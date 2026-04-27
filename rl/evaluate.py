"""
Evaluate a trained RL policy on one full A→B trip and export the notch
profile + physics state to a CSV file.

The output CSV can be loaded into NeTrainSim (or plotted) to compare the
RL energy profile against a baseline (e.g. constant notch or human driver).

Usage:
    source venv/bin/activate
    python rl/evaluate.py                                      # uses latest checkpoint
    python rl/evaluate.py --checkpoint checkpoints/policy_epoch050.pth
    python rl/evaluate.py --checkpoint checkpoints/policy_final.pth
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv, TOTAL_ROUTE_LENGTH_M

DEVICE       = "cpu"
HIDDEN_SIZES = [256, 128, 64]
N_ACTIONS    = 9
OBS_SHAPE    = (7,)
DISCOUNT     = 0.999

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def find_latest_checkpoint() -> str:
    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"No checkpoints directory found at {CHECKPOINT_DIR}")
    files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth"))
    if not files:
        raise FileNotFoundError("No .pth checkpoint files found in checkpoints/")
    if "policy_final.pth" in files:
        return os.path.join(CHECKPOINT_DIR, "policy_final.pth")
    return os.path.join(CHECKPOINT_DIR, files[-1])


def build_policy(checkpoint_path: str) -> PPOPolicy:
    net_actor  = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    net_critic = Net(state_shape=OBS_SHAPE, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    actor  = Actor(net_actor,  action_shape=N_ACTIONS, softmax_output=True, device=DEVICE).to(DEVICE)
    critic = Critic(net_critic, device=DEVICE).to(DEVICE)
    optim  = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        eps_clip=0.2,
        advantage_normalization=True,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        reward_normalization=True,
        max_grad_norm=0.5,
    )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy


def run_episode(policy: PPOPolicy, output_path: str) -> None:
    env = NeTrainSimEnv()
    obs, _ = env.reset()

    rows = []
    step = 0
    cum_energy = 0.0
    terminated = False
    truncated  = False

    print("Running evaluation episode...")
    print(f"{'Step':>6}  {'Position(m)':>12}  {'Speed(m/s)':>10}  {'Notch':>5}  "
          f"{'Grade(%)':>8}  {'SpeedLimit':>10}  {'StepEnergy':>10}  {'CumEnergy':>10}")
    print("-" * 80)

    while not (terminated or truncated):
        # Policy picks action deterministically (greedy — highest probability notch)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy.actor(obs_tensor)
            notch = int(logits.argmax(dim=-1).item())

        obs, reward, terminated, truncated, _ = env.step(notch)

        state = env._last_state
        position_m      = float(state["position_m"])
        speed_mps       = float(state["speed_mps"])
        grade_perc      = float(state["grade_perc"])
        curvature_perc  = float(state["curvature_perc"])
        link_max_speed  = float(state["link_max_speed_mps"])
        step_energy_kwh = float(state["energy_kwh"])
        cum_energy     += step_energy_kwh

        rows.append({
            "position_m": round(position_m, 3),
            "speed_mps":  round(speed_mps, 4),
            "notch":      notch,
        })

        step += 1
        if step % 500 == 0:
            pct = 100.0 * position_m / TOTAL_ROUTE_LENGTH_M
            print(f"{step:>6}  {position_m:>12.1f}  {speed_mps:>10.3f}  {notch:>5}  "
                  f"{grade_perc:>8.4f}  {link_max_speed:>10.3f}  "
                  f"{step_energy_kwh:>10.5f}  {cum_energy:>10.3f}")

    env.close()

    status = "ARRIVED" if terminated else "TIMEOUT"
    print(f"\n[{status}] steps={step}  total_energy={cum_energy:.2f} kWh\n")

    # Write CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["position_m", "speed_mps", "notch"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Notch profile saved → {output_path}")
    print(f"Rows: {len(rows)}  |  Total energy: {cum_energy:.2f} kWh  |  Steps: {step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (default: latest in checkpoints/)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/notch_profile.csv)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint()
    output_path = args.output or os.path.join(OUTPUT_DIR, "notch_profile.csv")

    policy = build_policy(checkpoint)
    run_episode(policy, output_path)


if __name__ == "__main__":
    main()
