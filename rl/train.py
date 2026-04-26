"""
REINFORCE (Vanilla Policy Gradient) training on NeTrainSimEnv using Tianshou 0.5.1.

Actions are fed to NeTrainSim interactive mode each timestep (notch 0-8).

Run:
    source venv/bin/activate
    python rl/train.py
"""

import os
import sys
import time

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv

DEVICE = "cpu"
HIDDEN_SIZES = [128, 64]
LR = 3e-4
DISCOUNT = 0.999  # 0.99 → horizon ~100 steps; 0.999 → ~1000 (visible over 6700-step episodes)
MAX_EPOCH = 200
# episode_per_collect=1: collect one full A→B episode per policy update
# (on-policy REINFORCE requires complete episodes)
EPISODES_PER_COLLECT = 1
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 1     # REINFORCE has no importance correction; >1 biases gradient
BATCH_SIZE           = 512   # rows sampled per gradient step (trajectory has ~6700 rows)
STEP_PER_EPOCH       = 7_000  # aligned to one full episode (~6700 steps + margin)


def make_env():
    return NeTrainSimEnv()


_train_start = time.time()


def train_fn(epoch: int, env_step: int) -> None:
    elapsed = time.time() - _train_start
    print(
        f"\n{'='*65}\n"
        f"  Epoch {epoch:3d}/{MAX_EPOCH}  |  env_step={env_step:,}  |  elapsed={elapsed/60:.1f}min\n"
        f"{'='*65}",
        flush=True,
    )


def test_fn(epoch: int, env_step: int) -> None:
    print(f"  [test episode — epoch {epoch}]", flush=True)


def main():
    # ── Environments ──────────────────────────────────────────────────────
    # n=1: each env spawns a simulator subprocess. Increase with SubprocVectorEnv
    # if the machine can run multiple simulator instances in parallel.
    train_envs = DummyVectorEnv([make_env])
    test_envs  = DummyVectorEnv([make_env])

    # ── Network + Policy ──────────────────────────────────────────────────
    obs_shape = (7,)   # matches NeTrainSimEnv.observation_space
    n_actions = 9      # notch 0-8

    net   = Net(state_shape=obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    actor = Actor(
        preprocess_net=net,
        action_shape=n_actions,
        softmax_output=True,
        device=DEVICE,
    ).to(DEVICE)

    optim = torch.optim.Adam(actor.parameters(), lr=LR)

    # PGPolicy in Tianshou 0.5.1 has no max_grad_norm param; clip via optimizer hook.
    _orig_step = optim.step
    def _clipped_step(*args, **kwargs):
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        return _orig_step(*args, **kwargs)
    optim.step = _clipped_step  # type: ignore[method-assign]

    policy = PGPolicy(
        model=actor,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        reward_normalization=True,  # normalize discounted returns across batch
    )

    # ── Collectors + Buffer ───────────────────────────────────────────────
    # Buffer sized for exactly one full episode (~6700 steps + margin)
    buffer = VectorReplayBuffer(total_size=8_000, buffer_num=1)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=MAX_EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=EPISODES_PER_TEST,
        batch_size=BATCH_SIZE,
        episode_per_collect=EPISODES_PER_COLLECT,
        train_fn=train_fn,
        test_fn=test_fn,
        verbose=True,
        show_progress=True,
    )

    print("Starting REINFORCE training on NeTrainSimEnv.")
    print("Each episode = one full A→B trip (~74.9 km, ~6700 simulator steps).")
    print(f"Training for {MAX_EPOCH} epochs × {STEP_PER_EPOCH} steps/epoch.")
    print("Progress: ep=episode  step=sim-step  pos=distance  energy=cumulative kWh\n")

    result = trainer.run()
    print(f"\nTraining complete: {result}")


if __name__ == "__main__":
    main()
