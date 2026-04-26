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
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv

# Auto-detect GPU — uses CUDA on DGX Spark, falls back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parallel simulator instances — each spawns one C++ subprocess.
# DGX Spark has many CPU cores; running 4 envs in parallel fills the GPU
# with data while C++ simulators run concurrently.
NUM_TRAIN_ENVS = 4
NUM_TEST_ENVS  = 1

# Larger network to take advantage of GPU compute
HIDDEN_SIZES = [256, 128, 64]
LR           = 3e-4
DISCOUNT     = 0.999   # long horizon — visible over ~7500-step episodes
MAX_EPOCH    = 200

# With NUM_TRAIN_ENVS=4, each collect gathers 4 episodes in parallel
EPISODES_PER_COLLECT = NUM_TRAIN_ENVS
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 1       # REINFORCE: no importance correction, >1 biases gradient
BATCH_SIZE           = 2048    # larger batch — GPU handles this cheaply
STEP_PER_EPOCH       = 7_000   # ~1 full episode per env per epoch

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


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
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"policy_epoch{epoch:03d}.pth")
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved → {path}", flush=True)


def test_fn(epoch: int, env_step: int) -> None:
    print(f"  [test episode — epoch {epoch}]", flush=True)


def main():
    global policy  # needed so train_fn can access it for checkpointing

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Environments ──────────────────────────────────────────────────────
    # SubprocVectorEnv runs each env in a separate process — C++ simulators
    # run in parallel, keeping the GPU fed with experience data.
    train_envs = SubprocVectorEnv([make_env] * NUM_TRAIN_ENVS)
    test_envs  = SubprocVectorEnv([make_env] * NUM_TEST_ENVS)

    # ── Network + Policy ──────────────────────────────────────────────────
    obs_shape = (7,)
    n_actions = 9

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
        reward_normalization=True,
    )

    # ── Collectors + Buffer ───────────────────────────────────────────────
    # Buffer sized for NUM_TRAIN_ENVS full episodes + margin
    buffer = VectorReplayBuffer(total_size=8_000 * NUM_TRAIN_ENVS, buffer_num=NUM_TRAIN_ENVS)
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
    print(f"Parallel train envs: {NUM_TRAIN_ENVS}  |  device: {DEVICE}")
    print("Each episode = one full A→B trip (~74.9 km, ~7500 simulator steps).")
    print(f"Training for {MAX_EPOCH} epochs × {STEP_PER_EPOCH} steps/epoch.")
    print("Checkpoints saved every 10 epochs → checkpoints/\n")

    result = trainer.run()
    print(f"\nTraining complete: {result}")

    # Save final model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Final model saved → {final_path}")


if __name__ == "__main__":
    main()
