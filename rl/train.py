"""
REINFORCE (Vanilla Policy Gradient) training on NeTrainSimEnv using Tianshou 0.5.1.

Phase 1 note: actions do not yet feed back to the simulator (see train_env.py).
The pipeline is correct end-to-end; replace Phase 1 env with Phase 2 (interactive
C++ mode) to get real RL control. See CLAUDE.md "C++ Modifications Required".

Run:
    source venv/bin/activate
    python rl/train.py
"""

import os
import sys

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
DISCOUNT = 0.99
MAX_EPOCH = 200
# episode_per_collect=1: collect one full A→B episode per policy update
# (on-policy REINFORCE requires complete episodes)
EPISODES_PER_COLLECT = 1
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 4     # gradient update passes over the collected batch
BATCH_SIZE           = 512   # rows sampled per gradient step (trajectory has ~6700 rows)
STEP_PER_EPOCH       = 10_000  # roughly one full A→B episode at default speed


def make_env():
    return NeTrainSimEnv()


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

    policy = PGPolicy(
        model=actor,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        reward_normalization=True,  # normalize discounted returns across batch
    )

    # ── Collectors + Buffer ───────────────────────────────────────────────
    # Buffer large enough for one full episode (~6700 steps + headroom)
    buffer = VectorReplayBuffer(total_size=15_000, buffer_num=1)
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
    )

    print("Starting REINFORCE training on NeTrainSimEnv.")
    print("Each episode = one full A→B trip (~74.9 km, ~6700 simulator steps).")
    print(f"Training for {MAX_EPOCH} epochs × {STEP_PER_EPOCH} steps/epoch.\n")

    result = trainer.run()
    print(f"\nTraining complete: {result}")


if __name__ == "__main__":
    main()
