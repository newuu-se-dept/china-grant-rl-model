"""
PPO training on NeTrainSimEnv using Tianshou 0.5.1.

Improvements over REINFORCE:
  - PPO clips policy updates (stable learning, no entropy collapse)
  - Critic (value network) reduces gradient variance vs pure REINFORCE
  - ent_coef=0.01 keeps exploration alive throughout training
  - repeat_per_collect=4 reuses each batch of episodes 4× (more efficient)

Run:
    source venv/bin/activate
    python rl/train.py 2>&1 | tee train_log.txt
"""

import os
import sys
import time

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.train_env import NeTrainSimEnv

# CPU is faster: network is tiny and inference runs one step at a time,
# so GPU kernel-launch overhead exceeds any compute gain.
DEVICE = "cpu"

NUM_TRAIN_ENVS = 8
NUM_TEST_ENVS  = 1

HIDDEN_SIZES = [256, 128, 64]
LR           = 3e-4
DISCOUNT     = 0.99    # was 0.999 — shorter effective horizon (~100 steps), keeps advantages from collapsing
MAX_EPOCH    = 300

EPISODES_PER_COLLECT = NUM_TRAIN_ENVS   # 8 parallel episodes before each update
EPISODES_PER_TEST    = 1
REPEAT_PER_COLLECT   = 4                # PPO reuses each collected batch 4×
BATCH_SIZE           = 2048
STEP_PER_EPOCH       = 7_000

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

_train_start = time.time()


def make_env():
    return NeTrainSimEnv()


def train_fn(epoch: int, env_step: int) -> None:
    elapsed = (time.time() - _train_start) / 60
    print(
        f"\n{'━'*65}\n"
        f"  Epoch {epoch:3d}/{MAX_EPOCH}  │  {env_step:>10,} steps  │  {elapsed:.1f} min elapsed\n"
        f"{'━'*65}",
        flush=True,
    )
    if epoch % 10 == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINT_DIR, f"policy_epoch{epoch:03d}.pth")
        torch.save(policy.state_dict(), path)
        print(f"  → checkpoint: {path}", flush=True)


def test_fn(epoch: int, env_step: int) -> None:
    print(f"  [test]", flush=True)


def main():
    global policy

    print("━" * 65)
    print("  PPO Training — NeTrainSim Energy Optimization")
    print("━" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Train envs   : {NUM_TRAIN_ENVS} parallel C++ simulators")
    print(f"  Epochs       : {MAX_EPOCH}  (checkpoint every 10)")
    print(f"  Algorithm    : PPO  (eps_clip=0.2, ent_coef=0.01, repeat=4×)")
    print(f"  Reward       : -energy/step  -speed_penalty  -time_penalty  +100 arrival")
    print("━" * 65 + "\n")

    train_envs = SubprocVectorEnv([make_env] * NUM_TRAIN_ENVS)
    test_envs  = SubprocVectorEnv([make_env] * NUM_TEST_ENVS)

    obs_shape = (7,)
    n_actions = 9

    # Separate networks for actor and critic — PPO needs a value estimate
    net_actor  = Net(state_shape=obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)
    net_critic = Net(state_shape=obs_shape, hidden_sizes=HIDDEN_SIZES, device=DEVICE)

    actor = Actor(
        preprocess_net=net_actor,
        action_shape=n_actions,
        softmax_output=True,
        device=DEVICE,
    ).to(DEVICE)

    critic = Critic(
        preprocess_net=net_critic,
        device=DEVICE,
    ).to(DEVICE)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR
    )

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        discount_factor=DISCOUNT,
        eps_clip=0.2,               # clip ratio — prevents large destructive updates
        advantage_normalization=True,
        vf_coef=0.5,                # value loss weight
        ent_coef=0.001,             # was 0.01 — was dominating the gradient and forcing uniform-random policy
        gae_lambda=0.95,            # GAE smoothing for advantage estimates
        reward_normalization=False, # was True — was shrinking advantages to ~0 and starving policy gradient
        max_grad_norm=0.5,
    )

    buffer = VectorReplayBuffer(
        total_size=8_000 * NUM_TRAIN_ENVS,
        buffer_num=NUM_TRAIN_ENVS,
    )
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

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

    result = trainer.run()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)

    total_min = (time.time() - _train_start) / 60
    print(f"\n{'━'*65}")
    print(f"  Training complete in {total_min:.1f} min")
    print(f"  Best reward : {result['best_reward']:.3f}")
    print(f"  Final model : {final_path}")
    print(f"{'━'*65}")


if __name__ == "__main__":
    main()
