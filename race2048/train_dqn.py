"""Train DQN on Game2048Env."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from race2048.dqn import DQNAgent, ReplayBuffer
from race2048.env import Game2048Env


def _flatten(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def train(
    *,
    total_steps: int = 200_000,
    learning_starts: int = 2_000,
    batch_size: int = 128,
    buffer_size: int = 100_000,
    train_frequency: int = 1,
    env_seed: int = 0,
    save_path: Path | None = None,
    device: str | None = None,
    log_every: int = 5_000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    eps_decay_steps: int = 100_000,
    target_update_every: int = 1_000,
) -> DQNAgent:
    env = Game2048Env(seed=env_seed, max_steps=50_000)
    buffer = ReplayBuffer(buffer_size)
    agent = DQNAgent(
        device=device,
        gamma=gamma,
        lr=lr,
        eps_decay_steps=eps_decay_steps,
        target_update_every=target_update_every,
    )

    obs, info = env.reset(seed=env_seed)
    episode = 0
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        mask = info["legal_action_mask"]
        action = agent.select_action(obs, mask, step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["legal_action_mask"]

        buffer.push(
            _flatten(obs),
            action,
            float(reward),
            _flatten(next_obs),
            done,
            next_mask,
        )

        if step >= learning_starts and step % train_frequency == 0:
            agent.train_step(buffer, batch_size)

        if done:
            episode += 1
            obs, info = env.reset(seed=env_seed + episode)
        else:
            obs = next_obs

        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            eps = agent.epsilon(step)
            print(
                f"step {step}/{total_steps}  eps {eps:.3f}  "
                f"buffer {len(buffer)}  train_steps {agent.train_steps}  "
                f"episodes {episode}  {elapsed:.0f}s"
            )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state": agent.policy_net.state_dict(),
                "target_state": agent.target_net.state_dict(),
                "train_steps": agent.train_steps,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

    return agent


def main() -> None:
    p = argparse.ArgumentParser(description="Train DQN on 2048")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--target-update-every", type=int, default=1000)
    args = p.parse_args()

    train(
        total_steps=args.steps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_frequency=args.train_freq,
        env_seed=args.seed,
        save_path=args.save,
        device=args.device,
        log_every=args.log_every,
        gamma=args.gamma,
        lr=args.lr,
        eps_decay_steps=args.eps_decay_steps,
        target_update_every=args.target_update_every,
    )


if __name__ == "__main__":
    main()
