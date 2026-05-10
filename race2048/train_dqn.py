"""Train DQN on Game2048Env (episode loop + replay + resume)."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from race2048.board import Game2048
from race2048.dqn import DQNAgent, ReplayBuffer
from race2048.env import Game2048Env


def _flatten(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _load_training_checkpoint(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _apply_checkpoint(agent: DQNAgent, ck: dict[str, Any]) -> tuple[int, int]:
    """Restore weights and optional optimizer. Returns (env_steps, train_steps)."""
    agent.policy_net.load_state_dict(ck["policy_state"])
    if "target_state" in ck:
        agent.target_net.load_state_dict(ck["target_state"])
    else:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    ts = int(ck.get("train_steps", 0))
    agent.train_steps = ts
    if ck.get("optimizer_state") is not None:
        try:
            agent.optimizer.load_state_dict(ck["optimizer_state"])
        except Exception:
            pass
    env_steps = int(ck.get("env_steps", ck.get("global_step", ck.get("train_steps", 0))))
    return env_steps, ts


def _build_checkpoint(agent: DQNAgent, env_steps: int) -> dict[str, Any]:
    return {
        "policy_state": agent.policy_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "train_steps": agent.train_steps,
        "env_steps": env_steps,
    }


def train(
    *,
    total_env_steps: int = 200_000,
    learning_starts: int = 2_000,
    batch_size: int = 128,
    buffer_size: int = 100_000,
    train_frequency: int = 1,
    env_seed: int = 0,
    max_episodes: int | None = None,
    save_path: Path | None = None,
    save_every_env_steps: int | None = None,
    load_path: Path | None = None,
    device: str | None = None,
    log_every_episodes: int = 50,
    log_every_env_steps: int | None = 10_000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    eps_decay_steps: int = 100_000,
    target_update_every: int = 1_000,
) -> DQNAgent:
    """
    Standard DQN loop in Python:

    for each episode:
        reset env
        while not terminal:
            ε-greedy action → step → store transition
            if replay big enough: sample batch, loss, backprop
            ε decays with **global env step** (not episode index)
    """
    map_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env(seed=env_seed, max_steps=50_000)
    buffer = ReplayBuffer(buffer_size)
    agent = DQNAgent(
        device=device,
        gamma=gamma,
        lr=lr,
        eps_decay_steps=eps_decay_steps,
        target_update_every=target_update_every,
    )

    env_t = 0
    if load_path is not None:
        ck = _load_training_checkpoint(load_path, map_dev)
        env_t, _ = _apply_checkpoint(agent, ck)
        print(
            f"Loaded {load_path} — resume env_steps={env_t}, "
            f"train_steps={agent.train_steps}"
        )

    target_env_t = env_t + total_env_steps
    episode_idx = 0
    t0 = time.perf_counter()
    win_cap = max(1, log_every_episodes)
    recent_ret: deque[float] = deque(maxlen=win_cap)
    recent_len: deque[int] = deque(maxlen=win_cap)
    recent_win: deque[int] = deque(maxlen=win_cap)

    while env_t < target_env_t:
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        obs, info = env.reset(seed=env_seed + episode_idx)
        ep_reward = 0.0
        ep_env_steps = 0
        ep_max_tile = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            if env_t >= target_env_t:
                break

            env_t += 1
            mask = info["legal_action_mask"]
            action = agent.select_action(obs, mask, env_t)
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

            if env_t >= learning_starts and env_t % train_frequency == 0:
                agent.train_step(buffer, batch_size)

            ep_reward += float(reward)
            ep_env_steps += 1
            ep_max_tile = max(ep_max_tile, int(info.get("max_tile", 0)))

            obs = next_obs

            if (
                save_path is not None
                and save_every_env_steps is not None
                and env_t % save_every_env_steps == 0
            ):
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_build_checkpoint(agent, env_t), save_path)
                print(f"[checkpoint] env_steps={env_t} → {save_path}")

            if log_every_env_steps is not None and env_t % log_every_env_steps == 0:
                elapsed = time.perf_counter() - t0
                eps = agent.epsilon(env_t)
                print(
                    f"env_step {env_t}/{target_env_t}  eps {eps:.3f}  "
                    f"buffer {len(buffer)}  grad {agent.train_steps}  "
                    f"ep {episode_idx}  {elapsed:.0f}s"
                )

        won = ep_max_tile >= Game2048.WIN_TILE
        if log_every_episodes > 0:
            recent_ret.append(ep_reward)
            recent_len.append(ep_env_steps)
            recent_win.append(1 if won else 0)

        episode_idx += 1
        if log_every_episodes > 0 and episode_idx % log_every_episodes == 0:
            elapsed = time.perf_counter() - t0
            eps = agent.epsilon(env_t)
            wr = 100.0 * sum(recent_win) / max(1, len(recent_win))
            mr = sum(recent_ret) / max(1, len(recent_ret))
            ml = sum(recent_len) / max(1, len(recent_len))
            print(
                f"episode {episode_idx}  env_step {env_t}/{target_env_t}  "
                f"eps {eps:.3f}  last_{len(recent_ret)}ep: "
                f"win% {wr:.1f}  mean_return {mr:.2f}  mean_len {ml:.1f}  "
                f"buffer {len(buffer)}  {elapsed:.0f}s"
            )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_build_checkpoint(agent, env_t), save_path)
        print(f"Saved checkpoint to {save_path}")

    return agent


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train DQN on 2048 (Python). Use --load to continue training."
    )
    p.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="How many **environment steps** to take after starting (or after --load resume point)",
    )
    p.add_argument("--learning-starts", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N env steps (optional)",
    )
    p.add_argument("--load", type=Path, default=None, help="Resume policy / target / optimizer")
    p.add_argument(
        "--log-every-episodes",
        type=int,
        default=50,
        help="Print rolling episode stats every N episodes (0 to disable)",
    )
    p.add_argument(
        "--log-every-steps",
        type=int,
        default=10_000,
        help="Print env-step progress every N steps (0 disables)",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--target-update-every", type=int, default=1000)
    args = p.parse_args()

    log_steps = args.log_every_steps if args.log_every_steps > 0 else None
    train(
        total_env_steps=args.steps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_frequency=args.train_freq,
        env_seed=args.seed,
        max_episodes=args.max_episodes,
        save_path=args.save,
        save_every_env_steps=args.save_every,
        load_path=args.load,
        device=args.device,
        log_every_episodes=args.log_every_episodes,
        log_every_env_steps=log_steps,
        gamma=args.gamma,
        lr=args.lr,
        eps_decay_steps=args.eps_decay_steps,
        target_update_every=args.target_update_every,
    )


if __name__ == "__main__":
    main()
