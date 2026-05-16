"""Evaluate baseline agents and greedy DQN over many episodes (Section 3 metrics)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from race2048.agents import GreedyEmptyAgent, OrderedAgent, RandomAgent
from race2048.dqn.qnet import load_qnetwork_from_checkpoint
from race2048.env import Game2048Env
from race2048.evaluation import aggregate, run_episode


def _make_agent(name: str, seed: int):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "ordered":
        return OrderedAgent()
    if name == "greedy":
        return GreedyEmptyAgent()
    raise ValueError(name)


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any] | Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _greedy_dqn_act_fn(
    checkpoint: Path,
    *,
    device: torch.device,
) -> Callable[[np.ndarray, dict[str, Any]], int]:
    """Argmax Q among legal moves only (no ε exploration)."""
    ck: dict[str, Any] | Any = _load_ckpt(checkpoint, device)
    net = load_qnetwork_from_checkpoint(ck, device)

    @torch.no_grad()
    def act(obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = np.asarray(info["legal_action_mask"], dtype=np.bool_).reshape(4)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0).to(device)
        q = net(x).squeeze(0).clone()
        m = torch.from_numpy(mask).to(device)
        q[~m] = -float("inf")
        return int(q.argmax().item())

    return act


def main() -> None:
    p = argparse.ArgumentParser(
        description="Roll out 2048 policies (baselines or greedy DQN) and aggregate metrics."
    )
    p.add_argument(
        "agent",
        nargs="?",
        default="random",
        choices=["random", "ordered", "greedy", "dqn"],
        help="Policy: baselines or greedy DQN from checkpoint",
    )
    p.add_argument("--episodes", type=int, default=200, help="Number of full games")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for env / random agent")
    p.add_argument("--max-steps", type=int, default=50_000, help="Truncation cap per episode")
    p.add_argument("--json", action="store_true", help="Print one JSON object with stats")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/dqn.pt"),
        help="For agent=dqn: path to training checkpoint (policy_state)",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Deprecated for dqn if checkpoint has qnet_config (auto-detected)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="For agent=dqn: torch device (default: cpu)",
    )
    p.add_argument(
        "--snake-top-left",
        action="store_true",
        help="Use snake env (overrides checkpoint default when set)",
    )
    p.add_argument(
        "--no-snake-top-left",
        action="store_true",
        help="Disable snake env (overrides checkpoint)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="MDP γ for potential shaping when snake env is on (match training)",
    )
    args = p.parse_args()

    map_dev = torch.device(args.device or "cpu")
    snake_tl = False
    ck_for_snake: dict[str, Any] | None = None
    if args.agent == "dqn":
        ck_for_snake = _load_ckpt(args.checkpoint.resolve(), map_dev)
        if not isinstance(ck_for_snake, dict):
            ck_for_snake = None
    if args.no_snake_top_left:
        snake_tl = False
    elif args.snake_top_left:
        snake_tl = True
    elif args.agent == "dqn" and ck_for_snake is not None:
        snake_tl = bool(ck_for_snake.get("env_snake_top_left", False))

    env_kw: dict[str, Any] = {}
    if snake_tl:
        env_kw["snake_top_left"] = True
        env_kw["snake_shaping_gamma"] = float(args.gamma)

    if args.agent == "dqn":
        act_fn = _greedy_dqn_act_fn(args.checkpoint.resolve(), device=map_dev)
    else:
        act_fn = None

    reports = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        env = Game2048Env(seed=ep_seed, max_steps=args.max_steps, **env_kw)
        if args.agent == "dqn":
            reports.append(run_episode(env, act_fn, seed=ep_seed))
        else:
            agent = _make_agent(args.agent, seed=ep_seed)
            reports.append(run_episode(env, agent.act, seed=ep_seed))

    stats = aggregate(reports)
    if args.json:
        payload: dict[str, Any] = {
            "agent": args.agent,
            "episodes": stats.episodes,
            "reach_2048_rate": stats.reach_2048_rate,
            "mean_moves_to_2048": stats.mean_moves_to_2048,
            "median_moves_to_2048": stats.median_moves_to_2048,
            "mean_valid_moves": stats.mean_valid_moves,
            "mean_max_tile": stats.mean_max_tile,
            "std_max_tile": stats.std_max_tile,
            "median_max_tile": stats.median_max_tile,
            "mean_final_tile_sum": stats.mean_final_tile_sum,
            "mean_invalid_moves": stats.mean_invalid_moves,
        }
        if args.agent == "dqn":
            payload["checkpoint"] = str(args.checkpoint.resolve())
        print(json.dumps(payload, indent=2))
        return

    print(f"Agent: {args.agent}   episodes: {stats.episodes}")
    if args.agent == "dqn":
        print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Reach 2048 rate:       {stats.reach_2048_rate:.3f}")
    if stats.mean_moves_to_2048 is not None:
        print(
            f"Mean moves to 2048:    {stats.mean_moves_to_2048:.2f}   "
            f"(median {stats.median_moves_to_2048:.1f})"
        )
    else:
        print("Mean moves to 2048:    n/a (never reached in any episode)")
    print(f"Mean valid moves/game: {stats.mean_valid_moves:.1f}")
    print(f"Mean max tile:         {stats.mean_max_tile:.1f}  (std {stats.std_max_tile:.2f}, median {stats.median_max_tile:.0f})")
    print(f"Mean final tile sum:   {stats.mean_final_tile_sum:.0f}")
    print(f"Mean invalid moves/ep: {stats.mean_invalid_moves:.2f}")


if __name__ == "__main__":
    main()
