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
from race2048.dqn.qnet import QNetwork
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


def _load_qnetwork(
    checkpoint: Path,
    *,
    device: torch.device,
    hidden_dim: int,
) -> QNetwork:
    net = QNetwork(hidden_dim=hidden_dim).to(device)
    try:
        ck: dict[str, Any] | Any = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
    except TypeError:
        ck = torch.load(checkpoint, map_location=device)
    sd = ck["policy_state"] if isinstance(ck, dict) and "policy_state" in ck else ck
    net.load_state_dict(sd)
    net.eval()
    return net


def _greedy_dqn_act_fn(
    checkpoint: Path,
    *,
    device: torch.device,
    hidden_dim: int,
) -> Callable[[np.ndarray, dict[str, Any]], int]:
    """Argmax Q among legal moves only (no ε exploration)."""
    net = _load_qnetwork(checkpoint, device=device, hidden_dim=hidden_dim)

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
        help="For agent=dqn: MLP width (must match checkpoint)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="For agent=dqn: torch device (default: cpu)",
    )
    args = p.parse_args()

    map_dev = torch.device(args.device or "cpu")
    if args.agent == "dqn":
        act_fn = _greedy_dqn_act_fn(
            args.checkpoint.resolve(),
            device=map_dev,
            hidden_dim=args.hidden_dim,
        )
    else:
        act_fn = None

    reports = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        env = Game2048Env(seed=ep_seed, max_steps=args.max_steps)
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
        }
        if args.agent == "dqn":
            payload["checkpoint"] = str(args.checkpoint.resolve())
            payload["hidden_dim"] = args.hidden_dim
        print(json.dumps(payload, indent=2))
        return

    print(f"Agent: {args.agent}   episodes: {stats.episodes}")
    if args.agent == "dqn":
        print(f"Checkpoint: {args.checkpoint.resolve()}  hidden_dim={args.hidden_dim}")
    print(f"Reach 2048 rate:       {stats.reach_2048_rate:.3f}")
    if stats.mean_moves_to_2048 is not None:
        print(
            f"Mean moves to 2048:    {stats.mean_moves_to_2048:.2f}   "
            f"(median {stats.median_moves_to_2048:.1f})"
        )
    else:
        print("Mean moves to 2048:    n/a (never reached in any episode)")
    print(f"Mean valid moves/game: {stats.mean_valid_moves:.1f}")
    print(f"Mean max tile:         {stats.mean_max_tile:.1f}")


if __name__ == "__main__":
    main()
