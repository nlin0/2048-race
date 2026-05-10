"""Evaluate baseline agents over many episodes (prints Section 3 metrics)."""

from __future__ import annotations

import argparse
import json

from race2048.agents import GreedyEmptyAgent, OrderedAgent, RandomAgent
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


def main() -> None:
    p = argparse.ArgumentParser(description="Roll out 2048 baselines and aggregate metrics.")
    p.add_argument(
        "agent",
        nargs="?",
        default="random",
        choices=["random", "ordered", "greedy"],
        help="Which baseline policy to run",
    )
    p.add_argument("--episodes", type=int, default=200, help="Number of full games")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for env / random agent")
    p.add_argument("--max-steps", type=int, default=50_000, help="Truncation cap per episode")
    p.add_argument("--json", action="store_true", help="Print one JSON object with stats")
    args = p.parse_args()

    reports = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        env = Game2048Env(seed=ep_seed, max_steps=args.max_steps)
        agent = _make_agent(args.agent, seed=ep_seed)
        reports.append(run_episode(env, agent.act, seed=ep_seed))

    stats = aggregate(reports)
    if args.json:
        print(
            json.dumps(
                {
                    "agent": args.agent,
                    "episodes": stats.episodes,
                    "reach_2048_rate": stats.reach_2048_rate,
                    "mean_moves_to_2048": stats.mean_moves_to_2048,
                    "median_moves_to_2048": stats.median_moves_to_2048,
                    "mean_valid_moves": stats.mean_valid_moves,
                    "mean_max_tile": stats.mean_max_tile,
                },
                indent=2,
            )
        )
        return

    print(f"Agent: {args.agent}   episodes: {stats.episodes}")
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
