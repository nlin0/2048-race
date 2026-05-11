from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from race2048.board import Game2048
from race2048.demo_bots import pick_bot_action, resolve_checkpoint_under


def play_one_game(policy: str, checkpoint: Path | None, seed: int) -> dict:
    game = Game2048(seed=seed)
    game.reset()
    rng = np.random.default_rng(seed)

    moves = 0

    while game.legal_actions():
        action = pick_bot_action(
            policy,
            game,
            rng=rng,
            checkpoint=checkpoint,
            device=torch.device("cpu"),
        )
        result = game.step(action)
        moves += 1

        if result.game_over:
            break

    board = game.board
    return {
        "score_proxy": int(board.sum()),
        "max_tile": int(board.max()),
        "moves": moves,
        "reached_2048": int(board.max() >= 2048),
    }


def evaluate(policy: str, games: int, checkpoint_name: str | None) -> None:
    root = Path(__file__).resolve().parents[1]

    checkpoint = None
    if policy == "dqn":
        checkpoint = resolve_checkpoint_under(root, checkpoint_name, "dqn.pt")
        if checkpoint is None:
            raise FileNotFoundError("Could not find DQN checkpoint")

    results = []

    for i in range(games):
        results.append(play_one_game(policy, checkpoint, seed=i))

    max_tiles = np.array([r["max_tile"] for r in results])
    moves = np.array([r["moves"] for r in results])
    scores = np.array([r["score_proxy"] for r in results])
    wins = np.array([r["reached_2048"] for r in results])

    print(f"\nPolicy: {policy}")
    print(f"Games: {games}")
    print(f"Average score proxy: {scores.mean():.2f}")
    print(f"Average max tile: {max_tiles.mean():.2f}")
    print(f"Median max tile: {np.median(max_tiles):.2f}")
    print(f"Best max tile: {max_tiles.max()}")
    print(f"Average moves: {moves.mean():.2f}")
    print(f"Reached 2048 rate: {100 * wins.mean():.2f}%")

    unique, counts = np.unique(max_tiles, return_counts=True)
    print("\nMax tile distribution:")
    for tile, count in zip(unique, counts):
        print(f"  {tile}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "ordered", "greedy", "corner", "dqn"], required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="dqn.pt")
    args = parser.parse_args()

    evaluate(args.policy, args.games, args.checkpoint)


if __name__ == "__main__":
    main()