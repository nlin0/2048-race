"""Rollouts and aggregate metrics (Section 3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from race2048.board import Game2048
from race2048.env import Game2048Env


@dataclass(slots=True)
class EpisodeReport:
    """One rollout."""

    steps: int  # env.step calls (valid + invalid)
    valid_moves: int
    reached_2048: bool
    moves_to_2048: int | None  # valid moves through first state with max tile >= 2048
    max_tile: int
    final_tile_sum: int  # sum of all tile values on final board (reporting metric)
    invalid_moves: int  # steps - valid_moves


def run_episode(
    env: Game2048Env,
    act_fn: Callable[[np.ndarray, dict[str, Any]], int],
    *,
    seed: int | None = None,
) -> EpisodeReport:
    """Run until terminated or truncated. `act_fn` is e.g. `agent.act`."""
    obs, info = env.reset(seed=seed)
    moves_to_2048: int | None = None
    valid_moves = 0
    total_steps = 0

    while True:
        action = int(act_fn(obs, info))
        obs, _reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if info.get("valid"):
            valid_moves += 1
        if moves_to_2048 is None and int(info.get("max_tile", 0)) >= Game2048.WIN_TILE:
            moves_to_2048 = valid_moves
        if terminated or truncated:
            break

    board = env.game.board
    max_tile = int(board.max())
    final_sum = int(board.sum())
    return EpisodeReport(
        steps=total_steps,
        valid_moves=valid_moves,
        reached_2048=max_tile >= Game2048.WIN_TILE,
        moves_to_2048=moves_to_2048,
        max_tile=max_tile,
        final_tile_sum=final_sum,
        invalid_moves=total_steps - valid_moves,
    )


@dataclass(slots=True)
class AggregateStats:
    episodes: int
    reach_2048_rate: float
    mean_moves_to_2048: float | None
    median_moves_to_2048: float | None
    mean_valid_moves: float
    mean_max_tile: float
    std_max_tile: float
    median_max_tile: float
    mean_final_tile_sum: float
    mean_invalid_moves: float


def aggregate(reports: Sequence[EpisodeReport]) -> AggregateStats:
    n = len(reports)
    if n == 0:
        return AggregateStats(0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    reached = sum(1 for r in reports if r.reached_2048)
    rate = reached / n
    mt = [r.moves_to_2048 for r in reports if r.moves_to_2048 is not None]
    mean_m = float(np.mean(mt)) if mt else None
    med_m = float(np.median(mt)) if mt else None
    max_tiles = [r.max_tile for r in reports]
    std_max = float(np.std(max_tiles, ddof=1)) if n > 1 else 0.0
    return AggregateStats(
        episodes=n,
        reach_2048_rate=rate,
        mean_moves_to_2048=mean_m,
        median_moves_to_2048=med_m,
        mean_valid_moves=float(np.mean([r.valid_moves for r in reports])),
        mean_max_tile=float(np.mean(max_tiles)),
        std_max_tile=std_max,
        median_max_tile=float(np.median(max_tiles)),
        mean_final_tile_sum=float(np.mean([r.final_tile_sum for r in reports])),
        mean_invalid_moves=float(np.mean([r.invalid_moves for r in reports])),
    )
