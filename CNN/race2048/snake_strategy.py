"""Top-left snake heuristics: safe action masking + potential-based reward shaping."""

from __future__ import annotations

import math

import numpy as np

from race2048.board import Action


def _log2_tile(v: int) -> float:
    if v <= 0:
        return 0.0
    return float(math.log2(v))


def _snake_path_cells() -> list[tuple[int, int]]:
    """Row-serpentine path from top-left through all 16 cells (standard 2048 snake)."""
    out: list[tuple[int, int]] = []
    for r in range(4):
        cols = range(4) if (r % 2) == 0 else range(3, -1, -1)
        for c in cols:
            out.append((r, c))
    return out


_SNAKE_PATH: tuple[tuple[int, int], ...] = tuple(_snake_path_cells())


def monotonic_edge_score(board: np.ndarray) -> float:
    """Non-negative score when top row (L→R) and left column (T→B) are non-increasing in log2."""
    b = np.asarray(board, dtype=np.int32)
    s = 0.0
    for c in range(3):
        a, d = int(b[0, c]), int(b[0, c + 1])
        if a > 0 and d > 0 and a >= d:
            s += _log2_tile(a) - _log2_tile(d)
    for r in range(3):
        a, d = int(b[r, 0]), int(b[r + 1, 0])
        if a > 0 and d > 0 and a >= d:
            s += _log2_tile(a) - _log2_tile(d)
    return s


def snake_path_monotonic_score(board: np.ndarray) -> float:
    """Sum of log drops along the full TL serpentine path for strictly decreasing adjacent tiles."""
    b = np.asarray(board, dtype=np.int32)
    s = 0.0
    for (r1, c1), (r2, c2) in zip(_SNAKE_PATH, _SNAKE_PATH[1:]):
        v1, v2 = int(b[r1, c1]), int(b[r2, c2])
        if v1 > 0 and v2 > 0 and v1 > v2:
            s += _log2_tile(v1) - _log2_tile(v2)
    return s


def max_tile_in_top_left_corner(board: np.ndarray) -> bool:
    b = np.asarray(board, dtype=np.int32)
    mx = int(b.max())
    return mx > 0 and int(b[0, 0]) == mx


def max_tile_in_any_corner(board: np.ndarray) -> bool:
    """True if the global maximum tile sits on at least one corner cell."""
    b = np.asarray(board, dtype=np.int32)
    mx = int(b.max())
    if mx <= 0:
        return False
    return (
        int(b[0, 0]) == mx
        or int(b[0, 3]) == mx
        or int(b[3, 0]) == mx
        or int(b[3, 3]) == mx
    )


def empty_fraction(board: np.ndarray) -> float:
    b = np.asarray(board, dtype=np.int32)
    return float(np.count_nonzero(b == 0)) / 16.0


def strategy_forbidden_mask(board: np.ndarray) -> np.ndarray:
    """Actions to discourage breaking a top-left anchor (same index order as Action).

    - If any maximum tile sits on row 0, forbid DOWN (would leave the top wall).
    - If any maximum tile sits on col 0, forbid RIGHT (would leave the left wall).

    Caller must intersect with legal moves and fall back if nothing remains.
    """
    b = np.asarray(board, dtype=np.int32)
    forbid = np.zeros(4, dtype=np.bool_)
    mx = int(b.max())
    if mx <= 0:
        return forbid
    rows, cols = np.where(b == mx)
    if np.any(rows == 0):
        forbid[int(Action.DOWN)] = True
    if np.any(cols == 0):
        forbid[int(Action.RIGHT)] = True
    return forbid


def apply_snake_strategy_mask(legal: np.ndarray, board: np.ndarray) -> np.ndarray:
    """legal & ~forbidden, or legal if that would forbid every legal move."""
    leg = np.asarray(legal, dtype=np.bool_).reshape(4)
    fb = strategy_forbidden_mask(board)
    masked = leg & ~fb
    if masked.any():
        return masked
    return leg.copy()


def strategy_potential(
    board: np.ndarray,
    *,
    corner_weight: float,
    monotonic_weight: float,
    empty_weight: float,
) -> float:
    """Scalar potential Φ(s) for potential-based shaping (same γ as MDP).

    Encourages: max tile on any corner, strict decrease along TL serpentine path,
    and more empty cells (reduces board lock).
    """
    b = np.asarray(board, dtype=np.int32)
    phi = monotonic_weight * snake_path_monotonic_score(b)
    if max_tile_in_any_corner(b):
        phi += corner_weight
    phi += empty_weight * empty_fraction(b)
    return phi
