"""Simple baseline players."""

from __future__ import annotations

import numpy as np

from race2048.board import Action, slide_board


class RandomAgent:
    """Uniform random among legal moves."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> int:
        mask = info["legal_action_mask"]
        idx = self._rng.choice(np.flatnonzero(mask))
        return int(idx)


class OrderedAgent:
    """Deterministic priority order (e.g. often DOWN-first). Skips illegal moves."""

    def __init__(self, order: list[int] | None = None) -> None:
        self._order = order or [Action.DOWN, Action.LEFT, Action.RIGHT, Action.UP]

    def act(self, obs: np.ndarray, info: dict) -> int:
        mask: np.ndarray = info["legal_action_mask"]
        for a in self._order:
            if mask[a]:
                return int(a)
        raise RuntimeError("no legal actions")


class GreedyEmptyAgent:
    """One-step lookahead without spawn: maximize empty cells after slide+merge."""

    def act(self, obs: np.ndarray, info: dict) -> int:
        mask: np.ndarray = info["legal_action_mask"]
        b = _obs_to_board(obs)
        best_a: int | None = None
        best_score = -1
        for a in range(4):
            if not mask[a]:
                continue
            slid = slide_board(b, a)
            score = int(np.sum(slid == 0))
            if score > best_score or (
                score == best_score
                and (best_a is None or a < int(best_a))
            ):
                best_score = score
                best_a = a
        assert best_a is not None
        return best_a


def _obs_to_board(obs: np.ndarray) -> np.ndarray:
    """Invert log2 encoding (approximate integer tiles)."""
    o = np.asarray(obs, dtype=np.float64)
    board = np.zeros((4, 4), dtype=np.int32)
    mask = o > 0
    board[mask] = (2 ** o[mask]).astype(np.int32)
    return board

class CornerHeuristicAgent:
    """
    Slightly smarter 2048 baseline.

    It looks one move ahead and scores the resulting board using:
    - number of empty cells
    - max tile value
    - whether the max tile is in a corner
    - board smoothness
    """

    def act(self, obs: np.ndarray, info: dict) -> int:
        mask: np.ndarray = info["legal_action_mask"]
        board = _obs_to_board(obs)

        best_action: int | None = None
        best_score = -float("inf")

        for a in range(4):
            if not mask[a]:
                continue

            next_board = slide_board(board, a)
            score = self._score_board(next_board)

            if score > best_score:
                best_score = score
                best_action = a

        assert best_action is not None
        return int(best_action)

    def _score_board(self, board: np.ndarray) -> float:
        empty_cells = int(np.sum(board == 0))
        max_tile = int(board.max())

        corners = [
            board[0, 0],
            board[0, 3],
            board[3, 0],
            board[3, 3],
        ]
        max_in_corner = 1 if max_tile in corners else 0

        # Smoothness: penalize big jumps between neighboring tiles
        smoothness_penalty = 0.0
        for r in range(4):
            for c in range(4):
                if board[r, c] == 0:
                    continue
                current = np.log2(board[r, c])
                if r + 1 < 4 and board[r + 1, c] != 0:
                    smoothness_penalty += abs(current - np.log2(board[r + 1, c]))
                if c + 1 < 4 and board[r, c + 1] != 0:
                    smoothness_penalty += abs(current - np.log2(board[r, c + 1]))

        return (
            10.0 * empty_cells
            + 2.0 * np.log2(max(max_tile, 2))
            + 15.0 * max_in_corner
            - 1.0 * smoothness_penalty
        )