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
