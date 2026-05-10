from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import numpy as np


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class StepResult(NamedTuple):
    """Outcome of attempting a single move."""

    board: np.ndarray
    valid: bool
    game_over: bool
    won: bool


def _merge_line_left(line: np.ndarray) -> np.ndarray:
    """Slide non-zero tiles left and merge equal neighbors once per pair.

    `line` is length 4. Classic 2048: [2, 2, 2, 0] -> [4, 2, 0, 0].
    """
    tiles = [int(x) for x in line if x != 0]
    merged: list[int] = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] * 2)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    while len(merged) < 4:
        merged.append(0)
    return np.array(merged[:4], dtype=np.int32)


def _move_left(board: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.int32)
    for r in range(4):
        out[r] = _merge_line_left(board[r])
    return out


def _move_right(board: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.int32)
    for r in range(4):
        out[r] = _merge_line_left(board[r, ::-1])[::-1]
    return out


def _move_up(board: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.int32)
    for c in range(4):
        col = board[:, c]
        out[:, c] = _merge_line_left(col)
    return out


def _move_down(board: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.int32)
    for c in range(4):
        col = board[:, c]
        out[:, c] = _merge_line_left(col[::-1])[::-1]
    return out


def _apply_action(board: np.ndarray, action: Action) -> np.ndarray:
    if action == Action.LEFT:
        return _move_left(board)
    if action == Action.RIGHT:
        return _move_right(board)
    if action == Action.UP:
        return _move_up(board)
    if action == Action.DOWN:
        return _move_down(board)
    raise ValueError(f"Unknown action: {action}")


def has_legal_move(board: np.ndarray) -> bool:
    """True if at least one action would change the board."""
    for a in Action:
        if not np.array_equal(board, _apply_action(board, a)):
            return True
    return False


def has_won(board: np.ndarray, target: int = 2048) -> bool:
    return bool(np.any(board >= target))


def slide_board(board: np.ndarray, action: Action | int) -> np.ndarray:
    """Slide and merge only (no new tile). Used for lookahead / baselines."""
    return _apply_action(np.asarray(board, dtype=np.int32), Action(action))


class Game2048:
    """Simulates a 4×4 2048 board (NumPy)."""

    SIZE = 4
    WIN_TILE = 2048

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._board = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)

    @property
    def board(self) -> np.ndarray:
        """Current grid; copy to avoid accidental mutation."""
        return self._board.copy()

    def reset(self) -> np.ndarray:
        """Clear the board, spawn two starting tiles, return the new state."""
        self._board.fill(0)
        self._spawn_tile()
        self._spawn_tile()
        return self.board

    def _spawn_tile(self) -> None:
        empty = np.argwhere(self._board == 0)
        if len(empty) == 0:
            return
        idx = self._rng.integers(len(empty))
        r, c = empty[idx]
        self._board[r, c] = 2 if self._rng.random() < 0.9 else 4

    def step(self, action: Action | int) -> StepResult:
        """Apply a move. Invalid if the board does not change (no spawn)."""
        a = Action(action)
        before = self._board
        after = _apply_action(before, a)

        if np.array_equal(before, after):
            return StepResult(
                board=self.board,
                valid=False,
                game_over=not has_legal_move(before),
                won=has_won(before, self.WIN_TILE),
            )

        self._board = after
        self._spawn_tile()

        return StepResult(
            board=self.board,
            valid=True,
            game_over=not has_legal_move(self._board),
            won=has_won(self._board, self.WIN_TILE),
        )

    def legal_actions(self) -> list[Action]:
        """Actions that would change the current board."""
        return [a for a in Action if not np.array_equal(self._board, _apply_action(self._board, a))]
