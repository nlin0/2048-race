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


def _merge_line_left_scored(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide + merge left; return new line and classic 2048 merge score (sum of merged tile values)."""
    tiles = [int(x) for x in line if x != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            merged.append(v)
            score += v
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    while len(merged) < 4:
        merged.append(0)
    return np.array(merged[:4], dtype=np.int32), score


def _merge_line_left(line: np.ndarray) -> np.ndarray:
    """Slide non-zero tiles left and merge equal neighbors once per pair.

    `line` is length 4. Classic 2048: [2, 2, 2, 0] -> [4, 2, 0, 0].
    """
    out, _ = _merge_line_left_scored(line)
    return out


def _move_left_scored(board: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros((4, 4), dtype=np.int32)
    total = 0
    for r in range(4):
        line, s = _merge_line_left_scored(board[r])
        out[r] = line
        total += s
    return out, total


def _move_left(board: np.ndarray) -> np.ndarray:
    out, _ = _move_left_scored(board)
    return out


def _move_right_scored(board: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros((4, 4), dtype=np.int32)
    total = 0
    for r in range(4):
        line, s = _merge_line_left_scored(board[r, ::-1])
        out[r] = line[::-1]
        total += s
    return out, total


def _move_right(board: np.ndarray) -> np.ndarray:
    out, _ = _move_right_scored(board)
    return out


def _move_up_scored(board: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros((4, 4), dtype=np.int32)
    total = 0
    for c in range(4):
        col = board[:, c]
        line, s = _merge_line_left_scored(col)
        out[:, c] = line
        total += s
    return out, total


def _move_up(board: np.ndarray) -> np.ndarray:
    out, _ = _move_up_scored(board)
    return out


def _move_down_scored(board: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros((4, 4), dtype=np.int32)
    total = 0
    for c in range(4):
        col = board[:, c]
        line, s = _merge_line_left_scored(col[::-1])
        out[:, c] = line[::-1]
        total += s
    return out, total


def _move_down(board: np.ndarray) -> np.ndarray:
    out, _ = _move_down_scored(board)
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


def _apply_action_scored(board: np.ndarray, action: Action) -> tuple[np.ndarray, int]:
    """Slide+merge only (no spawn); score is classic 2048 merge sum for that move."""
    if action == Action.LEFT:
        return _move_left_scored(board)
    if action == Action.RIGHT:
        return _move_right_scored(board)
    if action == Action.UP:
        return _move_up_scored(board)
    if action == Action.DOWN:
        return _move_down_scored(board)
    raise ValueError(f"Unknown action: {action}")


def merge_score_for_slide(board: np.ndarray, action: Action | int) -> int:
    """Merge points for `action` on `board` before tile spawn; 0 if move does not change the grid."""
    b = np.asarray(board, dtype=np.int32)
    slid, score = _apply_action_scored(b, Action(action))
    if np.array_equal(b, slid):
        return 0
    return int(score)


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
