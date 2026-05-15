"""Gymnasium-style environment wrapping `Game2048` for RL and evaluation."""

from __future__ import annotations

import math
from typing import Any, Literal, SupportsFloat, cast

import numpy as np
from gymnasium import Env, spaces

from race2048.board import Action, Game2048, StepResult, merge_score_for_slide

AnchorCorner = Literal["bl", "br", "tl", "tr"]

# log2 view of the board: empty cells = 0, tile 2 -> 1, 4 -> 2, ..., 2048 -> 11
_OBS_LOW = np.zeros((4, 4), dtype=np.float32)
_OBS_HIGH = np.full((4, 4), 16.0, dtype=np.float32)  # up to 2^16


def encode_board_log2(board: np.ndarray) -> np.ndarray:
    """Same encoding as Game2048Env observations (shape 4×4 float32)."""
    b = np.asarray(board, dtype=np.int32)
    o = np.zeros((4, 4), dtype=np.float32)
    mask = b > 0
    o[mask] = np.log2(b[mask].astype(np.float64)).astype(np.float32)
    return o


_CORNERS: tuple[tuple[int, int], ...] = ((0, 0), (0, 3), (3, 0), (3, 3))

_ANCHOR_CELL: dict[str, tuple[int, int]] = {
    "bl": (3, 0),
    "br": (3, 3),
    "tl": (0, 0),
    "tr": (0, 3),
}


def anchor_cell(corner: AnchorCorner) -> tuple[int, int]:
    return _ANCHOR_CELL[corner]


def orient_log2_for_anchor(L: np.ndarray, corner: AnchorCorner) -> np.ndarray:
    """Rotate/flip log2 grid so the strategy anchor sits at bottom-left (3,0) in the view."""
    if corner == "bl":
        return L
    if corner == "br":
        return L[:, ::-1]
    if corner == "tl":
        return L[::-1, :]
    if corner == "tr":
        return L[::-1, ::-1]
    raise ValueError(f"Unknown anchor corner: {corner!r}")


def max_tile_on_fixed_corner(board: np.ndarray, corner: AnchorCorner) -> bool:
    """True iff the board maximum value sits on the chosen anchor corner."""
    b = np.asarray(board, dtype=np.int32)
    m = int(b.max())
    if m == 0:
        return False
    r, c = anchor_cell(corner)
    return int(b[r, c]) == m


def monotonicity_for_anchor(board: np.ndarray, corner: AnchorCorner) -> float:
    """Weak BL-style monotonicity in log2 space for the orientation where ``corner`` is BL."""
    L = encode_board_log2(board)
    return _monotonicity_bottom_left_log(orient_log2_for_anchor(L, corner))


def edge_fullness_for_anchor(board: np.ndarray, corner: AnchorCorner) -> float:
    """Mean fill ratio (0..1) on the two edges that meet at the anchor corner."""
    b = np.asarray(board, dtype=np.int32)
    if corner == "bl":
        row_fill = np.count_nonzero(b[3]) / 4.0
        col_fill = np.count_nonzero(b[:, 0]) / 4.0
    elif corner == "br":
        row_fill = np.count_nonzero(b[3]) / 4.0
        col_fill = np.count_nonzero(b[:, 3]) / 4.0
    elif corner == "tl":
        row_fill = np.count_nonzero(b[0]) / 4.0
        col_fill = np.count_nonzero(b[:, 0]) / 4.0
    else:  # tr
        row_fill = np.count_nonzero(b[0]) / 4.0
        col_fill = np.count_nonzero(b[:, 3]) / 4.0
    return 0.5 * (row_fill + col_fill)


def danger_action_for_corner(corner: AnchorCorner) -> int:
    """Action index that slides tiles away from a bottom anchor (bl/br) or top anchor (tl/tr)."""
    if corner in ("bl", "br"):
        return int(Action.UP)
    return int(Action.DOWN)


def max_tile_on_any_corner(board: np.ndarray) -> bool:
    """True if some cell with the board maximum sits on a corner (handles multiple max tiles)."""
    m = int(np.asarray(board).max())
    if m == 0:
        return False
    b = np.asarray(board, dtype=np.int32)
    return any(int(b[r, c]) == m for r, c in _CORNERS)


def _monotonicity_bottom_left_log(L: np.ndarray) -> float:
    """Weak monotonicity toward bottom-left in log2 space: fraction of adjacent pairs that
    do not increase when moving right or up (0..1). Empty cells contribute 0 in L.
    """
    pts = 0
    for r in range(4):
        for c in range(3):
            if float(L[r, c]) >= float(L[r, c + 1]):
                pts += 1
    for r in range(1, 4):
        for c in range(4):
            if float(L[r, c]) >= float(L[r - 1, c]):
                pts += 1
    return pts / 24.0


def max_monotonicity_score_log2(board: np.ndarray) -> float:
    """Best bottom-left monotonicity over the four axis reflections (one corner anchor each)."""
    L = encode_board_log2(board)
    variants = (L, L[:, ::-1], L[::-1, :], L[::-1, ::-1])
    return max(_monotonicity_bottom_left_log(v) for v in variants)


class Game2048Env(Env):
    """2048 with invalid moves penalized; observation is log2 of tile values (0 if empty).

    Layout shaping (after valid moves) by default uses a **fixed anchor corner** (``anchor_corner``):
    bonus when the max tile sits on that corner, monotonicity/snake score in the matching
    orientation, mean fill on the two edges meeting the anchor, global empty-cell bonus, a
    reward when the max tile increases (log2 scale), and a soft penalty for the “danger” axis
    move (e.g. UP for bottom anchors) while the previous board still had the max on the anchor.

    Set ``legacy_corner_shaping=True`` to restore the older “any corner + best of four
    reflections” monotonicity (for matching old checkpoints). Set individual ``*_bonus`` /
    ``*_penalty`` weights to ``0.0`` to disable those terms.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_steps: int | None = 10_000,
        terminate_on_win: bool = False,
        invalid_move_penalty: float = 1.0,
        step_cost: float = 0.005,
        win_bonus: float = 50.0,
        anchor_corner: AnchorCorner = "bl",
        legacy_corner_shaping: bool = False,
        corner_max_bonus: float = 0.08,
        monotonicity_bonus: float = 0.08,
        empty_tile_bonus: float = 0.1,
        edge_fill_bonus: float = 0.05,
        max_log2_increase_bonus: float = 0.12,
        danger_axis_penalty: float = 0.04,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._max_steps = max_steps
        self._terminate_on_win = terminate_on_win
        self._invalid_penalty = invalid_move_penalty
        self._step_cost = step_cost
        self._win_bonus = win_bonus
        if anchor_corner not in _ANCHOR_CELL:
            raise ValueError(
                f"anchor_corner must be one of {tuple(_ANCHOR_CELL)}, got {anchor_corner!r}"
            )
        self._anchor_corner: AnchorCorner = anchor_corner
        self._legacy_corner_shaping = legacy_corner_shaping
        self._corner_max_bonus = corner_max_bonus
        self._monotonicity_bonus = monotonicity_bonus
        self._empty_tile_bonus = empty_tile_bonus
        self._edge_fill_bonus = edge_fill_bonus
        self._max_log2_increase_bonus = max_log2_increase_bonus
        self._danger_axis_penalty = danger_axis_penalty
        self._game = Game2048(seed=self._rng.integers(0, 2**31 - 1))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=_OBS_LOW,
            high=_OBS_HIGH,
            shape=(4, 4),
            dtype=np.float32,
        )
        self._step_count = 0

    @property
    def game(self) -> Game2048:
        return self._game

    @property
    def max_episode_steps(self) -> int | None:
        return self._max_steps

    def legal_action_mask(self) -> np.ndarray:
        m = np.zeros(4, dtype=np.bool_)
        for a in self._game.legal_actions():
            m[int(a)] = True
        return m

    def _encode_obs(self, board: np.ndarray) -> np.ndarray:
        return encode_board_log2(board)

    def _mono_and_anchor(self, board: np.ndarray) -> tuple[float, bool]:
        if self._legacy_corner_shaping:
            return (
                max_monotonicity_score_log2(board),
                max_tile_on_any_corner(board),
            )
        return (
            monotonicity_for_anchor(board, self._anchor_corner),
            max_tile_on_fixed_corner(board, self._anchor_corner),
        )

    def _soft_danger_penalty(self, prev_board: np.ndarray, action: int) -> float:
        if (
            self._legacy_corner_shaping
            or self._danger_axis_penalty == 0.0
            or not max_tile_on_fixed_corner(
                prev_board, self._anchor_corner
            )
        ):
            return 0.0
        if int(action) == danger_action_for_corner(self._anchor_corner):
            return self._danger_axis_penalty
        return 0.0

    def _reward_shaped(
        self,
        prev_max: int,
        prev_board: np.ndarray,
        action: int,
        res: StepResult,
    ) -> float:
        if not res.valid:
            return -self._invalid_penalty
        merge_pts = merge_score_for_slide(prev_board, action)
        new_max = int(res.board.max())
        r = 0.0
        if merge_pts > 0:
            r += float(math.log2(merge_pts))
        if new_max >= Game2048.WIN_TILE and prev_max < Game2048.WIN_TILE:
            r += self._win_bonus
        if (
            not self._legacy_corner_shaping
            and self._max_log2_increase_bonus != 0.0
            and new_max > prev_max
            and prev_max >= 2
        ):
            r += self._max_log2_increase_bonus * (
                float(math.log2(new_max)) - float(math.log2(prev_max))
            )
        r -= self._soft_danger_penalty(prev_board, action)

        board = res.board
        n_empty = int(np.count_nonzero(board == 0))
        mono, on_anchor = self._mono_and_anchor(board)
        if self._corner_max_bonus != 0.0 and on_anchor:
            r += self._corner_max_bonus
        if self._monotonicity_bonus != 0.0:
            r += self._monotonicity_bonus * mono
        if self._empty_tile_bonus != 0.0:
            r += self._empty_tile_bonus * (n_empty / 16.0)
        if self._edge_fill_bonus != 0.0 and not self._legacy_corner_shaping:
            edge_fill = edge_fullness_for_anchor(board, self._anchor_corner)
            r += self._edge_fill_bonus * edge_fill
        r -= self._step_cost
        return r

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game = Game2048(seed=int(self._rng.integers(0, 2**31 - 1)))
        self._game.reset()
        self._step_count = 0
        obs = self._encode_obs(self._game.board)
        return obs, self._info_dict(mid_reset=True)

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_count += 1
        prev_max = int(self._game.board.max())
        prev_board = self._game.board.copy()
        res = self._game.step(action)
        obs = self._encode_obs(res.board)
        reward = cast(
            SupportsFloat,
            self._reward_shaped(prev_max, prev_board, action, res),
        )
        terminated = res.game_over or (
            self._terminate_on_win and res.won and res.valid
        )
        truncated = (
            self._max_steps is not None and self._step_count >= self._max_steps
        )
        info = self._info_dict(
            res=res,
            prev_board=prev_board,
        )
        if terminated or truncated:
            info["final_max_tile"] = int(res.board.max())
        return obs, reward, terminated, truncated, info

    def _info_dict(
        self,
        *,
        res: StepResult | None = None,
        prev_board: np.ndarray | None = None,
        mid_reset: bool = False,
    ) -> dict[str, Any]:
        b = self._game.board
        max_tile = int(b.max())
        info: dict[str, Any] = {
            "max_tile": max_tile,
            "legal_actions": [int(a) for a in self._game.legal_actions()],
            "legal_action_mask": self.legal_action_mask().copy(),
            "step_count": self._step_count,
        }
        if mid_reset:
            info["reached_2048"] = False
            info["moves_to_2048"] = None
            info["valid"] = True
            info["won"] = info["max_tile"] >= Game2048.WIN_TILE
            info["just_reached_2048"] = False
            info["anchor_corner"] = self._anchor_corner
            info["legacy_corner_shaping"] = self._legacy_corner_shaping
            return info

        assert res is not None
        info["valid"] = res.valid
        info["won"] = res.won
        info["reached_2048"] = max_tile >= Game2048.WIN_TILE
        if prev_board is not None and max_tile >= Game2048.WIN_TILE and not (
            int(prev_board.max()) >= Game2048.WIN_TILE
        ):
            info["just_reached_2048"] = True
        else:
            info["just_reached_2048"] = False
        if res.valid:
            b = self._game.board
            n_empty = int(np.count_nonzero(b == 0))
            mono, on_anchor = self._mono_and_anchor(b)
            edge_fill = (
                0.0
                if self._legacy_corner_shaping
                else edge_fullness_for_anchor(b, self._anchor_corner)
            )
            layout = 0.0
            if self._corner_max_bonus != 0.0 and on_anchor:
                layout += self._corner_max_bonus
            if self._monotonicity_bonus != 0.0:
                layout += self._monotonicity_bonus * mono
            if self._empty_tile_bonus != 0.0:
                layout += self._empty_tile_bonus * (n_empty / 16.0)
            if self._edge_fill_bonus != 0.0 and not self._legacy_corner_shaping:
                layout += self._edge_fill_bonus * edge_fill
            info["n_empty"] = n_empty
            info["monotonicity"] = mono
            info["max_on_corner"] = on_anchor
            info["max_on_anchor"] = on_anchor
            info["anchor_corner"] = self._anchor_corner
            info["edge_fullness"] = edge_fill
            info["legacy_corner_shaping"] = self._legacy_corner_shaping
            info["layout_reward"] = layout
        return info
