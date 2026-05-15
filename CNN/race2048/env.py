"""Gymnasium-style environment wrapping `Game2048` for RL and evaluation."""

from __future__ import annotations

import math
from typing import Any, SupportsFloat, cast

import numpy as np
from gymnasium import Env, spaces

from race2048.board import Game2048, StepResult, merge_score_for_slide
from race2048.snake_strategy import (
    apply_snake_strategy_mask,
    strategy_potential,
)

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


class Game2048Env(Env):
    """2048 with invalid moves penalized; observation is log2 of tile values (0 if empty)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_steps: int | None = 10_000,
        terminate_on_win: bool = False,
        invalid_move_penalty: float = 1.0,
        step_cost: float = 0.005,
        win_bonus: float = 120.0,
        snake_top_left: bool = False,
        snake_corner_weight: float = 0.05,
        snake_monotonic_weight: float = 0.004,
        snake_empty_weight: float = 0.12,
        snake_shaping_gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._max_steps = max_steps
        self._terminate_on_win = terminate_on_win
        self._invalid_penalty = invalid_move_penalty
        self._step_cost = step_cost
        self._win_bonus = win_bonus
        self._snake_top_left = snake_top_left
        self._snake_corner_w = float(snake_corner_weight)
        self._snake_mono_w = float(snake_monotonic_weight)
        self._snake_empty_w = float(snake_empty_weight)
        self._snake_gamma = float(snake_shaping_gamma)
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
        if self._snake_top_left:
            m = apply_snake_strategy_mask(m, self._game.board)
        return m

    def _encode_obs(self, board: np.ndarray) -> np.ndarray:
        return encode_board_log2(board)

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
        r -= self._step_cost
        if self._snake_top_left:
            phi_s = strategy_potential(
                prev_board,
                corner_weight=self._snake_corner_w,
                monotonic_weight=self._snake_mono_w,
                empty_weight=self._snake_empty_w,
            )
            phi_sp = strategy_potential(
                res.board,
                corner_weight=self._snake_corner_w,
                monotonic_weight=self._snake_mono_w,
                empty_weight=self._snake_empty_w,
            )
            r += self._snake_gamma * phi_sp - phi_s
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
        raw_legal = np.zeros(4, dtype=np.bool_)
        for a in self._game.legal_actions():
            raw_legal[int(a)] = True
        info: dict[str, Any] = {
            "max_tile": max_tile,
            "legal_actions": [int(a) for a in self._game.legal_actions()],
            "legal_action_mask_raw": raw_legal.copy(),
            "legal_action_mask": self.legal_action_mask().copy(),
            "step_count": self._step_count,
        }
        if mid_reset:
            info["reached_2048"] = False
            info["moves_to_2048"] = None
            info["valid"] = True
            info["won"] = info["max_tile"] >= Game2048.WIN_TILE
            info["just_reached_2048"] = False
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
        return info
