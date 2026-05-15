import numpy as np
import pytest

from race2048.board import Action
from race2048.env import Game2048Env
from race2048.snake_strategy import (
    apply_snake_strategy_mask,
    max_tile_in_any_corner,
    max_tile_in_top_left_corner,
    monotonic_edge_score,
    snake_path_monotonic_score,
    strategy_forbidden_mask,
    strategy_potential,
)


def test_forbid_down_when_max_on_top_row():
    b = np.zeros((4, 4), dtype=np.int32)
    b[0, 2] = 128
    b[2, 2] = 128
    f = strategy_forbidden_mask(b)
    assert f[int(Action.DOWN)]


def test_forbid_right_when_max_on_left_col():
    b = np.zeros((4, 4), dtype=np.int32)
    b[2, 0] = 64
    b[0, 1] = 32
    f = strategy_forbidden_mask(b)
    assert f[int(Action.RIGHT)]


def test_apply_mask_falls_back_when_only_forbidden_remains():
    """If strategy would forbid all legal moves, keep full legal set."""
    legal = np.zeros(4, dtype=np.bool_)
    legal[int(Action.DOWN)] = True
    b = np.zeros((4, 4), dtype=np.int32)
    b[0, 0] = 2
    m = apply_snake_strategy_mask(legal, b)
    assert m[int(Action.DOWN)]


def test_monotonic_edge_score_non_negative():
    b = np.array(
        [[8, 4, 2, 0], [4, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int32,
    )
    assert monotonic_edge_score(b) >= 0.0


def test_corner_max_detected_tl():
    b = np.zeros((4, 4), dtype=np.int32)
    b[0, 0] = 16
    b[0, 1] = 8
    assert max_tile_in_top_left_corner(b)


def test_max_tile_any_corner():
    b = np.zeros((4, 4), dtype=np.int32)
    b[3, 3] = 32
    b[0, 0] = 4
    assert max_tile_in_any_corner(b)


def test_snake_path_monotonic_positive():
    b = np.array(
        [[16, 8, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.int32,
    )
    assert snake_path_monotonic_score(b) > 0.0


def test_potential_shaping_finite():
    b = np.ones((4, 4), dtype=np.int32) * 2
    p = strategy_potential(
        b, corner_weight=0.1, monotonic_weight=0.02, empty_weight=0.05
    )
    assert np.isfinite(p)


def test_snake_env_masks_and_never_empty_legal():
    env = Game2048Env(seed=42, max_steps=500, snake_top_left=True)
    obs, info = env.reset(seed=0)
    assert info["legal_action_mask"].any()
    assert info["legal_action_mask_raw"].any()
    for _ in range(300):
        m = info["legal_action_mask"]
        a = int(np.random.choice(np.flatnonzero(m)))
        obs, _r, term, trunc, info = env.step(a)
        if not (term or trunc):
            assert info["legal_action_mask"].any()
        if term or trunc:
            obs, info = env.reset(seed=_ + 1)
