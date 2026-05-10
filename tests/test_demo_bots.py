"""Smoke tests for web-driven bot policies."""

import numpy as np

from race2048.board import Action, Game2048
from race2048.demo_bots import pick_bot_action


def test_random_bot_legal():
    g = Game2048(seed=0)
    g.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = pick_bot_action("random", g, rng=rng, checkpoint=None)
        res = g.step(a)
        assert res.valid or res.game_over
        if res.game_over:
            break


def test_ordered_always_legal_when_moves_exist():
    g = Game2048(seed=2)
    g.reset()
    rng = np.random.default_rng(0)
    a = pick_bot_action("ordered", g, rng=rng, checkpoint=None)
    assert a in range(4)
    g.step(Action(a))
