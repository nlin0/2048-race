import numpy as np
import pytest

from race2048.board import Action
from race2048.env import Game2048Env


def test_reset_has_legal_mask():
    env = Game2048Env(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (4, 4)
    assert info["legal_action_mask"].sum() >= 1
    assert info["legal_action_mask"].shape == (4,)


def test_invalid_step_penalized():
    env = Game2048Env(seed=0, invalid_move_penalty=2.0)
    env.reset(seed=0)
    g = env.game
    g._board[:] = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )
    obs, r, term, trunc, info = env.step(Action.LEFT)
    assert info["valid"] is False
    assert float(r) == -2.0
    assert term is True


def test_episode_terminates():
    env = Game2048Env(seed=42, max_steps=5000)
    env.reset(seed=42)
    agent = __import__("race2048.agents.baselines", fromlist=["RandomAgent"]).RandomAgent(
        seed=99
    )
    steps = 0
    obs, info = env.reset(seed=42)
    while True:
        a = agent.act(obs, info)
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        assert steps < 6000
        if term or trunc:
            break


@pytest.mark.parametrize("name", ["random", "ordered", "greedy"])
def test_baseline_runs(name):
    from race2048.evaluation import run_episode

    mod = __import__("race2048.agents.baselines", fromlist=["*"])
    if name == "random":
        agent = mod.RandomAgent(seed=0)
    elif name == "ordered":
        agent = mod.OrderedAgent()
    else:
        agent = mod.GreedyEmptyAgent()
    env = Game2048Env(seed=0, max_steps=200)
    rep = run_episode(env, agent.act, seed=1)
    assert rep.steps <= 200
