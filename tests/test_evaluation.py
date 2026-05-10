from race2048.agents.baselines import RandomAgent
from race2048.env import Game2048Env
from race2048.evaluation import aggregate, run_episode


def test_run_episode_reproducible():
    env1 = Game2048Env(seed=0)
    env2 = Game2048Env(seed=0)
    r1 = run_episode(env1, RandomAgent(seed=7).act, seed=42)
    r2 = run_episode(env2, RandomAgent(seed=7).act, seed=42)
    assert r1 == r2


def test_aggregate_empty():
    a = aggregate([])
    assert a.episodes == 0
