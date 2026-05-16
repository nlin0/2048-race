from race2048.agents.baselines import RandomAgent
from race2048.env import Game2048Env
from race2048.evaluation import EpisodeReport, aggregate, run_episode


def test_run_episode_reproducible():
    env1 = Game2048Env(seed=0)
    env2 = Game2048Env(seed=0)
    r1 = run_episode(env1, RandomAgent(seed=7).act, seed=42)
    r2 = run_episode(env2, RandomAgent(seed=7).act, seed=42)
    assert r1 == r2


def test_aggregate_empty():
    a = aggregate([])
    assert a.episodes == 0


def test_aggregate_non_empty_std_and_sums():
    reports = [
        EpisodeReport(10, 9, False, None, 512, 1000, 1),
        EpisodeReport(10, 10, False, None, 1024, 2000, 0),
    ]
    a = aggregate(reports)
    assert a.episodes == 2
    assert a.std_max_tile > 0.0
    assert a.median_max_tile == 768.0
    assert a.mean_final_tile_sum == 1500.0
    assert a.mean_invalid_moves == 0.5
