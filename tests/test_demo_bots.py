"""Smoke tests for web-driven bot policies."""

import numpy as np

from race2048.board import Action, Game2048
from race2048.demo_bots import pick_bot_action, resolve_checkpoint_under


def test_resolve_checkpoint_under_checkpoints_only(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "policy.pt").write_bytes(b"y")
    p = resolve_checkpoint_under(tmp_path, "policy.pt", "dqn.pt")
    assert p is not None
    assert p.parent.name == "checkpoints"


def test_resolve_checkpoint_falls_back_to_mlp_subdir(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "MLP" / "checkpoints").mkdir(parents=True)
    (tmp_path / "MLP" / "checkpoints" / "only.pt").write_bytes(b"z")
    p = resolve_checkpoint_under(tmp_path, "only.pt", "dqn.pt")
    assert p is not None
    assert p.resolve() == (tmp_path / "MLP" / "checkpoints" / "only.pt").resolve()


def test_resolve_checkpoint_root_wins_over_subdir(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "MLP" / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "shared.pt").write_bytes(b"root")
    (tmp_path / "MLP" / "checkpoints" / "shared.pt").write_bytes(b"mlp")
    p = resolve_checkpoint_under(tmp_path, "shared.pt", "dqn.pt")
    assert p.read_bytes() == b"root"


def test_resolve_checkpoint_ignores_models_dir(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "models" / "only.pt").write_bytes(b"z")
    p = resolve_checkpoint_under(tmp_path, "only.pt", "dqn.pt")
    assert p is None


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
