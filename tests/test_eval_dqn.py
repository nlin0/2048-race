"""Greedy DQN evaluation CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from race2048.env import Game2048Env
from race2048.eval_cli import _greedy_dqn_act_fn


def test_greedy_dqn_act_runs_episode(tmp_path: Path) -> None:
    ck = tmp_path / "mini.pt"
    from race2048.dqn.qnet import QNetwork

    net = QNetwork(hidden_dim=256)
    torch.save({"policy_state": net.state_dict()}, ck)

    act = _greedy_dqn_act_fn(ck, device=torch.device("cpu"))
    env = Game2048Env(seed=0, max_steps=200)
    obs, info = env.reset(seed=0)
    for _ in range(50):
        a = act(obs, info)
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert int(env.game.board.max()) >= 2


def test_greedy_dqn_respects_legal_mask(tmp_path: Path) -> None:
    ck = tmp_path / "w.pt"
    from race2048.dqn.qnet import QNetwork

    torch.save({"policy_state": QNetwork().state_dict()}, ck)
    act = _greedy_dqn_act_fn(ck, device=torch.device("cpu"))
    obs = np.zeros((4, 4), dtype=np.float32)
    info = {"legal_action_mask": np.array([True, False, False, False], dtype=np.bool_)}
    assert act(obs, info) == 0
