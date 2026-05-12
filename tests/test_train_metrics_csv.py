"""CSV metrics logging from train_dqn."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from race2048.train_dqn import _append_metrics_csv, _loss_window_stats, train


def test_loss_window_stats():
    d: deque[float] = deque([1.0, float("nan"), 3.0], maxlen=10)
    mean, nf, nt = _loss_window_stats(d)
    assert mean == pytest.approx(2.0)
    assert nf == 2
    assert nt == 3


def test_append_metrics_csv_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.csv"
    _append_metrics_csv(
        p,
        {
            "event": "env_step",
            "wall_time_s": 1.5,
            "env_step": 100,
            "episode": 2,
            "epsilon": 0.99,
            "buffer_size": 100,
            "train_steps": 80,
            "mean_loss": 0.5,
            "loss_n_finite": 10,
            "loss_n_total": 10,
            "win_rate_pct": "",
            "mean_return": "",
            "mean_ep_len": "",
            "mean_max_tile": "",
            "best_max_tile": "",
        },
    )
    _append_metrics_csv(
        p,
        {
            "event": "episode",
            "wall_time_s": 2.0,
            "env_step": 200,
            "episode": 50,
            "epsilon": 0.9,
            "buffer_size": 200,
            "train_steps": 180,
            "mean_loss": 1.25,
            "loss_n_finite": 100,
            "loss_n_total": 100,
            "win_rate_pct": 0.0,
            "mean_return": 100.5,
            "mean_ep_len": 90.0,
            "mean_max_tile": 64.0,
            "best_max_tile": 128,
        },
    )
    text = p.read_text(encoding="utf-8")
    assert "event" in text
    assert "env_step" in text
    assert text.count("\n") >= 3  # header + 2 rows


def test_train_writes_metrics_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "run.csv"
    train(
        total_env_steps=4000,
        learning_starts=50,
        batch_size=32,
        buffer_size=10_000,
        train_frequency=1,
        env_seed=0,
        log_every_episodes=10,
        log_every_env_steps=500,
        metrics_csv=csv_path,
        eps_decay_steps=10_000,
    )
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
