from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from race2048.compare_policies import run_comparison
from race2048.dqn.qnet import QNetwork


def test_run_comparison_identical_weights_zero_delta(tmp_path: Path) -> None:
    ck = tmp_path / "same.pt"
    net = QNetwork(hidden_dim=64)
    torch.save(
        {"policy_state": net.state_dict(), "qnet_config": net.config_dict()},
        ck,
    )
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "out"
    summary = run_comparison(
        repo_root=repo,
        path_a=ck,
        path_b=ck,
        label_a="X",
        label_b="Y",
        episodes=4,
        base_seed=123,
        max_steps=800,
        gamma=0.99,
        out_dir=out,
        device=torch.device("cpu"),
    )
    assert summary["headline"]["mean_max_tile_delta_a_minus_b"] == 0.0
    assert summary["headline"]["mean_final_sum_delta_a_minus_b"] == 0.0
    paired = summary["paired"]
    assert paired["max_tile"]["wins_X"] == 0
    assert paired["max_tile"]["wins_Y"] == 0
    assert paired["max_tile"]["ties"] == 4
    assert (out / "summary.json").is_file()
    assert (out / "episodes_combined.csv").is_file()
