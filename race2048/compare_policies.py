"""Matched-seed rollouts to compare two greedy DQN checkpoints (e.g. CNN vs MLP).

Writes per-episode CSV files and a JSON summary for reports.

Example::

    PYTHONPATH=. python -m race2048.compare_policies \\
        --a checkpoints/dqn_cnn_n3_1.pt \\
        --b MLP/checkpoints/dqn_2_1.pt \\
        --episodes 500 \\
        --out-dir reports/cnn_vs_mlp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from race2048.demo_bots import resolve_checkpoint_under
from race2048.env import Game2048Env
from race2048.eval_cli import _greedy_dqn_act_fn, _load_ckpt
from race2048.evaluation import EpisodeReport, aggregate, run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_ckpt(repo: Path, p: Path) -> Path:
    if p.is_absolute() and p.is_file():
        return p.resolve()
    joined = (repo / p).resolve()
    if joined.is_file():
        return joined
    name = Path(p).name
    r = resolve_checkpoint_under(repo, name, name)
    if r is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {p} (tried {joined} and checkpoints/ + CNN|MLP/checkpoints/)"
        )
    return r


def _env_kw_from_checkpoint(ck: dict[str, Any] | Any, gamma: float) -> dict[str, Any]:
    if not isinstance(ck, dict):
        return {}
    if ck.get("env_snake_top_left"):
        return {"snake_top_left": True, "snake_shaping_gamma": float(gamma)}
    return {}


def _policy_meta(label: str, path: Path, ck: dict[str, Any] | Any, gamma: float) -> dict[str, Any]:
    ekw = _env_kw_from_checkpoint(ck, gamma)
    meta: dict[str, Any] = {
        "label": label,
        "checkpoint": str(path),
        "snake_top_left": bool(ekw.get("snake_top_left", False)),
        "qnet_type": None,
    }
    if isinstance(ck, dict) and isinstance(ck.get("qnet_config"), dict):
        meta["qnet_type"] = ck["qnet_config"].get("qnet_type")
    return meta


def _paired_comparison(
    reports_a: list[EpisodeReport],
    reports_b: list[EpisodeReport],
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Head-to-head stats on matched seeds (same episode index)."""
    n = len(reports_a)
    assert n == len(reports_b)

    max_wins_a = max_wins_b = max_ties = 0
    sum_wins_a = sum_wins_b = sum_ties = 0
    reach_only_a = reach_only_b = reach_both = reach_neither = 0

    for ra, rb in zip(reports_a, reports_b):
        if ra.max_tile > rb.max_tile:
            max_wins_a += 1
        elif ra.max_tile < rb.max_tile:
            max_wins_b += 1
        else:
            max_ties += 1

        if ra.final_tile_sum > rb.final_tile_sum:
            sum_wins_a += 1
        elif ra.final_tile_sum < rb.final_tile_sum:
            sum_wins_b += 1
        else:
            sum_ties += 1

        a_2048, b_2048 = ra.reached_2048, rb.reached_2048
        if a_2048 and b_2048:
            reach_both += 1
        elif a_2048:
            reach_only_a += 1
        elif b_2048:
            reach_only_b += 1
        else:
            reach_neither += 1

    return {
        "episodes": n,
        "max_tile": {
            f"wins_{label_a}": max_wins_a,
            f"wins_{label_b}": max_wins_b,
            "ties": max_ties,
            f"win_rate_{label_a}": max_wins_a / n if n else 0.0,
            f"win_rate_{label_b}": max_wins_b / n if n else 0.0,
            "tie_rate": max_ties / n if n else 0.0,
        },
        "final_tile_sum": {
            f"wins_{label_a}": sum_wins_a,
            f"wins_{label_b}": sum_wins_b,
            "ties": sum_ties,
            f"win_rate_{label_a}": sum_wins_a / n if n else 0.0,
            f"win_rate_{label_b}": sum_wins_b / n if n else 0.0,
            "tie_rate": sum_ties / n if n else 0.0,
        },
        "reach_2048": {
            f"only_{label_a}": reach_only_a,
            f"only_{label_b}": reach_only_b,
            "both": reach_both,
            "neither": reach_neither,
        },
    }


def _stats_as_dict(stats: Any, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "episodes": stats.episodes,
        "reach_2048_rate": stats.reach_2048_rate,
        "mean_moves_to_2048": stats.mean_moves_to_2048,
        "median_moves_to_2048": stats.median_moves_to_2048,
        "mean_valid_moves": stats.mean_valid_moves,
        "mean_max_tile": stats.mean_max_tile,
        "std_max_tile": stats.std_max_tile,
        "median_max_tile": stats.median_max_tile,
        "mean_final_tile_sum": stats.mean_final_tile_sum,
        "mean_invalid_moves": stats.mean_invalid_moves,
    }


def _write_episode_csv(path: Path, label: str, reports: list[EpisodeReport], seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "label",
                "env_seed",
                "steps",
                "valid_moves",
                "invalid_moves",
                "max_tile",
                "final_tile_sum",
                "reached_2048",
                "moves_to_2048",
            ]
        )
        for i, (r, sd) in enumerate(zip(reports, seeds)):
            w.writerow(
                [
                    i,
                    label,
                    sd,
                    r.steps,
                    r.valid_moves,
                    r.invalid_moves,
                    r.max_tile,
                    r.final_tile_sum,
                    int(r.reached_2048),
                    "" if r.moves_to_2048 is None else r.moves_to_2048,
                ]
            )


def run_comparison(
    *,
    repo_root: Path,
    path_a: Path,
    path_b: Path,
    label_a: str = "A",
    label_b: str = "B",
    episodes: int,
    base_seed: int,
    max_steps: int,
    gamma: float,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run ``episodes`` games per policy with the same env seeds; write CSV + JSON under ``out_dir``."""
    resolved_a = _resolve_ckpt(repo_root, path_a)
    resolved_b = _resolve_ckpt(repo_root, path_b)
    ck_a = _load_ckpt(resolved_a, device)
    ck_b = _load_ckpt(resolved_b, device)
    env_kw_a = _env_kw_from_checkpoint(ck_a, gamma)
    env_kw_b = _env_kw_from_checkpoint(ck_b, gamma)

    act_a = _greedy_dqn_act_fn(resolved_a, device=device)
    act_b = _greedy_dqn_act_fn(resolved_b, device=device)

    seeds = [base_seed + i for i in range(episodes)]
    reports_a: list[EpisodeReport] = []
    reports_b: list[EpisodeReport] = []
    for sd in seeds:
        env_a = Game2048Env(seed=sd, max_steps=max_steps, **env_kw_a)
        env_b = Game2048Env(seed=sd, max_steps=max_steps, **env_kw_b)
        reports_a.append(run_episode(env_a, act_a, seed=sd))
        reports_b.append(run_episode(env_b, act_b, seed=sd))

    agg_a = aggregate(reports_a)
    agg_b = aggregate(reports_b)
    meta_a = _policy_meta(label_a, resolved_a, ck_a, gamma)
    meta_b = _policy_meta(label_b, resolved_b, ck_b, gamma)

    out = out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    safe_a = label_a.lower().replace(" ", "_")
    safe_b = label_b.lower().replace(" ", "_")
    _write_episode_csv(out / f"episodes_{safe_a}.csv", label_a, reports_a, seeds)
    _write_episode_csv(out / f"episodes_{safe_b}.csv", label_b, reports_b, seeds)

    combined = out / "episodes_combined.csv"
    with combined.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "env_seed",
                f"label_{safe_a}",
                f"max_tile_{safe_a}",
                f"sum_{safe_a}",
                f"label_{safe_b}",
                f"max_tile_{safe_b}",
                f"sum_{safe_b}",
                "max_tile_diff_a_minus_b",
                "sum_diff_a_minus_b",
            ]
        )
        for i, sd in enumerate(seeds):
            ra, rb = reports_a[i], reports_b[i]
            w.writerow(
                [
                    i,
                    sd,
                    label_a,
                    ra.max_tile,
                    ra.final_tile_sum,
                    label_b,
                    rb.max_tile,
                    rb.final_tile_sum,
                    ra.max_tile - rb.max_tile,
                    ra.final_tile_sum - rb.final_tile_sum,
                ]
            )

    paired = _paired_comparison(reports_a, reports_b, label_a=label_a, label_b=label_b)
    summary: dict[str, Any] = {
        "episodes": episodes,
        "base_seed": base_seed,
        "max_steps": max_steps,
        "gamma": gamma,
        "policies": [_stats_as_dict(agg_a, meta_a), _stats_as_dict(agg_b, meta_b)],
        "paired": paired,
        "headline": {
            "reach_2048_delta_a_minus_b": agg_a.reach_2048_rate - agg_b.reach_2048_rate,
            "mean_max_tile_delta_a_minus_b": agg_a.mean_max_tile - agg_b.mean_max_tile,
            "mean_final_sum_delta_a_minus_b": agg_a.mean_final_tile_sum - agg_b.mean_final_tile_sum,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare two greedy DQN checkpoints with matched seeds per episode."
    )
    p.add_argument(
        "--a",
        type=Path,
        dest="path_a",
        default=Path("checkpoints/dqn_cnn_n3_1.pt"),
        help="First policy .pt (path or basename resolved under repo)",
    )
    p.add_argument(
        "--b",
        type=Path,
        dest="path_b",
        default=Path("MLP/checkpoints/dqn_2_1.pt"),
        help="Second policy .pt",
    )
    p.add_argument("--label-a", type=str, default="A", help="Short name for first policy in CSV/JSON")
    p.add_argument("--label-b", type=str, default="B", help="Short name for second policy")
    p.add_argument("--episodes", type=int, default=200, help="Rollouts per policy (same seed sequence)")
    p.add_argument("--seed", type=int, default=10_000, help="Base env seed; episode i uses seed+i")
    p.add_argument("--max-steps", type=int, default=50_000, help="Truncation cap per episode")
    p.add_argument("--gamma", type=float, default=0.99, help="Snake MDP γ when checkpoint enables snake")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/compare_dqn"),
        help="Directory for episodes_*.csv, episodes_combined.csv, summary.json",
    )
    p.add_argument("--device", type=str, default=None, help="torch device (default: cpu)")
    args = p.parse_args()

    dev = torch.device(args.device or "cpu")
    summary = run_comparison(
        repo_root=REPO_ROOT,
        path_a=args.path_a,
        path_b=args.path_b,
        label_a=args.label_a,
        label_b=args.label_b,
        episodes=args.episodes,
        base_seed=args.seed,
        max_steps=args.max_steps,
        gamma=args.gamma,
        out_dir=args.out_dir,
        device=dev,
    )
    print(json.dumps(summary["policies"], indent=2))
    print(f"\nPaired head-to-head ({args.label_a} vs {args.label_b}, same seeds):")
    print(json.dumps(summary["paired"], indent=2))
    print(f"\nHeadline deltas ({args.label_a} − {args.label_b}):")
    print(json.dumps(summary["headline"], indent=2))
    print(f"\nWrote CSV + summary under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
