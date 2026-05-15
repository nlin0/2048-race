"""Choose bot actions for demos (web server, bot-vs-bot)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from race2048.agents.baselines import GreedyEmptyAgent, OrderedAgent
from race2048.board import Game2048
from race2048.dqn.qnet import load_qnetwork_from_checkpoint
from race2048.env import encode_board_log2

_DQN_NETS: dict[str, torch.nn.Module] = {}


def legal_mask_numpy(game: Game2048) -> np.ndarray:
    m = np.zeros(4, dtype=np.bool_)
    for a in game.legal_actions():
        m[int(a)] = True
    return m


def _obs_and_info(game: Game2048) -> tuple[np.ndarray, dict]:
    obs = encode_board_log2(game.board)
    mask = legal_mask_numpy(game)
    return obs, {"legal_action_mask": mask}


def resolve_checkpoint_under(root: Path, rel: str | None, default_filename: str) -> Path | None:
    """Resolve a basename-only ``*.pt`` for demos and the web UI.

    Search order (first hit wins): ``<root>/checkpoints/``, then
    ``<root>/CNN/checkpoints/``, ``<root>/MLP/checkpoints/`` so presets work when
    weights only exist under the subproject trees.
    """
    fname = Path((rel or default_filename).strip()).name  # disallow path traversal
    if not fname:
        return None
    root_r = root.resolve()
    primary = (root_r / "checkpoints").resolve()
    primary.mkdir(parents=True, exist_ok=True)
    candidates = [
        primary,
        (root_r / "CNN" / "checkpoints").resolve(),
        (root_r / "MLP" / "checkpoints").resolve(),
    ]
    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if not base.is_dir():
            continue
        path = (base / fname).resolve()
        if not path.is_file():
            continue
        try:
            path.relative_to(base)
        except ValueError:
            continue
        return path
    return None


def load_qnetwork(path: Path, device: torch.device) -> torch.nn.Module:
    path = path.resolve()
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _DQN_NETS:
        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location=device)
        _DQN_NETS[key] = load_qnetwork_from_checkpoint(ckpt, device)
    return _DQN_NETS[key]


def pick_random_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    return int(rng.choice(legal))


@torch.no_grad()
def pick_dqn_action(
    game: Game2048,
    checkpoint: Path,
    *,
    device: torch.device | None = None,
) -> int:
    dev = device or torch.device("cpu")
    obs, info = _obs_and_info(game)
    mask = info["legal_action_mask"]
    if not mask.any():
        return 0
    net = load_qnetwork(checkpoint, dev)
    x = torch.from_numpy(obs.astype(np.float32).reshape(-1)).unsqueeze(0).to(dev)
    q = net(x).squeeze(0)
    qb = torch.from_numpy(mask.astype(np.bool_)).to(dev)
    q[~qb] = -float("inf")
    return int(q.argmax().item())


def pick_bot_action(
    policy: str,
    game: Game2048,
    *,
    rng: np.random.Generator,
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> int:
    """Return integer action index 0–3."""
    name = policy.strip().lower()
    obs, info = _obs_and_info(game)
    mask = legal_mask_numpy(game)

    if name == "random":
        return pick_random_action(mask, rng)
    if name == "ordered":
        return OrderedAgent().act(obs, info)
    if name == "greedy":
        return GreedyEmptyAgent().act(obs, info)
    if name == "dqn":
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("DQN checkpoint missing or invalid")
        return pick_dqn_action(game, checkpoint, device=device)
    raise ValueError(f"unknown bot policy {policy!r}")
