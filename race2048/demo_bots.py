"""Choose bot actions for demos (web server, bot-vs-bot)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from race2048.agents.baselines import GreedyEmptyAgent, OrderedAgent, CornerHeuristicAgent
from race2048.board import Game2048
from race2048.dqn.agent import DQNAgent
from race2048.env import encode_board_log2

_DQN_AGENTS: dict[str, DQNAgent] = {}


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
    """Return *.pt under `<root>/checkpoints/` given a basename-only `rel`; never escapes."""
    checkpoints = (root / "checkpoints").resolve()
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = (rel or default_filename).strip()
    fname = Path(name).name
    path = (checkpoints / fname).resolve()
    if not path.is_file():
        return None
    allowed = checkpoints
    try:
        path.relative_to(allowed)
    except ValueError:
        return None
    return path


def load_dqn_agent(path: Path, device: torch.device) -> DQNAgent:
    key = str(path.resolve())
    if key not in _DQN_AGENTS:
        agent = DQNAgent()
        agent.device = device
        agent.policy_net = agent.policy_net.to(device)
        agent.target_net = agent.target_net.to(device)
        agent.load(path)
        agent.policy_net.eval()
        agent.target_net.eval()
        _DQN_AGENTS[key] = agent
    return _DQN_AGENTS[key]


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

    mask = legal_mask_numpy(game)
    if not mask.any():
        return 0

    agent = load_dqn_agent(checkpoint, dev)

    action = int(
    agent.select_action(
        game.board,
        training=False,
        legal_actions=[int(a) for a in game.legal_actions()],
    )
    )

    if not mask[action]:
        legal = np.flatnonzero(mask)
        return int(legal[0])

    return action


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
    if name == "corner":
        return CornerHeuristicAgent().act(obs, info)
    if name == "dqn":
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("DQN checkpoint missing or invalid")
        return pick_dqn_action(game, checkpoint, device=device)

    raise ValueError(f"unknown bot policy {policy!r}")