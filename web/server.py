"""Browser UI server: wraps `Game2048` with a small HTTP API + bot matches."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from race2048.board import Action, Game2048, has_legal_move, has_won
from race2048.demo_bots import pick_bot_action, resolve_checkpoint_under

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default DQN weights under ``checkpoints/`` (basename only in API / UI).
DEFAULT_DQN_VERSUS = "dqn_cnn_n3_1.pt"
DEFAULT_DQN_ARENA_LEFT = "dqn_cnn_n3_1.pt"
DEFAULT_DQN_ARENA_RIGHT = "dqn_2_1.pt"

_ACTION_MAP: dict[str, Action] = {
    "up": Action.UP,
    "down": Action.DOWN,
    "left": Action.LEFT,
    "right": Action.RIGHT,
}

app = FastAPI(title="2048 Race UI")
_GAMES: dict[str, Game2048] = {}


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Avoid stale CSS/JS during local dev (browsers cache /static aggressively)."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def _snapshot(g: Game2048) -> dict:
    b = g.board
    return {
        "board": b.tolist(),
        "legal_actions": [a.name.lower() for a in g.legal_actions()],
        "won": has_won(b, Game2048.WIN_TILE),
        "game_over": not has_legal_move(b),
    }


@dataclass
class DuelMatch:
    id_str: str
    mode: str  # "versus" | "arena"
    left_kind: str
    right_kind: str
    left_id: str
    right_id: str
    left_ckpt: Path | None
    right_ckpt: Path | None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


_MATCHES: dict[str, DuelMatch] = {}


class NewGameResponse(BaseModel):
    id: str
    board: list[list[int]]
    legal_actions: list[str]
    game_over: bool
    won: bool


class StepRequest(BaseModel):
    action: str = Field(..., description="up | down | left | right")


class StepResponse(BaseModel):
    board: list[list[int]]
    valid: bool
    game_over: bool
    won: bool
    legal_actions: list[str]


class VersusDualStepResponse(BaseModel):
    """Human (left) step and bot reply when the human move was valid."""

    player: StepResponse
    opponent: StepResponse | None = None


class MatchCreate(BaseModel):
    """Create a duel (human vs bot) or arena (bot vs bot) match."""

    mode: str = Field(..., description='"versus" or "arena"')
    opponent: str | None = Field(
        default=None,
        description="versus: random | ordered | greedy | dqn",
    )
    checkpoint: str | None = Field(default=None, description="versus opponent dqn basename")
    left: str | None = Field(default=None, description="arena left policy")
    right: str | None = Field(default=None, description="arena right policy")
    left_checkpoint: str | None = None
    right_checkpoint: str | None = None
    seed: int | None = None


class MatchCreated(BaseModel):
    match_id: str
    mode: str
    left_kind: str
    right_kind: str
    left_game_id: str
    right_game_id: str


class TickBody(BaseModel):
    lanes: list[str] = Field(
        ...,
        description='Any of "left", "right" — order determines move sequence',
    )


def _maybe_ckpt(kind: str, rel: str | None, *, default_name: str) -> Path | None:
    if kind.lower() != "dqn":
        return None
    return resolve_checkpoint_under(REPO_ROOT, rel, default_name)


def _require_ckpt(kind: str, rel: str | None, *, default_name: str) -> Path:
    rel_eff = (rel or default_name).strip() or default_name
    basename = Path(rel_eff).name
    path = _maybe_ckpt(kind, rel, default_name=default_name)
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                f'Missing weight file "{basename}". Add it under checkpoints/ '
                f"(or CNN/checkpoints/, MLP/checkpoints/) at the repo root."
            ),
        )
    return path


@app.post("/api/games", response_model=NewGameResponse)
def new_game(seed: int | None = Query(default=None)) -> NewGameResponse:
    gid = str(uuid.uuid4())
    game = Game2048(seed=seed)
    game.reset()
    _GAMES[gid] = game
    st = _snapshot(game)
    return NewGameResponse(id=gid, **st)


@app.post("/api/games/{game_id}/step", response_model=StepResponse)
def step(game_id: str, body: StepRequest) -> StepResponse:
    game = _GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Unknown game id")

    key = body.action.strip().lower()
    if key not in _ACTION_MAP:
        raise HTTPException(status_code=400, detail="action must be up, down, left, or right")

    res = game.step(_ACTION_MAP[key])
    return StepResponse(
        board=res.board.tolist(),
        valid=res.valid,
        game_over=res.game_over,
        won=res.won,
        legal_actions=[a.name.lower() for a in game.legal_actions()],
    )


@app.get("/api/games/{game_id}")
def get_state(game_id: str) -> dict:
    game = _GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Unknown game id")
    return _snapshot(game)


@app.get("/api/checkpoints")
def list_checkpoints() -> dict:
    names: set[str] = set()
    root_r = REPO_ROOT.resolve()
    for rel in ("checkpoints", "CNN/checkpoints", "MLP/checkpoints"):
        ck = (root_r / rel).resolve()
        if not ck.is_dir():
            continue
        for p in ck.iterdir():
            if p.suffix.lower() == ".pt":
                names.add(p.name)
    (root_r / "checkpoints").mkdir(parents=True, exist_ok=True)
    return {"checkpoints": sorted(names)}


@app.post("/api/matches", response_model=MatchCreated)
def create_match(body: MatchCreate) -> MatchCreated:
    mode = body.mode.strip().lower()
    if mode not in ("versus", "arena"):
        raise HTTPException(status_code=400, detail='mode must be "versus" or "arena"')

    seed = body.seed if body.seed is not None else int(uuid.uuid4().int % (2**31))

    if mode == "versus":
        if not body.opponent:
            raise HTTPException(status_code=400, detail="versus requires opponent")
        opp = body.opponent.strip().lower()
        allowed = {"random", "ordered", "greedy", "dqn"}
        if opp not in allowed:
            raise HTTPException(status_code=400, detail=f"opponent must be one of {sorted(allowed)}")
        left_kind, right_kind = "human", opp
        left_ckpt, right_ckpt = None, None
        if opp == "dqn":
            right_ckpt = _require_ckpt("dqn", body.checkpoint, default_name=DEFAULT_DQN_VERSUS)
    else:
        if not body.left or not body.right:
            raise HTTPException(status_code=400, detail="arena requires left and right")
        left_kind = body.left.strip().lower()
        right_kind = body.right.strip().lower()
        allowed = {"random", "ordered", "greedy", "dqn"}
        if left_kind not in allowed or right_kind not in allowed:
            raise HTTPException(status_code=400, detail=f"policies must be in {sorted(allowed)}")
        left_ckpt = (
            _require_ckpt(left_kind, body.left_checkpoint, default_name=DEFAULT_DQN_ARENA_LEFT)
            if left_kind == "dqn"
            else None
        )
        right_ckpt = (
            _require_ckpt(right_kind, body.right_checkpoint, default_name=DEFAULT_DQN_ARENA_RIGHT)
            if right_kind == "dqn"
            else None
        )

    mid = str(uuid.uuid4())
    lid = str(uuid.uuid4())
    rid = str(uuid.uuid4())
    left_game = Game2048(seed=seed)
    right_game = Game2048(seed=seed + 1)
    left_game.reset()
    right_game.reset()
    _GAMES[lid] = left_game
    _GAMES[rid] = right_game

    _MATCHES[mid] = DuelMatch(
        id_str=mid,
        mode=mode,
        left_kind=left_kind,
        right_kind=right_kind,
        left_id=lid,
        right_id=rid,
        left_ckpt=left_ckpt,
        right_ckpt=right_ckpt,
        rng=np.random.default_rng(seed),
    )
    return MatchCreated(
        match_id=mid,
        mode=mode,
        left_kind=left_kind,
        right_kind=right_kind,
        left_game_id=lid,
        right_game_id=rid,
    )


def _run_bot_turn(game_id: str, m: DuelMatch, side: str) -> StepResponse:
    game = _GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Unknown game id")

    snap0 = _snapshot(game)
    if snap0["game_over"]:
        return StepResponse(
            board=snap0["board"],
            valid=False,
            game_over=True,
            won=snap0["won"],
            legal_actions=snap0["legal_actions"],
        )

    if side == "left":
        kind, ck = m.left_kind, m.left_ckpt
    else:
        kind, ck = m.right_kind, m.right_ckpt

    if kind == "human":
        raise HTTPException(status_code=400, detail="cannot auto-step human lane")

    try:
        act_idx = pick_bot_action(
            kind,
            game,
            rng=m.rng,
            checkpoint=ck,
            device=DEVICE,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    res = game.step(act_idx)
    return StepResponse(
        board=res.board.tolist(),
        valid=res.valid,
        game_over=res.game_over,
        won=res.won,
        legal_actions=[a.name.lower() for a in game.legal_actions()],
    )


@app.post("/api/matches/{match_id}/versus-step", response_model=VersusDualStepResponse)
def versus_dual_step(match_id: str, body: StepRequest) -> VersusDualStepResponse:
    """Apply the human move, then the bot reply, in one request (parallel boards)."""
    m = _MATCHES.get(match_id)
    if m is None:
        raise HTTPException(status_code=404, detail="unknown match")
    if m.mode != "versus":
        raise HTTPException(status_code=400, detail="only versus matches support versus-step")
    game = _GAMES.get(m.left_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Unknown game id")

    key = body.action.strip().lower()
    if key not in _ACTION_MAP:
        raise HTTPException(status_code=400, detail="action must be up, down, left, or right")

    res = game.step(_ACTION_MAP[key])
    player = StepResponse(
        board=res.board.tolist(),
        valid=res.valid,
        game_over=res.game_over,
        won=res.won,
        legal_actions=[a.name.lower() for a in game.legal_actions()],
    )
    if not res.valid:
        return VersusDualStepResponse(player=player, opponent=None)
    opponent = _run_bot_turn(m.right_id, m, "right")
    return VersusDualStepResponse(player=player, opponent=opponent)


@app.post("/api/matches/{match_id}/opponent-turn", response_model=StepResponse)
def opponent_turn(match_id: str) -> StepResponse:
    m = _MATCHES.get(match_id)
    if m is None:
        raise HTTPException(status_code=404, detail="unknown match")
    if m.mode != "versus":
        raise HTTPException(status_code=400, detail="only versus matches support opponent-turn")
    return _run_bot_turn(m.right_id, m, "right")


@app.post("/api/matches/{match_id}/tick")
def tick_match(match_id: str, body: TickBody) -> dict:
    """Step one or two bot-controlled lanes."""
    m = _MATCHES.get(match_id)
    if m is None:
        raise HTTPException(status_code=404, detail="unknown match")
    lanes = []
    for x in body.lanes:
        s = x.strip().lower()
        if s not in ("left", "right"):
            raise HTTPException(status_code=400, detail='lanes entries must be "left" or "right"')
        lanes.append(s)
    if len(lanes) == 0:
        raise HTTPException(status_code=400, detail="lanes must not be empty")

    out: dict[str, StepResponse | None] = {}
    for ln in lanes:
        gid = m.left_id if ln == "left" else m.right_id
        out[ln] = _run_bot_turn(gid, m, ln)
    return {"steps": {k: v.model_dump() for k, v in out.items()}}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
