import pytest

pytest.importorskip("httpx")

from starlette.testclient import TestClient

from web.server import REPO_ROOT, app


def test_list_checkpoints():
    c = TestClient(app)
    r = c.get("/api/checkpoints")
    assert r.status_code == 200
    assert "checkpoints" in r.json()


def test_versus_random_match():
    c = TestClient(app)
    r = c.post("/api/matches", json={"mode": "versus", "opponent": "random", "seed": 42})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "versus"
    assert d["left_kind"] == "human"
    s = c.post(f"/api/matches/{d['match_id']}/opponent-turn")
    assert s.status_code == 200


def test_versus_dual_step_advances_both_boards():
    c = TestClient(app)
    r = c.post("/api/matches", json={"mode": "versus", "opponent": "greedy", "seed": 7})
    assert r.status_code == 200
    mid = r.json()["match_id"]
    u = c.post(f"/api/matches/{mid}/versus-step", json={"action": "up"})
    assert u.status_code == 200
    pack = u.json()
    assert pack["player"]["valid"] is True
    assert pack["opponent"] is not None
    assert pack["opponent"]["valid"] is True


def test_arena_tick():
    c = TestClient(app)
    r = c.post(
        "/api/matches",
        json={"mode": "arena", "left": "greedy", "right": "ordered", "seed": 1},
    )
    assert r.status_code == 200
    mid = r.json()["match_id"]
    t = c.post(f"/api/matches/{mid}/tick", json={"lanes": ["left", "right"]})
    assert t.status_code == 200
    assert "left" in t.json()["steps"]


def test_dqn_match_requires_checkpoint():
    c = TestClient(app)
    r = c.post("/api/matches", json={"mode": "versus", "opponent": "dqn"})
    ck_dir = REPO_ROOT / "checkpoints"
    has_pt = any(ck_dir.glob("*.pt")) if ck_dir.is_dir() else False
    if has_pt:
        assert r.status_code == 200
    else:
        assert r.status_code == 400
