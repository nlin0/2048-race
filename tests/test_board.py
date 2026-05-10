import numpy as np
import pytest

from race2048.board import Action, Game2048, _merge_line_left, has_legal_move, slide_board


def test_merge_line_left_triple_twos():
    out = _merge_line_left(np.array([2, 2, 2, 0], dtype=np.int32))
    assert np.array_equal(out, np.array([4, 2, 0, 0], dtype=np.int32))


def test_merge_line_left_four_twos():
    out = _merge_line_left(np.array([2, 2, 2, 2], dtype=np.int32))
    assert np.array_equal(out, np.array([4, 4, 0, 0], dtype=np.int32))


def test_invalid_move_unchanged_state():
    g = Game2048(seed=0)
    g._board[:] = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )
    snap = g.board.copy()
    res = g.step(Action.LEFT)
    assert not res.valid
    assert np.array_equal(g.board, snap)


def test_invalid_move_no_spawn_on_full_gradient():
    g = Game2048(seed=0)
    g._board[:] = np.array(
        [
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
        ],
        dtype=np.int32,
    )
    snap = g.board.copy()
    res = g.step(Action.LEFT)
    assert not res.valid
    assert np.array_equal(g.board, snap)
    assert np.max(g.board) == 128


def test_valid_move_spawns_tile():
    g = Game2048(seed=42)
    g.reset()
    assert np.count_nonzero(g._board) == 2
    for a in Action:
        g2 = Game2048(seed=42)
        g2.reset()
        res = g2.step(a)
        if res.valid:
            assert np.count_nonzero(g2._board) in (2, 3)
            break
    else:
        pytest.fail("expected a valid move from reset with seed 42")


def test_win_flag_when_2048_present():
    g = Game2048(seed=0)
    g._board[:] = 0
    g._board[0, 0] = 2048
    res = g.step(Action.UP)
    assert res.won
    assert not res.valid


def test_game_over_full_stuck():
    g = Game2048(seed=0)
    g._board[:] = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )
    assert not has_legal_move(g._board)
    res = g.step(Action.LEFT)
    assert not res.valid
    assert res.game_over


def test_down_merge_column_no_spawn(monkeypatch):
    def no_spawn(self) -> None:
        return None

    monkeypatch.setattr(Game2048, "_spawn_tile", no_spawn)

    g = Game2048(seed=1)
    g._board[:] = 0
    g._board[0, 0] = 2
    g._board[1, 0] = 2

    res = g.step(Action.DOWN)
    assert res.valid
    assert g._board[3, 0] == 4
    assert g._board[0, 0] == 0
    assert g._board[1, 0] == 0


def test_slide_board_no_spawn():
    b = np.zeros((4, 4), dtype=np.int32)
    b[0, 0] = 2
    b[0, 1] = 2
    s = slide_board(b, Action.LEFT)
    assert s[0, 0] == 4 and s[0, 1] == 0


def test_spawn_uses_ninety_ten_split_approximately():
    rng = np.random.default_rng(7)
    tiles = [2 if rng.random() < 0.9 else 4 for _ in range(20_000)]
    fours = sum(1 for t in tiles if t == 4)
    assert 1500 < fours < 2500
