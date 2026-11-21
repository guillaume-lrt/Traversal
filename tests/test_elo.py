import os
import tempfile
import pytest

from elo_system import (
    EloDatabase,
    expected_score,
    compute_k_factor,
    calculate_elo_update,
    DEFAULT_INITIAL_RATING,
)


def test_expected_score():
    assert pytest.approx(expected_score(1500, 1500), 1e-4) == 0.5
    e_a = expected_score(1600, 1400)
    e_b = expected_score(1400, 1600)
    assert pytest.approx(e_a + e_b, 1e-4) == 1.0
    assert e_a > 0.75  # ~0.7597


def test_compute_k_factor():
    assert compute_k_factor(0) == 40.0
    assert compute_k_factor(9) == 40.0
    assert compute_k_factor(10) == 28.0
    assert compute_k_factor(29) == 28.0
    assert compute_k_factor(30) == 16.0
    assert compute_k_factor(100) == 16.0
    assert compute_k_factor(100, base_k=20.0) == 20.0


def test_calculate_elo_update():
    # Equal players, player 1 wins
    r1, r2, d1, d2 = calculate_elo_update(1500, 1500, score_a=1.0, games_a=0, games_b=0)
    assert d1 == 20.0
    assert d2 == -20.0
    assert r1 == 1520.0
    assert r2 == 1480.0

    # Draw between equal players
    r1, r2, d1, d2 = calculate_elo_update(1500, 1500, score_a=0.5, games_a=0, games_b=0)
    assert d1 == 0.0
    assert d2 == 0.0
    assert r1 == 1500.0
    assert r2 == 1500.0


def test_elo_database_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_elo.json")
        db = EloDatabase(db_path=db_path)

        # Seeds check
        assert "Guillaume" in db.players
        assert "heuristic" in db.players
        assert db.players["Guillaume"].rating == DEFAULT_INITIAL_RATING

        # Record a match: Guillaume beats Heuristic
        rec = db.record_match(
            p1_id="Guillaume",
            p2_id="heuristic",
            winner=0,
            ply_count=35,
            scores=[3, 1],
            termination_reason="sentinel_instant_win",
            source="visualizer",
        )

        assert rec.p1_rating_after > DEFAULT_INITIAL_RATING
        assert rec.p2_rating_after < DEFAULT_INITIAL_RATING
        assert rec.winner_name == "Guillaume"

        # Check leaderboard
        lb = db.get_leaderboard()
        assert len(lb) >= 2
        assert lb[0]["id"] == "Guillaume"
        assert lb[0]["rank"] == 1
        assert lb[0]["wins"] == 1
        assert lb[0]["games_played"] == 1
        assert lb[0]["win_rate"] == 1.0

        # Reload from disk
        db2 = EloDatabase(db_path=db_path)
        assert len(db2.matches) == 1
        assert db2.players["Guillaume"].rating == rec.p1_rating_after
        assert db2.players["Guillaume"].peak_rating == rec.p1_rating_after
        assert db2.players["heuristic"].rating == rec.p2_rating_after


def test_multiple_models_leaderboard():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_elo2.json")
        db = EloDatabase(db_path=db_path)

        # Record matches between multiple models
        db.record_match(p1_id="train_v4/best.pt", p2_id="train_v3/best.pt", winner=0)
        db.record_match(p1_id="train_v4/best.pt", p2_id="train_v2/best.pt", winner=0)
        db.record_match(p1_id="train_v3/best.pt", p2_id="train_v2/best.pt", winner=0)
        db.record_match(p1_id="Guillaume", p2_id="train_v4/best.pt", winner=0)

        lb = db.get_leaderboard()
        # Guillaume beat train_v4 who beat v3 and v2
        assert lb[0]["id"] == "Guillaume"
        assert lb[1]["id"] == "train_v4/best.pt"
        assert lb[2]["id"] == "train_v3/best.pt"
