import json
import os
import tempfile
import pytest
import torch

from visualizer import VisualizerSession
from elo_system import EloDatabase, DEFAULT_INITIAL_RATING


def test_visualizer_session_elo_integration(tmp_path):
    db_path = str(tmp_path / "test_elo.json")
    db = EloDatabase(db_path=db_path)
    session = VisualizerSession(elo_db=db)
    session.set_human_player_name("Guillaume")

    # Check state serialization
    state = session._serialize_current_state()
    assert "elo" in state
    assert state["elo"]["human_player_name"] == "Guillaume"
    assert state["elo"]["p1_rating"] == DEFAULT_INITIAL_RATING
    assert state["elo"]["p2_rating"] == DEFAULT_INITIAL_RATING

    # Check agent listing has Elo ratings
    agents = session.list_available_agents()
    human_agent = next(a for a in agents if a["id"] == "human")
    assert "Guillaume" in human_agent["label"]
    assert "1500 Elo" in human_agent["label"]
    assert human_agent["elo_rating"] == 1500

    # Simulate game finish and match recording
    game = session.history[0][0].clone()
    game.done = True
    game.winner = 0
    game.termination_reason = "sentinel_instant_win"

    assert not session.match_recorded
    session._check_and_record_match(game)
    assert session.match_recorded
    assert session.last_elo_update is not None
    assert session.last_elo_update["winner"] == 0
    assert session.last_elo_update["p1_delta"] > 0
    assert session.last_elo_update["p2_delta"] < 0

    # Verify leaderboard updated
    lb = session.elo_db.get_leaderboard()
    assert lb[0]["id"] == "Guillaume"
    assert lb[0]["rating"] > DEFAULT_INITIAL_RATING

    # New game resets match recording flags
    new_state = session.new_game()
    assert not session.match_recorded
    assert session.last_elo_update is None


def test_visualizer_vanguard_bonus_metadata(tmp_path):
    db = EloDatabase(db_path=str(tmp_path / "test_elo.json"))
    session = VisualizerSession(elo_db=db)
    session.new_game(seed=42)
    state = session._serialize_current_state()

    # Pieces 2 and 3 are Player 1 Vanguards
    p1_vanguard = state["inspector"]["current_player_pieces"][2]
    assert "Vanguard" in p1_vanguard["name"]
    v_bonus_moves = [m for m in p1_vanguard["legal_moves"] if m["is_vanguard_bonus"]]
    assert len(v_bonus_moves) > 0
    for m in v_bonus_moves:
        assert m["d_min"] == 4
        assert m["is_vanguard_bonus"] is True

    # Explorer should have no vanguard bonus
    p1_explorer = state["inspector"]["current_player_pieces"][0]
    assert all(m["is_vanguard_bonus"] is False for m in p1_explorer["legal_moves"])


def test_visualizer_multithreaded_locking_and_concurrency(tmp_path):
    """
    Verifies that VisualizerSession methods are thread-safe and can handle
    concurrent requests from ThreadingHTTPServer threads without deadlocks or race conditions (review #5.6).
    """
    import concurrent.futures

    db = EloDatabase(db_path=str(tmp_path / "test_elo.json"))
    session = VisualizerSession(elo_db=db)
    errors = []

    def worker_state():
        for _ in range(20):
            try:
                st = session._serialize_current_state()
                assert "pieces" in st
                assert "inspector" in st
            except Exception as e:
                errors.append(e)

    def worker_agents():
        for _ in range(15):
            try:
                agents = session.list_available_agents()
                assert len(agents) > 0
            except Exception as e:
                errors.append(e)

    def worker_moves():
        for _ in range(10):
            try:
                st = session._serialize_current_state()
                p_pieces = st["inspector"]["current_player_pieces"]
                moved = False
                for p in p_pieces:
                    if p["legal_moves"]:
                        act = p["legal_moves"][0]["action"]
                        session.apply_human_move(act)
                        moved = True
                        break
                if not moved:
                    session.new_game()
            except ValueError as e:
                # Stale move from concurrent turn transition or already finished game
                if "Illegal action" in str(e) or "already finished" in str(e):
                    pass
                else:
                    errors.append(e)
            except Exception as e:
                errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(worker_state),
            executor.submit(worker_state),
            executor.submit(worker_agents),
            executor.submit(worker_agents),
            executor.submit(worker_moves),
            executor.submit(worker_moves),
        ]
        done, not_done = concurrent.futures.wait(futures, timeout=10.0)
        assert len(not_done) == 0, "Deadlock detected: some thread workers did not finish within timeout"

    assert len(errors) == 0, f"Encountered thread concurrency errors: {errors}"


def test_checkpoint_discovery_filters_replay_buffers_and_optimizers(tmp_path):
    train_dir = tmp_path / "train_mock"
    train_dir.mkdir()
    torch.save({"model_state_dict": {}}, train_dir / "best.pt")
    torch.save({"model_state_dict": {}}, train_dir / "latest.pt")
    torch.save({"model_state_dict": {}}, train_dir / "iter_0100.pt")
    torch.save({"capacity": 1000}, train_dir / "replay_buffer.pt")
    torch.save({"capacity": 1000}, train_dir / "replay_buffer_iter_0100.pt")
    torch.save({"state": {}}, train_dir / "optimizer.pt")

    from elo_arena import discover_available_checkpoints
    found_arena = discover_available_checkpoints(str(tmp_path))
    assert "train_mock/best.pt" in found_arena
    assert "train_mock/latest.pt" in found_arena
    assert "train_mock/iter_0100.pt" in found_arena
    assert not any("replay_buffer" in p for p in found_arena)
    assert not any("optimizer" in p for p in found_arena)

    db = EloDatabase(db_path=str(tmp_path / "test_elo.json"))
    session = VisualizerSession(checkpoint_dir=str(tmp_path), elo_db=db)
    found_viz = session._discover_checkpoint_relpaths()
    assert "train_mock/best.pt" in found_viz
    assert "train_mock/latest.pt" in found_viz
    assert "train_mock/iter_0100.pt" in found_viz
    assert not any("replay_buffer" in p for p in found_viz)
    assert not any("optimizer" in p for p in found_viz)


def test_list_available_agents_includes_games_played(tmp_path):
    db_path = str(tmp_path / "test_elo.json")
    db = EloDatabase(db_path=db_path)
    # Record a match so Guillaume has 1 game
    db.record_match(p1_id="Guillaume", p2_id="heuristic:standard", winner=0)
    session = VisualizerSession(elo_db=db)
    session.set_human_player_name("Guillaume")

    agents = session.list_available_agents()
    human = next(a for a in agents if a["id"] == "human")
    assert "games_played" in human
    assert human["games_played"] == 1

    h_std = next(a for a in agents if a["id"] == "heuristic:standard")
    assert "games_played" in h_std
    assert h_std["games_played"] == 1


def test_targeted_gauntlet_tournament(tmp_path):
    from elo_arena import run_tournament
    db = EloDatabase(db_path=str(tmp_path / "test_elo_tourn.json"))

    # Target heuristic:blitz against two opponents with distinct IDs
    target = "heuristic:blitz"
    opponents = ["heuristic:standard", "heuristic:fast"]
    records = run_tournament(
        agent_ids=[target] + opponents,
        db=db,
        pairs_per_matchup=1,  # 1 pair = 2 games per opponent
        target_agent_id=target,
    )

    # 2 opponents * 1 pair * 2 games = 4 games total
    assert len(records) == 4

    # All matches MUST involve target
    for rec in records:
        assert target in (rec.player1_id, rec.player2_id)
        other = rec.player2_id if rec.player1_id == target else rec.player1_id
        assert other in opponents

    # Verify no games between opponents
    for rec in records:
        assert not (rec.player1_id in opponents and rec.player2_id in opponents)

    # Verify ratings updated in DB
    assert target in db.players
    assert db.players[target].games_played == 4


def test_visualizer_start_tournament_target(tmp_path):
    db = EloDatabase(db_path=str(tmp_path / "test_elo_session.json"))
    session = VisualizerSession(elo_db=db)

    # Start a targeted tournament via session
    target = "heuristic:blitz"
    opponents = ["heuristic:standard", "heuristic:master"]
    res = session.start_tournament(
        agent_ids=opponents,
        pairs=3,
        target_agent=target,
    )

    assert res["status"] == "started"
    # 2 opponents * 3 pairs * 2 games = 12 total games
    assert res["total"] == 12
    assert res["target_agent"] == target
    assert session.tournament_status["target_agent"] == target
    assert session.tournament_status["total"] == 12
    assert session.tournament_status["mode"] == "gauntlet"




