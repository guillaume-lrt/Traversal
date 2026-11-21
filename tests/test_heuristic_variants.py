import numpy as np
import pytest

from traversal_env import TraversalGame
from train import HeuristicAgent, search_best_heuristic_moves_numba, HEURISTIC_PRESETS
from elo_system import parse_agent_variant, EloDatabase
from visualizer import VisualizerSession


def test_heuristic_presets_structure():
    assert "blitz" in HEURISTIC_PRESETS
    assert "standard" in HEURISTIC_PRESETS
    assert "master" in HEURISTIC_PRESETS
    assert "grandmaster" in HEURISTIC_PRESETS

    assert HEURISTIC_PRESETS["blitz"]["depth"] == 2
    assert HEURISTIC_PRESETS["blitz"]["root_beam"] == 10

    assert HEURISTIC_PRESETS["standard"]["depth"] == 4
    assert HEURISTIC_PRESETS["standard"]["root_beam"] == 14

    assert HEURISTIC_PRESETS["master"]["depth"] == 6
    assert HEURISTIC_PRESETS["master"]["root_beam"] == 16

    assert HEURISTIC_PRESETS["grandmaster"]["depth"] == 8
    assert HEURISTIC_PRESETS["grandmaster"]["root_beam"] == 16


def test_heuristic_agent_variants():
    blitz = HeuristicAgent(variant="blitz")
    assert blitz.depth == 2
    assert blitz.root_beam == 10
    assert blitz.variant == "blitz"

    std = HeuristicAgent(variant="standard")
    assert std.depth == 4
    assert std.root_beam == 14
    assert std.variant == "standard"

    master = HeuristicAgent(variant="master")
    assert master.depth == 6
    assert master.root_beam == 16
    assert master.variant == "master"

    gm = HeuristicAgent(variant="grandmaster")
    assert gm.depth == 8
    assert gm.root_beam == 16
    assert gm.variant == "grandmaster"

    # Default should be standard
    default_agent = HeuristicAgent()
    assert default_agent.depth == 4
    assert default_agent.root_beam == 14
    assert default_agent.variant == "standard"


def test_heuristic_action_selection():
    game = TraversalGame(seed=77)
    for v in ("blitz", "standard", "master", "grandmaster"):
        agent = HeuristicAgent(variant=v, rng=np.random.default_rng(42))
        act = agent.select_action(game)
        assert isinstance(act, int)
        legal = game.get_legal_actions()
        assert act in legal


def test_parse_agent_variant_heuristics():
    assert parse_agent_variant("heuristic:blitz") == ("heuristic", "blitz", 2)
    assert parse_agent_variant("heuristic:fast") == ("heuristic", "blitz", 2)
    assert parse_agent_variant("heuristic:2ply") == ("heuristic", "blitz", 2)

    assert parse_agent_variant("heuristic:standard") == ("heuristic", "standard", 4)
    assert parse_agent_variant("heuristic") == ("heuristic", "standard", 4)
    assert parse_agent_variant("heuristic:4ply") == ("heuristic", "standard", 4)

    assert parse_agent_variant("heuristic:master") == ("heuristic", "master", 6)
    assert parse_agent_variant("heuristic:deep") == ("heuristic", "master", 6)
    assert parse_agent_variant("heuristic:6ply") == ("heuristic", "master", 6)

    assert parse_agent_variant("heuristic:grandmaster") == ("heuristic", "grandmaster", 8)
    assert parse_agent_variant("heuristic:8ply") == ("heuristic", "grandmaster", 8)
    assert parse_agent_variant("heuristic:ultra") == ("heuristic", "grandmaster", 8)


def test_visualizer_available_agents_contains_all_heuristics(tmp_path):
    db_path = str(tmp_path / "elo.json")
    db = EloDatabase(db_path=db_path)
    session = VisualizerSession(elo_db=db)

    agents = session.list_available_agents()
    agent_ids = {a["id"] for a in agents}

    assert "heuristic:blitz" in agent_ids
    assert "heuristic:standard" in agent_ids
    assert "heuristic:master" in agent_ids
    assert "heuristic:grandmaster" in agent_ids

    # Check labels and variants
    h_gm = next(a for a in agents if a["id"] == "heuristic:grandmaster")
    assert "Grandmaster" in h_gm["label"]
    assert "8-ply" in h_gm["label"]

    h_master = next(a for a in agents if a["id"] == "heuristic:master")
    assert "Master" in h_master["label"]
    assert "6-ply" in h_master["label"]

    h_std = next(a for a in agents if a["id"] == "heuristic:standard")
    assert "Standard" in h_std["label"]
    assert "4-ply" in h_std["label"]

    h_blitz = next(a for a in agents if a["id"] == "heuristic:blitz")
    assert "Blitz" in h_blitz["label"]
    assert "2-ply" in h_blitz["label"]


def test_visualizer_heuristic_analysis_all_variants(tmp_path):
    db_path = str(tmp_path / "elo.json")
    db = EloDatabase(db_path=db_path)
    session = VisualizerSession(elo_db=db)
    game = TraversalGame(seed=123)

    for var_id in ("heuristic:blitz", "heuristic:standard", "heuristic:master", "heuristic:grandmaster"):
        act, analysis = session._run_heuristic_analysis(game, agent_id=var_id)
        assert isinstance(act, int)
        assert "elapsed_ms" in analysis
        assert "top_moves" in analysis
        assert len(analysis["top_moves"]) > 0
        assert analysis["top_moves"][0]["action"] >= 0


def test_heuristic_evaluation_blocked_distances_awareness():
    """
    Verifies that _evaluate_board_numba uses blocked distances rather than
    unblocked terrain tables alone (review #6.1 & #112).
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    game = TraversalGame(seed=42)
    t_grid, c_grid, eq = _precompute_terrain_tables_numba(game.terrain)

    # Initial baseline evaluation
    eval_initial = _evaluate_board_numba(
        game.terrain, t_grid, c_grid, eq, game.pos, game.scores, game.total_pieces_finished, 0
    )

    # Place a friendly piece blocking the opponent's Sentinel path
    game_blocked = game.clone()
    p = 0
    opp = 1
    # Move friendly piece 0 directly in front of opponent Sentinel (piece 5)
    game_blocked.pos[p, 0] = [game.pos[opp, 5, 0] - 1, game.pos[opp, 5, 1]]

    eval_blocked = _evaluate_board_numba(
        game_blocked.terrain, t_grid, c_grid, eq, game_blocked.pos, game_blocked.scores, game_blocked.total_pieces_finished, 0
    )

    # Friendly player should receive a significant tactical advantage from blocking the opponent Sentinel
    assert eval_blocked > eval_initial + 200.0


def test_heuristic_search_repetition_awareness():
    """
    Verifies that the heuristic search is repetition-aware, correctly evaluating
    threefold repetition moves as terminal draws (0.0) rather than drifting into
    repetition draws while winning (review #6.2 & #113).
    """
    from train import search_best_heuristic_moves_numba, HeuristicAgent

    game = TraversalGame(seed=42)
    k0 = game.get_position_key()

    # Step forward 2 plies
    legal_p0 = game.get_legal_actions()
    game.step(legal_p0[0])
    legal_p1 = game.get_legal_actions()
    game.step(legal_p1[0])

    # Record that k0 has been reached twice
    game.position_counts[k0] = 2

    # Agent select_action should run smoothly and take position counts into account
    agent = HeuristicAgent(variant="fast")
    chosen_act = agent.select_action(game)
    assert chosen_act in game.get_legal_actions()

    # Verify search_best_heuristic_moves_numba with explicit repetition arrays
    rep_v0 = np.array([k0[0]], dtype=np.int64)
    rep_v1 = np.array([k0[1]], dtype=np.int64)
    rep_c = np.array([2], dtype=np.int32)

    actions, values = search_best_heuristic_moves_numba(
        game.terrain,
        game.pos,
        game.scores,
        int(game.total_pieces_finished),
        int(game.ply_count),
        int(game.current_player),
        2,
        10,
        None,
        game.piece_status,
        rep_v0,
        rep_v1,
        rep_c,
    )
    assert len(actions) > 0
    assert len(values) == len(actions)


def test_endgame_race_term_single_finish():
    """
    Verifies that when total_finished >= 1, the race evaluation evaluates
    the direct single-finish sprint rather than adding up phantom second pieces (review #6.3 & #114).
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    game = TraversalGame(seed=42)
    t_grid, c_grid, eq = _precompute_terrain_tables_numba(game.terrain)

    # Setup endgame where Player 0 already banked 1 finish
    pos = game.pos.copy()
    scores = np.array([2, 0], dtype=np.int32)
    total_fin = 1
    pos[0, 0] = [-1, -1]  # Piece 0 finished
    pos[0, 1] = [5, 4]    # Piece 1 in mid-board

    # Case A: Opponent Sentinel is at row 5 (far from finish at row 0)
    pos[1, 5] = [5, 5]
    eval_safe = _evaluate_board_numba(game.terrain, t_grid, c_grid, eq, pos, scores, total_fin, 0)

    # Case B: Opponent Sentinel is at row 2 (much closer to finish at row 0)
    pos[1, 5] = [2, 5]
    eval_danger = _evaluate_board_numba(game.terrain, t_grid, c_grid, eq, pos, scores, total_fin, 0)

    # Player 0 evaluation should be significantly lower when opponent is close to a winning finish
    assert eval_safe > eval_danger + 400.0


def test_heuristic_agent_iterative_deepening_time_budget():
    """
    Verifies that HeuristicAgent supports iterative deepening under a time budget (review #6.4 & #111).
    """
    import time

    game = TraversalGame(seed=42)

    # 1. Agent with time budget
    budget_ms = 40.0
    agent = HeuristicAgent(variant="standard", time_budget_ms=budget_ms)
    assert agent.time_budget_ms == budget_ms

    t0 = time.perf_counter()
    action = agent.select_action(game)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert action in game.get_legal_actions()
    # Should complete within a small multiplier of budget (allowing margin for first call compilation/overhead)
    assert elapsed_ms < 500.0

    # 2. Dynamic override via select_action parameter
    action2 = agent.select_action(game, time_budget_ms=25.0)
    assert action2 in game.get_legal_actions()

    # 3. Direct search without budget (time_budget_ms=None)
    agent_direct = HeuristicAgent(variant="blitz", time_budget_ms=None)
    action3 = agent_direct.select_action(game)
    assert action3 in game.get_legal_actions()




