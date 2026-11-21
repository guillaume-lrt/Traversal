import pytest
import numpy as np

from traversal_env import (
    TraversalGame,
    PIECE_NAMES,
    PIECE_MOVEMENT,
    PIECE_TYPES,
    PIECE_EXPLORER,
    PIECE_VANGUARD,
    PIECE_SENTINEL,
)
import visualizer
from visualizer import (
    VisualizerSession,
    compute_piece_finish_path,
    compute_detailed_ui_inspector,
    _compute_single_player_inspector,
)


class TestForbiddenMoves:
    def test_forbidden_moves_generated_on_initial_board(self):
        game = TraversalGame(terrain_mode="flat", seed=42)
        p0_info, _ = _compute_single_player_inspector(game, 0)

        # In initial setup on row 0, pieces are at columns 2..7.
        # Explorer at (0, 4) has friendly neighbors at (0, 3) and (0, 5), etc.
        exp_piece = p0_info[0]  # Explorer 0
        assert exp_piece["status"] == 0
        assert len(exp_piece["legal_moves"]) > 0
        assert len(exp_piece["forbidden_moves"]) > 0

        # Check fields of forbidden moves
        reasons = set()
        for f in exp_piece["forbidden_moves"]:
            assert "r" in f and "c" in f
            assert f["is_legal"] is False
            assert f["action"] is None
            assert "forbidden_reason" in f
            assert "forbidden_desc" in f
            reasons.add(f["forbidden_reason"])

        # There must be friendly-occupied forbidden moves (e.g. adjacent allies on row 0)
        assert "friendly_occupied" in reasons

    def test_forbidden_moves_include_blocked_attacks_and_paths(self):
        # Create a game and place an enemy piece with defense > attacker
        game = TraversalGame(terrain_mode="flat", seed=100)
        # Advance ply or inspect pieces
        inspector = compute_detailed_ui_inspector(game)
        assert "players_pieces" in inspector
        assert len(inspector["players_pieces"]) == 2

        for pl in (0, 1):
            for pc in inspector["players_pieces"][pl]:
                if pc["status"] == 0:
                    assert "forbidden_moves" in pc
                    assert isinstance(pc["forbidden_moves"], list)
                    assert "legal_moves" in pc
                    assert isinstance(pc["legal_moves"], list)


class TestFinishPathCalculation:
    def test_finish_path_flat_terrain(self):
        terrain = np.zeros((10, 10), dtype=np.int32)
        # Sentinel (M=3) at (0, 2) moving to Row 9
        path_p1 = compute_piece_finish_path(terrain, player=0, piece_idx=4, r0=0, c0=2)
        assert path_p1 is not None
        assert "turns" in path_p1
        assert "waypoints" in path_p1
        assert "turn_segments" in path_p1
        assert "total_cost" in path_p1

        # Distance from row 0 to row 9 is 9 steps on flat (cost 9), plus 1 step off board = 10 cost
        # With budget 3:
        # Turn 1: 3 cost (reaches row 3)
        # Turn 2: 3 cost (reaches row 6)
        # Turn 3: 3 cost (reaches row 9)
        # Turn 4: 1 cost (steps off)
        # Total turns = 4
        assert path_p1["turns"] == 4
        assert path_p1["total_cost"] == 10
        assert len(path_p1["turn_segments"]) == 4

    def test_finish_path_respects_step_costs_and_budget(self):
        # Create a terrain with an elevation step of +2 between (0, 2) and (1, 2)
        # Step cost = |2 - 0| + 1 = 3
        terrain = np.zeros((10, 10), dtype=np.int32)
        terrain[1, 2] = 2

        # Sentinel (M=3) at (0, 2):
        # Moving to (1, 2) costs 3, consuming its entire turn budget in 1 step!
        path = compute_piece_finish_path(terrain, player=0, piece_idx=4, r0=0, c0=2)
        assert path is not None
        first_seg = path["turn_segments"][0]
        assert first_seg["turn"] == 1
        assert first_seg["cost"] <= 3

    def test_finish_path_player2(self):
        terrain = np.zeros((10, 10), dtype=np.int32)
        # Player 2 piece at (9, 5) moving to Row 0
        path_p2 = compute_piece_finish_path(terrain, player=1, piece_idx=0, r0=9, c0=5)
        assert path_p2 is not None
        assert path_p2["turns"] > 0
        # Final waypoint should be on Row 0
        final_seg = path_p2["turn_segments"][-1]
        assert final_seg["action"] == "finish"
        assert final_seg["to"][0] == 0


class TestSimulationMode:
    def test_simulation_mode_lifecycle(self):
        session = VisualizerSession()
        initial_state = session._serialize_current_state()
        initial_ply = initial_state["ply_count"]
        assert not initial_state["simulation_mode"]["active"]

        # 1. Start simulation
        sim_state = session.start_simulation()
        assert sim_state["simulation_mode"]["active"] is True
        assert sim_state["simulation_mode"]["sim_start_ply"] == initial_ply
        assert sim_state["simulation_mode"]["sim_plies_count"] == 0

        # 2. Make move in simulation
        p0_pc = sim_state["inspector"]["players_pieces"][0][0]
        legal_move = p0_pc["legal_moves"][0]
        action = legal_move["action"]
        assert action is not None

        moved_state = session.apply_human_move(action)
        assert moved_state["ply_count"] == initial_ply + 1
        assert moved_state["simulation_mode"]["active"] is True
        assert moved_state["simulation_mode"]["sim_plies_count"] == 1

        # 3. Reset simulation branch
        reset_state = session.reset_simulation()
        assert reset_state["ply_count"] == initial_ply
        assert reset_state["simulation_mode"]["active"] is True
        assert reset_state["simulation_mode"]["sim_plies_count"] == 0

        # Make another move
        session.apply_human_move(action)

        # 4. Exit simulation with commit=False (discard)
        exit_state = session.exit_simulation(commit=False)
        assert exit_state["simulation_mode"]["active"] is False
        assert exit_state["ply_count"] == initial_ply

        # 5. Exit simulation with commit=True (apply to match)
        session.start_simulation()
        session.apply_human_move(action)
        committed_state = session.exit_simulation(commit=True)
        assert committed_state["simulation_mode"]["active"] is False
        assert committed_state["ply_count"] == initial_ply + 1
