"""
Comprehensive unit tests for Milestone 1 (`traversal_env.py`).
Tests terrain generation, movement, Explorer special jumps, Vanguard d_min=4 charge bonus,
self-exclusion in allied support, defense math, finishing, and all 4 termination conditions.
"""

import numpy as np
import pytest

from traversal_env import (
    TraversalGame,
    generate_terrain,
    TERRAIN_MODES,
    PIECE_MOVEMENT,
    PIECE_STRENGTH,
    STATUS_ACTIVE,
    STATUS_CAPTURED,
    STATUS_FINISHED,
    FINISH_TARGET,
    MAX_PLIES,
)


def make_empty_game(terrain=None) -> TraversalGame:
    """Creates a game with flat terrain (or custom terrain) and all pieces removed from board."""
    g = TraversalGame(terrain_mode="flat", terrain=terrain)
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)
    g._cached_legal_mask = None
    return g


def place_piece(g: TraversalGame, player: int, piece_idx: int, r: int, c: int) -> None:
    g.pos[player, piece_idx] = [r, c]
    g.piece_status[player, piece_idx] = STATUS_ACTIVE
    g._cached_legal_mask = None


# ---------------------------------------------------------------------------
# 1. Board & Terrain Generation Tests
# ---------------------------------------------------------------------------


def test_terrain_generation_modes_and_slope_constraint():
    rng = np.random.default_rng(42)
    for mode in TERRAIN_MODES:
        for _ in range(15):
            h = generate_terrain(mode=mode, rng=rng)
            assert h.shape == (10, 10)
            assert h.dtype == np.int32
            assert np.all(h >= 0)
            # Orthogonal slope constraint: |h(r1, c1) - h(r2, c2)| <= 2
            diff_r = np.abs(h[1:, :] - h[:-1, :])
            diff_c = np.abs(h[:, 1:] - h[:, :-1])
            assert np.all(diff_r <= 2), f"Row slope > 2 in mode {mode}"
            assert np.all(diff_c <= 2), f"Col slope > 2 in mode {mode}"


def test_starting_setup_and_piece_stats():
    g = TraversalGame(terrain_mode="flat")
    # Movement + Strength == 6 for all 6 pieces
    assert np.all(PIECE_MOVEMENT + PIECE_STRENGTH == 6)

    # P1 on Row 0, cols 2..7: [S0, V0, E0, E1, V1, S1]
    p1_cols = {tuple(g.pos[0, k]): k for k in range(6)}
    assert p1_cols[(0, 2)] == 4  # S0
    assert p1_cols[(0, 3)] == 2  # V0
    assert p1_cols[(0, 4)] == 0  # E0
    assert p1_cols[(0, 5)] == 1  # E1
    assert p1_cols[(0, 6)] == 3  # V1
    assert p1_cols[(0, 7)] == 5  # S1

    # P2 on Row 9, cols 2..7 (180-degree rotated canonical layout)
    p2_cols = {tuple(g.pos[1, k]): k for k in range(6)}
    assert p2_cols[(9, 7)] == 4  # S0
    assert p2_cols[(9, 6)] == 2  # V0
    assert p2_cols[(9, 5)] == 0  # E0
    assert p2_cols[(9, 4)] == 1  # E1
    assert p2_cols[(9, 3)] == 3  # V1
    assert p2_cols[(9, 2)] == 5  # S1


# ---------------------------------------------------------------------------
# 2. Movement & Blocking Tests
# ---------------------------------------------------------------------------


def test_step_cost_and_blocking():
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[1, 4] = 2  # Step from (0,4) to (1,4) costs |2 - 0| + 1 = 3
    terrain[2, 4] = 2  # Step from (1,4) to (2,4) costs |2 - 2| + 1 = 1 (total 4)

    g = make_empty_game(terrain=terrain)
    # Place Sentinel_0 (M=3) at (0, 4) and dummy enemy piece at (9, 9)
    place_piece(g, 0, 4, 0, 4)
    place_piece(g, 1, 0, 9, 9)

    legal_dest, _, d_min, _, _, _ = g.get_player_combat_and_reach(0)
    # Sentinel has M=3: can reach (1, 4) at cost 3, but NOT (2, 4) at cost 4
    assert d_min[4, 1, 4] == 3
    assert legal_dest[4, 1, 4] == 1
    assert legal_dest[4, 2, 4] == 0
    # No null move (d=0 is illegal)
    assert legal_dest[4, 0, 4] == 0

    # Blocking: place a friendly piece at (1, 4) -> blocks path and cannot land on friendly
    place_piece(g, 0, 5, 1, 4)
    legal_dest_blocked, _, _, _, _, _ = g.get_player_combat_and_reach(0)
    assert legal_dest_blocked[4, 1, 4] == 0


def test_explorer_special_moves_ignore_height_and_jump_over_pieces():
    terrain = np.zeros((10, 10), dtype=np.int32)
    # Make adjacent squares very high so standard movement costs 3 per step
    terrain[4, :] = 6
    terrain[5, :] = 6
    terrain[3, 3] = 0

    g = make_empty_game(terrain=terrain)
    # Place Explorer_0 (idx 0, M=5) at (3, 3)
    place_piece(g, 0, 0, 3, 3)
    # Block orthogonal neighbor (4, 3) with a friendly piece
    place_piece(g, 0, 2, 4, 3)
    # Place opponent piece far away
    place_piece(g, 1, 0, 9, 9)

    legal_dest, _, d_min, _, _, _ = g.get_player_combat_and_reach(0)

    # Diagonal step to (4, 4) and (4, 2) should be legal despite height difference of 6!
    assert legal_dest[0, 4, 4] == 1
    assert legal_dest[0, 4, 2] == 1

    # 2-square orthogonal leap to (5, 3) over friendly piece at (4, 3) and height=6 should be legal!
    assert d_min[0, 5, 3] > 5  # Unreachable by standard movement
    assert legal_dest[0, 5, 3] == 1


# ---------------------------------------------------------------------------
# 3. Combat Rules & Edge Cases (Section 2.4)
# ---------------------------------------------------------------------------


def test_combat_height_modifier_uses_starting_square():
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[2, 2] = 2  # Attacker starts at height 2
    terrain[2, 3] = 0  # Intermediate square at height 0 (step cost 3)
    terrain[2, 4] = 1  # Defender at height 1 (step cost 2, total cost 5)

    g = make_empty_game(terrain=terrain)
    # Explorer_0 (M=5, S=1) at (2, 2); Opponent Vanguard_0 (S=2) at (2, 4)
    place_piece(g, 0, 0, 2, 2)
    place_piece(g, 1, 2, 2, 4)

    legal_dest, _, d_min, max_atk, _, opp_def = g.get_player_combat_and_reach(0)
    assert d_min[0, 2, 4] == 5
    # Even though intermediate (2,3) has h=0 < h(2,4)=1, starting square (2,2) has h=2 > h(2,4)=1!
    # So HeightMod = +1 -> Attack = S_0(1) + 1 = 2 == Defense(2) -> legal capture!
    assert opp_def[2, 4] == 2.0
    assert max_atk[2, 4] == 2.0
    assert legal_dest[0, 2, 4] == 1


def test_vanguard_charge_bonus_requires_exact_shortest_path_4():
    g = make_empty_game()
    # Vanguard_0 (idx 2, M=4, S=2) at (2, 2) on flat terrain
    place_piece(g, 0, 2, 2, 2)
    # Enemy Sentinel_0 (idx 4, S=3) at (2, 6) -> shortest path on flat terrain is d_min = 4
    place_piece(g, 1, 4, 2, 6)
    # Another Enemy Sentinel_1 (idx 5, S=3) at (2, 4) -> shortest path is d_min = 2 (even though a 4-step detour exists!)
    place_piece(g, 1, 5, 2, 4)

    legal_dest, _, d_min, max_atk, _, _ = g.get_player_combat_and_reach(0)

    # Target (2, 4) is at d_min = 2 -> no charge bonus -> Attack = 2 < Defense = 3 (illegal!)
    assert d_min[2, 2, 4] == 2
    assert max_atk[2, 4] == 2.0
    assert legal_dest[2, 2, 4] == 0

    # Wait: (2, 4) blocks the direct straight line to (2, 6)!
    # Let's check: around (2, 4), going via (3,2)->(3,6) takes 6 steps > 4.
    # Move Enemy Sentinel_1 from (2, 4) to (2, 0) (also d_min = 2 from (2, 2)) so (2, 6) is unblocked!
    g.pos[1, 5] = [2, 0]
    g._cached_legal_mask = None

    legal_dest, _, d_min, max_atk, _, _ = g.get_player_combat_and_reach(0)
    # Target (2, 0) has d_min = 2 -> Attack = 2 < Defense = 3 -> illegal
    assert d_min[2, 2, 0] == 2
    assert max_atk[2, 0] == 2.0
    assert legal_dest[2, 2, 0] == 0

    # Target (2, 6) has d_min = 4 -> Vanguard charge bonus +1 -> Attack = 2 + 1 = 3 >= Defense = 3 -> legal!
    assert d_min[2, 2, 6] == 4
    assert max_atk[2, 6] == 3.0
    assert legal_dest[2, 2, 6] == 1


def test_allied_support_excludes_attacking_piece_itself():
    """
    Crucial edge case (Section 2.4):
    If attacking piece i started the turn already orthogonally adjacent to (r_t, c_t),
    piece i must NOT count itself as an adjacent friendly supporter!
    """
    g = make_empty_game()
    # Vanguard_0 (idx 2, S=2) at (4, 4), adjacent to enemy Sentinel_0 (idx 4, S=3) at (4, 5)
    place_piece(g, 0, 2, 4, 4)
    place_piece(g, 1, 4, 4, 5)

    legal_dest, _, _, max_atk, _, opp_def = g.get_player_combat_and_reach(0)
    # Without other allies, Vanguard_0 at (4,4) has Attack = 2 (NOT 2 + 2 = 4!) vs Defense = 3
    assert opp_def[4, 5] == 3.0
    assert max_atk[4, 5] == 2.0
    assert legal_dest[2, 4, 5] == 0

    # Now add a friendly Explorer_0 (idx 0, S=1) at (3, 5) adjacent to target (4, 5)
    place_piece(g, 0, 0, 3, 5)
    legal_dest2, _, _, max_atk2, _, _ = g.get_player_combat_and_reach(0)
    # Vanguard_0 now gets +1 support from Explorer_0 -> Attack = 2 + 1 = 3 >= Defense 3 -> legal!
    assert max_atk2[4, 5] == 3.0
    assert legal_dest2[2, 4, 5] == 1


def test_defense_adds_one_per_adjacent_friendly_regardless_of_type():
    g = make_empty_game()
    # Enemy Explorer_0 (S=1) at (5, 5), supported by 2 adjacent Enemy Sentinels at (5, 4) and (6, 5)
    place_piece(g, 1, 0, 5, 5)
    place_piece(g, 1, 4, 5, 4)
    place_piece(g, 1, 5, 6, 5)

    # Friendly Vanguard_0 (S=2) at (4, 5)
    place_piece(g, 0, 2, 4, 5)

    legal_dest, _, _, max_atk, _, opp_def = g.get_player_combat_and_reach(0)
    # Defense at (5, 5) = S_explorer(1) + 2 adjacent allies = 3
    assert opp_def[5, 5] == 3.0
    # Vanguard at (4, 5) has Attack = 2 < 3 -> illegal!
    assert max_atk[5, 5] == 2.0
    assert legal_dest[2, 5, 5] == 0


# ---------------------------------------------------------------------------
# 4. Finishing & Termination Condition Tests (Section 2.5)
# ---------------------------------------------------------------------------


def test_finishing_rules_and_explorer_special_move_cannot_finish():
    g = make_empty_game()
    # Explorer_0 (M=5) at (5, 2) -> distance to Row 9 is 4 <= M - 1 (4) -> CAN finish!
    place_piece(g, 0, 0, 5, 2)
    # Explorer_1 (M=5) at (4, 7) -> distance to Row 9 is 5 == M (needs M-1=4) -> CANNOT finish!
    place_piece(g, 0, 1, 4, 7)
    # Sentinel_0 (M=3) at (9, 4) -> starts on Row 9 -> CAN finish!
    place_piece(g, 0, 4, 9, 4)
    place_piece(g, 1, 0, 0, 0)

    _, can_finish, _, _, _, _ = g.get_player_combat_and_reach(0)
    assert can_finish[0] == 1
    assert can_finish[1] == 0
    assert can_finish[4] == 1

    # Also check Explorer jumping onto Row 9 via 2-tile leap over a high cliff cannot finish on the same turn
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[8, :] = 2
    terrain[9, :] = 4  # Stepping 7->8 costs 3, 8->9 costs 3 (total 6 > 4)
    g2 = make_empty_game(terrain=terrain)
    place_piece(g2, 0, 0, 7, 3)  # Can 2-leap to (9, 3), but d_min to Row 9 is 6 > 4
    place_piece(g2, 1, 0, 0, 0)
    legal_dest2, can_finish2, _, _, _, _ = g2.get_player_combat_and_reach(0)
    assert legal_dest2[0, 9, 3] == 1  # Can leap onto (9, 3)
    assert can_finish2[0] == 0  # Cannot step off the board on that same turn!


def test_sentinel_instant_win():
    g = make_empty_game()
    place_piece(g, 0, 4, 8, 4)  # P1 Sentinel_0 at (8, 4), 1 step from Row 9 (M=3, 1 <= 2)
    place_piece(g, 1, 0, 1, 4)  # P2 Explorer_0 at (1, 4)

    done, outcome_for_next = g.step_move(piece_idx=4, finish=True)
    assert done is True
    assert g.termination_reason == "sentinel_instant_win"
    assert g.winner == 0
    assert g.get_outcome(0) == 1.0
    assert g.get_outcome(1) == -1.0
    assert outcome_for_next == -1.0  # Next player is P2 (1), who lost


def test_two_piece_finish_and_second_finisher_sentinel_is_not_instant_win():
    g = make_empty_game()
    # P1 has Explorer_0 (1 pt) and Vanguard_0 (2 pts) ready to finish (total 3 pts)
    place_piece(g, 0, 0, 9, 2)
    place_piece(g, 0, 2, 9, 3)
    # P2 has Sentinel_0 (3 pts) ready to finish on Row 0
    place_piece(g, 1, 4, 0, 5)
    place_piece(g, 1, 0, 5, 5)

    # 1st finish: P1 finishes Explorer_0 (score P1=1, P2=0, total_finished=1)
    done, _ = g.step_move(piece_idx=0, finish=True)
    assert not done
    assert g.scores[0] == 1
    assert g.total_pieces_finished == 1

    # 2nd finish: P2 finishes Sentinel_0 (score P1=1, P2=3, total_finished=2)
    # Ends by two_piece_finish (NOT sentinel_instant_win since it was the 2nd piece to finish), P2 wins 3 vs 1!
    done, _ = g.step_move(piece_idx=4, finish=True)
    assert done is True
    assert g.termination_reason == "two_piece_finish"
    assert g.winner == 1
    assert g.scores[1] == 3


def test_elimination_and_200_ply_tiebreakers():
    # 1. Elimination: P1 captures P2's only piece
    g = make_empty_game()
    place_piece(g, 0, 4, 4, 4)  # P1 Sentinel_0 (S=3)
    place_piece(g, 1, 0, 4, 5)  # P2 Explorer_0 (S=1)

    done, _ = g.step_move(piece_idx=4, target_r=4, target_c=5)
    assert done is True
    assert g.termination_reason == "elimination"
    assert g.winner == 0

    # 2. 200-Ply Limit Tiebreaker: closest piece wins; if tied distance, stronger piece wins
    g2 = make_empty_game()
    place_piece(g2, 0, 0, 6, 2)  # P1 Explorer_0 (S=1) at row 6 -> 3 rows remaining (9 - 6 = 3)
    place_piece(g2, 1, 2, 4, 7)  # P2 Vanguard_0 (S=2) at row 4 -> 4 rows remaining
    g2.ply_count = MAX_PLIES - 1

    # P1 moves Explorer_0 from (6, 2) to (6, 3) -> still 3 rows remaining, ply reaches 200
    done, _ = g2.step_move(piece_idx=0, target_r=6, target_c=3)
    assert done is True
    assert g2.termination_reason == "ply_limit"
    assert g2.winner == 0  # P1 has 3 rows remaining vs P2's 4 rows remaining

    # Now test equal distance tiebreaker (stronger piece wins)
    g3 = make_empty_game()
    place_piece(g3, 0, 0, 6, 2)  # P1 Explorer_0 (S=1) at row 6 (3 rows remaining)
    place_piece(g3, 1, 2, 3, 7)  # P2 Vanguard_0 (S=2) at row 3 (3 rows remaining to row 0!)
    g3.ply_count = MAX_PLIES - 1
    done, _ = g3.step_move(piece_idx=0, target_r=6, target_c=3)
    assert done is True
    assert g3.termination_reason == "ply_limit"
    assert g3.winner == 1  # Both 3 rows from finish, but P2's Vanguard (S=2) > P1's Explorer (S=1)


def test_random_self_play_rollouts():
    rng = np.random.default_rng(123)
    for game_idx in range(30):
        g = TraversalGame(terrain_mode="random", seed=int(rng.integers(0, 100000)))
        while not g.done:
            legal_actions = g.get_legal_actions()
            assert len(legal_actions) > 0
            action = int(rng.choice(legal_actions))
            g.step(action)
        assert g.ply_count <= MAX_PLIES
        assert g.winner in (0, 1, -1)
        assert g.termination_reason in (
            "sentinel_instant_win",
            "two_piece_finish",
            "elimination",
            "no_legal_moves",
            "ply_limit",
            "threefold_repetition",
        )


def test_threefold_repetition_exact_cycle():
    """
    Verifies that reaching the exact same position for the 3rd time results in
    an immediate draw (done=True, winner=-1, termination_reason='threefold_repetition', outcome=0.0).
    """
    g = make_empty_game()
    # P1 Explorer_0 at (3, 3), P2 Explorer_0 at (7, 7)
    place_piece(g, 0, 0, 3, 3)
    place_piece(g, 1, 0, 7, 7)
    g.reset_repetition_history()

    # Move 1: P1 (3,3)->(3,4)
    done, _ = g.step_move(piece_idx=0, target_r=3, target_c=4)
    assert not done
    # Move 2: P2 (7,7)->(7,8)
    done, _ = g.step_move(piece_idx=0, target_r=7, target_c=8)
    assert not done
    # Move 3: P1 (3,4)->(3,3)
    done, _ = g.step_move(piece_idx=0, target_r=3, target_c=3)
    assert not done
    # Move 4: P2 (7,8)->(7,7) -> 2nd occurrence of initial position
    done, _ = g.step_move(piece_idx=0, target_r=7, target_c=7)
    assert not done
    assert g.winner is None

    # Move 5: P1 (3,3)->(3,4)
    done, _ = g.step_move(piece_idx=0, target_r=3, target_c=4)
    assert not done
    # Move 6: P2 (7,7)->(7,8)
    done, _ = g.step_move(piece_idx=0, target_r=7, target_c=8)
    assert not done
    # Move 7: P1 (3,4)->(3,3)
    done, _ = g.step_move(piece_idx=0, target_r=3, target_c=3)
    assert not done
    # Move 8: P2 (7,8)->(7,7) -> 3rd occurrence of initial position!
    done, outcome = g.step_move(piece_idx=0, target_r=7, target_c=7)
    assert done is True
    assert g.winner == -1
    assert g.termination_reason == "threefold_repetition"
    assert outcome == 0.0
    assert g.get_outcome(0) == 0.0
    assert g.get_outcome(1) == 0.0


def test_threefold_repetition_non_consecutive():
    """
    Verifies that threefold repetition works even when non-consecutive moves are interspersed.
    """
    g = make_empty_game()
    place_piece(g, 0, 0, 2, 2)
    place_piece(g, 1, 0, 8, 8)
    g.reset_repetition_history()

    # Initial position at ply 0 (count = 1)
    # Ply 1: P1 (2,2) -> (2,3)
    g.step_move(piece_idx=0, target_r=2, target_c=3)
    # Ply 2: P2 (8,8) -> (8,7)
    g.step_move(piece_idx=0, target_r=8, target_c=7)
    # Ply 3: P1 (2,3) -> (2,2)
    g.step_move(piece_idx=0, target_r=2, target_c=2)
    # Ply 4: P2 (8,7) -> (8,8) -> Return to initial position (count = 2)
    g.step_move(piece_idx=0, target_r=8, target_c=8)
    assert not g.done

    # Interspersed wander:
    # Ply 5: P1 (2,2) -> (3,2)
    g.step_move(piece_idx=0, target_r=3, target_c=2)
    # Ply 6: P2 (8,8) -> (7,8)
    g.step_move(piece_idx=0, target_r=7, target_c=8)
    # Ply 7: P1 (3,2) -> (2,2)
    g.step_move(piece_idx=0, target_r=2, target_c=2)
    # Ply 8: P2 (7,8) -> (8,8) -> Return to initial position (count = 3)!
    g.step_move(piece_idx=0, target_r=8, target_c=8)
    assert g.done is True
    assert g.winner == -1
    assert g.termination_reason == "threefold_repetition"


def test_threefold_repetition_capture_resets_history():
    """
    Verifies that captures permanently alter material and reset repetition history
    so pre-capture positions do not cause false repetition.
    """
    g = make_empty_game()
    # P1 Sentinel_0 at (4, 4), P1 Explorer_0 at (2, 2)
    place_piece(g, 0, 4, 4, 4)
    place_piece(g, 0, 0, 2, 2)
    # P2 Sentinel_0 at (4, 6), P2 Explorer_0 at (4, 5) [adjacent to P1 Sentinel]
    place_piece(g, 1, 4, 4, 6)
    place_piece(g, 1, 0, 4, 5)
    g.reset_repetition_history()

    # P1 moves Explorer back and forth to reach count=2
    # Ply 1: P1 E (2,2)->(2,3)
    g.step_move(piece_idx=0, target_r=2, target_c=3)
    # Ply 2: P2 S (4,6)->(3,6)
    g.step_move(piece_idx=4, target_r=3, target_c=6)
    # Ply 3: P1 E (2,3)->(2,2)
    g.step_move(piece_idx=0, target_r=2, target_c=2)
    # Ply 4: P2 S (3,6)->(4,6) -> initial pos count = 2
    g.step_move(piece_idx=4, target_r=4, target_c=6)
    assert not g.done

    # Ply 5: P1 Sentinel captures P2 Explorer at (4, 5)!
    done, _ = g.step_move(piece_idx=4, target_r=4, target_c=5)
    assert not done
    assert g.pos[1, 0, 0] == -1  # Captured!
    # Repetition history must be reset after capture
    assert len(g.position_counts) == 1


def test_threefold_repetition_identical_piece_swap():
    """
    Verifies that identical pieces swapping squares is recognized as the same position.
    """
    g = make_empty_game()
    # P1 has two Explorers at (2, 2) and (2, 4)
    place_piece(g, 0, 0, 2, 2)
    place_piece(g, 0, 1, 2, 4)
    # P2 has one piece far away at (8, 8)
    place_piece(g, 1, 0, 8, 8)
    g.reset_repetition_history()

    key_orig = g.get_position_key()

    # Swap P1 E0 and E1 in place
    g.pos[0, 0] = [2, 4]
    g.pos[0, 1] = [2, 2]
    key_swapped = g.get_position_key()

    assert key_orig == key_swapped, "Identical piece permutation must produce the same position key"


def test_explorer_leap_cannot_capture():
    """
    Verifies that an Explorer's 2-square orthogonal leap is strictly for movement
    to an empty square and CANNOT be used to capture an enemy piece.
    """
    # Create terrain where standard movement cannot go around due to cliffs
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[3, :] = 4
    terrain[5, :] = 4  # (3, c) and (5, c) are height 4 cliffs, impassable in standard budget
    g = make_empty_game(terrain=terrain)

    # P1 Explorer_0 at (4, 4) (height 0)
    place_piece(g, 0, 0, 4, 4)
    # Enemy piece at (4, 5) blocks standard orthogonal path to (4, 6)
    place_piece(g, 1, 1, 4, 5)
    # Enemy piece (Explorer) at (4, 6) - reachable ONLY via 2-square orthogonal leap
    place_piece(g, 1, 0, 4, 6)

    legal_dest, _, d_min, max_atk, _, _ = g.get_player_combat_and_reach(0)

    # Standard movement to (4, 6) is blocked
    assert d_min[0, 4, 6] > 5
    # 2-square leap onto enemy at (4, 6) must NOT be legal
    assert legal_dest[0, 4, 6] == 0
    # Attack threat on (4, 6) from the leap must NOT be registered
    assert max_atk[4, 6] == 0.0

    # However, 2-square leap in another direction to an EMPTY square (4, 2) IS legal
    assert legal_dest[0, 4, 2] == 1


def test_explorer_standard_walking_capture_to_diagonal_square():
    """
    Verifies that when orthogonal paths are clear, an Explorer can walk 2 orthogonal
    steps around a corner to capture an enemy at (5, 5).
    """
    g = make_empty_game()
    # P1 Explorer_0 at (4, 4) (base attack 1)
    place_piece(g, 0, 0, 4, 4)
    # P2 Explorer_0 at (5, 5) (diagonal, base defense 1)
    place_piece(g, 1, 0, 5, 5)
    # Second enemy piece far away
    place_piece(g, 1, 1, 8, 8)

    legal_dest, _, d_min, max_atk, _, _ = g.get_player_combat_and_reach(0)

    # Reachable via 2 standard orthogonal steps around the corner
    assert d_min[0, 5, 5] == 2
    assert legal_dest[0, 5, 5] == 1
    assert max_atk[5, 5] >= 1.0

    # Step move: P1 Explorer captures P2 Explorer at (5, 5)
    done, _ = g.step_move(piece_idx=0, target_r=5, target_c=5)
    assert not done
    assert g.pos[0, 0, 0] == 5 and g.pos[0, 0, 1] == 5
    assert g.pos[1, 0, 0] == -1  # Captured!


def test_explorer_special_diagonal_step_cannot_capture():
    """
    Verifies that when orthogonal paths are blocked, an Explorer CANNOT use its
    special diagonal step to capture an enemy piece (movement to empty squares only).
    """
    # Surround (4, 4) orthogonally with cliffs at (4, 5) and (5, 4) so standard path is blocked
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[4, 5] = 4
    terrain[5, 4] = 4
    terrain[3, :] = 4
    g = make_empty_game(terrain=terrain)

    place_piece(g, 0, 0, 4, 4)
    place_piece(g, 1, 0, 5, 5)
    place_piece(g, 1, 1, 8, 8)

    legal_dest, _, d_min, max_atk, _, _ = g.get_player_combat_and_reach(0)
    # Orthogonal movement cannot reach (5, 5) within budget
    assert d_min[0, 5, 5] > 5
    # Special diagonal step onto enemy is NOT legal
    assert legal_dest[0, 5, 5] == 0
    assert max_atk[5, 5] == 0.0

    # Special diagonal step to an EMPTY square (5, 3) across cliffs IS legal
    assert legal_dest[0, 5, 3] == 1

