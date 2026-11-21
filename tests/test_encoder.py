"""
Unit tests for Milestone 2 (`encoder.py`).
Verifies static/dynamic tensor shapes, normalization constants, origin exit flags,
180-degree canonical perspective rotation, and exact commutation of horizontal symmetry.
"""

import numpy as np
import pytest

from traversal_env import TraversalGame, STATUS_ACTIVE, STATUS_CAPTURED
from encoder import (
    encode_static_tensor,
    encode_dynamic_tensor,
    encode_game_state,
    mirror_static,
    mirror_dynamic,
    mirror_policy,
    PIECE_SWAP_PERM,
)


def test_static_tensor_encoding_and_normalization():
    terrain = np.zeros((10, 10), dtype=np.int32)
    terrain[5, 5] = 3  # Isolated spike: median remains 0.0

    x_s0 = encode_static_tensor(terrain, player=0)
    assert x_s0.shape == (4, 10, 10)
    assert x_s0.dtype == np.float32

    # Static 0: median-centered height / 5.0 -> delta h = 1 is 0.20, so h=3 is 0.60
    assert np.isclose(x_s0[0, 0, 0], 0.0)
    assert np.isclose(x_s0[0, 5, 5], 0.60)

    # On completely flat board, verify exact unblocked distances
    flat = np.zeros((10, 10), dtype=np.int32)
    x_flat = encode_static_tensor(flat, player=0)
    for r in range(10):
        # Distance to step off Row 9 is (10 - r) / 30.0
        assert np.allclose(x_flat[1, r, :], (10 - r) / 30.0)
        # Distance to step off Row 0 is (r + 1) / 30.0
        assert np.allclose(x_flat[3, r, :], (r + 1) / 30.0)

    # Static 2: Explorer turns to step off Row 9 on flat board
    # Rows 5..9 have dist <= 5 -> 1 turn (0.1); Rows 0..4 take 2 turns (0.2)
    assert np.allclose(x_flat[2, 5:, :], 0.1)
    assert np.allclose(x_flat[2, :5, :], 0.2)


def test_dynamic_tensor_origin_exit_flag_and_mask_bijection():
    g = TraversalGame(terrain_mode="flat")
    # Move P1 Explorer_0 to (6, 2) so it can finish next turn (dist to Row 9 is 3 <= 4)
    g.pos[0, 0] = [6, 2]
    # Also place P2 Sentinel_0 at (1, 1) so P2 can finish on P2's turn via empty (0, 1) (dist 1 <= 2)
    g.pos[1, 4] = [1, 1]
    g.scores[0] = 2
    g.scores[1] = 1
    g.total_pieces_finished = 1
    g.ply_count = 40
    g._cached_legal_mask = None

    x_static, x_dyn, legal_mask = encode_game_state(g)
    assert x_static.shape == (4, 10, 10)
    assert x_dyn.shape == (34, 10, 10)
    assert legal_mask.shape == (606,)

    # Dyn 0..23 must be strictly binary {0.0, 1.0}
    unique_vals = np.unique(x_dyn[:24])
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    # Verify origin exit flag for P1 Explorer_0 at canonical (6, 2)
    assert x_dyn[0, 6, 2] == 1.0
    assert x_dyn[12, 6, 2] == 1.0  # Origin exit flag set because Explorer_0 can finish!
    assert legal_mask[0 * 101 + 100] == True

    # Verify origin exit flag for P2 Sentinel_0 at physical (1, 1) (canonical (1, 1) in P1's view)
    assert x_dyn[6 + 4, 1, 1] == 1.0
    assert x_dyn[18 + 4, 1, 1] == 1.0  # Opponent origin exit flag set!

    # Check exact bijection between my_reach[k] (Dyn 12..17) and legal_mask (606,)
    for k in range(6):
        r0, c0 = g.to_canonical_coords(int(g.pos[0, k, 0]), int(g.pos[0, k, 1]), 0)
        reach_plane = x_dyn[12 + k].copy()
        exit_flag = reach_plane[r0, c0]
        reach_plane[r0, c0] = 0.0  # Clear origin pixel to compare on-board destinations
        mask_slice = legal_mask[k * 101 : k * 101 + 100].reshape(10, 10)
        assert np.array_equal(reach_plane > 0.5, mask_slice)
        assert bool(exit_flag > 0.5) == bool(legal_mask[k * 101 + 100])

    # Check scalar planes (Dyn 30..33)
    assert np.allclose(x_dyn[30], 2.0 / 6.0)
    assert np.allclose(x_dyn[31], 1.0 / 6.0)
    assert np.allclose(x_dyn[32], 1.0 / 2.0)
    assert np.allclose(x_dyn[33], 40.0 / 200.0)


def test_180_degree_canonical_perspective_rotation_for_player_2():
    # Construct a state for P1 and the exact 180-degree rotated state for P2
    rng = np.random.default_rng(7)
    g1 = TraversalGame(terrain_mode="mountains", seed=7)
    for _ in range(11):  # Play 11 plies so it becomes P2's turn (current_player == 1)
        actions = g1.get_legal_actions()
        g1.step(int(rng.choice(actions)))

    assert g1.current_player == 1
    x_s_p2, x_d_p2, mask_p2 = encode_game_state(g1)

    # Check bijection between P2's encoded my_reach (Dyn 12..17) and P2's canonical legal_mask
    for k in range(6):
        r_phys, c_phys = int(g1.pos[1, k, 0]), int(g1.pos[1, k, 1])
        if r_phys < 0:
            continue
        r_canon, c_canon = g1.to_canonical_coords(r_phys, c_phys, 1)
        assert x_d_p2[k, r_canon, c_canon] == 1.0
        reach_plane = x_d_p2[12 + k].copy()
        exit_flag = reach_plane[r_canon, c_canon]
        reach_plane[r_canon, c_canon] = 0.0
        mask_slice = mask_p2[k * 101 : k * 101 + 100].reshape(10, 10)
        assert np.array_equal(reach_plane > 0.5, mask_slice)
        assert bool(exit_flag > 0.5) == bool(mask_p2[k * 101 + 100])


def test_horizontal_symmetry_augmentation_exact_commutation():
    """
    Verifies that reflecting a physical board state horizontally (flipping c <-> 9 - c
    and swapping E0<->E1, V0<->V1, S0<->S1) and encoding it gives the exact same
    tensors and legal mask as calling mirror_static / mirror_dynamic / mirror_policy.
    """
    rng = np.random.default_rng(99)
    for ply_target in (0, 7, 14, 23):
        g = TraversalGame(terrain_mode="random", seed=100 + ply_target)
        for _ in range(ply_target):
            if g.done:
                break
            g.step(int(rng.choice(g.get_legal_actions())))

        # Build the physically mirrored game state
        g_mir = g.clone()
        g_mir.terrain = np.ascontiguousarray(g.terrain[:, ::-1])
        g_mir.pos = g.pos[:, PIECE_SWAP_PERM, :].copy()
        g_mir.piece_status = g.piece_status[:, PIECE_SWAP_PERM].copy()
        for p in (0, 1):
            for k in range(6):
                if g_mir.pos[p, k, 0] >= 0:
                    g_mir.pos[p, k, 1] = 9 - g_mir.pos[p, k, 1]
        g_mir._cached_legal_mask = None

        xs, xd, mask = encode_game_state(g)
        xs_m, xd_m, mask_m = encode_game_state(g_mir)

        assert np.allclose(mirror_static(xs), xs_m, atol=1e-6)
        assert np.allclose(mirror_dynamic(xd), xd_m, atol=1e-6)
        assert np.array_equal(mirror_policy(mask), mask_m)


def test_blocked_distance_v2_vs_v1():
    """
    Verifies that compute_blocked_dist_std_v2 computes realistic distances for occupied squares
    (review #6), whereas v1 clamped every occupied square to 30.0 (providing 0 distance info).
    Also verifies that v2 strictly forbids paths from passing THROUGH occupied squares.
    """
    from encoder import compute_blocked_dist_std, compute_blocked_dist_std_v2

    terrain = np.zeros((10, 10), dtype=np.int32)
    my_pos = np.full((6, 2), -1, dtype=np.int32)
    opp_pos = np.full((6, 2), -1, dtype=np.int32)

    # Place a friendly piece at (8, 5) with goal row 9 on flat terrain
    # Distance to step off row 9 from row 8 is (abs(9-8)+1) + 1 = 2
    my_pos[0] = [8, 5]

    d_v1 = compute_blocked_dist_std(terrain, my_pos, opp_pos, target_row=9)
    d_v2 = compute_blocked_dist_std_v2(terrain, my_pos, opp_pos, target_row=9)

    # In v1, occupied square (8, 5) was never relaxed and stayed at 30
    assert d_v1[8, 5] == 30.0
    # In v2, occupied square (8, 5) receives the valid distance 2.0 (step from (8,5) to (9,5) costs 1, step off row 9 costs 1)
    assert d_v2[8, 5] == 2.0

    # Now build a horizontal wall of pieces across row 8, columns 1..9 (leaving ONLY column 0 open)
    for i in range(6):
        opp_pos[i] = [8, i + 1]  # cols 1..6
    for i in range(3):
        my_pos[i + 1] = [8, 7 + i]  # cols 7..9
    # Place friendly piece 0 behind the wall at (7, 5)
    my_pos[0] = [7, 5]

    d_v2_walled = compute_blocked_dist_std_v2(terrain, my_pos, opp_pos, target_row=9)
    # The path cannot penetrate through the occupied pieces on row 8.
    # It must navigate around row 8 col 0:
    # (7, 5) -> ... -> (7, 0) -> (8, 0) -> (9, 0) -> step off
    # Steps: 5 steps left to col 0 (cost 5), 1 step down to (8,0) (cost 1), 1 step to (9,0) (cost 1), 1 step off (cost 1) = 8
    assert d_v2_walled[7, 5] == 8.0
    # And row 8 col 5 (which is occupied by an opponent piece) cannot be traversed through,
    # but gets its distance from unblocked neighbors (7, 5) or (9, 5): from (9, 5) it is 1 + 1 = 2
    assert d_v2_walled[8, 5] == 2.0


def test_encoder_v2_channels_attack_cleanup_and_repetition():
    """
    Verifies:
      1. encode_dynamic_tensor with version='v2' has 35 channels (vs 34 for v1).
      2. Friendly attack cleanup (review #5.4): friendly occupied squares have 0.0 attack value.
      3. Repetition plane Dyn 34 (review #10): 0.0 initially, 1.0 when position repeated >= 2 times.
    """
    g = TraversalGame(terrain_mode="flat", seed=42)
    # Move P0 Explorer_0
    g.pos[0, 0] = [2, 2]
    # Place P0 Explorer_1 adjacent at [2, 3]
    g.pos[0, 1] = [2, 3]

    x_dyn_v1 = encode_dynamic_tensor(g, version="v1")
    assert x_dyn_v1.shape == (34, 10, 10)

    x_dyn_v2 = encode_dynamic_tensor(g, version="v2")
    assert x_dyn_v2.shape == (35, 10, 10)

    # In v2, friendly occupied square (2, 3) must have 0.0 attack value in my_max_atk (Dyn 24)
    assert x_dyn_v2[24, 2, 3] == 0.0

    # Initial position repetition count is 1 -> Dyn 34 must be 0.0
    assert np.all(x_dyn_v2[34] == 0.0)

    # Artificially record the position key a second time
    key = g.get_position_key()
    g.position_counts[key] = 2

    x_dyn_v2_rep = encode_dynamic_tensor(g, version="v2")
    assert np.all(x_dyn_v2_rep[34] == 1.0)


def test_mirror_dynamic_v2_commutation():
    """
    Verifies horizontal symmetry mirror_dynamic on 35-channel tensor (v2).
    """
    g = TraversalGame(terrain_mode="random", seed=123)
    x_dyn_v2 = encode_dynamic_tensor(g, version="v2")
    assert x_dyn_v2.shape == (35, 10, 10)

    mirrored = mirror_dynamic(x_dyn_v2)
    assert mirrored.shape == (35, 10, 10)
    # Double mirror must be identity
    double_mirrored = mirror_dynamic(mirrored)
    assert np.allclose(x_dyn_v2, double_mirrored, atol=1e-6)
