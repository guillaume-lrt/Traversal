"""
Canonical State Tensor Encoder & Symmetry Augmentation for Traversal (Milestone 2).

Implements:
  - `encode_static_tensor(terrain, player)` -> (4, 10, 10) float32
  - `encode_dynamic_tensor(game)` -> (34, 10, 10) float32
  - `encode_game_state(game)` -> (X_static, X_dynamic, legal_mask)
  - Horizontal left-right reflection augmentation (`mirror_static`, `mirror_dynamic`,
    `mirror_policy`, `augment_horizontal`) for 2x sample efficiency (Section 6.2).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numba import njit

from traversal_env import (
    TraversalGame,
    ORTHO_DIRS,
    EXPLORER_SPECIAL_OFFSETS,
    ACTION_SPACE_SIZE,
    NUM_PIECES,
    NUM_TARGETS,
)

STATIC_CHANNELS = 4
DYNAMIC_CHANNELS_V1 = 34
DYNAMIC_CHANNELS_V2 = 35
DYNAMIC_CHANNELS = 34  # Default legacy channel count

# Permutation swapping pairs of identical pieces: [E0<->E1, V0<->V1, S0<->S1]
PIECE_SWAP_PERM = np.array([1, 0, 3, 2, 5, 4], dtype=np.int64)

# Precompute dynamic channel permutations under horizontal reflection
_DYN_CHANNEL_PERM_V1 = np.arange(DYNAMIC_CHANNELS_V1, dtype=np.int64)
for _base in (0, 6, 12, 18):
    _DYN_CHANNEL_PERM_V1[_base : _base + 6] = _base + PIECE_SWAP_PERM

_DYN_CHANNEL_PERM_V2 = np.arange(DYNAMIC_CHANNELS_V2, dtype=np.int64)
for _base in (0, 6, 12, 18):
    _DYN_CHANNEL_PERM_V2[_base : _base + 6] = _base + PIECE_SWAP_PERM

_DYN_CHANNEL_PERM = _DYN_CHANNEL_PERM_V1

# Precompute 606-action permutation under horizontal reflection
_ACTION_MIRROR_PERM = np.empty(ACTION_SPACE_SIZE, dtype=np.int64)
for _k in range(NUM_PIECES):
    _swapped_k = int(PIECE_SWAP_PERM[_k])
    for _r in range(10):
        for _c in range(10):
            _old_t = _r * 10 + _c
            _new_t = _r * 10 + (9 - _c)
            _ACTION_MIRROR_PERM[_k * NUM_TARGETS + _old_t] = _swapped_k * NUM_TARGETS + _new_t
    _ACTION_MIRROR_PERM[_k * NUM_TARGETS + 100] = _swapped_k * NUM_TARGETS + 100


# ---------------------------------------------------------------------------
# Numba-Accelerated Whole-Board Dijkstra Kernels for Static & Dynamic Channels
# ---------------------------------------------------------------------------


@njit(cache=True)
def compute_unblocked_dist_std(terrain: np.ndarray, target_row: int) -> np.ndarray:
    """
    Computes empty-board Dijkstra minimum movement cost from every (r, c)
    to step off `target_row` (0 or 9) using standard orthogonal movement.
    Stepping off from `target_row` costs 1 movement point.
    """
    dist = np.full((10, 10), 999, dtype=np.int32)
    visited = np.zeros((10, 10), dtype=np.uint8)

    for c in range(10):
        dist[target_row, c] = 1

    for _ in range(100):
        # Find unvisited node with smallest dist
        best_d = 999
        br = -1
        bc = -1
        for r in range(10):
            for c in range(10):
                if visited[r, c] == 0 and dist[r, c] < best_d:
                    best_d = dist[r, c]
                    br = r
                    bc = c
        if br == -1:
            break
        visited[br, bc] = 1

        for d in range(4):
            nr = br + ORTHO_DIRS[d, 0]
            nc = bc + ORTHO_DIRS[d, 1]
            if 0 <= nr < 10 and 0 <= nc < 10 and visited[nr, nc] == 0:
                step_cost = abs(terrain[br, bc] - terrain[nr, nc]) + 1
                nd = best_d + step_cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd

    return dist.astype(np.float32)


@njit(cache=True)
def compute_unblocked_dist_exp(terrain: np.ndarray, target_row: int) -> np.ndarray:
    """
    Computes empty-board minimum turns from every (r, c) to step off `target_row`
    using Explorer movement (M=5 standard movement OR 1-square diagonal step OR
    2-square orthogonal leap per turn).
    """
    # Standard distance to step off target_row
    std_exit_dist = compute_unblocked_dist_std(terrain, target_row)

    turns = np.full((10, 10), 99, dtype=np.int32)
    queue_r = np.zeros(100, dtype=np.int32)
    queue_c = np.zeros(100, dtype=np.int32)
    q_head = 0
    q_tail = 0

    # Any square on target_row OR with std_exit_dist <= 5 (reaches target_row with <= 4 + 1 exit)
    # can finish in 1 turn
    for r in range(10):
        for c in range(10):
            if r == target_row or std_exit_dist[r, c] <= 5.0:
                turns[r, c] = 1
                queue_r[q_tail] = r
                queue_c[q_tail] = c
                q_tail += 1

    # Precompute single-turn standard reachability (cost <= 5) between all pairs on empty board
    # Since step_cost is symmetric, a square (sr, sc) can reach (r, c) in 1 turn iff
    # standard distance <= 5 OR (sr, sc) is a special jump/diagonal neighbor of (r, c).
    bucket_r = np.zeros((6, 100), dtype=np.int32)
    bucket_c = np.zeros((6, 100), dtype=np.int32)
    bucket_len = np.zeros(6, dtype=np.int32)
    d_local = np.full((10, 10), 999, dtype=np.int32)

    while q_head < q_tail:
        r = queue_r[q_head]
        c = queue_c[q_head]
        q_head += 1
        cur_t = turns[r, c]

        # 1. Special Explorer jumps/diagonals into (r, c)
        for s in range(8):
            sr = r - EXPLORER_SPECIAL_OFFSETS[s, 0]
            sc = c - EXPLORER_SPECIAL_OFFSETS[s, 1]
            if 0 <= sr < 10 and 0 <= sc < 10:
                if turns[sr, sc] > cur_t + 1:
                    turns[sr, sc] = cur_t + 1
                    queue_r[q_tail] = sr
                    queue_c[q_tail] = sc
                    q_tail += 1

        # 2. Standard orthogonal movement (cost <= 5) into (r, c)
        for rr in range(10):
            for cc in range(10):
                d_local[rr, cc] = 999
        for b in range(6):
            bucket_len[b] = 0

        d_local[r, c] = 0
        bucket_r[0, 0] = r
        bucket_c[0, 0] = c
        bucket_len[0] = 1

        for cost in range(5):
            cnt = bucket_len[cost]
            for idx in range(cnt):
                cr = bucket_r[cost, idx]
                cc = bucket_c[cost, idx]
                if d_local[cr, cc] < cost:
                    continue
                for d in range(4):
                    nr = cr + ORTHO_DIRS[d, 0]
                    nc = cc + ORTHO_DIRS[d, 1]
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        scost = abs(terrain[nr, nc] - terrain[cr, cc]) + 1
                        ncost = cost + scost
                        if ncost <= 5 and ncost < d_local[nr, nc]:
                            d_local[nr, nc] = ncost
                            b_idx = bucket_len[ncost]
                            bucket_r[ncost, b_idx] = nr
                            bucket_c[ncost, b_idx] = nc
                            bucket_len[ncost] = b_idx + 1
                            if turns[nr, nc] > cur_t + 1:
                                turns[nr, nc] = cur_t + 1
                                queue_r[q_tail] = nr
                                queue_c[q_tail] = nc
                                q_tail += 1

    return turns.astype(np.float32)


@njit(cache=True)
def compute_blocked_dist_std(
    terrain: np.ndarray,
    my_pos: np.ndarray,
    opp_pos: np.ndarray,
    target_row: int,
) -> np.ndarray:
    """
    Computes dynamic Dijkstra minimum movement cost from every (r, c) to step off
    `target_row` treating all currently occupied squares as impassable walls,
    clamped to 30.0.
    """
    occ = np.zeros((10, 10), dtype=np.uint8)
    for i in range(6):
        r, c = my_pos[i, 0], my_pos[i, 1]
        if r >= 0 and c >= 0:
            occ[r, c] = 1
        r2, c2 = opp_pos[i, 0], opp_pos[i, 1]
        if r2 >= 0 and c2 >= 0:
            occ[r2, c2] = 1

    dist = np.full((10, 10), 30, dtype=np.int32)
    visited = np.zeros((10, 10), dtype=np.uint8)

    for c in range(10):
        if occ[target_row, c] == 0:
            dist[target_row, c] = 1

    for _ in range(100):
        best_d = 30
        br = -1
        bc = -1
        for r in range(10):
            for c in range(10):
                # Only expand backwards from unoccupied squares
                if visited[r, c] == 0 and occ[r, c] == 0 and dist[r, c] < best_d:
                    best_d = dist[r, c]
                    br = r
                    bc = c
        if br == -1:
            break
        visited[br, bc] = 1

        for d in range(4):
            nr = br + ORTHO_DIRS[d, 0]
            nc = bc + ORTHO_DIRS[d, 1]
            if 0 <= nr < 10 and 0 <= nc < 10 and occ[nr, nc] == 0 and visited[nr, nc] == 0:
                step_cost = abs(terrain[br, bc] - terrain[nr, nc]) + 1
                nd = best_d + step_cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd

    return dist.astype(np.float32)


@njit(cache=True)
def compute_blocked_dist_std_v2(
    terrain: np.ndarray,
    my_pos: np.ndarray,
    opp_pos: np.ndarray,
    target_row: int,
) -> np.ndarray:
    """
    Version 2 Dijkstra minimum movement cost from every (r, c) to step off target_row.
    Occupied squares cannot be traversed through, but can take a distance from adjacent
    unoccupied paths (review #6). Clamped to 30.0.
    """
    occ = np.zeros((10, 10), dtype=np.uint8)
    for i in range(6):
        r, c = my_pos[i, 0], my_pos[i, 1]
        if r >= 0 and c >= 0:
            occ[r, c] = 1
        r2, c2 = opp_pos[i, 0], opp_pos[i, 1]
        if r2 >= 0 and c2 >= 0:
            occ[r2, c2] = 1

    dist = np.full((10, 10), 30, dtype=np.int32)
    visited = np.zeros((10, 10), dtype=np.uint8)

    # Any square on target_row costs 1 to step off
    for c in range(10):
        dist[target_row, c] = 1

    for _ in range(100):
        best_d = 30
        br = -1
        bc = -1
        for r in range(10):
            for c in range(10):
                # Expand backwards ONLY from unvisited, UNOCCUPIED squares (paths cannot pass through occupied squares)
                if visited[r, c] == 0 and occ[r, c] == 0 and dist[r, c] < best_d:
                    best_d = dist[r, c]
                    br = r
                    bc = c
        if br == -1:
            break
        visited[br, bc] = 1

        for d in range(4):
            nr = br + ORTHO_DIRS[d, 0]
            nc = bc + ORTHO_DIRS[d, 1]
            if 0 <= nr < 10 and 0 <= nc < 10 and visited[nr, nc] == 0:
                step_cost = abs(terrain[br, bc] - terrain[nr, nc]) + 1
                nd = best_d + step_cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd

    return dist.astype(np.float32)


# ---------------------------------------------------------------------------
# Tensor Encoding Functions (Section 3.1 & 3.2)
# ---------------------------------------------------------------------------


def encode_static_tensor(terrain: np.ndarray, player: int = 0) -> np.ndarray:
    """
    Computes Block 1 (`X_static` of shape (4, 10, 10), dtype=float32) from `player`'s
    canonical perspective (where `player` moves from Row 0 toward Row 9):
      - Static 0: (h - median(h)) / 5.0
      - Static 1: my_unblocked_dist_std / 30.0 (to Row 9)
      - Static 2: my_unblocked_dist_exp / 10.0 (turns to Row 9)
      - Static 3: opp_unblocked_dist_std / 30.0 (to Row 0)
    """
    h_canon = terrain if player == 0 else np.ascontiguousarray(terrain[::-1, ::-1])
    x_static = np.empty((STATIC_CHANNELS, 10, 10), dtype=np.float32)

    median_h = float(np.median(h_canon))
    x_static[0] = (h_canon.astype(np.float32) - median_h) / 5.0
    x_static[1] = compute_unblocked_dist_std(h_canon, 9) / 30.0
    x_static[2] = compute_unblocked_dist_exp(h_canon, 9) / 10.0
    x_static[3] = compute_unblocked_dist_std(h_canon, 0) / 30.0

    return x_static


def encode_both_static_tensors(terrain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper returning `(X_static_p0, X_static_p1)` at Move 0.
    """
    return encode_static_tensor(terrain, player=0), encode_static_tensor(terrain, player=1)


def encode_dynamic_tensor(game: TraversalGame, version: str = "v1") -> np.ndarray:
    """
    Computes Block 2 (`X_dynamic` of shape (C, 10, 10), dtype=float32) from
    `game.current_player`'s canonical perspective:
      - 'v1': 34 channels (legacy)
      - 'v2': 35 channels with blocked-distance fix (review #6), attack plane cleanup
        on friendly squares (review #5.4), and repetition plane Dyn 34 (review #10).
    """
    norm_ver = version.lower().strip() if version else "v1"
    is_v2 = (norm_ver == "v2")
    channels = DYNAMIC_CHANNELS_V2 if is_v2 else DYNAMIC_CHANNELS_V1
    p = game.current_player
    opp = 1 - p
    x_dyn = np.zeros((channels, 10, 10), dtype=np.float32)

    my_legal_dest, my_can_finish, _, my_max_atk, my_def, opp_def = game.get_player_combat_and_reach(p)
    opp_legal_dest, opp_can_finish, _, opp_max_atk, _, _ = game.get_player_combat_and_reach(opp)

    if is_v2:
        # Review #5.4: Do not write attack values onto friendly-occupied squares
        for k in range(6):
            r, c = int(game.pos[p, k, 0]), int(game.pos[p, k, 1])
            if r >= 0 and c >= 0:
                my_max_atk[r, c] = 0.0
            r2, c2 = int(game.pos[opp, k, 0]), int(game.pos[opp, k, 1])
            if r2 >= 0 and c2 >= 0:
                opp_max_atk[r2, c2] = 0.0

    # Dyn 0..5 (my_pos) & Dyn 12..17 (my_reach with origin exit flag)
    for k in range(6):
        r, c = int(game.pos[p, k, 0]), int(game.pos[p, k, 1])
        if r >= 0 and c >= 0:
            rc, cc = (r, c) if p == 0 else (9 - r, 9 - c)
            x_dyn[k, rc, cc] = 1.0

            reach_grid = my_legal_dest[k] if p == 0 else my_legal_dest[k, ::-1, ::-1]
            x_dyn[12 + k] = reach_grid.astype(np.float32)
            # Origin exit flag (Section 3.2): set (r0, c0) to 1.0 iff piece k can finish this turn
            if my_can_finish[k]:
                x_dyn[12 + k, rc, cc] = 1.0

    # Dyn 6..11 (opp_pos) & Dyn 18..23 (opp_reach with origin exit flag)
    for k in range(6):
        r, c = int(game.pos[opp, k, 0]), int(game.pos[opp, k, 1])
        if r >= 0 and c >= 0:
            rc, cc = (r, c) if p == 0 else (9 - r, 9 - c)
            x_dyn[6 + k, rc, cc] = 1.0

            opp_grid = opp_legal_dest[k] if p == 0 else opp_legal_dest[k, ::-1, ::-1]
            x_dyn[18 + k] = opp_grid.astype(np.float32)
            if opp_can_finish[k]:
                x_dyn[18 + k, rc, cc] = 1.0

    # Dyn 24..27: Combat projection & defense planes
    if p == 0:
        x_dyn[24] = my_max_atk / 10.0
        x_dyn[25] = my_def / 7.0
        x_dyn[26] = opp_max_atk / 10.0
        x_dyn[27] = opp_def / 7.0
    else:
        x_dyn[24] = my_max_atk[::-1, ::-1] / 10.0
        x_dyn[25] = my_def[::-1, ::-1] / 7.0
        x_dyn[26] = opp_max_atk[::-1, ::-1] / 10.0
        x_dyn[27] = opp_def[::-1, ::-1] / 7.0

    # Dyn 28..29: Dynamic blocked Dijkstra distance maps
    my_target_row = 9 if p == 0 else 0
    opp_target_row = 0 if p == 0 else 9
    if is_v2:
        my_blk = compute_blocked_dist_std_v2(game.terrain, game.pos[p], game.pos[opp], my_target_row)
        opp_blk = compute_blocked_dist_std_v2(game.terrain, game.pos[p], game.pos[opp], opp_target_row)
    else:
        my_blk = compute_blocked_dist_std(game.terrain, game.pos[p], game.pos[opp], my_target_row)
        opp_blk = compute_blocked_dist_std(game.terrain, game.pos[p], game.pos[opp], opp_target_row)

    if p == 0:
        x_dyn[28] = my_blk / 30.0
        x_dyn[29] = opp_blk / 30.0
    else:
        x_dyn[28] = my_blk[::-1, ::-1] / 30.0
        x_dyn[29] = opp_blk[::-1, ::-1] / 30.0

    # Dyn 30..33: Scalar planes
    x_dyn[30].fill(float(game.scores[p]) / 6.0)
    x_dyn[31].fill(float(game.scores[opp]) / 6.0)
    x_dyn[32].fill(float(game.total_pieces_finished) / 2.0)
    x_dyn[33].fill(float(game.ply_count) / 200.0)

    # Dyn 34 (v2 only): Repetition plane (review #10)
    if is_v2:
        pos_key = game.get_position_key()
        rep_count = game.position_counts.get(pos_key, 1)
        x_dyn[34].fill(1.0 if rep_count >= 2 else 0.0)

    return x_dyn


def encode_game_state(game: TraversalGame, version: str = "v1") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns `(X_static, X_dynamic, legal_mask)` for `game.current_player`.
    """
    x_static = encode_static_tensor(game.terrain, player=game.current_player)
    x_dynamic = encode_dynamic_tensor(game, version=version)
    legal_mask = game.get_legal_mask()
    return x_static, x_dynamic, legal_mask


# ---------------------------------------------------------------------------
# Horizontal Symmetry Augmentation (Section 6.2)
# ---------------------------------------------------------------------------


def mirror_static(x_static: np.ndarray) -> np.ndarray:
    """Horizontally mirrors `X_static` of shape (..., 4, 10, 10) across columns c <-> 9 - c."""
    return np.ascontiguousarray(x_static[..., :, ::-1])


def mirror_dynamic(x_dynamic: np.ndarray) -> np.ndarray:
    """
    Horizontally mirrors `X_dynamic` of shape (34, 10, 10), (35, 10, 10), or (B, C, 10, 10):
      1. Flips columns c <-> 9 - c
      2. Swaps paired identical piece channels E0<->E1, V0<->V1, S0<->S1
         across my_pos (0..5), opp_pos (6..11), my_reach (12..17), and opp_reach (18..23).
    """
    c_dim = x_dynamic.shape[-3]
    perm = _DYN_CHANNEL_PERM_V2 if c_dim == 35 else _DYN_CHANNEL_PERM_V1
    flipped = x_dynamic[..., perm, :, ::-1]
    return np.ascontiguousarray(flipped)


def mirror_policy(pi: np.ndarray) -> np.ndarray:
    """
    Horizontally mirrors a 606-element policy vector or boolean legal move mask
    (shape (606,) or (B, 606)), swapping piece pairs E0<->E1, V0<->V1, S0<->S1
    and flipping target columns c <-> 9 - c while preserving target 100 (finish).
    """
    return np.ascontiguousarray(pi[..., _ACTION_MIRROR_PERM])


def augment_horizontal(
    x_static: np.ndarray,
    x_dynamic: np.ndarray,
    pi_mcts: np.ndarray,
    z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns the horizontally mirrored `(X_static, X_dynamic, pi_mcts, z)` tuple."""
    return (
        mirror_static(x_static),
        mirror_dynamic(x_dynamic),
        mirror_policy(pi_mcts),
        float(z),
    )
