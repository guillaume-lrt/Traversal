"""
Headless Game Engine for Traversal (Milestone 1).

Implements the exact board, terrain generation, movement (Dijkstra + Explorer special moves),
combat (height modifier, Vanguard d_min=4 charge bonus, allied support with self-exclusion,
and defense), finishing rules, and game termination conditions as specified in
`traversal_alphazero_ai_architecture_specification.md`.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Constants & Canonical Piece Definitions
# ---------------------------------------------------------------------------

BOARD_SIZE = 10
NUM_PIECES = 6
NUM_TARGETS = 101  # 0..99 = canonical board squares (r*10 + c), 100 = finish off-board
ACTION_SPACE_SIZE = NUM_PIECES * NUM_TARGETS  # 606
FINISH_TARGET = 100
MAX_PLIES = 200

# Piece types
PIECE_EXPLORER = 0
PIECE_VANGUARD = 1
PIECE_SENTINEL = 2

PIECE_NAMES = (
    "Explorer_0",
    "Explorer_1",
    "Vanguard_0",
    "Vanguard_1",
    "Sentinel_0",
    "Sentinel_1",
)

# Canonical ordering: [Explorer_0, Explorer_1, Vanguard_0, Vanguard_1, Sentinel_0, Sentinel_1]
PIECE_TYPES = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
PIECE_MOVEMENT = np.array([5, 5, 4, 4, 3, 3], dtype=np.int32)
PIECE_STRENGTH = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
PIECE_SCORE = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)

# Piece status codes
STATUS_ACTIVE = 0
STATUS_CAPTURED = 1
STATUS_FINISHED = 2

# Starting positions in canonical view (Row 0, Columns 2..7):
# Col 2: S0, Col 3: V0, Col 4: E0, Col 5: E1, Col 6: V1, Col 7: S1
P1_START_POS = np.array(
    [
        [0, 4],  # Explorer_0
        [0, 5],  # Explorer_1
        [0, 3],  # Vanguard_0
        [0, 6],  # Vanguard_1
        [0, 2],  # Sentinel_0
        [0, 7],  # Sentinel_1
    ],
    dtype=np.int32,
)

# Player 2 starts on Row 9, Columns 2..7 (rotated 180 degrees from P1's canonical layout
# so that in P2's 180-degree rotated canonical perspective, P2's S0 is also at (0, 2)):
P2_START_POS = np.array(
    [
        [9, 5],  # Explorer_0 (canonical (0, 4) for P2)
        [9, 4],  # Explorer_1 (canonical (0, 5) for P2)
        [9, 6],  # Vanguard_0 (canonical (0, 3) for P2)
        [9, 3],  # Vanguard_1 (canonical (0, 6) for P2)
        [9, 7],  # Sentinel_0 (canonical (0, 2) for P2)
        [9, 2],  # Sentinel_1 (canonical (0, 7) for P2)
    ],
    dtype=np.int32,
)

TERRAIN_MODES = ("random", "mountains", "valleys", "highlands", "woodlands", "desert", "flat")

# Orthogonal directions: Up, Down, Left, Right
ORTHO_DIRS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

# Explorer special move offsets: 4 diagonal steps + 4 orthogonal 2-square leaps
EXPLORER_SPECIAL_OFFSETS = np.array(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
        [-2, 0],
        [2, 0],
        [0, -2],
        [0, 2],
    ],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Numba-Accelerated Terrain & Movement / Combat Kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def enforce_slope_constraint(heights: np.ndarray) -> np.ndarray:
    """
    Projects an integer 10x10 height grid onto the feasible set where:
      1. h(r, c) >= 0 for all (r, c)
      2. |h(r1, c1) - h(r2, c2)| <= 2 for all orthogonally adjacent pairs.
    """
    h = heights.copy()
    for r in range(10):
        for c in range(10):
            if h[r, c] < 0:
                h[r, c] = 0

    changed = True
    while changed:
        changed = False
        for r in range(10):
            for c in range(10):
                val = h[r, c]
                if r > 0 and h[r - 1, c] + 2 < val:
                    val = h[r - 1, c] + 2
                if r < 9 and h[r + 1, c] + 2 < val:
                    val = h[r + 1, c] + 2
                if c > 0 and h[r, c - 1] + 2 < val:
                    val = h[r, c - 1] + 2
                if c < 9 and h[r, c + 1] + 2 < val:
                    val = h[r, c + 1] + 2
                if val < h[r, c]:
                    h[r, c] = val
                    changed = True
    return h


@njit(cache=True)
def compute_defense_grid(my_pos: np.ndarray) -> np.ndarray:
    """
    Computes effective defense for every square occupied by a piece in `my_pos`:
      Defense(r_t, c_t) = S_defender + sum_{friendly k adj. to (r_t, c_t)} 1
    Returns a (10, 10) int32 array (0 on squares not occupied by `my_pos`).
    """
    def_grid = np.zeros((10, 10), dtype=np.int32)
    occ_Friendly = np.zeros((10, 10), dtype=np.int32)

    for i in range(6):
        r = my_pos[i, 0]
        c = my_pos[i, 1]
        if r >= 0 and c >= 0:
            occ_Friendly[r, c] = 1

    for i in range(6):
        r = my_pos[i, 0]
        c = my_pos[i, 1]
        if r >= 0 and c >= 0:
            adj_count = 0
            if r > 0 and occ_Friendly[r - 1, c] == 1:
                adj_count += 1
            if r < 9 and occ_Friendly[r + 1, c] == 1:
                adj_count += 1
            if c > 0 and occ_Friendly[r, c - 1] == 1:
                adj_count += 1
            if c < 9 and occ_Friendly[r, c + 1] == 1:
                adj_count += 1
            def_grid[r, c] = PIECE_STRENGTH[i] + adj_count

    return def_grid


@njit(cache=True)
def compute_player_reach_and_combat(
    terrain: np.ndarray,
    my_pos: np.ndarray,
    opp_pos: np.ndarray,
    target_row: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes for the given player (`my_pos` moving toward `target_row` in {0, 9}):
      - legal_dest: (6, 10, 10) uint8 (1 if piece i can legally move to or capture (r, c))
      - can_finish: (6,) uint8 (1 if piece i can legally step off `target_row` this turn)
      - d_min_all: (6, 10, 10) int32 (shortest standard orthogonal movement cost, 999 if unreachable)
      - my_max_attack: (10, 10) float32 (max effective attack across all friendly pieces i that
        can reach (r, c) this turn, unnormalized)
      - my_defense: (10, 10) float32 (effective defense on friendly-occupied tiles, unnormalized)
      - opp_defense: (10, 10) float32 (effective defense on opponent-occupied tiles, unnormalized)
    """
    # Occupancy grid: 0 = empty, 1 = friendly, 2 = opponent
    occ = np.zeros((10, 10), dtype=np.int32)
    friendly_strength_at = np.zeros((10, 10), dtype=np.int32)
    friendly_idx_at = np.full((10, 10), -1, dtype=np.int32)

    for i in range(6):
        r = my_pos[i, 0]
        c = my_pos[i, 1]
        if r >= 0 and c >= 0:
            occ[r, c] = 1
            friendly_strength_at[r, c] = PIECE_STRENGTH[i]
            friendly_idx_at[r, c] = i

    for i in range(6):
        r = opp_pos[i, 0]
        c = opp_pos[i, 1]
        if r >= 0 and c >= 0:
            occ[r, c] = 2

    my_def_int = compute_defense_grid(my_pos)
    opp_def_int = compute_defense_grid(opp_pos)

    legal_dest = np.zeros((6, 10, 10), dtype=np.uint8)
    can_finish = np.zeros(6, dtype=np.uint8)
    d_min_all = np.full((6, 10, 10), 999, dtype=np.int32)
    my_max_attack = np.zeros((10, 10), dtype=np.float32)

    # Dial's algorithm buckets for costs 0..5 (max movement is 5)
    bucket_r = np.zeros((6, 100), dtype=np.int32)
    bucket_c = np.zeros((6, 100), dtype=np.int32)
    bucket_len = np.zeros(6, dtype=np.int32)

    for i in range(6):
        r0 = my_pos[i, 0]
        c0 = my_pos[i, 1]
        if r0 < 0 or c0 < 0:
            continue

        move_budget = PIECE_MOVEMENT[i]
        base_str = PIECE_STRENGTH[i]
        h0 = terrain[r0, c0]

        # Reset buckets
        for b in range(6):
            bucket_len[b] = 0

        d_min_all[i, r0, c0] = 0
        bucket_r[0, 0] = r0
        bucket_c[0, 0] = c0
        bucket_len[0] = 1

        # Standard orthogonal Dijkstra up to move_budget
        for cost in range(move_budget):
            count = bucket_len[cost]
            for idx in range(count):
                r = bucket_r[cost, idx]
                c = bucket_c[cost, idx]

                if d_min_all[i, r, c] < cost:
                    continue

                # Cannot move through any occupied square (except starting square r0, c0)
                if not (r == r0 and c == c0) and occ[r, c] != 0:
                    continue

                for d in range(4):
                    nr = r + ORTHO_DIRS[d, 0]
                    nc = c + ORTHO_DIRS[d, 1]
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        step_cost = abs(terrain[nr, nc] - terrain[r, c]) + 1
                        new_cost = cost + step_cost
                        if new_cost <= move_budget and new_cost < d_min_all[i, nr, nc]:
                            d_min_all[i, nr, nc] = new_cost
                            b_idx = bucket_len[new_cost]
                            bucket_r[new_cost, b_idx] = nr
                            bucket_c[new_cost, b_idx] = nc
                            bucket_len[new_cost] = b_idx + 1

        # Check finishing condition (Section 2.5):
        # Starts on target_row OR can reach any empty square on target_row with d_min <= M_i - 1
        if r0 == target_row:
            can_finish[i] = 1
        else:
            max_finish_prefix = move_budget - 1
            for c_end in range(10):
                if occ[target_row, c_end] == 0 and d_min_all[i, target_row, c_end] <= max_finish_prefix:
                    can_finish[i] = 1
                    break

        # Determine all reachable candidate squares (excluding friendly-occupied squares & origin)
        # and evaluate combat / attack values
        for rt in range(10):
            for ct in range(10):
                if rt == r0 and ct == c0:
                    continue

                d_val = d_min_all[i, rt, ct]
                reachable = d_val <= move_budget
                is_special_move = False

                # Explorer special moves from (r0, c0)
                if not reachable and PIECE_TYPES[i] == PIECE_EXPLORER:
                    dr = abs(rt - r0)
                    dc = abs(ct - c0)
                    if (dr == 1 and dc == 1) or (dr == 2 and dc == 0) or (dr == 0 and dc == 2):
                        reachable = True
                        is_special_move = True

                if not reachable:
                    continue

                # Explorer special moves (1-square diagonal step and 2-square orthogonal leap)
                # are strictly standalone movement abilities to empty tiles;
                # they cannot land on an enemy piece to capture it. All captures must be
                # reachable via standard orthogonal movement steps.
                if is_special_move and occ[rt, ct] != 0:
                    continue

                # Compute effective Attack_i(rt, ct) (Section 2.4)
                ht = terrain[rt, ct]
                if h0 > ht:
                    h_mod = 1
                elif h0 < ht:
                    h_mod = -1
                else:
                    h_mod = 0

                # Vanguard charge bonus: +1 iff piece is Vanguard and shortest valid movement cost d_min == 4
                v_bonus = 1 if (PIECE_TYPES[i] == PIECE_VANGUARD and d_val == 4) else 0

                # Allied support: sum of S_j for friendly j != i orthogonally adjacent to (rt, ct)
                ally_support = 0
                if rt > 0 and friendly_idx_at[rt - 1, ct] != -1 and friendly_idx_at[rt - 1, ct] != i:
                    ally_support += friendly_strength_at[rt - 1, ct]
                if rt < 9 and friendly_idx_at[rt + 1, ct] != -1 and friendly_idx_at[rt + 1, ct] != i:
                    ally_support += friendly_strength_at[rt + 1, ct]
                if ct > 0 and friendly_idx_at[rt, ct - 1] != -1 and friendly_idx_at[rt, ct - 1] != i:
                    ally_support += friendly_strength_at[rt, ct - 1]
                if ct < 9 and friendly_idx_at[rt, ct + 1] != -1 and friendly_idx_at[rt, ct + 1] != i:
                    ally_support += friendly_strength_at[rt, ct + 1]

                eff_attack = base_str + h_mod + v_bonus + ally_support
                if not is_special_move and eff_attack > my_max_attack[rt, ct]:
                    my_max_attack[rt, ct] = float(eff_attack)

                if occ[rt, ct] == 0:
                    legal_dest[i, rt, ct] = 1
                elif occ[rt, ct] == 2 and not is_special_move:
                    # Enemy square: legal capture iff Attack_i(rt, ct) >= Defense(rt, ct)
                    if eff_attack >= opp_def_int[rt, ct]:
                        legal_dest[i, rt, ct] = 1

    return (
        legal_dest,
        can_finish,
        d_min_all,
        my_max_attack,
        my_def_int.astype(np.float32),
        opp_def_int.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Procedural Terrain Generation (Section 2.1)
# ---------------------------------------------------------------------------


def generate_terrain(
    mode: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generates a 10x10 integer terrain grid satisfying:
      - h(r, c) >= 0
      - |h(r1, c1) - h(r2, c2)| <= 2 for all orthogonally adjacent squares
      - Structural fairness between Player 1 (Row 0 -> 9) and Player 2 (Row 9 -> 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    mode = mode.lower()
    if mode not in TERRAIN_MODES:
        raise ValueError(f"Unknown terrain mode '{mode}'. Choose from {TERRAIN_MODES}.")

    if mode == "flat":
        return np.zeros((10, 10), dtype=np.int32)

    if mode == "random":
        chosen_mode = rng.choice(["mountains", "valleys", "highlands", "woodlands", "desert"])
        return generate_terrain(chosen_mode, rng=rng)

    grid_r, grid_c = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    raw = np.zeros((10, 10), dtype=np.float64)

    if mode == "mountains":
        # High central peaks sloping down toward edges
        num_peaks = int(rng.integers(1, 3))
        for _ in range(num_peaks):
            pr = rng.uniform(2.5, 6.5)
            pc = rng.uniform(2.0, 7.0)
            peak_h = rng.uniform(6.0, 9.0)
            dist = np.abs(grid_r - pr) + np.abs(grid_c - pc)
            raw = np.maximum(raw, peak_h - 1.3 * dist)
        raw += rng.normal(0.0, 0.6, size=(10, 10))

    elif mode == "valleys":
        # High flanks / outer edges with a low central valley corridor
        center_c = rng.uniform(3.5, 5.5)
        dist_c = np.abs(grid_c - center_c)
        raw = 1.2 * dist_c + rng.uniform(0.0, 2.0, size=(10, 10))

    elif mode == "highlands":
        # Elevated undulating plateaus (heights mostly 3..7)
        base = rng.uniform(3.5, 5.5)
        raw = base + rng.normal(0.0, 1.5, size=(10, 10))
        # Add a gentle ridge
        rc = rng.uniform(3.0, 6.0)
        raw += 2.0 * np.exp(-0.15 * ((grid_r - 4.5) ** 2 + (grid_c - rc) ** 2))

    elif mode == "woodlands":
        # Gentle rolling hills (heights mostly 0..4)
        raw = rng.uniform(0.5, 3.5, size=(10, 10))
        # Smooth locally with simple 5-point average
        padded = np.pad(raw, 1, mode="edge")
        raw = (
            padded[1:-1, 1:-1]
            + 0.5 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        ) / 3.0
        raw += rng.uniform(-0.8, 0.8, size=(10, 10))

    elif mode == "desert":
        # Mostly flat low terrain (0..2) with 1-2 small dunes
        raw = rng.choice([0.0, 0.0, 1.0, 1.0, 2.0], size=(10, 10))
        dr = rng.uniform(3.0, 6.0)
        dc = rng.uniform(2.0, 7.0)
        dist = np.abs(grid_r - dr) + np.abs(grid_c - dc)
        raw = np.maximum(raw, 3.5 - 1.2 * dist)

    # Balance Player 1 and Player 2 structurally by blending with 180-degree rotated profile
    # plus a small asymmetric perturbation so boards aren't strictly identical halves
    blend_weight = rng.uniform(0.35, 0.5)
    sym_raw = (1.0 - blend_weight) * raw + blend_weight * raw[::-1, ::-1]
    heights = np.rint(np.clip(sym_raw, 0.0, 12.0)).astype(np.int32)

    # Guarantee exact |h(r1, c1) - h(r2, c2)| <= 2 slope constraint everywhere
    heights = enforce_slope_constraint(heights)
    return heights


# ---------------------------------------------------------------------------
# Position Key Computation for Repetition Detection (Numba Accelerated)
# ---------------------------------------------------------------------------


@njit(cache=True)
def compute_position_key(
    pos: np.ndarray,
    piece_status: np.ndarray,
    current_player: int,
) -> Tuple[np.int64, np.int64]:
    """
    Computes a compact 64-bit integer pair (v0, v1) uniquely identifying the board position
    and player to move, invariant under permutations of identical friendly pieces
    (Explorers, Vanguards, Sentinels):
      - v0: current_player (1 bit) + 6 piece codes for Player 0 (6 * 7 = 42 bits)
      - v1: 6 piece codes for Player 1 (6 * 7 = 42 bits)
    Piece code in 0..101:
      - 0..99: square (r * 10 + c) on board
      - 100: captured / off-board
      - 101: finished off-board
    """
    v0 = np.int64(current_player)
    v1 = np.int64(0)

    # Player 0 pieces: 3 pairs (0,1), (2,3), (4,5)
    for pair in range(3):
        k0 = pair * 2
        k1 = k0 + 1
        s0 = piece_status[0, k0]
        if s0 == 2:
            c0 = 101
        elif s0 == 1 or pos[0, k0, 0] < 0:
            c0 = 100
        else:
            c0 = pos[0, k0, 0] * 10 + pos[0, k0, 1]

        s1 = piece_status[0, k1]
        if s1 == 2:
            c1 = 101
        elif s1 == 1 or pos[0, k1, 0] < 0:
            c1 = 100
        else:
            c1 = pos[0, k1, 0] * 10 + pos[0, k1, 1]

        if c0 > c1:
            c0, c1 = c1, c0
        v0 = (v0 << 7) | c0
        v0 = (v0 << 7) | c1

    # Player 1 pieces: 3 pairs (0,1), (2,3), (4,5)
    for pair in range(3):
        k0 = pair * 2
        k1 = k0 + 1
        s0 = piece_status[1, k0]
        if s0 == 2:
            c0 = 101
        elif s0 == 1 or pos[1, k0, 0] < 0:
            c0 = 100
        else:
            c0 = pos[1, k0, 0] * 10 + pos[1, k0, 1]

        s1 = piece_status[1, k1]
        if s1 == 2:
            c1 = 101
        elif s1 == 1 or pos[1, k1, 0] < 0:
            c1 = 100
        else:
            c1 = pos[1, k1, 0] * 10 + pos[1, k1, 1]

        if c0 > c1:
            c0, c1 = c1, c0
        v1 = (v1 << 7) | c0
        v1 = (v1 << 7) | c1

    return (v0, v1)


# ---------------------------------------------------------------------------
# Headless TraversalGame Class
# ---------------------------------------------------------------------------


class TraversalGame:
    """
    Pure Python / NumPy / Numba headless engine for Traversal.

    Canonical Action Space (606 actions):
      - action in 0..605:
        piece_idx = action // 101 in 0..5 ([E0, E1, V0, V1, S0, S1])
        target_idx = action % 101 in 0..100:
          * 0..99: canonical destination square (r_canon, c_canon) = (target_idx // 10, target_idx % 10)
            where (r_phys, c_phys) = (r_canon, c_canon) for Player 0 (P1)
            and   (r_phys, c_phys) = (9 - r_canon, 9 - c_canon) for Player 1 (P2).
          * 100: piece steps off Row 9 (canonical) to finish and score.
    """

    def __init__(
        self,
        terrain_mode: str = "random",
        terrain: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        _skip_init: bool = False,
    ) -> None:
        if _skip_init:
            return

        self.rng = np.random.default_rng(seed)
        self.terrain_mode = terrain_mode

        if terrain is not None:
            t = np.asarray(terrain, dtype=np.int32)
            if t.shape != (10, 10):
                raise ValueError(f"Expected (10, 10) terrain, got {t.shape}")
            if np.any(t < 0):
                raise ValueError("Terrain heights must be >= 0")
            self.terrain = t.copy()
        else:
            self.terrain = generate_terrain(mode=terrain_mode, rng=self.rng)

        # Piece positions: shape (2, 6, 2), [-1, -1] when off-board (captured or finished)
        self.pos = np.empty((2, 6, 2), dtype=np.int32)
        self.pos[0] = P1_START_POS
        self.pos[1] = P2_START_POS

        # Piece status: shape (2, 6) in {STATUS_ACTIVE, STATUS_CAPTURED, STATUS_FINISHED}
        self.piece_status = np.zeros((2, 6), dtype=np.int8)

        # Game state scalars
        self.current_player: int = 0  # 0 = Player 1 (Row 0->9), 1 = Player 2 (Row 9->0)
        self.scores = np.zeros(2, dtype=np.int32)
        self.total_pieces_finished: int = 0
        self.ply_count: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None  # 0 = P1, 1 = P2, -1 = Draw, None = ongoing
        self.termination_reason: Optional[str] = None

        # Cached legal move mask for current state
        self._cached_legal_mask: Optional[np.ndarray] = None

        # Position repetition tracking for Threefold Repetition Draw
        self.position_counts: Dict[Tuple[int, int], int] = {}
        init_key = self.get_position_key()
        self.position_counts[init_key] = 1

    def get_position_key(self) -> Tuple[int, int]:
        """Returns the compact canonical position key (v0, v1) for the current board state and turn."""
        return compute_position_key(self.pos, self.piece_status, self.current_player)

    def reset_repetition_history(self) -> None:
        """Resets repetition history and records the current position with count 1."""
        self.position_counts = {self.get_position_key(): 1}

    def clone(self) -> "TraversalGame":
        """Fast state copy for MCTS simulations (shares immutable terrain by reference)."""
        new_game = TraversalGame(_skip_init=True)
        new_game.rng = self.rng
        new_game.terrain_mode = self.terrain_mode
        new_game.terrain = self.terrain  # Immutable during a match
        new_game.pos = self.pos.copy()
        new_game.piece_status = self.piece_status.copy()
        new_game.current_player = self.current_player
        new_game.scores = self.scores.copy()
        new_game.total_pieces_finished = self.total_pieces_finished
        new_game.ply_count = self.ply_count
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.termination_reason = self.termination_reason
        new_game._cached_legal_mask = (
            self._cached_legal_mask.copy() if self._cached_legal_mask is not None else None
        )
        new_game.position_counts = self.position_counts.copy()
        return new_game

    # -----------------------------------------------------------------------
    # Coordinate & Action Mapping Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def to_canonical_coords(r: int, c: int, player: int) -> Tuple[int, int]:
        """Converts physical board coordinates (r, c) to `player`'s canonical view."""
        if player == 0:
            return r, c
        return 9 - r, 9 - c

    @staticmethod
    def to_physical_coords(r_canon: int, c_canon: int, player: int) -> Tuple[int, int]:
        """Converts canonical coordinates (r_canon, c_canon) to physical board coordinates."""
        if player == 0:
            return r_canon, c_canon
        return 9 - r_canon, 9 - c_canon

    def encode_action(
        self,
        piece_idx: int,
        target_r: Optional[int] = None,
        target_c: Optional[int] = None,
        finish: bool = False,
        canonical: bool = False,
    ) -> int:
        """
        Encodes a move for `self.current_player` into a discrete action index in 0..605.
        By default, `target_r` and `target_c` are physical board coordinates unless `canonical=True`.
        """
        if not (0 <= piece_idx < 6):
            raise ValueError(f"piece_idx must be in 0..5, got {piece_idx}")
        if finish:
            return piece_idx * NUM_TARGETS + FINISH_TARGET
        if target_r is None or target_c is None:
            raise ValueError("target_r and target_c must be provided when finish=False")
        if not canonical:
            target_r, target_c = self.to_canonical_coords(target_r, target_c, self.current_player)
        return piece_idx * NUM_TARGETS + (target_r * 10 + target_c)

    def decode_action(
        self, action: int, canonical: bool = False
    ) -> Tuple[int, int, int, bool]:
        """
        Decodes an action index in 0..605 for `self.current_player` into
        `(piece_idx, r, c, is_finish)`.
        Returns physical `(r, c)` by default (or `(-1, -1)` if `is_finish=True`).
        """
        if not (0 <= action < ACTION_SPACE_SIZE):
            raise ValueError(f"Action must be in 0..{ACTION_SPACE_SIZE - 1}, got {action}")
        piece_idx = action // NUM_TARGETS
        t = action % NUM_TARGETS
        if t == FINISH_TARGET:
            return piece_idx, -1, -1, True
        r_canon = t // 10
        c_canon = t % 10
        if canonical:
            return piece_idx, r_canon, c_canon, False
        r_phys, c_phys = self.to_physical_coords(r_canon, c_canon, self.current_player)
        return piece_idx, r_phys, c_phys, False

    # -----------------------------------------------------------------------
    # Reachability, Combat & Legal Move Generation
    # -----------------------------------------------------------------------

    def get_player_combat_and_reach(
        self, player: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns `(legal_dest, can_finish, d_min_all, max_attack, my_def, opp_def)`
        in physical board coordinates for `player` (defaults to `self.current_player`).
        """
        if player is None:
            player = self.current_player
        opp = 1 - player
        target_row = 9 if player == 0 else 0
        return compute_player_reach_and_combat(
            self.terrain,
            self.pos[player],
            self.pos[opp],
            target_row,
        )

    def get_legal_mask(self) -> np.ndarray:
        """
        Returns the boolean legal move mask of shape (606,) in canonical perspective
        for `self.current_player`.
        """
        if self.done:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

        if self._cached_legal_mask is not None:
            return self._cached_legal_mask.copy()

        p = self.current_player
        legal_dest, can_finish, _, _, _, _ = self.get_player_combat_and_reach(p)

        mask_2d = np.zeros((6, 101), dtype=np.bool_)
        for k in range(6):
            if self.pos[p, k, 0] < 0:
                continue
            dest_grid = legal_dest[k]
            if p == 1:
                # Rotate 180 degrees to canonical coordinates for Player 2
                dest_grid = dest_grid[::-1, ::-1]
            mask_2d[k, :100] = dest_grid.reshape(100) > 0
            mask_2d[k, FINISH_TARGET] = bool(can_finish[k])

        flat_mask = mask_2d.reshape(ACTION_SPACE_SIZE)
        self._cached_legal_mask = flat_mask
        return flat_mask.copy()

    def get_legal_actions(self) -> np.ndarray:
        """Returns 1D array of legal action indices in 0..605."""
        return np.flatnonzero(self.get_legal_mask())

    # -----------------------------------------------------------------------
    # Step & Termination Logic (Section 2.3, 2.4, 2.5)
    # -----------------------------------------------------------------------

    def _resolve_score_or_progress_winner(self) -> int:
        """
        Determines the winner (0 = P1, 1 = P2, -1 = Draw) when the game ends via
        Elimination, No Legal Moves, or the 200-Ply Limit (Section 2.5):
          1. Higher score wins.
          2. If tied on score, the player whose on-board piece is closest to finishing
             (fewest rows remaining to cross) wins.
          3. If both players' closest pieces are equally close, the stronger piece (S_i) wins.
          4. If also equally strong, draw (-1).
        """
        if self.scores[0] > self.scores[1]:
            return 0
        if self.scores[1] > self.scores[0]:
            return 1

        min_rows = [999, 999]
        best_strength = [0, 0]

        for p in (0, 1):
            for k in range(6):
                r = int(self.pos[p, k, 0])
                if r < 0:
                    continue
                rows_rem = (9 - r) if p == 0 else r
                s_k = int(PIECE_STRENGTH[k])
                if rows_rem < min_rows[p]:
                    min_rows[p] = rows_rem
                    best_strength[p] = s_k
                elif rows_rem == min_rows[p] and s_k > best_strength[p]:
                    best_strength[p] = s_k

        if min_rows[0] < min_rows[1]:
            return 0
        if min_rows[1] < min_rows[0]:
            return 1
        if best_strength[0] > best_strength[1]:
            return 0
        if best_strength[1] > best_strength[0]:
            return 1
        return -1

    def step(self, action: int) -> Tuple[bool, float]:
        """
        Executes canonical `action` in 0..605 for `self.current_player`.
        Returns `(done, outcome_for_new_current_player)`.
        """
        if self.done:
            raise RuntimeError("Cannot call step() on a finished game.")

        legal_mask = self.get_legal_mask()
        if not (0 <= action < ACTION_SPACE_SIZE) or not legal_mask[action]:
            piece_idx, r_phys, c_phys, is_fin = self.decode_action(action)
            raise ValueError(
                f"Illegal action {action} (player={self.current_player}, "
                f"piece={PIECE_NAMES[piece_idx]}, dest=({r_phys}, {c_phys}), finish={is_fin})"
            )

        p = self.current_player
        opp = 1 - p
        piece_idx, dest_r, dest_c, is_finish = self.decode_action(action, canonical=False)

        # Ensure starting position is recorded if board was initialized/mutated externally
        if len(self.position_counts) == 0:
            self.position_counts[self.get_position_key()] = 1

        self.ply_count += 1
        self._cached_legal_mask = None

        captured_enemy = False
        if is_finish:
            # Remove piece from board and add score
            self.pos[p, piece_idx, 0] = -1
            self.pos[p, piece_idx, 1] = -1
            self.piece_status[p, piece_idx] = STATUS_FINISHED

            was_very_first_finish = self.total_pieces_finished == 0
            self.total_pieces_finished += 1
            self.scores[p] += PIECE_SCORE[piece_idx]

            # 1. Sentinel Instant Win: very first piece of the entire game to finish is a Sentinel
            if was_very_first_finish and PIECE_TYPES[piece_idx] == PIECE_SENTINEL:
                self.done = True
                self.winner = p
                self.termination_reason = "sentinel_instant_win"
            # 2. Two-Piece Finish: any 2 pieces total have finished
            elif self.total_pieces_finished >= 2:
                self.done = True
                self.termination_reason = "two_piece_finish"
                if self.scores[0] > self.scores[1]:
                    self.winner = 0
                elif self.scores[1] > self.scores[0]:
                    self.winner = 1
                else:
                    self.winner = -1
        else:
            # Check if destination square holds an enemy piece -> capture it
            for opp_k in range(6):
                if self.pos[opp, opp_k, 0] == dest_r and self.pos[opp, opp_k, 1] == dest_c:
                    self.pos[opp, opp_k, 0] = -1
                    self.pos[opp, opp_k, 1] = -1
                    self.piece_status[opp, opp_k] = STATUS_CAPTURED
                    captured_enemy = True
                    break

            # Move piece to destination square
            self.pos[p, piece_idx, 0] = dest_r
            self.pos[p, piece_idx, 1] = dest_c

        # Advance turn to next player
        self.current_player = opp

        # Update position repetition counts
        if is_finish or captured_enemy:
            # Irreversible move: piece counts or scores changed permanently
            self.position_counts.clear()

        pos_key = self.get_position_key()
        repetition_count = self.position_counts.get(pos_key, 0) + 1
        self.position_counts[pos_key] = repetition_count

        # Check remaining termination conditions if game hasn't ended yet
        if not self.done:
            p0_has_pieces = bool(np.any(self.pos[0, :, 0] >= 0))
            p1_has_pieces = bool(np.any(self.pos[1, :, 0] >= 0))

            # 3a. Elimination: a player has no remaining pieces on the board
            if not p0_has_pieces or not p1_has_pieces:
                self.done = True
                self.termination_reason = "elimination"
                self.winner = self._resolve_score_or_progress_winner()
            # 3b. Threefold Repetition: exact same position reached 3 times
            elif repetition_count >= 3:
                self.done = True
                self.termination_reason = "threefold_repetition"
                self.winner = -1
            # 4. 200-Move Limit Tiebreaker
            elif self.ply_count >= MAX_PLIES:
                self.done = True
                self.termination_reason = "ply_limit"
                self.winner = self._resolve_score_or_progress_winner()
            else:
                # 3c. No Legal Moves for the player whose turn it now is
                next_mask = self.get_legal_mask()
                if not np.any(next_mask):
                    self.done = True
                    self.termination_reason = "no_legal_moves"
                    self.winner = self._resolve_score_or_progress_winner()

        return self.done, self.get_outcome(self.current_player)

    def step_move(
        self,
        piece_idx: int,
        target_r: Optional[int] = None,
        target_c: Optional[int] = None,
        finish: bool = False,
        canonical: bool = False,
    ) -> Tuple[bool, float]:
        """Convenience wrapper to execute a move specified by physical or canonical coordinates."""
        action = self.encode_action(
            piece_idx=piece_idx,
            target_r=target_r,
            target_c=target_c,
            finish=finish,
            canonical=canonical,
        )
        return self.step(action)

    def get_outcome(self, perspective_player: Optional[int] = None) -> float:
        """
        Returns exact game outcome z in {-1.0, 0.0, +1.0} from `perspective_player`'s view
        (defaults to `self.current_player`). Returns 0.0 if the game is not done or is a draw.
        """
        if not self.done or self.winner is None or self.winner == -1:
            return 0.0
        if perspective_player is None:
            perspective_player = self.current_player
        return 1.0 if self.winner == perspective_player else -1.0

    # -----------------------------------------------------------------------
    # Text Visualization for Debugging
    # -----------------------------------------------------------------------

    def render_ascii(self) -> str:
        """Renders an ASCII view of the 10x10 board with terrain heights and pieces."""
        short_labels = ("E0", "E1", "V0", "V1", "S0", "S1")
        cell_str = [[f"  .{self.terrain[r, c]} " for c in range(10)] for r in range(10)]

        for p in (0, 1):
            prefix = "A" if p == 0 else "B"
            for k in range(6):
                r, c = int(self.pos[p, k, 0]), int(self.pos[p, k, 1])
                if r >= 0 and c >= 0:
                    cell_str[r][c] = f"{prefix}{short_labels[k]}:{self.terrain[r, c]}"

        lines = [
            f"Ply: {self.ply_count} | Turn: P{self.current_player + 1} | "
            f"Scores: P1={self.scores[0]}, P2={self.scores[1]} | "
            f"Finished: {self.total_pieces_finished} | Done: {self.done} (Winner: {self.winner})"
        ]
        header = "    " + " ".join(f"  c{c} " for c in range(10))
        lines.append(header)
        for r in range(9, -1, -1):
            row_line = f"r{r}  " + " ".join(cell_str[r])
            lines.append(row_line)
        return "\n".join(lines)
