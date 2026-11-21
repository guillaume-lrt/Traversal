"""
AlphaZero Self-Play, Replay Buffer, Evaluation Arena & Training Pipeline for Traversal (Milestone 4).

Implements:
  - Memory-efficient `ReplayBuffer` storing `X_static` once per game ID (Section 5.2)
    with automatic 2x Horizontal Symmetry Augmentation (Section 6.2).
  - Batched GPU self-play loop with cached `F_static` inference and temperature schedule.
  - `HeuristicAgent` baseline and symmetric 2-game-per-terrain Evaluation Arena.
  - End-to-end joint training with `AdamW(lr=1e-3, weight_decay=1e-4)` and cosine annealing to `1e-4`.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numba import njit
import psutil
import torch
from aim import Run as AimRun
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from traversal_env import (
    TraversalGame,
    PIECE_STRENGTH,
    PIECE_SCORE,
    PIECE_TYPES,
    PIECE_EXPLORER,
    PIECE_SENTINEL,
    STATUS_CAPTURED,
    FINISH_TARGET,
    NUM_TARGETS,
    MAX_PLIES,
    ORTHO_DIRS,
    EXPLORER_SPECIAL_OFFSETS,
    compute_player_reach_and_combat,
    compute_position_key,
)
from encoder import (
    compute_blocked_dist_std_v2,
    compute_unblocked_dist_std,
    encode_both_static_tensors,
    encode_dynamic_tensor,
    mirror_static,
    mirror_dynamic,
    mirror_policy,
)
from model import (
    TraversalDualStemNet,
    compute_alphazero_loss,
    load_model_from_checkpoint,
    masked_softmax,
)
from mcts import MCTS, run_batched_mcts_roots
from elo_system import EloDatabase, DEFAULT_DATABASE_PATH


# ---------------------------------------------------------------------------
# 1. Memory-Efficient Replay Buffer with Per-Game Static Deduplication
# ---------------------------------------------------------------------------


@dataclass
class ReplaySample:
    static_key: int
    x_dynamic: np.ndarray  # (34, 10, 10) float32
    pi_mcts: np.ndarray    # (606,) float32
    legal_mask: np.ndarray # (606,) bool
    z: float               # outcome in {-1.0, 0.0, +1.0}


class _BufferView:
    """Lightweight sequence view wrapping ReplayBuffer for backward compatibility with .buffer[i]."""
    __slots__ = ("_rb",)

    def __init__(self, rb: "ReplayBuffer") -> None:
        self._rb = rb

    def __len__(self) -> int:
        return self._rb.size

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._rb.size))]
        if idx < 0:
            idx += self._rb.size
        if idx < 0 or idx >= self._rb.size:
            raise IndexError(f"ReplayBuffer index {idx} out of range (size {self._rb.size})")
        start_idx = self._rb.head if self._rb.size == self._rb.capacity else 0
        actual_idx = (start_idx + idx) % self._rb.capacity
        return ReplaySample(
            static_key=int(self._rb.buf_static_key[actual_idx]),
            x_dynamic=self._rb.buf_x_dyn[actual_idx].astype(np.float32),
            pi_mcts=self._rb.buf_pi[actual_idx],
            legal_mask=self._rb.buf_mask[actual_idx],
            z=float(self._rb.buf_z[actual_idx]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ReplayBuffer:
    """
    High-performance NumPy Ring Buffer that stores `x_dynamic` as float16 to halve RAM
    (review #5.2), indexes mini-batches in O(1) vectorized slices, and stores `X_static`
    once per (game_id, player, is_mirrored) key to minimize memory on limited-RAM systems.
    """

    def __init__(self, capacity: int = 40000) -> None:
        self.capacity = capacity
        self.size: int = 0
        self.head: int = 0
        self.buf_static_key = np.zeros(capacity, dtype=np.int64)
        self.buf_x_dyn: Optional[np.ndarray] = None  # (capacity, C, 10, 10) float16
        self.buf_pi = np.zeros((capacity, 606), dtype=np.float32)
        self.buf_mask = np.zeros((capacity, 606), dtype=np.bool_)
        self.buf_z = np.zeros(capacity, dtype=np.float32)

        self.static_store: Dict[int, np.ndarray] = {}
        self.static_refcount: Dict[int, int] = {}
        self._next_game_id: int = 1
        self.buffer = _BufferView(self)

    def __len__(self) -> int:
        return self.size

    def _register_static(self, key: int, x_static: np.ndarray) -> None:
        if key not in self.static_store:
            self.static_store[key] = x_static.astype(np.float32, copy=True)
            self.static_refcount[key] = 0

    def _push_sample(self, sample: ReplaySample) -> None:
        if self.buf_x_dyn is None:
            dyn_c = sample.x_dynamic.shape[0]
            self.buf_x_dyn = np.zeros((self.capacity, dyn_c, 10, 10), dtype=np.float16)

        # Evict old sample if buffer is full
        if self.size >= self.capacity:
            old_k = int(self.buf_static_key[self.head])
            if old_k in self.static_refcount:
                self.static_refcount[old_k] -= 1
                if self.static_refcount[old_k] <= 0:
                    del self.static_refcount[old_k]
                    self.static_store.pop(old_k, None)

        self.buf_static_key[self.head] = sample.static_key
        self.buf_x_dyn[self.head] = sample.x_dynamic.astype(np.float16, copy=False)
        self.buf_pi[self.head] = sample.pi_mcts
        self.buf_mask[self.head] = sample.legal_mask
        self.buf_z[self.head] = sample.z

        self.static_refcount[sample.static_key] = self.static_refcount.get(sample.static_key, 0) + 1
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    @staticmethod
    def _compute_shaped_outcome(final_game: TraversalGame, player: int) -> float:
        """
        Computes a shaped terminal value target z in [-1.0, +1.0] for `player`:
          - Decisive wins/losses combine a speed bonus and a material preservation bonus:
              z = +/-(0.72 + 0.14 * (1 - ply / 200) + 0.14 * clip(mat_diff / 6, -1, 1))
            so the Value Head prefers fast wins AND penalizes throwing away pieces for free.
          - On 1-1 / 2-2 two-piece-finish draws, breaks ties using remaining on-board
            piece progress and material (+/-0.25) to prevent Value Head collapse around 0.0.
        """
        opp = 1 - player
        my_surviving_str = float(
            np.sum(PIECE_STRENGTH[final_game.piece_status[player] != STATUS_CAPTURED])
        )
        opp_surviving_str = float(
            np.sum(PIECE_STRENGTH[final_game.piece_status[opp] != STATUS_CAPTURED])
        )
        mat_factor = max(-1.0, min(1.0, (my_surviving_str - opp_surviving_str) / 6.0))
        speed_factor = max(0.0, 1.0 - float(final_game.ply_count) / 200.0)
        decisive_mag = max(0.58, min(1.0, 0.72 + 0.14 * speed_factor + 0.14 * mat_factor))

        if final_game.winner == player:
            return decisive_mag
        if final_game.winner is not None and final_game.winner == opp:
            return -decisive_mag

        # Draw (winner == -1):
        if final_game.termination_reason == "threefold_repetition":
            return 0.0

        # On 1-1 / 2-2 two-piece-finish draws, break ties using remaining on-board progress & material
        saved_scores = final_game.scores.copy()
        final_game.scores[:] = 0
        prog_winner = final_game._resolve_score_or_progress_winner()
        final_game.scores[:] = saved_scores
        base_draw = 0.15 * mat_factor
        if prog_winner == player:
            return max(-0.40, min(0.40, 0.20 + base_draw))
        if prog_winner == opp:
            return max(-0.40, min(0.40, -0.20 + base_draw))
        return base_draw

    def add_game_trajectory(
        self,
        x_static_pair: Tuple[np.ndarray, np.ndarray],
        trajectory: List[Tuple[Any, ...]],
        final_game: TraversalGame,
        augment_symmetry: bool = True,
        q_blend: float = 0.0,
        value_target: str = "raw",
    ) -> int:
        """
        Adds a completed game's trajectory `[(player, x_dynamic, pi_mcts, legal_mask, [root_q]), ...]`
        to the replay buffer (plus horizontal reflections if `augment_symmetry=True`).
        `value_target` controls the terminal outcome scale:
          - 'raw': exact {-1.0, 0.0, +1.0} matching MCTS terminal values (review #2.2).
          - 'shaped': heuristic speed/material shaping.
        If `q_blend > 0.0`, blends target outcome with MCTS root Q:
          z' = (1 - q_blend) * z + q_blend * root_q (review #1.4).
        Returns total number of samples added.
        """
        gid = self._next_game_id
        self._next_game_id += 1

        # Keys: gid * 4 + (player * 2 + is_mirror)
        x_s0, x_s1 = x_static_pair
        key_p0_orig = gid * 4 + 0
        key_p1_orig = gid * 4 + 2
        self._register_static(key_p0_orig, x_s0)
        self._register_static(key_p1_orig, x_s1)

        if augment_symmetry:
            key_p0_mir = gid * 4 + 1
            key_p1_mir = gid * 4 + 3
            self._register_static(key_p0_mir, mirror_static(x_s0))
            self._register_static(key_p1_mir, mirror_static(x_s1))

        added = 0
        for item in trajectory:
            player = int(item[0])
            x_dyn = item[1]
            pi_mcts = item[2]
            legal_mask = item[3]
            root_q = float(item[4]) if len(item) > 4 else 0.0

            if value_target == "shaped":
                z_base = ReplayBuffer._compute_shaped_outcome(final_game, player)
            else:
                z_base = float(final_game.get_outcome(player))

            if q_blend > 0.0:
                z = float((1.0 - q_blend) * z_base + q_blend * root_q)
            else:
                z = z_base

            s_key = key_p0_orig if player == 0 else key_p1_orig
            self._push_sample(
                ReplaySample(
                    static_key=s_key,
                    x_dynamic=x_dyn.astype(np.float32, copy=False),
                    pi_mcts=pi_mcts.astype(np.float32, copy=False),
                    legal_mask=legal_mask.astype(np.bool_, copy=False),
                    z=z,
                )
            )
            added += 1

            if augment_symmetry:
                m_key = key_p0_mir if player == 0 else key_p1_mir
                self._push_sample(
                    ReplaySample(
                        static_key=m_key,
                        x_dynamic=mirror_dynamic(x_dyn),
                        pi_mcts=mirror_policy(pi_mcts),
                        legal_mask=mirror_policy(legal_mask),
                        z=z,
                    )
                )
                added += 1

        return added

    def sample_batch(
        self, batch_size: int, device: torch.device, rng: Optional[np.random.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniformly samples a mini-batch `(x_static, x_dynamic, pi_mcts, legal_mask, z)` on `device`
        using O(1) vectorized NumPy contiguous array indexing (review #5.2).
        """
        if rng is None:
            rng = np.random.default_rng()
        n = self.size
        if n == 0 or self.buf_x_dyn is None:
            raise ValueError("Cannot sample from an empty ReplayBuffer")

        k = min(batch_size, n)
        raw_indices = rng.choice(n, size=k, replace=False)
        start_idx = self.head if self.size == self.capacity else 0
        actual_indices = (start_idx + raw_indices) % self.capacity

        static_keys = self.buf_static_key[actual_indices]
        x_statics = np.stack([self.static_store[key] for key in static_keys], axis=0)
        x_dynamics = self.buf_x_dyn[actual_indices].astype(np.float32)
        pis = self.buf_pi[actual_indices]
        masks = self.buf_mask[actual_indices]
        zs = self.buf_z[actual_indices]

        return (
            torch.from_numpy(x_statics).to(device),
            torch.from_numpy(x_dynamics).to(device),
            torch.from_numpy(pis).to(device),
            torch.from_numpy(masks).to(device),
            torch.from_numpy(zs).to(device),
        )

    def save(self, file_path: str) -> None:
        """Saves replay buffer state to disk atomically."""
        dir_name = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(dir_name, exist_ok=True)
        tmp_path = f"{file_path}.tmp"
        payload = {
            "capacity": self.capacity,
            "size": self.size,
            "head": self.head,
            "buf_static_key": self.buf_static_key,
            "buf_x_dyn": self.buf_x_dyn,
            "buf_pi": self.buf_pi,
            "buf_mask": self.buf_mask,
            "buf_z": self.buf_z,
            "static_store": self.static_store,
            "static_refcount": self.static_refcount,
            "_next_game_id": self._next_game_id,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, file_path)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores replay buffer state from a state dictionary (supporting array ring buffers and legacy sample lists)."""
        self._next_game_id = int(state_dict.get("_next_game_id", 1))
        self.static_store = dict(state_dict.get("static_store", {}))
        self.static_refcount = dict(state_dict.get("static_refcount", {}))

        if "buf_x_dyn" in state_dict and state_dict["buf_x_dyn"] is not None:
            self.size = int(state_dict.get("size", 0))
            self.head = int(state_dict.get("head", 0))
            saved_cap = int(state_dict.get("capacity", self.capacity))
            if saved_cap == self.capacity:
                self.buf_static_key = state_dict["buf_static_key"].copy()
                self.buf_x_dyn = state_dict["buf_x_dyn"].copy()
                self.buf_pi = state_dict["buf_pi"].copy()
                self.buf_mask = state_dict["buf_mask"].copy()
                self.buf_z = state_dict["buf_z"].copy()
            else:
                self.size = 0
                self.head = 0
                self.buf_static_key = np.zeros(self.capacity, dtype=np.int64)
                self.buf_x_dyn = None
                self.buf_pi = np.zeros((self.capacity, 606), dtype=np.float32)
                self.buf_mask = np.zeros((self.capacity, 606), dtype=np.bool_)
                self.buf_z = np.zeros(self.capacity, dtype=np.float32)
                old_x = state_dict["buf_x_dyn"]
                old_pi = state_dict["buf_pi"]
                old_m = state_dict["buf_mask"]
                old_z = state_dict["buf_z"]
                old_k = state_dict["buf_static_key"]
                old_size = int(state_dict.get("size", len(old_k)))
                old_head = int(state_dict.get("head", old_size))
                start_i = old_head if old_size == saved_cap else 0
                for i in range(old_size):
                    act_i = (start_i + i) % saved_cap
                    self._push_sample(
                        ReplaySample(
                            static_key=int(old_k[act_i]),
                            x_dynamic=old_x[act_i].astype(np.float32),
                            pi_mcts=old_pi[act_i],
                            legal_mask=old_m[act_i],
                            z=float(old_z[act_i]),
                        )
                    )
        elif "samples" in state_dict:
            samples = state_dict.get("samples", [])
            self.size = 0
            self.head = 0
            self.buf_static_key = np.zeros(self.capacity, dtype=np.int64)
            self.buf_x_dyn = None
            self.buf_pi = np.zeros((self.capacity, 606), dtype=np.float32)
            self.buf_mask = np.zeros((self.capacity, 606), dtype=np.bool_)
            self.buf_z = np.zeros(self.capacity, dtype=np.float32)
            for s in samples:
                self._push_sample(s)

    @classmethod
    def load(cls, file_path: str, capacity: Optional[int] = None) -> "ReplayBuffer":
        """Loads a replay buffer from disk."""
        payload = torch.load(file_path, weights_only=False, map_location="cpu")
        cap = capacity if capacity is not None else int(payload.get("capacity", 40000))
        buf = cls(capacity=cap)
        buf.load_state_dict(payload)
        return buf


# ---------------------------------------------------------------------------
# 2. Batched Self-Play Generator & Fresh Validation Metrics (Section 5.2, 6.1, 6.2)
# ---------------------------------------------------------------------------


def compute_fresh_trajectory_metrics(
    model: TraversalDualStemNet,
    device: torch.device,
    static_np_pairs: List[Tuple[np.ndarray, np.ndarray]],
    trajectories: List[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
    games: List[TraversalGame],
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    Evaluates the model on freshly collected self-play trajectories BEFORE gradient updates
    to compute held-out value and policy generalisation metrics (review #1.1).
    """
    xs_list: List[np.ndarray] = []
    xd_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    z_shaped_list: List[float] = []
    z_raw_list: List[float] = []

    for game_i, (final_game, (s0, s1), traj) in enumerate(zip(games, static_np_pairs, trajectories)):
        for entry in traj:
            player, x_dyn, pi_target, legal_mask = entry[:4]
            xs = s0 if player == 0 else s1
            xs_list.append(xs)
            xd_list.append(x_dyn)
            mask_list.append(legal_mask)
            pi_list.append(pi_target)
            z_shaped_list.append(ReplayBuffer._compute_shaped_outcome(final_game, player))
            z_raw_list.append(final_game.get_outcome(player))

    num_samples = len(xs_list)
    if num_samples == 0:
        return {
            "fresh_value_mse": 0.0,
            "fresh_value_raw_mse": 0.0,
            "fresh_value_sign_acc": 0.0,
            "fresh_value_corr": 0.0,
            "fresh_policy_kl": 0.0,
            "fresh_policy_entropy": 0.0,
            "fresh_policy_ce": 0.0,
        }

    v_pred_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []

    model_was_training = model.training
    model.eval()

    with torch.inference_mode():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            b_xs = torch.from_numpy(np.stack(xs_list[start_idx:end_idx])).to(device)
            b_xd = torch.from_numpy(np.stack(xd_list[start_idx:end_idx])).to(device)
            b_mask = torch.from_numpy(np.stack(mask_list[start_idx:end_idx])).to(device)

            logits, v_pred = model(b_xs, b_xd)
            probs = masked_softmax(logits, b_mask)

            v_pred_list.append(v_pred.squeeze(-1).cpu().numpy())
            probs_list.append(probs.cpu().numpy())

    if model_was_training:
        model.train()

    v_all = np.concatenate(v_pred_list, axis=0)
    probs_all = np.concatenate(probs_list, axis=0)
    pi_all = np.stack(pi_list, axis=0)
    z_shaped_all = np.array(z_shaped_list, dtype=np.float32)
    z_raw_all = np.array(z_raw_list, dtype=np.float32)

    val_mse = float(np.mean((v_all - z_shaped_all) ** 2))
    raw_val_mse = float(np.mean((v_all - z_raw_all) ** 2))

    decisive = z_raw_all != 0
    if np.any(decisive):
        sign_acc = float(np.mean(np.sign(v_all[decisive]) == np.sign(z_raw_all[decisive])))
    else:
        sign_acc = 1.0

    if np.std(v_all) > 1e-6 and np.std(z_shaped_all) > 1e-6:
        corr = float(np.corrcoef(v_all, z_shaped_all)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    eps = 1e-12
    ce = float(-np.sum(pi_all * np.log(probs_all + eps), axis=-1).mean())
    ent = float(-np.sum(np.where(pi_all > 0, pi_all * np.log(np.clip(pi_all, 1e-12, 1.0)), 0.0), axis=-1).mean())
    kl = max(0.0, float(ce - ent))

    return {
        "fresh_value_mse": val_mse,
        "fresh_value_raw_mse": raw_val_mse,
        "fresh_value_sign_acc": sign_acc,
        "fresh_value_corr": corr,
        "fresh_policy_kl": kl,
        "fresh_policy_entropy": ent,
        "fresh_policy_ce": ce,
    }


def generate_self_play_games(
    model: TraversalDualStemNet,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    num_games: int = 8,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    temp_threshold_ply: int = 30,
    temp_floor: float = 0.0,
    heuristic_fraction: float = 0.25,
    heuristic_imitation_weight: float = 0.50,
    terrain_mode: str = "random",
    augment_symmetry: bool = True,
    q_blend: float = 0.0,
    value_target: str = "raw",
    shaping_weight: float = 1.0,
    v_tac_weight: float = 0.30,
    dirichlet_alpha: float = 0.18,
    dirichlet_eps: float = 0.25,
    subtree_reuse: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Plays `num_games` self-play games concurrently in lockstep on `device`,
    caching `F_static` once at Move 0 per game and storing trajectories in `replay_buffer`.
    Includes `heuristic_fraction` pro-am curriculum games against `HeuristicAgent` to prevent
    early draw saturation and bootstrap fast race trajectories.
    """
    if rng is None:
        rng = np.random.default_rng()

    model.eval()
    heuristic = HeuristicAgent(rng=rng)
    num_heur_games = int(round(num_games * heuristic_fraction)) if num_games >= 4 else 0

    games = [
        TraversalGame(terrain_mode=terrain_mode, seed=int(rng.integers(0, 1_000_000_000)))
        for _ in range(num_games)
    ]

    # Compute X_static and cached F_static once per game at Move 0 (Section 5.2)
    static_np_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    f_static_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    all_s = []
    for g in games:
        s0, s1 = encode_both_static_tensors(g.terrain)
        static_np_pairs.append((s0, s1))
        all_s.append(s0)
        all_s.append(s1)

    with torch.inference_mode():
        t_all = torch.from_numpy(np.stack(all_s, axis=0)).to(device)
        f_all = model.encode_static(t_all)  # (2 * num_games, 32, 10, 10)
        for i in range(num_games):
            f_static_pairs.append((f_all[2 * i : 2 * i + 1], f_all[2 * i + 1 : 2 * i + 2]))

    trajectories: List[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = [
        [] for _ in range(num_games)
    ]
    active_indices = list(range(num_games))
    active_roots: Dict[int, Optional[MCTSNode]] = {i: None for i in range(num_games)}

    while active_indices:
        active_games = [games[i] for i in active_indices]
        active_f_statics = [f_static_pairs[i] for i in active_indices]
        active_cur_roots = [active_roots[i] for i in active_indices] if subtree_reuse else None

        roots = run_batched_mcts_roots(
            model=model,
            device=device,
            games=active_games,
            f_static_per_game=active_f_statics,
            num_simulations=num_simulations,
            c_puct=c_puct,
            add_dirichlet_noise=True,
            dirichlet_eps=dirichlet_eps,
            dirichlet_alpha=dirichlet_alpha,
            shaping_weight=shaping_weight,
            v_tac_weight=v_tac_weight,
            existing_roots=active_cur_roots,
            rng=rng,
        )

        next_active: List[int] = []
        for local_k, game_i in enumerate(active_indices):
            g = games[game_i]
            root = roots[local_k]
            player = g.current_player
            encoder_ver = getattr(model, "encoder_version", "v1")
            # Reuse the root x_dyn instead of re-encoding on CPU (review #5.5)
            x_dyn = (
                root.x_dyn
                if getattr(root, "x_dyn", None) is not None
                else encode_dynamic_tensor(g, version=encoder_ver)
            )
            legal_mask = g.get_legal_mask()

            # Check if this turn belongs to the curriculum HeuristicAgent (in games 0..num_heur_games-1)
            is_heur_turn = (game_i < num_heur_games) and (player == (game_i % 2))

            if is_heur_turn:
                _, pi_mcts = root.select_move_by_temperature(temperature=0.0, rng=rng)
                chosen_action = heuristic.select_action(g)
                # Blend MCTS visit distribution with expert heuristic move target (review #2.3)
                if heuristic_imitation_weight > 0.0:
                    pi_target = (1.0 - heuristic_imitation_weight) * pi_mcts
                    pi_target[chosen_action] += heuristic_imitation_weight
                else:
                    pi_target = pi_mcts
            else:
                # Temperature schedule: tau=1.0 for plies 0..half, tau=0.5 for plies half..thresh, tau=temp_floor for >= thresh (review #11)
                if g.ply_count < max(1, temp_threshold_ply // 2):
                    temp = 1.0
                elif g.ply_count < temp_threshold_ply:
                    temp = 0.5
                else:
                    temp = temp_floor
                chosen_action, pi_target = root.select_move_by_temperature(
                    temperature=temp, rng=rng
                )

            root_q = root.get_root_q()
            trajectories[game_i].append((player, x_dyn, pi_target, legal_mask, root_q))

            # Subtree reuse (review #9): retain child subtree for chosen action
            next_root: Optional[MCTSNode] = None
            if subtree_reuse and root.legal_actions is not None:
                matches = np.flatnonzero(root.legal_actions == chosen_action)
                if len(matches) > 0:
                    act_idx = int(matches[0])
                    if act_idx in root.children:
                        next_root = root.children[act_idx]

            g.step(chosen_action)

            if not g.done:
                next_active.append(game_i)
                if next_root is not None:
                    next_root.game = g
                    active_roots[game_i] = next_root
                else:
                    active_roots[game_i] = None
            else:
                active_roots.pop(game_i, None)

        active_indices = next_active

    total_samples = 0
    total_plies = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for i in range(num_games):
        added = replay_buffer.add_game_trajectory(
            x_static_pair=static_np_pairs[i],
            trajectory=trajectories[i],
            final_game=games[i],
            augment_symmetry=augment_symmetry,
            q_blend=q_blend,
            value_target=value_target,
        )
        total_samples += added
        total_plies += games[i].ply_count
        if games[i].winner == 0:
            p1_wins += 1
        elif games[i].winner == 1:
            p2_wins += 1
        else:
            draws += 1

    fresh_metrics = compute_fresh_trajectory_metrics(
        model=model,
        device=device,
        static_np_pairs=static_np_pairs,
        trajectories=trajectories,
        games=games,
    )

    term_counts: Dict[str, int] = {}
    for g in games:
        reason = g.termination_reason or "unknown"
        term_counts[reason] = term_counts.get(reason, 0) + 1

    stats = {
        "games": float(num_games),
        "samples_added": float(total_samples),
        "avg_plies": float(total_plies / max(1, num_games)),
        "p1_win_rate": float(p1_wins / max(1, num_games)),
        "p2_win_rate": float(p2_wins / max(1, num_games)),
        "draw_rate": float(draws / max(1, num_games)),
        "sentinel_win_rate": float(term_counts.get("sentinel_instant_win", 0) / max(1, num_games)),
        "two_finish_rate": float(term_counts.get("two_piece_finish", 0) / max(1, num_games)),
        "elimination_rate": float(term_counts.get("elimination", 0) / max(1, num_games)),
        "repetition_rate": float(term_counts.get("threefold_repetition", 0) / max(1, num_games)),
        "ply_limit_rate": float(term_counts.get("ply_limit", 0) / max(1, num_games)),
    }
    stats.update(fresh_metrics)
    return stats


# ---------------------------------------------------------------------------
# 3. Numba-Accelerated 3-Ply Alpha-Beta Heuristic Baseline Agent & Arena
# ---------------------------------------------------------------------------


@njit(cache=True)
def _resolve_winner_numba(pos: np.ndarray, scores: np.ndarray) -> int:
    if scores[0] > scores[1]:
        return 0
    if scores[1] > scores[0]:
        return 1

    min_rows0 = 999
    min_rows1 = 999
    best_str0 = 0
    best_str1 = 0

    for k in range(6):
        r0 = int(pos[0, k, 0])
        if r0 >= 0:
            rem0 = 9 - r0
            sk = int(PIECE_STRENGTH[k])
            if rem0 < min_rows0 or (rem0 == min_rows0 and sk > best_str0):
                min_rows0 = rem0
                best_str0 = sk

        r1 = int(pos[1, k, 0])
        if r1 >= 0:
            rem1 = r1
            sk = int(PIECE_STRENGTH[k])
            if rem1 < min_rows1 or (rem1 == min_rows1 and sk > best_str1):
                min_rows1 = rem1
                best_str1 = sk

    if min_rows0 < min_rows1:
        return 0
    if min_rows1 < min_rows0:
        return 1
    if best_str0 > best_str1:
        return 0
    if best_str1 > best_str0:
        return 1
    return -1


@njit(cache=True)
def _precompute_terrain_tables_numba(
    terrain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes once per terrain (in ~25 microseconds):
      1. `turns_grid`: shape (2, 3, 10, 10) int32
         Exact minimum number of turns for `(player, piece_type, r, c)` to finish off
         the target row on `terrain` (accounting for Explorer M=5 + leaps/diagonals,
         Vanguard M=4, and Sentinel M=3).
      2. `cost_grid`: shape (2, 10, 10) float32
         Exact standard orthogonal Dijkstra movement cost from `(r, c)` to step off
         the target row for `player in (0, 1)`.
      3. `elev_quality`: shape (10, 10) float32
         Local tactical elevation & steepness score for `(r, c)` across its orthogonal
         neighbors `(nr, nc)` with `dh = h(r, c) - h(nr, nc)`:
           - Combat edge: `sign(dh)` in {-1, 0, +1}
           - Steepness penalty: `-0.65 * (|dh| - 1)` when `|dh| >= 2`
         so a +1 plateau scores +1.0 while a steep +2 peak scores +0.35 and a -2 pit scores -1.65.
    """
    cost_grid = np.empty((2, 10, 10), dtype=np.float32)
    cost_grid[0] = compute_unblocked_dist_std(terrain, 9)
    cost_grid[1] = compute_unblocked_dist_std(terrain, 0)

    turns_grid = np.full((2, 3, 10, 10), 99, dtype=np.int32)
    queue_r = np.empty(100, dtype=np.int32)
    queue_c = np.empty(100, dtype=np.int32)
    bucket_r = np.empty((6, 100), dtype=np.int32)
    bucket_c = np.empty((6, 100), dtype=np.int32)
    bucket_len = np.empty(6, dtype=np.int32)
    d_local = np.empty((10, 10), dtype=np.int32)

    budgets = np.array([5, 4, 3], dtype=np.int32)

    for p in range(2):
        target_row = 9 if p == 0 else 0
        std_dist = cost_grid[p]

        for ptype in range(3):
            mb = int(budgets[ptype])
            q_head = 0
            q_tail = 0

            for r in range(10):
                for c in range(10):
                    if r == target_row or std_dist[r, c] <= float(mb):
                        turns_grid[p, ptype, r, c] = 1
                        queue_r[q_tail] = r
                        queue_c[q_tail] = c
                        q_tail += 1

            while q_head < q_tail:
                r = queue_r[q_head]
                c = queue_c[q_head]
                q_head += 1
                cur_t = turns_grid[p, ptype, r, c]

                # Explorer special leaps/diagonals into (r, c)
                if ptype == PIECE_EXPLORER:
                    for s in range(8):
                        sr = r - EXPLORER_SPECIAL_OFFSETS[s, 0]
                        sc = c - EXPLORER_SPECIAL_OFFSETS[s, 1]
                        if 0 <= sr < 10 and 0 <= sc < 10:
                            if turns_grid[p, ptype, sr, sc] > cur_t + 1:
                                turns_grid[p, ptype, sr, sc] = cur_t + 1
                                queue_r[q_tail] = sr
                                queue_c[q_tail] = sc
                                q_tail += 1

                # Standard orthogonal Dijkstra within budget `mb` into (r, c)
                for rr in range(10):
                    for cc in range(10):
                        d_local[rr, cc] = 999
                for b in range(mb + 1):
                    bucket_len[b] = 0

                d_local[r, c] = 0
                bucket_r[0, 0] = r
                bucket_c[0, 0] = c
                bucket_len[0] = 1

                for cost in range(mb):
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
                                scost = abs(int(terrain[nr, nc]) - int(terrain[cr, cc])) + 1
                                ncost = cost + scost
                                if ncost <= mb and ncost < d_local[nr, nc]:
                                    d_local[nr, nc] = ncost
                                    b_idx = bucket_len[ncost]
                                    bucket_r[ncost, b_idx] = nr
                                    bucket_c[ncost, b_idx] = nc
                                    bucket_len[ncost] = b_idx + 1
                                    if turns_grid[p, ptype, nr, nc] > cur_t + 1:
                                        turns_grid[p, ptype, nr, nc] = cur_t + 1
                                        queue_r[q_tail] = nr
                                        queue_c[q_tail] = nc
                                        q_tail += 1

    elev_quality = np.zeros((10, 10), dtype=np.float32)
    for r in range(10):
        for c in range(10):
            h0 = int(terrain[r, c])
            score_sum = 0.0
            n_cnt = 0
            for d in range(4):
                nr = r + ORTHO_DIRS[d, 0]
                nc = c + ORTHO_DIRS[d, 1]
                if 0 <= nr < 10 and 0 <= nc < 10:
                    dh = h0 - int(terrain[nr, nc])
                    if dh > 0:
                        edge_val = 1.0
                    elif dh < 0:
                        edge_val = -1.0
                    else:
                        edge_val = 0.0
                    if abs(dh) >= 2:
                        edge_val -= 0.65 * float(abs(dh) - 1)
                    score_sum += edge_val
                    n_cnt += 1
            if n_cnt > 0:
                elev_quality[r, c] = score_sum / float(n_cnt)

    return turns_grid, cost_grid, elev_quality


@njit(cache=True)
def _evaluate_board_numba(
    terrain: np.ndarray,
    turns_grid: np.ndarray,
    cost_grid: np.ndarray,
    elev_quality: np.ndarray,
    pos: np.ndarray,
    scores: np.ndarray,
    total_finished: int,
    p: int,
) -> float:
    """
    Threat-aware & terrain-aware static board evaluation from player `p`'s perspective
    (where it is currently `p`'s turn to move). Accounts for:
      - Banked finish scores & active piece material
      - Exact terrain `turns_to_finish` and Dijkstra movement cost to finish
      - Steep penalties for opponent pieces with `turns_to_finish <= 2` (and Sentinels `<= 3`)
      - Local `+1` vs `+2` elevation quality (`elev_quality`)
      - Piece cohesion / phalanx support (orthogonally adjacent & escorting friendly pairs)
      - Immediate captures & forks
    """
    opp = 1 - p
    my_target = 9 if p == 0 else 0
    opp_target = 0 if p == 0 else 9

    my_dest, my_fin, _, my_max_atk, my_def, opp_def = compute_player_reach_and_combat(
        terrain, pos[p], pos[opp], my_target
    )
    opp_dest, opp_fin, _, opp_max_atk, _, _ = compute_player_reach_and_combat(
        terrain, pos[opp], pos[p], opp_target
    )

    my_blk_dist = compute_blocked_dist_std_v2(terrain, pos[p], pos[opp], my_target)
    opp_blk_dist = compute_blocked_dist_std_v2(terrain, pos[p], pos[opp], opp_target)

    # Check if player `p` has an immediate winning finish on this turn
    my_best_fin_pts = 0.0
    for k in range(6):
        if pos[p, k, 0] >= 0 and my_fin[k] == 1:
            pts = float(PIECE_SCORE[k])
            if total_finished == 0 and PIECE_TYPES[k] == PIECE_SENTINEL:
                return 9000.0
            if total_finished >= 1 and (scores[p] + int(pts) > scores[opp]):
                return 8800.0
            if total_finished >= 1 and (scores[p] + int(pts) < scores[opp]):
                continue  # Suicidal 2nd finish hands match to opponent; do not reward as positive finish
            if pts > my_best_fin_pts:
                my_best_fin_pts = pts

    # Check if player `p` has any legal moves at all
    has_any_move = my_best_fin_pts > 0.0
    if not has_any_move:
        for k in range(6):
            if pos[p, k, 0] >= 0 and np.any(my_dest[k] > 0):
                has_any_move = True
                break
    if not has_any_move:
        w = _resolve_winner_numba(pos, scores)
        if w == p:
            return 8500.0
        if w == opp:
            return -8500.0
        return 0.0

    # 1. Banked finish points + active material on board
    val = float(scores[p] - scores[opp]) * (550.0 + 185.0)

    my_mat = 0.0
    opp_mat = 0.0
    # Top-2 fewest turns to finish (0 if already finished, 99 if captured)
    my_best_t1 = 0 if scores[p] > 0 else 99
    my_best_t2 = 99
    opp_best_t1 = 0 if scores[opp] > 0 else 99
    opp_best_t2 = 99
    opp_dir = 1 if opp == 0 else -1

    my_occ = np.full((10, 10), -1, dtype=np.int32)
    for k in range(6):
        mr, mc = int(pos[p, k, 0]), int(pos[p, k, 1])
        if mr >= 0 and mc >= 0:
            my_occ[mr, mc] = k

    for k in range(6):
        pt = int(PIECE_TYPES[k])
        sk = float(PIECE_STRENGTH[k])
        is_sent = pt == PIECE_SENTINEL

        mr = int(pos[p, k, 0])
        mc = int(pos[p, k, 1])
        if mr >= 0:
            my_mat += sk
            m_turns = int(turns_grid[p, pt, mr, mc])
            m_cost = float(my_blk_dist[mr, mc])
            if m_turns <= 2 and m_cost >= 26.0:
                m_turns = max(m_turns, 4)

            # Terrain-aware progress (both Dijkstra cost and exact turn count)
            val += (3.8 + 1.2 * sk) * max(0.0, 22.0 - m_cost)
            val += (18.0 + 6.0 * sk) * max(0.0, 7.0 - float(m_turns))

            if m_turns == 1:
                if (is_sent and total_finished == 0) or (
                    total_finished >= 1 and (scores[p] + int(PIECE_SCORE[k]) > scores[opp])
                ):
                    val += 850.0
                elif total_finished >= 1 and (scores[p] + int(PIECE_SCORE[k]) < scores[opp]):
                    pass
                else:
                    val += 170.0 * sk
            elif m_turns == 2:
                if (is_sent and total_finished == 0) or (
                    total_finished >= 1 and (scores[p] + int(PIECE_SCORE[k]) > scores[opp])
                ):
                    val += 310.0
                elif total_finished >= 1 and (scores[p] + int(PIECE_SCORE[k]) < scores[opp]):
                    pass
                else:
                    val += 65.0 * sk

            is_suicidal_finish = total_finished >= 1 and (scores[p] + int(PIECE_SCORE[k]) < scores[opp])
            if not is_suicidal_finish:
                if m_turns < my_best_t1:
                    my_best_t2 = my_best_t1
                    my_best_t1 = m_turns
                elif m_turns < my_best_t2:
                    my_best_t2 = m_turns

            # Check if an enemy piece is within tactical range (Manhattan distance <= 4)
            near_enemy = False
            for ok in range(6):
                er, ec = int(pos[opp, ok, 0]), int(pos[opp, ok, 1])
                if er >= 0 and (abs(mr - er) + abs(mc - ec) <= 4):
                    near_enemy = True
                    break

            # Local +1 vs +2 elevation quality bonus
            eq = float(elev_quality[mr, mc])
            if near_enemy:
                val += (22.0 + 6.0 * sk) * eq
            else:
                val += 8.0 * eq

            # Piece cohesion / phalanx support bonus
            adj_f = int(my_def[mr, mc]) - int(PIECE_STRENGTH[k])
            if adj_f > 0:
                # Each adjacent link is counted across pairs; reward stronger in combat zones or on Sentinels
                cohesion_w = 18.0 if (near_enemy or is_sent) else 11.0
                val += cohesion_w * float(adj_f)

            # Mutual escort bonus (friendly piece at Manhattan distance 2 ready to support)
            for j in range(k + 1, 6):
                jr, jc = int(pos[p, j, 0]), int(pos[p, j, 1])
                if jr >= 0 and (abs(mr - jr) + abs(mc - jc) == 2):
                    val += 10.0
                    break

        orow = int(pos[opp, k, 0])
        ocol = int(pos[opp, k, 1])
        if orow >= 0:
            opp_mat += sk
            o_turns = int(turns_grid[opp, pt, orow, ocol])
            o_cost = float(opp_blk_dist[orow, ocol])
            if o_turns <= 2 and o_cost >= 26.0:
                o_turns = max(o_turns, 4)

            val -= (3.8 + 1.2 * sk) * max(0.0, 22.0 - o_cost)
            val -= (18.0 + 6.0 * sk) * max(0.0, 7.0 - float(o_turns))

            # Steep terrain-aware penalty when opponent pieces are <= 2 turns (or Sentinel <= 3 turns) from finishing!
            can_kill_now = (my_max_atk[orow, ocol] >= opp_def[orow, ocol]) and (
                opp_def[orow, ocol] > 0
            )
            if not can_kill_now:
                if is_sent and total_finished == 0 and o_turns <= 3:
                    if o_turns == 1:
                        sent_danger = 2400.0
                    elif o_turns == 2:
                        sent_danger = 1250.0
                    else:
                        sent_danger = 420.0

                    front_r = orow + opp_dir
                    if 0 <= front_r < 10 and my_occ[front_r, ocol] != -1:
                        if my_def[front_r, ocol] > 3.0:
                            sent_danger *= 0.55
                        else:
                            sent_danger *= 0.80
                    val -= sent_danger
                elif o_turns <= 2:
                    if total_finished >= 1 and (
                        scores[opp] + int(PIECE_SCORE[k]) > scores[p]
                    ):
                        val -= 2400.0 if o_turns == 1 else 750.0
                    elif total_finished >= 1 and (
                        scores[opp] + int(PIECE_SCORE[k]) < scores[p]
                    ):
                        # Opponent finish terminates the match with a win for player p!
                        pass
                    else:
                        val -= (250.0 if o_turns == 1 else 95.0) * sk

            is_opp_suicidal_finish = total_finished >= 1 and (scores[opp] + int(PIECE_SCORE[k]) < scores[p])
            if not is_opp_suicidal_finish:
                if o_turns < opp_best_t1:
                    opp_best_t2 = opp_best_t1
                    opp_best_t1 = o_turns
                elif o_turns < opp_best_t2:
                    opp_best_t2 = o_turns

            near_my = False
            for mk in range(6):
                mr2, mc2 = int(pos[p, mk, 0]), int(pos[p, mk, 1])
                if mr2 >= 0 and (abs(orow - mr2) + abs(ocol - mc2) <= 4):
                    near_my = True
                    break

            oeq = float(elev_quality[orow, ocol])
            if near_my:
                val -= (22.0 + 6.0 * sk) * oeq
            else:
                val -= 8.0 * oeq

            adj_o = int(opp_def[orow, ocol]) - int(PIECE_STRENGTH[k])
            if adj_o > 0:
                val -= (18.0 if (near_my or is_sent) else 11.0) * float(adj_o)

    val += (my_mat - opp_mat) * 185.0
    if total_finished >= 1:
        # Endgame: exactly ONE piece finish ends the game (review #6.3 & #114)
        my_endgame_t = float(min(10, my_best_t2 if scores[p] > 0 else my_best_t1))
        opp_endgame_t = float(min(10, opp_best_t2 if scores[opp] > 0 else opp_best_t1))
        val += 85.0 * (opp_endgame_t - my_endgame_t)
    else:
        # Opening/Midgame: both pieces needed to complete the traversal
        my_race_t = float(min(10, my_best_t1) + min(10, my_best_t2))
        opp_race_t = float(min(10, opp_best_t1) + min(10, opp_best_t2))
        val += 55.0 * (opp_race_t - my_race_t)

    # 2. Best immediate capture available to `p` right now
    best_my_cap = 0.0
    for ok in range(6):
        er = int(pos[opp, ok, 0])
        ec = int(pos[opp, ok, 1])
        if er >= 0 and ec >= 0:
            if my_max_atk[er, ec] >= opp_def[er, ec] and opp_def[er, ec] > 0:
                s_vic = float(PIECE_STRENGTH[ok])
                # Check if another opponent piece oj != ok can recapture at (er, ec)
                min_cap_str = 99
                for mk in range(6):
                    if my_dest[mk, er, ec] == 1:
                        sk_mk = int(PIECE_STRENGTH[mk])
                        if sk_mk < min_cap_str:
                            min_cap_str = sk_mk
                recap = (min_cap_str <= 6) and (opp_max_atk[er, ec] >= float(min_cap_str))
                gain = max(0.0, s_vic - float(min_cap_str)) if recap else s_vic
                if gain > best_my_cap:
                    best_my_cap = gain

    # 3. Opponent threats against `p`'s pieces (and opponent finish threats)
    threat1 = 0.0
    threat2 = 0.0
    for k in range(6):
        mr = int(pos[p, k, 0])
        mc = int(pos[p, k, 1])
        if mr >= 0 and mc >= 0:
            if opp_max_atk[mr, mc] >= my_def[mr, mc] and my_def[mr, mc] > 0:
                sk = float(PIECE_STRENGTH[k])
                if sk > threat1:
                    threat2 = threat1
                    threat1 = sk
                elif sk > threat2:
                    threat2 = sk

    opp_fin_threat = 0.0
    for ok in range(6):
        er = int(pos[opp, ok, 0])
        ec = int(pos[opp, ok, 1])
        if er >= 0 and ec >= 0 and opp_fin[ok] == 1:
            if my_max_atk[er, ec] < opp_def[er, ec]:
                if total_finished == 0 and PIECE_TYPES[ok] == PIECE_SENTINEL:
                    opp_fin_threat = max(opp_fin_threat, 4200.0)
                elif total_finished >= 1:
                    if scores[opp] + int(PIECE_SCORE[ok]) > scores[p]:
                        opp_fin_threat = max(opp_fin_threat, 3600.0)
                    elif scores[opp] + int(PIECE_SCORE[ok]) == scores[p]:
                        opp_fin_threat = max(opp_fin_threat, 180.0)
                    # When scores[opp] + PIECE_SCORE[ok] < scores[p], opponent finish is suicidal for them (we win)
                else:
                    opp_fin_threat = max(opp_fin_threat, 420.0 * float(PIECE_SCORE[ok]))

    val += 240.0 * my_best_fin_pts - opp_fin_threat
    val += 135.0 * best_my_cap - 45.0 * threat1 - 145.0 * threat2
    return val


@njit(cache=True)
def _grow_move_buffers(
    move_k: np.ndarray,
    move_r: np.ndarray,
    move_c: np.ndarray,
    move_fin: np.ndarray,
    move_score: np.ndarray,
    num_moves: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dynamically doubles the capacity of candidate move buffers when full (review #5.3)."""
    new_cap = max(move_k.shape[0] * 2, 512)
    new_k = np.empty(new_cap, dtype=np.int32)
    new_k[:num_moves] = move_k[:num_moves]
    new_r = np.empty(new_cap, dtype=np.int32)
    new_r[:num_moves] = move_r[:num_moves]
    new_c = np.empty(new_cap, dtype=np.int32)
    new_c[:num_moves] = move_c[:num_moves]
    new_fin = np.empty(new_cap, dtype=np.int32)
    new_fin[:num_moves] = move_fin[:num_moves]
    new_score = np.empty(new_cap, dtype=np.float32)
    new_score[:num_moves] = move_score[:num_moves]
    return new_k, new_r, new_c, new_fin, new_score


@njit(cache=True)
def _generate_ordered_moves_numba(
    terrain: np.ndarray,
    turns_grid: np.ndarray,
    cost_grid: np.ndarray,
    elev_quality: np.ndarray,
    pos: np.ndarray,
    scores: np.ndarray,
    total_finished: int,
    p: int,
    max_candidates: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generates legal moves for player `p`, scores them with a terrain-aware and threat-aware
    tactical heuristic (finishes, MVV-LVA captures, `turns_to_finish <= 2` interceptions,
    threat escapes, hanging penalties, `+1` vs `+2` elevation quality, phalanx cohesion,
    and exact Dijkstra progress), and returns the top `max_candidates` moves sorted descending.
    """
    opp = 1 - p
    my_target = 9 if p == 0 else 0
    opp_target = 0 if p == 0 else 9

    my_dest, my_fin, _, _, my_def, _ = compute_player_reach_and_combat(
        terrain, pos[p], pos[opp], my_target
    )
    opp_dest, opp_fin, _, opp_max_atk, _, _ = compute_player_reach_and_combat(
        terrain, pos[opp], pos[p], opp_target
    )

    my_occ = np.full((10, 10), -1, dtype=np.int32)
    opp_occ = np.full((10, 10), -1, dtype=np.int32)
    opp_dir = 1 if opp == 0 else -1
    opp_sent_emergency = False
    for k in range(6):
        mr, mc = int(pos[p, k, 0]), int(pos[p, k, 1])
        if mr >= 0 and mc >= 0:
            my_occ[mr, mc] = k
        er, ec = int(pos[opp, k, 0]), int(pos[opp, k, 1])
        if er >= 0 and ec >= 0:
            opp_occ[er, ec] = k
            opp_pt = int(PIECE_TYPES[k])
            o_turns = int(turns_grid[opp, opp_pt, er, ec])
            if total_finished == 0 and opp_pt == PIECE_SENTINEL and o_turns <= 2:
                opp_sent_emergency = True

    initial_cap = 512
    move_k = np.empty(initial_cap, dtype=np.int32)
    move_r = np.empty(initial_cap, dtype=np.int32)
    move_c = np.empty(initial_cap, dtype=np.int32)
    move_fin = np.empty(initial_cap, dtype=np.int32)
    move_score = np.empty(initial_cap, dtype=np.float32)
    num_moves = 0

    for k in range(6):
        r0 = int(pos[p, k, 0])
        c0 = int(pos[p, k, 1])
        if r0 < 0 or c0 < 0:
            continue

        pt = int(PIECE_TYPES[k])
        sk = int(PIECE_STRENGTH[k])
        pts_k = int(PIECE_SCORE[k])
        t0 = int(turns_grid[p, pt, r0, c0])
        cost0 = float(cost_grid[p, r0, c0])
        is_threatened_orig = (opp_max_atk[r0, c0] >= my_def[r0, c0]) and (my_def[r0, c0] > 0)

        # 1. Finish action
        if my_fin[k] == 1:
            if total_finished == 0 and pt == PIECE_SENTINEL:
                sc = 20000.0
            elif total_finished >= 1:
                if scores[p] + pts_k > scores[opp]:
                    sc = 15000.0 + 100.0 * float(pts_k)
                elif scores[p] + pts_k == scores[opp]:
                    sc = 40.0
                else:
                    sc = -15000.0
            else:
                # Finishing 1st also disarms the opponent's Sentinel Instant-Win rule!
                sc = 850.0 + 240.0 * float(pts_k)
                if opp_sent_emergency:
                    sc += 1800.0

            if num_moves >= move_k.shape[0]:
                move_k, move_r, move_c, move_fin, move_score = _grow_move_buffers(
                    move_k, move_r, move_c, move_fin, move_score, num_moves
                )

            move_k[num_moves] = k
            move_r[num_moves] = -1
            move_c[num_moves] = -1
            move_fin[num_moves] = 1
            move_score[num_moves] = sc
            num_moves += 1

        # 2. Board destinations
        for rt in range(10):
            for ct in range(10):
                if my_dest[k, rt, ct] == 0:
                    continue

                tt = int(turns_grid[p, pt, rt, ct])
                cost_t = float(cost_grid[p, rt, ct])
                cost_adv = cost0 - cost_t
                turn_adv = t0 - tt
                ct_canon = ct if p == 0 else (9 - ct)

                # Terrain-aware base progress score + local elevation quality (+1 ridge vs +2 peak)
                sc = (
                    cost_adv * (11.0 + 2.5 * float(sk))
                    + float(turn_adv) * (38.0 + 8.0 * float(sk))
                    + max(0.0, 22.0 - cost_t) * 2.5
                    + 16.0 * float(elev_quality[rt, ct])
                    - 0.35 * abs(float(ct_canon) - 4.5)
                )

                # Compute new defense & friendly adjacency (phalanx cohesion) at (rt, ct)
                adj_friends = 0
                for d in range(4):
                    nr = rt + ORTHO_DIRS[d, 0]
                    nc = ct + ORTHO_DIRS[d, 1]
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        fk = my_occ[nr, nc]
                        if fk != -1 and fk != k:
                            adj_friends += 1
                new_def = sk + adj_friends
                sc += 22.0 * float(adj_friends)
                opp_atk_dest = float(opp_max_atk[rt, ct])

                opp_k = opp_occ[rt, ct]
                if opp_k != -1:
                    s_opp = int(PIECE_STRENGTH[opp_k])
                    opp_pt = int(PIECE_TYPES[opp_k])
                    o_turns = int(turns_grid[opp, opp_pt, rt, ct])
                    is_opp_sent = opp_pt == PIECE_SENTINEL

                    # Huge bonus if capturing an opponent piece that can finish in 1 or 2 turns!
                    if opp_fin[opp_k] == 1:
                        if total_finished == 0 and is_opp_sent:
                            sc += 8000.0
                        elif total_finished >= 1 and (scores[opp] + int(PIECE_SCORE[opp_k]) > scores[p]):
                            sc += 7000.0
                        else:
                            sc += 1800.0 + 400.0 * float(s_opp)
                    elif total_finished == 0 and is_opp_sent and o_turns <= 3:
                        # Stop enemy Sentinel at all costs!
                        sc += 750.0 + 420.0 * float(4 - o_turns)
                    elif o_turns <= 2:
                        sc += 260.0 * float((3 - o_turns) * s_opp)

                    # Check if another opponent piece oj != opp_k can recapture at (rt, ct)
                    is_trade = opp_atk_dest >= float(new_def)
                    if not is_trade:
                        # Free / safe capture
                        sc += 480.0 + 185.0 * float(s_opp) - 12.0 * float(sk)
                    else:
                        # Trade capture
                        sc += 200.0 + 175.0 * float(s_opp - sk)
                else:
                    # Empty destination square: check if blocking/converging on a dangerous enemy piece
                    block_bonus = 0.0
                    is_emergency_block = False
                    for ok in range(6):
                        er = int(pos[opp, ok, 0])
                        ec = int(pos[opp, ok, 1])
                        if er < 0 or ec < 0:
                            continue
                        opp_pt = int(PIECE_TYPES[ok])
                        o_turns = int(turns_grid[opp, opp_pt, er, ec])
                        is_opp_sent = opp_pt == PIECE_SENTINEL
                        if (is_opp_sent and total_finished == 0 and o_turns <= 3) or o_turns <= 2:
                            # 1. Directly in front of the advancing piece (blocking its path)
                            if rt == er + opp_dir and ct == ec:
                                if is_opp_sent and total_finished == 0:
                                    block_bonus += 520.0 + 180.0 * float(4 - o_turns)
                                    if o_turns <= 2 or new_def > 3:
                                        is_emergency_block = True
                                else:
                                    if total_finished >= 1 and (scores[opp] + int(PIECE_SCORE[ok]) > scores[p]):
                                        block_bonus += 360.0 + 120.0 * float(3 - o_turns)
                                        if o_turns <= 2 or new_def > 3:
                                            is_emergency_block = True
                                    else:
                                        block_bonus += 240.0 + 90.0 * float(3 - o_turns)
                            # 2. Orthogonally adjacent to an enemy Sentinel (setting up +S_k allied support to kill it)
                            elif is_opp_sent and total_finished == 0 and (abs(rt - er) + abs(ct - ec) == 1):
                                block_bonus += 260.0 + 60.0 * float(sk)

                    sc += block_bonus
                    if opp_atk_dest >= float(new_def):
                        if is_emergency_block and sk == 1:
                            sc -= 40.0
                        else:
                            sc -= 290.0 * float(sk)
                    elif is_threatened_orig:
                        sc += 230.0 * float(sk)

                if num_moves >= move_k.shape[0]:
                    move_k, move_r, move_c, move_fin, move_score = _grow_move_buffers(
                        move_k, move_r, move_c, move_fin, move_score, num_moves
                    )

                move_k[num_moves] = k
                move_r[num_moves] = rt
                move_c[num_moves] = ct
                move_fin[num_moves] = 0
                move_score[num_moves] = sc
                num_moves += 1

    limit = min(num_moves, max_candidates)
    for i in range(limit):
        best_idx = i
        best_val = move_score[i]
        for j in range(i + 1, num_moves):
            if move_score[j] > best_val:
                best_val = move_score[j]
                best_idx = j
        if best_idx != i:
            move_score[i], move_score[best_idx] = move_score[best_idx], move_score[i]
            move_k[i], move_k[best_idx] = move_k[best_idx], move_k[i]
            move_r[i], move_r[best_idx] = move_r[best_idx], move_r[i]
            move_c[i], move_c[best_idx] = move_c[best_idx], move_c[i]
            move_fin[i], move_fin[best_idx] = move_fin[best_idx], move_fin[i]

    return (
        move_k[:limit],
        move_r[:limit],
        move_c[:limit],
        move_fin[:limit],
        limit,
    )


@njit(cache=True)
def _step_and_check_terminal_numba(
    pos: np.ndarray,
    scores: np.ndarray,
    total_finished: int,
    ply_count: int,
    p: int,
    k: int,
    rt: int,
    ct: int,
    is_fin: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Applies move `(k, rt, ct, is_fin)` for player `p` on copied state arrays and returns
    `(next_pos, next_scores, next_finished, next_ply, is_done, winner)`.
    """
    opp = 1 - p
    next_pos = pos.copy()
    next_scores = scores.copy()
    next_finished = total_finished
    next_ply = ply_count + 1
    is_done = 0
    winner = -1

    if is_fin == 1:
        next_pos[p, k, 0] = -1
        next_pos[p, k, 1] = -1
        was_first = next_finished == 0
        next_finished += 1
        next_scores[p] += PIECE_SCORE[k]

        if was_first and PIECE_TYPES[k] == PIECE_SENTINEL:
            is_done = 1
            winner = p
        elif next_finished >= 2:
            is_done = 1
            if next_scores[0] > next_scores[1]:
                winner = 0
            elif next_scores[1] > next_scores[0]:
                winner = 1
            else:
                winner = -1
    else:
        for ok in range(6):
            if next_pos[opp, ok, 0] == rt and next_pos[opp, ok, 1] == ct:
                next_pos[opp, ok, 0] = -1
                next_pos[opp, ok, 1] = -1
                break
        next_pos[p, k, 0] = rt
        next_pos[p, k, 1] = ct

    if is_done == 0:
        p0_alive = False
        p1_alive = False
        for i in range(6):
            if next_pos[0, i, 0] >= 0:
                p0_alive = True
            if next_pos[1, i, 0] >= 0:
                p1_alive = True
        if not p0_alive or not p1_alive or next_ply >= MAX_PLIES:
            is_done = 1
            winner = _resolve_winner_numba(next_pos, next_scores)

    return next_pos, next_scores, next_finished, next_ply, is_done, winner


HEURISTIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "blitz": {
        "key": "blitz",
        "name": "Blitz",
        "icon": "⚡",
        "depth": 2,
        "root_beam": 10,
        "inner_limits": np.array([0, 6, 6], dtype=np.int32),
        "description": "Fast & intuitive (2-ply α-β, 10/6 beam, ~0.8ms/move)",
    },
    "standard": {
        "key": "standard",
        "name": "Standard",
        "icon": "⚖️",
        "depth": 4,
        "root_beam": 14,
        "inner_limits": np.array([0, 5, 7, 10, 10], dtype=np.int32),
        "description": "Balanced tactical depth (4-ply α-β, 14/10/7/5 beam, ~5.1ms/move)",
    },
    "master": {
        "key": "master",
        "name": "Master",
        "icon": "🧠",
        "depth": 6,
        "root_beam": 16,
        "inner_limits": np.array([0, 5, 6, 7, 8, 10, 10], dtype=np.int32),
        "description": "Deep tactical search (6-ply α-β, 16/10/8/7/6/5 beam, ~76ms/move)",
    },
    "grandmaster": {
        "key": "grandmaster",
        "name": "Grandmaster",
        "icon": "👑",
        "depth": 8,
        "root_beam": 16,
        "inner_limits": np.array([0, 4, 4, 5, 5, 6, 7, 8, 8], dtype=np.int32),
        "description": "Ultra deep tactical search (8-ply α-β, 16/8/8/7/6/5/5/4 beam, ~480ms/move)",
    },
}


@njit(cache=True)
def _alphabeta_negamax_numba(
    terrain: np.ndarray,
    turns_grid: np.ndarray,
    cost_grid: np.ndarray,
    elev_quality: np.ndarray,
    pos: np.ndarray,
    scores: np.ndarray,
    total_finished: int,
    ply_count: int,
    p: int,
    depth: int,
    alpha: float,
    beta: float,
    inner_limits: Optional[np.ndarray] = None,
    piece_status: Optional[np.ndarray] = None,
    rep_keys_v0: Optional[np.ndarray] = None,
    rep_keys_v1: Optional[np.ndarray] = None,
    rep_counts: Optional[np.ndarray] = None,
) -> float:
    if depth <= 0:
        return _evaluate_board_numba(
            terrain, turns_grid, cost_grid, elev_quality, pos, scores, total_finished, p
        )

    if inner_limits is None:
        max_cands = 8 if depth >= 3 else (6 if depth == 2 else 5)
    else:
        max_cands = inner_limits[depth] if depth < len(inner_limits) else inner_limits[-1]

    cand_k, cand_r, cand_c, cand_fin, count = _generate_ordered_moves_numba(
        terrain,
        turns_grid,
        cost_grid,
        elev_quality,
        pos,
        scores,
        total_finished,
        p,
        max_cands,
    )

    if count == 0:
        w = _resolve_winner_numba(pos, scores)
        if w == p:
            return 9500.0 - float(ply_count)
        if w == (1 - p):
            return -9500.0 + float(ply_count)
        return 0.0

    best_val = -1e9
    for i in range(count):
        cand_k_i = int(cand_k[i])
        cand_r_i = int(cand_r[i])
        cand_c_i = int(cand_c[i])
        cand_fin_i = int(cand_fin[i])

        is_cap = False
        if cand_fin_i == 0:
            for ok in range(6):
                if pos[1 - p, ok, 0] == cand_r_i and pos[1 - p, ok, 1] == cand_c_i:
                    is_cap = True
                    break

        next_pos, next_scores, next_fin, next_ply, is_done, winner = (
            _step_and_check_terminal_numba(
                pos,
                scores,
                total_finished,
                ply_count,
                p,
                cand_k_i,
                cand_r_i,
                cand_c_i,
                cand_fin_i,
            )
        )

        is_rep_draw = False
        is_rep_twice = False
        if is_done == 0 and cand_fin_i == 0 and not is_cap and rep_keys_v0 is not None and len(rep_keys_v0) > 0 and piece_status is not None:
            v0, v1 = compute_position_key(next_pos, piece_status, 1 - p)
            for r_idx in range(len(rep_keys_v0)):
                if rep_keys_v0[r_idx] == v0 and rep_keys_v1[r_idx] == v1:
                    if rep_counts[r_idx] >= 2:
                        is_rep_draw = True
                    elif rep_counts[r_idx] == 1:
                        is_rep_twice = True
                    break

        if is_rep_draw:
            val = 0.0
        elif is_done == 1:
            if winner == p:
                val = 9500.0 - float(next_ply)
            elif winner == (1 - p):
                val = -9500.0 + float(next_ply)
            else:
                val = 0.0
        else:
            val = -_alphabeta_negamax_numba(
                terrain,
                turns_grid,
                cost_grid,
                elev_quality,
                next_pos,
                next_scores,
                next_fin,
                next_ply,
                1 - p,
                depth - 1,
                -beta,
                -alpha,
                inner_limits,
                piece_status,
                rep_keys_v0,
                rep_keys_v1,
                rep_counts,
            )
            if is_rep_twice and val > 50.0:
                val -= 120.0

        if val > best_val:
            best_val = val
        if best_val > alpha:
            alpha = best_val
        if alpha >= beta:
            break

    return best_val


@njit(cache=True)
def search_best_heuristic_moves_numba(
    terrain: np.ndarray,
    pos: np.ndarray,
    scores: np.ndarray,
    total_finished: int,
    ply_count: int,
    p: int,
    depth: int = 4,
    root_beam: int = 14,
    inner_limits: Optional[np.ndarray] = None,
    piece_status: Optional[np.ndarray] = None,
    rep_keys_v0: Optional[np.ndarray] = None,
    rep_keys_v1: Optional[np.ndarray] = None,
    rep_counts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a `depth`-ply Negamax Alpha-Beta search over the top `root_beam` tactically
    ordered candidate moves at root and returns `(canonical_actions, move_values)`.
    Precomputes exact terrain turns-to-finish, Dijkstra cost-to-finish, and local
    +1 vs +2 elevation quality tables once at the root.
    """
    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    if piece_status is None:
        p_status = np.zeros((2, 6), dtype=np.int8)
        for pl in range(2):
            for k_idx in range(6):
                if pos[pl, k_idx, 0] < 0:
                    p_status[pl, k_idx] = 1
    else:
        p_status = piece_status

    if rep_keys_v0 is None:
        r_v0 = np.empty(0, dtype=np.int64)
        r_v1 = np.empty(0, dtype=np.int64)
        r_counts = np.empty(0, dtype=np.int32)
    else:
        r_v0 = rep_keys_v0
        r_v1 = rep_keys_v1
        r_counts = rep_counts

    cand_k, cand_r, cand_c, cand_fin, count = _generate_ordered_moves_numba(
        terrain,
        turns_grid,
        cost_grid,
        elev_quality,
        pos,
        scores,
        total_finished,
        p,
        root_beam,
    )
    actions = np.empty(count, dtype=np.int32)
    values = np.empty(count, dtype=np.float32)

    alpha = -1e9
    beta = 1e9

    for i in range(count):
        k = int(cand_k[i])
        rt = int(cand_r[i])
        ct = int(cand_c[i])
        is_fin = int(cand_fin[i])

        if is_fin == 1:
            actions[i] = k * NUM_TARGETS + FINISH_TARGET
        else:
            r_can = rt if p == 0 else (9 - rt)
            c_can = ct if p == 0 else (9 - ct)
            actions[i] = k * NUM_TARGETS + (r_can * 10 + c_can)

        is_cap = False
        if is_fin == 0:
            for ok in range(6):
                if pos[1 - p, ok, 0] == rt and pos[1 - p, ok, 1] == ct:
                    is_cap = True
                    break

        next_pos, next_scores, next_fin, next_ply, is_done, winner = (
            _step_and_check_terminal_numba(
                pos, scores, total_finished, ply_count, p, k, rt, ct, is_fin
            )
        )

        is_rep_draw = False
        is_rep_twice = False
        if is_done == 0 and is_fin == 0 and not is_cap and len(r_v0) > 0:
            v0, v1 = compute_position_key(next_pos, p_status, 1 - p)
            for r_idx in range(len(r_v0)):
                if r_v0[r_idx] == v0 and r_v1[r_idx] == v1:
                    if r_counts[r_idx] >= 2:
                        is_rep_draw = True
                    elif r_counts[r_idx] == 1:
                        is_rep_twice = True
                    break

        if is_rep_draw:
            val = 0.0
        elif is_done == 1:
            if winner == p:
                val = 9500.0 - float(next_ply)
            elif winner == (1 - p):
                val = -9500.0 + float(next_ply)
            else:
                val = 0.0
        else:
            # Evaluate every root candidate accurately while passing alpha for pruning deeper branches
            val = -_alphabeta_negamax_numba(
                terrain,
                turns_grid,
                cost_grid,
                elev_quality,
                next_pos,
                next_scores,
                next_fin,
                next_ply,
                1 - p,
                depth - 1,
                -beta,
                -(alpha - 15.0),
                inner_limits,
                p_status,
                r_v0,
                r_v1,
                r_counts,
            )
            if is_rep_twice and val > 50.0:
                val -= 120.0

        values[i] = float(val)
        if val > alpha:
            alpha = val

    return actions, values


class HeuristicAgent:
    """
    Numba-accelerated Alpha-Beta Tactical Agent for evaluating AlphaZero checkpoints
    and bootstrapping curriculum games, supporting 3 calibrated difficulty tiers:
      - 'blitz': 2-ply, root beam 10, inner [6] (~0.8 ms/move)
      - 'standard': 4-ply, root beam 14, inner [10, 7, 5] (~5.1 ms/move)
      - 'master': 6-ply, root beam 16, inner [10, 8, 7, 6, 5] (~76 ms/move)
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        depth: Optional[int] = None,
        root_beam: Optional[int] = None,
        variant: str = "standard",
        inner_limits: Optional[np.ndarray] = None,
        time_budget_ms: Optional[float] = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        norm_var = variant.lower().strip()
        if norm_var in ("fast", "2ply"):
            norm_var = "blitz"
        elif norm_var in ("deep", "6ply"):
            norm_var = "master"
        elif norm_var in ("ultra", "8ply", "grandmaster", "gm"):
            norm_var = "grandmaster"
        elif norm_var not in HEURISTIC_PRESETS:
            norm_var = "standard"

        preset = HEURISTIC_PRESETS[norm_var]
        self.variant = norm_var
        self.depth = depth if depth is not None else int(preset["depth"])
        self.root_beam = root_beam if root_beam is not None else int(preset["root_beam"])
        self.inner_limits = inner_limits if inner_limits is not None else preset["inner_limits"]
        self.time_budget_ms = time_budget_ms

    def select_action(
        self, game: TraversalGame, time_budget_ms: Optional[float] = None
    ) -> int:
        if len(game.position_counts) > 0:
            rep_keys_v0 = np.empty(len(game.position_counts), dtype=np.int64)
            rep_keys_v1 = np.empty(len(game.position_counts), dtype=np.int64)
            rep_counts = np.empty(len(game.position_counts), dtype=np.int32)
            idx = 0
            for (v0, v1), count in game.position_counts.items():
                rep_keys_v0[idx] = v0
                rep_keys_v1[idx] = v1
                rep_counts[idx] = count
                idx += 1
        else:
            rep_keys_v0 = np.empty(0, dtype=np.int64)
            rep_keys_v1 = np.empty(0, dtype=np.int64)
            rep_counts = np.empty(0, dtype=np.int32)

        budget = time_budget_ms if time_budget_ms is not None else self.time_budget_ms

        if budget is not None and budget > 0:
            # Iterative deepening search with time budget (review #6.4 & #111)
            t_start = time.perf_counter()
            best_actions = np.empty(0, dtype=np.int32)
            best_values = np.empty(0, dtype=np.float32)
            max_depth = max(2, self.depth)

            for d in range(2, max_depth + 1):
                cur_actions, cur_values = search_best_heuristic_moves_numba(
                    game.terrain,
                    game.pos,
                    game.scores,
                    int(game.total_pieces_finished),
                    int(game.ply_count),
                    int(game.current_player),
                    int(d),
                    int(self.root_beam),
                    self.inner_limits,
                    game.piece_status,
                    rep_keys_v0,
                    rep_keys_v1,
                    rep_counts,
                )
                if len(cur_actions) > 0:
                    best_actions = cur_actions
                    best_values = cur_values

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                # If an immediate decisive win was found, or budget exceeded 60%, stop deepening
                if len(best_values) > 0 and (float(np.max(best_values)) >= 8000.0 or len(best_actions) <= 1):
                    break
                if elapsed_ms >= budget * 0.6:
                    break

            actions = best_actions
            values = best_values
        else:
            # Direct fixed-depth search
            actions, values = search_best_heuristic_moves_numba(
                game.terrain,
                game.pos,
                game.scores,
                int(game.total_pieces_finished),
                int(game.ply_count),
                int(game.current_player),
                int(self.depth),
                int(self.root_beam),
                self.inner_limits,
                game.piece_status,
                rep_keys_v0,
                rep_keys_v1,
                rep_counts,
            )

        if len(actions) == 0:
            legal = game.get_legal_actions()
            return int(self.rng.choice(legal))

        max_val = float(np.max(values))
        # Break near-ties (within 5.0 eval points) randomly for variety
        best_mask = values >= (max_val - 5.0)
        best_actions = actions[best_mask]
        return int(self.rng.choice(best_actions))


def evaluate_against_heuristic(
    model: TraversalDualStemNet,
    device: torch.device,
    num_games: int = 40,
    num_simulations: int = 48,
    shaping_weight: float = 1.0,
    v_tac_weight: float = 0.30,
    terrain_mode: str = "random",
    heuristic_variant: str = "standard",
    rng: Optional[np.random.Generator] = None,
    elo_db: Optional[EloDatabase] = None,
    model_id: Optional[str] = None,
    opponent_id: str = "heuristic:standard",
) -> Dict[str, float]:
    """
    Evaluates `model` (using deterministic MCTS tau->0, no Dirichlet noise) against
    `HeuristicAgent` in lockstep GPU batches across `num_games` paired symmetric games
    (for every terrain, AI plays both Player 0 and Player 1 to eliminate first-player & terrain bias).

    Results are recorded in `elo_db` (if provided) under `source="training_eval"`.
    """
    if num_games <= 0:
        return {
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "win_rate": 0.0,
            "total_games": 0.0,
        }

    if rng is None:
        rng = np.random.default_rng()

    # Ensure even number of games for paired terrain symmetry
    if num_games % 2 != 0:
        num_games += 1

    num_pairs = num_games // 2
    games: List[TraversalGame] = []
    ai_players: List[int] = []

    for _ in range(num_pairs):
        shared_terrain = TraversalGame(
            terrain_mode=terrain_mode,
            seed=int(rng.integers(0, 1_000_000_000)),
        ).terrain
        games.append(TraversalGame(terrain=shared_terrain))
        ai_players.append(0)  # Game 2p: AI is Player 0
        games.append(TraversalGame(terrain=shared_terrain))
        ai_players.append(1)  # Game 2p+1: AI is Player 1

    model_was_training = model.training
    model.eval()

    # Pre-encode static tensors once per game at Move 0 (Section 5.2)
    all_s = []
    for g in games:
        s0, s1 = encode_both_static_tensors(g.terrain)
        all_s.append(s0)
        all_s.append(s1)

    f_static_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        t_all = torch.from_numpy(np.stack(all_s, axis=0)).to(device)
        f_all = model.encode_static(t_all)  # (2 * num_games, 32, 10, 10)
        for i in range(num_games):
            f_static_pairs.append((f_all[2 * i : 2 * i + 1], f_all[2 * i + 1 : 2 * i + 2]))

    heuristic = HeuristicAgent(rng=rng, variant=heuristic_variant)
    active_indices = list(range(num_games))

    while active_indices:
        # 1. Step HeuristicAgent on games where it's the heuristic's turn
        heur_indices = [
            i for i in active_indices
            if games[i].current_player != ai_players[i]
        ]
        for i in heur_indices:
            action = heuristic.select_action(games[i])
            games[i].step(action)

        active_indices = [i for i in active_indices if not games[i].done]
        if not active_indices:
            break

        # 2. Step AI via batched MCTS on games where it's the AI's turn
        ai_indices = [
            i for i in active_indices
            if games[i].current_player == ai_players[i]
        ]
        if ai_indices:
            ai_games = [games[i] for i in ai_indices]
            ai_f_statics = [f_static_pairs[i] for i in ai_indices]

            roots = run_batched_mcts_roots(
                model=model,
                device=device,
                games=ai_games,
                f_static_per_game=ai_f_statics,
                num_simulations=num_simulations,
                add_dirichlet_noise=False,
                shaping_weight=shaping_weight,
                v_tac_weight=v_tac_weight,
                rng=rng,
            )

            for i, root in zip(ai_indices, roots):
                action, _ = root.select_move_by_temperature(temperature=0.0, rng=rng)
                games[i].step(action)

        active_indices = [i for i in active_indices if not games[i].done]

    if model_was_training:
        model.train()

    wins = 0
    draws = 0
    losses = 0

    for i, g in enumerate(games):
        ai_p = ai_players[i]
        if g.winner == ai_p:
            wins += 1
        elif g.winner == -1:
            draws += 1
        else:
            losses += 1

        if elo_db is not None and model_id is not None:
            p1_id = model_id if ai_p == 0 else opponent_id
            p2_id = opponent_id if ai_p == 0 else model_id
            elo_db.record_match(
                p1_id=p1_id,
                p2_id=p2_id,
                winner=g.winner,
                ply_count=g.ply_count,
                scores=[int(g.scores[0]), int(g.scores[1])],
                termination_reason=g.termination_reason,
                terrain_mode=terrain_mode,
                source="training_eval",
            )

    total = max(1, len(games))
    win_rate = float((wins + 0.5 * draws) / total)
    res: Dict[str, float] = {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "win_rate": win_rate,
        "total_games": float(total),
    }
    if elo_db is not None and model_id is not None:
        p_rec = elo_db.players.get(model_id)
        if p_rec is not None:
            res["elo_rating"] = float(p_rec.rating)

    return res


def evaluate_checkpoint_arena(
    candidate_model: TraversalDualStemNet,
    incumbent_model: TraversalDualStemNet,
    device: torch.device,
    num_games: int = 40,
    num_simulations: int = 48,
    shaping_weight: float = 1.0,
    v_tac_weight: float = 0.30,
    terrain_mode: str = "random",
    rng: Optional[np.random.Generator] = None,
    elo_db: Optional[EloDatabase] = None,
    candidate_id: Optional[str] = None,
    incumbent_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Pits `candidate_model` against `incumbent_model` across `num_games` paired symmetric games
    in lockstep GPU batches (for every terrain, Candidate plays both Player 0 and Player 1).

    Results are recorded in `elo_db` (if provided) under `source="arena_gating"`.
    """
    if num_games <= 0:
        return {
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "score_rate": 0.0,
            "total_games": 0.0,
        }

    if rng is None:
        rng = np.random.default_rng()

    if num_games % 2 != 0:
        num_games += 1

    num_pairs = num_games // 2
    games: List[TraversalGame] = []
    cand_players: List[int] = []

    for _ in range(num_pairs):
        shared_terrain = TraversalGame(
            terrain_mode=terrain_mode,
            seed=int(rng.integers(0, 1_000_000_000)),
        ).terrain
        games.append(TraversalGame(terrain=shared_terrain))
        cand_players.append(0)  # Game 2p: Candidate is Player 0
        games.append(TraversalGame(terrain=shared_terrain))
        cand_players.append(1)  # Game 2p+1: Candidate is Player 1

    cand_was_training = candidate_model.training
    inc_was_training = incumbent_model.training
    candidate_model.eval()
    incumbent_model.eval()

    # Pre-encode static tensors with both models
    all_s = []
    for g in games:
        s0, s1 = encode_both_static_tensors(g.terrain)
        all_s.append(s0)
        all_s.append(s1)

    cand_f_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    inc_f_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        t_all = torch.from_numpy(np.stack(all_s, axis=0)).to(device)
        cand_f_all = candidate_model.encode_static(t_all)
        inc_f_all = incumbent_model.encode_static(t_all)
        for i in range(num_games):
            cand_f_pairs.append((cand_f_all[2 * i : 2 * i + 1], cand_f_all[2 * i + 1 : 2 * i + 2]))
            inc_f_pairs.append((inc_f_all[2 * i : 2 * i + 1], inc_f_all[2 * i + 1 : 2 * i + 2]))

    active_indices = list(range(num_games))

    while active_indices:
        # 1. Step Candidate moves for active games where it's the candidate's turn
        cand_indices = [
            i for i in active_indices
            if games[i].current_player == cand_players[i]
        ]
        if cand_indices:
            c_games = [games[i] for i in cand_indices]
            c_f_statics = [cand_f_pairs[i] for i in cand_indices]
            roots = run_batched_mcts_roots(
                model=candidate_model,
                device=device,
                games=c_games,
                f_static_per_game=c_f_statics,
                num_simulations=num_simulations,
                add_dirichlet_noise=False,
                shaping_weight=shaping_weight,
                v_tac_weight=v_tac_weight,
                rng=rng,
            )
            for i, root in zip(cand_indices, roots):
                action, _ = root.select_move_by_temperature(temperature=0.0, rng=rng)
                games[i].step(action)

        active_indices = [i for i in active_indices if not games[i].done]
        if not active_indices:
            break

        # 2. Step Incumbent moves for active games where it's the incumbent's turn
        inc_indices = [
            i for i in active_indices
            if games[i].current_player != cand_players[i]
        ]
        if inc_indices:
            inc_games = [games[i] for i in inc_indices]
            inc_f_statics = [inc_f_pairs[i] for i in inc_indices]
            roots = run_batched_mcts_roots(
                model=incumbent_model,
                device=device,
                games=inc_games,
                f_static_per_game=inc_f_statics,
                num_simulations=num_simulations,
                add_dirichlet_noise=False,
                shaping_weight=shaping_weight,
                v_tac_weight=v_tac_weight,
                rng=rng,
            )
            for i, root in zip(inc_indices, roots):
                action, _ = root.select_move_by_temperature(temperature=0.0, rng=rng)
                games[i].step(action)

        active_indices = [i for i in active_indices if not games[i].done]

    if cand_was_training:
        candidate_model.train()
    if inc_was_training:
        incumbent_model.train()

    wins = 0
    draws = 0
    losses = 0

    for i, g in enumerate(games):
        c_p = cand_players[i]
        if g.winner == c_p:
            wins += 1
        elif g.winner == -1:
            draws += 1
        else:
            losses += 1

        if elo_db is not None and candidate_id is not None and incumbent_id is not None:
            p1_id = candidate_id if c_p == 0 else incumbent_id
            p2_id = incumbent_id if c_p == 0 else candidate_id
            elo_db.record_match(
                p1_id=p1_id,
                p2_id=p2_id,
                winner=g.winner,
                ply_count=g.ply_count,
                scores=[int(g.scores[0]), int(g.scores[1])],
                termination_reason=g.termination_reason,
                terrain_mode=terrain_mode,
                source="arena_gating",
            )

    total = max(1, len(games))
    score_rate = float((wins + 0.5 * draws) / total)
    res: Dict[str, float] = {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "score_rate": score_rate,
        "total_games": float(total),
    }
    if elo_db is not None and candidate_id is not None:
        p_rec = elo_db.players.get(candidate_id)
        if p_rec is not None:
            res["elo_rating"] = float(p_rec.rating)

    return res


# ---------------------------------------------------------------------------
# 4. Main AlphaZero Training Loop (Section 6.3)
# ---------------------------------------------------------------------------


def _resolve_experiment_dir(
    checkpoint_dir: str,
    resume_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Resolves `(run_dir, resolved_run_name, resolved_resume_file)`:
      - If `resume_path` is provided:
          * If it is a directory, resumes from `latest.pt` (or `best.pt`) inside it.
          * If it is a file, resumes in that file's parent directory (unless `run_name` overrides).
      - If `resume_path` is None:
          * Creates a new experiment subfolder inside `checkpoint_dir`.
          * If `run_name` is None, automatically increments `train_v1`, `train_v2`, ...
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_path:
        if os.path.isdir(resume_path):
            run_dir = os.path.normpath(resume_path)
            cand_latest = os.path.join(run_dir, "latest.pt")
            cand_best = os.path.join(run_dir, "best.pt")
            resolved_resume = cand_latest if os.path.isfile(cand_latest) else cand_best
        else:
            resolved_resume = resume_path
            parent = os.path.dirname(os.path.normpath(resume_path))
            if run_name:
                run_dir = os.path.join(checkpoint_dir, run_name)
            elif parent and parent != ".":
                run_dir = parent
            else:
                run_dir = checkpoint_dir
        resolved_name = run_name if run_name else os.path.basename(os.path.normpath(run_dir))
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, resolved_name, resolved_resume

    if not run_name:
        max_v = 0
        for entry in os.listdir(checkpoint_dir):
            full_p = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(full_p):
                m = re.match(r"^train_v(\d+)$", entry)
                if m:
                    max_v = max(max_v, int(m.group(1)))
        run_name = f"train_v{max_v + 1}"

    run_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name, None


def train_alphazero(
    num_iterations: int = 20,
    games_per_iter: int = 8,
    mcts_sims: int = 96,
    temp_threshold_ply: int = 30,
    temp_floor: float = 0.0,
    heuristic_fraction: float = 0.25,
    heuristic_anneal_iters: int = 0,
    heuristic_imitation_weight: float = 0.50,
    train_steps_per_iter: int = 60,
    target_reuse: float = 3.0,
    value_head: str = "gap",
    encoder_version: str = "v2",
    q_blend: float = 0.0,
    value_target: str = "raw",
    shaping_weight: float = 1.0,
    shaping_anneal_iters: int = 0,
    v_tac_weight: float = 0.30,
    v_tac_anneal_iters: int = 0,
    dirichlet_alpha: float = 0.18,
    dirichlet_eps: float = 0.25,
    subtree_reuse: bool = True,
    batch_size: int = 64,
    replay_capacity: int = 40000,
    eval_games: int = 20,
    eval_sims: int = 48,
    eval_interval: int = 5,
    gating: str = "none",
    gating_threshold: float = 0.55,
    elo_db: Optional[str] = "elo_ratings.json",
    eval_opponents: str = "heuristic",
    save_buffer: bool = False,
    fresh_buffer: bool = False,
    lr_init: float = 1e-3,
    lr_min: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    run_name: Optional[str] = None,
    resume_path: Optional[str] = None,
    init_weights_path: Optional[str] = None,
    aim_repo: str = ".",
    experiment_name: str = "traversal_alphazero",
    seed: int = 42,
) -> TraversalDualStemNet:
    run_dir, resolved_run_name, resolved_resume = _resolve_experiment_dir(
        checkpoint_dir=checkpoint_dir,
        resume_path=resume_path,
        run_name=run_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    start_iter = 1
    best_win_rate = 0.0
    ckpt_meta: Dict[str, Any] = {}

    resolved_init_file: Optional[str] = None
    if init_weights_path:
        if os.path.isdir(init_weights_path):
            cand_b = os.path.join(init_weights_path, "best.pt")
            cand_l = os.path.join(init_weights_path, "latest.pt")
            resolved_init_file = cand_b if os.path.isfile(cand_b) else cand_l
        elif os.path.isfile(init_weights_path):
            resolved_init_file = init_weights_path
        else:
            raise FileNotFoundError(f"Checkpoint file not found for --init-weights: {init_weights_path}")

    if resolved_init_file:
        model, _ = load_model_from_checkpoint(resolved_init_file, device=device)
        best_model = TraversalDualStemNet(
            value_head_type=model.value_head_type,
            dynamic_channels=model.dynamic_channels,
            encoder_version=model.encoder_version,
        ).to(device)
        best_model.load_state_dict(model.state_dict())
        optimizer = AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
        start_iter = 1
        best_win_rate = 0.0
        print(
            f"[Traversal AlphaZero] Initialized weights from {resolved_init_file} ({model.value_head_type} value head, "
            f"encoder {model.encoder_version} ({model.dynamic_channels}ch)) for new run '{resolved_run_name}' -> saving in {run_dir}"
        )
    elif resolved_resume and os.path.isfile(resolved_resume):
        model, ckpt_meta = load_model_from_checkpoint(resolved_resume, device=device)
        best_model = TraversalDualStemNet(
            value_head_type=model.value_head_type,
            dynamic_channels=model.dynamic_channels,
            encoder_version=model.encoder_version,
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
        if "optimizer_state_dict" in ckpt_meta:
            optimizer.load_state_dict(ckpt_meta["optimizer_state_dict"])
        start_iter = int(ckpt_meta.get("iteration", 0)) + 1
        best_win_rate = float(ckpt_meta.get("best_win_rate", ckpt_meta.get("heuristic_win_rate", 0.0)))
        print(
            f"[Traversal AlphaZero] Resumed ({model.value_head_type} value head, encoder {model.encoder_version} ({model.dynamic_channels}ch)) from {resolved_resume} at iteration {start_iter} "
            f"(best_win_rate={best_win_rate*100:.1f}%) -> saving in {run_dir}"
        )
    else:
        model = TraversalDualStemNet(value_head_type=value_head, encoder_version=encoder_version).to(device)
        best_model = TraversalDualStemNet(value_head_type=value_head, encoder_version=encoder_version).to(device)
        optimizer = AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
        print(f"[Traversal AlphaZero] Starting new experiment '{resolved_run_name}' ({value_head} value head, encoder {model.encoder_version} ({model.dynamic_channels}ch)) -> saving in {run_dir}")

    # Seed RNG correctly to avoid repeating earlier games on resume (review #5)
    if "np_rng_state" in ckpt_meta:
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt_meta["np_rng_state"]
        print("[Traversal AlphaZero] Restored NumPy RNG state from checkpoint.")
    else:
        # If no saved RNG state, seed with seed + (start_iter - 1) so resumed sequence is unique
        resumed_seed = seed + (start_iter - 1) * 1000
        rng = np.random.default_rng(resumed_seed)
        if start_iter > 1:
            print(f"[Traversal AlphaZero] Resumed RNG seeded with {resumed_seed} (seed + (start_iter-1)*1000).")

    if "torch_rng_state" in ckpt_meta:
        try:
            torch.set_rng_state(ckpt_meta["torch_rng_state"])
            print("[Traversal AlphaZero] Restored PyTorch RNG state from checkpoint.")
        except Exception:
            torch.manual_seed(seed + (start_iter - 1) * 1000)
    else:
        torch.manual_seed(seed + (start_iter - 1) * 1000)

    # Initialize scheduler spanning the total trajectory
    total_grad_steps = max(1, (start_iter - 1 + num_iterations) * train_steps_per_iter)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_grad_steps, eta_min=lr_min)

    if "scheduler_state_dict" in ckpt_meta:
        try:
            scheduler.load_state_dict(ckpt_meta["scheduler_state_dict"])
            scheduler.T_max = max(getattr(scheduler, "T_max", total_grad_steps), total_grad_steps)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = scheduler.get_last_lr()[i]
            print(f"[Traversal AlphaZero] Restored scheduler state (current LR: {optimizer.param_groups[0]['lr']:.2e}, T_max: {scheduler.T_max})")
        except Exception as exc:
            print(f"[Traversal AlphaZero] Warning: could not restore scheduler state ({exc}); advancing to start_iter.")
            past_steps = (start_iter - 1) * train_steps_per_iter
            for _ in range(past_steps):
                scheduler.step()
    elif start_iter > 1:
        # Fallback for older checkpoints: advance scheduler to iteration start_iter
        past_steps = (start_iter - 1) * train_steps_per_iter
        for _ in range(past_steps):
            scheduler.step()
        print(f"[Traversal AlphaZero] Advanced scheduler by {past_steps} steps (current LR: {optimizer.param_groups[0]['lr']:.2e})")

    best_ckpt_path = os.path.join(run_dir, "best.pt")
    if os.path.isfile(best_ckpt_path):
        best_model, _ = load_model_from_checkpoint(best_ckpt_path, device=device)
    else:
        best_model.load_state_dict(model.state_dict())

    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    buffer_path = os.path.join(run_dir, "replay_buffer.pt")
    load_buffer_path = buffer_path
    if not fresh_buffer and not resolved_init_file and not os.path.isfile(load_buffer_path) and resolved_resume:
        resume_dir = os.path.dirname(os.path.normpath(resolved_resume))
        cand_buf = os.path.join(resume_dir, "replay_buffer.pt")
        if os.path.isfile(cand_buf):
            load_buffer_path = cand_buf
    if not fresh_buffer and (save_buffer or os.path.isfile(load_buffer_path)):
        if os.path.isfile(load_buffer_path):
            try:
                replay_buffer.load_state_dict(torch.load(load_buffer_path, weights_only=False, map_location="cpu"))
                print(f"[Traversal AlphaZero] Restored replay buffer with {len(replay_buffer)} samples from {load_buffer_path}")
            except Exception as exc:
                print(f"[Traversal AlphaZero] Warning: failed to load replay buffer from {load_buffer_path} ({exc})")
    elif fresh_buffer:
        print("[Traversal AlphaZero] Starting with fresh empty replay buffer (--fresh-buffer).")

    elo_database: Optional[EloDatabase] = None
    if elo_db and elo_db.lower() not in ("", "none", "null"):
        elo_database = EloDatabase(db_path=elo_db)

    aim_run = AimRun(repo=aim_repo, experiment=experiment_name)
    try:
        aim_run.name = resolved_run_name
    except Exception:
        pass
    aim_run["hparams"] = {
        "run_name": resolved_run_name,
        "run_dir": run_dir,
        "num_iterations": num_iterations,
        "games_per_iter": games_per_iter,
        "mcts_sims": mcts_sims,
        "temp_threshold_ply": temp_threshold_ply,
        "temp_floor": temp_floor,
        "heuristic_fraction": heuristic_fraction,
        "heuristic_anneal_iters": heuristic_anneal_iters,
        "heuristic_imitation_weight": heuristic_imitation_weight,
        "train_steps_per_iter": train_steps_per_iter,
        "target_reuse": target_reuse,
        "value_head": model.value_head_type,
        "encoder_version": getattr(model, "encoder_version", "v1"),
        "dynamic_channels": getattr(model, "dynamic_channels", 34),
        "q_blend": q_blend,
        "value_target": value_target,
        "shaping_weight": shaping_weight,
        "shaping_anneal_iters": shaping_anneal_iters,
        "v_tac_weight": v_tac_weight,
        "v_tac_anneal_iters": v_tac_anneal_iters,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_eps": dirichlet_eps,
        "subtree_reuse": subtree_reuse,
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "eval_games": eval_games,
        "eval_sims": eval_sims,
        "eval_interval": eval_interval,
        "gating": gating,
        "gating_threshold": gating_threshold,
        "elo_db": elo_db,
        "eval_opponents": eval_opponents,
        "save_buffer": save_buffer,
        "lr_init": lr_init,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(device),
    }

    q_blend_str = f" | QBlend: {q_blend:.2f}" if q_blend > 0.0 else ""
    shaping_str = f" | Shaping: {shaping_weight:.2f}->0 ({shaping_anneal_iters} iters)" if shaping_anneal_iters > 0 else f" | Shaping: {shaping_weight:.2f}"
    v_tac_str = f" | VTac: {v_tac_weight:.2f}->0 ({v_tac_anneal_iters} iters)" if v_tac_anneal_iters > 0 else f" | VTac: {v_tac_weight:.2f}"
    heur_str = f" | HeurFrac: {heuristic_fraction:.2f}->0 ({heuristic_anneal_iters} iters)" if heuristic_anneal_iters > 0 else f" | HeurFrac: {heuristic_fraction:.2f}"
    eval_str = f" | Eval: {eval_games}g@{eval_sims}s every {eval_interval}i (gating={gating})" if eval_games > 0 else " | Eval: disabled"
    print(
        f"[Traversal AlphaZero] Run: {resolved_run_name} ({run_dir}) | Device: {device} | Head: {model.value_head_type} | Target: {value_target}{q_blend_str}{shaping_str}{v_tac_str}{heur_str}{eval_str} | "
        f"Iterations: {num_iterations} | Games/iter: {games_per_iter} | "
        f"MCTS sims: {mcts_sims} | TempPly: {temp_threshold_ply} | Batch size: {batch_size} | "
        f"Target reuse: {target_reuse:.1f}x (max {train_steps_per_iter} steps)"
    )

    global_step = 0
    end_iter = start_iter + num_iterations - 1
    last_eval_win_rate: Optional[float] = best_win_rate if best_win_rate > 0 else None
    last_eval_elo: Optional[float] = ckpt_meta.get("elo_rating")
    for it in range(start_iter, end_iter + 1):
        t0 = time.time()

        if shaping_anneal_iters > 0:
            cur_shaping = max(0.0, shaping_weight * (1.0 - min(1.0, (it - 1) / max(1, shaping_anneal_iters))))
        else:
            cur_shaping = shaping_weight

        if v_tac_anneal_iters > 0:
            cur_v_tac = max(0.0, v_tac_weight * (1.0 - min(1.0, (it - 1) / max(1, v_tac_anneal_iters))))
        else:
            cur_v_tac = v_tac_weight

        if heuristic_anneal_iters > 0:
            cur_heur_progress = min(1.0, (it - 1) / max(1, heuristic_anneal_iters))
            cur_heur_frac = max(0.0, heuristic_fraction * (1.0 - cur_heur_progress))
            cur_heur_imit = max(0.0, heuristic_imitation_weight * (1.0 - cur_heur_progress))
        else:
            cur_heur_frac = heuristic_fraction
            cur_heur_imit = heuristic_imitation_weight

        # 1. Self-Play Phase
        sp_stats = generate_self_play_games(
            model=model,
            device=device,
            replay_buffer=replay_buffer,
            num_games=games_per_iter,
            num_simulations=mcts_sims,
            temp_threshold_ply=temp_threshold_ply,
            temp_floor=temp_floor,
            heuristic_fraction=cur_heur_frac,
            heuristic_imitation_weight=cur_heur_imit,
            augment_symmetry=True,
            q_blend=q_blend,
            value_target=value_target,
            shaping_weight=cur_shaping,
            v_tac_weight=cur_v_tac,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            subtree_reuse=subtree_reuse,
            rng=rng,
        )

        # 2. Gradient Update Phase (sample-reuse controlled to prevent value-head memorisation, review #1.2)
        model.train()
        loss_totals, loss_vals, loss_pols, loss_kls, loss_ents = [], [], [], [], []

        new_samples = max(1, int(sp_stats.get("samples_added", 0)))
        if target_reuse > 0.0:
            desired_samples = target_reuse * float(new_samples)
            steps_from_reuse = int(round(desired_samples / max(1, batch_size)))
            effective_steps = max(4, min(steps_from_reuse, train_steps_per_iter))
        else:
            effective_steps = min(
                train_steps_per_iter,
                max(8, (len(replay_buffer) * 2) // max(1, batch_size)),
            )
        # Cap by available buffer data during early warmup
        effective_steps = min(effective_steps, max(1, (len(replay_buffer) * 2) // max(1, batch_size)))
        actual_reuse = float((effective_steps * batch_size) / float(new_samples))

        for _ in range(effective_steps):
            if len(replay_buffer) < max(8, min(batch_size, 16)):
                break
            x_s, x_d, pi_t, mask_t, z_t = replay_buffer.sample_batch(batch_size, device, rng=rng)
            optimizer.zero_grad(set_to_none=True)
            logits, v_pred = model(x_s, x_d)
            loss, metrics = compute_alphazero_loss(logits, v_pred, pi_t, z_t, mask_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_totals.append(metrics["loss_total"])
            loss_vals.append(metrics["loss_value"])
            loss_pols.append(metrics["loss_policy"])
            if "policy_kl" in metrics:
                loss_kls.append(metrics["policy_kl"])
            if "policy_entropy" in metrics:
                loss_ents.append(metrics["policy_entropy"])

        avg_loss = float(np.mean(loss_totals)) if loss_totals else 0.0
        avg_v_loss = float(np.mean(loss_vals)) if loss_vals else 0.0
        avg_p_loss = float(np.mean(loss_pols)) if loss_pols else 0.0
        avg_kl = float(np.mean(loss_kls)) if loss_kls else 0.0
        avg_ent = float(np.mean(loss_ents)) if loss_ents else 0.0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        # 3. Evaluation Phase vs Heuristic Baseline & Opponents / Gating
        eval_ran = eval_games > 0 and (it % eval_interval == 0)
        eval_stats: Optional[Dict[str, float]] = None
        candidate_id = resolved_run_name
        arena_stats = None

        if eval_ran:
            eval_stats = evaluate_against_heuristic(
                model=model,
                device=device,
                num_games=eval_games,
                num_simulations=eval_sims,
                shaping_weight=cur_shaping,
                v_tac_weight=cur_v_tac,
                rng=rng,
                elo_db=elo_database,
                model_id=candidate_id,
                opponent_id="heuristic:standard",
            )
            last_eval_win_rate = eval_stats["win_rate"]
            last_eval_elo = eval_stats.get("elo_rating")

            # Optional: evaluate against additional opponents (e.g. past frozen checkpoints)
            opp_list = [op.strip() for op in eval_opponents.split(",") if op.strip()]
            for opp_spec in opp_list:
                if opp_spec in ("heuristic", "heuristic:standard", "heuristic:blitz", "heuristic:master"):
                    continue
                opp_path = opp_spec
                if ":" in opp_path and not os.path.isfile(opp_path):
                    opp_path = opp_path.split(":")[0]
                if not os.path.isfile(opp_path):
                    cand_p = os.path.join(checkpoint_dir, opp_spec.split(":")[0] if ":" in opp_spec else opp_spec)
                    if os.path.isfile(cand_p):
                        opp_path = cand_p
                if os.path.isfile(opp_path):
                    opp_model, _ = load_model_from_checkpoint(opp_path, device=device)
                    opp_stats = evaluate_checkpoint_arena(
                        candidate_model=model,
                        incumbent_model=opp_model,
                        device=device,
                        num_games=eval_games,
                        num_simulations=eval_sims,
                        shaping_weight=cur_shaping,
                        v_tac_weight=cur_v_tac,
                        rng=rng,
                        elo_db=elo_database,
                        candidate_id=candidate_id,
                        incumbent_id=opp_spec,
                    )
                    clean_opp_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", opp_spec)
                    aim_run.track(opp_stats["score_rate"], name=f"score_vs_{clean_opp_name}", step=it, context={"subset": "eval"})

        # Gate best.pt promotion
        promote_best = False
        if eval_ran and eval_stats is not None:
            if gating == "none":
                # AlphaZero default: promote if eval win rate is at least as good as best_win_rate
                if eval_stats["win_rate"] >= best_win_rate:
                    promote_best = True
            elif gating == "arena":
                if eval_games > 0:
                    arena_stats = evaluate_checkpoint_arena(
                        candidate_model=model,
                        incumbent_model=best_model,
                        device=device,
                        num_games=eval_games,
                        num_simulations=eval_sims,
                        shaping_weight=cur_shaping,
                        v_tac_weight=cur_v_tac,
                        rng=rng,
                        elo_db=elo_database,
                        candidate_id=candidate_id,
                        incumbent_id=f"{resolved_run_name}:best",
                    )
                    if arena_stats["score_rate"] >= gating_threshold:
                        promote_best = True
                else:
                    promote_best = True
            elif gating == "heuristic":
                if eval_stats["win_rate"] >= gating_threshold and eval_stats["win_rate"] >= best_win_rate:
                    promote_best = True
        elif eval_games == 0 and gating == "none":
            promote_best = True

        if promote_best:
            best_win_rate = max(best_win_rate, eval_stats["win_rate"] if eval_stats is not None else best_win_rate)
            best_model.load_state_dict(model.state_dict())
            torch.save(
                {
                    "iteration": it,
                    "run_name": resolved_run_name,
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "np_rng_state": rng.bit_generator.state,
                    "loss_total": avg_loss,
                    "heuristic_win_rate": eval_stats["win_rate"] if eval_stats is not None else (last_eval_win_rate or 0.0),
                    "best_win_rate": best_win_rate,
                    "elo_rating": eval_stats.get("elo_rating") if eval_stats is not None else last_eval_elo,
                },
                os.path.join(run_dir, "best.pt"),
            )
            if elo_database is not None and eval_stats is not None and "elo_rating" in eval_stats:
                best_id = f"{resolved_run_name}/best.pt"
                elo_database.get_or_create_player(best_id, kind="alphazero")
                elo_database.players[best_id].rating = eval_stats["elo_rating"]
                elo_database.players[best_id].peak_rating = max(
                    elo_database.players[best_id].peak_rating, eval_stats["elo_rating"]
                )
                elo_database._save()

        latest_path = os.path.join(run_dir, "latest.pt")
        cur_eval_win_rate = eval_stats["win_rate"] if (eval_ran and eval_stats is not None) else (last_eval_win_rate or 0.0)
        cur_eval_elo = eval_stats.get("elo_rating") if (eval_ran and eval_stats is not None) else last_eval_elo
        ckpt_payload = {
            "iteration": it,
            "run_name": resolved_run_name,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "np_rng_state": rng.bit_generator.state,
            "loss_total": avg_loss,
            "heuristic_win_rate": cur_eval_win_rate,
            "best_win_rate": best_win_rate,
            "elo_rating": cur_eval_elo,
        }
        torch.save(ckpt_payload, latest_path)
        if it % 100 == 0:
            torch.save(ckpt_payload, os.path.join(run_dir, f"iter_{it:04d}.pt"))

        if save_buffer and (it % 5 == 0 or it == end_iter):
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            replay_buffer.save(buffer_path)
            if it % 100 == 0:
                replay_buffer.save(os.path.join(run_dir, f"replay_buffer_iter_{it:04d}.pt"))

        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        elapsed = time.time() - t0

        aim_run.track(avg_loss, name="loss_total", step=it, context={"subset": "train"})
        aim_run.track(avg_v_loss, name="loss_value", step=it, context={"subset": "train"})
        aim_run.track(avg_p_loss, name="loss_policy", step=it, context={"subset": "train"})
        aim_run.track(avg_kl, name="policy_kl", step=it, context={"subset": "train"})
        aim_run.track(avg_ent, name="policy_entropy", step=it, context={"subset": "train"})
        aim_run.track(cur_lr, name="learning_rate", step=it, context={"subset": "train"})
        aim_run.track(sp_stats["avg_plies"], name="avg_plies", step=it, context={"subset": "self_play"})
        aim_run.track(sp_stats["draw_rate"], name="draw_rate", step=it, context={"subset": "self_play"})
        aim_run.track(sp_stats.get("sentinel_win_rate", 0.0), name="sentinel_win_rate", step=it, context={"subset": "self_play"})
        aim_run.track(sp_stats.get("two_finish_rate", 0.0), name="two_finish_rate", step=it, context={"subset": "self_play"})
        aim_run.track(sp_stats.get("elimination_rate", 0.0), name="elimination_rate", step=it, context={"subset": "self_play"})
        aim_run.track(len(replay_buffer), name="buffer_size", step=it, context={"subset": "self_play"})
        if "fresh_value_mse" in sp_stats:
            aim_run.track(sp_stats["fresh_value_mse"], name="fresh_value_mse", step=it, context={"subset": "self_play"})
            aim_run.track(sp_stats["fresh_value_raw_mse"], name="fresh_value_raw_mse", step=it, context={"subset": "self_play"})
            aim_run.track(sp_stats["fresh_policy_kl"], name="fresh_policy_kl", step=it, context={"subset": "self_play"})
            aim_run.track(sp_stats["fresh_value_sign_acc"], name="fresh_value_sign_acc", step=it, context={"subset": "self_play"})
            aim_run.track(sp_stats["fresh_value_corr"], name="fresh_value_corr", step=it, context={"subset": "self_play"})
        aim_run.track(actual_reuse, name="sample_reuse", step=it, context={"subset": "train"})
        aim_run.track(cur_shaping, name="mcts_shaping_weight", step=it, context={"subset": "mcts"})
        aim_run.track(cur_v_tac, name="mcts_v_tac_weight", step=it, context={"subset": "mcts"})
        aim_run.track(cur_heur_frac, name="heuristic_fraction", step=it, context={"subset": "curriculum"})
        aim_run.track(cur_heur_imit, name="heuristic_imitation_weight", step=it, context={"subset": "curriculum"})
        if eval_ran and eval_stats is not None:
            aim_run.track(eval_stats["win_rate"], name="heuristic_win_rate", step=it, context={"subset": "eval"})
            if "elo_rating" in eval_stats:
                aim_run.track(eval_stats["elo_rating"], name="elo_rating", step=it, context={"subset": "eval"})
        if arena_stats is not None:
            aim_run.track(arena_stats["score_rate"], name="arena_candidate_score_rate", step=it, context={"subset": "eval"})
        aim_run.track(ram_gb, name="ram_used_gb", step=it, context={"subset": "system"})

        fresh_str = ""
        if "fresh_value_mse" in sp_stats:
            fresh_str = f" | Fresh(V_MSE={sp_stats['fresh_value_mse']:.3f}, Sign={sp_stats['fresh_value_sign_acc']*100:.0f}%, P_KL={sp_stats['fresh_policy_kl']:.3f})"
        if eval_ran and eval_stats is not None:
            elo_part = f" (Elo {eval_stats['elo_rating']:.0f})" if "elo_rating" in eval_stats else ""
            eval_log_str = f"vsHeuristic: {eval_stats['win_rate']*100:.1f}%{elo_part} ({int(eval_stats['wins'])}W-{int(eval_stats['draws'])}D-{int(eval_stats['losses'])}L)"
        elif last_eval_win_rate is not None:
            elo_part = f" (Elo {last_eval_elo:.0f})" if last_eval_elo is not None else ""
            eval_log_str = f"vsHeuristic: [{last_eval_win_rate*100:.1f}%{elo_part}]"
        else:
            eval_log_str = f"vsHeuristic: [eval every {eval_interval} iters]"
        print(
            f"[{resolved_run_name}] Iter {it:03d}/{end_iter:03d} ({elapsed:.1f}s) | "
            f"Buf: {len(replay_buffer)} | AvgPlies: {sp_stats['avg_plies']:.1f} (Draw={sp_stats['draw_rate']*100:.0f}%) | "
            f"Loss: {avg_loss:.4f} (V={avg_v_loss:.4f}, P={avg_p_loss:.4f}, KL={avg_kl:.3f}){fresh_str} | "
            f"Reuse: {actual_reuse:.1f}x ({effective_steps} steps) | "
            f"{eval_log_str} | "
            f"LR: {cur_lr:.2e} | RAM: {ram_gb:.2f}GB"
        )

    aim_run.close()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Traversal AlphaZero AI")
    parser.add_argument("--iterations", "--num-iterations", dest="iterations", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=8, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=96, help="MCTS simulations per turn")
    parser.add_argument("--temp-ply", type=int, default=30, help="Ply threshold to transition to temp floor/0 (default 30, review #11)")
    parser.add_argument("--temp-floor", type=float, default=0.0, help="Temperature floor applied after temp-ply (default 0.0, set e.g. 0.25 to maintain late-game diversity: review #11)")
    parser.add_argument("--heuristic-frac", "--heuristic-fraction", dest="heuristic_frac", type=float, default=0.25, help="Initial fraction of self-play games vs HeuristicAgent")
    parser.add_argument(
        "--heuristic-anneal-iters",
        type=int,
        default=0,
        help="Iterations over which to linearly anneal heuristic-fraction and imitation weight to 0 (0 = constant)",
    )
    parser.add_argument(
        "--heuristic-imitation-weight",
        type=float,
        default=0.50,
        help="Initial imitation weight on heuristic turns: target = (1-w)*pi_mcts + w*one_hot(heur) (default 0.50)",
    )
    parser.add_argument("--train-steps", type=int, default=60, help="Gradient steps per iteration")
    parser.add_argument(
        "--target-reuse",
        type=float,
        default=3.0,
        help="Target sample reuse ratio (gradient samples / new samples). Capped by --train-steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--value-head",
        type=str,
        default="gap",
        choices=["gap", "flat"],
        help="Value head architecture: 'gap' (global pooling, default) or 'flat' (legacy 800->64)",
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="State encoder version: 'v2' (35 channels: blocked-dist fix, friendly attack cleanup, repetition plane) or 'v1' (34 channels: legacy)",
    )
    parser.add_argument(
        "--value-target",
        type=str,
        default="raw",
        choices=["raw", "shaped"],
        help="Terminal value target for training: 'raw' (exact ±1/0 matching MCTS terminal values, default) or 'shaped' (speed/material shaping)",
    )
    parser.add_argument(
        "--q-blend",
        type=float,
        default=0.0,
        help="Blend factor between final outcome z and MCTS root Q: z' = (1-lambda)*z + lambda*root_Q (review #1.4)",
    )
    parser.add_argument(
        "--shaping-weight",
        type=float,
        default=1.0,
        help="Initial MCTS tactical prior shaping weight (1.0 = full shaping, 0.0 = pure network priors)",
    )
    parser.add_argument(
        "--shaping-anneal-iters",
        type=int,
        default=0,
        help="Iterations over which to linearly anneal shaping-weight to 0 (0 = constant)",
    )
    parser.add_argument(
        "--v-tac-weight",
        type=float,
        default=0.30,
        help="Initial weight of tactical heuristic value anchor v_tac (0.30 = default, 0.0 = pure v_net)",
    )
    parser.add_argument(
        "--v-tac-anneal-iters",
        type=int,
        default=0,
        help="Iterations over which to linearly anneal v-tac-weight to 0 (0 = constant)",
    )
    parser.add_argument(
        "--mcts-anneal-iters",
        type=int,
        default=None,
        help="Convenience flag: linearly anneal both prior shaping and v_tac to 0 over N iterations",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.18,
        help="Root Dirichlet exploration noise alpha parameter (default 0.18, calibrated for ~55 legal moves: review #7)",
    )
    parser.add_argument(
        "--dirichlet-eps",
        type=float,
        default=0.25,
        help="Root Dirichlet exploration noise mixture weight epsilon (default 0.25)",
    )
    parser.add_argument(
        "--subtree-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse MCTS search subtree from chosen action across plies in self-play (default True, review #9)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Training mini-batch size")
    parser.add_argument("--replay-capacity", type=int, default=40000, help="Replay buffer capacity")
    parser.add_argument("--eval-games", type=int, default=20, help="Arena games vs heuristic per iter (default 20)")
    parser.add_argument("--eval-sims", type=int, default=48, help="MCTS sims during evaluation")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N iterations (default 5)")
    parser.add_argument(
        "--gating",
        type=str,
        default="none",
        choices=["none", "arena", "heuristic"],
        help="Model promotion gating: 'none' (AlphaZero standard, default), 'arena' (candidate vs incumbent best_model), or 'heuristic'",
    )
    parser.add_argument(
        "--gating-threshold",
        type=float,
        default=0.55,
        help="Score rate threshold (win_rate + 0.5*draw_rate) to promote candidate model in gating (default 0.55)",
    )
    parser.add_argument(
        "--elo-db",
        type=str,
        default="elo_ratings.json",
        help="Path to Elo ratings database JSON file (default 'elo_ratings.json'). Set empty or 'none' to disable.",
    )
    parser.add_argument(
        "--eval-opponents",
        type=str,
        default="heuristic",
        help="Comma-separated list of opponents to evaluate against (e.g. 'heuristic', or 'heuristic,checkpoints/train_v4/best.pt')",
    )
    parser.add_argument(
        "--save-buffer",
        action="store_true",
        default=False,
        help="Persist replay buffer to disk across iterations (saved as replay_buffer.pt in run_dir)",
    )
    parser.add_argument(
        "--name",
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Experiment folder name inside checkpoints/ (auto-increments train_v1, train_v2, ... if omitted)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Base directory for experiment checkpoint folders",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file or experiment folder (e.g. checkpoints/train_v1/latest.pt) to resume from",
    )
    parser.add_argument(
        "--init-weights",
        "--warmstart",
        "--load-weights",
        dest="init_weights",
        type=str,
        default=None,
        help="Path to checkpoint file or folder (e.g. checkpoints/train_v7/best.pt) to initialize weights from for a fresh run, without resuming iteration count, optimizer, or buffer",
    )
    parser.add_argument(
        "--fresh-buffer",
        action="store_true",
        default=False,
        help="Start with an empty replay buffer even if a saved replay buffer exists in the resume directory",
    )
    parser.add_argument(
        "--lr-init",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW (default 1e-3, recommend 3e-4 or 5e-4 for fine-tuning)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-4,
        help="Minimum learning rate for cosine scheduler (default 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shaping_anneal = args.shaping_anneal_iters
    v_tac_anneal = args.v_tac_anneal_iters
    if args.mcts_anneal_iters is not None:
        if shaping_anneal == 0:
            shaping_anneal = args.mcts_anneal_iters
        if v_tac_anneal == 0:
            v_tac_anneal = args.mcts_anneal_iters

    train_alphazero(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        temp_threshold_ply=args.temp_ply,
        temp_floor=args.temp_floor,
        heuristic_fraction=args.heuristic_frac,
        heuristic_anneal_iters=args.heuristic_anneal_iters,
        heuristic_imitation_weight=args.heuristic_imitation_weight,
        train_steps_per_iter=args.train_steps,
        target_reuse=args.target_reuse,
        value_head=args.value_head,
        encoder_version=args.encoder_version,
        q_blend=args.q_blend,
        value_target=args.value_target,
        shaping_weight=args.shaping_weight,
        shaping_anneal_iters=shaping_anneal,
        v_tac_weight=args.v_tac_weight,
        v_tac_anneal_iters=v_tac_anneal,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        subtree_reuse=args.subtree_reuse,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_interval=args.eval_interval,
        gating=args.gating,
        gating_threshold=args.gating_threshold,
        elo_db=args.elo_db,
        eval_opponents=args.eval_opponents,
        save_buffer=args.save_buffer,
        fresh_buffer=args.fresh_buffer,
        lr_init=args.lr_init,
        lr_min=args.lr_min,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        resume_path=args.resume,
        init_weights_path=args.init_weights,
        seed=args.seed,
    )
