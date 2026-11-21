"""
AlphaZero Monte Carlo Tree Search (MCTS) with Cached Static Stem Inference (Milestone 4).

Implements:
  - `MCTSNode` and `MCTS` (single-game and batched multi-game lockstep MCTS)
  - Inference caching of `F_static_p0` and `F_static_p1` at Move 0 (Section 5.2)
  - Root Dirichlet exploration noise (eps=0.25, alpha=0.3) (Section 6.1)
  - PUCT selection (c_puct=1.5) and alternating-turn sign-flipped backpropagation
  - Temperature schedule (tau=1.0 for plies < 30, tau->0 after ply 30) (Section 6.2)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import math
import numpy as np
from numba import njit
import torch

from traversal_env import (
    TraversalGame,
    ACTION_SPACE_SIZE,
    NUM_TARGETS,
    FINISH_TARGET,
    PIECE_STRENGTH,
    PIECE_SCORE,
    ORTHO_DIRS,
)
from encoder import encode_both_static_tensors, encode_dynamic_tensor
from model import TraversalDualStemNet, masked_softmax


@njit(cache=True)
def shape_tactical_priors_and_anchor(
    raw_priors: np.ndarray,
    legal_actions: np.ndarray,
    x_dyn: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Numba-accelerated tactical prior shaping and material/threat value anchor using
    the already-computed dynamic state tensor `x_dyn` of shape (34, 10, 10):
      1. Boosts MCTS policy priors for finishing moves and safe/winning captures.
      2. Dampens MCTS policy priors for moves that step onto empty squares where
         the opponent can immediately capture the piece (`opp_max_atk >= new_def`).
      3. Computes a bounded tactical value anchor `v_tac` in [-1.0, 1.0] reflecting
         score difference, active piece material, immediate capture gains, and race progress.
    """
    num_legal = len(legal_actions)
    priors = raw_priors.astype(np.float32).copy()

    my_occ = np.full((10, 10), -1, dtype=np.int32)
    opp_occ = np.full((10, 10), -1, dtype=np.int32)
    my_pos_r = np.full(6, -1, dtype=np.int32)
    my_pos_c = np.full(6, -1, dtype=np.int32)
    opp_pos_r = np.full(6, -1, dtype=np.int32)
    opp_pos_c = np.full(6, -1, dtype=np.int32)

    my_active_str = 0.0
    opp_active_str = 0.0
    my_best_row1 = 0
    my_best_row2 = 0
    opp_best_row1 = 0
    opp_best_row2 = 0

    for k in range(6):
        s_k = float(PIECE_STRENGTH[k])
        for r in range(10):
            for c in range(10):
                if x_dyn[k, r, c] > 0.5:
                    my_occ[r, c] = k
                    my_pos_r[k] = r
                    my_pos_c[k] = c
                    my_active_str += s_k
                    if r > my_best_row1:
                        my_best_row2 = my_best_row1
                        my_best_row1 = r
                    elif r > my_best_row2:
                        my_best_row2 = r
                if x_dyn[6 + k, r, c] > 0.5:
                    opp_occ[r, c] = k
                    opp_pos_r[k] = r
                    opp_pos_c[k] = c
                    opp_active_str += s_k
                    opp_prog = 9 - r
                    if opp_prog > opp_best_row1:
                        opp_best_row2 = opp_best_row1
                        opp_best_row1 = opp_prog
                    elif opp_prog > opp_best_row2:
                        opp_best_row2 = opp_prog

    my_score = float(np.round(x_dyn[30, 0, 0] * 6.0))
    opp_score = float(np.round(x_dyn[31, 0, 0] * 6.0))
    total_finished = int(np.round(x_dyn[32, 0, 0] * 2.0))

    # Check which friendly pieces are currently threatened at their origin
    my_threatened = np.zeros(6, dtype=np.uint8)
    threatened_str_sum = 0.0
    for k in range(6):
        r0 = my_pos_r[k]
        c0 = my_pos_c[k]
        if r0 >= 0 and c0 >= 0:
            opp_atk0 = int(np.round(x_dyn[26, r0, c0] * 10.0))
            my_def0 = int(np.round(x_dyn[25, r0, c0] * 7.0))
            if opp_atk0 >= my_def0 and my_def0 > 0:
                my_threatened[k] = 1
                threatened_str_sum += float(PIECE_STRENGTH[k])

    best_capture_gain = 0.0
    has_immediate_win = False
    can_finish_max_score = 0.0

    for m in range(num_legal):
        act = int(legal_actions[m])
        k = act // NUM_TARGETS
        t = act % NUM_TARGETS
        s_k = int(PIECE_STRENGTH[k])

        if t == FINISH_TARGET:
            p_score = float(PIECE_SCORE[k])
            if total_finished == 0 and k >= 4:
                # Sentinel instant win
                priors[m] = priors[m] * 20.0 + 0.50
                has_immediate_win = True
                can_finish_max_score = max(can_finish_max_score, 4.5)
            elif total_finished >= 1 and (my_score + p_score > opp_score):
                # Game-winning 2nd finish
                priors[m] = priors[m] * 15.0 + 0.40
                has_immediate_win = True
                can_finish_max_score = max(can_finish_max_score, 4.5)
            elif total_finished >= 1 and (my_score + p_score < opp_score):
                # Suicidal 2nd finish: terminates game with lower score and hands win to opponent!
                priors[m] = priors[m] * 0.01
            elif total_finished >= 1 and (my_score + p_score == opp_score):
                # 2nd finish resulting in a tie/draw
                priors[m] = priors[m] * 1.2 + 0.05
                can_finish_max_score = max(can_finish_max_score, 0.5)
            else:
                # 1st finish (Explorer or Vanguard): score points, continue game, disarm opponent Sentinel
                priors[m] = priors[m] * (3.5 + 0.5 * float(s_k)) + 0.12
                can_finish_max_score = max(can_finish_max_score, p_score)
        else:
            rt = t // 10
            ct = t % 10

            # Compute friendly support at destination (excluding moving piece k itself)
            adj_friends = 0
            for d in range(4):
                nr = rt + ORTHO_DIRS[d, 0]
                nc = ct + ORTHO_DIRS[d, 1]
                if 0 <= nr < 10 and 0 <= nc < 10:
                    fk = my_occ[nr, nc]
                    if fk != -1 and fk != k:
                        adj_friends += 1
            new_def = s_k + adj_friends
            opp_atk_dest = int(np.round(x_dyn[26, rt, ct] * 10.0))

            opp_k = opp_occ[rt, ct]
            if opp_k != -1:
                # Capturing enemy piece `opp_k`
                s_opp = int(PIECE_STRENGTH[opp_k])
                # Check if any other opponent piece can reach (rt, ct) to recapture.
                # Since opp_k occupies (rt, ct), it cannot attack its own square; any attack in
                # opp_atk_dest (Dyn 26) comes exclusively from other opponent pieces ok != opp_k.
                is_recapturable = opp_atk_dest >= new_def
                net_gain = float(s_opp - s_k) if is_recapturable else float(s_opp)
                if net_gain > best_capture_gain:
                    best_capture_gain = net_gain

                opp_prog = 9 - rt
                is_opp_sent_danger = (total_finished == 0) and (opp_k >= 4) and (opp_prog >= 5)
                # Check if opponent piece opp_k is threatening a game-ending finish on its next turn
                is_opp_win_danger = (
                    (x_dyn[18 + opp_k, rt, ct] > 0.5)
                    and (
                        (total_finished == 0 and opp_k >= 4)
                        or (total_finished >= 1 and (opp_score + float(PIECE_SCORE[opp_k]) > my_score))
                    )
                )

                if is_opp_sent_danger or is_opp_win_danger:
                    # Stopping a game-ending finish threat near the finish line is top priority even if traded
                    priors[m] = priors[m] * (5.0 + float(max(1, opp_prog - 4))) + 0.12
                elif not is_recapturable or s_opp >= s_k:
                    # Free capture or favorable/equal trade
                    priors[m] = priors[m] * (3.0 + 0.8 * float(s_opp)) + 0.05 * float(s_opp)
                else:
                    # Unfavorable trade (e.g. sacrificing Sentinel for Explorer)
                    priors[m] = priors[m] * 0.30
            else:
                # Moving to an empty square: check if blocking an advancing dangerous enemy piece
                blocks_finish = False
                for ok in range(6):
                    sr = opp_pos_r[ok]
                    sc_col = opp_pos_c[ok]
                    if sr < 0:
                        continue
                    opp_prog = 9 - sr
                    is_sent_threat = (total_finished == 0) and (ok >= 4) and (opp_prog >= 5)
                    is_win_threat = (
                        (total_finished >= 1)
                        and (opp_score + float(PIECE_SCORE[ok]) > my_score)
                        and (opp_prog >= 6)
                    )
                    if is_sent_threat or is_win_threat:
                        if rt == sr - 1 and ct == sc_col:
                            blocks_finish = True
                            break

                if opp_atk_dest >= new_def:
                    if blocks_finish and s_k == 1:
                        priors[m] = priors[m] * 1.5 + 0.04
                    elif s_k == 1:
                        priors[m] *= 0.12
                    else:
                        priors[m] *= 0.06
                else:
                    if blocks_finish:
                        priors[m] = priors[m] * 3.0 + 0.06
                    elif my_threatened[k] == 1:
                        priors[m] = priors[m] * 2.2 + 0.025 * float(s_k)

    p_sum = float(np.sum(priors))
    if p_sum > 1e-8:
        priors /= p_sum
    else:
        priors.fill(1.0 / max(1, num_legal))

    if has_immediate_win:
        return priors, 1.0

    # Check if opponent has an unstoppable finish threat or deep Sentinel threat
    opp_finish_threat = 0.0
    for ok in range(6):
        or0 = opp_pos_r[ok]
        oc0 = opp_pos_c[ok]
        if or0 >= 0 and oc0 >= 0:
            opp_prog = 9 - or0
            # Check if current player can immediately capture the threatening opponent piece
            my_atk_on_target = int(np.round(x_dyn[24, or0, oc0] * 10.0))
            opp_def_on_target = int(np.round(x_dyn[27, or0, oc0] * 7.0))
            can_kill_threat = (my_atk_on_target >= opp_def_on_target) and (opp_def_on_target > 0)

            # Origin exit flag in x_dyn[18 + ok, or0, oc0] indicates opponent piece ok can finish
            if x_dyn[18 + ok, or0, oc0] > 0.5:
                if not can_kill_threat:
                    if total_finished == 0 and ok >= 4:
                        opp_finish_threat = max(opp_finish_threat, 4.5)
                    elif total_finished >= 1 and (opp_score + float(PIECE_SCORE[ok]) > my_score):
                        opp_finish_threat = max(opp_finish_threat, 4.5)
                    elif total_finished >= 1 and (opp_score + float(PIECE_SCORE[ok]) == my_score):
                        opp_finish_threat = max(opp_finish_threat, 1.0)
                    elif total_finished >= 1 and (opp_score + float(PIECE_SCORE[ok]) < my_score):
                        pass  # Suicidal for opponent: finishing gives an instant win to our side!
                    else:
                        opp_finish_threat = max(opp_finish_threat, float(PIECE_SCORE[ok]))
            elif not can_kill_threat:
                if total_finished == 0 and ok >= 4 and opp_prog >= 6:
                    opp_finish_threat = max(opp_finish_threat, 0.65 * float(opp_prog - 5))
                elif (
                    total_finished >= 1
                    and (opp_score + float(PIECE_SCORE[ok]) > my_score)
                    and opp_prog >= 6
                ):
                    opp_finish_threat = max(opp_finish_threat, 0.65 * float(opp_prog - 5))

    # Effective material includes both on-board strength and already-banked finish points
    eff_my_mat = my_active_str + 2.5 * my_score
    eff_opp_mat = opp_active_str + 2.5 * opp_score
    mat_diff = (
        (eff_my_mat - eff_opp_mat)
        + 0.85 * best_capture_gain
        - 0.35 * max(0.0, threatened_str_sum - best_capture_gain)
    )

    score_diff = (my_score + 0.7 * can_finish_max_score) - (opp_score + 0.7 * opp_finish_threat)
    race_diff = float((my_best_row1 + my_best_row2) - (opp_best_row1 + opp_best_row2))

    raw_adv = 0.40 * score_diff + 0.16 * mat_diff + 0.03 * race_diff
    v_tac = math.tanh(raw_adv)
    return priors, float(v_tac)


class MCTSNode:
    """
    Search tree node storing statistics strictly over the legal actions at state `game`.
    `W` and `Q = W / N` are stored from the perspective of `game.current_player` (the player
    choosing the action at this node).
    """

    __slots__ = (
        "game",
        "legal_actions",
        "P",
        "N",
        "W",
        "V",
        "total_N",
        "children",
        "is_expanded",
        "x_dyn",
    )

    def __init__(self, game: TraversalGame) -> None:
        self.game = game
        self.legal_actions: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.N: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.V: float = 0.0
        self.total_N: int = 0
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False
        self.x_dyn: Optional[np.ndarray] = None

    def expand(
        self, legal_actions: np.ndarray, priors: np.ndarray, value: float = 0.0
    ) -> None:
        num_legal = len(legal_actions)
        self.legal_actions = legal_actions
        self.P = priors.astype(np.float32, copy=True)
        self.N = np.zeros(num_legal, dtype=np.int32)
        self.W = np.zeros(num_legal, dtype=np.float32)
        self.V = float(value)
        self.total_N = 0
        self.is_expanded = True

    def add_dirichlet_noise(
        self, eps: float = 0.25, alpha: float = 0.3, rng: Optional[np.random.Generator] = None
    ) -> None:
        if not self.is_expanded or self.P is None or len(self.P) <= 1:
            return
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.dirichlet(np.full(len(self.P), alpha, dtype=np.float64)).astype(np.float32)
        self.P = (1.0 - eps) * self.P + eps * noise

    def select_action_idx(self, c_puct: float = 1.5, fpu_reduction: float = 0.20) -> int:
        """
        Selects the index into `self.legal_actions` maximizing:
          PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
        Uses First Play Urgency (FPU) relative to the node's running value estimate so
        unvisited actions at disadvantageous nodes do not default to 0.0 and flatten search.
        """
        assert self.N is not None and self.W is not None and self.P is not None
        sqrt_total = math.sqrt(max(1, self.total_N))
        if self.total_N > 0:
            node_q = (self.V + float(np.sum(self.W))) / (1.0 + float(self.total_N))
            fpu_val = max(-1.0, node_q - fpu_reduction)
        else:
            fpu_val = self.V
        q_vals = np.where(self.N > 0, self.W / np.maximum(1, self.N), fpu_val)
        u_vals = c_puct * self.P * (sqrt_total / (1.0 + self.N))
        puct = q_vals + u_vals
        return int(np.argmax(puct))

    def get_visit_distribution(self) -> np.ndarray:
        """
        Returns the full 606-dim normalized visit count distribution `pi_mcts` for training.
        """
        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not self.is_expanded or self.legal_actions is None or self.N is None:
            return pi
        total = float(np.sum(self.N))
        if total > 0:
            pi[self.legal_actions] = self.N.astype(np.float32) / total
        else:
            pi[self.legal_actions] = 1.0 / len(self.legal_actions)
        return pi

    def get_root_q(self) -> float:
        """
        Returns the expected MCTS search value from the node player's perspective:
          root_Q = sum_a (N(a) / total_N) * Q(a) = sum_a W(a) / total_N
        Clamped to [-1.0, 1.0]. Falls back to self.V if total_N == 0.
        """
        if self.total_N > 0 and self.W is not None:
            q = float(np.sum(self.W)) / float(self.total_N)
            return float(np.clip(q, -1.0, 1.0))
        return float(np.clip(self.V, -1.0, 1.0))

    def select_move_by_temperature(
        self, temperature: float, rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Returns `(chosen_action, pi_mcts_606)` according to temperature schedule (Section 6.2):
          - pi_mcts_606 is the full 606-dim search policy vector.
          - If temperature <= 1e-3 (after ply 30 or evaluation): picks argmax_a N(s_root, a).
          - Otherwise: samples proportionally to N(s_root, a)^(1 / temperature).
        """
        assert self.legal_actions is not None and self.N is not None and self.P is not None
        if rng is None:
            rng = np.random.default_rng()

        pi_mcts = self.get_visit_distribution()

        if temperature <= 1e-3:
            # Break ties among max visit count using prior probability P
            max_n = np.max(self.N)
            candidates = np.flatnonzero(self.N == max_n)
            if len(candidates) == 1:
                best_idx = int(candidates[0])
            else:
                best_idx = int(candidates[np.argmax(self.P[candidates])])
            return int(self.legal_actions[best_idx]), pi_mcts

        counts = self.N.astype(np.float64)
        if abs(temperature - 1.0) > 1e-6:
            counts = np.power(counts, 1.0 / temperature)
        total = np.sum(counts)
        if total <= 0:
            probs = np.full(len(self.legal_actions), 1.0 / len(self.legal_actions))
        else:
            probs = counts / total
        chosen_idx = int(rng.choice(len(self.legal_actions), p=probs))
        return int(self.legal_actions[chosen_idx]), pi_mcts


class MCTS:
    """
    AlphaZero MCTS runner for a single game, caching `F_static` for both players at Move 0.
    """

    def __init__(
        self,
        model: TraversalDualStemNet,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_eps: float = 0.25,
        dirichlet_alpha: float = 0.18,
        shaping_weight: float = 1.0,
        v_tac_weight: float = 0.30,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.shaping_weight = float(shaping_weight)
        self.v_tac_weight = float(v_tac_weight)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Cached static stem outputs for the active game's terrain: shape (2, 1, 32, 10, 10)
        self._cached_terrain_id: Optional[int] = None
        self._f_static_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def cache_game_static(self, terrain: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and caches `(F_static_p0, F_static_p1)` once per game terrain (Section 5.2).
        """
        x_s0, x_s1 = encode_both_static_tensors(terrain)
        x_pair = np.stack([x_s0, x_s1], axis=0)  # (2, 4, 10, 10)
        with torch.inference_mode():
            t_pair = torch.from_numpy(x_pair).to(self.device)
            f_pair = self.model.encode_static(t_pair)  # (2, 32, 10, 10)
            f_s0 = f_pair[0:1]
            f_s1 = f_pair[1:2]
        self._cached_terrain_id = id(terrain)
        self._f_static_cache = (f_s0, f_s1)
        return f_s0, f_s1

    def _get_f_static(self, game: TraversalGame) -> torch.Tensor:
        if self._f_static_cache is None or self._cached_terrain_id != id(game.terrain):
            self.cache_game_static(game.terrain)
        assert self._f_static_cache is not None
        return self._f_static_cache[game.current_player]

    def _evaluate_leaf(self, node: MCTSNode) -> float:
        """
        Expands `node` using `forward_with_cached_static` and returns value `v` in [-1, 1]
        from `node.game.current_player`'s perspective.
        """
        game = node.game
        if game.done:
            return game.get_outcome(game.current_player)

        legal_mask_np = game.get_legal_mask()
        legal_actions = np.flatnonzero(legal_mask_np).astype(np.int32)
        encoder_ver = getattr(self.model, "encoder_version", "v1")
        x_dyn_np = encode_dynamic_tensor(game, version=encoder_ver)
        f_static = self._get_f_static(game)

        with torch.inference_mode():
            x_dyn_t = torch.from_numpy(x_dyn_np).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)
            logits, val = self.model.forward_with_cached_static(f_static, x_dyn_t)
            probs = masked_softmax(logits, mask_t)[0].cpu().numpy()
            v_net = float(val[0, 0].item())

        if self.shaping_weight <= 0.0 and self.v_tac_weight <= 0.0:
            # Pure neural AlphaZero fast path: no tactical prior shaping or heuristic anchor
            raw_p = probs[legal_actions]
            p_sum = float(np.sum(raw_p))
            priors = raw_p / p_sum if p_sum > 1e-8 else np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
            v = v_net
            # Immediate win shortcut: if an immediate finishing move wins the game, anchor v = 1.0
            total_finished = int(np.round(x_dyn_np[32, 0, 0] * 2.0))
            my_score = int(np.round(x_dyn_np[32, 0, 1] * 8.0))
            opp_score = int(np.round(x_dyn_np[32, 0, 2] * 8.0))
            for act_i in range(len(legal_actions)):
                act = int(legal_actions[act_i])
                if act % NUM_TARGETS == FINISH_TARGET:
                    k_pc = act // NUM_TARGETS
                    if (total_finished == 0 and k_pc >= 4) or (total_finished >= 1 and my_score + float(PIECE_SCORE[k_pc]) > opp_score):
                        v = 1.0
                        break
        else:
            shaped_priors, v_tac = shape_tactical_priors_and_anchor(
                probs[legal_actions], legal_actions, x_dyn_np
            )
            if v_tac >= 0.999:
                v = 1.0  # Exact immediate win shortcut preserved (review #2.1)
            else:
                v = (1.0 - self.v_tac_weight) * v_net + self.v_tac_weight * v_tac

            if self.shaping_weight <= 0.0:
                raw_p = probs[legal_actions]
                p_sum = float(np.sum(raw_p))
                priors = raw_p / p_sum if p_sum > 1e-8 else np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
            elif self.shaping_weight >= 1.0:
                priors = shaped_priors
            else:
                raw_p = probs[legal_actions]
                p_sum = float(np.sum(raw_p))
                raw_norm = raw_p / p_sum if p_sum > 1e-8 else np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
                priors = (1.0 - self.shaping_weight) * raw_norm + self.shaping_weight * shaped_priors
                p_sum = float(np.sum(priors))
                priors = priors / p_sum if p_sum > 1e-8 else raw_norm

        node.expand(legal_actions, priors, value=v)
        node.x_dyn = x_dyn_np
        return v

    def run(
        self,
        root_game: TraversalGame,
        add_dirichlet_noise: bool = True,
        num_simulations: Optional[int] = None,
        existing_root: Optional[MCTSNode] = None,
    ) -> MCTSNode:
        """
        Runs `num_simulations` MCTS simulations starting from `root_game` and returns the expanded root node.
        Supports subtree reuse via `existing_root` (review #9).
        """
        n_sims = num_simulations if num_simulations is not None else self.num_simulations
        if existing_root is not None:
            root = existing_root
            root.game = root_game
        else:
            root = MCTSNode(root_game.clone())

        # Expand root node if not already expanded
        if not root.is_expanded:
            self._evaluate_leaf(root)
        if add_dirichlet_noise:
            root.add_dirichlet_noise(
                eps=self.dirichlet_eps, alpha=self.dirichlet_alpha, rng=self.rng
            )

        for _ in range(n_sims):
            node = root
            search_path: List[Tuple[MCTSNode, int]] = []

            # 1. Selection: descend tree while expanded and non-terminal
            while node.is_expanded and not node.game.done:
                idx = node.select_action_idx(self.c_puct)
                search_path.append((node, idx))
                if idx not in node.children:
                    next_game = node.game.clone()
                    assert node.legal_actions is not None
                    next_game.step(int(node.legal_actions[idx]))
                    node.children[idx] = MCTSNode(next_game)
                node = node.children[idx]

            # 2. Leaf Expansion & Evaluation
            v = self._evaluate_leaf(node)

            # 3. Backpropagation: walk back up path, flipping v <- -v at each step
            for parent, action_idx in reversed(search_path):
                v = -v
                assert parent.N is not None and parent.W is not None
                parent.N[action_idx] += 1
                parent.W[action_idx] += v
                parent.total_N += 1

        return root


def run_batched_mcts_roots(
    model: TraversalDualStemNet,
    device: torch.device,
    games: List[TraversalGame],
    f_static_per_game: List[Tuple[torch.Tensor, torch.Tensor]],
    num_simulations: int,
    c_puct: float = 1.5,
    add_dirichlet_noise: bool = True,
    dirichlet_eps: float = 0.25,
    dirichlet_alpha: float = 0.18,
    shaping_weight: float = 1.0,
    v_tac_weight: float = 0.30,
    existing_roots: Optional[List[Optional[MCTSNode]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[MCTSNode]:
    """
    Runs MCTS in lockstep across a batch of active games, batching all neural network
    leaf evaluations (`forward_with_cached_static`) into a single GPU tensor call per simulation.
    Supports subtree reuse across plies via `existing_roots` (review #9).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_games = len(games)
    if num_games == 0:
        return []

    roots: List[MCTSNode] = []
    unexpanded_nodes: List[Tuple[int, MCTSNode]] = []
    for i in range(num_games):
        if existing_roots is not None and i < len(existing_roots) and existing_roots[i] is not None:
            r = existing_roots[i]
            r.game = games[i]
            roots.append(r)
            if not r.is_expanded:
                unexpanded_nodes.append((i, r))
        else:
            r = MCTSNode(games[i].clone())
            roots.append(r)
            unexpanded_nodes.append((i, r))

    def _batch_expand_nodes(nodes_to_eval: List[Tuple[int, MCTSNode]]) -> List[float]:
        values = [0.0] * len(nodes_to_eval)
        non_term_indices: List[int] = []
        f_statics: List[torch.Tensor] = []
        x_dyns: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        legal_lists: List[np.ndarray] = []

        for local_i, (game_i, node) in enumerate(nodes_to_eval):
            g = node.game
            if g.done:
                values[local_i] = g.get_outcome(g.current_player)
            else:
                non_term_indices.append(local_i)
                f_statics.append(f_static_per_game[game_i][g.current_player])
                encoder_ver = getattr(model, "encoder_version", "v1")
                x_dyns.append(encode_dynamic_tensor(g, version=encoder_ver))
                mask_np = g.get_legal_mask()
                masks.append(mask_np)
                legal_lists.append(np.flatnonzero(mask_np).astype(np.int32))

        if non_term_indices:
            with torch.inference_mode():
                f_batch = torch.cat(f_statics, dim=0)  # (B_eval, 32, 10, 10)
                x_batch = torch.from_numpy(np.stack(x_dyns, axis=0)).to(device)
                m_batch = torch.from_numpy(np.stack(masks, axis=0)).to(device)
                logits_b, val_b = model.forward_with_cached_static(f_batch, x_batch)
                probs_b = masked_softmax(logits_b, m_batch).cpu().numpy()
                vals_np = val_b.squeeze(-1).cpu().numpy()

            for k_idx, local_i in enumerate(non_term_indices):
                _, node = nodes_to_eval[local_i]
                leg = legal_lists[k_idx]
                v_net = float(vals_np[k_idx])

                if shaping_weight <= 0.0 and v_tac_weight <= 0.0:
                    # Pure neural AlphaZero fast path: no tactical prior shaping or heuristic anchor
                    raw_p = probs_b[k_idx, leg]
                    p_sum = float(np.sum(raw_p))
                    priors = raw_p / p_sum if p_sum > 1e-8 else np.full(len(leg), 1.0 / len(leg), dtype=np.float32)
                    v = v_net
                    # Immediate win shortcut: if an immediate finishing move wins the game, anchor v = 1.0
                    total_finished = int(np.round(x_dyns[k_idx][32, 0, 0] * 2.0))
                    my_score = int(np.round(x_dyns[k_idx][32, 0, 1] * 8.0))
                    opp_score = int(np.round(x_dyns[k_idx][32, 0, 2] * 8.0))
                    for act_i in range(len(leg)):
                        act = int(leg[act_i])
                        if act % NUM_TARGETS == FINISH_TARGET:
                            k_pc = act // NUM_TARGETS
                            if (total_finished == 0 and k_pc >= 4) or (total_finished >= 1 and my_score + float(PIECE_SCORE[k_pc]) > opp_score):
                                v = 1.0
                                break
                else:
                    shaped_p, v_tac = shape_tactical_priors_and_anchor(
                        probs_b[k_idx, leg], leg, x_dyns[k_idx]
                    )
                    if v_tac >= 0.999:
                        v = 1.0  # Exact immediate win shortcut preserved (review #2.1)
                    else:
                        v = (1.0 - v_tac_weight) * v_net + v_tac_weight * v_tac

                    if shaping_weight <= 0.0:
                        raw_p = probs_b[k_idx, leg]
                        p_sum = float(np.sum(raw_p))
                        priors = raw_p / p_sum if p_sum > 1e-8 else np.full(len(leg), 1.0 / len(leg), dtype=np.float32)
                    elif shaping_weight >= 1.0:
                        priors = shaped_p
                    else:
                        raw_p = probs_b[k_idx, leg]
                        p_sum = float(np.sum(raw_p))
                        raw_norm = raw_p / p_sum if p_sum > 1e-8 else np.full(len(leg), 1.0 / len(leg), dtype=np.float32)
                        priors = (1.0 - shaping_weight) * raw_norm + shaping_weight * shaped_p
                        p_sum = float(np.sum(priors))
                        priors = priors / p_sum if p_sum > 1e-8 else raw_norm

                node.expand(leg, priors, value=v)
                node.x_dyn = x_dyns[k_idx]
                values[local_i] = v

        return values

    # Expand any unexpanded roots in one batched GPU call
    if unexpanded_nodes:
        _batch_expand_nodes(unexpanded_nodes)

    if add_dirichlet_noise:
        for r_node in roots:
            r_node.add_dirichlet_noise(eps=dirichlet_eps, alpha=dirichlet_alpha, rng=rng)

    # Run simulations in lockstep across all games
    for _ in range(num_simulations):
        paths: List[List[Tuple[MCTSNode, int]]] = []
        leaves: List[Tuple[int, MCTSNode]] = []

        for i in range(num_games):
            node = roots[i]
            path: List[Tuple[MCTSNode, int]] = []
            while node.is_expanded and not node.game.done:
                idx = node.select_action_idx(c_puct)
                path.append((node, idx))
                if idx not in node.children:
                    next_g = node.game.clone()
                    assert node.legal_actions is not None
                    next_g.step(int(node.legal_actions[idx]))
                    node.children[idx] = MCTSNode(next_g)
                node = node.children[idx]
            paths.append(path)
            leaves.append((i, node))

        leaf_vals = _batch_expand_nodes(leaves)

        for i in range(num_games):
            v = leaf_vals[i]
            for parent, action_idx in reversed(paths[i]):
                v = -v
                assert parent.N is not None and parent.W is not None
                parent.N[action_idx] += 1
                parent.W[action_idx] += v
                parent.total_N += 1

    return roots
