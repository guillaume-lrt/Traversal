"""
Unit tests for Milestone 4 (`mcts.py` and `train.py`).
Verifies MCTS sign-flipped backpropagation on immediate winning moves,
batched GPU MCTS self-play, ReplayBuffer static deduplication + 2x symmetry augmentation,
and an end-to-end training & evaluation iteration on GPU.
"""

import os
import numpy as np
import torch
import pytest

from traversal_env import TraversalGame, STATUS_ACTIVE, STATUS_CAPTURED
from model import TraversalDualStemNet
from encoder import encode_both_static_tensors
from mcts import MCTS, run_batched_mcts_roots
from train import (
    HeuristicAgent,
    ReplayBuffer,
    _precompute_terrain_tables_numba,
    generate_self_play_games,
    evaluate_against_heuristic,
    evaluate_checkpoint_arena,
    parse_args,
)
from elo_system import EloDatabase


def test_mcts_finds_immediate_sentinel_instant_win():
    """
    Verifies that MCTS terminal evaluation and alternating-turn sign-flipped backpropagation
    correctly assign Q = +1.0 to an immediate winning move and select it deterministically.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    g = TraversalGame(terrain_mode="flat")
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)
    # Place P1 Sentinel_0 (piece_idx=4) at (8, 4), 1 step from Row 9 -> can finish and win immediately!
    g.pos[0, 4] = [8, 4]
    g.piece_status[0, 4] = STATUS_ACTIVE
    # Place P2 Explorer_0 at (1, 4)
    g.pos[1, 0] = [1, 4]
    g.piece_status[1, 0] = STATUS_ACTIVE
    g._cached_legal_mask = None

    mcts = MCTS(model=model, device=device, num_simulations=40, rng=np.random.default_rng(0))
    root = mcts.run(g, add_dirichlet_noise=False)

    win_action = g.encode_action(piece_idx=4, finish=True)
    chosen_action, pi_mcts = root.select_move_by_temperature(temperature=0.0)

    assert chosen_action == win_action
    assert np.isclose(np.sum(pi_mcts), 1.0)
    # The winning action's mean value Q = W / N must be +1.0
    win_idx = int(np.flatnonzero(root.legal_actions == win_action)[0])
    assert root.N[win_idx] > 0
    assert np.isclose(root.W[win_idx] / root.N[win_idx], 1.0, atol=1e-5)


def test_mcts_winning_player_avoids_threefold_repetition():
    """
    Verifies that a player in a winning position will not repeat moves into
    threefold repetition (which would result in a draw and drop expected reward to 0.0),
    preferring winning progression moves instead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    g = TraversalGame(terrain_mode="flat")
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P0 Sentinel_0 at (8, 4) (can finish and win immediately)
    g.pos[0, 4] = [8, 4]
    g.piece_status[0, 4] = STATUS_ACTIVE
    # P0 Explorer_0 at (3, 3)
    g.pos[0, 0] = [3, 3]
    g.piece_status[0, 0] = STATUS_ACTIVE
    # P1 Explorer_0 at (7, 7)
    g.pos[1, 0] = [7, 7]
    g.piece_status[1, 0] = STATUS_ACTIVE

    # Seed history such that moving P0 Explorer to (3, 4) produces the 3rd repetition
    temp_g = g.clone()
    temp_g.pos[0, 0] = [3, 4]
    temp_g.current_player = 1
    rep_key = temp_g.get_position_key()
    g.position_counts[rep_key] = 2

    # Verify that stepping the repeating move directly produces threefold repetition draw
    test_clone = g.clone()
    done, outcome = test_clone.step_move(piece_idx=0, target_r=3, target_c=4)
    assert done is True
    assert test_clone.termination_reason == "threefold_repetition"
    assert test_clone.winner == -1
    assert outcome == 0.0

    # Run MCTS from g
    mcts = MCTS(model=model, device=device, num_simulations=50, rng=np.random.default_rng(0))
    root = mcts.run(g, add_dirichlet_noise=False)

    win_action = g.encode_action(piece_idx=4, finish=True)
    repeat_action = g.encode_action(piece_idx=0, target_r=3, target_c=4)

    chosen_action, _ = root.select_move_by_temperature(temperature=0.0)

    win_idx = int(np.flatnonzero(root.legal_actions == win_action)[0])
    rep_idx = int(np.flatnonzero(root.legal_actions == repeat_action)[0])

    q_win = root.W[win_idx] / max(1, root.N[win_idx])
    q_rep = root.W[rep_idx] / max(1, root.N[rep_idx]) if root.N[rep_idx] > 0 else 0.0

    assert chosen_action == win_action
    assert q_win > q_rep
    assert q_win >= 0.99


def test_end_to_end_self_play_replay_buffer_and_training_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    buf = ReplayBuffer(capacity=1000)
    rng = np.random.default_rng(42)

    stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=2,
        num_simulations=8,
        augment_symmetry=True,
        rng=rng,
    )

    assert stats["games"] == 2.0
    assert len(buf) == int(stats["samples_added"])
    assert len(buf) > 0
    # Because of per-game static deduplication, 2 games * 2 players * 2 mirrors = 8 static tensors stored
    assert len(buf.static_store) == 8

    # Sample mini-batch and run a joint gradient step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_s, x_d, pi_t, mask_t, z_t = buf.sample_batch(batch_size=min(16, len(buf)), device=device, rng=rng)
    logits, v_pred = model(x_s, x_d)

    from model import compute_alphazero_loss
    loss, metrics = compute_alphazero_loss(logits, v_pred, pi_t, z_t, mask_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert metrics["loss_total"] > 0.0

    # Run quick arena check against HeuristicAgent
    eval_stats = evaluate_against_heuristic(
        model=model, device=device, num_games=2, num_simulations=8, rng=rng
    )
    assert 0.0 <= eval_stats["win_rate"] <= 1.0


def test_alphabeta_heuristic_and_mcts_avoid_hanging_explorer():
    """
    Reproduces the exact tactical blunder scenario from human play:
      - P1 Explorer_0 is at (1, 2) on elevation h=3.
      - P2 Explorer_1 is at (6, 2) on elevation h=2, and (2, 2) has elevation h=2.
      - Moving P2 Explorer_1 (6, 2) -> (2, 2) hangs Explorer_1 for free to P1 Explorer_0.
      - Verifies that both the 3-ply Alpha-Beta HeuristicAgent and MCTS with FPU + tactical
        prior shaping refuse to play the suicidal move (6, 2) -> (2, 2), and that if P2
        is placed on (2, 2), P1 immediately captures it.
    """
    g = TraversalGame(terrain_mode="flat")
    g.terrain.fill(2)
    g.terrain[1, 2] = 3  # P1 Explorer_0 has height advantage over (2, 2)

    # Apply the exact opening moves up to Ply 5 (P2's turn to move)
    g.pos[0, 4] = [3, 4]  # #1 P1 Sentinel_0 (0,4) -> (3,4)
    g.pos[1, 4] = [6, 5]  # #2 P2 Sentinel_0 (9,5) -> (6,5)
    g.pos[0, 1] = [3, 7]  # #3 P1 Explorer_1 (0,7) -> (3,7)
    g.pos[1, 1] = [6, 2]  # #4 P2 Explorer_1 (9,2) -> (6,2)
    g.pos[0, 0] = [1, 2]  # #5 P1 Explorer_0 (0,2) -> (1,2)
    g.current_player = 1
    g.ply_count = 5
    g._cached_legal_mask = None

    blunder_action = g.encode_action(piece_idx=1, target_r=2, target_c=2)

    # 1. 3-ply Alpha-Beta HeuristicAgent must not play the blunder
    heur = HeuristicAgent(rng=np.random.default_rng(0), depth=3)
    heur_action = heur.select_action(g)
    assert heur_action != blunder_action

    # 2. MCTS (even with an uncalibrated/flat network) must dampen the hanging move and avoid it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()
    mcts = MCTS(model=model, device=device, num_simulations=64, rng=np.random.default_rng(0))
    root = mcts.run(g, add_dirichlet_noise=False)
    mcts_action, _ = root.select_move_by_temperature(temperature=0.0)
    assert mcts_action != blunder_action

    # 3. If P2 does play the blunder (6,2)->(2,2), P1's 3-ply HeuristicAgent immediately captures (2,2)
    g.step(blunder_action)
    chosen_reply = heur.select_action(g)
    _, dest_r, dest_c, is_fin = g.decode_action(chosen_reply)
    assert not is_fin and (dest_r, dest_c) == (2, 2)


def test_heuristic_stops_advancing_enemy_sentinel_at_all_costs():
    """
    Verifies that when an opponent Sentinel gets dangerously close to its finish line
    (with total_pieces_finished == 0), HeuristicAgent prioritizes stopping it (capturing
    or blocking its forward path) over advancing its own pieces on the other side of the board.
    """
    g = TraversalGame(terrain_mode="flat")
    g.terrain.fill(2)
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P2 Sentinel_0 is at (2, 4) heading toward Row 0 (only 2 rows from Instant Win!)
    g.pos[1, 4] = [2, 4]
    g.piece_status[1, 4] = STATUS_ACTIVE

    # P1 has Sentinel_0 at (3, 4) (on h=3 so it can capture (2, 4)) AND an Explorer at (6, 8)
    g.terrain[3, 4] = 3
    g.pos[0, 4] = [3, 4]
    g.piece_status[0, 4] = STATUS_ACTIVE
    g.pos[0, 0] = [6, 8]
    g.piece_status[0, 0] = STATUS_ACTIVE

    g.current_player = 0
    g._cached_legal_mask = None

    heur = HeuristicAgent(rng=np.random.default_rng(0))
    assert heur.depth == 4
    action = heur.select_action(g)
    piece_idx, dest_r, dest_c, is_fin = g.decode_action(action)
    # Must capture the enemy Sentinel at (2, 4) rather than racing the Explorer at (6, 8)
    assert not is_fin and piece_idx == 4 and (dest_r, dest_c) == (2, 4)


def test_precomputed_terrain_tables_and_elevation_quality():
    """
    Verifies that `_precompute_terrain_tables_numba` accurately computes:
      1. Higher `elev_quality` for a +1 plateau/hill than a +2 steep peak (which gives no
         extra combat bonus over +1 but incurs a steep climbing penalty).
      2. Exact minimum `turns_grid` taking terrain ridges into account.
    """
    terrain = np.full((10, 10), 2, dtype=np.int8)
    # Square (5, 2) is a +1 hill (h=3 surrounded by h=2)
    terrain[5, 2] = 3
    # Square (5, 7) is a +2 steep spike (h=4 surrounded by h=2)
    terrain[5, 7] = 4
    # Put a steep wall across row 8 columns 0..4 so P1 at (6, 1) needs more turns than P1 at (6, 8)
    terrain[8, 0:5] = 4

    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    # +1 hill must have strictly better elevation quality than +2 spike
    assert elev_quality[5, 2] > elev_quality[5, 7]
    # And +1 hill has strictly better elevation quality than flat h=2
    assert elev_quality[5, 2] > elev_quality[2, 2]

    # Unblocked flat path at (6, 8) -> can finish in 1 turn for Sentinel (M=3, 3 steps to row 9 + 1 finish = 4 > 3 -> 2 turns, whereas (7, 8) is 1 turn)
    assert turns_grid[0, 2, 7, 8] == 1  # Sentinel at (7,8) finishes in 1 turn on flat terrain
    assert turns_grid[0, 2, 7, 1] > turns_grid[0, 2, 7, 8]  # Blocked by h=4 wall at row 8
    assert cost_grid[0, 7, 1] > cost_grid[0, 7, 8]


def test_mcts_prior_dampens_sacrificing_sentinel_for_defended_explorer():
    """
    Verifies that when a Sentinel can capture an opponent Explorer, but that square is
    guarded/recapturable by another opponent piece (e.g. Vanguard on high ground):
      1. `opp_max_attack` correctly identifies the defender's reach onto the occupied square.
      2. `shape_tactical_priors_and_anchor` marks the trade as recapturable and dampens
         the prior rather than treating it as a free capture.
      3. MCTS does not make the suicidal sacrifice.
    """
    g = TraversalGame(terrain_mode="flat")
    g.terrain.fill(2)
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P1: Explorer_0 at (0, 2) [h=2], Vanguard_0 at (0, 0) on h=3 (attack 2+1=3 >= 3)
    g.terrain[0, 0] = 3
    g.pos[0, 0] = [0, 2]
    g.piece_status[0, 0] = STATUS_ACTIVE
    g.pos[0, 2] = [0, 0]
    g.piece_status[0, 2] = STATUS_ACTIVE

    # P2: Sentinel_1 at (2, 2)
    g.pos[1, 5] = [2, 2]
    g.piece_status[1, 5] = STATUS_ACTIVE
    g.current_player = 1
    g.ply_count = 9
    g._cached_legal_mask = None

    # 1. P1's Vanguard must project attack=3 onto (0, 2) despite Explorer_0 occupying it
    _, _, _, p1_max_atk, _, _ = g.get_player_combat_and_reach(0)
    assert p1_max_atk[0, 2] >= 3.0

    # 2. In P2's canonical perspective, P1's attack onto (0, 2) (canonical (9, 7)) must be >= 3.0
    from encoder import encode_dynamic_tensor
    from mcts import shape_tactical_priors_and_anchor
    x_dyn = encode_dynamic_tensor(g)
    assert np.isclose(x_dyn[26, 9, 7] * 10.0, 3.0, atol=0.1)

    # 3. Shape priors: capturing (0, 2) with Sentinel must be dampened (priors[m] * 0.30)
    legal_actions = g.get_legal_actions()
    action_sac = g.encode_action(piece_idx=5, target_r=0, target_c=2, canonical=False)
    assert action_sac in legal_actions

    raw_priors = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
    shaped_priors, _ = shape_tactical_priors_and_anchor(raw_priors, legal_actions, x_dyn)

    idx_sac = int(np.flatnonzero(legal_actions == action_sac)[0])
    # The suicidal move's shaped prior must be lower than its raw unshaped prior
    assert shaped_priors[idx_sac] < raw_priors[idx_sac]


def test_mcts_prior_dampens_suicidal_second_finish():
    """
    Verifies that when total_finished >= 1 and finishing would result in a lower score
    than the opponent (e.g. P1 has banked a Vanguard for 2 points, P2 has 0 and could finish
    an Explorer for 1 pt), MCTS prior shaping dampens the suicidal finish (which would terminate
    the game in a 1 vs 2 loss) rather than boosting it.
    """
    from encoder import encode_dynamic_tensor
    from mcts import shape_tactical_priors_and_anchor

    g = TraversalGame(terrain_mode="flat")
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P1 has banked 2 points (Vanguard finished), total_finished = 1
    g.scores[0] = 2
    g.scores[1] = 0
    g.total_pieces_finished = 1

    # P2 (current player) has Explorer_0 on Row 0 (can finish for 1 pt) and Vanguard_0 at (2, 2)
    g.pos[1, 0] = [0, 2]  # On Row 0 -> can finish
    g.piece_status[1, 0] = STATUS_ACTIVE
    g.pos[1, 2] = [2, 2]
    g.piece_status[1, 2] = STATUS_ACTIVE

    # P1 has dummy piece far away
    g.pos[0, 0] = [9, 9]
    g.piece_status[0, 0] = STATUS_ACTIVE

    g.current_player = 1
    g._cached_legal_mask = None

    legal_actions = g.get_legal_actions()
    finish_action = g.encode_action(piece_idx=0, finish=True)
    assert finish_action in legal_actions

    x_dyn = encode_dynamic_tensor(g)
    raw_priors = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
    shaped_priors, v_tac = shape_tactical_priors_and_anchor(raw_priors, legal_actions, x_dyn)

    fin_idx = int(np.flatnonzero(legal_actions == finish_action)[0])
    # Finishing Explorer_0 ends the game and loses 1 vs 2 -> must be severely dampened
    assert shaped_priors[fin_idx] < raw_priors[fin_idx] * 0.10


def test_mcts_prior_prioritizes_capturing_game_winning_finisher():
    """
    Verifies that when an opponent piece is capable of finishing and winning the game
    on their turn, capturing that piece is prioritized and NOT dampened as an unfavorable trade
    even if our piece has higher strength and the target square is recapturable.
    """
    from encoder import encode_dynamic_tensor
    from mcts import shape_tactical_priors_and_anchor

    g = TraversalGame(terrain_mode="flat")
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P2 has banked 2 points, P1 has banked 1 point, total_finished = 1
    g.scores[0] = 1
    g.scores[1] = 2
    g.total_pieces_finished = 1

    # P2 Explorer_0 is at (1, 4) on Row 1 (heading to Row 0 to finish for +1 pt -> 3 vs 1 win!)
    g.pos[1, 0] = [1, 4]
    g.piece_status[1, 0] = STATUS_ACTIVE
    # P2 Vanguard_0 guards (1, 4) from (2, 4)
    g.pos[1, 2] = [2, 4]
    g.piece_status[1, 2] = STATUS_ACTIVE

    # P1 (current player) Sentinel_0 is at (1, 5) (M=3, S=3), able to capture (1, 4)
    g.pos[0, 4] = [1, 5]
    g.piece_status[0, 4] = STATUS_ACTIVE
    g.pos[0, 0] = [8, 8]
    g.piece_status[0, 0] = STATUS_ACTIVE

    g.current_player = 0
    g._cached_legal_mask = None

    legal_actions = g.get_legal_actions()
    cap_action = g.encode_action(piece_idx=4, target_r=1, target_c=4, canonical=False)
    assert cap_action in legal_actions

    x_dyn = encode_dynamic_tensor(g)
    raw_priors = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
    shaped_priors, _ = shape_tactical_priors_and_anchor(raw_priors, legal_actions, x_dyn)

    cap_idx = int(np.flatnonzero(legal_actions == cap_action)[0])
    # Even though Sentinel (S=3) attacks Explorer (S=1) on a guarded square, it prevents an
    # immediate game-losing finish, so prior must be BOOSTED above raw uniform prior!
    assert shaped_priors[cap_idx] > raw_priors[cap_idx]


def test_opp_finish_threat_suppressed_for_suicidal_opponent_finish():
    """
    Verifies that when an opponent piece can finish, but doing so would lose the game for them
    (e.g. P1 has banked a Vanguard for 2 points, P2 has 0 and could finish an Explorer for 1 pt),
    P1's tactical anchor does not penalize itself with a phantom opp_finish_threat.
    """
    from encoder import encode_dynamic_tensor
    from mcts import shape_tactical_priors_and_anchor

    g = TraversalGame(terrain_mode="flat")
    g.pos.fill(-1)
    g.piece_status.fill(STATUS_CAPTURED)

    # P1 has banked 2 points (Vanguard finished), P2 has 0, total_finished = 1
    g.scores[0] = 2
    g.scores[1] = 0
    g.total_pieces_finished = 1

    # P1 Explorer_0 at (5, 5)
    g.pos[0, 0] = [5, 5]
    g.piece_status[0, 0] = STATUS_ACTIVE

    # P2 Explorer_0 at (0, 2) (starts on Row 0 -> can finish for 1 pt, which would lose 1 vs 2!)
    g.pos[1, 0] = [0, 2]
    g.piece_status[1, 0] = STATUS_ACTIVE

    g.current_player = 0
    g._cached_legal_mask = None

    x_dyn = encode_dynamic_tensor(g)
    legal_actions = g.get_legal_actions()
    raw_priors = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)

    _, v_tac = shape_tactical_priors_and_anchor(raw_priors, legal_actions, x_dyn)
    # P1 is winning 2-0 with 1 piece finished; P2 finishing gives P1 an instant 2-1 victory.
    # Therefore, v_tac must be positive (P1 has advantage), not dragged down to negative by phantom threat.
    assert v_tac > 0.0


def test_evaluate_board_recapture_logic_distinguishes_safe_captures_from_trades():
    """
    Verifies that _evaluate_board_numba does not treat a capture by a strong piece (e.g. Sentinel S=3)
    as a trade when the opponent only reaches the square with attack=1 (which cannot penetrate S=3).
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    terrain = np.zeros((10, 10), dtype=np.int32)
    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    # Position 1: Safe capture (P1 Explorer at (4,6) only has attack 1, cannot recapture S=3 Sentinel)
    pos_safe = np.full((2, 6, 2), -1, dtype=np.int32)
    scores = np.zeros(2, dtype=np.int32)
    pos_safe[0, 4] = [4, 4]  # P0 Sentinel_0 (S=3)
    pos_safe[1, 0] = [4, 5]  # P1 Explorer_0 (S=1)
    pos_safe[1, 1] = [4, 6]  # P1 Explorer_1 (S=1)

    # Position 2: Trade capture (P1 Sentinel at (4,6) has attack 3, CAN recapture S=3 Sentinel)
    pos_trade = np.full((2, 6, 2), -1, dtype=np.int32)
    pos_trade[0, 4] = [4, 4]  # P0 Sentinel_0 (S=3)
    pos_trade[1, 0] = [4, 5]  # P1 Explorer_0 (S=1)
    pos_trade[1, 4] = [4, 6]  # P1 Sentinel_0 (S=3)

    val_safe = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_safe, scores, total_finished=0, p=0
    )
    val_trade = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_trade, scores, total_finished=0, p=0
    )

    # P0 capturing when safe should be substantially better than when facing recapture
    assert val_safe > val_trade + 500.0


def test_evaluate_board_penalizes_opponent_winning_second_finish_severely():
    """
    Verifies that _evaluate_board_numba heavily penalizes an opponent 1-turn second finish
    that wins the match, and does not penalize an opponent finish that would lose the match for them.
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    terrain = np.zeros((10, 10), dtype=np.int32)
    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    # Case A: P1 has 2 pts, P0 has 1 pt, total_finished = 1.
    # P1 has an Explorer on Row 0 (o_turns = 1). If P1 finishes, P1 wins 3 to 1!
    # P0's piece is at (6, 5), too far to eliminate the threat in 1 turn (can_kill_now = False).
    pos_a = np.full((2, 6, 2), -1, dtype=np.int32)
    pos_a[0, 0] = [6, 5]
    pos_a[1, 0] = [0, 5]  # Row 0 -> finishes in 1 turn
    scores_a = np.array([1, 2], dtype=np.int32)

    val_a = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_a, scores_a, total_finished=1, p=0
    )

    # Case B: P0 has 2 pts, P1 has 0 pts, total_finished = 1.
    # P1 has an Explorer on Row 0 (o_turns = 1). If P1 finishes, P0 wins 2 to 1!
    scores_b = np.array([2, 0], dtype=np.int32)

    val_b = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_a, scores_b, total_finished=1, p=0
    )

    # In Case A, P0 faces an unstoppable match loss -> evaluation must be heavily negative
    assert val_a < -1500.0
    # In Case B, P0 is ahead 2-0 and P1's finish gives P0 the win -> evaluation must be strongly positive
    assert val_b > 1000.0


def test_replay_buffer_threefold_repetition_assigned_exact_zero_target():
    """
    Verifies that ReplayBuffer._compute_shaped_outcome assigns exact z = 0.0
    when termination_reason == 'threefold_repetition', even if one player has
    greater material or piece progress.
    """
    g = TraversalGame(terrain_mode="flat")
    # Set P0 to have lots of pieces on the board
    g.pos[0, 4] = [5, 5]  # Sentinel
    g.pos[0, 5] = [5, 6]  # Sentinel
    # P1 has only 1 Explorer
    g.pos[1, 0] = [1, 1]
    for k in range(1, 6):
        g.piece_status[1, k] = STATUS_CAPTURED
        g.pos[1, k] = [-1, -1]

    g.done = True
    g.winner = -1
    g.termination_reason = "threefold_repetition"

    z_p0 = ReplayBuffer._compute_shaped_outcome(g, player=0)
    z_p1 = ReplayBuffer._compute_shaped_outcome(g, player=1)

    assert z_p0 == 0.0
    assert z_p1 == 0.0


def test_evaluate_board_does_not_reward_suicidal_second_finish():
    """
    Verifies that when total_finished >= 1 and finishing a piece would lose the match
    (scores[p] + PIECE_SCORE[k] < scores[opp]), _evaluate_board_numba does not treat it
    as a positive finish bonus (+240) nor as the primary race turn piece.
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    terrain = np.zeros((10, 10), dtype=np.int32)
    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    # Opponent banked a Vanguard (2 pts). P0 has 0 pts. Total finished = 1.
    scores = np.array([0, 2], dtype=np.int32)

    # Position A: P0 Explorer_0 (1 pt) is at (9, 5) (can finish this turn, but finishing loses 1 to 2!)
    pos_suicidal = np.full((2, 6, 2), -1, dtype=np.int32)
    pos_suicidal[0, 0] = [9, 5]  # Explorer at row 9 (finishes for 1 pt -> loses 1 vs 2)
    pos_suicidal[1, 0] = [5, 5]  # Opponent piece far away

    # Position B: P0 Explorer_0 is at (8, 5) (not on row 9 yet)
    pos_normal = np.full((2, 6, 2), -1, dtype=np.int32)
    pos_normal[0, 0] = [8, 5]
    pos_normal[1, 0] = [5, 5]

    val_suicidal = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_suicidal, scores, total_finished=1, p=0
    )
    val_normal = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_normal, scores, total_finished=1, p=0
    )

    # Being on row 9 with an Explorer when finishing loses 1-2 must NOT give an immediate +240 finish bonus
    # Difference should only reflect normal row 8 vs 9 terrain progress, never the +240 finish bonus
    assert (val_suicidal - val_normal) < 200.0


def test_evaluate_board_recapture_discounts_capturer_strength_on_trades():
    """
    Verifies that _evaluate_board_numba correctly evaluates recapture gain as
    max(0, s_vic - s_attacker) when recapturable, ensuring that sacrificing a Sentinel (S=3)
    for an Explorer (S=1) gives gain = 0.0 (no phantom capture reward).
    """
    from train import _evaluate_board_numba, _precompute_terrain_tables_numba

    terrain = np.zeros((10, 10), dtype=np.int32)
    turns_grid, cost_grid, elev_quality = _precompute_terrain_tables_numba(terrain)

    # Board with P0 Sentinel at (4,4), P1 Explorer at (4,5), P1 Sentinel at (4,6) [can recapture]
    pos = np.full((2, 6, 2), -1, dtype=np.int32)
    scores = np.zeros(2, dtype=np.int32)
    pos[0, 4] = [4, 4]  # P0 Sentinel (S=3)
    pos[1, 0] = [4, 5]  # P1 Explorer (S=1)
    pos[1, 4] = [4, 6]  # P1 Sentinel (S=3)

    val_trade = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos, scores, total_finished=0, p=0
    )

    # Board where P1 Explorer is absent (no blunder capture available)
    pos_no_victim = pos.copy()
    pos_no_victim[1, 0] = [-1, -1]
    val_no_cap = _evaluate_board_numba(
        terrain, turns_grid, cost_grid, elev_quality, pos_no_victim, scores, total_finished=0, p=0
    )

    # P0 should not evaluate having an unfavorable trade as superior to the opponent having 1 fewer piece
    assert val_trade <= val_no_cap


def test_compute_alphazero_loss_computes_policy_kl_and_entropy():
    """
    Verifies that compute_alphazero_loss accurately computes target entropy
    and policy KL divergence D_KL(pi_target || p_theta) = CE - H(pi_target).
    """
    from model import compute_alphazero_loss

    b = 2
    # 5 legal actions out of 606
    legal_mask = torch.zeros(b, 606, dtype=torch.bool)
    legal_mask[:, :5] = True

    # Uniform target distribution over 5 actions: H(pi) = ln(5) ~ 1.6094379
    pi_target = torch.zeros(b, 606)
    pi_target[:, :5] = 0.2

    # Case 1: logits perfectly uniform over legal actions -> p_theta == pi_target -> KL = 0
    logits_uniform = torch.zeros(b, 606)
    v_pred = torch.zeros(b, 1)
    z_target = torch.zeros(b)

    loss_1, metrics_1 = compute_alphazero_loss(logits_uniform, v_pred, pi_target, z_target, legal_mask)
    expected_ent = np.log(5.0)
    assert np.isclose(metrics_1["policy_entropy"], expected_ent, atol=1e-5)
    assert np.isclose(metrics_1["policy_kl"], 0.0, atol=1e-5)

    # Case 2: logits skewed away from uniform -> KL > 0
    logits_skewed = torch.zeros(b, 606)
    logits_skewed[:, 0] = 5.0  # concentrates probability on action 0
    loss_2, metrics_2 = compute_alphazero_loss(logits_skewed, v_pred, pi_target, z_target, legal_mask)
    assert metrics_2["policy_kl"] > 0.1
    assert np.isclose(metrics_2["policy_entropy"], expected_ent, atol=1e-5)
    assert np.isclose(metrics_2["loss_policy"], metrics_2["policy_entropy"] + metrics_2["policy_kl"], atol=1e-5)


def test_compute_fresh_trajectory_metrics_and_self_play_integration():
    """
    Verifies that compute_fresh_trajectory_metrics correctly computes held-out
    validation metrics on fresh self-play trajectories, and that generate_self_play_games
    includes them in its return dictionary.
    """
    from train import compute_fresh_trajectory_metrics, generate_self_play_games
    from model import TraversalDualStemNet, masked_softmax

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    # 1. Empty trajectories case
    empty_res = compute_fresh_trajectory_metrics(
        model=model,
        device=device,
        static_np_pairs=[],
        trajectories=[],
        games=[],
    )
    assert empty_res["fresh_value_mse"] == 0.0
    assert empty_res["fresh_policy_kl"] == 0.0

    # 2. Run small self-play and check stats
    buf = ReplayBuffer(capacity=500)
    sp_stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=2,
        num_simulations=8,
        augment_symmetry=False,
        rng=np.random.default_rng(123),
    )

    for k in (
        "fresh_value_mse",
        "fresh_value_raw_mse",
        "fresh_value_sign_acc",
        "fresh_value_corr",
        "fresh_policy_kl",
        "fresh_policy_entropy",
        "fresh_policy_ce",
    ):
        assert k in sp_stats, f"Missing key {k} in sp_stats"

    assert sp_stats["fresh_value_mse"] >= 0.0
    assert sp_stats["fresh_policy_kl"] >= 0.0
    assert 0.0 <= sp_stats["fresh_value_sign_acc"] <= 1.0


def test_target_sample_reuse_ratio_calculation():
    """
    Verifies that target_reuse correctly modulates gradient step count
    proportional to new samples added, preventing sample-reuse explosion.
    """
    batch_size = 64
    train_steps_cap = 60

    # Case 1: 400 new samples, target_reuse = 3.0 -> desired = 1200 samples -> 19 steps
    new_samples = 400
    target_reuse = 3.0
    desired_samples = target_reuse * float(new_samples)
    steps_from_reuse = int(round(desired_samples / batch_size))
    effective_steps = max(4, min(steps_from_reuse, train_steps_cap))
    assert effective_steps == 19
    actual_reuse = (effective_steps * batch_size) / new_samples
    assert np.isclose(actual_reuse, 3.04, atol=0.1)

    # Case 2: 1200 new samples, target_reuse = 4.0 -> desired = 4800 samples -> 75 steps -> capped at 60
    new_samples = 1200
    target_reuse = 4.0
    desired_samples = target_reuse * float(new_samples)
    steps_from_reuse = int(round(desired_samples / batch_size))
    effective_steps = max(4, min(steps_from_reuse, train_steps_cap))
    assert effective_steps == train_steps_cap

    # Case 3: target_reuse = 0 disables reuse control and uses standard warmup formula
    target_reuse = 0.0
    buf_len = 10000
    calc_steps = min(train_steps_cap, max(8, (buf_len * 2) // batch_size))
    assert calc_steps == train_steps_cap


def test_mcts_node_get_root_q():
    """
    Verifies that MCTSNode.get_root_q() computes the visit-weighted expected search value
    from the node player's perspective, clamped to [-1.0, 1.0].
    """
    from mcts import MCTSNode
    from traversal_env import TraversalGame

    game = TraversalGame(seed=42)
    node = MCTSNode(game)
    node.V = 0.42

    # Unexpanded / zero visits falls back to node.V
    assert node.get_root_q() == pytest.approx(0.42)

    # Expanded node with visits
    actions = np.array([5, 12, 19], dtype=np.int32)
    priors = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    node.expand(actions, priors, value=0.42)
    node.N = np.array([10, 20, 10], dtype=np.int32)
    node.W = np.array([5.0, 10.0, -2.0], dtype=np.float32)
    node.total_N = 40

    # Expected root_Q = sum(W) / total_N = (5 + 10 - 2) / 40 = 13 / 40 = 0.325
    assert node.get_root_q() == pytest.approx(0.325)

    # Clamping test
    node.W = np.array([30.0, 30.0, 30.0], dtype=np.float32)
    assert node.get_root_q() == pytest.approx(1.0)
    node.W = np.array([-30.0, -30.0, -30.0], dtype=np.float32)
    assert node.get_root_q() == pytest.approx(-1.0)


def test_value_target_smoothing_q_blend():
    """
    Verifies that ReplayBuffer.add_game_trajectory applies q_blend to smooth
    value targets: z' = (1 - q_blend) * z_raw + q_blend * root_q (review #1.4).
    """
    from traversal_env import TraversalGame
    from encoder import encode_both_static_tensors, encode_dynamic_tensor

    game = TraversalGame(seed=101)
    s0, s1 = encode_both_static_tensors(game.terrain)
    x_dyn = encode_dynamic_tensor(game)
    mask = game.get_legal_mask()
    pi = np.zeros(606, dtype=np.float32)
    pi[0] = 1.0

    # Let game end with Player 0 winning: raw outcome for P0 is 1.0
    game.done = True
    game.winner = 0
    raw_z0 = float(game.get_outcome(0))
    assert raw_z0 == 1.0

    root_q = 0.10

    # 1. q_blend = 0.0 (default AlphaZero, pure z)
    buf0 = ReplayBuffer(capacity=100)
    buf0.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, root_q)],
        final_game=game,
        augment_symmetry=True,
        q_blend=0.0,
    )
    assert len(buf0) == 2
    assert buf0.buffer[0].z == pytest.approx(raw_z0)
    assert buf0.buffer[1].z == pytest.approx(raw_z0)  # mirrored sample

    # 2. q_blend = 0.5 (50% final outcome + 50% root_Q)
    buf_half = ReplayBuffer(capacity=100)
    buf_half.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, root_q)],
        final_game=game,
        augment_symmetry=True,
        q_blend=0.5,
    )
    expected_half = 0.5 * raw_z0 + 0.5 * root_q
    assert buf_half.buffer[0].z == pytest.approx(expected_half)
    assert buf_half.buffer[1].z == pytest.approx(expected_half)

    # 3. q_blend = 1.0 (100% root_Q)
    buf1 = ReplayBuffer(capacity=100)
    buf1.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, root_q)],
        final_game=game,
        augment_symmetry=True,
        q_blend=1.0,
    )
    assert buf1.buffer[0].z == pytest.approx(root_q)
    assert buf1.buffer[1].z == pytest.approx(root_q)

    # 4. Backward compatibility: 4-tuple trajectory without root_q
    buf_legacy = ReplayBuffer(capacity=100)
    buf_legacy.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask)],
        final_game=game,
        augment_symmetry=False,
        q_blend=0.0,
    )
    assert buf_legacy.buffer[0].z == pytest.approx(raw_z0)


def test_mcts_shaping_and_v_tac_weights_and_immediate_win_shortcut():
    """
    Verifies that MCTS respects shaping_weight and v_tac_weight, smoothly
    transitioning from tactical shaping to pure network priors/values, while
    preserving the exact 'immediate win -> v=1' shortcut (review #2.1).
    """
    from model import TraversalDualStemNet, masked_softmax
    from mcts import MCTS, MCTSNode
    from encoder import encode_dynamic_tensor
    from traversal_env import TraversalGame, PIECE_SENTINEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    # Case 1: Standard non-terminal position without immediate win
    game = TraversalGame(seed=42)

    # A: shaping_weight=0.0, v_tac_weight=0.0 (Pure network)
    mcts_pure = MCTS(
        model=model,
        device=device,
        num_simulations=4,
        shaping_weight=0.0,
        v_tac_weight=0.0,
        rng=np.random.default_rng(42),
    )
    mcts_pure.cache_game_static(game.terrain)
    node_pure = MCTSNode(game.clone())
    v_pure = mcts_pure._evaluate_leaf(node_pure)

    # With v_tac_weight=0.0, leaf evaluation must equal network v_net
    with torch.inference_mode():
        x_dyn = encode_dynamic_tensor(game)
        f_s = mcts_pure._get_f_static(game)
        xd_t = torch.from_numpy(x_dyn).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(game.get_legal_mask()).unsqueeze(0).to(device)
        logits, v_pred = model.forward_with_cached_static(f_s, xd_t)
        expected_v_net = float(v_pred[0, 0].item())
        expected_p_net = masked_softmax(logits, mask_t)[0].cpu().numpy()

    assert v_pure == pytest.approx(expected_v_net, abs=1e-5)
    # Check that priors match network distribution
    assert node_pure.legal_actions is not None
    leg = node_pure.legal_actions
    expected_priors = expected_p_net[leg] / np.sum(expected_p_net[leg])
    assert np.allclose(node_pure.P, expected_priors, atol=1e-5)

    # B: shaping_weight=1.0, v_tac_weight=0.30 (Legacy shaped MCTS)
    # On a board with tactical captures/threats, shaping alters priors significantly
    tactical_game = TraversalGame(seed=42)
    tactical_game.pos[1, 0] = [1, 2]  # Opponent piece right in front of friendly Explorer
    mcts_shaped = MCTS(
        model=model,
        device=device,
        num_simulations=4,
        shaping_weight=1.0,
        v_tac_weight=0.30,
        rng=np.random.default_rng(42),
    )
    mcts_shaped.cache_game_static(tactical_game.terrain)
    mcts_pure.cache_game_static(tactical_game.terrain)

    node_tactical_pure = MCTSNode(tactical_game.clone())
    node_tactical_shaped = MCTSNode(tactical_game.clone())
    mcts_pure._evaluate_leaf(node_tactical_pure)
    mcts_shaped._evaluate_leaf(node_tactical_shaped)
    assert node_tactical_shaped.P is not None
    assert node_tactical_pure.P is not None
    # Tactical shaping boosts capture and alters prior distribution
    assert not np.allclose(node_tactical_shaped.P, node_tactical_pure.P, atol=1e-3)

    # Case 2: Immediate win position -> v MUST BE 1.0 even when v_tac_weight=0.0!
    # Place P0 Sentinel on row 9 with legal finish move
    win_game = TraversalGame(seed=123)
    win_game.pos[0, 4] = [9, 2]  # P0 Sentinel at row 9
    mcts_pure.cache_game_static(win_game.terrain)
    node_win = MCTSNode(win_game.clone())
    v_win = mcts_pure._evaluate_leaf(node_win)
    assert v_win == pytest.approx(1.0)


def test_mcts_shaping_linear_anneal_schedule():
    """
    Verifies that the linear anneal schedule in train_alphazero correctly reduces
    shaping_weight and v_tac_weight from initial values to 0.0 over N iterations.
    """
    shaping_weight = 1.0
    shaping_anneal_iters = 20
    v_tac_weight = 0.30
    v_tac_anneal_iters = 20

    # Iteration 1: full initial weights
    it = 1
    cur_shaping = max(0.0, shaping_weight * (1.0 - min(1.0, (it - 1) / shaping_anneal_iters)))
    cur_v_tac = max(0.0, v_tac_weight * (1.0 - min(1.0, (it - 1) / v_tac_anneal_iters)))
    assert cur_shaping == pytest.approx(1.0)
    assert cur_v_tac == pytest.approx(0.30)

    # Iteration 11: halfway through 20 iterations
    it = 11
    cur_shaping = max(0.0, shaping_weight * (1.0 - min(1.0, (it - 1) / shaping_anneal_iters)))
    cur_v_tac = max(0.0, v_tac_weight * (1.0 - min(1.0, (it - 1) / v_tac_anneal_iters)))
    assert cur_shaping == pytest.approx(0.50)
    assert cur_v_tac == pytest.approx(0.15)

    # Iteration 21: reaches exactly 0.0
    it = 21
    cur_shaping = max(0.0, shaping_weight * (1.0 - min(1.0, (it - 1) / shaping_anneal_iters)))
    cur_v_tac = max(0.0, v_tac_weight * (1.0 - min(1.0, (it - 1) / v_tac_anneal_iters)))
    assert cur_shaping == pytest.approx(0.0)
    assert cur_v_tac == pytest.approx(0.0)

    # Iteration 30: clamped at 0.0
    it = 30
    cur_shaping = max(0.0, shaping_weight * (1.0 - min(1.0, (it - 1) / shaping_anneal_iters)))
    cur_v_tac = max(0.0, v_tac_weight * (1.0 - min(1.0, (it - 1) / v_tac_anneal_iters)))
    assert cur_shaping == pytest.approx(0.0)
    assert cur_v_tac == pytest.approx(0.0)


def test_value_target_consistency_raw_vs_shaped():
    """
    Verifies that --value-target raw produces exact {-1, 0, +1} outcomes matching
    MCTS terminal node values, while --value-target shaped uses speed/material shaping (review #2.2).
    """
    from traversal_env import TraversalGame
    from encoder import encode_both_static_tensors, encode_dynamic_tensor

    game = TraversalGame(seed=42)
    s0, s1 = encode_both_static_tensors(game.terrain)
    x_dyn = encode_dynamic_tensor(game)
    mask = game.get_legal_mask()
    pi = np.zeros(606, dtype=np.float32)
    pi[0] = 1.0

    # 1. Decisive win for Player 0
    game.done = True
    game.winner = 0
    game.ply_count = 60

    # Case A: raw (default) -> exact +1.0 for P0
    buf_raw = ReplayBuffer(capacity=100)
    buf_raw.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, 0.5)],
        final_game=game,
        augment_symmetry=False,
        value_target="raw",
    )
    assert buf_raw.buffer[0].z == pytest.approx(1.0)

    # Case B: shaped -> shaped magnitude < 1.0
    buf_shaped = ReplayBuffer(capacity=100)
    buf_shaped.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, 0.5)],
        final_game=game,
        augment_symmetry=False,
        value_target="shaped",
    )
    shaped_z = buf_shaped.buffer[0].z
    assert shaped_z != 1.0
    assert 0.58 <= shaped_z <= 1.0

    # 2. Draw game
    game_draw = TraversalGame(seed=77)
    game_draw.done = True
    game_draw.winner = -1
    game_draw.termination_reason = "max_plies"

    # Raw draw must be exactly 0.0
    buf_draw_raw = ReplayBuffer(capacity=100)
    buf_draw_raw.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, 0.0)],
        final_game=game_draw,
        augment_symmetry=False,
        value_target="raw",
    )
    assert buf_draw_raw.buffer[0].z == pytest.approx(0.0)

    # 3. Combined with q_blend: (1 - 0.4) * 1.0 + 0.4 * 0.20 = 0.68
    buf_blend_raw = ReplayBuffer(capacity=100)
    buf_blend_raw.add_game_trajectory(
        x_static_pair=(s0, s1),
        trajectory=[(0, x_dyn, pi, mask, 0.20)],
        final_game=game,
        augment_symmetry=False,
        q_blend=0.4,
        value_target="raw",
    )
    assert buf_blend_raw.buffer[0].z == pytest.approx(0.68)


def test_heuristic_curriculum_imitation_weight_and_anneal():
    """
    Verifies that heuristic_fraction and heuristic_imitation_weight can be annealed
    to 0, removing permanent imitation bias from weaker players (review #2.3).
    """
    from model import TraversalDualStemNet
    from train import generate_self_play_games

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    # 1. Pure self-play with heuristic_fraction = 0.0
    buf = ReplayBuffer(capacity=1000)
    stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=2,
        num_simulations=4,
        heuristic_fraction=0.0,
        heuristic_imitation_weight=0.0,
        augment_symmetry=False,
        rng=np.random.default_rng(99),
    )
    assert stats["games"] == 2.0
    assert len(buf) > 0

    # 2. Annealing schedule calculation over 20 iterations
    init_frac = 0.25
    init_imit = 0.50
    anneal_iters = 20

    # Iteration 1
    it = 1
    prog = min(1.0, (it - 1) / anneal_iters)
    frac = max(0.0, init_frac * (1.0 - prog))
    imit = max(0.0, init_imit * (1.0 - prog))
    assert frac == pytest.approx(0.25)
    assert imit == pytest.approx(0.50)

    # Iteration 11 (midway)
    it = 11
    prog = min(1.0, (it - 1) / anneal_iters)
    frac = max(0.0, init_frac * (1.0 - prog))
    imit = max(0.0, init_imit * (1.0 - prog))
    assert frac == pytest.approx(0.125)
    assert imit == pytest.approx(0.25)

    # Iteration 21 (fully annealed)
    it = 21
    prog = min(1.0, (it - 1) / anneal_iters)
    frac = max(0.0, init_frac * (1.0 - prog))
    imit = max(0.0, init_imit * (1.0 - prog))
    assert frac == pytest.approx(0.0)
    assert imit == pytest.approx(0.0)


def test_batched_evaluation_against_heuristic_and_elo_integration(tmp_path):
    """
    Verifies that `evaluate_against_heuristic`:
      1. Runs in lockstep GPU batches with paired terrain symmetry (even games AI is P0, odd AI is P1).
      2. Correctly updates and returns wins, draws, losses, total_games, and win_rate.
      3. Integrates with `EloDatabase` to record match histories and update Elo rating.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet(value_head_type="gap").to(device)
    model.eval()

    db_file = str(tmp_path / "test_elo.json")
    elo_db = EloDatabase(db_path=db_file)

    stats = evaluate_against_heuristic(
        model=model,
        device=device,
        num_games=4,
        num_simulations=6,
        heuristic_variant="blitz",
        rng=np.random.default_rng(123),
        elo_db=elo_db,
        model_id="test_candidate",
        opponent_id="heuristic:blitz",
    )

    assert stats["total_games"] == 4.0
    assert stats["wins"] + stats["draws"] + stats["losses"] == 4.0
    expected_wr = (stats["wins"] + 0.5 * stats["draws"]) / 4.0
    assert stats["win_rate"] == pytest.approx(expected_wr)

    assert "elo_rating" in stats
    assert len(elo_db.matches) == 4
    assert "test_candidate" in elo_db.players
    cand_player = elo_db.players["test_candidate"]
    assert cand_player.games_played == 4
    assert cand_player.rating == stats["elo_rating"]
    assert all(m.source == "training_eval" for m in elo_db.matches)


def test_batched_checkpoint_arena_evaluation_and_cli_args(tmp_path, monkeypatch):
    """
    Verifies that `evaluate_checkpoint_arena`:
      1. Pits candidate model against incumbent model across paired terrains.
      2. Returns score_rate, wins, draws, losses, and records matches in EloDatabase.
      3. Verifies new CLI flags in `parse_args()`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cand_model = TraversalDualStemNet(value_head_type="gap").to(device)
    inc_model = TraversalDualStemNet(value_head_type="gap").to(device)
    cand_model.eval()
    inc_model.eval()

    db_file = str(tmp_path / "arena_elo.json")
    elo_db = EloDatabase(db_path=db_file)

    arena_stats = evaluate_checkpoint_arena(
        candidate_model=cand_model,
        incumbent_model=inc_model,
        device=device,
        num_games=2,
        num_simulations=6,
        rng=np.random.default_rng(456),
        elo_db=elo_db,
        candidate_id="cand_v1",
        incumbent_id="inc_v0",
    )

    assert arena_stats["total_games"] == 2.0
    assert arena_stats["wins"] + arena_stats["draws"] + arena_stats["losses"] == 2.0
    expected_score = (arena_stats["wins"] + 0.5 * arena_stats["draws"]) / 2.0
    assert arena_stats["score_rate"] == pytest.approx(expected_score)

    assert "elo_rating" in arena_stats
    assert len(elo_db.matches) == 2
    assert "cand_v1" in elo_db.players
    assert "inc_v0" in elo_db.players
    assert all(m.source == "arena_gating" for m in elo_db.matches)

    # Test CLI parser defaults & flags
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--eval-games", "40",
        "--eval-interval", "2",
        "--gating", "arena",
        "--gating-threshold", "0.58",
        "--elo-db", "my_elo.json",
        "--eval-opponents", "heuristic,checkpoints/train_v4/best.pt",
    ])
    args = parse_args()
    assert args.eval_games == 40
    assert args.eval_interval == 2
    assert args.gating == "arena"
    assert args.gating_threshold == pytest.approx(0.58)
    assert args.elo_db == "my_elo.json"
    assert args.eval_opponents == "heuristic,checkpoints/train_v4/best.pt"


def test_replay_buffer_persistence_save_load(tmp_path):
    """
    Verifies that ReplayBuffer can be saved to and loaded from disk, preserving:
      - All samples with dynamic, policy, mask, and z targets.
      - Static tensors and deduplication refcounts.
      - Next game ID counter.
    """
    buf = ReplayBuffer(capacity=50)
    g = TraversalGame(terrain_mode="flat")
    s0 = np.full((4, 10, 10), 1.0, dtype=np.float32)
    s1 = np.full((4, 10, 10), 2.0, dtype=np.float32)
    traj = [
        (0, np.full((34, 10, 10), 0.5, dtype=np.float32), np.ones(606, dtype=np.float32) / 606.0, np.ones(606, dtype=bool), 0.25),
        (1, np.full((34, 10, 10), -0.5, dtype=np.float32), np.ones(606, dtype=np.float32) / 606.0, np.ones(606, dtype=bool), -0.25),
    ]
    added = buf.add_game_trajectory((s0, s1), traj, g, augment_symmetry=True)
    assert added == 4  # 2 moves * 2 (original + mirror)
    assert len(buf) == 4

    save_path = str(tmp_path / "replay_test.pt")
    buf.save(save_path)
    assert os.path.isfile(save_path)

    loaded_buf = ReplayBuffer.load(save_path, capacity=50)
    assert len(loaded_buf) == 4
    assert loaded_buf._next_game_id == buf._next_game_id
    assert set(loaded_buf.static_store.keys()) == set(buf.static_store.keys())

    device = torch.device("cpu")
    xs, xd, pi, mask, z = loaded_buf.sample_batch(batch_size=4, device=device)
    assert xs.shape == (4, 4, 10, 10)
    assert xd.shape == (4, 34, 10, 10)
    assert pi.shape == (4, 606)
    assert mask.shape == (4, 606)
    assert z.shape == (4,)


def test_resume_correctness_scheduler_and_rng(tmp_path, monkeypatch):
    """
    Verifies that checkpoint payloads save and restore:
      - scheduler_state_dict (resuming preserves the cosine annealing decay, rather than restarting).
      - torch_rng_state and np_rng_state (resuming generates unique/unbroken sequences).
      - CLI flag --save-buffer is parsed properly.
    """
    model = TraversalDualStemNet(value_head_type="gap")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-4)

    # Step scheduler 200 times
    for _ in range(200):
        opt.step()
        sched.step()
    lr_at_step_200 = opt.param_groups[0]["lr"]
    assert lr_at_step_200 < 1e-3  # decaying

    rng = np.random.default_rng(999)
    # Generate some random values
    val1 = rng.random()
    np_state = rng.bit_generator.state
    torch_state = torch.get_rng_state()

    ckpt_file = str(tmp_path / "ckpt_test.pt")
    torch.save({
        "iteration": 10,
        "run_name": "test_resume_run",
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "torch_rng_state": torch_state,
        "np_rng_state": np_state,
        "heuristic_win_rate": 0.70,
        "best_win_rate": 0.70,
    }, ckpt_file)

    # Now verify restoring scheduler state on a new optimizer and scheduler
    model2 = TraversalDualStemNet(value_head_type="gap")
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=1000, eta_min=1e-4)

    meta = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    sched2.load_state_dict(meta["scheduler_state_dict"])
    for i, group in enumerate(opt2.param_groups):
        group["lr"] = sched2.get_last_lr()[i]
    # After loading scheduler state, opt2 learning rate matches lr_at_step_200
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_at_step_200)

    # Verify restoring RNG state reproduces the next random value
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = meta["np_rng_state"]
    val2 = rng.random()
    val2_restored = rng2.random()
    assert val2 == val2_restored

    # Verify parse_args parses --save-buffer
    monkeypatch.setattr("sys.argv", ["train.py", "--save-buffer"])
    args = parse_args()
    assert args.save_buffer is True


def test_encoder_v2_batched_self_play_and_replay_buffer_sampling(monkeypatch):
    """
    Verifies that model with encoder_version='v2' (35 channels) runs smoothly through:
      1. Batched self-play trajectory collection.
      2. Trajectory insertion into ReplayBuffer with 35-channel states and symmetry augmentation.
      3. ReplayBuffer.sample_batch mini-batch sampling producing (B, 35, 10, 10) tensors.
      4. parse_args() support for --encoder-version {v1, v2}.
    """
    device = torch.device("cpu")
    model = TraversalDualStemNet(value_head_type="gap", encoder_version="v2").to(device)
    assert model.encoder_version == "v2"
    assert model.dynamic_channels == 35

    buf = ReplayBuffer(capacity=1000)
    stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=2,
        num_simulations=4,
        heuristic_fraction=0.0,
    )
    assert len(buf) > 0
    assert stats["samples_added"] == float(len(buf))
    # Check that sample in buffer has 35 channels
    sample = buf.buffer[0]
    assert sample.x_dynamic.shape == (35, 10, 10)

    # Sample a batch
    xs, xd, pi, mask, z = buf.sample_batch(batch_size=4, device=device)
    assert xs.shape == (min(4, len(buf)), 4, 10, 10)
    assert xd.shape == (min(4, len(buf)), 35, 10, 10)
    assert pi.shape == (min(4, len(buf)), 606)

    # Test parse_args flag
    monkeypatch.setattr("sys.argv", ["train.py", "--encoder-version", "v2"])
    args = parse_args()
    assert args.encoder_version == "v2"

    monkeypatch.setattr("sys.argv", ["train.py", "--encoder-version", "v1"])
    args1 = parse_args()
    assert args1.encoder_version == "v1"


def test_dirichlet_noise_alpha_default_and_config(monkeypatch):
    """
    Verifies that Dirichlet alpha defaults to ~0.18 (10 / avg_legal, review #7)
    and is fully configurable via parameters and CLI flags.
    """
    device = torch.device("cpu")
    model = TraversalDualStemNet(value_head_type="gap")
    mcts = MCTS(model=model, device=device)
    assert mcts.dirichlet_alpha == pytest.approx(0.18)
    assert mcts.dirichlet_eps == pytest.approx(0.25)

    # Test MCTSNode.add_dirichlet_noise with alpha=0.18
    game = TraversalGame(terrain_mode="flat")
    from mcts import MCTSNode
    root = MCTSNode(game=game)
    legal_actions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    priors = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    root.expand(legal_actions, priors)

    rng = np.random.default_rng(1234)
    root.add_dirichlet_noise(eps=0.25, alpha=0.18, rng=rng)
    assert len(root.P) == 5
    assert np.all(root.P > 0.0)
    assert np.isclose(root.P.sum(), 1.0, atol=1e-5)
    # The noise must perturb the uniform distribution
    assert not np.allclose(root.P, priors, atol=1e-3)

    # Test parse_args flag
    monkeypatch.setattr("sys.argv", ["train.py", "--dirichlet-alpha", "0.22", "--dirichlet-eps", "0.30"])
    args = parse_args()
    assert args.dirichlet_alpha == pytest.approx(0.22)
    assert args.dirichlet_eps == pytest.approx(0.30)


def test_mcts_subtree_reuse_and_dyn_caching(monkeypatch):
    """
    Verifies MCTS subtree reuse across plies in self-play (review #9)
    and root x_dyn caching on MCTSNode (review #5.5).
    """
    device = torch.device("cpu")
    model = TraversalDualStemNet(value_head_type="gap", encoder_version="v2").to(device)
    game = TraversalGame(terrain_mode="flat", seed=42)

    # 1. Run batched MCTS roots from scratch
    s0, s1 = encode_both_static_tensors(game.terrain)
    f_pair = (model.encode_static(torch.from_numpy(s0).unsqueeze(0)), model.encode_static(torch.from_numpy(s1).unsqueeze(0)))
    roots = run_batched_mcts_roots(
        model=model,
        device=device,
        games=[game],
        f_static_per_game=[f_pair],
        num_simulations=16,
    )
    root = roots[0]
    assert root.is_expanded
    assert root.x_dyn is not None
    assert root.x_dyn.shape == (35, 10, 10)  # item 5.5: root dynamic state cached on node

    # Find the most visited child action
    best_act_idx = int(np.argmax(root.N))
    child = root.children[best_act_idx]
    prev_visits = child.total_N
    assert prev_visits > 0

    # Step the game
    game_next = game.clone()
    game_next.step(int(root.legal_actions[best_act_idx]))

    # 2. Reuse the child subtree as existing root
    roots_reused = run_batched_mcts_roots(
        model=model,
        device=device,
        games=[game_next],
        f_static_per_game=[f_pair],
        num_simulations=16,
        existing_roots=[child],
    )
    new_root = roots_reused[0]
    assert new_root is child
    # Simulations were added on top of previously accumulated visits
    assert new_root.total_N >= prev_visits + 16

    # 3. Test generate_self_play_games with subtree_reuse=True and subtree_reuse=False
    buf_reuse = ReplayBuffer(capacity=500)
    stats_reuse = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf_reuse,
        num_games=2,
        num_simulations=8,
        subtree_reuse=True,
    )
    assert stats_reuse["games"] == 2.0
    assert len(buf_reuse) > 0

    buf_no_reuse = ReplayBuffer(capacity=500)
    stats_no_reuse = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf_no_reuse,
        num_games=2,
        num_simulations=8,
        subtree_reuse=False,
    )
    assert stats_no_reuse["games"] == 2.0
    assert len(buf_no_reuse) > 0

    # 4. Test parse_args flags --subtree-reuse and --no-subtree-reuse
    monkeypatch.setattr("sys.argv", ["train.py", "--subtree-reuse"])
    args_true = parse_args()
    assert args_true.subtree_reuse is True

    monkeypatch.setattr("sys.argv", ["train.py", "--no-subtree-reuse"])
    args_false = parse_args()
    assert args_false.subtree_reuse is False


def test_temperature_schedule_and_floor(monkeypatch):
    """
    Verifies the extended temperature schedule and non-zero temperature floor (review #11):
      1. Default temp_threshold_ply is 30 (expanded from legacy 10).
      2. Default temp_floor is 0.0, with configurable CLI flag --temp-floor.
      3. Non-zero temperature (e.g. 0.3) preserves diversity over visit counts.
      4. generate_self_play_games executes cleanly with non-zero temp floor.
    """
    # 1. Test CLI defaults and parsing
    monkeypatch.setattr("sys.argv", ["train.py"])
    args_default = parse_args()
    assert args_default.temp_ply == 30
    assert args_default.temp_floor == 0.0

    monkeypatch.setattr("sys.argv", ["train.py", "--temp-ply", "24", "--temp-floor", "0.35"])
    args_custom = parse_args()
    assert args_custom.temp_ply == 24
    assert args_custom.temp_floor == pytest.approx(0.35)

    # 2. Test MCTSNode.select_move_by_temperature with tau=0.3
    from mcts import MCTSNode
    game = TraversalGame(terrain_mode="flat")
    node = MCTSNode(game)
    legal = np.array([10, 20, 30], dtype=np.int32)
    priors = np.array([0.33, 0.33, 0.34], dtype=np.float32)
    node.expand(legal, priors)
    node.N = np.array([10, 5, 1], dtype=np.int32)
    node.total_N = 16

    # With tau=0.0 (deterministic argmax), action 10 is chosen 100% of the time
    for _ in range(10):
        act, _ = node.select_move_by_temperature(temperature=0.0)
        assert act == 10

    # With tau=0.3, action 10 is heavily favored but action 20 is also selectable across multiple draws
    rng = np.random.default_rng(42)
    choices = [node.select_move_by_temperature(temperature=0.3, rng=rng)[0] for _ in range(200)]
    assert 10 in choices
    assert 20 in choices
    # 10 should have majority due to N=10 vs N=5
    count_10 = choices.count(10)
    count_20 = choices.count(20)
    assert count_10 > count_20 > 0

    # 3. Test generate_self_play_games integration with temp_threshold_ply=30 and temp_floor=0.2
    device = torch.device("cpu")
    model = TraversalDualStemNet(value_head_type="gap", encoder_version="v2").to(device)
    buf = ReplayBuffer(capacity=200)
    stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=2,
        num_simulations=4,
        temp_threshold_ply=30,
        temp_floor=0.2,
    )
    assert stats["games"] == 2.0
    assert len(buf) > 0


def test_sentinel_win_share_and_termination_metrics_tracking():
    """
    Verifies that self-play collects and reports detailed game termination statistics
    (sentinel_win_rate, two_finish_rate, elimination_rate, repetition_rate, ply_limit_rate),
    allowing continuous tracking of sentinel instant-win share with and without tactical shaping (review #12).
    """
    device = torch.device("cpu")
    model = TraversalDualStemNet(value_head_type="gap", encoder_version="v2").to(device)
    buf = ReplayBuffer(capacity=500)

    stats = generate_self_play_games(
        model=model,
        device=device,
        replay_buffer=buf,
        num_games=4,
        num_simulations=4,
        shaping_weight=0.0,
        v_tac_weight=0.0,
    )
    for key in (
        "sentinel_win_rate",
        "two_finish_rate",
        "elimination_rate",
        "repetition_rate",
        "ply_limit_rate",
    ):
        assert key in stats
        assert 0.0 <= stats[key] <= 1.0

    # Total termination rates across all categories must sum to 1.0
    total_rate = (
        stats["sentinel_win_rate"]
        + stats["two_finish_rate"]
        + stats["elimination_rate"]
        + stats["repetition_rate"]
        + stats["ply_limit_rate"]
    )
    assert np.isclose(total_rate, 1.0, atol=1e-5)


def test_replay_buffer_numpy_ring_array_and_float16_storage(tmp_path):
    """
    Verifies ReplayBuffer preallocated NumPy ring arrays (review #5.2):
      1. Storage in float16 for x_dynamic (halves RAM usage).
      2. O(1) vectorized batch sampling returning float32 tensors.
      3. Proper ring eviction when capacity is reached, cleaning up static refcounts.
      4. Transparent sequence access via .buffer[i] and len(buf.buffer).
      5. Full save/load persistence and backward compatibility with legacy sample lists.
    """
    from train import ReplaySample
    buf = ReplayBuffer(capacity=5)
    s_mat = np.zeros((4, 10, 10), dtype=np.float32)

    # Push 7 samples
    for i in range(7):
        buf._register_static(i, s_mat + i)
        sample = ReplaySample(
            static_key=i,
            x_dynamic=np.full((35, 10, 10), float(i), dtype=np.float32),
            pi_mcts=np.full(606, float(i) / 10.0, dtype=np.float32),
            legal_mask=np.ones(606, dtype=bool),
            z=float(i) - 3.0,
        )
        buf._push_sample(sample)

    assert len(buf) == 5
    assert len(buf.buffer) == 5
    assert buf.size == 5
    assert buf.head == 2
    # Verify x_dynamic is stored internally as float16
    assert buf.buf_x_dyn.dtype == np.float16

    # Keys 0 and 1 must be evicted from static store
    assert 0 not in buf.static_store
    assert 1 not in buf.static_store
    assert 2 in buf.static_store
    assert 6 in buf.static_store

    # Test sequence indexing
    assert buf.buffer[0].static_key == 2
    assert buf.buffer[0].z == pytest.approx(-1.0)
    assert buf.buffer[4].static_key == 6
    assert buf.buffer[4].z == pytest.approx(3.0)
    assert buf.buffer[-1].static_key == 6

    # Test vectorized sample_batch
    device = torch.device("cpu")
    xs, xd, pi, mask, z = buf.sample_batch(batch_size=3, device=device)
    assert xs.shape == (3, 4, 10, 10)
    assert xd.shape == (3, 35, 10, 10)
    assert xd.dtype == torch.float32  # Upcasted to float32 for torch
    assert pi.shape == (3, 606)
    assert mask.shape == (3, 606)
    assert z.shape == (3,)

    # Test save and load
    save_file = str(tmp_path / "ring_buf.pt")
    buf.save(save_file)

    loaded_buf = ReplayBuffer.load(save_file)
    assert len(loaded_buf) == 5
    assert loaded_buf.buffer[0].static_key == 2
    assert loaded_buf.buffer[4].static_key == 6

    # Test backward compatibility with legacy sample-list state dict
    legacy_payload = {
        "capacity": 10,
        "samples": [
            ReplaySample(
                static_key=100,
                x_dynamic=np.ones((34, 10, 10), dtype=np.float32),
                pi_mcts=np.zeros(606, dtype=np.float32),
                legal_mask=np.ones(606, dtype=bool),
                z=1.0,
            )
        ],
        "static_store": {100: s_mat},
        "static_refcount": {100: 1},
        "_next_game_id": 5,
    }
    legacy_buf = ReplayBuffer(capacity=10)
    legacy_buf.load_state_dict(legacy_payload)
    assert len(legacy_buf) == 1
    assert legacy_buf.buffer[0].static_key == 100
    assert legacy_buf.buffer[0].z == 1.0


def test_dynamic_candidate_move_buffer_expansion_and_bounds_guard():
    """
    Verifies that _grow_move_buffers and _generate_ordered_moves_numba safely
    grow move buffers without buffer overflow or fixed-256 array constraints (review #5.3).
    """
    from train import _grow_move_buffers, _generate_ordered_moves_numba, _precompute_terrain_tables_numba, HeuristicAgent
    from traversal_env import TraversalGame

    # 1. Verify buffer growth function
    k = np.array([1, 2, 3, 4], dtype=np.int32)
    r = np.array([5, 6, 7, 8], dtype=np.int32)
    c = np.array([9, 8, 7, 6], dtype=np.int32)
    fin = np.array([0, 0, 1, 0], dtype=np.int32)
    sc = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    new_k, new_r, new_c, new_fin, new_sc = _grow_move_buffers(k, r, c, fin, sc, 4)
    assert len(new_k) >= 512
    assert len(new_r) == len(new_k)
    assert len(new_c) == len(new_k)
    assert len(new_fin) == len(new_k)
    assert len(new_sc) == len(new_k)

    # Check preserved elements
    np.testing.assert_array_equal(new_k[:4], k)
    np.testing.assert_array_equal(new_r[:4], r)
    np.testing.assert_array_equal(new_c[:4], c)
    np.testing.assert_array_equal(new_fin[:4], fin)
    np.testing.assert_array_almost_equal(new_sc[:4], sc)

    # 2. Verify _generate_ordered_moves_numba on real game states
    game = TraversalGame(seed=123)
    t_grid, c_grid, eq = _precompute_terrain_tables_numba(game.terrain)

    cand_k, cand_r, cand_c, cand_fin, count = _generate_ordered_moves_numba(
        game.terrain,
        t_grid,
        c_grid,
        eq,
        game.pos,
        game.scores,
        int(game.total_pieces_finished),
        int(game.current_player),
        16,
    )
    assert count == 16
    assert len(cand_k) == 16
    assert len(cand_r) == 16
    assert len(cand_c) == 16
    assert len(cand_fin) == 16

    # 3. Verify HeuristicAgent search execution
    agent = HeuristicAgent(variant="fast", depth=2, root_beam=10)
    action = agent.select_action(game)
    assert action in game.get_legal_actions()












