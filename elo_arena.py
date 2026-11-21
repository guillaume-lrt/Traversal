"""
Traversal Elo Arena & Model Tournament Engine.

Runs automated competitive matches and tournaments between AlphaZero checkpoints,
baseline Heuristic agents, and tracks player ratings (including Guillaume)
in the persistent Elo database.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from elo_system import EloDatabase, MatchRecord, parse_agent_variant, VARIANT_PRESETS
from encoder import encode_dynamic_tensor
from mcts import MCTS
from model import TraversalDualStemNet, load_model_from_checkpoint
from train import HeuristicAgent
from traversal_env import TraversalGame, TERRAIN_MODES


class AgentWrapper:
    """Unified interface for playing moves across AlphaZero checkpoints and Heuristic agents."""

    def __init__(
        self,
        agent_id: str,
        device: torch.device,
        num_simulations: int = 48,
        checkpoint_dir: str = "checkpoints",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.agent_id = agent_id
        self.device = device
        self.num_simulations = num_simulations
        self.rng = rng or np.random.default_rng()
        self.checkpoint_dir = checkpoint_dir

        self.kind: str
        self.display_name: str
        self.mcts: Optional[MCTS] = None
        self.heuristic: Optional[HeuristicAgent] = None
        self.metadata: Dict[str, Any] = {}

        self._setup()

    def _setup(self) -> None:
        base_id, var_key, sims = parse_agent_variant(self.agent_id, default_sims=self.num_simulations)
        self.num_simulations = sims

        if base_id == "heuristic":
            self.kind = "heuristic"
            if var_key in ("ultra", "8ply", "grandmaster", "gm"):
                self.display_name = "Heuristic Baseline — 👑 Grandmaster (8-ply α-β)"
                self.heuristic = HeuristicAgent(rng=self.rng, variant="grandmaster")
            elif var_key in ("deep", "6ply", "master"):
                self.display_name = "Heuristic Baseline — 🧠 Master (6-ply α-β)"
                self.heuristic = HeuristicAgent(rng=self.rng, variant="master")
            elif var_key in ("fast", "2ply", "blitz"):
                self.display_name = "Heuristic Baseline — ⚡ Blitz (2-ply α-β)"
                self.heuristic = HeuristicAgent(rng=self.rng, variant="blitz")
            else:
                self.display_name = "Heuristic Baseline — ⚖️ Standard (4-ply α-β)"
                self.heuristic = HeuristicAgent(rng=self.rng, variant="standard")
            return

        self.kind = "alphazero"
        # Find checkpoint
        path = base_id
        if not os.path.isfile(path):
            cand = os.path.join(self.checkpoint_dir, base_id)
            if os.path.isfile(cand):
                path = cand
            else:
                raise FileNotFoundError(f"Checkpoint not found: {base_id}")

        model, meta = load_model_from_checkpoint(path, device=self.device)
        for k in ("iteration", "loss_total", "heuristic_win_rate", "best_win_rate"):
            if k in meta:
                self.metadata[k] = float(meta[k]) if isinstance(meta[k], (int, float)) else meta[k]
        model.eval()

        it = self.metadata.get("iteration")
        it_str = f" Iter {it}" if it is not None else ""
        rel = os.path.relpath(path, self.checkpoint_dir).replace("\\", "/")

        if var_key in VARIANT_PRESETS:
            var_info = VARIANT_PRESETS[var_key]
            var_badge = f"{var_info['icon']} {var_info['name']} ({self.num_simulations}s)"
        else:
            var_badge = f"{self.num_simulations}s"

        self.display_name = f"AlphaZero ({rel}{it_str} — {var_badge})"
        self.mcts = MCTS(
            model=model,
            device=self.device,
            num_simulations=self.num_simulations,
            rng=self.rng,
        )

    def prepare_terrain(self, terrain: np.ndarray) -> None:
        if self.mcts is not None:
            self.mcts.cache_game_static(terrain)

    def select_action(self, game: TraversalGame) -> int:
        if self.heuristic is not None:
            return int(self.heuristic.select_action(game))

        assert self.mcts is not None
        root = self.mcts.run(
            game,
            add_dirichlet_noise=False,
            num_simulations=self.num_simulations,
        )
        action, _ = root.select_move_by_temperature(temperature=0.0, rng=self.rng)
        return int(action)


def play_game(
    agent_p1: AgentWrapper,
    agent_p2: AgentWrapper,
    terrain: Optional[np.ndarray] = None,
    terrain_mode: str = "random",
    seed: Optional[int] = None,
    max_plies: int = 200,
) -> Tuple[int, int, List[int], str]:
    """
    Plays a single game between agent_p1 (Player 1) and agent_p2 (Player 2).
    Returns (winner, ply_count, [score0, score1], termination_reason).
    winner: 0 (P1 win), 1 (P2 win), -1 (draw).
    """
    if terrain is not None:
        game = TraversalGame(terrain=terrain)
    else:
        game = TraversalGame(terrain_mode=terrain_mode, seed=seed)

    agent_p1.prepare_terrain(game.terrain)
    agent_p2.prepare_terrain(game.terrain)

    while not game.done:
        cur_agent = agent_p1 if game.current_player == 0 else agent_p2
        action = cur_agent.select_action(game)
        game.step(action)

    return (
        int(game.winner) if game.winner is not None else -1,
        int(game.ply_count),
        [int(game.scores[0]), int(game.scores[1])],
        game.termination_reason or "unknown",
    )


def discover_available_checkpoints(checkpoint_dir: str = "checkpoints") -> List[str]:
    """Finds all valid .pt checkpoints relative to checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        return []
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt")) + glob.glob(
        os.path.join(checkpoint_dir, "*", "*.pt")
    )
    # Filter out non-model files (e.g. replay buffers, optimizers, temporary files)
    pt_files = [
        p
        for p in pt_files
        if not os.path.basename(p).startswith(("replay_buffer", "buffer", "optimizer"))
        and "replay_buffer" not in os.path.basename(p)
        and not p.endswith((".tmp", ".temp"))
    ]
    # Prefer best.pt, then latest.pt, then iter_*.pt
    priority = {"best.pt": 0, "latest.pt": 1}

    def _sort_key(full_path: str) -> Tuple[int, float, int, str]:
        rel = os.path.relpath(full_path, checkpoint_dir).replace("\\", "/")
        parent = os.path.dirname(rel)
        base = os.path.basename(rel)
        dir_mtime = (
            os.path.getmtime(os.path.join(checkpoint_dir, parent)) if parent else 0.0
        )
        return (0 if parent else 1, -dir_mtime, priority.get(base, 2), rel)

    pt_files.sort(key=_sort_key)
    return [os.path.relpath(p, checkpoint_dir).replace("\\", "/") for p in pt_files]


def run_matched_pair(
    agent_a_id: str,
    agent_b_id: str,
    db: EloDatabase,
    device: torch.device,
    num_simulations: int = 48,
    terrain_mode: str = "random",
    checkpoint_dir: str = "checkpoints",
    source: str = "arena",
    on_game_finish: Optional[Callable[[MatchRecord], None]] = None,
    wrapper_a: Optional[AgentWrapper] = None,
    wrapper_b: Optional[AgentWrapper] = None,
) -> Tuple[MatchRecord, MatchRecord]:
    """
    Plays 2 games between Agent A and Agent B on the EXACT same terrain:
      - Game 1: A is P1, B is P2
      - Game 2: B is P1, A is P2
    Records results in the database and returns both MatchRecords.
    """
    rng = np.random.default_rng()
    if wrapper_a is None:
        wrapper_a = AgentWrapper(agent_a_id, device, num_simulations, checkpoint_dir, rng)
    if wrapper_b is None:
        wrapper_b = AgentWrapper(agent_b_id, device, num_simulations, checkpoint_dir, rng)

    # Generate identical shared terrain
    base_game = TraversalGame(
        terrain_mode=terrain_mode,
        seed=int(rng.integers(0, 1_000_000_000)),
    )
    shared_terrain = base_game.terrain.copy()

    # Game 1: A (P1) vs B (P2)
    win1, plies1, scores1, reason1 = play_game(
        wrapper_a, wrapper_b, terrain=shared_terrain
    )
    rec1 = db.record_match(
        p1_id=agent_a_id,
        p2_id=agent_b_id,
        winner=win1,
        p1_name=wrapper_a.display_name,
        p2_name=wrapper_b.display_name,
        p1_kind=wrapper_a.kind,
        p2_kind=wrapper_b.kind,
        ply_count=plies1,
        scores=scores1,
        termination_reason=reason1,
        terrain_mode=terrain_mode,
        source=source,
    )
    if on_game_finish:
        on_game_finish(rec1)

    # Game 2: B (P1) vs A (P2) on same terrain
    win2, plies2, scores2, reason2 = play_game(
        wrapper_b, wrapper_a, terrain=shared_terrain
    )
    rec2 = db.record_match(
        p1_id=agent_b_id,
        p2_id=agent_a_id,
        winner=win2,
        p1_name=wrapper_b.display_name,
        p2_name=wrapper_a.display_name,
        p1_kind=wrapper_b.kind,
        p2_kind=wrapper_a.kind,
        ply_count=plies2,
        scores=scores2,
        termination_reason=reason2,
        terrain_mode=terrain_mode,
        source=source,
    )
    if on_game_finish:
        on_game_finish(rec2)

    return rec1, rec2


def run_tournament(
    agent_ids: List[str],
    db: Optional[EloDatabase] = None,
    pairs_per_matchup: int = 1,  # 1 pair = 2 games (1 as P1, 1 as P2)
    num_simulations: int = 48,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    terrain_mode: str = "random",
    progress_callback: Optional[Callable[[int, int, MatchRecord], None]] = None,
    target_agent_id: Optional[str] = None,
) -> List[MatchRecord]:
    """
    Runs an automated tournament between agents:
      - If target_agent_id is None (Round-Robin):
          Every pair of agents in agent_ids plays each other. Total matchups: N*(N-1)/2.
      - If target_agent_id is provided (Targeted Gauntlet):
          Only target_agent_id plays against each opponent in agent_ids. Opponents do NOT
          play each other, saving significant compute and focusing games entirely on the target.
    """
    if db is None:
        db = EloDatabase()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matchups: List[Tuple[str, str]] = []
    if target_agent_id:
        if target_agent_id not in agent_ids:
            agent_ids = [target_agent_id] + list(agent_ids)
        opponents = [a for a in agent_ids if a != target_agent_id]
        for opp in opponents:
            matchups.append((target_agent_id, opp))
    else:
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                matchups.append((agent_ids[i], agent_ids[j]))

    if not matchups:
        return []

    # Pre-instantiate agent wrappers to reuse warm models across matchups
    wrappers: Dict[str, AgentWrapper] = {}
    needed_ids = set()
    for a, b in matchups:
        needed_ids.add(a)
        needed_ids.add(b)

    for aid in needed_ids:
        try:
            wrappers[aid] = AgentWrapper(
                aid,
                device=device,
                num_simulations=num_simulations,
                checkpoint_dir=checkpoint_dir,
            )
        except Exception as exc:
            print(f"[Elo Arena] Warning: Could not initialize agent '{aid}': {exc}")

    all_records: List[MatchRecord] = []
    total_games = len(matchups) * pairs_per_matchup * 2
    completed_games = 0

    def _callback(record: MatchRecord) -> None:
        nonlocal completed_games
        completed_games += 1
        if progress_callback:
            progress_callback(completed_games, total_games, record)

    for agent_a, agent_b in matchups:
        wrap_a = wrappers.get(agent_a)
        wrap_b = wrappers.get(agent_b)
        if wrap_a is None or wrap_b is None:
            print(f"[Elo Arena] Skipping matchup {agent_a} vs {agent_b} due to initialization error.")
            continue

        for _ in range(pairs_per_matchup):
            r1, r2 = run_matched_pair(
                agent_a_id=agent_a,
                agent_b_id=agent_b,
                db=db,
                device=device,
                num_simulations=num_simulations,
                terrain_mode=terrain_mode,
                checkpoint_dir=checkpoint_dir,
                on_game_finish=_callback,
                wrapper_a=wrap_a,
                wrapper_b=wrap_b,
            )
            all_records.extend([r1, r2])

    return all_records


def format_leaderboard_table(leaderboard: List[Dict[str, Any]]) -> str:
    """Formats the leaderboard into a clean, beautiful terminal table."""
    headers = ["Rank", "Name", "Type", "Elo Rating", "Peak Elo", "W / D / L", "Win %", "Games"]
    rows = []
    for p in leaderboard:
        rank_str = f"#{p['rank']}"
        if p["rank"] == 1:
            rank_str = "🥇 #1"
        elif p["rank"] == 2:
            rank_str = "🥈 #2"
        elif p["rank"] == 3:
            rank_str = "🥉 #3"

        kind_badge = {
            "human": "👤 Human",
            "alphazero": "🧠 AlphaZero",
            "heuristic": "⚙️ Heuristic",
        }.get(p.get("kind", ""), p.get("kind", ""))

        record_str = f"{p['wins']}-{p['draws']}-{p['losses']}"
        win_pct = f"{p['win_rate'] * 100:.1f}%" if p["games_played"] > 0 else "-"
        rows.append([
            rank_str,
            p["name"],
            kind_badge,
            f"{p['rating']:.1f}",
            f"{p['peak_rating']:.1f}",
            record_str,
            win_pct,
            str(p["games_played"]),
        ])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))

    def _line(sep_left, sep_mid, sep_right, fill="-"):
        return sep_left + sep_mid.join(fill * (w + 2) for w in widths) + sep_right

    out = []
    out.append(_line("┌", "┬", "┐"))
    header_cells = [f" {headers[i].ljust(widths[i])} " for i in range(len(headers))]
    out.append("│" + "│".join(header_cells) + "│")
    out.append(_line("├", "┼", "┤"))

    for row in rows:
        cells = [f" {row[i].ljust(widths[i])} " for i in range(len(row))]
        out.append("│" + "│".join(cells) + "│")

    out.append(_line("└", "┴", "┘"))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Traversal Elo Arena & Model Tournament CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: leaderboard
    subparsers.add_parser("leaderboard", help="Display current Elo ratings leaderboard")

    # Command: history
    hist_parser = subparsers.add_parser("history", help="Show recent match history")
    hist_parser.add_argument("--limit", type=int, default=20, help="Number of matches to show")
    hist_parser.add_argument("--player", type=str, default=None, help="Filter matches by player ID")

    # Command: tournament
    tourn_parser = subparsers.add_parser("tournament", help="Run a round-robin tournament or targeted gauntlet between models")
    tourn_parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="List of checkpoint relpaths or 'heuristic'. Defaults to best checkpoints found.",
    )
    tourn_parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target specific agent (e.g. train_v8/best.pt:master). Only plays games between this target and opponents (Gauntlet mode, skips opponent-vs-opponent matches).",
    )
    tourn_parser.add_argument(
        "--games-per-pair",
        type=int,
        default=2,
        help="Number of matched pairs (each pair = 2 games swapped) per matchup",
    )
    tourn_parser.add_argument(
        "--sims",
        type=int,
        default=32,
        help="Number of MCTS simulations per move (default: 32 for arena)",
    )
    tourn_parser.add_argument(
        "--terrain-mode",
        type=str,
        default="random",
        choices=list(TERRAIN_MODES),
        help="Terrain mode for tournament games",
    )
    tourn_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory with .pt files",
    )

    # Command: match
    match_parser = subparsers.add_parser("match", help="Run a direct head-to-head match between two agents")
    match_parser.add_argument("agent1", type=str, help="First agent (checkpoint or 'heuristic')")
    match_parser.add_argument("agent2", type=str, help="Second agent (checkpoint or 'heuristic')")
    match_parser.add_argument("--pairs", type=int, default=1, help="Number of balanced pairs (x2 games)")
    match_parser.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")

    # Command: record
    rec_parser = subparsers.add_parser("record", help="Manually record a match result")
    rec_parser.add_argument("--p1", type=str, required=True, help="Player 1 ID (e.g. Guillaume)")
    rec_parser.add_argument("--p2", type=str, required=True, help="Player 2 ID (e.g. train_v4/best.pt)")
    rec_parser.add_argument(
        "--winner",
        type=str,
        required=True,
        choices=["p1", "p2", "draw"],
        help="Match winner",
    )
    rec_parser.add_argument("--reason", type=str, default="manual_entry", help="Termination reason")

    # Command: reset
    reset_parser = subparsers.add_parser("reset", help="Reset all Elo ratings to 1500")
    reset_parser.add_argument("--force", action="store_true", help="Confirm reset without prompting")

    args = parser.parse_args()
    db = EloDatabase()

    if args.command in (None, "leaderboard"):
        lb = db.get_leaderboard()
        print("\n" + format_leaderboard_table(lb) + "\n")
        return

    if args.command == "history":
        matches = db.get_matches(limit=args.limit, player_id=args.player)
        if not matches:
            print("No matches recorded yet.")
            return
        print(f"\nRecent Match History ({len(matches)} matches):")
        for m in matches:
            win_sym = "⚔️"
            if m["winner"] == 0:
                win_txt = f"{m['player1_name']} won"
            elif m["winner"] == 1:
                win_txt = f"{m['player2_name']} won"
            else:
                win_txt = "Draw"
            d1_sign = f"+{m['p1_delta']:.1f}" if m['p1_delta'] >= 0 else f"{m['p1_delta']:.1f}"
            d2_sign = f"+{m['p2_delta']:.1f}" if m['p2_delta'] >= 0 else f"{m['p2_delta']:.1f}"
            print(
                f"[{m['timestamp'][:19]}] {m['player1_name']} ({d1_sign} -> {m['p1_rating_after']:.1f}) "
                f"vs {m['player2_name']} ({d2_sign} -> {m['p2_rating_after']:.1f}) "
                f"| Result: {win_txt} ({m['ply_count']} plies, {m['termination_reason']})"
            )
        print()
        return

    if args.command == "record":
        w_code = 0 if args.winner == "p1" else (1 if args.winner == "p2" else -1)
        rec = db.record_match(
            p1_id=args.p1,
            p2_id=args.p2,
            winner=w_code,
            termination_reason=args.reason,
            source="manual_cli",
        )
        print(f"Recorded match: {rec.player1_name} vs {rec.player2_name}")
        print(f"  {rec.player1_name}: {rec.p1_rating_before:.1f} -> {rec.p1_rating_after:.1f} ({rec.p1_delta:+.1f})")
        print(f"  {rec.player2_name}: {rec.p2_rating_before:.1f} -> {rec.p2_rating_after:.1f} ({rec.p2_delta:+.1f})")
        return

    if args.command == "reset":
        if not args.force:
            conf = input("Are you sure you want to reset all ratings to 1500? (y/N): ")
            if conf.lower() != "y":
                print("Reset cancelled.")
                return
        db.reset(keep_players=True)
        print("Elo database successfully reset to default 1500.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Elo Arena] Using device: {device}")

    if args.command == "match":
        print(f"Starting match: {args.agent1} vs {args.agent2} ({args.pairs * 2} games, {args.sims} sims)...")
        for p_idx in range(args.pairs):
            r1, r2 = run_matched_pair(
                agent_a_id=args.agent1,
                agent_b_id=args.agent2,
                db=db,
                device=device,
                num_simulations=args.sims,
            )
            print(f"  Game {p_idx*2 + 1}: Winner: {r1.winner_name} ({r1.ply_count} plies) | {r1.p1_name} ({r1.p1_delta:+.1f}), {r1.p2_name} ({r1.p2_delta:+.1f})")
            print(f"  Game {p_idx*2 + 2}: Winner: {r2.winner_name} ({r2.ply_count} plies) | {r2.p1_name} ({r2.p1_delta:+.1f}), {r2.p2_name} ({r2.p2_delta:+.1f})")

        print("\nUpdated Standings:")
        print(format_leaderboard_table(db.get_leaderboard()))
        return

    if args.command == "tournament":
        agents = args.checkpoints
        if not agents:
            # Auto-pick best checkpoints across train versions
            all_ckpts = discover_available_checkpoints(args.checkpoint_dir)
            bests = [c for c in all_ckpts if os.path.basename(c) == "best.pt"]
            agents = bests + ["heuristic:grandmaster", "heuristic:master", "heuristic:standard", "heuristic:blitz"]

        if args.target:
            if args.target not in agents:
                agents = [args.target] + list(agents)
            opponents = [a for a in agents if a != args.target]
            total_matches = len(opponents) * args.games_per_pair * 2
            print(f"[Elo Arena] Running TARGET GAUNTLET for '{args.target}' against {len(opponents)} opponents ({total_matches} total games, {args.games_per_pair} pairs each):")
            for ag in opponents:
                print(f"  vs {ag}")
        else:
            total_matches = (len(agents) * (len(agents) - 1) // 2) * args.games_per_pair * 2
            print(f"[Elo Arena] Running ROUND-ROBIN tournament among {len(agents)} agents ({total_matches} total games, {args.games_per_pair} pairs each):")
            for ag in agents:
                print(f"  - {ag}")

        def _prog(cur, tot, rec: MatchRecord) -> None:
            w_str = rec.winner_name
            print(f"  [{cur}/{tot}] {rec.player1_name} vs {rec.player2_name} -> {w_str} ({rec.ply_count} plies) [{rec.p1_delta:+.1f} / {rec.p2_delta:+.1f}]")

        run_tournament(
            agent_ids=agents,
            db=db,
            pairs_per_matchup=args.games_per_pair,
            num_simulations=args.sims,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            terrain_mode=args.terrain_mode,
            progress_callback=_prog,
            target_agent_id=args.target,
        )

        print("\n=== Final Tournament Standings ===")
        print(format_leaderboard_table(db.get_leaderboard()))


if __name__ == "__main__":
    main()
