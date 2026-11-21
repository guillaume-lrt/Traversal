"""
Elo Rating System & Match Tracking Engine for Traversal.

Features:
  - Standard FIDE-style Elo calculation with dynamic K-factors (provisional -> established).
  - Tracks Human players (e.g. Guillaume) and AI models/checkpoints side-by-side.
  - Atomic JSON database persistence with thread safety.
  - Leaderboard generation with ranks, peak ratings, win rates, and streak tracking.
  - Full match history logging with Elo deltas and game statistics.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_ratings.json")

VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "blitz": {
        "key": "blitz",
        "name": "Blitz",
        "icon": "⚡",
        "sims": 32,
        "description": "Fast & intuitive (32 sims, ~40ms/move)",
    },
    "standard": {
        "key": "standard",
        "name": "Standard",
        "icon": "⚖️",
        "sims": 96,
        "description": "Balanced tactical depth (96 sims, ~180ms/move)",
    },
    "master": {
        "key": "master",
        "name": "Master",
        "icon": "🧠",
        "sims": 256,
        "description": "Deep tactical search (256 sims, ~500ms/move)",
    },
    "grandmaster": {
        "key": "grandmaster",
        "name": "Grandmaster",
        "icon": "👑",
        "sims": 512,
        "description": "Ultra deep search (512 sims, ~1000ms/move)",
    },
}

HEURISTIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "blitz": {
        "key": "blitz",
        "name": "Blitz",
        "icon": "⚡",
        "depth": 2,
        "root_beam": 10,
        "description": "Fast & intuitive (2-ply α-β, 10/6 beam, ~0.8ms/move)",
    },
    "standard": {
        "key": "standard",
        "name": "Standard",
        "icon": "⚖️",
        "depth": 4,
        "root_beam": 14,
        "description": "Balanced tactical depth (4-ply α-β, 14/10/7/5 beam, ~5.1ms/move)",
    },
    "master": {
        "key": "master",
        "name": "Master",
        "icon": "🧠",
        "depth": 6,
        "root_beam": 16,
        "description": "Deep tactical search (6-ply α-β, 16/10/8/7/6/5 beam, ~76ms/move)",
    },
    "grandmaster": {
        "key": "grandmaster",
        "name": "Grandmaster",
        "icon": "👑",
        "depth": 8,
        "root_beam": 16,
        "description": "Ultra deep tactical search (8-ply α-β, 16/8/8/7/6/5/5/4 beam, ~480ms/move)",
    },
}


def parse_agent_variant(agent_id: str, default_sims: int = 96) -> Tuple[str, str, int]:
    """
    Parses an agent ID into (base_id, variant_key, sims_or_depth).
    Examples:
      'train_v4/best.pt:blitz' -> ('train_v4/best.pt', 'blitz', 32)
      'train_v4/best.pt:master' -> ('train_v4/best.pt', 'master', 256)
      'train_v4/best.pt@64' -> ('train_v4/best.pt', 'custom', 64)
      'train_v4/best.pt' -> ('train_v4/best.pt', 'standard', 96)
      'heuristic:blitz' -> ('heuristic', 'blitz', 2)
      'heuristic:master' -> ('heuristic', 'master', 6)
      'heuristic:standard' -> ('heuristic', 'standard', 4)
      'heuristic' -> ('heuristic', 'standard', 4)
      'Guillaume' -> ('Guillaume', 'human', 0)
    """
    norm = agent_id.strip()
    if norm.lower() in ("human", "guillaume") or (
        ":" not in norm and "@" not in norm and not norm.endswith(".pt") and "heuristic" not in norm.lower()
    ):
        return norm, "human", 0

    if "@" in norm:
        base, sims_str = norm.split("@", 1)
        sims = int(sims_str) if sims_str.isdigit() else default_sims
        return base, f"custom_{sims}", sims

    if ":" in norm:
        base, var_key = norm.split(":", 1)
        if base == "heuristic":
            if var_key in ("ultra", "8ply", "grandmaster", "gm"):
                return "heuristic", "grandmaster", 8
            if var_key in ("deep", "6ply", "master"):
                return "heuristic", "master", 6
            if var_key in ("fast", "2ply", "blitz"):
                return "heuristic", "blitz", 2
            return "heuristic", "standard", 4
        if var_key in VARIANT_PRESETS:
            return base, var_key, VARIANT_PRESETS[var_key]["sims"]
        return base, var_key, default_sims

    if norm == "heuristic":
        return "heuristic", "standard", 4

    return norm, "standard", default_sims


def expected_score(rating_a: float, rating_b: float) -> float:
    """Computes expected score E_A for Player A against Player B in [0.0, 1.0]."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_k_factor(games_played: int, base_k: Optional[float] = None) -> float:
    """
    Dynamic K-factor:
      - Provisional (< 10 games): K = 40 (faster calibration)
      - Developing (10 <= games < 30): K = 28
      - Established (>= 30 games): K = 16
    """
    if base_k is not None:
        return float(base_k)
    if games_played < 10:
        return 40.0
    if games_played < 30:
        return 28.0
    return 16.0


def calculate_elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    games_a: int = 0,
    games_b: int = 0,
    k_a: Optional[float] = None,
    k_b: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculates new ratings and rating deltas for player A and player B.
    score_a: 1.0 (win), 0.5 (draw), 0.0 (loss).
    Returns (new_rating_a, new_rating_b, delta_a, delta_b).
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a
    score_b = 1.0 - score_a

    act_k_a = compute_k_factor(games_a, k_a)
    act_k_b = compute_k_factor(games_b, k_b)

    delta_a = act_k_a * (score_a - e_a)
    delta_b = act_k_b * (score_b - e_b)

    new_a = round(rating_a + delta_a, 1)
    new_b = round(rating_b + delta_b, 1)
    return new_a, new_b, round(delta_a, 1), round(delta_b, 1)


@dataclass
class PlayerRecord:
    id: str
    name: str
    kind: str  # "human", "alphazero", "heuristic", "random"
    rating: float = DEFAULT_INITIAL_RATING
    peak_rating: float = DEFAULT_INITIAL_RATING
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_played: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return round((self.wins + 0.5 * self.draws) / self.games_played, 3)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        d["rating_int"] = int(round(self.rating))
        d["peak_rating_int"] = int(round(self.peak_rating))
        return d


@dataclass
class MatchRecord:
    match_id: str
    timestamp: str
    player1_id: str
    player2_id: str
    player1_name: str
    player2_name: str
    winner: int  # 0 = P1 win, 1 = P2 win, -1 = draw
    winner_name: str
    score_p1: float  # 1.0, 0.5, 0.0
    score_p2: float
    p1_rating_before: float
    p1_rating_after: float
    p1_delta: float
    p2_rating_before: float
    p2_rating_after: float
    p2_delta: float
    ply_count: int = 0
    scores: List[int] = field(default_factory=lambda: [0, 0])
    termination_reason: Optional[str] = None
    terrain_mode: Optional[str] = None
    source: str = "visualizer"  # "visualizer", "arena", "cli"

    @property
    def p1_name(self) -> str:
        return self.player1_name

    @property
    def p2_name(self) -> str:
        return self.player2_name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["p1_name"] = self.player1_name
        d["p2_name"] = self.player2_name
        return d


class EloDatabase:
    """Thread-safe persistent Elo database."""

    def __init__(self, db_path: str = DEFAULT_DATABASE_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.players: Dict[str, PlayerRecord] = {}
        self.matches: List[MatchRecord] = []
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not os.path.isfile(self.db_path):
                self._seed_defaults()
                self._save()
                return

            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                player_fields = {f.name for f in fields(PlayerRecord)}
                match_fields = {f.name for f in fields(MatchRecord)}

                self.players.clear()
                for p_id, p_data in data.get("players", {}).items():
                    p_clean = {k: v for k, v in p_data.items() if k in player_fields}
                    self.players[p_id] = PlayerRecord(**p_clean)

                self.matches.clear()
                for m_data in data.get("matches", []):
                    m_clean = {k: v for k, v in m_data.items() if k in match_fields}
                    self.matches.append(MatchRecord(**m_clean))

                # Ensure defaults exist
                self._seed_defaults()
            except Exception as exc:
                print(f"[EloDatabase] Warning: failed to parse {self.db_path} ({exc}); initializing fresh.")
                self.players.clear()
                self.matches.clear()
                self._seed_defaults()
                self._save()

    def _seed_defaults(self) -> None:
        """Seeds standard baseline players if not present."""
        if "Guillaume" not in self.players:
            self.players["Guillaume"] = PlayerRecord(
                id="Guillaume",
                name="Guillaume",
                kind="human",
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
            )
        if "heuristic" not in self.players:
            self.players["heuristic"] = PlayerRecord(
                id="heuristic",
                name="Heuristic Baseline (4-ply Standard)",
                kind="heuristic",
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
            )
        if "heuristic:grandmaster" not in self.players:
            self.players["heuristic:grandmaster"] = PlayerRecord(
                id="heuristic:grandmaster",
                name="Heuristic Baseline — 👑 Grandmaster (8-ply α-β)",
                kind="heuristic",
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
            )
        if "heuristic:master" not in self.players:
            self.players["heuristic:master"] = PlayerRecord(
                id="heuristic:master",
                name="Heuristic Baseline — 🧠 Master (6-ply α-β)",
                kind="heuristic",
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
            )
        if "heuristic:standard" not in self.players:
            std_r = self.players["heuristic"].rating if "heuristic" in self.players else DEFAULT_INITIAL_RATING
            self.players["heuristic:standard"] = PlayerRecord(
                id="heuristic:standard",
                name="Heuristic Baseline — ⚖️ Standard (4-ply α-β)",
                kind="heuristic",
                rating=std_r,
                peak_rating=std_r,
            )
        if "heuristic:blitz" not in self.players:
            self.players["heuristic:blitz"] = PlayerRecord(
                id="heuristic:blitz",
                name="Heuristic Baseline — ⚡ Blitz (2-ply α-β)",
                kind="heuristic",
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
            )

    def _save(self) -> None:
        with self._lock:
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "players": {p_id: p.to_dict() for p_id, p in self.players.items()},
                "matches": [m.to_dict() for m in self.matches],
            }
            dir_name = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(dir_name, exist_ok=True)
            temp_path = f"{self.db_path}.tmp.{uuid.uuid4().hex[:8]}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.db_path)

    def get_or_create_player(
        self,
        player_id: str,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlayerRecord:
        with self._lock:
            norm_id = player_id.strip()
            if norm_id in self.players:
                p = self.players[norm_id]
                if name and p.name != name:
                    p.name = name
                if kind and p.kind != kind:
                    p.kind = kind
                if metadata:
                    p.metadata.update(metadata)
                return p

            base_id, var_key, sims = parse_agent_variant(norm_id)

            # Infer kind
            if kind is None:
                if base_id.lower() in ("human", "guillaume") or var_key == "human":
                    kind = "human"
                elif "heuristic" in base_id.lower():
                    kind = "heuristic"
                elif ".pt" in base_id:
                    kind = "alphazero"
                else:
                    kind = "other"

            if name is None:
                if kind == "alphazero":
                    if var_key in VARIANT_PRESETS:
                        preset = VARIANT_PRESETS[var_key]
                        name = f"AlphaZero ({base_id} — {preset['icon']} {preset['name']} {preset['sims']}s)"
                    elif var_key.startswith("custom_"):
                        name = f"AlphaZero ({base_id} — {sims}s)"
                    else:
                        name = f"AlphaZero ({base_id})"
                elif kind == "heuristic":
                    if var_key in ("ultra", "8ply", "grandmaster", "gm"):
                        name = "Heuristic Baseline — 👑 Grandmaster (8-ply α-β)"
                    elif var_key in ("deep", "6ply", "master"):
                        name = "Heuristic Baseline — 🧠 Master (6-ply α-β)"
                    elif var_key in ("fast", "2ply", "blitz"):
                        name = "Heuristic Baseline — ⚡ Blitz (2-ply α-β)"
                    else:
                        name = "Heuristic Baseline — ⚖️ Standard (4-ply α-β)"
                else:
                    name = norm_id

            new_player = PlayerRecord(
                id=norm_id,
                name=name,
                kind=kind,
                rating=DEFAULT_INITIAL_RATING,
                peak_rating=DEFAULT_INITIAL_RATING,
                metadata=metadata or {},
            )
            self.players[norm_id] = new_player
            self._save()
            return new_player

    def record_match(
        self,
        p1_id: str,
        p2_id: str,
        winner: int,  # 0 = P1 win, 1 = P2 win, -1 = draw
        p1_name: Optional[str] = None,
        p2_name: Optional[str] = None,
        p1_kind: Optional[str] = None,
        p2_kind: Optional[str] = None,
        ply_count: int = 0,
        scores: Optional[List[int]] = None,
        termination_reason: Optional[str] = None,
        terrain_mode: Optional[str] = None,
        source: str = "visualizer",
        match_id: Optional[str] = None,
    ) -> MatchRecord:
        """
        Records a completed game and updates Elo ratings accordingly.
        winner: 0 (p1 wins), 1 (p2 wins), -1 (draw).
        """
        with self._lock:
            p1 = self.get_or_create_player(p1_id, kind=p1_kind, name=p1_name)
            p2 = self.get_or_create_player(p2_id, kind=p2_kind, name=p2_name)

            if winner == 0:
                s1, s2 = 1.0, 0.0
                win_name = p1.name
            elif winner == 1:
                s1, s2 = 0.0, 1.0
                win_name = p2.name
            else:
                s1, s2 = 0.5, 0.5
                win_name = "Draw"

            r1_before = p1.rating
            r2_before = p2.rating

            r1_after, r2_after, d1, d2 = calculate_elo_update(
                rating_a=r1_before,
                rating_b=r2_before,
                score_a=s1,
                games_a=p1.games_played,
                games_b=p2.games_played,
            )

            # Update stats
            p1.rating = r1_after
            p1.peak_rating = max(p1.peak_rating, r1_after)
            p1.games_played += 1

            p2.rating = r2_after
            p2.peak_rating = max(p2.peak_rating, r2_after)
            p2.games_played += 1

            if winner == 0:
                p1.wins += 1
                p2.losses += 1
            elif winner == 1:
                p2.wins += 1
                p1.losses += 1
            else:
                p1.draws += 1
                p2.draws += 1

            now_str = datetime.now(timezone.utc).isoformat()
            p1.last_played = now_str
            p2.last_played = now_str

            m_id = match_id or f"m_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            record = MatchRecord(
                match_id=m_id,
                timestamp=now_str,
                player1_id=p1.id,
                player2_id=p2.id,
                player1_name=p1.name,
                player2_name=p2.name,
                winner=winner,
                winner_name=win_name,
                score_p1=s1,
                score_p2=s2,
                p1_rating_before=r1_before,
                p1_rating_after=r1_after,
                p1_delta=d1,
                p2_rating_before=r2_before,
                p2_rating_after=r2_after,
                p2_delta=d2,
                ply_count=ply_count,
                scores=scores or [0, 0],
                termination_reason=termination_reason,
                terrain_mode=terrain_mode,
                source=source,
            )
            self.matches.append(record)
            self._save()
            return record

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Returns all players sorted by Elo rating descending, decorated with rank."""
        with self._lock:
            # Sort by rating desc, then wins desc, then games desc
            sorted_players = sorted(
                self.players.values(),
                key=lambda p: (p.rating, p.wins, p.games_played),
                reverse=True,
            )
            leaderboard: List[Dict[str, Any]] = []
            for rank, p in enumerate(sorted_players, start=1):
                p_dict = p.to_dict()
                p_dict["rank"] = rank
                leaderboard.append(p_dict)
            return leaderboard

    def get_matches(
        self, limit: int = 50, player_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns recent matches in reverse chronological order."""
        with self._lock:
            matches_to_filter = self.matches
            if player_id:
                matches_to_filter = [
                    m for m in matches_to_filter
                    if m.player1_id == player_id or m.player2_id == player_id
                ]
            recent = list(reversed(matches_to_filter[-limit:]))
            return [m.to_dict() for m in recent]

    def reset(self, keep_players: bool = True) -> None:
        """Resets matches and ratings."""
        with self._lock:
            self.matches.clear()
            if keep_players:
                for p in self.players.values():
                    p.rating = DEFAULT_INITIAL_RATING
                    p.peak_rating = DEFAULT_INITIAL_RATING
                    p.games_played = 0
                    p.wins = 0
                    p.losses = 0
                    p.draws = 0
                    p.last_played = None
            else:
                self.players.clear()
                self._seed_defaults()
            self._save()
