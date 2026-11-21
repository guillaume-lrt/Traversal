"""
Interactive Local Web Visualizer & Human/Spectator Play Server for Traversal.

Features:
  - Play as Player 1 or Player 2 against any trained AlphaZero checkpoint (`best.pt`,
    `latest.pt`, `iter_*.pt`), `HeuristicAgent`, or Human.
  - AI vs. AI / Heuristic spectator mode with Play/Pause auto-step and full move history
    scrubber (Step Back, Step Forward, Jump to any ply, or branch from history).
  - Live AI Analysis Overlay: MCTS root value / win-rate gauge, raw neural network value,
    and Top Candidate Moves table (visits N, policy share pi, prior P, Q-value, win%)
    drawn as arrows on the board.
  - Interactive 3D Isometric Elevation View & 2D Top-Down Tactical Grid View with
    detailed Combat & Movement Inspector (showing exact Dijkstra movement cost, Explorer
    special moves, Vanguard charge bonus, height modifier, allied support, and defense).

Run with:
  .venv/bin/python visualizer.py --port 8000
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import threading
import time
import base64
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch

logger = logging.getLogger("visualizer")

from elo_system import EloDatabase, parse_agent_variant, VARIANT_PRESETS

from traversal_env import (
    TraversalGame,
    TERRAIN_MODES,
    PIECE_NAMES,
    PIECE_TYPES,
    PIECE_MOVEMENT,
    PIECE_STRENGTH,
    PIECE_SCORE,
    PIECE_EXPLORER,
    PIECE_VANGUARD,
    PIECE_SENTINEL,
    FINISH_TARGET,
    NUM_TARGETS,
)
from encoder import encode_both_static_tensors, encode_dynamic_tensor
from model import TraversalDualStemNet, load_model_from_checkpoint, masked_softmax
from mcts import MCTS, MCTSNode
from train import HeuristicAgent, search_best_heuristic_moves_numba, HEURISTIC_PRESETS


# ---------------------------------------------------------------------------
# Official traversalgame.com Board Codec & Online Game Import
# ---------------------------------------------------------------------------

OFFICIAL_CODEC_KEY = "ff7e3ca46483315761dcdf1ab63dc8ac"


def _fnv1a(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def _rotl32(x: int, k: int) -> int:
    return (((x & 0xFFFFFFFF) << k) | ((x & 0xFFFFFFFF) >> (32 - k))) & 0xFFFFFFFF


class _CodecPRNG:
    def __init__(self, seed: bytes) -> None:
        e = bytearray(seed) + b"\x00"
        self.s = [0] * 4
        for t in range(4):
            e[-1] = t
            val = _fnv1a(e)
            self.s[t] = val if val < 0x80000000 else val - 0x100000000
        if all(x == 0 for x in self.s):
            self.s[0] = 1

    def next_u32(self) -> int:
        s0, s1, s2, s3 = self.s
        res = (_rotl32((s1 * 5) & 0xFFFFFFFF, 7) * 9) & 0xFFFFFFFF
        t = (s1 << 9) & 0xFFFFFFFF
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl32(s3, 11)

        def to_i32(v: int) -> int:
            v &= 0xFFFFFFFF
            return v if v < 0x80000000 else v - 0x100000000

        self.s = [to_i32(s0), to_i32(s1), to_i32(s2), to_i32(s3)]
        return res


def decode_traversal_terrain(
    b64_blob: str, game_id: str, key: str = OFFICIAL_CODEC_KEY
) -> List[List[int]]:
    """
    Decodes the obfuscated base64 terrain blob `s` returned by traversalgame.com
    into a 10x10 list of tile elevation integers.
    """
    raw = base64.b64decode(b64_blob)
    if len(raw) < 13 or raw[0] != 1:
        raise ValueError("Invalid or unknown terrain blob header")
    rows, cols = raw[1], raw[2]
    c = rows * cols
    if len(raw) != 11 + c + 2:
        raise ValueError(
            f"Terrain blob size mismatch (expected {11 + c + 2}, got {len(raw)})"
        )
    salt = raw[3:11]
    payload = raw[11 : 11 + c]
    seed = key.encode("utf-8") + b"\x00" + game_id.encode("utf-8") + b"\x00" + salt
    rng = _CodecPRNG(seed)

    perm = list(range(c))
    for t in range(c - 1, 0, -1):
        o = rng.next_u32() % (t + 1)
        perm[t], perm[o] = perm[o], perm[t]

    tiles = bytearray(c)
    for i in range(c):
        tiles[perm[i]] = payload[i] ^ (rng.next_u32() & 0xFF)

    chk = _fnv1a(tiles) & 0xFFFF
    expected_chk = (raw[-2] << 8) | raw[-1]
    if chk != expected_chk:
        raise ValueError(f"Terrain checksum mismatch ({chk} != {expected_chk})")

    return [list(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]


def parse_online_import_request(
    game_url_or_id: str,
    cookie: Optional[str] = None,
    raw_json: Optional[Any] = None,
) -> Tuple[str, Dict[str, str], Optional[Dict[str, Any]]]:
    """
    Normalizes user input for importing online games from traversalgame.com.

    Supports:
      - Clean URL or UUID (e.g. 'https://www.traversalgame.com/game/<uuid>' or '<uuid>')
      - Direct pasted cURL command in URL, Cookie, or JSON input
      - Full-state JSON or terrain blob in URL, Cookie, or JSON input
      - Browser cookies copied from DevTools (Storage / Stockage tab):
        * Raw JWT token (with or without 'traversal_session=' or 'Bearer')
        * Full cookie table copy (tab-separated lines from Firefox / Chrome)
        * Semicolon-separated cookie strings
      - Detection of truncated/partial tokens (with informative error guidance)

    Returns:
      (game_id, http_headers, pre_parsed_game_data)
    """
    raw_url = str(game_url_or_id or "").strip()
    raw_cookie = str(cookie or "").strip()
    raw_json_str = str(raw_json or "").strip() if isinstance(raw_json, str) else ""

    all_text_inputs = [raw_url, raw_cookie, raw_json_str]

    # 1. Check for cURL command in any of the fields
    curl_input = next((s for s in all_text_inputs if "curl " in s or s.startswith("curl")), None)
    extracted_url = ""
    extracted_cookie_hdr = ""
    extracted_bearer = ""

    if curl_input:
        # Extract target URL
        url_m = re.search(r"[\'\"](https?://[^\'\"]+)[\'\"]", curl_input)
        if url_m:
            extracted_url = url_m.group(1)
        # Extract Cookie header
        c_m = re.search(r"-(?:H|-header)\s+[\'\"][Cc]ookie:\s*([^\'\"]+)[\'\"]", curl_input)
        if c_m:
            extracted_cookie_hdr = c_m.group(1)
        # Extract Authorization header
        a_m = re.search(r"-(?:H|-header)\s+[\'\"][Aa]uthorization:\s*(?:Bearer\s+)?([^\'\"]+)[\'\"]", curl_input)
        if a_m:
            extracted_bearer = a_m.group(1)

    # 2. Check for direct JSON in any of the fields
    parsed_game_data: Optional[Dict[str, Any]] = None
    if isinstance(raw_json, dict):
        parsed_game_data = raw_json
    else:
        for cand in [raw_json_str, raw_url]:
            if cand and (cand.startswith("{") or '"s":' in cand or '"board":' in cand or '"move_history":' in cand):
                try:
                    parsed_game_data = json.loads(cand)
                    break
                except Exception:
                    pass

    # Unwrap nested envelopes if present: {"game": {...}} or {"data": {...}}
    if isinstance(parsed_game_data, dict):
        if "game" in parsed_game_data and isinstance(parsed_game_data["game"], dict):
            inner = parsed_game_data["game"]
            if "move_history" not in inner and "move_history" in parsed_game_data:
                inner["move_history"] = parsed_game_data["move_history"]
            parsed_game_data = inner
        elif "data" in parsed_game_data and isinstance(parsed_game_data["data"], dict):
            inner = parsed_game_data["data"]
            if "move_history" not in inner and "move_history" in parsed_game_data:
                inner["move_history"] = parsed_game_data["move_history"]
            parsed_game_data = inner

    # 3. Extract Game UUID
    candidate_sources = [extracted_url, raw_url]
    if isinstance(parsed_game_data, dict):
        for k in ("game_id", "id", "gameId"):
            if k in parsed_game_data and isinstance(parsed_game_data[k], str):
                candidate_sources.append(parsed_game_data[k])

    game_id = ""
    for cand in candidate_sources:
        if cand:
            uuid_m = re.search(
                r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
                cand,
            )
            if uuid_m:
                game_id = uuid_m.group(1).lower()
                break

    # 4. Parse Cookies and Bearer tokens
    cookie_sources = [s for s in [extracted_cookie_hdr, raw_cookie, os.getenv("TRAVERSAL_COOKIE", "")] if s]
    cookie_dict: Dict[str, str] = {}

    for src in cookie_sources:
        for line in src.splitlines():
            line = line.strip()
            if not line:
                continue

            # DevTools table copy (tab or multiple spaces separated)
            parts = re.split(r"\t+|\s{2,}", line)
            if len(parts) >= 2 and re.match(r"^[a-zA-Z0-9_\-]+$", parts[0].strip()):
                cookie_dict[parts[0].strip()] = parts[1].strip().strip("\"'")
                continue

            # Standard semicolon separated key=val
            if "=" in line:
                for pair in line.split(";"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        cookie_dict[k.strip()] = v.strip().strip("\"'")
                continue

            # DevTools single row key: value format
            if ":" in line and not line.startswith("http"):
                k, v = line.split(":", 1)
                k_clean = k.strip().lstrip("▼").lstrip("▶").strip().strip("\"'")
                if re.match(r"^[a-zA-Z0-9_\-]+$", k_clean):
                    cookie_dict[k_clean] = v.strip().strip("\"'")
                    continue

            # Line starting with Bearer
            if line.startswith("Bearer "):
                extracted_bearer = line[7:].strip().strip("\"'")
                continue

            # Line starting with eyJ (JWT token)
            if line.startswith("eyJ"):
                extracted_bearer = line.strip().strip("\"'")
                cookie_dict["traversal_session"] = extracted_bearer
                continue

            # Space-separated tokens (e.g. user pasted multiple values from DevTools)
            tokens = line.split()
            for tok in tokens:
                tok_clean = tok.strip().strip("\"'").rstrip(";")
                if tok_clean.startswith("eyJ"):
                    extracted_bearer = tok_clean
                    cookie_dict["traversal_session"] = tok_clean
                elif re.match(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                    tok_clean,
                ):
                    cookie_dict["traversal_refresh"] = tok_clean

    # If raw_cookie was provided but didn't match any format, check if it's a bare token or truncated token
    if raw_cookie and not cookie_dict and not extracted_bearer:
        cleaned_tok = raw_cookie.strip().strip("\"'")
        if cleaned_tok.startswith("eyJ"):
            extracted_bearer = cleaned_tok
            cookie_dict["traversal_session"] = cleaned_tok
        elif len(cleaned_tok) < 150 and "." not in cleaned_tok:
            # Likely truncated signature or incomplete value (e.g. ~86 chars)
            raise ValueError(
                f"The token entered ({len(cleaned_tok)} chars) appears to be incomplete or truncated.\n\n"
                "Traversal session JWT tokens start with 'eyJ...' and are ~320 characters long.\n"
                "In Firefox DevTools (F12) → Stockage → Cookies → traversalgame.com:\n"
                "  1. Click 'traversal_session' in the table.\n"
                "  2. Copy its complete value from the right-hand 'Données' pane.\n\n"
                "Or even simpler: In DevTools → 'Réseau' (Network) tab, right-click 'full-state' → "
                "Copier → 'Copier comme cURL', and paste it directly into the URL/cURL input!"
            )
        else:
            cookie_dict["traversal_session"] = cleaned_tok
            extracted_bearer = cleaned_tok

    if "traversal_session" in cookie_dict and not extracted_bearer:
        extracted_bearer = cookie_dict["traversal_session"]
    elif extracted_bearer and "traversal_session" not in cookie_dict:
        cookie_dict["traversal_session"] = extracted_bearer

    # Build HTTP headers mimicking official client
    headers: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.traversalgame.com",
        "Referer": "https://www.traversalgame.com/",
        "X-Client-Platform": "web",
        "X-Client-Protocol": "2",
        "X-Client-Features": "presence,terms",
    }
    if cookie_dict:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
    if extracted_bearer:
        headers["Authorization"] = f"Bearer {extracted_bearer}"

    return game_id, headers, parsed_game_data


# ---------------------------------------------------------------------------
# Terrain Path Precomputation & Optimal Finish Route Search
# ---------------------------------------------------------------------------

_TERRAIN_PATH_CACHE: Dict[bytes, Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}


def _get_terrain_path_cache(terrain: np.ndarray):
    """
    Computes and caches all-pairs Dijkstra distance and predecessor tables
    plus backwards BFS turn-to-finish tables for all 3 piece types and 2 players.
    Executes in ~18ms once per unique terrain and ~0ms thereafter.
    """
    key = terrain.tobytes()
    if key in _TERRAIN_PATH_CACHE:
        return _TERRAIN_PATH_CACHE[key]

    ortho = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cost = np.full((10, 10, 10, 10), 999, dtype=np.int32)
    pred = np.full((10, 10, 10, 10, 2), -1, dtype=np.int32)

    import heapq
    from collections import deque

    for sr in range(10):
        for sc in range(10):
            d = cost[sr, sc]
            d[sr, sc] = 0
            q = [(0, sr, sc)]
            visited = np.zeros((10, 10), dtype=bool)
            while q:
                c, r, col = heapq.heappop(q)
                if visited[r, col]:
                    continue
                visited[r, col] = True
                for dr, dc in ortho:
                    nr, nc = r + dr, col + dc
                    if 0 <= nr < 10 and 0 <= nc < 10 and not visited[nr, nc]:
                        ncost = c + abs(int(terrain[nr, nc]) - int(terrain[r, col])) + 1
                        if ncost < d[nr, nc]:
                            d[nr, nc] = ncost
                            pred[sr, sc, nr, nc] = [r, col]
                            heapq.heappush(q, (ncost, nr, nc))

    min_turns = {}
    for p_type in (0, 1, 2):
        budget = 5 if p_type == 0 else (4 if p_type == 1 else 3)
        can_reach_1turn = np.zeros((10, 10, 10, 10), dtype=bool)
        is_special_1turn = np.zeros((10, 10, 10, 10), dtype=bool)
        for r1 in range(10):
            for c1 in range(10):
                for r2 in range(10):
                    for c2 in range(10):
                        if r1 == r2 and c1 == c2:
                            continue
                        if cost[r1, c1, r2, c2] <= budget:
                            can_reach_1turn[r1, c1, r2, c2] = True
                        elif p_type == 0:  # Explorer special moves
                            dr = abs(r2 - r1)
                            dc = abs(c2 - c1)
                            if (dr == 1 and dc == 1) or (dr == 2 and dc == 0) or (dr == 0 and dc == 2):
                                can_reach_1turn[r1, c1, r2, c2] = True
                                is_special_1turn[r1, c1, r2, c2] = True

        for player in (0, 1):
            target_row = 9 if player == 0 else 0
            mt = np.full((10, 10), 99, dtype=np.int32)
            queue = deque()
            for r in range(10):
                for c in range(10):
                    if r == target_row:
                        mt[r, c] = 1
                        queue.append((r, c))
                    else:
                        for tc in range(10):
                            if cost[r, c, target_row, tc] <= budget - 1:
                                mt[r, c] = 1
                                queue.append((r, c))
                                break
            while queue:
                vr, vc = queue.popleft()
                d = mt[vr, vc]
                for ur in range(10):
                    for uc in range(10):
                        if can_reach_1turn[ur, uc, vr, vc]:
                            if mt[ur, uc] > d + 1:
                                mt[ur, uc] = d + 1
                                queue.append((ur, uc))

            min_turns[(p_type, player)] = (mt, can_reach_1turn, is_special_1turn)

    res = (cost, pred, min_turns)
    _TERRAIN_PATH_CACHE[key] = res
    return res


def compute_piece_finish_path(
    terrain: np.ndarray,
    player: int,
    piece_idx: int,
    r0: int,
    c0: int,
    friendly_at: Optional[Dict[Tuple[int, int], int]] = None,
    enemy_at: Optional[Dict[Tuple[int, int], int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Computes the shortest path to finish and exact number of turns it would take
    for a piece at (r0, c0) moving toward player's finish row (Row 9 for P1, Row 0 for P2).
    Respects terrain elevation movement cost, piece movement budget, Explorer special moves,
    and the final step-off-board finishing requirement.
    """
    if r0 < 0 or c0 < 0:
        return None

    p_type = int(PIECE_TYPES[piece_idx])
    budget = int(PIECE_MOVEMENT[piece_idx])
    target_row = 9 if player == 0 else 0

    cost, pred, min_turns_dict = _get_terrain_path_cache(terrain)
    mt, can_reach_1turn, is_special_1turn = min_turns_dict[(p_type, player)]

    total_turns = int(mt[r0, c0])
    if total_turns >= 90:
        return None

    turn_segments: List[Dict[str, Any]] = []
    all_tiles: List[List[int]] = [[r0, c0]]
    total_cost = 0
    curr = [r0, c0]

    for turn_idx in range(1, total_turns + 1):
        d_curr = int(mt[curr[0], curr[1]])
        if d_curr == 1:
            if curr[0] == target_row:
                turn_segments.append(
                    {
                        "turn": turn_idx,
                        "from": list(curr),
                        "to": list(curr),
                        "path": [list(curr)],
                        "action": "finish",
                        "cost": 1,
                        "desc": f"Turn {turn_idx}: Step off board from Row {target_row} to Finish (+1 pt)",
                    }
                )
                total_cost += 1
                break

            best_tc = -1
            best_cost = 999
            for tc in range(10):
                c_val = int(cost[curr[0], curr[1], target_row, tc])
                if c_val <= budget - 1 and c_val < best_cost:
                    best_cost = c_val
                    best_tc = tc

            if best_tc == -1:
                break

            path_pts: List[List[int]] = []
            trace = [target_row, best_tc]
            while trace != curr:
                path_pts.append(list(trace))
                p_r, p_c = pred[curr[0], curr[1], trace[0], trace[1]]
                trace = [int(p_r), int(p_c)]
            path_pts.append(list(curr))
            path_pts.reverse()

            turn_segments.append(
                {
                    "turn": turn_idx,
                    "from": list(curr),
                    "to": [target_row, best_tc],
                    "path": path_pts,
                    "action": "finish",
                    "cost": int(best_cost) + 1,
                    "desc": f"Turn {turn_idx}: Move ({curr[0]},{curr[1]}) \u2192 ({target_row},{best_tc}) and step off to FINISH",
                }
            )
            for pt in path_pts[1:]:
                all_tiles.append(pt)
            total_cost += int(best_cost) + 1
            break
        else:
            candidates: List[Tuple[int, int, int, int]] = []
            for nr in range(10):
                for nc in range(10):
                    if can_reach_1turn[curr[0], curr[1], nr, nc] and int(mt[nr, nc]) == d_curr - 1:
                        c_val = (
                            int(cost[curr[0], curr[1], nr, nc])
                            if not is_special_1turn[curr[0], curr[1], nr, nc]
                            else 1
                        )
                        progress = nr if player == 0 else (9 - nr)
                        blocked_penalty = 0
                        if turn_idx == 1 and friendly_at and (nr, nc) in friendly_at:
                            blocked_penalty = 50
                        candidates.append((progress - blocked_penalty, -c_val, nr, nc))

            if not candidates:
                break
            candidates.sort(reverse=True)
            _, _, next_r, next_c = candidates[0]

            if is_special_1turn[curr[0], curr[1], next_r, next_c]:
                path_pts = [list(curr), [next_r, next_c]]
                cost_val = 1
            else:
                path_pts = []
                trace = [next_r, next_c]
                while trace != curr:
                    path_pts.append(list(trace))
                    p_r, p_c = pred[curr[0], curr[1], trace[0], trace[1]]
                    trace = [int(p_r), int(p_c)]
                path_pts.append(list(curr))
                path_pts.reverse()
                cost_val = int(cost[curr[0], curr[1], next_r, next_c])

            turn_segments.append(
                {
                    "turn": turn_idx,
                    "from": list(curr),
                    "to": [next_r, next_c],
                    "path": path_pts,
                    "action": "move",
                    "cost": int(cost_val),
                    "desc": f"Turn {turn_idx}: Move ({curr[0]},{curr[1]}) \u2192 ({next_r},{next_c}) (cost {cost_val}/{budget})",
                }
            )
            for pt in path_pts[1:]:
                all_tiles.append(pt)
            total_cost += int(cost_val)
            curr = [next_r, next_c]

    waypoints = [seg["to"] for seg in turn_segments]
    total_steps = len(all_tiles) - 1
    summary = f"{total_turns} turns ({total_cost} movement cost across {total_steps} steps)"

    return {
        "turns": total_turns,
        "total_cost": total_cost,
        "total_steps": total_steps,
        "waypoints": waypoints,
        "path": all_tiles,
        "turn_segments": turn_segments,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Detailed Combat & Reachability Breakdown for UI Tooltips
# ---------------------------------------------------------------------------


def _compute_single_player_inspector(
    game: TraversalGame, player: int
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Computes per-piece movement, reachability, and combat breakdown data
    for `player` (0 or 1), regardless of whether it is currently `player`'s turn.
    """
    opp = 1 - player
    is_active_turn = (player == game.current_player) and (not game.done)
    terrain = game.terrain
    legal_dest, can_finish, d_min_all, max_attack, my_def, opp_def = (
        game.get_player_combat_and_reach(player)
    )

    friendly_at: Dict[Tuple[int, int], int] = {}
    enemy_at: Dict[Tuple[int, int], int] = {}
    for k in range(6):
        r, c = int(game.pos[player, k, 0]), int(game.pos[player, k, 1])
        if r >= 0 and c >= 0:
            friendly_at[(r, c)] = k
        er, ec = int(game.pos[opp, k, 0]), int(game.pos[opp, k, 1])
        if er >= 0 and ec >= 0:
            enemy_at[(er, ec)] = k

    pieces_info: List[Dict[str, Any]] = []
    for i in range(6):
        r0, c0 = int(game.pos[player, i, 0]), int(game.pos[player, i, 1])
        status = int(game.piece_status[player, i])
        p_type = int(PIECE_TYPES[i])
        move_budget = int(PIECE_MOVEMENT[i])
        base_str = int(PIECE_STRENGTH[i])

        moves_list: List[Dict[str, Any]] = []
        forbidden_moves: List[Dict[str, Any]] = []
        blocked_attacks: List[Dict[str, Any]] = []
        finish_path = None

        if r0 >= 0 and c0 >= 0:
            h0 = int(terrain[r0, c0])
            finish_path = compute_piece_finish_path(
                terrain=terrain,
                player=player,
                piece_idx=i,
                r0=r0,
                c0=c0,
                friendly_at=friendly_at,
                enemy_at=enemy_at,
            )

            # Compute unblocked terrain movement cost from (r0, c0) up to move_budget
            d_unblocked = np.full((10, 10), 999, dtype=np.int32)
            d_unblocked[r0, c0] = 0
            bucket: List[List[Tuple[int, int]]] = [[] for _ in range(move_budget + 1)]
            bucket[0].append((r0, c0))
            for cost in range(move_budget):
                for r_cur, c_cur in bucket[cost]:
                    if d_unblocked[r_cur, c_cur] < cost:
                        continue
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r_cur + dr, c_cur + dc
                        if 0 <= nr < 10 and 0 <= nc < 10:
                            step_c = abs(int(terrain[nr, nc]) - int(terrain[r_cur, c_cur])) + 1
                            ncost = cost + step_c
                            if ncost <= move_budget and ncost < d_unblocked[nr, nc]:
                                d_unblocked[nr, nc] = ncost
                                bucket[ncost].append((nr, nc))

            for rt in range(10):
                for ct in range(10):
                    if rt == r0 and ct == c0:
                        continue

                    d_val = int(d_min_all[i, rt, ct])
                    std_reachable = d_val <= move_budget
                    d_unb = int(d_unblocked[rt, ct])
                    std_unblocked = d_unb <= move_budget
                    special_type: Optional[str] = None

                    if p_type == PIECE_EXPLORER:
                        dr = abs(rt - r0)
                        dc = abs(ct - c0)
                        if dr == 1 and dc == 1:
                            special_type = "diagonal"
                        elif (dr == 2 and dc == 0) or (dr == 0 and dc == 2):
                            special_type = "leap"

                    if not std_unblocked and not std_reachable and special_type is None:
                        continue

                    ht = int(terrain[rt, ct])
                    h_mod = 1 if h0 > ht else (-1 if h0 < ht else 0)
                    v_bonus = (
                        1
                        if (
                            p_type == PIECE_VANGUARD
                            and (d_val == 4 or (std_unblocked and d_unb == 4))
                        )
                        else 0
                    )

                    ally_sup = 0
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rt + dr, ct + dc
                        adj_k = friendly_at.get((nr, nc), -1)
                        if adj_k != -1 and adj_k != i:
                            ally_sup += int(PIECE_STRENGTH[adj_k])

                    eff_attack = base_str + h_mod + v_bonus + ally_sup
                    is_enemy = (rt, ct) in enemy_at
                    is_friendly = (rt, ct) in friendly_at
                    is_legal = bool(legal_dest[i, rt, ct] == 1)

                    combat_detail = None
                    if is_enemy:
                        opp_k = enemy_at[(rt, ct)]
                        def_base = int(PIECE_STRENGTH[opp_k])
                        def_total = int(opp_def[rt, ct])
                        def_sup = def_total - def_base
                        can_cap = (
                            eff_attack >= def_total
                            and not (special_type is not None and not std_reachable)
                        )
                        combat_detail = {
                            "enemy_piece_idx": opp_k,
                            "enemy_piece_name": PIECE_NAMES[opp_k],
                            "base_str": base_str,
                            "height_mod": h_mod,
                            "vanguard_bonus": v_bonus,
                            "ally_support": ally_sup,
                            "total_attack": eff_attack,
                            "def_base": def_base,
                            "def_support": def_sup,
                            "total_defense": def_total,
                            "can_capture": can_cap,
                        }

                    action_code = (
                        game.encode_action(piece_idx=i, target_r=rt, target_c=ct)
                        if (is_active_turn and is_legal)
                        else None
                    )
                    entry = {
                        "r": rt,
                        "c": ct,
                        "d_min": d_val if std_reachable else (d_unb if std_unblocked else None),
                        "d_unblocked": d_unb if std_unblocked else None,
                        "special": special_type,
                        "is_enemy": is_enemy,
                        "is_friendly": is_friendly,
                        "is_vanguard_bonus": bool(
                            p_type == PIECE_VANGUARD
                            and ((std_reachable and d_val == 4) or (std_unblocked and d_unb == 4))
                        ),
                        "combat": combat_detail,
                        "action": action_code,
                        "is_legal": is_legal,
                    }

                    if is_legal:
                        moves_list.append(entry)
                    else:
                        if is_friendly:
                            f_idx = friendly_at[(rt, ct)]
                            forbidden_reason = "friendly_occupied"
                            forbidden_desc = f"Occupied by friendly {PIECE_NAMES[f_idx]}"
                        elif is_enemy:
                            if special_type is not None and not std_reachable:
                                forbidden_reason = "special_cannot_capture"
                                forbidden_desc = f"Explorer {special_type} move cannot be used to capture enemy pieces"
                                blocked_attacks.append(entry)
                            elif not std_reachable and special_type is None:
                                forbidden_reason = "enemy_path_blocked"
                                forbidden_desc = f"Path to enemy {PIECE_NAMES[opp_k]} is blocked by other pieces"
                            else:
                                forbidden_reason = "combat_failed"
                                forbidden_desc = (
                                    f"Cannot capture: Attack ({eff_attack}) < Defense ({def_total})"
                                )
                                blocked_attacks.append(entry)
                        else:
                            forbidden_reason = "path_blocked"
                            forbidden_desc = "Path blocked by other piece(s)"

                        entry["forbidden_reason"] = forbidden_reason
                        entry["forbidden_desc"] = forbidden_desc
                        forbidden_moves.append(entry)

        finish_action = (
            game.encode_action(piece_idx=i, finish=True)
            if (r0 >= 0 and is_active_turn and bool(can_finish[i]))
            else None
        )

        pieces_info.append(
            {
                "player": player,
                "piece_idx": i,
                "name": PIECE_NAMES[i],
                "type": p_type,
                "movement": move_budget,
                "strength": base_str,
                "score_val": int(PIECE_SCORE[i]),
                "r": r0,
                "c": c0,
                "status": status,
                "is_active_turn": is_active_turn,
                "can_finish": bool(can_finish[i]) and (r0 >= 0),
                "finish_action": finish_action,
                "is_sentinel_instant_win": bool(
                    can_finish[i]
                    and (r0 >= 0)
                    and game.total_pieces_finished == 0
                    and p_type == PIECE_SENTINEL
                ),
                "legal_moves": moves_list,
                "forbidden_moves": forbidden_moves,
                "blocked_attacks": blocked_attacks,
                "finish_path": finish_path,
            }
        )

    return pieces_info, my_def


def compute_detailed_ui_inspector(game: TraversalGame) -> Dict[str, Any]:
    """
    Computes rich per-piece movement, reachability, and combat breakdown data
    in physical coordinates for BOTH Player 1 and Player 2 so the frontend can
    display possible moves and combat math for any piece clicked at any time.
    """
    p0_info, p0_def = _compute_single_player_inspector(game, 0)
    p1_info, p1_def = _compute_single_player_inspector(game, 1)

    p = game.current_player
    cur_info = p0_info if p == 0 else p1_info
    my_def = p0_def if p == 0 else p1_def
    opp_def = p1_def if p == 0 else p0_def

    return {
        "players_pieces": [p0_info, p1_info],
        "player_def": [p0_def.astype(int).tolist(), p1_def.astype(int).tolist()],
        "current_player_pieces": cur_info,
        "my_def": my_def.astype(int).tolist(),
        "opp_def": opp_def.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Session & Model Manager
# ---------------------------------------------------------------------------


class VisualizerSession:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        elo_db: Optional[EloDatabase] = None,
    ) -> None:
        self.lock = threading.RLock()
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng()

        # Loaded neural network models cache: path -> (mtime, TraversalDualStemNet, meta)
        self.model_cache: Dict[str, Tuple[float, TraversalDualStemNet, Dict[str, Any]]] = {}
        self.meta_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.heuristic_agent = HeuristicAgent(rng=self.rng)

        # Elo system and rating database
        self.elo_db = elo_db if elo_db is not None else EloDatabase()
        self.human_player_name: str = "Guillaume"
        self.match_session_id: str = uuid.uuid4().hex
        self.match_recorded: bool = False
        self.last_elo_update: Optional[Dict[str, Any]] = None
        self.tournament_status: Dict[str, Any] = {
            "running": False,
            "progress": 0,
            "total": 0,
            "current": "",
            "last_result": "",
        }

        # Active game configuration
        self.p1_agent: str = "human"
        self.p2_agent: str = "best.pt"
        self.p1_sims: int = 96
        self.p2_sims: int = 96
        self.terrain_mode: str = "random"

        # History list for step-by-step playback:
        # Each item is (TraversalGame clone, move_record_dict_or_None, analysis_dict_or_None)
        self.history: List[Tuple[TraversalGame, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
        self.cursor: int = 0

        # Simulation Mode: test variations and explore branches without affecting the real game
        self.is_simulating: bool = False
        self.saved_real_history: Optional[List[Tuple[TraversalGame, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None
        self.saved_real_cursor: int = 0
        self.sim_start_ply: int = 0
        self.sim_start_cursor: int = 0

        self.new_game(
            terrain_mode="random",
            p1_agent="human",
            p2_agent=self._default_ai_agent(),
            p1_sims=96,
            p2_sims=96,
        )

    def _discover_checkpoint_relpaths(self) -> List[str]:
        with self.lock:
            if not os.path.isdir(self.checkpoint_dir):
                return []
            pt_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pt")) + glob.glob(
                os.path.join(self.checkpoint_dir, "*", "*.pt")
            )
            # Filter out non-model files (e.g. replay buffers, optimizers, temporary files)
            pt_files = [
                p
                for p in pt_files
                if not os.path.basename(p).startswith(("replay_buffer", "buffer", "optimizer"))
                and "replay_buffer" not in os.path.basename(p)
                and not p.endswith((".tmp", ".temp"))
            ]
            priority = {"best.pt": 0, "latest.pt": 1}

            def _sort_key(full_path: str) -> Tuple[int, float, int, str]:
                rel = os.path.relpath(full_path, self.checkpoint_dir).replace("\\", "/")
                parent = os.path.dirname(rel)
                base = os.path.basename(rel)
                # Prefer newer experiment folders first, and best.pt / latest.pt first within each folder
                dir_mtime = (
                    os.path.getmtime(os.path.join(self.checkpoint_dir, parent))
                    if parent
                    else 0.0
                )
                return (0 if parent else 1, -dir_mtime, priority.get(base, 2), rel)

            pt_files.sort(key=_sort_key)
            return [
                os.path.relpath(p, self.checkpoint_dir).replace("\\", "/") for p in pt_files
            ]

    def set_human_player_name(self, name: str) -> None:
        with self.lock:
            clean_name = name.strip() or "Guillaume"
            self.human_player_name = clean_name
            self.elo_db.get_or_create_player(clean_name, kind="human", name=clean_name)

    def _get_agent_rating(self, agent_id: str) -> float:
        with self.lock:
            target_id = self.human_player_name if agent_id == "human" else agent_id
            if target_id in self.elo_db.players:
                return self.elo_db.players[target_id].rating
            base, var_key, _ = parse_agent_variant(target_id)
            if base in self.elo_db.players:
                return self.elo_db.players[base].rating
            if f"{base}:standard" in self.elo_db.players:
                return self.elo_db.players[f"{base}:standard"].rating
            return 1500.0

    def _get_agent_games_played(self, agent_id: str) -> int:
        with self.lock:
            target_id = self.human_player_name if agent_id == "human" else agent_id
            if target_id in self.elo_db.players:
                return int(self.elo_db.players[target_id].games_played)
            base, var_key, _ = parse_agent_variant(target_id)
            if base in self.elo_db.players:
                return int(self.elo_db.players[base].games_played)
            if f"{base}:standard" in self.elo_db.players:
                return int(self.elo_db.players[f"{base}:standard"].games_played)
            return 0

    def _get_agent_display_name(self, agent_id: str) -> str:
        with self.lock:
            if agent_id == "human":
                return self.human_player_name
            p = self.elo_db.players.get(agent_id)
            if p and p.name:
                return p.name
            base_id, var_key, sims = parse_agent_variant(agent_id)
            if base_id == "heuristic":
                if var_key in ("ultra", "8ply", "grandmaster", "gm"):
                    return "Heuristic Baseline — 👑 Grandmaster (8-ply α-β)"
                elif var_key in ("deep", "6ply", "master"):
                    return "Heuristic Baseline — 🧠 Master (6-ply α-β)"
                elif var_key in ("fast", "2ply", "blitz"):
                    return "Heuristic Baseline — ⚡ Blitz (2-ply α-β)"
                return "Heuristic Baseline — ⚖️ Standard (4-ply α-β)"
            if var_key in VARIANT_PRESETS:
                preset = VARIANT_PRESETS[var_key]
                return f"AlphaZero ({base_id} — {preset['icon']} {preset['name']} {preset['sims']}s)"
            return f"AlphaZero ({agent_id})"

    def list_available_agents(self, only_best: bool = True) -> List[Dict[str, Any]]:
        with self.lock:
            human_elo = int(round(self._get_agent_rating("human")))
            human_games = self._get_agent_games_played("human")
            agents = [
                {
                    "id": "human",
                    "label": f"Human Player ({self.human_player_name}) [{human_elo} Elo]",
                    "kind": "human",
                    "elo_rating": human_elo,
                    "games_played": human_games,
                    "default_sims": 96,
                    "variant": "human",
                    "is_best": True,
                },
            ]

            active_bases = set()
            for cur_agent in (self.p1_agent, self.p2_agent):
                if cur_agent and cur_agent not in ("human", "heuristic"):
                    b_id, _, _ = parse_agent_variant(cur_agent)
                    active_bases.add(b_id)

            # Checkpoints with 3 difficulty variants each (Master, Standard, Blitz)
            for rel_path in self._discover_checkpoint_relpaths():
                is_best = os.path.basename(rel_path) == "best.pt"
                if only_best and not is_best and rel_path not in active_bases:
                    continue
                try:
                    meta = self._get_checkpoint_meta(rel_path)
                except Exception as e:
                    logger.warning("Could not load metadata for checkpoint '%s': %s", rel_path, e)
                    continue
                it = meta.get("iteration", "?")
                it_str = f" Iter {it}" if it != "?" else ""

                # Master (256 sims)
                master_id = f"{rel_path}:master"
                m_elo = int(round(self._get_agent_rating(master_id)))
                m_games = self._get_agent_games_played(master_id)
                agents.append(
                    {
                        "id": master_id,
                        "label": f"AlphaZero ({rel_path}{it_str}) — 🧠 Master (256 sims) [{m_elo} Elo]",
                        "kind": "alphazero",
                        "meta": meta,
                        "elo_rating": m_elo,
                        "games_played": m_games,
                        "default_sims": 256,
                        "variant": "master",
                        "base_id": rel_path,
                        "is_best": is_best,
                    }
                )

                # Standard (96 sims)
                standard_id = f"{rel_path}:standard"
                s_elo = int(round(self._get_agent_rating(standard_id)))
                s_games = self._get_agent_games_played(standard_id)
                agents.append(
                    {
                        "id": standard_id,
                        "label": f"AlphaZero ({rel_path}{it_str}) — ⚖️ Standard (96 sims) [{s_elo} Elo]",
                        "kind": "alphazero",
                        "meta": meta,
                        "elo_rating": s_elo,
                        "games_played": s_games,
                        "default_sims": 96,
                        "variant": "standard",
                        "base_id": rel_path,
                        "is_best": is_best,
                    }
                )

                # Blitz (32 sims)
                blitz_id = f"{rel_path}:blitz"
                b_elo = int(round(self._get_agent_rating(blitz_id)))
                b_games = self._get_agent_games_played(blitz_id)
                agents.append(
                    {
                        "id": blitz_id,
                        "label": f"AlphaZero ({rel_path}{it_str}) — ⚡ Blitz (32 sims) [{b_elo} Elo]",
                        "kind": "alphazero",
                        "meta": meta,
                        "elo_rating": b_elo,
                        "games_played": b_games,
                        "default_sims": 32,
                        "variant": "blitz",
                        "base_id": rel_path,
                        "is_best": is_best,
                    }
                )

            # Heuristic variants: Grandmaster (8-ply), Master (6-ply), Standard (4-ply), and Blitz (2-ply)
            h_gm_elo = int(round(self._get_agent_rating("heuristic:grandmaster")))
            agents.append(
                {
                    "id": "heuristic:grandmaster",
                    "label": f"Heuristic Baseline — 👑 Grandmaster (8-ply α-β) [{h_gm_elo} Elo]",
                    "kind": "heuristic",
                    "elo_rating": h_gm_elo,
                    "games_played": self._get_agent_games_played("heuristic:grandmaster"),
                    "default_sims": 512,
                    "variant": "grandmaster",
                    "base_id": "heuristic",
                }
            )
            h_mas_elo = int(round(self._get_agent_rating("heuristic:master")))
            agents.append(
                {
                    "id": "heuristic:master",
                    "label": f"Heuristic Baseline — 🧠 Master (6-ply α-β) [{h_mas_elo} Elo]",
                    "kind": "heuristic",
                    "elo_rating": h_mas_elo,
                    "games_played": self._get_agent_games_played("heuristic:master"),
                    "default_sims": 256,
                    "variant": "master",
                    "base_id": "heuristic",
                }
            )
            h_std_elo = int(round(self._get_agent_rating("heuristic:standard")))
            agents.append(
                {
                    "id": "heuristic:standard",
                    "label": f"Heuristic Baseline — ⚖️ Standard (4-ply α-β) [{h_std_elo} Elo]",
                    "kind": "heuristic",
                    "elo_rating": h_std_elo,
                    "games_played": self._get_agent_games_played("heuristic:standard"),
                    "default_sims": 96,
                    "variant": "standard",
                    "base_id": "heuristic",
                }
            )
            h_blitz_elo = int(round(self._get_agent_rating("heuristic:blitz")))
            agents.append(
                {
                    "id": "heuristic:blitz",
                    "label": f"Heuristic Baseline — ⚡ Blitz (2-ply α-β) [{h_blitz_elo} Elo]",
                    "kind": "heuristic",
                    "elo_rating": h_blitz_elo,
                    "games_played": self._get_agent_games_played("heuristic:blitz"),
                    "default_sims": 32,
                    "variant": "blitz",
                    "base_id": "heuristic",
                }
            )
            return agents

    def _default_ai_agent(self) -> str:
        with self.lock:
            rel_paths = self._discover_checkpoint_relpaths()
            for rel in rel_paths:
                if os.path.basename(rel) == "best.pt":
                    return f"{rel}:standard"
            if rel_paths:
                return f"{rel_paths[0]}:standard"
            return "heuristic:standard"

    def _get_checkpoint_meta(self, ckpt_name: str) -> Dict[str, Any]:
        with self.lock:
            base_name, _, _ = parse_agent_variant(ckpt_name)
            norm_rel = os.path.normpath(base_name).lstrip("/\\")
            if norm_rel.startswith(".."):
                raise ValueError(f"Invalid checkpoint path: {ckpt_name}")
            path = os.path.join(self.checkpoint_dir, norm_rel)
            if not os.path.isfile(path):
                cand = os.path.join(self.checkpoint_dir, os.path.basename(base_name))
                if os.path.isfile(cand):
                    path = cand
                else:
                    raise FileNotFoundError(
                        f"Checkpoint '{base_name}' not found in {self.checkpoint_dir}"
                    )

            mtime = os.path.getmtime(path)
            if base_name in self.model_cache:
                cached_mtime, _, cached_meta = self.model_cache[base_name]
                if abs(cached_mtime - mtime) < 1e-3:
                    return cached_meta

            if base_name in self.meta_cache:
                cached_mtime, cached_meta = self.meta_cache[base_name]
                if abs(cached_mtime - mtime) < 1e-3:
                    return cached_meta

            m = re.search(r"iter_(\d+)\.pt$", os.path.basename(path))
            file_it = int(m.group(1)) if m else None

            meta: Dict[str, Any] = {}
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    for k in ("iteration", "loss_total", "heuristic_win_rate", "best_win_rate"):
                        if k in ckpt:
                            val = ckpt[k]
                            meta[k] = float(val) if isinstance(val, (int, float)) else val
                elif any(
                    k.startswith(("static_stem", "dynamic_stem", "shared_trunk", "policy_", "value_"))
                    for k in ckpt.keys()
                ):
                    pass
                else:
                    raise ValueError(f"File '{base_name}' does not contain valid model parameters")
            else:
                raise ValueError(f"Unrecognized checkpoint format in '{base_name}'")

            if "iteration" not in meta and file_it is not None:
                meta["iteration"] = file_it

            self.meta_cache[base_name] = (mtime, meta)
            return meta

    def _get_or_load_model(
        self, ckpt_name: str
    ) -> Tuple[TraversalDualStemNet, Dict[str, Any]]:
        with self.lock:
            base_name, _, _ = parse_agent_variant(ckpt_name)
            norm_rel = os.path.normpath(base_name).lstrip("/\\")
            if norm_rel.startswith(".."):
                raise ValueError(f"Invalid checkpoint path: {ckpt_name}")
            path = os.path.join(self.checkpoint_dir, norm_rel)
            if not os.path.isfile(path):
                # Fallback to searching by basename if needed
                cand = os.path.join(self.checkpoint_dir, os.path.basename(base_name))
                if os.path.isfile(cand):
                    path = cand
                else:
                    raise FileNotFoundError(
                        f"Checkpoint '{base_name}' not found in {self.checkpoint_dir}"
                    )

            mtime = os.path.getmtime(path)
            if base_name in self.model_cache:
                cached_mtime, cached_model, cached_meta = self.model_cache[base_name]
                if abs(cached_mtime - mtime) < 1e-3:
                    return cached_model, cached_meta

            model, meta_loaded = load_model_from_checkpoint(path, device=self.device)
            meta: Dict[str, Any] = {}
            for k in ("iteration", "loss_total", "heuristic_win_rate", "best_win_rate"):
                if k in meta_loaded:
                    meta[k] = float(meta_loaded[k]) if isinstance(meta_loaded[k], (int, float)) else meta_loaded[k]
            model.eval()
            self.model_cache[base_name] = (mtime, model, meta)
            self.meta_cache[base_name] = (mtime, meta)
            return model, meta

    def new_game(
        self,
        terrain_mode: str = "random",
        p1_agent: Optional[str] = None,
        p2_agent: Optional[str] = None,
        p1_sims: Optional[int] = None,
        p2_sims: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if terrain_mode in TERRAIN_MODES:
                self.terrain_mode = terrain_mode
            if p1_agent is not None:
                self.p1_agent = p1_agent
            if p2_agent is not None:
                self.p2_agent = p2_agent
            if p1_sims is not None:
                self.p1_sims = max(1, min(2000, int(p1_sims)))
            if p2_sims is not None:
                self.p2_sims = max(1, min(2000, int(p2_sims)))

            game = TraversalGame(terrain_mode=self.terrain_mode, seed=seed)
            self.history = [(game.clone(), None, None)]
            self.cursor = 0
            self.match_session_id = uuid.uuid4().hex
            self.match_recorded = False
            self.last_elo_update = None
            self.is_simulating = False
            self.saved_real_history = None
            return self._serialize_current_state()

    def update_settings(
        self,
        p1_agent: Optional[str] = None,
        p2_agent: Optional[str] = None,
        p1_sims: Optional[int] = None,
        p2_sims: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if p1_agent is not None:
                self.p1_agent = p1_agent
            if p2_agent is not None:
                self.p2_agent = p2_agent
            if p1_sims is not None:
                self.p1_sims = max(1, min(2000, int(p1_sims)))
            if p2_sims is not None:
                self.p2_sims = max(1, min(2000, int(p2_sims)))
            return self._serialize_current_state()

    def jump_to_ply(self, target_cursor: int) -> Dict[str, Any]:
        with self.lock:
            self.cursor = max(0, min(len(self.history) - 1, int(target_cursor)))
            return self._serialize_current_state()

    def start_simulation(self) -> Dict[str, Any]:
        with self.lock:
            if self.is_simulating:
                return self._serialize_current_state()
            self.is_simulating = True
            self.saved_real_history = [
                (g.clone(), m.copy() if m else None, a.copy() if a else None)
                for (g, m, a) in self.history
            ]
            self.saved_real_cursor = self.cursor
            self.sim_start_cursor = self.cursor
            self.sim_start_ply = self.history[self.cursor][0].ply_count
            return self._serialize_current_state()

    def reset_simulation(self) -> Dict[str, Any]:
        with self.lock:
            if not self.is_simulating:
                return self._serialize_current_state()
            self.history = self.history[: self.sim_start_cursor + 1]
            self.cursor = self.sim_start_cursor
            return self._serialize_current_state()

    def exit_simulation(self, commit: bool = False) -> Dict[str, Any]:
        with self.lock:
            if not self.is_simulating:
                return self._serialize_current_state()
            if commit:
                self.is_simulating = False
                self.saved_real_history = None
                cur_game = self.history[self.cursor][0]
                if cur_game.done:
                    self._check_and_record_match(cur_game)
            else:
                if self.saved_real_history is not None:
                    self.history = self.saved_real_history
                    self.cursor = min(self.saved_real_cursor, len(self.history) - 1)
                self.is_simulating = False
                self.saved_real_history = None
            return self._serialize_current_state()

    def import_online_game(
        self,
        game_url_or_id: str,
        mode: str = "full",  # "full" or "terrain_only"
        cookie: Optional[str] = None,
        raw_json: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            game_id, headers, game_data = parse_online_import_request(
                game_url_or_id=game_url_or_id,
                cookie=cookie,
                raw_json=raw_json,
            )

            # If no raw_json was provided or parsed, fetch via network
            if game_data is None:
                if not game_id:
                    raise ValueError(
                        "Please provide a valid Traversal game link (e.g. https://www.traversalgame.com/game/<uuid>), UUID, cURL command, or paste game JSON."
                    )

                api_url = f"https://api.traversalgame.com/api/games/{game_id}/full-state"
                req = urllib.request.Request(api_url, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        game_data = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        # Attempt to query public /invite endpoint for creator context
                        invite_data: Optional[Dict[str, Any]] = None
                        try:
                            inv_req = urllib.request.Request(
                                f"https://api.traversalgame.com/api/games/{game_id}/invite",
                                headers={"User-Agent": headers["User-Agent"], "Accept": "application/json"},
                            )
                            with urllib.request.urlopen(inv_req, timeout=10) as inv_resp:
                                invite_data = json.loads(inv_resp.read().decode("utf-8"))
                        except Exception:
                            pass

                        creator_txt = (
                            f" created by '{invite_data.get('creator_username')}'"
                            if (invite_data and invite_data.get("creator_username"))
                            else ""
                        )
                        preset_txt = (
                            f" (preset: {invite_data.get('terrain_selected')})"
                            if (invite_data and invite_data.get("terrain_selected"))
                            else ""
                        )
                        has_auth = bool("Cookie" in headers or "Authorization" in headers)
                        token_reason = (
                            "The session cookie/token provided was invalid, expired, or truncated."
                            if has_auth
                            else "This match is private and requires logging into traversalgame.com."
                        )

                        raise PermissionError(
                            f"Game {game_id}{creator_txt}{preset_txt} requires login on traversalgame.com to view.\n"
                            f"{token_reason}\n\n"
                            "Quick solutions (from DevTools [F12]):\n"
                            "  1. [Simplest & 100% Reliable] In DevTools → 'Réseau' (Network) tab, click 'full-state' → right-click → 'Copier comme cURL' (Copy as cURL), and paste directly into the URL/cURL box!\n"
                            "  2. Or in 'Réseau' → 'full-state' → click 'Réponse' (Response) tab → Copy all JSON and paste into 'Paste Game JSON'.\n"
                            "  3. Or in 'Stockage' (Storage) → Cookies, select 'traversal_session' and copy its full ~320 character value (starts with 'eyJ...') from the right-hand 'Données' pane."
                        )
                    elif e.code == 404:
                        raise ValueError(f"Game '{game_id}' was not found on traversalgame.com (HTTP 404).")
                    else:
                        raise RuntimeError(f"Server returned HTTP {e.code}: {e.reason}")
                except Exception as e:
                    if not isinstance(e, (ValueError, PermissionError)):
                        raise RuntimeError(f"Could not connect to traversalgame.com: {e}")
                    raise

            # 3. Extract and orient terrain
            raw_board: Optional[List[List[int]]] = None
            if "board" in game_data and isinstance(game_data["board"], list):
                raw_board = game_data["board"]
            elif "s" in game_data and isinstance(game_data["s"], str):
                target_gid = game_id or game_data.get("id") or game_data.get("game_id", "00000000-0000-0000-0000-000000000000")
                raw_board = decode_traversal_terrain(game_data["s"], target_gid)

            if raw_board is None:
                raise ValueError("Game payload did not contain readable terrain ('board' or 's' field).")

            # Flip vertically for engine canonical perspective (P1 moving Row 0 -> Row 9)
            terrain_arr = np.asarray(raw_board, dtype=np.int32)[::-1, :].copy()

            # 4. Mode: Terrain Only
            if mode == "terrain_only":
                game = TraversalGame(terrain_mode="imported", terrain=terrain_arr)
                self.history = [(game.clone(), None, None)]
                self.cursor = 0
                self.match_session_id = uuid.uuid4().hex
                self.match_recorded = False
                self.last_elo_update = None
                state = self._serialize_current_state()
                state["import_info"] = {
                    "mode": "terrain_only",
                    "game_id": game_id,
                    "message": f"Successfully loaded terrain from online game {game_id or 'custom'}!",
                }
                return state

            # 5. Mode: Full Game
            game = TraversalGame(terrain_mode="imported", terrain=terrain_arr)
            move_history = game_data.get("move_history") or []

            p1_map = {"p1_4": 0, "p1_5": 1, "p1_3": 2, "p1_6": 3, "p1_2": 4, "p1_7": 5}
            p2_map = {"p2_5": 0, "p2_4": 1, "p2_6": 2, "p2_3": 3, "p2_7": 4, "p2_2": 5}

            history_records: List[Tuple[TraversalGame, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = [
                (game.clone(), None, None)
            ]

            for ply_idx, m in enumerate(move_history):
                curr_p = game.current_player
                pid = str(m.get("pieceId", ""))
                piece_idx: Optional[int] = None
                if curr_p == 0 and pid in p1_map:
                    piece_idx = p1_map[pid]
                elif curr_p == 1 and pid in p2_map:
                    piece_idx = p2_map[pid]

                # Fallback matching by from square
                if piece_idx is None and "fromRow" in m and "fromCol" in m:
                    our_from_r = 9 - int(m["fromRow"])
                    our_from_c = int(m["fromCol"])
                    for k in range(6):
                        if game.pos[curr_p, k, 0] == our_from_r and game.pos[curr_p, k, 1] == our_from_c:
                            piece_idx = k
                            break

                if piece_idx is None:
                    raise ValueError(f"Could not resolve moving piece '{pid}' at ply {ply_idx + 1}")

                to_row = int(m.get("toRow", 0))
                to_col = int(m.get("toCol", 0))
                is_finish = bool(m.get("isFinish", False) or m.get("is_finish", False) or (to_row < 0 or to_row >= 10))

                dest_r = 9 - to_row
                dest_c = to_col
                action = game.encode_action(
                    piece_idx=piece_idx,
                    target_r=dest_r,
                    target_c=dest_c,
                    finish=is_finish,
                    canonical=False,
                )
                legal_actions = game.get_legal_actions()

                if action not in legal_actions:
                    # Fallback check for action matching target
                    matched = False
                    for la in legal_actions:
                        la_k, la_r, la_c, la_fin = game.decode_action(la, canonical=False)
                        if la_k == piece_idx:
                            if is_finish and la_fin:
                                action = la
                                matched = True
                                break
                            elif not is_finish and la_r == dest_r and la_c == dest_c:
                                action = la
                                matched = True
                                break
                    if not matched:
                        raise ValueError(
                            f"Move at ply {ply_idx + 1} (piece {PIECE_NAMES[piece_idx]} to ({dest_r},{dest_c})) is illegal."
                        )

                move_desc = self._describe_move(game, action)
                if "notation" in m:
                    move_desc["notation"] = m["notation"]
                if "elapsedMs" in m:
                    move_desc["elapsed_ms"] = m["elapsedMs"]

                game.step(action)
                history_records.append((game.clone(), move_desc, None))

            self.history = history_records
            self.cursor = len(self.history) - 1
            self.match_session_id = uuid.uuid4().hex
            self.match_recorded = True

            state = self._serialize_current_state()
            state["import_info"] = {
                "mode": "full",
                "game_id": game_id,
                "plies_imported": len(move_history),
                "message": f"Successfully loaded full match ({len(move_history)} plies) from {game_id or 'custom import'}!",
            }
            return state

    def _describe_move(self, game_before: TraversalGame, action: int) -> Dict[str, Any]:
        p = game_before.current_player
        opp = 1 - p
        piece_idx, dest_r, dest_c, is_finish = game_before.decode_action(action, canonical=False)
        r0, c0 = int(game_before.pos[p, piece_idx, 0]), int(game_before.pos[p, piece_idx, 1])

        captured_piece: Optional[str] = None
        if not is_finish:
            for ok in range(6):
                if int(game_before.pos[opp, ok, 0]) == dest_r and int(game_before.pos[opp, ok, 1]) == dest_c:
                    captured_piece = PIECE_NAMES[ok]
                    break

        if is_finish:
            text = f"P{p + 1} {PIECE_NAMES[piece_idx]} ({r0},{c0}) -> FINISH (+{PIECE_SCORE[piece_idx]} pt)"
        elif captured_piece:
            text = f"P{p + 1} {PIECE_NAMES[piece_idx]} ({r0},{c0}) x ({dest_r},{dest_c}) [{captured_piece}]"
        else:
            text = f"P{p + 1} {PIECE_NAMES[piece_idx]} ({r0},{c0}) -> ({dest_r},{dest_c})"

        return {
            "ply": game_before.ply_count + 1,
            "player": p,
            "action": int(action),
            "piece_idx": int(piece_idx),
            "piece_name": PIECE_NAMES[piece_idx],
            "from_r": r0,
            "from_c": c0,
            "to_r": int(dest_r),
            "to_c": int(dest_c),
            "is_finish": bool(is_finish),
            "captured": captured_piece,
            "text": text,
        }

    def apply_human_move(self, action: int) -> Dict[str, Any]:
        with self.lock:
            game_before = self.history[self.cursor][0]
            if game_before.done:
                raise ValueError("Game is already finished.")

            next_game = game_before.clone()
            move_desc = self._describe_move(next_game, action)
            next_game.step(action)

            # Truncate future branch if stepping from a historical state
            self.history = self.history[: self.cursor + 1]
            self.history.append((next_game, move_desc, None))
            self.cursor = len(self.history) - 1

            if next_game.done and not self.is_simulating:
                self._check_and_record_match(next_game)

            return self._serialize_current_state()

    def _run_mcts_analysis(
        self, game: TraversalGame, ckpt_name: str, num_sims: int
    ) -> Tuple[int, Dict[str, Any]]:
        t0 = time.time()
        model, meta = self._get_or_load_model(ckpt_name)
        mcts = MCTS(model=model, device=self.device, num_simulations=num_sims, rng=self.rng)
        mcts.cache_game_static(game.terrain)

        # Also get raw network value & policy prior at root
        encoder_ver = getattr(model, "encoder_version", "v1")
        x_dyn_np = encode_dynamic_tensor(game, version=encoder_ver)
        legal_mask_np = game.get_legal_mask()
        f_static = mcts._get_f_static(game)
        with torch.inference_mode():
            x_dyn_t = torch.from_numpy(x_dyn_np).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)
            logits, raw_val_t = model.forward_with_cached_static(f_static, x_dyn_t)
            raw_val = float(raw_val_t[0, 0].item())

        root: MCTSNode = mcts.run(game, add_dirichlet_noise=False, num_simulations=num_sims)
        chosen_action, pi_mcts = root.select_move_by_temperature(temperature=0.0, rng=self.rng)
        elapsed_ms = (time.time() - t0) * 1000.0

        assert root.legal_actions is not None and root.N is not None and root.W is not None and root.P is not None

        # Root MCTS value from current_player's perspective (weighted average Q over visits)
        total_visits = int(np.sum(root.N))
        if total_visits > 0:
            root_q = float(np.sum(root.W) / total_visits)
        else:
            root_q = raw_val

        # Convert current_player value in [-1, 1] to P1 win probability in [0, 1]
        p = game.current_player
        p1_mcts_val = root_q if p == 0 else -root_q
        p1_raw_val = raw_val if p == 0 else -raw_val
        p1_win_prob = 0.5 * (p1_mcts_val + 1.0)
        current_player_win_prob = 0.5 * (root_q + 1.0)

        # Build top candidate moves sorted by visit count N, then prior P
        order = np.lexsort((-root.P, -root.N))
        top_candidates: List[Dict[str, Any]] = []
        for idx in order[:10]:
            act = int(root.legal_actions[idx])
            n_vis = int(root.N[idx])
            w_sum = float(root.W[idx])
            q_val = (w_sum / n_vis) if n_vis > 0 else 0.0
            prior_p = float(root.P[idx])
            share = (n_vis / total_visits) if total_visits > 0 else prior_p
            desc = self._describe_move(game, act)
            top_candidates.append(
                {
                    "action": act,
                    "piece_idx": desc["piece_idx"],
                    "piece_name": desc["piece_name"],
                    "from_r": desc["from_r"],
                    "from_c": desc["from_c"],
                    "to_r": desc["to_r"],
                    "to_c": desc["to_c"],
                    "is_finish": desc["is_finish"],
                    "captured": desc["captured"],
                    "text": desc["text"],
                    "visits": n_vis,
                    "policy_share": share,
                    "prior": prior_p,
                    "q_value": q_val,
                    "win_prob": 0.5 * (q_val + 1.0),
                }
            )

        analysis = {
            "checkpoint": ckpt_name,
            "iteration": meta.get("iteration", "?"),
            "simulations": num_sims,
            "elapsed_ms": round(elapsed_ms, 1),
            "perspective_player": p,
            "root_q": root_q,
            "raw_value": raw_val,
            "current_player_win_prob": current_player_win_prob,
            "p1_win_prob": p1_win_prob,
            "top_moves": top_candidates,
        }
        return chosen_action, analysis

    def _run_heuristic_analysis(
        self, game: TraversalGame, agent_id: str = "heuristic"
    ) -> Tuple[int, Dict[str, Any]]:
        t0 = time.time()
        p = game.current_player
        base_id, var_key, _ = parse_agent_variant(agent_id)
        if var_key in ("ultra", "8ply", "grandmaster", "gm"):
            preset = HEURISTIC_PRESETS["grandmaster"]
            var_label = "8-ply 👑 Grandmaster"
        elif var_key in ("deep", "6ply", "master"):
            preset = HEURISTIC_PRESETS["master"]
            var_label = "6-ply 🧠 Master"
        elif var_key in ("fast", "2ply", "blitz"):
            preset = HEURISTIC_PRESETS["blitz"]
            var_label = "2-ply ⚡ Blitz"
        else:
            preset = HEURISTIC_PRESETS["standard"]
            var_label = "4-ply ⚖️ Standard"

        depth = int(preset["depth"])
        root_beam = int(preset["root_beam"])
        inner_limits = preset["inner_limits"]

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

        actions, values = search_best_heuristic_moves_numba(
            game.terrain,
            game.pos,
            game.scores,
            int(game.total_pieces_finished),
            int(game.ply_count),
            int(p),
            depth,
            root_beam,
            inner_limits,
            game.piece_status,
            rep_keys_v0,
            rep_keys_v1,
            rep_counts,
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        if len(actions) == 0:
            chosen_action = int(self.heuristic_agent.select_action(game))
            actions = np.array([chosen_action], dtype=np.int32)
            values = np.array([0.0], dtype=np.float32)
        else:
            order = np.argsort(-values)
            actions = actions[order]
            values = values[order]
            max_val = float(values[0])
            best_mask = values >= (max_val - 5.0)
            chosen_action = int(self.rng.choice(actions[best_mask]))

        best_q = float(np.tanh(float(values[0]) / 600.0))
        p1_val = best_q if p == 0 else -best_q
        p1_win_prob = 0.5 * (p1_val + 1.0)
        cur_win_prob = 0.5 * (best_q + 1.0)

        # Softmax over scaled values for visual policy_share / prior display
        shifted = (values - float(np.max(values))) / 80.0
        weights = np.exp(np.clip(shifted, -20.0, 0.0))
        shares = weights / max(1e-8, float(np.sum(weights)))

        top_candidates: List[Dict[str, Any]] = []
        for idx in range(min(10, len(actions))):
            act = int(actions[idx])
            v_sc = float(values[idx])
            q_val = float(np.tanh(v_sc / 600.0))
            sh = float(shares[idx])
            desc = self._describe_move(game, act)
            top_candidates.append(
                {
                    "action": act,
                    "piece_idx": desc["piece_idx"],
                    "piece_name": desc["piece_name"],
                    "from_r": desc["from_r"],
                    "from_c": desc["from_c"],
                    "to_r": desc["to_r"],
                    "to_c": desc["to_c"],
                    "is_finish": desc["is_finish"],
                    "captured": desc["captured"],
                    "text": desc["text"],
                    "visits": int(round(sh * 100)),
                    "policy_share": sh,
                    "prior": sh,
                    "q_value": q_val,
                    "win_prob": 0.5 * (q_val + 1.0),
                }
            )

        analysis = {
            "checkpoint": f"HeuristicAgent ({var_label} α-β)",
            "iteration": "-",
            "simulations": len(actions),
            "elapsed_ms": round(elapsed_ms, 1),
            "perspective_player": p,
            "root_q": best_q,
            "raw_value": best_q,
            "current_player_win_prob": cur_win_prob,
            "p1_win_prob": p1_win_prob,
            "top_moves": top_candidates,
        }
        return chosen_action, analysis

    def analyze_current_state(
        self, ckpt_name: Optional[str] = None, num_sims: Optional[int] = None
    ) -> Dict[str, Any]:
        with self.lock:
            game, move_rec, _ = self.history[self.cursor]
            if game.done:
                return self._serialize_current_state()

            p = game.current_player
            agent_id = ckpt_name
            if not agent_id or agent_id in ("human", "heuristic"):
                # Pick the active AI checkpoint or default best.pt
                other_agent = self.p2_agent if p == 0 else self.p1_agent
                if other_agent not in ("human", "heuristic"):
                    agent_id = other_agent
                elif ckpt_name == "heuristic" or other_agent == "heuristic":
                    agent_id = "heuristic:standard"
                else:
                    agent_id = self._default_ai_agent()

            sims = num_sims if num_sims is not None else (self.p1_sims if p == 0 else self.p2_sims)
            sims = max(1, min(2000, int(sims)))

            base_id, var_key, preset_sims = parse_agent_variant(agent_id, default_sims=sims)
            if "heuristic" in base_id:
                _, analysis = self._run_heuristic_analysis(game, agent_id=agent_id)
            else:
                effective_sims = sims if num_sims is not None else preset_sims
                _, analysis = self._run_mcts_analysis(game, base_id, effective_sims)

            self.history[self.cursor] = (game, move_rec, analysis)
            return self._serialize_current_state()

    def compute_board_analytics(
        self, ckpt_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Computes multi-layer neural network and terrain analytics for the board
        at the current history cursor:
          - Static inputs: terrain height, P1 Dijkstra, P2 Dijkstra, topological advantage
          - Static stem activations: L2 norm (32 channels) and PC1 (latent topology)
          - End-to-end model outputs: Sentinel, Vanguard, and Explorer positional value maps,
            plus move policy landing priors.
        """
        with self.lock:
            game = self.history[self.cursor][0]
            terrain = game.terrain
            x_static_p1, _ = encode_both_static_tensors(terrain)

            agent_id = ckpt_name
            if not agent_id or agent_id in ("human", "heuristic"):
                for candidate in (self.p2_agent, self.p1_agent, "train_v5/best.pt", "best.pt"):
                    if candidate and candidate not in ("human", "heuristic"):
                        agent_id = candidate
                        break
            if not agent_id or agent_id in ("human", "heuristic"):
                pts = self._discover_checkpoint_relpaths()
                agent_id = pts[0] if pts else "best.pt"

            base_id, _, _ = parse_agent_variant(agent_id)
            try:
                model, meta = self._get_or_load_model(base_id)
            except Exception:
                pts = self._discover_checkpoint_relpaths()
                if pts:
                    base_id, _, _ = parse_agent_variant(pts[0])
                    model, meta = self._get_or_load_model(base_id)
                else:
                    raise

            with torch.no_grad():
                t_static = torch.from_numpy(x_static_p1).to(self.device)
                f_static = model.encode_static(t_static)
                f_static_np = f_static.cpu().numpy()

                # 1. Static Stem L2 Norm
                l2_norm = np.linalg.norm(f_static_np, axis=0)

                # 2. Static Stem PC1 (PCA)
                flat = f_static_np.reshape(32, 100).T
                centered = flat - flat.mean(axis=0)
                u, _, _ = np.linalg.svd(centered, full_matrices=False)
                pc1 = u[:, 0].reshape(10, 10)

                # 3. Dynamic current game state
                version = getattr(model, "encoder_version", "v2")
                x_dyn = encode_dynamic_tensor(game, version=version)
                legal_mask = game.get_legal_mask()
                logits, root_val = model.forward_with_cached_static(
                    f_static, torch.from_numpy(x_dyn).to(self.device)
                )
                probs = masked_softmax(logits, legal_mask).cpu().numpy()

                # Policy landing heatmap across all pieces
                policy_tiles = np.zeros((10, 10), dtype=np.float32)
                for k in range(6):
                    for r in range(10):
                        for c in range(10):
                            policy_tiles[r, c] += probs[k * NUM_TARGETS + (r * 10 + c)]

                # 4. Positional value probes: Sentinel (4, 5), Vanguard (2, 3), Explorer (0, 1)
                # Average both pieces of each type (e.g. S0 and S1) to avoid unilateral lateral bias
                probe_pairs = [
                    (4, 5),  # Sentinel_0, Sentinel_1
                    (2, 3),  # Vanguard_0, Vanguard_1
                    (0, 1),  # Explorer_0, Explorer_1
                ]
                batch_dyn = []
                for s0, s1 in probe_pairs:
                    for s in (s0, s1):
                        for r in range(10):
                            for c in range(10):
                                tg = game.clone()
                                tg.piece_status[0, s] = 0
                                tg.pos[0, s] = [r, c]
                                batch_dyn.append(encode_dynamic_tensor(tg, version=version))

                batch_t = torch.from_numpy(np.stack(batch_dyn, axis=0)).to(self.device)
                _, vals = model.forward_with_cached_static(f_static, batch_t)
                vals_all = vals.cpu().numpy().reshape(3, 2, 10, 10)
                vals_np = vals_all.mean(axis=1)  # (3, 10, 10)

            my_dist = x_static_p1[1] * 30.0
            opp_dist = x_static_p1[3] * 30.0
            topo_adv = (x_static_p1[3] - x_static_p1[1]) * 30.0

            # List active pieces currently on the board
            board_pieces = []
            for pl in (0, 1):
                for k in range(6):
                    r0, c0 = int(game.pos[pl, k, 0]), int(game.pos[pl, k, 1])
                    if r0 >= 0 and c0 >= 0 and game.piece_status[pl, k] == 0:
                        board_pieces.append({
                            "player": pl,
                            "piece_idx": k,
                            "name": PIECE_NAMES[k],
                            "type": int(PIECE_TYPES[k]),
                            "r": r0,
                            "c": c0,
                        })

            return {
                "checkpoint": base_id,
                "ply_count": int(game.ply_count),
                "cursor": int(self.cursor),
                "current_player": int(game.current_player),
                "terrain_mode": game.terrain_mode,
                "terrain": terrain.astype(int).tolist(),
                "my_dist": my_dist.tolist(),
                "opp_dist": opp_dist.tolist(),
                "topo_adv": topo_adv.tolist(),
                "static_l2_norm": l2_norm.tolist(),
                "static_pc1": pc1.tolist(),
                "policy_priors": policy_tiles.tolist(),
                "value_maps": {
                    "Sentinel": vals_np[0].tolist(),
                    "Vanguard": vals_np[1].tolist(),
                    "Explorer": vals_np[2].tolist(),
                },
                "root_val": float(root_val.item()),
                "pieces": board_pieces,
                "available_checkpoints": self._discover_checkpoint_relpaths(),
            }

    def step_ai_move(self) -> Dict[str, Any]:
        with self.lock:
            game_before, _, _ = self.history[self.cursor]
            if game_before.done:
                return self._serialize_current_state()

            p = game_before.current_player
            agent_id = self.p1_agent if p == 0 else self.p2_agent
            sims = self.p1_sims if p == 0 else self.p2_sims

            if agent_id == "human":
                # If user clicks "Force AI Move" on a human turn, use the other player's AI or best.pt
                fallback = self.p2_agent if p == 0 else self.p1_agent
                agent_id = fallback if fallback != "human" else self._default_ai_agent()

            base_id, var_key, preset_sims = parse_agent_variant(agent_id, default_sims=sims)

            if "heuristic" in base_id:
                chosen_action, analysis = self._run_heuristic_analysis(game_before, agent_id=agent_id)
            else:
                effective_sims = sims if sims is not None else preset_sims
                chosen_action, analysis = self._run_mcts_analysis(game_before, base_id, effective_sims)

            # Save the analysis onto the pre-move state as well
            prev_move_rec = self.history[self.cursor][1]
            self.history[self.cursor] = (game_before, prev_move_rec, analysis)

            next_game = game_before.clone()
            move_desc = self._describe_move(next_game, chosen_action)
            next_game.step(chosen_action)

            self.history = self.history[: self.cursor + 1]
            self.history.append((next_game, move_desc, analysis))
            self.cursor = len(self.history) - 1

            if next_game.done and not self.is_simulating:
                self._check_and_record_match(next_game)

            return self._serialize_current_state()

    def _normalize_agent_id(self, agent_id: str, sims: int) -> str:
        if agent_id == "human":
            return self.human_player_name
        base_id, var_key, parsed_sims = parse_agent_variant(agent_id, default_sims=sims)
        if ":" in agent_id or "@" in agent_id:
            return agent_id
        if base_id == "heuristic":
            if var_key in ("ultra", "8ply", "grandmaster", "gm") or sims >= 384:
                return "heuristic:grandmaster"
            elif var_key in ("deep", "6ply", "master") or sims >= 192:
                return "heuristic:master"
            elif var_key in ("fast", "2ply", "blitz") or sims <= 48:
                return "heuristic:blitz"
            return "heuristic:standard"
        if sims <= 48:
            return f"{base_id}:blitz"
        elif sims >= 192:
            return f"{base_id}:master"
        else:
            return f"{base_id}:standard"

    def _check_and_record_match(self, game: TraversalGame) -> None:
        with self.lock:
            if self.match_recorded or not game.done:
                return

            p1_id = self._normalize_agent_id(self.p1_agent, self.p1_sims)
            p2_id = self._normalize_agent_id(self.p2_agent, self.p2_sims)

            p1_kind = "human" if self.p1_agent == "human" else ("heuristic" if "heuristic" in p1_id else "alphazero")
            p2_kind = "human" if self.p2_agent == "human" else ("heuristic" if "heuristic" in p2_id else "alphazero")

            p1_name = self.human_player_name if self.p1_agent == "human" else self._get_agent_display_name(p1_id)
            p2_name = self.human_player_name if self.p2_agent == "human" else self._get_agent_display_name(p2_id)

            winner_code = int(game.winner) if game.winner is not None else -1

            rec = self.elo_db.record_match(
                p1_id=p1_id,
                p2_id=p2_id,
                winner=winner_code,
                p1_name=p1_name,
                p2_name=p2_name,
                p1_kind=p1_kind,
                p2_kind=p2_kind,
                ply_count=int(game.ply_count),
                scores=[int(game.scores[0]), int(game.scores[1])],
                termination_reason=game.termination_reason,
                terrain_mode=game.terrain_mode,
                source="visualizer",
                match_id=f"vis_{self.match_session_id[:8]}_{int(time.time())}",
            )
            self.match_recorded = True
            self.last_elo_update = {
                "match_id": rec.match_id,
                "winner": rec.winner,
                "winner_name": rec.winner_name,
                "p1_name": rec.player1_name,
                "p2_name": rec.player2_name,
                "p1_rating_before": rec.p1_rating_before,
                "p1_rating_after": rec.p1_rating_after,
                "p1_delta": rec.p1_delta,
                "p2_rating_before": rec.p2_rating_before,
                "p2_rating_after": rec.p2_rating_after,
                "p2_delta": rec.p2_delta,
                "termination_reason": rec.termination_reason,
            }

    def start_tournament(
        self,
        agent_ids: List[str],
        pairs: int = 1,
        sims: int = 96,
        terrain_mode: str = "random",
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if self.tournament_status.get("running"):
                return {"error": "A tournament is already running"}

            if target_agent:
                if target_agent not in agent_ids:
                    agent_ids = [target_agent] + list(agent_ids)
                opponents = [a for a in agent_ids if a != target_agent]
                if not opponents:
                    return {"error": "Target gauntlet requires at least 1 opponent"}
                total_games = len(opponents) * pairs * 2
                mode_desc = f"Gauntlet ({target_agent} vs {len(opponents)} opponents)"
            else:
                if len(agent_ids) < 2:
                    return {"error": "Round-robin tournament requires at least 2 agents"}
                total_games = (len(agent_ids) * (len(agent_ids) - 1) // 2) * pairs * 2
                mode_desc = f"Round-Robin ({len(agent_ids)} models)"

            self.tournament_status = {
                "running": True,
                "progress": 0,
                "total": total_games,
                "current": f"Starting {mode_desc}...",
                "last_result": "",
                "target_agent": target_agent,
                "mode": "gauntlet" if target_agent else "round_robin",
            }

        def _run():
            try:
                from elo_arena import run_tournament
                def _progress(cur, tot, rec):
                    with self.lock:
                        self.tournament_status["progress"] = cur
                        self.tournament_status["total"] = tot
                        self.tournament_status["current"] = f"{rec.player1_name} vs {rec.player2_name}"
                        self.tournament_status["last_result"] = f"{rec.winner_name} won ({rec.p1_delta:+.1f} / {rec.p2_delta:+.1f})"
                run_tournament(
                    agent_ids=agent_ids,
                    db=self.elo_db,
                    pairs_per_matchup=pairs,
                    num_simulations=sims,
                    terrain_mode=terrain_mode,
                    device=self.device,
                    checkpoint_dir=self.checkpoint_dir,
                    progress_callback=_progress,
                    target_agent_id=target_agent,
                )
            except Exception as exc:
                print(f"[VisualizerSession] Tournament error: {exc}")
                with self.lock:
                    self.tournament_status["last_result"] = f"Error: {exc}"
            finally:
                with self.lock:
                    self.tournament_status["running"] = False
                    self.tournament_status["current"] = "Completed"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return {"status": "started", "total": total_games, "target_agent": target_agent}

    def _serialize_current_state(self) -> Dict[str, Any]:
        with self.lock:
            game, last_move, analysis = self.history[self.cursor]
            inspector = compute_detailed_ui_inspector(game)

            all_pieces: List[List[Dict[str, Any]]] = [[], []]
            for pl in (0, 1):
                for k in range(6):
                    all_pieces[pl].append(
                        {
                            "piece_idx": k,
                            "name": PIECE_NAMES[k],
                            "type": int(PIECE_TYPES[k]),
                            "movement": int(PIECE_MOVEMENT[k]),
                            "strength": int(PIECE_STRENGTH[k]),
                            "score_val": int(PIECE_SCORE[k]),
                            "r": int(game.pos[pl, k, 0]),
                            "c": int(game.pos[pl, k, 1]),
                            "status": int(game.piece_status[pl, k]),
                        }
                    )

            move_log = [
                item[1] for item in self.history[1:] if item[1] is not None
            ]

            return {
                "cursor": self.cursor,
                "max_cursor": len(self.history) - 1,
                "ply_count": int(game.ply_count),
                "current_player": int(game.current_player),
                "scores": [int(game.scores[0]), int(game.scores[1])],
                "total_pieces_finished": int(game.total_pieces_finished),
                "done": bool(game.done),
                "winner": int(game.winner) if game.winner is not None else None,
                "termination_reason": game.termination_reason,
                "terrain_mode": game.terrain_mode,
                "terrain": game.terrain.astype(int).tolist(),
                "pieces": all_pieces,
                "inspector": inspector,
                "last_move": last_move,
                "analysis": analysis,
                "move_log": move_log,
                "simulation_mode": {
                    "active": bool(self.is_simulating),
                    "sim_start_ply": int(self.sim_start_ply) if self.is_simulating else None,
                    "sim_start_cursor": int(self.sim_start_cursor) if self.is_simulating else None,
                    "sim_plies_count": int(max(0, len(self.history) - 1 - self.sim_start_cursor))
                    if self.is_simulating
                    else 0,
                },
                "settings": {
                    "p1_agent": self.p1_agent,
                    "p2_agent": self.p2_agent,
                    "p1_sims": self.p1_sims,
                    "p2_sims": self.p2_sims,
                    "terrain_mode": self.terrain_mode,
                    "device": str(self.device),
                },
                "elo": {
                    "human_player_name": self.human_player_name,
                    "p1_rating": round(self._get_agent_rating(self.p1_agent), 1),
                    "p2_rating": round(self._get_agent_rating(self.p2_agent), 1),
                    "last_update": self.last_elo_update,
                    "tournament": self.tournament_status,
                },
                "available_agents": self.list_available_agents(),
                "terrain_modes": list(TERRAIN_MODES),
            }


# ---------------------------------------------------------------------------
# HTTP Server & JSON API Handler
# ---------------------------------------------------------------------------

SESSION: Optional[VisualizerSession] = None
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class VisualizerHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        assert SESSION is not None
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            if not os.path.isfile(index_path):
                self._send_json({"error": "static/index.html not found"}, status=404)
                return
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/state":
            with SESSION.lock:
                state = SESSION._serialize_current_state()
            self._send_json(state)
            return

        if parsed.path == "/api/elo/leaderboard":
            with SESSION.lock:
                lb = SESSION.elo_db.get_leaderboard()
            self._send_json({"leaderboard": lb})
            return

        if parsed.path == "/api/elo/matches":
            with SESSION.lock:
                matches = SESSION.elo_db.get_matches(limit=50)
            self._send_json({"matches": matches})
            return

        if parsed.path == "/api/elo/tournament/status":
            with SESSION.lock:
                status = dict(SESSION.tournament_status)
            self._send_json({"tournament": status})
            return

        if parsed.path == "/api/analytics":
            try:
                data = SESSION.compute_board_analytics()
                self._send_json(data)
            except Exception as e:
                logger.exception("Failed to compute board analytics: %s", e)
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": f"Unknown endpoint {parsed.path}"}, status=404)

    def do_POST(self) -> None:
        assert SESSION is not None
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/analytics":
                ckpt = body.get("checkpoint") if isinstance(body, dict) else None
                try:
                    data = SESSION.compute_board_analytics(ckpt_name=ckpt)
                    self._send_json(data)
                except Exception as e:
                    logger.exception("Failed to compute board analytics: %s", e)
                    self._send_json({"error": str(e)}, status=500)
                return
            if parsed.path == "/api/new_game":
                state = SESSION.new_game(
                    terrain_mode=body.get("terrain_mode", "random"),
                    p1_agent=body.get("p1_agent"),
                    p2_agent=body.get("p2_agent"),
                    p1_sims=body.get("p1_sims"),
                    p2_sims=body.get("p2_sims"),
                    seed=body.get("seed"),
                )
                self._send_json(state)
                return

            if parsed.path == "/api/settings":
                state = SESSION.update_settings(
                    p1_agent=body.get("p1_agent"),
                    p2_agent=body.get("p2_agent"),
                    p1_sims=body.get("p1_sims"),
                    p2_sims=body.get("p2_sims"),
                )
                self._send_json(state)
                return

            if parsed.path == "/api/move":
                action = int(body["action"])
                state = SESSION.apply_human_move(action)
                self._send_json(state)
                return

            if parsed.path == "/api/ai_move":
                state = SESSION.step_ai_move()
                self._send_json(state)
                return

            if parsed.path == "/api/analyze":
                state = SESSION.analyze_current_state(
                    ckpt_name=body.get("checkpoint"),
                    num_sims=body.get("simulations"),
                )
                self._send_json(state)
                return

            if parsed.path == "/api/jump":
                cursor = int(body.get("cursor", 0))
                state = SESSION.jump_to_ply(cursor)
                self._send_json(state)
                return

            if parsed.path == "/api/simulation/start":
                state = SESSION.start_simulation()
                self._send_json(state)
                return

            if parsed.path == "/api/simulation/reset":
                state = SESSION.reset_simulation()
                self._send_json(state)
                return

            if parsed.path == "/api/simulation/exit":
                commit = bool(body.get("commit", False)) if isinstance(body, dict) else False
                state = SESSION.exit_simulation(commit=commit)
                self._send_json(state)
                return

            if parsed.path == "/api/piece_path":
                p = int(body.get("player", 0))
                idx = int(body.get("piece_idx", 0))
                with SESSION.lock:
                    game = SESSION.history[SESSION.cursor][0]
                    r0 = int(game.pos[p, idx, 0])
                    c0 = int(game.pos[p, idx, 1])
                    path_info = compute_piece_finish_path(game.terrain, p, idx, r0, c0)
                self._send_json({"piece_idx": idx, "player": p, "path": path_info})
                return

            if parsed.path == "/api/elo/player_name":
                name = body.get("name", "Guillaume")
                SESSION.set_human_player_name(name)
                with SESSION.lock:
                    state = SESSION._serialize_current_state()
                self._send_json(state)
                return

            if parsed.path == "/api/elo/reset":
                with SESSION.lock:
                    SESSION.elo_db.reset(keep_players=True)
                    state = SESSION._serialize_current_state()
                self._send_json({"success": True, "state": state})
                return

            if parsed.path == "/api/elo/tournament":
                agents = body.get("checkpoints")
                if not agents:
                    all_ckpts = SESSION._discover_checkpoint_relpaths()
                    bests = [c for c in all_ckpts if os.path.basename(c) == "best.pt"]
                    agents = bests + ["heuristic:grandmaster", "heuristic:master", "heuristic:standard", "heuristic:blitz"]
                pairs = int(body.get("pairs", 1))
                sims = int(body.get("sims", 96))
                terrain_mode = body.get("terrain_mode", "random")
                target_agent = body.get("target_agent") or None
                res = SESSION.start_tournament(
                    agents,
                    pairs=pairs,
                    sims=sims,
                    terrain_mode=terrain_mode,
                    target_agent=target_agent,
                )
                self._send_json(res)
                return

            if parsed.path == "/api/import_online_game":
                res = SESSION.import_online_game(
                    game_url_or_id=body.get("game_url_or_id", ""),
                    mode=body.get("mode", "full"),
                    cookie=body.get("cookie"),
                    raw_json=body.get("raw_json"),
                )
                self._send_json(res)
                return

            self._send_json({"error": f"Unknown POST endpoint {parsed.path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def main() -> None:
    global SESSION
    parser = argparse.ArgumentParser(description="Traversal 3D/2D Local Web Visualizer & AI Arena")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained .pt checkpoints",
    )
    args = parser.parse_args()

    SESSION = VisualizerSession(checkpoint_dir=args.checkpoint_dir)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHTTPHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[Traversal Visualizer] Device: {SESSION.device}")
    print(f"[Traversal Visualizer] Serving interactive 3D/2D Web UI at: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Traversal Visualizer] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

