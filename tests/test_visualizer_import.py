"""
Unit tests for online game & terrain import functionality in visualizer.py.
"""

import base64
import json
import numpy as np
import pytest

from visualizer import (
    VisualizerSession,
    OFFICIAL_CODEC_KEY,
    _CodecPRNG,
    _fnv1a,
    decode_traversal_terrain,
    parse_online_import_request,
)
from traversal_env import FINISH_TARGET, PIECE_NAMES


def _create_mock_blob(board: list[list[int]], game_id: str, key: str = OFFICIAL_CODEC_KEY) -> str:
    """Encodes a 10x10 board into the traversalgame.com binary blob format."""
    rows = len(board)
    cols = len(board[0])
    c = rows * cols
    flat = bytearray(c)
    for r in range(rows):
        for col in range(cols):
            flat[r * cols + col] = board[r][col]

    chk = _fnv1a(flat) & 0xFFFF
    salt = b"12345678"
    seed = key.encode("utf-8") + b"\x00" + game_id.encode("utf-8") + b"\x00" + salt
    rng = _CodecPRNG(seed)

    perm = list(range(c))
    for t in range(c - 1, 0, -1):
        o = rng.next_u32() % (t + 1)
        perm[t], perm[o] = perm[o], perm[t]

    payload = bytearray(c)
    for i in range(c):
        payload[i] = flat[perm[i]] ^ (rng.next_u32() & 0xFF)

    out = bytearray(11 + c + 2)
    out[0] = 1
    out[1] = rows
    out[2] = cols
    out[3:11] = salt
    out[11 : 11 + c] = payload
    out[-2] = (chk >> 8) & 0xFF
    out[-1] = chk & 0xFF
    return base64.b64encode(out).decode("utf-8")


def test_decode_traversal_terrain_roundtrip():
    test_board = [[(r + c) % 4 for c in range(10)] for r in range(10)]
    game_id = "e82df628-2029-492a-b5b4-948d10216088"
    blob = _create_mock_blob(test_board, game_id)

    decoded = decode_traversal_terrain(blob, game_id)
    assert decoded == test_board

    # Checksum failure on corrupted byte
    raw = bytearray(base64.b64decode(blob))
    raw[15] ^= 0x55
    bad_blob = base64.b64encode(raw).decode("utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        decode_traversal_terrain(bad_blob, game_id)


def test_import_online_game_terrain_only():
    session = VisualizerSession()
    test_board = [[(r + c) % 3 for c in range(10)] for r in range(10)]
    game_id = "e82df628-2029-492a-b5b4-948d10216088"
    blob = _create_mock_blob(test_board, game_id)

    raw_json = {"s": blob, "game_id": game_id}
    state = session.import_online_game(
        game_url_or_id=f"https://www.traversalgame.com/game/{game_id}",
        mode="terrain_only",
        raw_json=raw_json,
    )

    assert state["import_info"]["mode"] == "terrain_only"
    assert len(session.history) == 1
    assert session.cursor == 0
    # Engine inverts rows vertically for canonical player 0 perspective
    expected_terrain = np.asarray(test_board, dtype=np.int32)[::-1, :].tolist()
    assert state["terrain"] == expected_terrain


def test_import_online_game_full():
    session = VisualizerSession()
    # Flat terrain
    test_board = [[0 for _ in range(10)] for _ in range(10)]
    game_id = "e82df628-2029-492a-b5b4-948d10216088"
    blob = _create_mock_blob(test_board, game_id)

    # 2 sample moves:
    # 1. P1 moves Explorer_0 (p1_4) from (9, 4) to (7, 4) -> in engine coords: (0, 4) to (2, 4)
    # 2. P2 moves Explorer_0 (p2_5) from (0, 5) to (2, 5) -> in engine coords: (9, 5) to (7, 5)
    move_history = [
        {"player": 1, "pieceId": "p1_4", "fromRow": 9, "fromCol": 4, "toRow": 7, "toCol": 4, "notation": "Ee3"},
        {"player": 2, "pieceId": "p2_5", "fromRow": 0, "fromCol": 5, "toRow": 2, "toCol": 5, "notation": "Ee8"},
    ]

    raw_json = {
        "s": blob,
        "game_id": game_id,
        "move_history": move_history,
    }

    state = session.import_online_game(
        game_url_or_id=game_id,
        mode="full",
        raw_json=raw_json,
    )

    assert state["import_info"]["mode"] == "full"
    assert state["import_info"]["plies_imported"] == 2
    assert len(session.history) == 3
    assert session.cursor == 2

    # Verify ply 1 record
    game_ply1, move_ply1, _ = session.history[1]
    assert move_ply1 is not None
    assert move_ply1["player"] == 0
    assert move_ply1["piece_name"] == "Explorer_0"
    assert move_ply1["from_r"] == 0 and move_ply1["from_c"] == 4
    assert move_ply1["to_r"] == 2 and move_ply1["to_c"] == 4

    # Verify ply 2 record
    game_ply2, move_ply2, _ = session.history[2]
    assert move_ply2 is not None
    assert move_ply2["player"] == 1
    assert move_ply2["piece_name"] == "Explorer_0"
    assert move_ply2["from_r"] == 9 and move_ply2["from_c"] == 5
    assert move_ply2["to_r"] == 7 and move_ply2["to_c"] == 5


def test_import_online_game_permission_error():
    session = VisualizerSession()
    # When calling without raw_json and without valid cookie for private match, raises error
    # (or permission error explaining auth is needed)
    with pytest.raises(Exception) as exc_info:
        session.import_online_game(
            game_url_or_id="https://www.traversalgame.com/game/00000000-0000-0000-0000-000000000000",
            mode="full",
        )
    assert "requires" in str(exc_info.value) or "connect" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


def test_parse_online_import_request_curl():
    curl_cmd = (
        "curl 'https://api.traversalgame.com/api/games/0e2d2a54-81d4-4e7f-b570-583f4cc6984b/full-state' "
        "-H 'Cookie: traversal_session=eyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456; "
        "cf_clearance=test_clearance' "
        "-H 'Authorization: Bearer eyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456'"
    )
    game_id, headers, data = parse_online_import_request(curl_cmd)
    assert game_id == "0e2d2a54-81d4-4e7f-b570-583f4cc6984b"
    assert "traversal_session=" in headers.get("Cookie", "")
    assert "cf_clearance=" in headers.get("Cookie", "")
    assert headers.get("Authorization") == "Bearer eyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456"
    assert headers.get("Origin") == "https://www.traversalgame.com"


def test_parse_online_import_request_devtools_table():
    firefox_table = (
        "cf_clearance\tCaTbVT_iNegAlh05Cal4k4TSX6Y7LisVvqTcXpQf1k\t.traversalgame.com\n"
        "traversal_session\teyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJ1\t.traversalgame.com\n"
    )
    game_id, headers, data = parse_online_import_request(
        "https://www.traversalgame.com/game/0e2d2a54-81d4-4e7f-b570-583f4cc6984b",
        cookie=firefox_table,
    )
    assert game_id == "0e2d2a54-81d4-4e7f-b570-583f4cc6984b"
    assert "cf_clearance=CaTbVT_iNegAlh05Cal4k4TSX6Y7LisVvqTcXpQf1k" in headers.get("Cookie", "")
    assert "traversal_session=eyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJ1" in headers.get("Cookie", "")
    assert headers.get("Authorization") == "Bearer eyJhbGciOiJ1NilsInR5cCI6IkpXVCJ9.eyJ1"


def test_parse_online_import_request_truncated_warning():
    # User's exact pasted 86-char signature from the screenshot
    truncated = "_JX-rtqCcHt4jedDqm6dfviW4zJff078J0RgI_dG6AMeqbAzWGmUfPs4eXbzU6zi0HVJacdFwadQltuPqjAZs"
    with pytest.raises(ValueError, match="incomplete or truncated"):
        parse_online_import_request(
            "https://www.traversalgame.com/game/0e2d2a54-81d4-4e7f-b570-583f4cc6984b",
            cookie=truncated,
        )


def test_parse_online_import_request_direct_json():
    json_str = json.dumps({"game_id": "0e2d2a54-81d4-4e7f-b570-583f4cc6984b", "s": "mock_blob"})
    game_id, headers, data = parse_online_import_request(json_str)
    assert game_id == "0e2d2a54-81d4-4e7f-b570-583f4cc6984b"
    assert data is not None
    assert data["s"] == "mock_blob"
