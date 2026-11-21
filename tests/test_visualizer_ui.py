"""
Tests for the Obsidian Traversal web visualizer routes, static asset serving,
security checks, and clean UI structure.
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import visualizer
from elo_system import EloDatabase
from visualizer import VisualizerHTTPHandler, VisualizerSession


@pytest.fixture()
def live_visualizer_server(tmp_path):
    db = EloDatabase(db_path=str(tmp_path / "test_ui_elo.json"))
    session = VisualizerSession(checkpoint_dir=str(tmp_path / "ckpts"), elo_db=db)
    prev_session = visualizer.SESSION
    visualizer.SESSION = session

    server = ThreadingHTTPServer(("127.0.0.1", 0), VisualizerHTTPHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", session
    finally:
        server.shutdown()
        server.server_close()
        visualizer.SESSION = prev_session


def _http_get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode("utf-8")


def test_obsidian_index_routes_serve_html(live_visualizer_server):
    base_url, _ = live_visualizer_server
    for path in ("/", "/index.html"):
        status, ctype, body = _http_get(f"{base_url}{path}")
        assert status == 200, f"Expected 200 for {path}, got {status}"
        assert "text/html" in ctype
        assert 'href="/static/themes/base.css"' in body
        assert 'href="/static/themes/obsidian.css"' in body
        assert 'src="/static/app.js"' in body
        assert 'id="shortcutsModal"' in body
        assert 'id="toggleShortcutsHintsBtn"' in body
        assert "atelier.css" not in body
        assert "topograph.css" not in body
        assert "ui-switcher" not in body


def test_static_assets_served_with_correct_mime(live_visualizer_server):
    base_url, _ = live_visualizer_server
    expected = [
        ("/static/app.js", "application/javascript"),
        ("/static/themes/base.css", "text/css"),
        ("/static/themes/obsidian.css", "text/css"),
    ]
    for path, mime_prefix in expected:
        status, ctype, body = _http_get(f"{base_url}{path}")
        assert status == 200, f"Failed to fetch {path}"
        assert mime_prefix in ctype
        assert len(body) > 100


def test_removed_themes_traversal_and_missing_files_return_404(live_visualizer_server):
    base_url, _ = live_visualizer_server
    for bad_path in (
        "/static/themes/atelier.css",
        "/static/themes/topograph.css",
        "/static/../visualizer.py",
        "/static/themes/../../visualizer.py",
        "/static/does_not_exist.css",
        "/static/",
    ):
        status, _, _ = _http_get(f"{base_url}{bad_path}")
        assert status == 404, f"Expected 404 for {bad_path}, got {status}"


def test_api_state_and_clean_agent_labels(live_visualizer_server):
    base_url, _ = live_visualizer_server
    status, ctype, body = _http_get(f"{base_url}/api/state")
    assert status == 200
    assert "application/json" in ctype
    state = json.loads(body)
    assert "available_agents" in state
    assert len(state["available_agents"]) >= 4
    emoji_chars = ("🧠", "⚖", "⚡", "👑", "🎲", "⛰", "🏞", "🌄", "🌲", "🏜", "🎮", "📊", "🧪", "🏆")
    for ag in state["available_agents"]:
        for ch in emoji_chars:
            assert ch not in ag["label"], f"Unexpected emoji {ch} in agent label {ag['label']}"


def test_all_dom_ids_referenced_in_js_exist_in_html():
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(static_dir, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    missing = js_ids - html_ids
    assert not missing, f"JS references missing DOM IDs: {sorted(missing)}"
