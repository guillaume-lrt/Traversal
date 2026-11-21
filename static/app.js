/* ==========================================================================
   Traversal Visualizer — Application Logic & Keyboard Shortcuts (Obsidian UI)
   ========================================================================== */

// ---------------------------------------------------------------------------
// Obsidian Canvas Palette
// ---------------------------------------------------------------------------
const CANVAS_THEME = {
  // Obsidian: Neutral matte basalt & graphite dark studio (zero blue-slate tint)
  elevation: [
    { top: "#242422", left: "#181816", right: "#121210" }, // 0
    { top: "#2b2b28", left: "#1d1d1a", right: "#151513" }, // 1
    { top: "#33332e", left: "#23231f", right: "#191916" }, // 2
    { top: "#3c3c35", left: "#292924", right: "#1e1e1a" }, // 3
    { top: "#46473d", left: "#30312a", right: "#23241e" }, // 4
    { top: "#535446", left: "#393a30", right: "#292a22" }, // 5
    { top: "#62614f", left: "#444337", right: "#313027" }, // 6
    { top: "#736f5a", left: "#514e3f", right: "#3b382d" }, // 7
    { top: "#867e66", left: "#5f5948", right: "#454134" }, // 8
    { top: "#9c9178", left: "#706856", right: "#524c3e" }, // 9
    { top: "#b8ad96", left: "#877f6d", right: "#635d50" }, // 10+
  ],
  gridStroke: "rgba(234, 231, 223, 0.16)",
  gridStrokeAnalytics: "rgba(255, 255, 255, 0.25)",
  pillarStroke: "rgba(0, 0, 0, 0.42)",
  contourStroke: "rgba(255, 255, 255, 0.10)",
  coordColor: "rgba(186, 181, 169, 0.78)",
  heightLabelColor: "rgba(234, 231, 223, 0.62)",
  fontSans: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  fontMono: 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
  finishZone: {
    idleFill: "rgba(38, 38, 34, 0.68)",
    activeFill: "rgba(212, 163, 72, 0.28)",
    p1Stroke: "rgba(100, 161, 217, 0.50)",
    p2Stroke: "rgba(217, 115, 106, 0.50)",
    activeStroke: "#d4a348",
    idleText: "#a5a095",
    activeText: "#eab85c",
  },
  overlays: {
    lastMoveFill: "rgba(168, 153, 214, 0.22)",
    lastMoveStroke: "#a899d6",
    legalTurnFill: "rgba(105, 176, 130, 0.24)",
    legalTurnStroke: "#69b082",
    legalPreviewFill: "rgba(100, 161, 217, 0.22)",
    legalPreviewStroke: "#64a1d9",
    captureFill: "rgba(212, 163, 72, 0.30)",
    captureStroke: "#d4a348",
    vanguardFill: "rgba(212, 138, 44, 0.32)",
    vanguardStroke: "#d48a2c",
    vanguardCapFill: "rgba(217, 108, 60, 0.36)",
    vanguardCapStroke: "#e07b4f",
    blockedFill: "rgba(220, 109, 100, 0.24)",
    blockedStroke: "#dc6d64",
    blockedVanguardFill: "rgba(220, 109, 100, 0.32)",
    blockedVanguardStroke: "#e85d54",
    forbiddenCombatFill: "rgba(220, 109, 100, 0.16)",
    forbiddenCombatStroke: "rgba(220, 109, 100, 0.42)",
    forbiddenFriendlyFill: "rgba(143, 139, 128, 0.14)",
    forbiddenFriendlyStroke: "rgba(143, 139, 128, 0.38)",
    forbiddenEnemyFill: "rgba(212, 138, 44, 0.13)",
    forbiddenEnemyStroke: "rgba(212, 138, 44, 0.35)",
    forbiddenPathFill: "rgba(143, 139, 128, 0.13)",
    forbiddenPathStroke: "rgba(143, 139, 128, 0.35)",
    hoverFill: "rgba(240, 237, 228, 0.16)",
    hoverStroke: "#f0ede4",
    dotTurn: "#69b082",
    dotPreview: "#64a1d9",
    dotVanguard: "#d48a2c",
    dotSpecial: "#a899d6",
    dotForbiddenFill: "rgba(143, 139, 128, 0.28)",
    dotForbiddenStroke: "rgba(143, 139, 128, 0.55)",
    badgeAccent: "#e08e45",
    analyticsText: "#f0ede4",
    analyticsOutline: "rgba(16, 16, 14, 0.92)",
  },
  pieces: {
    p1: { base: "#366c9c", lit: "#5a96cc", dark: "#244c70", top: "#86b9e6", rim: "#64a1d9" },
    p2: { base: "#9e3b34", lit: "#c75850", dark: "#732621", top: "#e08982", rim: "#d9736a" },
    selectedHaloFill: "rgba(212, 163, 72, 0.28)",
    selectedRim: "#d4a348",
    ridgeColor: "#f0ede4",
    labelFill: "#ffffff",
    labelStroke: "rgba(12, 12, 10, 0.85)",
  },
  arrows: {
    highlight: "#d4a348",
    primary: "rgba(212, 163, 72, 0.90)",
    secondary: "rgba(100, 161, 217, 0.72)",
  },
  finishPath: {
    p1Line: "#64a1d9",
    p2Line: "#d9736a",
    badgeBg: "#1a1a18",
    badgeText: "#f0ede4",
    finishBg: "#d4a348",
    finishStroke: "#a87e2d",
    finishText: "#161614",
    costBg: "rgba(24, 24, 22, 0.92)",
    costStroke: "rgba(240, 237, 228, 0.25)",
  },
};

function getCanvasTheme() {
  return CANVAS_THEME;
}

// ---------------------------------------------------------------------------
// Optional Keyboard Shortcuts Display & Centralized Shortcuts Registry
// ---------------------------------------------------------------------------
const SHORTCUT_GROUPS = [
  {
    title: "Camera & View",
    items: [
      { keys: ["Q", "["], desc: "Rotate camera 45° counter-clockwise" },
      { keys: ["E", "]"], desc: "Rotate camera 45° clockwise" },
      { keys: ["F"], desc: "Flip perspective 180°" },
      { keys: ["V"], desc: "Toggle 3D / 2D board projection" },
      { keys: ["R"], desc: "Reset zoom and pan" },
    ],
  },
  {
    title: "Play & Analysis",
    items: [
      { keys: ["H"], desc: "Run MCTS analysis / hint" },
      { keys: ["P"], desc: "Toggle shortest finish path overlay" },
      { keys: ["S"], desc: "Toggle Simulation sandbox mode" },
      { keys: ["A"], desc: "Toggle Neural Spatial Analytics" },
      { keys: ["G"], desc: "Return to Game view" },
    ],
  },
  {
    title: "Timeline & Replay",
    items: [
      { keys: ["←"], desc: "Step backward 1 ply" },
      { keys: ["→"], desc: "Step forward 1 ply" },
      { keys: ["Home"], desc: "Jump to initial setup (Ply 0)" },
      { keys: ["End"], desc: "Jump to latest ply" },
    ],
  },
  {
    title: "Board & Interface",
    items: [
      { keys: ["Wheel"], desc: "Zoom board in / out" },
      { keys: ["Ctrl + Drag"], desc: "Pan board (or Middle-Click drag)" },
      { keys: ["Dbl-Click"], desc: "Reset board zoom and pan" },
      { keys: ["Right-Click"], desc: "Select piece for inspection" },
      { keys: ["Shift + Click"], desc: "Inspect square without moving" },
      { keys: ["M"], desc: "Mute / unmute move audio" },
      { keys: ["K"], desc: "Show / hide inline keycap hints" },
      { keys: ["?"], desc: "Open this shortcuts reference" },
      { keys: ["Esc"], desc: "Close open modal or exit Analytics" },
    ],
  },
];

let showShortcutHints = localStorage.getItem("traversal_show_shortcuts") === "1";

function setShortcutHintsVisible(visible) {
  showShortcutHints = Boolean(visible);
  document.body.classList.toggle("show-shortcuts", showShortcutHints);
  localStorage.setItem("traversal_show_shortcuts", showShortcutHints ? "1" : "0");

  const headerBtn = document.getElementById("toggleShortcutsHintsBtn");
  if (headerBtn) {
    headerBtn.classList.toggle("active-toggle", showShortcutHints);
    headerBtn.setAttribute("aria-pressed", showShortcutHints ? "true" : "false");
  }

  const modalCb = document.getElementById("modalShortcutsHintsToggle");
  if (modalCb) {
    modalCb.checked = showShortcutHints;
  }

  // Trigger redraw since bottom bar height changes slightly when mouse bar toggles
  requestAnimationFrame(() => drawBoard());
}

function renderShortcutsModalContent() {
  const grid = document.getElementById("shortcutsGrid");
  if (!grid) return;
  grid.innerHTML = "";
  for (const grp of SHORTCUT_GROUPS) {
    const card = document.createElement("div");
    card.className = "shortcut-group";
    const rowsHtml = grp.items
      .map((item) => {
        const keyCaps = item.keys
          .map((k) => `<span class="kbd-cap">${escapeHtml(k)}</span>`)
          .join('<span class="shortcut-sep">/</span>');
        return `
          <div class="shortcut-row">
            <span class="shortcut-desc">${escapeHtml(item.desc)}</span>
            <span class="shortcut-keys">${keyCaps}</span>
          </div>
        `;
      })
      .join("");
    card.innerHTML = `
      <div class="shortcut-group-title">${escapeHtml(grp.title)}</div>
      ${rowsHtml}
    `;
    grid.appendChild(card);
  }
}

function openShortcutsModal() {
  const modal = document.getElementById("shortcutsModal");
  if (!modal) return;
  const cb = document.getElementById("modalShortcutsHintsToggle");
  if (cb) cb.checked = showShortcutHints;
  modal.classList.add("open");
}

function closeShortcutsModal() {
  const modal = document.getElementById("shortcutsModal");
  if (modal) modal.classList.remove("open");
}

function toggleShortcutsModal() {
  const modal = document.getElementById("shortcutsModal");
  if (!modal) return;
  if (modal.classList.contains("open")) closeShortcutsModal();
  else openShortcutsModal();
}

function isAnyModalOpen() {
  return ["eloModal", "importModal", "shortcutsModal"].some((id) => {
    const el = document.getElementById(id);
    return el && el.classList.contains("open");
  });
}

// ---------------------------------------------------------------------------
// Global UI State
// ---------------------------------------------------------------------------
let appState = null;
let selectedPlayer = null;    // 0 (P1) or 1 (P2)
let selectedPieceIdx = null;  // 0..5
let hoveredCell = null;       // { r, c } or { finishPlayer: 0|1 }
let hoveredCandidate = null;  // candidate move object from table
let viewMode = "3d";          // "3d" | "2d"
let viewpointIndex = 1;       // 0..7 (S, SW, W, NW, N, NE, E, SE)
let flipBoard = false;        // true when viewpointIndex is North-facing
let showAiArrows = true;
let elevScale = 12;
let isBusy = false;
let autoPlayActive = false;
let autoPlayTimer = null;
let showPieceFinishPath = false;

function getSelectedPieceInfo() {
  if (!appState || selectedPlayer === null || selectedPieceIdx === null) return null;
  const byPlayer = appState.inspector.players_pieces;
  if (byPlayer && byPlayer[selectedPlayer]) {
    const pInfo = byPlayer[selectedPlayer][selectedPieceIdx];
    return pInfo && pInfo.status === 0 ? pInfo : null;
  }
  if (selectedPlayer === appState.current_player) {
    const pInfo = appState.inspector.current_player_pieces[selectedPieceIdx];
    return pInfo && pInfo.status === 0 ? pInfo : null;
  }
  return null;
}

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const canvas = document.getElementById("boardCanvas");
const ctx = canvas.getContext("2d");

const PIECE_SHORT = ["E0", "E1", "V0", "V1", "S0", "S1"];
const PIECE_ROLE_DESC = {
  0: "Explorer (M=5, S=1, 1pt) — Pentagon base, 1-point dome. Special: 1-sq diagonal step or 2-sq orthogonal leap.",
  1: "Vanguard (M=4, S=2, 2pts) — Diamond base, summit ridge. Special: +1 Charge Attack Bonus when d=4.",
  2: "Sentinel (M=3, S=3, 3pts) — Triangle base, prism summit. Special: Instant Victory if 1st piece to finish.",
};

// ---------------------------------------------------------------------------
// API Helpers
// ---------------------------------------------------------------------------
async function apiPost(endpoint, payload = {}) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Request failed");
  }
  return await res.json();
}

async function fetchInitialState() {
  const savedName = localStorage.getItem("traversal_player_name");
  if (savedName) {
    try {
      await apiPost("/api/elo/player_name", { name: savedName });
    } catch (_) {}
  }
  const res = await fetch("/api/state");
  appState = await res.json();
  populateAgentDropdowns();
  syncControlsFromState();
  autoSelectDefaultPiece();
  renderAll();
  scheduleAiIfNeeded();
  initEloSystem();
  initImportSystem();
}

function setBusy(flag) {
  isBusy = flag;
  const sp = document.getElementById("aiSpinner");
  if (flag) sp.classList.add("visible");
  else sp.classList.remove("visible");
  document.getElementById("newGameBtn").disabled = flag;
  document.getElementById("analyzeBtn").disabled = flag || (appState && appState.done);
  document.getElementById("stepAiBtn").disabled = flag || (appState && appState.done);
}

// ---------------------------------------------------------------------------
// Tactile Chess Sound Effects (Web Audio API)
// ---------------------------------------------------------------------------
let audioCtx = null;
let soundEnabled = localStorage.getItem("traversal_sound_enabled") !== "0";

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

window.addEventListener("pointerdown", () => {
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
}, { passive: true });

function playChessMoveSound(lastMove) {
  if (!soundEnabled) return;
  try {
    const aCtx = getAudioContext();
    if (!aCtx) return;
    const now = aCtx.currentTime;
    const isCapture = Boolean(lastMove && lastMove.captured);

    if (isCapture) {
      playChessCaptureSound(aCtx, now);
    } else {
      playChessStandardMoveSound(aCtx, now);
    }
  } catch (err) {
    console.warn("Audio playback error:", err);
  }
}

function playChessStandardMoveSound(aCtx, t) {
  const bufferSize = Math.floor(aCtx.sampleRate * 0.012);
  const buffer = aCtx.createBuffer(1, bufferSize, aCtx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.25));
  }
  const noise = aCtx.createBufferSource();
  noise.buffer = buffer;

  const filter = aCtx.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.setValueAtTime(3200, t);
  filter.Q.setValueAtTime(1.8, t);

  const noiseGain = aCtx.createGain();
  noiseGain.gain.setValueAtTime(0.35, t);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.014);

  noise.connect(filter);
  filter.connect(noiseGain);
  noiseGain.connect(aCtx.destination);
  noise.start(t);
  noise.stop(t + 0.016);

  const osc = aCtx.createOscillator();
  const oscGain = aCtx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(360, t);
  osc.frequency.exponentialRampToValueAtTime(120, t + 0.045);

  oscGain.gain.setValueAtTime(0.45, t);
  oscGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.065);

  osc.connect(oscGain);
  oscGain.connect(aCtx.destination);
  osc.start(t);
  osc.stop(t + 0.07);

  const subOsc = aCtx.createOscillator();
  const subGain = aCtx.createGain();
  subOsc.type = "triangle";
  subOsc.frequency.setValueAtTime(120, t);
  subOsc.frequency.exponentialRampToValueAtTime(65, t + 0.04);

  subGain.gain.setValueAtTime(0.25, t);
  subGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);

  subOsc.connect(subGain);
  subGain.connect(aCtx.destination);
  subOsc.start(t);
  subOsc.stop(t + 0.055);
}

function playChessCaptureSound(aCtx, t) {
  _playSubImpact(aCtx, t, 420, 160, 3800, 0.45);
  _playSubImpact(aCtx, t + 0.014, 280, 110, 2400, 0.38);
}

function _playSubImpact(aCtx, t, startFreq, endFreq, noiseFilterFreq, vol) {
  const bufferSize = Math.floor(aCtx.sampleRate * 0.014);
  const buffer = aCtx.createBuffer(1, bufferSize, aCtx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.22));
  }
  const noise = aCtx.createBufferSource();
  noise.buffer = buffer;

  const filter = aCtx.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.setValueAtTime(noiseFilterFreq, t);
  filter.Q.setValueAtTime(2.0, t);

  const noiseGain = aCtx.createGain();
  noiseGain.gain.setValueAtTime(vol * 0.75, t);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.015);

  noise.connect(filter);
  filter.connect(noiseGain);
  noiseGain.connect(aCtx.destination);
  noise.start(t);
  noise.stop(t + 0.018);

  const osc = aCtx.createOscillator();
  const oscGain = aCtx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(startFreq, t);
  osc.frequency.exponentialRampToValueAtTime(endFreq, t + 0.045);

  oscGain.gain.setValueAtTime(vol, t);
  oscGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);

  osc.connect(oscGain);
  oscGain.connect(aCtx.destination);
  osc.start(t);
  osc.stop(t + 0.065);
}

function updateSoundButtonState() {
  const btn = document.getElementById("toggleSoundBtn");
  if (!btn) return;
  btn.classList.toggle("active-toggle", soundEnabled);
  const label = btn.querySelector(".btn-label");
  if (label) {
    label.textContent = soundEnabled ? "Sound" : "Muted";
  }
  btn.title = soundEnabled ? "Move sounds: ON [M]" : "Move sounds: MUTED [M]";
}

// ---------------------------------------------------------------------------
// Controls Synchronization
// ---------------------------------------------------------------------------
function populateAgentDropdowns() {
  if (!appState || !appState.available_agents) return;
  const p1Sel = document.getElementById("p1AgentSelect");
  const p2Sel = document.getElementById("p2AgentSelect");
  if (!p1Sel || !p2Sel) return;

  const p1Val = p1Sel.value || (appState.settings && appState.settings.p1_agent);
  const p2Val = p2Sel.value || (appState.settings && appState.settings.p2_agent);

  p1Sel.innerHTML = "";
  p2Sel.innerHTML = "";

  const agents = appState.available_agents.filter((ag) => {
    if (!ag || ag.id === "human") return true;
    if (ag.kind === "heuristic") return true;
    if (ag.is_best !== undefined) return !!ag.is_best;
    const baseId = (ag.base_id || ag.id.split(":")[0] || "").toLowerCase().replace(/\\/g, "/");
    const filename = baseId.split("/").pop();
    return filename === "best.pt";
  });

  for (const ag of agents) {
    const o1 = document.createElement("option");
    o1.value = ag.id;
    o1.textContent = ag.label;
    p1Sel.appendChild(o1);

    const o2 = document.createElement("option");
    o2.value = ag.id;
    o2.textContent = ag.label;
    p2Sel.appendChild(o2);
  }

  if (p1Val && !agents.some((a) => a.id === p1Val)) {
    const cur = appState.available_agents.find((a) => a.id === p1Val);
    if (cur) {
      const o1 = document.createElement("option");
      o1.value = cur.id;
      o1.textContent = cur.label;
      p1Sel.appendChild(o1);
    }
  }
  if (p2Val && !agents.some((a) => a.id === p2Val)) {
    const cur = appState.available_agents.find((a) => a.id === p2Val);
    if (cur) {
      const o2 = document.createElement("option");
      o2.value = cur.id;
      o2.textContent = cur.label;
      p2Sel.appendChild(o2);
    }
  }

  if (p1Val) p1Sel.value = p1Val;
  if (p2Val) p2Sel.value = p2Val;
}

function syncControlsFromState() {
  if (!appState) return;
  const s = appState.settings;
  document.getElementById("deviceBadge").textContent = s.device.toUpperCase();
  document.getElementById("p1AgentSelect").value = s.p1_agent;
  document.getElementById("p2AgentSelect").value = s.p2_agent;
  document.getElementById("p1SimsInput").value = s.p1_sims;
  document.getElementById("p2SimsInput").value = s.p2_sims;
  document.getElementById("terrainSelect").value = s.terrain_mode;
}

function hasHumanPlayer() {
  if (!appState || !appState.settings) return false;
  return (
    appState.settings.p1_agent === "human" ||
    appState.settings.p2_agent === "human"
  );
}

function isCurrentPlayerHuman() {
  if (!appState || appState.done || !appState.settings) return false;
  const agent =
    appState.current_player === 0
      ? appState.settings.p1_agent
      : appState.settings.p2_agent;
  return agent === "human";
}

function autoSelectDefaultPiece() {
  if (!appState) {
    selectedPlayer = null;
    selectedPieceIdx = null;
    return;
  }
  if (appState.done) {
    const curInfo = getSelectedPieceInfo();
    if (!curInfo) {
      selectedPlayer = null;
      selectedPieceIdx = null;
    }
    return;
  }
  selectedPlayer = appState.current_player;
  const pieces = appState.inspector.players_pieces
    ? appState.inspector.players_pieces[selectedPlayer]
    : appState.inspector.current_player_pieces;
  if (
    selectedPieceIdx !== null &&
    pieces[selectedPieceIdx] &&
    pieces[selectedPieceIdx].status === 0
  ) {
    return;
  }
  const firstActive = pieces.find(
    (p) => p.status === 0 && (p.legal_moves.length > 0 || p.can_finish)
  );
  selectedPieceIdx = firstActive ? firstActive.piece_idx : null;
}

// ---------------------------------------------------------------------------
// Auto-Play & AI Turn Triggering
// ---------------------------------------------------------------------------
function scheduleAiIfNeeded() {
  if (autoPlayTimer) {
    clearTimeout(autoPlayTimer);
    autoPlayTimer = null;
  }
  if (!appState || appState.done || isBusy) {
    if (appState && appState.done && autoPlayActive) {
      toggleAutoPlay(false);
    }
    return;
  }

  if (appState.simulation_mode && appState.simulation_mode.active) {
    return;
  }

  if (appState.cursor < appState.max_cursor) {
    if (autoPlayActive) {
      const delay = parseInt(document.getElementById("autoSpeedSelect").value, 10) || 600;
      autoPlayTimer = setTimeout(async () => {
        appState = await apiPost("/api/jump", { cursor: appState.cursor + 1 });
        playChessMoveSound(appState.last_move);
        autoSelectDefaultPiece();
        renderAll();
        scheduleAiIfNeeded();
      }, delay);
    }
    return;
  }

  const curAgent =
    appState.current_player === 0
      ? appState.settings.p1_agent
      : appState.settings.p2_agent;
  const otherAgent =
    appState.current_player === 0
      ? appState.settings.p2_agent
      : appState.settings.p1_agent;

  const bothAi = curAgent !== "human" && otherAgent !== "human";

  if (curAgent !== "human" && (!bothAi || autoPlayActive)) {
    const delay = bothAi
      ? parseInt(document.getElementById("autoSpeedSelect").value, 10) || 600
      : 180;
    autoPlayTimer = setTimeout(() => {
      triggerAiMove();
    }, delay);
  }
}

function toggleAutoPlay(forceState) {
  autoPlayActive = forceState !== undefined ? forceState : !autoPlayActive;
  const btn = document.getElementById("autoPlayBtn");
  const label = btn ? btn.querySelector(".btn-label") : null;
  if (autoPlayActive) {
    if (label) label.textContent = "⏸ Pause";
    else if (btn) btn.textContent = "⏸ Pause";
    if (btn) btn.classList.add("active-toggle");
    scheduleAiIfNeeded();
  } else {
    if (label) label.textContent = "▶ Auto";
    else if (btn) btn.textContent = "▶ Auto";
    if (btn) btn.classList.remove("active-toggle");
    if (autoPlayTimer) {
      clearTimeout(autoPlayTimer);
      autoPlayTimer = null;
    }
  }
}

async function triggerAiMove() {
  if (isBusy || !appState || appState.done) return;
  try {
    setBusy(true);
    renderStatusBanner();
    appState = await apiPost("/api/ai_move", {});
    playChessMoveSound(appState.last_move);
    autoSelectDefaultPiece();
  } catch (err) {
    console.error(err);
  } finally {
    setBusy(false);
    renderAll();
    scheduleAiIfNeeded();
  }
}

async function executeHumanAction(action) {
  if (isBusy || !appState || appState.done) return;
  try {
    setBusy(true);
    appState = await apiPost("/api/move", { action });
    playChessMoveSound(appState.last_move);
    autoSelectDefaultPiece();
  } catch (err) {
    alert(err.message);
  } finally {
    setBusy(false);
    renderAll();
    scheduleAiIfNeeded();
  }
}

// ---------------------------------------------------------------------------
// UI Panels Rendering
// ---------------------------------------------------------------------------
function renderAll() {
  if (!appState) return;
  renderStatusBanner();
  renderSimulationBanner();
  renderPlayerCards();
  renderSelectedPiecePanel();
  renderHoverInspector();
  renderAnalysisPanel();
  renderMoveLog();
  drawBoard();
  if (analyticsActive && isAnalyticsStale() && !analyticsLoading) {
    fetchAnalytics();
  }
}

function renderSimulationBanner() {
  const banner = document.getElementById("simulationBanner");
  const toggleBtn = document.getElementById("toggleSimulationBtn");
  if (!banner) return;
  const sim = appState && appState.simulation_mode;
  if (sim && sim.active) {
    banner.style.display = "flex";
    if (toggleBtn) toggleBtn.classList.add("sim-active");
    const plyEl = document.getElementById("simStartPly");
    const countEl = document.getElementById("simPliesCount");
    if (plyEl) plyEl.textContent = sim.sim_start_ply !== null ? sim.sim_start_ply : 0;
    if (countEl) countEl.textContent = sim.sim_plies_count !== null ? sim.sim_plies_count : 0;

    const stepAiBtn = document.getElementById("simStepAiBtn");
    if (stepAiBtn) {
      stepAiBtn.disabled = isBusy || (appState && appState.done);
      const curAgent = appState.current_player === 0 ? appState.settings.p1_agent : appState.settings.p2_agent;
      stepAiBtn.title = curAgent === "human"
        ? `Optionally force an AI move for Player ${appState.current_player + 1}`
        : `Optionally play AI move for Player ${appState.current_player + 1} (${curAgent})`;
    }
  } else {
    banner.style.display = "none";
    if (toggleBtn) toggleBtn.classList.remove("sim-active");
  }
}

function renderStatusBanner() {
  const dot = document.getElementById("turnDot");
  const txt = document.getElementById("statusText");
  document.getElementById("plyCounter").textContent = `${appState.ply_count} / 200`;
  document.getElementById("finishedCounter").textContent = `${appState.total_pieces_finished} / 2`;
  document.getElementById("terrainModeLabel").textContent = appState.terrain_mode;

  const alertBanner = document.getElementById("eloAlertBanner");

  if (appState.done) {
    const reasonMap = {
      sentinel_instant_win: "Sentinel Instant Victory",
      two_piece_finish: "Two-Piece Finish Score Comparison",
      elimination: "Opponent Eliminated",
      no_legal_moves: "Opponent Has No Legal Moves",
      ply_limit: "200-Ply Limit Tiebreaker",
      threefold_repetition: "Threefold Repetition Draw",
    };
    const rText = reasonMap[appState.termination_reason] || appState.termination_reason || "";
    if (appState.winner === 0) {
      dot.style.background = "var(--p1-color)";
      txt.innerHTML = `<span style="color:var(--p1-color); font-weight:700;">Player 1 Wins</span> — ${escapeHtml(rText)}`;
    } else if (appState.winner === 1) {
      dot.style.background = "var(--p2-color)";
      txt.innerHTML = `<span style="color:var(--p2-color); font-weight:700;">Player 2 Wins</span> — ${escapeHtml(rText)}`;
    } else {
      dot.style.background = "var(--accent-gold)";
      txt.innerHTML = `<span style="color:var(--accent-gold); font-weight:700;">Draw Game</span> — ${escapeHtml(rText)}`;
    }

    if (alertBanner && appState.elo && appState.elo.last_update) {
      const up = appState.elo.last_update;
      const d1Sign = up.p1_delta >= 0 ? `+${up.p1_delta.toFixed(1)}` : `${up.p1_delta.toFixed(1)}`;
      const d2Sign = up.p2_delta >= 0 ? `+${up.p2_delta.toFixed(1)}` : `${up.p2_delta.toFixed(1)}`;
      alertBanner.innerHTML = `
        <strong>Elo Result:</strong>
        <span style="color:var(--p1-color); font-weight:700;">${escapeHtml(up.p1_name)}</span>
        <span class="tabular">(${d1Sign} → <strong>${up.p1_rating_after.toFixed(1)}</strong>)</span>
        <span class="muted-hint">vs</span>
        <span style="color:var(--p2-color); font-weight:700;">${escapeHtml(up.p2_name)}</span>
        <span class="tabular">(${d2Sign} → <strong>${up.p2_rating_after.toFixed(1)}</strong>)</span>
        <span style="color:var(--accent-gold); font-weight:700;">— Winner: ${escapeHtml(up.winner_name)}</span>
      `;
      alertBanner.style.display = "flex";
    } else if (alertBanner) {
      alertBanner.style.display = "none";
    }
    return;
  }

  if (alertBanner) alertBanner.style.display = "none";

  const p = appState.current_player;
  const agent = p === 0 ? appState.settings.p1_agent : appState.settings.p2_agent;
  const isSim = appState.simulation_mode && appState.simulation_mode.active;
  dot.style.background = p === 0 ? "var(--p1-color)" : "var(--p2-color)";
  if (isBusy) {
    txt.textContent = `Player ${p + 1} (${agent}) thinking...`;
  } else if (isSim && agent !== "human") {
    txt.innerHTML = `Player ${p + 1} (${escapeHtml(agent)}) — <strong style="color:var(--sim-accent);">Simulation: play move or click Step AI</strong>`;
  } else if (agent === "human") {
    txt.textContent = `Player ${p + 1}'s Turn (Human) — Select a piece and destination`;
  } else {
    txt.textContent = `Player ${p + 1}'s Turn (${agent})`;
  }
}

function renderPlayerCards() {
  document.getElementById("p1Score").textContent = `${appState.scores[0]} pts`;
  document.getElementById("p2Score").textContent = `${appState.scores[1]} pts`;

  if (appState.elo) {
    const p1Pill = document.getElementById("p1EloPill");
    const p2Pill = document.getElementById("p2EloPill");
    if (p1Pill) p1Pill.textContent = `${Math.round(appState.elo.p1_rating)} Elo`;
    if (p2Pill) p2Pill.textContent = `${Math.round(appState.elo.p2_rating)} Elo`;

    const hdrBadge = document.getElementById("headerEloBadge");
    if (hdrBadge) {
      const humanRating = appState.settings.p1_agent === "human"
        ? appState.elo.p1_rating
        : (appState.settings.p2_agent === "human" ? appState.elo.p2_rating : null);
      if (humanRating !== null) {
        hdrBadge.textContent = `${Math.round(humanRating)}`;
      }
    }
  }

  const p1Card = document.getElementById("p1Card");
  const p2Card = document.getElementById("p2Card");
  p1Card.classList.toggle("p1-active", !appState.done && appState.current_player === 0);
  p2Card.classList.toggle("p2-active", !appState.done && appState.current_player === 1);

  for (let pl = 0; pl < 2; pl++) {
    const container = document.getElementById(pl === 0 ? "p1Roster" : "p2Roster");
    container.innerHTML = "";
    for (const pc of appState.pieces[pl]) {
      const div = document.createElement("div");
      div.className = "piece-chip";
      if (pc.status === 1) div.classList.add("captured");
      else if (pc.status === 2) div.classList.add("finished");
      else if (pl === selectedPlayer && pc.piece_idx === selectedPieceIdx) {
        div.classList.add("selected");
      }

      const locText =
        pc.status === 0
          ? `(${pc.r},${pc.c})`
          : pc.status === 2
          ? "FIN"
          : "OUT";

      const shapeIcon = pc.type === 0 ? "⬟" : pc.type === 1 ? "◆" : "▲";
      div.innerHTML = `
        <span class="piece-chip-name">
          <span>${shapeIcon}</span>
          <strong>${PIECE_SHORT[pc.piece_idx]}</strong>
        </span>
        <span class="piece-chip-loc">${locText}</span>
      `;

      if (pc.status === 0) {
        div.addEventListener("click", () => {
          if (selectedPlayer === pl && selectedPieceIdx === pc.piece_idx) {
            selectedPlayer = null;
            selectedPieceIdx = null;
          } else {
            selectedPlayer = pl;
            selectedPieceIdx = pc.piece_idx;
          }
          renderAll();
        });
      }
      container.appendChild(div);
    }
  }
}

function renderSelectedPiecePanel() {
  const infoEl = document.getElementById("selectedPieceInfo");
  const finishBtn = document.getElementById("finishMoveBtn");
  const togglePathBtn = document.getElementById("toggleFinishPathBtn");

  const pInfo = getSelectedPieceInfo();
  if (!appState || !pInfo) {
    infoEl.innerHTML =
      '<span class="muted-hint">Click any active piece on the board or in either player\'s roster to inspect destinations, special rules, and combat math.</span>';
    finishBtn.style.display = "none";
    if (togglePathBtn) togglePathBtn.style.display = "none";
    return;
  }

  const h0 = appState.terrain[pInfo.r][pInfo.c];
  const isTurnPiece = !appState.done && selectedPlayer === appState.current_player;
  const ownerColor = selectedPlayer === 0 ? "var(--p1-color)" : "var(--p2-color)";
  const modeBadge = isTurnPiece
    ? `<span class="badge badge-turn-active">Active Turn</span>`
    : `<span class="badge badge-preview">Preview</span>`;

  let finishPreviewHtml = "";
  if (pInfo.can_finish && !isTurnPiece) {
    finishPreviewHtml = `
      <div class="notice-box notice-finish">
        Eligible to finish on P${selectedPlayer + 1}'s turn (+${pInfo.score_val} pt)${pInfo.is_sentinel_instant_win ? " — Sentinel Instant Win" : ""}
      </div>
    `;
  }

  let vanguardNoticeHtml = "";
  if (pInfo.type === 1) {
    const bonusCount = pInfo.legal_moves.filter(
      (m) => m.is_vanguard_bonus || m.d_min === 4
    ).length;
    vanguardNoticeHtml = `
      <div class="notice-box notice-vanguard">
        <strong>Vanguard Charge:</strong> Squares at <em>d=4</em> receive <strong>+1 Attack Strength</strong> (${bonusCount} reachable).
      </div>
    `;
  }

  const forbiddenCount = pInfo.forbidden_moves ? pInfo.forbidden_moves.length : 0;

  let finishPathHtml = "";
  if (showPieceFinishPath && pInfo.finish_path) {
    const fp = pInfo.finish_path;
    const segsHtml = (fp.turn_segments || [])
      .map(
        (s) =>
          `<div class="path-turn-row">
            <span><strong>Turn ${s.turn}:</strong> (${s.from[0]},${s.from[1]}) → (${s.to[0]},${s.to[1]})</span>
            <span style="color:var(--p1-color); font-weight:600;">${s.action === "finish" ? "FINISH" : `cost ${s.cost}/${pInfo.movement}`}</span>
          </div>`
      )
      .join("");

    finishPathHtml = `
      <div class="notice-box notice-path">
        <div class="section-header-row" style="margin-bottom:4px;">
          <strong style="color:var(--p1-color); font-size:11.5px;">Shortest Route to Finish</strong>
          <span class="badge badge-turns">${fp.turns} ${fp.turns === 1 ? "TURN" : "TURNS"}</span>
        </div>
        <div style="font-size:11px; margin-bottom:4px;">
          Total Cost: <strong>${fp.total_cost}</strong> across <strong>${fp.total_steps}</strong> steps (Target r${selectedPlayer === 0 ? 9 : 0})
        </div>
        <div class="path-turn-list">${segsHtml}</div>
      </div>
    `;
  }

  infoEl.innerHTML = `
    <div class="inspector-header">
      <strong style="font-size:12.5px; color:${ownerColor};">
        P${selectedPlayer + 1} ${escapeHtml(pInfo.name)} at (${pInfo.r}, ${pInfo.c})
      </strong>
      <div style="display:flex; gap:5px; align-items:center;">
        ${modeBadge}
        <span class="badge tabular">h=${h0}</span>
      </div>
    </div>
    <div class="inspector-sub">
      ${PIECE_ROLE_DESC[pInfo.type]}
    </div>
    <div class="inspector-counts tabular">
      Destinations: <strong style="color:var(--accent-green);">${pInfo.legal_moves.length} legal</strong>
      ${forbiddenCount > 0 ? ` · <strong class="muted-hint">${forbiddenCount} blocked</strong>` : ""}
      ${pInfo.blocked_attacks.length > 0 ? ` · <strong style="color:var(--danger);">${pInfo.blocked_attacks.length} outmatched</strong>` : ""}
    </div>
    ${vanguardNoticeHtml}
    ${finishPreviewHtml}
    ${finishPathHtml}
  `;

  if (togglePathBtn) {
    if (pInfo && pInfo.status === 0 && pInfo.finish_path) {
      togglePathBtn.style.display = "inline-flex";
      const labelSpan = togglePathBtn.querySelector(".btn-label");
      const text = showPieceFinishPath
        ? "Hide Path to Finish"
        : `Path to Finish (${pInfo.finish_path.turns} Turns)`;
      if (labelSpan) labelSpan.textContent = text;
      else togglePathBtn.textContent = text;
      togglePathBtn.classList.toggle("active-toggle", showPieceFinishPath);
      togglePathBtn.onclick = () => {
        showPieceFinishPath = !showPieceFinishPath;
        renderSelectedPiecePanel();
        drawBoard();
      };
    } else {
      togglePathBtn.style.display = "none";
    }
  }

  if (isTurnPiece && pInfo.can_finish && pInfo.finish_action !== null) {
    finishBtn.style.display = "inline-flex";
    if (pInfo.is_sentinel_instant_win) {
      finishBtn.textContent = `Finish ${pInfo.name} — Sentinel Instant Victory`;
    } else {
      finishBtn.textContent = `Step Off Board to Finish (+${pInfo.score_val} pt)`;
    }
    finishBtn.onclick = () => executeHumanAction(pInfo.finish_action);
  } else {
    finishBtn.style.display = "none";
  }
}

function renderHoverInspector() {
  const box = document.getElementById("hoverInspector");
  if (!appState || !hoveredCell || hoveredCell.r === undefined) {
    box.innerHTML = `<span class="muted-hint">Hover over any board square to inspect its elevation, occupant defense, and exact movement/combat math from the selected piece.</span>`;
    return;
  }

  const { r, c } = hoveredCell;
  const h = appState.terrain[r][c];

  let occInfo = `<span class="muted-hint">Empty Square</span>`;
  for (let pl = 0; pl < 2; pl++) {
    for (const pc of appState.pieces[pl]) {
      if (pc.status === 0 && pc.r === r && pc.c === c) {
        const defGrid = appState.inspector.player_def
          ? appState.inspector.player_def[pl]
          : (pl === appState.current_player ? appState.inspector.my_def : appState.inspector.opp_def);
        const defVal = defGrid ? defGrid[r][c] : "-";
        const color = pl === 0 ? "var(--p1-color)" : "var(--p2-color)";
        occInfo = `<strong style="color:${color};">P${pl + 1} ${escapeHtml(pc.name)}</strong> (Base S=${pc.strength}, Eff. Def=${defVal})`;
      }
    }
  }

  if (analyticsActive && analyticsData) {
    const meta = ANALYTICS_LAYER_META[analyticsLayer];
    const curVal = getAnalyticsValue(r, c, analyticsLayer);
    const curFmt = curVal !== null && curVal !== undefined ? meta.format(curVal) : "-";
    const sVal = getAnalyticsValue(r, c, "val_Sentinel");
    const vVal = getAnalyticsValue(r, c, "val_Vanguard");
    const eVal = getAnalyticsValue(r, c, "val_Explorer");
    const pol = getAnalyticsValue(r, c, "policy_priors");

    box.innerHTML = `
      <div class="inspector-header" style="border-bottom:1px solid var(--border); padding-bottom:4px;">
        <strong>Tile (${r}, ${c}) · h=${h}</strong>
        <span class="badge badge-turn-active">Analytics</span>
      </div>
      <div class="inspector-metric-box">
        <div class="inspector-metric-label">${escapeHtml(meta.title)}</div>
        <div class="inspector-metric-val">${escapeHtml(curFmt)}</div>
      </div>
      <div class="inspector-grid">
        <div>P1 Dijkstra: <strong style="color:var(--accent-green);">${analyticsData.my_dist[r][c].toFixed(1)}</strong></div>
        <div>P2 Dijkstra: <strong style="color:var(--p2-color);">${analyticsData.opp_dist[r][c].toFixed(1)}</strong></div>
        <div>Topo Adv: <strong>${(analyticsData.topo_adv[r][c] > 0 ? "+" : "") + analyticsData.topo_adv[r][c].toFixed(1)}</strong></div>
        <div>L2 Norm: <strong style="color:var(--accent-gold);">${analyticsData.static_l2_norm[r][c].toFixed(1)}</strong></div>
        <div>Sentinel V(s): <strong style="color:${sVal > 0.8 ? "var(--accent-green)" : "var(--text-main)"};">${(sVal > 0 ? "+" : "") + (sVal !== null ? sVal.toFixed(2) : "-")}</strong></div>
        <div>Vanguard V(s): <strong>${(vVal > 0 ? "+" : "") + (vVal !== null ? vVal.toFixed(2) : "-")}</strong></div>
        <div>Explorer V(s): <strong>${(eVal > 0 ? "+" : "") + (eVal !== null ? eVal.toFixed(2) : "-")}</strong></div>
        <div>Move Policy: <strong style="color:var(--p2-color);">${pol > 0.0005 ? (pol * 100).toFixed(1) + "%" : "0%"}</strong></div>
      </div>
      <div class="inspector-footer">Occupant: ${occInfo}</div>
    `;
    return;
  }

  let moveDetailHtml = "";
  const pInfo = getSelectedPieceInfo();
  if (pInfo) {
    const legalMove = pInfo.legal_moves.find((m) => m.r === r && m.c === c);
    const forbiddenMove = pInfo.forbidden_moves
      ? pInfo.forbidden_moves.find((m) => m.r === r && m.c === c)
      : null;
    const blockedMove = pInfo.blocked_attacks.find((m) => m.r === r && m.c === c);
    const m = legalMove || forbiddenMove || blockedMove;

    if (m) {
      const cVal = m.d_min !== null && m.d_min !== undefined ? m.d_min : m.d_unblocked;
      const costText = m.special
        ? `Explorer Special (${m.special.toUpperCase()})`
        : `Dijkstra Cost = ${cVal !== null && cVal !== undefined ? cVal : "?"} / ${pInfo.movement}`;

      const isTurnPiece = !appState.done && selectedPlayer === appState.current_player;
      let statusBadge = "";
      if (legalMove) {
        statusBadge = `<span class="status-legal">${isTurnPiece ? "LEGAL MOVE" : `POSSIBLE P${selectedPlayer + 1} MOVE`}</span>`;
      } else if (forbiddenMove) {
        statusBadge = `<span class="status-forbidden">FORBIDDEN (${escapeHtml(forbiddenMove.forbidden_desc)})</span>`;
      } else {
        statusBadge = `<span class="status-blocked">ILLEGAL ATTACK (Attack &lt; Defense)</span>`;
      }

      moveDetailHtml += `
        <div class="inspector-move-detail">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap;">
            ${statusBadge}
            <span class="muted-hint tabular" style="font-size:11px;">${escapeHtml(costText)}</span>
          </div>
      `;

      if (showPieceFinishPath && pInfo.finish_path && pInfo.finish_path.turn_segments) {
        const seg = pInfo.finish_path.turn_segments.find(
          (s) => s.path && s.path.some((pt) => pt[0] === r && pt[1] === c)
        );
        if (seg) {
          moveDetailHtml += `
            <div class="notice-box notice-path">
              On Finish Route: Turn ${seg.turn} (${seg.action === "finish" ? "Step off board" : `Waypoint (${seg.to[0]},${seg.to[1]})`})
            </div>
          `;
        }
      }

      if (pInfo.type === 1 && (m.is_vanguard_bonus || m.d_min === 4)) {
        moveDetailHtml += `
          <div class="notice-box notice-vanguard">
            <strong>Vanguard Charge (+1):</strong> Maximum range d=4 adds +1 Attack Strength.
          </div>
        `;
      }

      if (m.combat) {
        const cb = m.combat;
        const hSign = cb.height_mod > 0 ? `+${cb.height_mod}` : `${cb.height_mod}`;
        moveDetailHtml += `
          <div class="formula-grid">
            <div class="formula-card">
              <div class="formula-attack-title">Attack = ${cb.total_attack}</div>
              <div>Base S: ${cb.base_str}</div>
              <div>Height Mod: ${hSign}</div>
              <div class="${cb.vanguard_bonus > 0 ? "formula-vanguard-line" : ""}">Vanguard Charge: +${cb.vanguard_bonus}</div>
              <div>Allied Support: +${cb.ally_support}</div>
            </div>
            <div class="formula-card">
              <div class="formula-def-title">Defense = ${cb.total_defense}</div>
              <div>Defender S: ${cb.def_base}</div>
              <div>Friendly Adj: +${cb.def_support}</div>
              <div class="formula-result" style="color:${cb.can_capture ? "var(--accent-green)" : "var(--danger)"};">
                ${cb.total_attack} ${cb.can_capture ? "≥" : "<"} ${cb.total_defense}
              </div>
            </div>
          </div>
        `;
      }
      moveDetailHtml += `</div>`;
    }
  }

  box.innerHTML = `
    <div class="inspector-header">
      <strong>Square (r=${r}, c=${c})</strong>
      <span class="badge tabular">Height h = ${h}</span>
    </div>
    <div style="margin-top:3px;">Occupant: ${occInfo}</div>
    ${moveDetailHtml}
  `;
}

function renderAnalysisPanel() {
  const metaEl = document.getElementById("analysisMeta");
  const barP1 = document.getElementById("evalBarP1");
  const p1Txt = document.getElementById("evalP1Text");
  const p2Txt = document.getElementById("evalP2Text");
  const rootQEl = document.getElementById("rootQVal");
  const rawValEl = document.getElementById("rawNetVal");
  const tbody = document.getElementById("candidatesBody");

  const an = appState.analysis;
  if (!an) {
    metaEl.textContent = "No search yet";
    barP1.style.width = "50%";
    p1Txt.textContent = "P1: 50.0%";
    p2Txt.textContent = "P2: 50.0%";
    rootQEl.textContent = "—";
    rawValEl.textContent = "—";
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="muted-hint" style="text-align:center; padding:18px 6px;">
          Click <strong>Analyze / Hint</strong> to run MCTS on this board state.
        </td>
      </tr>
    `;
    return;
  }

  const p1Pct = (an.p1_win_prob * 100).toFixed(1);
  const p2Pct = ((1.0 - an.p1_win_prob) * 100).toFixed(1);
  barP1.style.width = `${p1Pct}%`;
  p1Txt.textContent = `P1: ${p1Pct}%`;
  p2Txt.textContent = `P2: ${p2Pct}%`;

  metaEl.textContent = `${an.checkpoint} (${an.simulations}s, ${an.elapsed_ms}ms)`;
  rootQEl.textContent = (an.root_q >= 0 ? "+" : "") + an.root_q.toFixed(3);
  rawValEl.textContent = (an.raw_value >= 0 ? "+" : "") + an.raw_value.toFixed(3);

  tbody.innerHTML = "";
  an.top_moves.forEach((cand, idx) => {
    const tr = document.createElement("tr");
    tr.className = "cand-row";
    const moveShort = cand.is_finish
      ? `${PIECE_SHORT[cand.piece_idx]} → FINISH`
      : `${PIECE_SHORT[cand.piece_idx]} (${cand.from_r},${cand.from_c})→(${cand.to_r},${cand.to_c})${cand.captured ? "×" : ""}`;

    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td class="${idx === 0 ? "cand-top-move" : "cand-move"}">${moveShort}</td>
      <td>${cand.visits}</td>
      <td>${(cand.policy_share * 100).toFixed(1)}%</td>
      <td>${(cand.prior * 100).toFixed(1)}%</td>
      <td class="${cand.win_prob >= 0.5 ? "cand-win-pos" : "cand-win-neg"}">
        ${(cand.win_prob * 100).toFixed(1)}%
      </td>
    `;

    tr.addEventListener("mouseenter", () => {
      hoveredCandidate = cand;
      drawBoard();
    });
    tr.addEventListener("mouseleave", () => {
      hoveredCandidate = null;
      drawBoard();
    });
    tr.addEventListener("click", () => {
      if (!appState.done && appState.cursor === appState.max_cursor) {
        executeHumanAction(cand.action);
      }
    });
    tbody.appendChild(tr);
  });
}

function renderMoveLog() {
  const logEl = document.getElementById("moveLog");
  logEl.innerHTML = "";

  const startDiv = document.createElement("div");
  startDiv.className = "log-entry" + (appState.cursor === 0 ? " active-ply" : "");
  startDiv.innerHTML = `<span>0. Initial Board Setup</span>`;
  startDiv.onclick = async () => {
    toggleAutoPlay(false);
    appState = await apiPost("/api/jump", { cursor: 0 });
    autoSelectDefaultPiece();
    renderAll();
  };
  logEl.appendChild(startDiv);

  appState.move_log.forEach((m, idx) => {
    const curIdx = idx + 1;
    const div = document.createElement("div");
    div.className = "log-entry" + (appState.cursor === curIdx ? " active-ply" : "");
    const pColor = m.player === 0 ? "var(--p1-color)" : "var(--p2-color)";
    div.innerHTML = `
      <span><strong style="color:${pColor};">#${m.ply}</strong> ${escapeHtml(m.text)}</span>
    `;
    div.onclick = async () => {
      toggleAutoPlay(false);
      appState = await apiPost("/api/jump", { cursor: curIdx });
      autoSelectDefaultPiece();
      renderAll();
    };
    logEl.appendChild(div);
  });

  logEl.scrollTop = logEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// 8 Viewpoints & Camera State (4 Sides Perspective + 4 Corners Isometric)
// ---------------------------------------------------------------------------
function getDefaultZoom(idx = viewpointIndex) {
  const isCorner = idx % 2 === 1;
  return isCorner ? 1.4 : 1.1;
}

function setViewpoint(idx) {
  viewpointIndex = ((idx % 8) + 8) % 8;
  flipBoard = viewpointIndex === 3 || viewpointIndex === 4 || viewpointIndex === 5;
  const sel = document.getElementById("viewpointSelect");
  if (sel) sel.value = String(viewpointIndex);
  document.querySelectorAll(".compass-btn").forEach((btn) => {
    btn.classList.toggle("active-toggle", parseInt(btn.dataset.vp, 10) === viewpointIndex);
  });
  const flipBtn = document.getElementById("flipBoardBtn");
  if (flipBtn) flipBtn.classList.toggle("active-toggle", flipBoard);

  zoomLevel = getDefaultZoom(viewpointIndex);
  panOffsetX = 0;
  panOffsetY = 0;
  updateZoomIndicator();

  drawBoard();
}

function rotateView(delta) {
  setViewpoint(viewpointIndex + delta);
}

// ---------------------------------------------------------------------------
// Board Canvas Renderer & Hit-Testing Setup
// ---------------------------------------------------------------------------
let cellPolygons = [];
let finishZones = [];

function elevationColor(h) {
  const palette = getCanvasTheme().elevation;
  const idx = Math.max(0, Math.min(palette.length - 1, h));
  return palette[idx];
}

function pointInPoly(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// ---------------------------------------------------------------------------
// Zoom and Pan Board State
// ---------------------------------------------------------------------------
let zoomLevel = 1.4;
let panOffsetX = 0;
let panOffsetY = 0;
const MIN_ZOOM = 0.4;
const MAX_ZOOM = 3.0;

let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panInitialOffsetX = 0;
let panInitialOffsetY = 0;
let hasDraggedPan = false;

function updateZoomIndicator() {
  const el = document.getElementById("zoomResetBtn");
  if (el) {
    el.textContent = `${Math.round(zoomLevel * 100)}%`;
  }
}

function setZoom(newZoom, focusX = null, focusY = null) {
  const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
  if (Math.abs(clamped - zoomLevel) < 0.001) return;

  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;

  if (focusX !== null && focusY !== null) {
    const boardX = (focusX - (W / 2 + panOffsetX)) / zoomLevel + W / 2;
    const boardY = (focusY - (H / 2 + panOffsetY)) / zoomLevel + H / 2;

    panOffsetX = focusX - (boardX - W / 2) * clamped - W / 2;
    panOffsetY = focusY - (boardY - H / 2) * clamped - H / 2;
  }

  zoomLevel = Math.round(clamped * 100) / 100;
  updateZoomIndicator();
  drawBoard();
}

function zoomIn() {
  setZoom(zoomLevel * 1.2);
}

function zoomOut() {
  setZoom(zoomLevel / 1.2);
}

function resetZoomPan() {
  zoomLevel = getDefaultZoom();
  panOffsetX = 0;
  panOffsetY = 0;
  updateZoomIndicator();
  drawBoard();
}

function drawBoard() {
  if (!appState) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.save();
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  ctx.save();
  ctx.translate(W / 2 + panOffsetX, H / 2 + panOffsetY);
  ctx.scale(zoomLevel, zoomLevel);
  ctx.translate(-W / 2, -H / 2);

  cellPolygons = [];
  finishZones = [];

  const legalMap = new Map();
  const forbiddenMap = new Map();
  const blockedMap = new Map();
  let canFinishSelected = false;
  let finishActionSelected = null;

  const pInfo = getSelectedPieceInfo();
  if (pInfo) {
    for (const m of pInfo.legal_moves) {
      legalMap.set(`${m.r},${m.c}`, m);
    }
    if (pInfo.forbidden_moves) {
      for (const f of pInfo.forbidden_moves) {
        forbiddenMap.set(`${f.r},${f.c}`, f);
      }
    }
    for (const b of pInfo.blocked_attacks) {
      blockedMap.set(`${b.r},${b.c}`, b);
    }
    canFinishSelected = pInfo.can_finish;
    finishActionSelected = pInfo.finish_action;
  }

  if (viewMode === "2d") {
    drawBoard2D(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected);
  } else {
    const isCorner = viewpointIndex % 2 === 1;
    if (isCorner) {
      drawBoard3DCorner(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected);
    } else {
      drawBoard3DSide(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected);
    }
  }

  if (showPieceFinishPath && pInfo && pInfo.finish_path) {
    drawFinishPathOverlay(pInfo.finish_path, selectedPlayer);
  }

  if (showAiArrows && appState.analysis && appState.analysis.top_moves) {
    drawCandidateArrows(appState.analysis.top_moves);
  }
  if (hoveredCandidate) {
    drawCandidateArrows([hoveredCandidate], true);
  }

  ctx.restore();
  ctx.restore();
}

function getCellCenter(r, c) {
  const found = cellPolygons.find((item) => item.r === r && item.c === c);
  return found ? found.topCenter : null;
}

// ---------------------------------------------------------------------------
// 3D CORNER VIEWS (EXACT 2:1 ISOMETRIC PROJECTION - 4 CLOCKWISE CORNERS)
// ---------------------------------------------------------------------------
function drawBoard3DCorner(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const tw = Math.max(22, Math.min(W / 24, H / 16));
  const th = tw * 0.52;
  const originX = W / 2;
  const originY = H * 0.54 - 10 * th + 25;

  function isoProject(dr, dc, hVal = 0) {
    const x = originX + (dc - dr) * tw;
    const y = originY + (dc + dr) * th - hVal * elevScale;
    return { x, y };
  }

  function cornerMap(dr, dc) {
    if (viewpointIndex === 1) return [9 - dc, 9 - dr];
    if (viewpointIndex === 3) return [dr, 9 - dc];
    if (viewpointIndex === 5) return [dc, dr];
    return [9 - dr, dc];
  }

  if (viewpointIndex === 1) {
    drawFinishZoneIso(isoProject, "dc", -0.95, 0, canFinishSelected, finishActionSelected);
    drawFinishZoneIso(isoProject, "dc", 10.35, 1, canFinishSelected, finishActionSelected);
  } else if (viewpointIndex === 3) {
    drawFinishZoneIso(isoProject, "dr", -0.95, 1, canFinishSelected, finishActionSelected);
    drawFinishZoneIso(isoProject, "dr", 10.35, 0, canFinishSelected, finishActionSelected);
  } else if (viewpointIndex === 5) {
    drawFinishZoneIso(isoProject, "dc", -0.95, 1, canFinishSelected, finishActionSelected);
    drawFinishZoneIso(isoProject, "dc", 10.35, 0, canFinishSelected, finishActionSelected);
  } else {
    drawFinishZoneIso(isoProject, "dr", -0.95, 0, canFinishSelected, finishActionSelected);
    drawFinishZoneIso(isoProject, "dr", 10.35, 1, canFinishSelected, finishActionSelected);
  }

  for (let dr = 0; dr < 10; dr++) {
    for (let dc = 0; dc < 10; dc++) {
      const [r, c] = cornerMap(dr, dc);
      const h = appState.terrain[r][c];
      const col = elevationColor(h);

      const baseHeight = 10;
      const totalLift = h * elevScale + baseHeight;

      const pN = isoProject(dr, dc, h);
      const pE = isoProject(dr, dc + 1, h);
      const pS = isoProject(dr + 1, dc + 1, h);
      const pW = isoProject(dr + 1, dc, h);

      const bE = { x: pE.x, y: pE.y + totalLift };
      const bS = { x: pS.x, y: pS.y + totalLift };
      const bW = { x: pW.x, y: pW.y + totalLift };

      ctx.beginPath();
      ctx.moveTo(pW.x, pW.y);
      ctx.lineTo(pS.x, pS.y);
      ctx.lineTo(bS.x, bS.y);
      ctx.lineTo(bW.x, bW.y);
      ctx.closePath();
      ctx.fillStyle = col.left;
      ctx.fill();
      ctx.strokeStyle = theme.pillarStroke;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(pS.x, pS.y);
      ctx.lineTo(pE.x, pE.y);
      ctx.lineTo(bE.x, bE.y);
      ctx.lineTo(bS.x, bS.y);
      ctx.closePath();
      ctx.fillStyle = col.right;
      ctx.fill();
      ctx.stroke();

      if (h > 0 && elevScale >= 4) {
        ctx.strokeStyle = theme.contourStroke;
        ctx.lineWidth = 0.7;
        for (let step = 1; step <= h; step++) {
          const dy = step * elevScale;
          ctx.beginPath();
          ctx.moveTo(pW.x, pW.y + dy);
          ctx.lineTo(pS.x, pS.y + dy);
          ctx.lineTo(pE.x, pE.y + dy);
          ctx.stroke();
        }
      }

      const poly = [pN, pE, pS, pW];
      const topCenter = {
        x: (pN.x + pS.x) / 2,
        y: (pN.y + pS.y) / 2,
      };
      cellPolygons.push({ r, c, poly, topCenter });

      ctx.beginPath();
      ctx.moveTo(pN.x, pN.y);
      ctx.lineTo(pE.x, pE.y);
      ctx.lineTo(pS.x, pS.y);
      ctx.lineTo(pW.x, pW.y);
      ctx.closePath();
      if (analyticsActive && analyticsData) {
        const aVal = getAnalyticsValue(r, c, analyticsLayer);
        if (aVal !== null && aVal !== undefined) {
          const mm = getAnalyticsMinMax(analyticsLayer);
          ctx.fillStyle = getAnalyticsColor(aVal, mm.min, mm.max, ANALYTICS_LAYER_META[analyticsLayer].type);
        } else {
          ctx.fillStyle = col.top;
        }
      } else {
        ctx.fillStyle = col.top;
      }
      ctx.fill();
      ctx.strokeStyle = analyticsActive ? theme.gridStrokeAnalytics : theme.gridStroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      decorateCellTop(r, c, poly, topCenter, tw * 0.42, legalMap, blockedMap, forbiddenMap, true);
    }
  }

  ctx.fillStyle = theme.coordColor;
  ctx.font = `600 10.5px ${theme.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const [r00] = cornerMap(0, 0);
  const [r90] = cornerMap(9, 0);
  const [r09] = cornerMap(0, 9);

  for (let i = 0; i < 10; i++) {
    const [ri, ci] = cornerMap(i, 0);
    const labelL = r00 !== r90 ? `r${ri}` : `c${ci}`;
    const posL = isoProject(i + 0.5, -0.45, 0);
    ctx.fillText(labelL, posL.x, posL.y);

    const [rj, cj] = cornerMap(0, i);
    const labelT = r00 !== r09 ? `r${rj}` : `c${cj}`;
    const posT = isoProject(-0.45, i + 0.5, 0);
    ctx.fillText(labelT, posT.x, posT.y);
  }
}

function drawFinishZoneIso(isoProject, axis, val, finishOwnerPlayer, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const fzTheme = theme.finishZone;
  const isSelectedOwner = canFinishSelected && selectedPlayer === finishOwnerPlayer;
  const isExecutableFinish =
    isSelectedOwner && !appState.done && appState.current_player === finishOwnerPlayer && finishActionSelected !== null;
  let p0, p1, p2, p3;
  const stripW = 0.6;
  if (axis === "dr") {
    p0 = isoProject(val, 0, 0);
    p1 = isoProject(val, 10, 0);
    p2 = isoProject(val + stripW, 10, 0);
    p3 = isoProject(val + stripW, 0, 0);
  } else {
    p0 = isoProject(0, val, 0);
    p1 = isoProject(10, val, 0);
    p2 = isoProject(10, val + stripW, 0);
    p3 = isoProject(0, val + stripW, 0);
  }
  const poly = [p0, p1, p2, p3];
  const center = {
    x: (p0.x + p1.x + p2.x + p3.x) / 4,
    y: (p0.y + p1.y + p2.y + p3.y) / 4,
  };
  finishZones.push({
    player: finishOwnerPlayer,
    poly,
    center,
    action: isExecutableFinish ? finishActionSelected : null,
  });

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.lineTo(p3.x, p3.y);
  ctx.closePath();
  ctx.fillStyle = isSelectedOwner ? fzTheme.activeFill : fzTheme.idleFill;
  ctx.strokeStyle = isSelectedOwner
    ? fzTheme.activeStroke
    : finishOwnerPlayer === 0
    ? fzTheme.p1Stroke
    : fzTheme.p2Stroke;
  ctx.lineWidth = isSelectedOwner ? 2.2 : 1;
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isSelectedOwner ? fzTheme.activeText : fzTheme.idleText;
  ctx.font = `700 10.5px ${theme.fontSans}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(
    isExecutableFinish
      ? `CLICK TO FINISH P${finishOwnerPlayer + 1} PIECE`
      : isSelectedOwner
      ? `P${finishOwnerPlayer + 1} PIECE CAN FINISH HERE`
      : `P${finishOwnerPlayer + 1} FINISH ROW ${finishOwnerPlayer === 0 ? 9 : 0}`,
    center.x,
    center.y
  );
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 3D SIDE VIEWS (PERSPECTIVE PROJECTION)
// ---------------------------------------------------------------------------
function drawBoard3DSide(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const D = 13.5;
  const W_front = Math.min(W * 0.82, 880);
  const baseY_front = H * 0.84;
  const baseY_back = H * 0.16;

  const H_cam = (baseY_front - baseY_back) / (1.0 / D - 1.0 / (10.0 + D));
  const y_horizon = baseY_front - H_cam / D;
  const baseHeight = 12;

  function sideProject(u, v, hVal = 0) {
    const persp = D / (v + D);
    const x = W / 2 + ((u - 5.0) / 10.0) * W_front * persp;
    const y_ground = y_horizon + H_cam / (v + D);
    const y = y_ground - hVal * elevScale * 1.6 * persp;
    return { x, y, persp };
  }

  function sideMap(u, v) {
    if (viewpointIndex === 0) return [v, u];
    if (viewpointIndex === 2) return [9 - u, v];
    if (viewpointIndex === 4) return [9 - v, 9 - u];
    return [u, 9 - v];
  }

  if (viewpointIndex === 0) {
    drawFinishZoneSide(sideProject, "v", 10.15, 0.7, 0, canFinishSelected, finishActionSelected);
    drawFinishZoneSide(sideProject, "v", -0.85, 0.7, 1, canFinishSelected, finishActionSelected);
  } else if (viewpointIndex === 2) {
    drawFinishZoneSide(sideProject, "u", -0.85, 0.7, 0, canFinishSelected, finishActionSelected);
    drawFinishZoneSide(sideProject, "u", 10.15, 0.7, 1, canFinishSelected, finishActionSelected);
  } else if (viewpointIndex === 4) {
    drawFinishZoneSide(sideProject, "v", -0.85, 0.7, 0, canFinishSelected, finishActionSelected);
    drawFinishZoneSide(sideProject, "v", 10.15, 0.7, 1, canFinishSelected, finishActionSelected);
  } else if (viewpointIndex === 6) {
    drawFinishZoneSide(sideProject, "u", -0.85, 0.7, 1, canFinishSelected, finishActionSelected);
    drawFinishZoneSide(sideProject, "u", 10.15, 0.7, 0, canFinishSelected, finishActionSelected);
  }

  for (let v = 9; v >= 0; v--) {
    const colOrder = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5];
    for (const u of colOrder) {
      const [r, c] = sideMap(u, v);
      const h = appState.terrain[r][c];
      const col = elevationColor(h);

      const pFL = sideProject(u, v, h);
      const pFR = sideProject(u + 1, v, h);
      const pBR = sideProject(u + 1, v + 1, h);
      const pBL = sideProject(u, v + 1, h);

      const bFL = sideProject(u, v, 0);
      const bFR = sideProject(u + 1, v, 0);
      const drop = baseHeight * pFL.persp;
      const bFL_y = bFL.y + drop;
      const bFR_y = bFR.y + drop;

      ctx.beginPath();
      ctx.moveTo(pFL.x, pFL.y);
      ctx.lineTo(pFR.x, pFR.y);
      ctx.lineTo(bFR.x, bFR_y);
      ctx.lineTo(bFL.x, bFL_y);
      ctx.closePath();
      ctx.fillStyle = col.left;
      ctx.fill();
      ctx.strokeStyle = theme.pillarStroke;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      if (u < 5) {
        const bBR = sideProject(u + 1, v + 1, 0);
        const bBR_y = bBR.y + drop;
        ctx.beginPath();
        ctx.moveTo(pFR.x, pFR.y);
        ctx.lineTo(pBR.x, pBR.y);
        ctx.lineTo(bBR.x, bBR_y);
        ctx.lineTo(bFR.x, bFR_y);
        ctx.closePath();
        ctx.fillStyle = col.right;
        ctx.fill();
        ctx.strokeStyle = theme.pillarStroke;
        ctx.lineWidth = 0.7;
        ctx.stroke();
      } else {
        const bBL = sideProject(u, v + 1, 0);
        const bBL_y = bBL.y + drop;
        ctx.beginPath();
        ctx.moveTo(pBL.x, pBL.y);
        ctx.lineTo(pFL.x, pFL.y);
        ctx.lineTo(bFL.x, bFL_y);
        ctx.lineTo(bBL.x, bBL_y);
        ctx.closePath();
        ctx.fillStyle = col.left;
        ctx.fill();
        ctx.strokeStyle = theme.pillarStroke;
        ctx.lineWidth = 0.7;
        ctx.stroke();
      }

      if (h > 0 && elevScale >= 4) {
        ctx.strokeStyle = theme.contourStroke;
        ctx.lineWidth = 0.7;
        for (let step = 0; step < h; step++) {
          const ptL = sideProject(u, v, step);
          const ptR = sideProject(u + 1, v, step);
          ctx.beginPath();
          ctx.moveTo(ptL.x, ptL.y);
          ctx.lineTo(ptR.x, ptR.y);
          ctx.stroke();

          if (u < 5) {
            const ptBR_k = sideProject(u + 1, v + 1, step);
            ctx.beginPath();
            ctx.moveTo(ptR.x, ptR.y);
            ctx.lineTo(ptBR_k.x, ptBR_k.y);
            ctx.stroke();
          } else {
            const ptBL_k = sideProject(u, v + 1, step);
            ctx.beginPath();
            ctx.moveTo(ptBL_k.x, ptBL_k.y);
            ctx.lineTo(ptL.x, ptL.y);
            ctx.stroke();
          }
        }
      }

      const poly = [pFL, pFR, pBR, pBL];
      const topCenter = {
        x: (pFL.x + pFR.x + pBR.x + pBL.x) / 4,
        y: (pFL.y + pFR.y + pBR.y + pBL.y) / 4,
      };
      cellPolygons.push({ r, c, poly, topCenter });

      ctx.beginPath();
      ctx.moveTo(pFL.x, pFL.y);
      ctx.lineTo(pFR.x, pFR.y);
      ctx.lineTo(pBR.x, pBR.y);
      ctx.lineTo(pBL.x, pBL.y);
      ctx.closePath();
      if (analyticsActive && analyticsData) {
        const aVal = getAnalyticsValue(r, c, analyticsLayer);
        if (aVal !== null && aVal !== undefined) {
          const mm = getAnalyticsMinMax(analyticsLayer);
          ctx.fillStyle = getAnalyticsColor(aVal, mm.min, mm.max, ANALYTICS_LAYER_META[analyticsLayer].type);
        } else {
          ctx.fillStyle = col.top;
        }
      } else {
        ctx.fillStyle = col.top;
      }
      ctx.fill();
      ctx.strokeStyle = analyticsActive ? theme.gridStrokeAnalytics : theme.gridStroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      const tileRad = Math.abs(pFR.x - pFL.x) * 0.40;
      decorateCellTop(r, c, poly, topCenter, tileRad, legalMap, blockedMap, forbiddenMap, true);
    }
  }

  ctx.fillStyle = theme.coordColor;
  ctx.font = `600 10.5px ${theme.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const [r_f0] = sideMap(0, 0);
  const [r_f9] = sideMap(9, 0);
  for (let u = 0; u < 10; u++) {
    const [ru, cu] = sideMap(u, 0);
    const colLabel = r_f0 !== r_f9 ? `r${ru}` : `c${cu}`;
    const posFront = sideProject(u + 0.5, -0.42, 0);
    ctx.fillText(colLabel, posFront.x, posFront.y);
  }

  const [r_s0] = sideMap(0, 0);
  const [r_s9] = sideMap(0, 9);
  for (let v = 0; v < 10; v++) {
    const [rv, cv] = sideMap(0, v);
    const rowLabel = r_s0 !== r_s9 ? `r${rv}` : `c${cv}`;
    const posSide = sideProject(-0.55, v + 0.5, 0);
    ctx.fillText(rowLabel, posSide.x, posSide.y);
  }
}

function drawFinishZoneSide(sideProject, axis, val, widthStrip, finishOwnerPlayer, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const fzTheme = theme.finishZone;
  const isSelectedOwner = canFinishSelected && selectedPlayer === finishOwnerPlayer;
  const isExecutableFinish =
    isSelectedOwner && !appState.done && appState.current_player === finishOwnerPlayer && finishActionSelected !== null;
  let p0, p1, p2, p3;
  if (axis === "v") {
    p0 = sideProject(0, val, 0);
    p1 = sideProject(10, val, 0);
    p2 = sideProject(10, val + widthStrip, 0);
    p3 = sideProject(0, val + widthStrip, 0);
  } else {
    p0 = sideProject(val, 0, 0);
    p1 = sideProject(val, 10, 0);
    p2 = sideProject(val + widthStrip, 10, 0);
    p3 = sideProject(val + widthStrip, 0, 0);
  }
  const poly = [p0, p1, p2, p3];
  const center = {
    x: (p0.x + p1.x + p2.x + p3.x) / 4,
    y: (p0.y + p1.y + p2.y + p3.y) / 4,
  };
  finishZones.push({
    player: finishOwnerPlayer,
    poly,
    center,
    action: isExecutableFinish ? finishActionSelected : null,
  });

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.lineTo(p3.x, p3.y);
  ctx.closePath();
  ctx.fillStyle = isSelectedOwner ? fzTheme.activeFill : fzTheme.idleFill;
  ctx.strokeStyle = isSelectedOwner
    ? fzTheme.activeStroke
    : finishOwnerPlayer === 0
    ? fzTheme.p1Stroke
    : fzTheme.p2Stroke;
  ctx.lineWidth = isSelectedOwner ? 2.2 : 1;
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isSelectedOwner ? fzTheme.activeText : fzTheme.idleText;
  ctx.font = `700 10.5px ${theme.fontSans}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(
    isExecutableFinish
      ? `CLICK TO FINISH P${finishOwnerPlayer + 1} PIECE`
      : isSelectedOwner
      ? `P${finishOwnerPlayer + 1} PIECE CAN FINISH HERE`
      : `PLAYER ${finishOwnerPlayer + 1} FINISH ROW ${finishOwnerPlayer === 0 ? 9 : 0}`,
    center.x,
    center.y
  );
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 2D Tactical Renderer Supporting All 8 Viewpoints
// ---------------------------------------------------------------------------
function drawBoard2D(W, H, legalMap, blockedMap, forbiddenMap, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const marginY = 65;
  const boardPx = Math.min(W - 80, H - marginY * 2);
  const cellSz = Math.max(26, Math.floor(boardPx / 10));
  const totalSz = cellSz * 10;
  const startX = (W - totalSz) / 2;
  const startY = (H - totalSz) / 2;

  const isFlipped2D = viewpointIndex === 3 || viewpointIndex === 4 || viewpointIndex === 5;
  const topFinPlayer = isFlipped2D ? 1 : 0;
  const botFinPlayer = isFlipped2D ? 0 : 1;

  drawFinishZoneBanner2D(startX, startY - 36, totalSz, 26, topFinPlayer, canFinishSelected, finishActionSelected);
  drawFinishZoneBanner2D(startX, startY + totalSz + 10, totalSz, 26, botFinPlayer, canFinishSelected, finishActionSelected);

  for (let dr = 0; dr < 10; dr++) {
    for (let dc = 0; dc < 10; dc++) {
      const r = isFlipped2D ? dr : 9 - dr;
      const c = isFlipped2D ? 9 - dc : dc;
      const x = startX + dc * cellSz;
      const y = startY + dr * cellSz;
      const h = appState.terrain[r][c];
      const col = elevationColor(h);

      const poly = [
        { x, y },
        { x: x + cellSz, y },
        { x: x + cellSz, y: y + cellSz },
        { x, y: y + cellSz },
      ];
      const topCenter = { x: x + cellSz / 2, y: y + cellSz / 2 };
      cellPolygons.push({ r, c, poly, topCenter });

      if (analyticsActive && analyticsData) {
        const aVal = getAnalyticsValue(r, c, analyticsLayer);
        if (aVal !== null && aVal !== undefined) {
          const mm = getAnalyticsMinMax(analyticsLayer);
          ctx.fillStyle = getAnalyticsColor(aVal, mm.min, mm.max, ANALYTICS_LAYER_META[analyticsLayer].type);
        } else {
          ctx.fillStyle = col.top;
        }
      } else {
        ctx.fillStyle = col.top;
      }
      ctx.fillRect(x, y, cellSz, cellSz);

      ctx.strokeStyle = analyticsActive ? theme.gridStrokeAnalytics : theme.gridStroke;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSz, cellSz);

      decorateCellTop(r, c, poly, topCenter, cellSz * 0.38, legalMap, blockedMap, forbiddenMap, false);
    }
  }

  ctx.fillStyle = theme.coordColor;
  ctx.font = `600 10.5px ${theme.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < 10; i++) {
    const r = isFlipped2D ? i : 9 - i;
    const c = isFlipped2D ? 9 - i : i;
    ctx.fillText(`r${r}`, startX - 16, startY + i * cellSz + cellSz / 2);
    ctx.fillText(`c${c}`, startX + i * cellSz + cellSz / 2, startY - 9);
  }
}

function drawFinishZoneBanner2D(x, y, w, h, finishOwnerPlayer, canFinishSelected, finishActionSelected) {
  const theme = getCanvasTheme();
  const fzTheme = theme.finishZone;
  const isSelectedOwner = canFinishSelected && selectedPlayer === finishOwnerPlayer;
  const isExecutableFinish =
    isSelectedOwner && !appState.done && appState.current_player === finishOwnerPlayer && finishActionSelected !== null;
  const poly = [
    { x, y },
    { x: x + w, y },
    { x: x + w, y: y + h },
    { x, y: y + h },
  ];
  const center = { x: x + w / 2, y: y + h / 2 };
  finishZones.push({ player: finishOwnerPlayer, poly, center, action: isExecutableFinish ? finishActionSelected : null });

  ctx.save();
  ctx.fillStyle = isSelectedOwner ? fzTheme.activeFill : fzTheme.idleFill;
  ctx.strokeStyle = isSelectedOwner
    ? fzTheme.activeStroke
    : finishOwnerPlayer === 0
    ? fzTheme.p1Stroke
    : fzTheme.p2Stroke;
  ctx.lineWidth = isSelectedOwner ? 2 : 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isSelectedOwner ? fzTheme.activeText : fzTheme.idleText;
  ctx.font = `700 11px ${theme.fontSans}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const label = isExecutableFinish
    ? `CLICK TO FINISH P${finishOwnerPlayer + 1} PIECE`
    : isSelectedOwner
    ? `P${finishOwnerPlayer + 1} PIECE CAN FINISH HERE`
    : `PLAYER ${finishOwnerPlayer + 1} FINISH ZONE (Beyond Row ${finishOwnerPlayer === 0 ? 9 : 0})`;
  ctx.fillText(label, center.x, center.y);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Cell Top Overlays & Piece Shape Rendering
// ---------------------------------------------------------------------------
function decorateCellTop(r, c, poly, topCenter, radius, legalMap, blockedMap, forbiddenMap, is3D = true) {
  const theme = getCanvasTheme();
  const ov = theme.overlays;
  const key = `${r},${c}`;
  const isHovered = hoveredCell && hoveredCell.r === r && hoveredCell.c === c;
  const legalMove = legalMap ? legalMap.get(key) : null;
  const forbiddenMove = forbiddenMap ? forbiddenMap.get(key) : null;
  const blockedMove = blockedMap ? blockedMap.get(key) : null;
  const lastMove = appState.last_move;
  const isLastFrom = lastMove && lastMove.from_r === r && lastMove.from_c === c;
  const isLastTo = lastMove && !lastMove.is_finish && lastMove.to_r === r && lastMove.to_c === c;
  const isTurnPiece = !appState.done && selectedPlayer === appState.current_player;

  const pInfo = getSelectedPieceInfo();
  const isVanguard = pInfo && pInfo.type === 1;
  const isVanguardBonus =
    (legalMove && (legalMove.is_vanguard_bonus || (isVanguard && legalMove.d_min === 4))) ||
    (forbiddenMove && (forbiddenMove.is_vanguard_bonus || (isVanguard && (forbiddenMove.d_min === 4 || forbiddenMove.d_unblocked === 4)))) ||
    (blockedMove && (blockedMove.is_vanguard_bonus || (isVanguard && blockedMove.d_min === 4)));

  function fillPoly(color, strokeColor = null, lineW = 1.5) {
    ctx.beginPath();
    ctx.moveTo(poly[0].x, poly[0].y);
    for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
    ctx.closePath();
    if (color) {
      ctx.fillStyle = color;
      ctx.fill();
    }
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineW;
      ctx.stroke();
    }
  }

  if (isLastFrom || isLastTo) {
    fillPoly(ov.lastMoveFill, ov.lastMoveStroke, 1.6);
  }

  if (legalMove) {
    if (isVanguardBonus) {
      if (legalMove.is_enemy) {
        fillPoly(ov.vanguardCapFill, ov.vanguardCapStroke, 2.4);
      } else {
        fillPoly(ov.vanguardFill, ov.vanguardStroke, 2.0);
      }
    } else if (legalMove.is_enemy) {
      fillPoly(ov.captureFill, ov.captureStroke, 2.0);
    } else if (isTurnPiece) {
      fillPoly(ov.legalTurnFill, ov.legalTurnStroke, 1.5);
    } else {
      fillPoly(ov.legalPreviewFill, ov.legalPreviewStroke, 1.5);
    }
  } else if (forbiddenMove) {
    if (forbiddenMove.forbidden_reason === "combat_failed" || forbiddenMove.forbidden_reason === "leap_cannot_capture" || forbiddenMove.forbidden_reason === "special_cannot_capture") {
      fillPoly(ov.forbiddenCombatFill, ov.forbiddenCombatStroke, 1.3);
    } else if (forbiddenMove.forbidden_reason === "friendly_occupied") {
      fillPoly(ov.forbiddenFriendlyFill, ov.forbiddenFriendlyStroke, 1.1);
    } else if (forbiddenMove.forbidden_reason === "enemy_path_blocked") {
      fillPoly(ov.forbiddenEnemyFill, ov.forbiddenEnemyStroke, 1.1);
    } else {
      fillPoly(ov.forbiddenPathFill, ov.forbiddenPathStroke, 1.1);
    }
  } else if (blockedMove) {
    if (isVanguardBonus) {
      fillPoly(ov.blockedVanguardFill, ov.blockedVanguardStroke, 1.8);
    } else {
      fillPoly(ov.blockedFill, ov.blockedStroke, 1.5);
    }
  }

  if (isHovered) {
    fillPoly(ov.hoverFill, ov.hoverStroke, 2.0);
  }

  const h = appState.terrain[r][c];
  ctx.fillStyle = theme.heightLabelColor;
  ctx.font = `600 9.5px ${theme.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const labelPos = {
    x: poly[0].x * 0.55 + topCenter.x * 0.45,
    y: poly[0].y * 0.55 + topCenter.y * 0.45,
  };
  ctx.fillText(`${h}`, labelPos.x, labelPos.y);

  let pieceOnSquare = null;
  let ownerPl = -1;
  for (let pl = 0; pl < 2; pl++) {
    for (const pc of appState.pieces[pl]) {
      if (pc.status === 0 && pc.r === r && pc.c === c) {
        pieceOnSquare = pc;
        ownerPl = pl;
        break;
      }
    }
  }

  if (analyticsActive && analyticsData && showAnalyticsValues) {
    const aVal = getAnalyticsValue(r, c, analyticsLayer);
    if (aVal !== null && aVal !== undefined) {
      const meta = ANALYTICS_LAYER_META[analyticsLayer];
      const formatted = meta.format(aVal);
      ctx.save();
      const fontSize = Math.max(9, Math.min(12.5, Math.round(radius * 0.42)));
      ctx.font = `700 ${fontSize}px ${theme.fontMono}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = ov.analyticsText;
      const yOff = (pieceOnSquare && showAnalyticsPieces) ? -radius * 0.72 : 0;
      ctx.strokeStyle = ov.analyticsOutline;
      ctx.lineWidth = 2.6;
      ctx.strokeText(formatted, topCenter.x, topCenter.y + yOff);
      ctx.fillText(formatted, topCenter.x, topCenter.y + yOff);
      ctx.restore();
    }
  }

  if (pieceOnSquare) {
    if (!analyticsActive || showAnalyticsPieces) {
      drawPieceToken(topCenter.x, topCenter.y, radius, ownerPl, pieceOnSquare, is3D);
      if (legalMove && isVanguardBonus) {
        ctx.fillStyle = ov.badgeAccent;
        ctx.font = `700 10px ${theme.fontMono}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("+1", topCenter.x + radius * 0.85, topCenter.y - radius * 0.85);
      } else if (forbiddenMove && forbiddenMove.forbidden_reason === "friendly_occupied") {
        ctx.fillStyle = theme.coordColor;
        ctx.font = `700 10px ${theme.fontSans}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("×", topCenter.x + radius * 0.82, topCenter.y - radius * 0.82);
      }
    }
  } else if (legalMove && !analyticsActive) {
    ctx.beginPath();
    const dotRadius = radius * (isVanguardBonus ? 0.34 : 0.28);
    ctx.arc(topCenter.x, topCenter.y, dotRadius, 0, Math.PI * 2);
    let dotCol = isTurnPiece ? ov.dotTurn : ov.dotPreview;
    if (isVanguardBonus) {
      dotCol = ov.dotVanguard;
    } else if (legalMove.special) {
      dotCol = ov.dotSpecial;
    }
    ctx.fillStyle = dotCol;
    ctx.fill();

    if (isVanguardBonus) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = `700 ${Math.max(8, Math.round(radius * 0.34))}px ${theme.fontMono}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("+1", topCenter.x, topCenter.y);
    }
  } else if (forbiddenMove && !analyticsActive) {
    ctx.beginPath();
    const dotRadius = radius * 0.21;
    ctx.arc(topCenter.x, topCenter.y, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = ov.dotForbiddenFill;
    ctx.fill();
    ctx.strokeStyle = ov.dotForbiddenStroke;
    ctx.lineWidth = 1.0;
    ctx.stroke();
  }

  if (!pieceOnSquare && blockedMove && !analyticsActive) {
    ctx.strokeStyle = ov.blockedStroke;
    ctx.lineWidth = 2.0;
    const s = radius * 0.28;
    ctx.beginPath();
    ctx.moveTo(topCenter.x - s, topCenter.y - s);
    ctx.lineTo(topCenter.x + s, topCenter.y + s);
    ctx.moveTo(topCenter.x + s, topCenter.y - s);
    ctx.lineTo(topCenter.x - s, topCenter.y + s);
    ctx.stroke();
  }
}

function drawPieceToken(cx, cy, radius, player, pc, is3D = true) {
  const isSelected = player === selectedPlayer && pc.piece_idx === selectedPieceIdx;
  if (is3D) {
    drawPieceShape3D(cx, cy, radius, player, pc, isSelected);
  } else {
    drawPieceShape2D(cx, cy, radius, player, pc, isSelected);
  }
}

function drawPieceShape3D(cx, cy, radius, player, pc, isSelected) {
  const theme = getCanvasTheme();
  const pt = theme.pieces;
  const pCol = player === 0 ? pt.p1 : pt.p2;
  const pType = pc.type;
  const cylH = Math.max(12, Math.round(radius * 1.15));

  const colBase = pCol.base;
  const colLit = pCol.lit;
  const colDark = pCol.dark;
  const colTop = pCol.top;
  const rimCol = isSelected ? pt.selectedRim : pCol.rim;

  ctx.save();

  if (isSelected) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius * 1.35, radius * 0.68, 0, 0, Math.PI * 2);
    ctx.fillStyle = pt.selectedHaloFill;
    ctx.fill();
    ctx.strokeStyle = pt.selectedRim;
    ctx.lineWidth = 2.4;
    ctx.stroke();
  }

  if (pType === 2) {
    const b0 = { x: cx, y: cy + radius * 0.42 };
    const b1 = { x: cx + radius * 0.85, y: cy - radius * 0.28 };
    const b2 = { x: cx - radius * 0.85, y: cy - radius * 0.28 };

    const ty = cy - cylH;
    const t0 = { x: cx, y: ty + radius * 0.20 };
    const t1 = { x: cx + radius * 0.38, y: ty - radius * 0.14 };
    const t2 = { x: cx - radius * 0.38, y: ty - radius * 0.14 };

    ctx.beginPath();
    ctx.moveTo(b2.x, b2.y);
    ctx.lineTo(b0.x, b0.y);
    ctx.lineTo(t0.x, t0.y);
    ctx.lineTo(t2.x, t2.y);
    ctx.closePath();
    ctx.fillStyle = colDark;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(b0.x, b0.y);
    ctx.lineTo(b1.x, b1.y);
    ctx.lineTo(t1.x, t1.y);
    ctx.lineTo(t0.x, t0.y);
    ctx.closePath();
    ctx.fillStyle = colLit;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(t0.x, t0.y);
    ctx.lineTo(t1.x, t1.y);
    ctx.lineTo(t2.x, t2.y);
    ctx.closePath();
    ctx.fillStyle = colTop;
    ctx.fill();
    ctx.strokeStyle = rimCol;
    ctx.lineWidth = isSelected ? 2.2 : 1.2;
    ctx.stroke();

    drawPieceLabel(cx, ty - radius * 0.05, pc.piece_idx, radius);
  } else if (pType === 1) {
    const bf = { x: cx, y: cy + radius * 0.44 };
    const br = { x: cx + radius * 0.85, y: cy };
    const bl = { x: cx - radius * 0.85, y: cy };

    const ty = cy - cylH;
    const tl = { x: cx - radius * 0.44, y: ty };
    const tr = { x: cx + radius * 0.44, y: ty };
    const tbl = { x: cx - radius * 0.38, y: ty - radius * 0.18 };
    const tbr = { x: cx + radius * 0.38, y: ty - radius * 0.18 };

    ctx.beginPath();
    ctx.moveTo(bl.x, bl.y);
    ctx.lineTo(bf.x, bf.y);
    ctx.lineTo(tl.x, tl.y);
    ctx.closePath();
    ctx.fillStyle = colDark;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(bf.x, bf.y);
    ctx.lineTo(br.x, br.y);
    ctx.lineTo(tr.x, tr.y);
    ctx.lineTo(tl.x, tl.y);
    ctx.closePath();
    ctx.fillStyle = colLit;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(tl.x, tl.y);
    ctx.lineTo(tr.x, tr.y);
    ctx.lineTo(tbr.x, tbr.y);
    ctx.lineTo(tbl.x, tbl.y);
    ctx.closePath();
    ctx.fillStyle = colTop;
    ctx.fill();
    ctx.strokeStyle = rimCol;
    ctx.lineWidth = isSelected ? 2.2 : 1.2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(tl.x, tl.y);
    ctx.lineTo(tr.x, tr.y);
    ctx.strokeStyle = pt.ridgeColor;
    ctx.lineWidth = 2.0;
    ctx.stroke();

    drawPieceLabel(cx, ty - radius * 0.05, pc.piece_idx, radius);
  } else {
    const ty = cy - cylH;
    const basePts = [];
    for (let i = 0; i < 5; i++) {
      const ang = (i * 2 * Math.PI) / 5 + Math.PI / 2;
      basePts.push({
        x: cx + radius * 0.82 * Math.cos(ang),
        y: cy + radius * 0.46 * Math.sin(ang),
      });
    }
    const topPts = [];
    for (let i = 0; i < 5; i++) {
      const ang = (i * 2 * Math.PI) / 5 + Math.PI / 2;
      topPts.push({
        x: cx + radius * 0.42 * Math.cos(ang),
        y: ty + radius * 0.24 * Math.sin(ang),
      });
    }

    ctx.beginPath();
    ctx.moveTo(basePts[2].x, basePts[2].y);
    ctx.lineTo(basePts[1].x, basePts[1].y);
    ctx.lineTo(topPts[1].x, topPts[1].y);
    ctx.lineTo(topPts[2].x, topPts[2].y);
    ctx.closePath();
    ctx.fillStyle = colDark;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 0.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(basePts[3].x, basePts[3].y);
    ctx.lineTo(basePts[2].x, basePts[2].y);
    ctx.lineTo(topPts[2].x, topPts[2].y);
    ctx.lineTo(topPts[3].x, topPts[3].y);
    ctx.closePath();
    ctx.fillStyle = colDark;
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(basePts[0].x, basePts[0].y);
    ctx.lineTo(basePts[4].x, basePts[4].y);
    ctx.lineTo(topPts[4].x, topPts[4].y);
    ctx.lineTo(topPts[0].x, topPts[0].y);
    ctx.closePath();
    ctx.fillStyle = colLit;
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(basePts[1].x, basePts[1].y);
    ctx.lineTo(basePts[0].x, basePts[0].y);
    ctx.lineTo(topPts[0].x, topPts[0].y);
    ctx.lineTo(topPts[1].x, topPts[1].y);
    ctx.closePath();
    ctx.fillStyle = colLit;
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(basePts[4].x, basePts[4].y);
    ctx.lineTo(basePts[3].x, basePts[3].y);
    ctx.lineTo(topPts[3].x, topPts[3].y);
    ctx.lineTo(topPts[4].x, topPts[4].y);
    ctx.closePath();
    ctx.fillStyle = colBase;
    ctx.fill();
    ctx.strokeStyle = theme.pillarStroke;
    ctx.lineWidth = 0.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(cx, ty, radius * 0.48, radius * 0.28, 0, 0, Math.PI * 2);
    ctx.fillStyle = colTop;
    ctx.fill();
    ctx.strokeStyle = rimCol;
    ctx.lineWidth = isSelected ? 2.2 : 1.2;
    ctx.stroke();

    drawPieceLabel(cx, ty, pc.piece_idx, radius);
  }

  ctx.restore();
}

function drawPieceShape2D(cx, cy, radius, player, pc, isSelected) {
  const theme = getCanvasTheme();
  const pt = theme.pieces;
  const pCol = player === 0 ? pt.p1 : pt.p2;
  const pType = pc.type;
  const baseColor = pCol.base;
  const rimColor = pCol.rim;

  ctx.save();

  if (isSelected) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.38, 0, Math.PI * 2);
    ctx.fillStyle = pt.selectedHaloFill;
    ctx.fill();
    ctx.strokeStyle = pt.selectedRim;
    ctx.lineWidth = 2.4;
    ctx.stroke();
  }

  if (pType === 2) {
    ctx.beginPath();
    for (let i = 0; i < 3; i++) {
      const ang = (i * 2 * Math.PI) / 3 - Math.PI / 2;
      const px = cx + radius * 1.05 * Math.cos(ang);
      const py = cy + radius * 1.05 * Math.sin(ang);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.strokeStyle = isSelected ? pt.selectedRim : rimColor;
    ctx.lineWidth = isSelected ? 2.4 : 1.8;
    ctx.stroke();

    for (let i = 0; i < 3; i++) {
      const ang = (i * 2 * Math.PI) / 3 - Math.PI / 2;
      const px = cx + radius * 0.78 * Math.cos(ang);
      const py = cy + radius * 0.78 * Math.sin(ang);
      ctx.beginPath();
      ctx.arc(px, py, 2.0, 0, Math.PI * 2);
      ctx.fillStyle = pCol.top;
      ctx.fill();
    }
  } else if (pType === 1) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - radius * 0.98);
    ctx.lineTo(cx + radius * 0.88, cy);
    ctx.lineTo(cx, cy + radius * 0.98);
    ctx.lineTo(cx - radius * 0.88, cy);
    ctx.closePath();
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.strokeStyle = isSelected ? pt.selectedRim : rimColor;
    ctx.lineWidth = isSelected ? 2.4 : 1.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx - radius * 0.45, cy);
    ctx.lineTo(cx + radius * 0.45, cy);
    ctx.strokeStyle = pCol.top;
    ctx.lineWidth = 2.0;
    ctx.stroke();
  } else {
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const ang = (i * 2 * Math.PI) / 5 - Math.PI / 2;
      const px = cx + radius * 0.95 * Math.cos(ang);
      const py = cy + radius * 0.95 * Math.sin(ang);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.strokeStyle = isSelected ? pt.selectedRim : rimColor;
    ctx.lineWidth = isSelected ? 2.4 : 1.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.28, 0, Math.PI * 2);
    ctx.fillStyle = pCol.top;
    ctx.fill();
  }

  ctx.fillStyle = pt.labelFill;
  ctx.font = `800 ${Math.max(10, Math.round(radius * 0.65))}px ${theme.fontSans}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(PIECE_SHORT[pc.piece_idx], cx, cy);

  ctx.restore();
}

function drawPieceLabel(cx, cy, pieceIdx, radius) {
  const theme = getCanvasTheme();
  const pt = theme.pieces;
  ctx.save();
  ctx.fillStyle = pt.labelFill;
  ctx.strokeStyle = pt.labelStroke;
  ctx.lineWidth = 2.4;
  ctx.font = `800 ${Math.max(10, Math.round(radius * 0.65))}px ${theme.fontSans}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.strokeText(PIECE_SHORT[pieceIdx], cx, cy);
  ctx.fillText(PIECE_SHORT[pieceIdx], cx, cy);
  ctx.restore();
}

function drawCandidateArrows(candidates, isHighlight = false) {
  const arrTheme = getCanvasTheme().arrows;
  const topList = isHighlight ? candidates : candidates.slice(0, 4);
  for (let i = topList.length - 1; i >= 0; i--) {
    const cand = topList[i];
    if (!isHighlight && cand.policy_share < 0.04 && i > 0) continue;

    const fromPt = getCellCenter(cand.from_r, cand.from_c);
    let toPt = null;
    if (cand.is_finish) {
      const fz = finishZones.find((z) => z.player === appState.analysis.perspective_player);
      if (fz) toPt = fz.center;
    } else {
      toPt = getCellCenter(cand.to_r, cand.to_c);
    }
    if (!fromPt || !toPt) continue;

    const weight = isHighlight ? 1.0 : Math.max(0.25, cand.policy_share);
    const lineW = isHighlight ? 4.2 : 2.0 + 4.0 * weight;
    const color = isHighlight
      ? arrTheme.highlight
      : i === 0
      ? arrTheme.primary
      : arrTheme.secondary;

    drawArrow(fromPt.x, fromPt.y - 6, toPt.x, toPt.y - 6, lineW, color);
  }
}

function drawArrow(x1, y1, x2, y2, width, color) {
  const headLen = Math.max(9, width * 2.5);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2 - headLen * 0.6 * Math.cos(angle), y2 - headLen * 0.6 * Math.sin(angle));
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLen * Math.cos(angle - Math.PI / 6),
    y2 - headLen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 6),
    y2 - headLen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawFinishPathOverlay(finishPath, player) {
  if (!finishPath || !finishPath.path || finishPath.path.length < 2) return;
  const theme = getCanvasTheme();
  const fpTheme = theme.finishPath;
  const path = finishPath.path;
  const isP1 = player === 0;
  const mainCol = isP1 ? fpTheme.p1Line : fpTheme.p2Line;

  ctx.save();

  ctx.beginPath();
  let first = true;
  for (const pt of path) {
    const center = getCellCenter(pt[0], pt[1]);
    if (!center) continue;
    if (first) {
      ctx.moveTo(center.x, center.y);
      first = false;
    } else {
      ctx.lineTo(center.x, center.y);
    }
  }
  ctx.strokeStyle = mainCol;
  ctx.lineWidth = 3.0;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash([7, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  for (let i = 1; i < path.length; i++) {
    const pt = path[i];
    const center = getCellCenter(pt[0], pt[1]);
    if (!center) continue;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = mainCol;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  if (finishPath.turn_segments) {
    for (const seg of finishPath.turn_segments) {
      const toPt = seg.to;
      const center = getCellCenter(toPt[0], toPt[1]);
      if (!center) continue;

      const isFinish = seg.action === "finish";
      const badgeY = center.y - 14;

      ctx.save();
      ctx.beginPath();
      ctx.arc(center.x, badgeY, 11, 0, Math.PI * 2);
      ctx.fillStyle = isFinish ? fpTheme.finishBg : fpTheme.badgeBg;
      ctx.fill();
      ctx.strokeStyle = isFinish ? fpTheme.finishStroke : mainCol;
      ctx.lineWidth = 1.8;
      ctx.stroke();

      ctx.fillStyle = isFinish ? fpTheme.finishText : fpTheme.badgeText;
      ctx.font = `700 9.5px ${theme.fontMono}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(isFinish ? "FIN" : `T${seg.turn}`, center.x, badgeY);

      if (!isFinish) {
        ctx.fillStyle = fpTheme.costBg;
        ctx.beginPath();
        ctx.roundRect(center.x - 14, badgeY + 12, 28, 11, 3);
        ctx.fill();
        ctx.strokeStyle = fpTheme.costStroke;
        ctx.lineWidth = 0.8;
        ctx.stroke();

        ctx.fillStyle = mainCol;
        ctx.font = `700 8px ${theme.fontMono}`;
        ctx.fillText(`c=${seg.cost}`, center.x, badgeY + 17.5);
      }
      ctx.restore();
    }
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Canvas Mouse Interaction & Zoom/Pan Sliding
// ---------------------------------------------------------------------------
function hitTestCanvas(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const screenX = clientX - rect.left;
  const screenY = clientY - rect.top;

  const W = rect.width;
  const H = rect.height;
  const x = (screenX - (W / 2 + panOffsetX)) / zoomLevel + W / 2;
  const y = (screenY - (H / 2 + panOffsetY)) / zoomLevel + H / 2;

  for (const fz of finishZones) {
    if (pointInPoly(x, y, fz.poly)) {
      return { finishZone: fz };
    }
  }

  for (let i = cellPolygons.length - 1; i >= 0; i--) {
    const cell = cellPolygons[i];
    if (pointInPoly(x, y, cell.poly)) {
      return { r: cell.r, c: cell.c };
    }
  }
  return null;
}

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const focusX = e.clientX - rect.left;
  const focusY = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.12 : 0.893;
  setZoom(zoomLevel * factor, focusX, focusY);
}, { passive: false });

canvas.addEventListener("mousedown", (e) => {
  const isPanButton = (e.ctrlKey && e.button === 0) || e.button === 1 || (e.altKey && e.button === 0);
  if (isPanButton) {
    isPanning = true;
    hasDraggedPan = false;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panInitialOffsetX = panOffsetX;
    panInitialOffsetY = panOffsetY;
    canvas.style.cursor = "grabbing";
    e.preventDefault();
  }
});

window.addEventListener("mousemove", (e) => {
  if (isPanning) {
    const dx = e.clientX - panStartX;
    const dy = e.clientY - panStartY;
    if (Math.hypot(dx, dy) > 2) {
      hasDraggedPan = true;
    }
    panOffsetX = panInitialOffsetX + dx;
    panOffsetY = panInitialOffsetY + dy;
    drawBoard();
  }
});

window.addEventListener("mouseup", (e) => {
  if (isPanning) {
    isPanning = false;
    canvas.style.cursor = e.ctrlKey ? "grab" : "pointer";
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Control" && !isPanning) {
    canvas.style.cursor = "grab";
  }
});

window.addEventListener("keyup", (e) => {
  if (e.key === "Control" && !isPanning) {
    canvas.style.cursor = "pointer";
  }
});

canvas.addEventListener("dblclick", () => resetZoomPan());

canvas.addEventListener("mousemove", (e) => {
  if (isPanning) return;
  canvas.style.cursor = e.ctrlKey ? "grab" : "pointer";

  const hit = hitTestCanvas(e.clientX, e.clientY);
  const prevKey = hoveredCell ? `${hoveredCell.r},${hoveredCell.c}` : "";
  const nextKey = hit && hit.r !== undefined ? `${hit.r},${hit.c}` : "";
  hoveredCell = hit;
  if (prevKey !== nextKey) {
    renderHoverInspector();
    drawBoard();
  }
});

canvas.addEventListener("mouseleave", () => {
  if (!isPanning) {
    hoveredCell = null;
    renderHoverInspector();
    drawBoard();
  }
});

canvas.addEventListener("contextmenu", (e) => {
  if (!appState) return;
  const hit = hitTestCanvas(e.clientX, e.clientY);
  if (hit && hit.r !== undefined) {
    for (let pl = 0; pl < 2; pl++) {
      const pc = appState.pieces[pl].find(
        (item) => item.status === 0 && item.r === hit.r && item.c === hit.c
      );
      if (pc) {
        e.preventDefault();
        selectedPlayer = pl;
        selectedPieceIdx = pc.piece_idx;
        renderAll();
        return;
      }
    }
  }
});

canvas.addEventListener("click", (e) => {
  if (!appState || isBusy) return;
  if (hasDraggedPan || e.ctrlKey) {
    hasDraggedPan = false;
    return;
  }

  const hit = hitTestCanvas(e.clientX, e.clientY);
  if (!hit) return;

  if (!appState.done && hit.finishZone && hit.finishZone.action !== null) {
    executeHumanAction(hit.finishZone.action);
    return;
  }

  if (hit.r === undefined) return;
  const { r, c } = hit;

  const curSelectedInfo = getSelectedPieceInfo();
  const isTurnSelection =
    !appState.done &&
    curSelectedInfo &&
    selectedPlayer === appState.current_player;

  let clickedPiece = null;
  let clickedOwner = null;
  for (let pl = 0; pl < 2; pl++) {
    const found = appState.pieces[pl].find(
      (pc) => pc.status === 0 && pc.r === r && pc.c === c
    );
    if (found) {
      clickedPiece = found;
      clickedOwner = pl;
      break;
    }
  }

  if (isTurnSelection && !e.shiftKey) {
    const legalMove = curSelectedInfo.legal_moves.find(
      (m) => m.r === r && m.c === c
    );
    if (legalMove && legalMove.action !== null) {
      executeHumanAction(legalMove.action);
      return;
    }
  }

  if (clickedPiece !== null) {
    if (
      selectedPlayer === clickedOwner &&
      selectedPieceIdx === clickedPiece.piece_idx
    ) {
      selectedPlayer = null;
      selectedPieceIdx = null;
    } else {
      selectedPlayer = clickedOwner;
      selectedPieceIdx = clickedPiece.piece_idx;
    }
    renderAll();
    return;
  }

  if (curSelectedInfo && curSelectedInfo.forbidden_moves) {
    const forbiddenMove = curSelectedInfo.forbidden_moves.find((m) => m.r === r && m.c === c);
    if (forbiddenMove) {
      hoveredCell = { r, c };
      renderHoverInspector();
      drawBoard();
      return;
    }
  }

  selectedPlayer = null;
  selectedPieceIdx = null;
  renderAll();
});

// ---------------------------------------------------------------------------
// Button & Toolbar Event Listeners
// ---------------------------------------------------------------------------
document.getElementById("zoomInBtn").addEventListener("click", () => zoomIn());
document.getElementById("zoomOutBtn").addEventListener("click", () => zoomOut());
document.getElementById("zoomResetBtn").addEventListener("click", () => resetZoomPan());

document.getElementById("newGameBtn").addEventListener("click", async () => {
  toggleAutoPlay(false);
  setBusy(true);
  try {
    appState = await apiPost("/api/new_game", {
      terrain_mode: document.getElementById("terrainSelect").value,
      p1_agent: document.getElementById("p1AgentSelect").value,
      p2_agent: document.getElementById("p2AgentSelect").value,
      p1_sims: parseInt(document.getElementById("p1SimsInput").value, 10) || 96,
      p2_sims: parseInt(document.getElementById("p2SimsInput").value, 10) || 96,
    });
    analyticsData = null;
    autoSelectDefaultPiece();
  } finally {
    setBusy(false);
    renderAll();
    scheduleAiIfNeeded();
  }
});

document.getElementById("swapSidesBtn").addEventListener("click", async () => {
  toggleAutoPlay(false);
  const p1A = document.getElementById("p1AgentSelect").value;
  const p2A = document.getElementById("p2AgentSelect").value;
  const p1S = parseInt(document.getElementById("p1SimsInput").value, 10) || 96;
  const p2S = parseInt(document.getElementById("p2SimsInput").value, 10) || 96;

  document.getElementById("p1AgentSelect").value = p2A;
  document.getElementById("p2AgentSelect").value = p1A;
  document.getElementById("p1SimsInput").value = p2S;
  document.getElementById("p2SimsInput").value = p1S;

  if (p2A === "human" && p1A !== "human") setViewpoint(0);
  else if (p1A === "human" && p2A !== "human") setViewpoint(4);

  document.getElementById("newGameBtn").click();
});

async function pushSettingsChange() {
  if (!appState) return;
  appState = await apiPost("/api/settings", {
    p1_agent: document.getElementById("p1AgentSelect").value,
    p2_agent: document.getElementById("p2AgentSelect").value,
    p1_sims: parseInt(document.getElementById("p1SimsInput").value, 10) || 96,
    p2_sims: parseInt(document.getElementById("p2SimsInput").value, 10) || 96,
  });
  renderAll();
  scheduleAiIfNeeded();
}

async function onP1AgentChange() {
  if (!appState) return;
  const selVal = document.getElementById("p1AgentSelect").value;
  const ag = appState.available_agents.find((a) => a.id === selVal);
  if (ag && ag.default_sims) {
    document.getElementById("p1SimsInput").value = ag.default_sims;
  }
  await pushSettingsChange();
}

async function onP2AgentChange() {
  if (!appState) return;
  const selVal = document.getElementById("p2AgentSelect").value;
  const ag = appState.available_agents.find((a) => a.id === selVal);
  if (ag && ag.default_sims) {
    document.getElementById("p2SimsInput").value = ag.default_sims;
  }
  await pushSettingsChange();
}

document.getElementById("p1AgentSelect").addEventListener("change", onP1AgentChange);
document.getElementById("p2AgentSelect").addEventListener("change", onP2AgentChange);
document.getElementById("p1SimsInput").addEventListener("change", pushSettingsChange);
document.getElementById("p2SimsInput").addEventListener("change", pushSettingsChange);

function setProjectionMode(mode) {
  viewMode = mode === "2d" ? "2d" : "3d";
  document.getElementById("view3dBtn").classList.toggle("active-toggle", viewMode === "3d");
  document.getElementById("view2dBtn").classList.toggle("active-toggle", viewMode === "2d");
  document.getElementById("elevationControlBar").style.display = viewMode === "3d" ? "inline-flex" : "none";
  drawBoard();
}

document.getElementById("view3dBtn").addEventListener("click", () => setProjectionMode("3d"));
document.getElementById("view2dBtn").addEventListener("click", () => setProjectionMode("2d"));

document.getElementById("viewpointSelect").addEventListener("change", (e) => {
  setViewpoint(parseInt(e.target.value, 10) || 0);
});
document.getElementById("rotCcwBtn").addEventListener("click", () => rotateView(-1));
document.getElementById("rotCwBtn").addEventListener("click", () => rotateView(1));
document.getElementById("flipBoardBtn").addEventListener("click", () => rotateView(4));

document.querySelectorAll(".compass-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const vp = parseInt(btn.dataset.vp, 10);
    setViewpoint(vp);
  });
});

let isScrubbing = false;

async function jumpToStart() {
  if (isBusy || isScrubbing || !appState || appState.cursor <= 0) return;
  toggleAutoPlay(false);
  isScrubbing = true;
  try {
    appState = await apiPost("/api/jump", { cursor: 0 });
    autoSelectDefaultPiece();
    renderAll();
  } catch (err) {
    console.error(err);
  } finally {
    isScrubbing = false;
  }
}

async function jumpToEnd() {
  if (isBusy || isScrubbing || !appState || appState.cursor >= appState.max_cursor) return;
  toggleAutoPlay(false);
  isScrubbing = true;
  try {
    appState = await apiPost("/api/jump", { cursor: appState.max_cursor });
    autoSelectDefaultPiece();
    renderAll();
    scheduleAiIfNeeded();
  } catch (err) {
    console.error(err);
  } finally {
    isScrubbing = false;
  }
}

async function stepTimelineBackward() {
  if (isBusy || isScrubbing || !appState || appState.cursor <= 0) return;
  toggleAutoPlay(false);
  isScrubbing = true;
  try {
    appState = await apiPost("/api/jump", { cursor: appState.cursor - 1 });
    autoSelectDefaultPiece();
    renderAll();
  } catch (err) {
    console.error(err);
  } finally {
    isScrubbing = false;
  }
}

async function stepTimelineForward() {
  if (isBusy || isScrubbing || !appState) return;
  toggleAutoPlay(false);
  if (appState.cursor < appState.max_cursor) {
    isScrubbing = true;
    try {
      appState = await apiPost("/api/jump", { cursor: appState.cursor + 1 });
      playChessMoveSound(appState.last_move);
      autoSelectDefaultPiece();
      renderAll();
      if (appState.cursor === appState.max_cursor) {
        scheduleAiIfNeeded();
      }
    } catch (err) {
      console.error(err);
    } finally {
      isScrubbing = false;
    }
  } else if (!appState.done && !isCurrentPlayerHuman() && !hasHumanPlayer() && !(appState.simulation_mode && appState.simulation_mode.active)) {
    triggerAiMove();
  }
}

async function runAnalyzeHint() {
  if (isBusy || !appState || appState.done) return;
  try {
    setBusy(true);
    appState = await apiPost("/api/analyze", {});
  } finally {
    setBusy(false);
    renderAll();
  }
}

function toggleSoundMute() {
  soundEnabled = !soundEnabled;
  localStorage.setItem("traversal_sound_enabled", soundEnabled ? "1" : "0");
  updateSoundButtonState();
  if (soundEnabled) {
    playChessMoveSound({ captured: false });
  }
}

function toggleFinishPathOverlay() {
  if (selectedPieceIdx === null) {
    const curPieces = appState ? appState.pieces[appState.current_player] : null;
    const firstAct = curPieces ? curPieces.find((p) => p.status === 0) : null;
    if (firstAct) {
      selectedPlayer = appState.current_player;
      selectedPieceIdx = firstAct.piece_idx;
    }
  }
  showPieceFinishPath = !showPieceFinishPath;
  renderSelectedPiecePanel();
  drawBoard();
}

// ---------------------------------------------------------------------------
// Centralized Global Keyboard Shortcut Dispatcher
// ---------------------------------------------------------------------------
function handleGlobalKeydown(e) {
  const target = e.target;
  const inInput =
    target &&
    (target.tagName === "INPUT" ||
      target.tagName === "SELECT" ||
      target.tagName === "TEXTAREA" ||
      target.isContentEditable);

  if (e.key === "Escape") {
    const scModal = document.getElementById("shortcutsModal");
    if (scModal && scModal.classList.contains("open")) {
      closeShortcutsModal();
      return;
    }
    const impModal = document.getElementById("importModal");
    if (impModal && impModal.classList.contains("open")) {
      impModal.classList.remove("open");
      return;
    }
    const eloModal = document.getElementById("eloModal");
    if (eloModal && eloModal.classList.contains("open")) {
      closeEloModal();
      return;
    }
    if (analyticsActive) {
      setAnalyticsMode(false);
    }
    return;
  }

  if (inInput || e.ctrlKey || e.metaKey || e.altKey) {
    return;
  }

  if (e.key === "?") {
    e.preventDefault();
    toggleShortcutsModal();
    return;
  }

  if (isAnyModalOpen()) {
    return;
  }

  const key = e.key;
  const lower = key.toLowerCase();

  if (key === "ArrowLeft") {
    e.preventDefault();
    stepTimelineBackward();
  } else if (key === "ArrowRight") {
    e.preventDefault();
    stepTimelineForward();
  } else if (key === "Home") {
    e.preventDefault();
    jumpToStart();
  } else if (key === "End") {
    e.preventDefault();
    jumpToEnd();
  } else if (lower === "q" || key === "[") {
    rotateView(-1);
  } else if (lower === "e" || key === "]") {
    rotateView(1);
  } else if (lower === "f") {
    rotateView(4);
  } else if (lower === "v") {
    setProjectionMode(viewMode === "3d" ? "2d" : "3d");
  } else if (lower === "r") {
    resetZoomPan();
  } else if (lower === "p") {
    e.preventDefault();
    toggleFinishPathOverlay();
  } else if (lower === "s") {
    e.preventDefault();
    toggleSimulationMode();
  } else if (lower === "a") {
    setAnalyticsMode(!analyticsActive);
  } else if (lower === "g") {
    setAnalyticsMode(false);
  } else if (lower === "h") {
    runAnalyzeHint();
  } else if (lower === "m") {
    toggleSoundMute();
  } else if (lower === "k") {
    e.preventDefault();
    setShortcutHintsVisible(!showShortcutHints);
  }
}

window.addEventListener("keydown", handleGlobalKeydown);

// ---------------------------------------------------------------------------
// Simulation Mode Actions
// ---------------------------------------------------------------------------
async function toggleSimulationMode() {
  if (!appState || isBusy) return;
  const sim = appState.simulation_mode;
  if (sim && sim.active) {
    if (confirm("Exit Simulation Mode and discard simulated moves? Click OK to discard, or Cancel to stay in simulation.")) {
      await exitSimulation(false);
    }
  } else {
    await startSimulation();
  }
}

async function startSimulation() {
  if (isBusy) return;
  try {
    setBusy(true);
    if (autoPlayTimer) {
      clearTimeout(autoPlayTimer);
      autoPlayTimer = null;
    }
    toggleAutoPlay(false);
    appState = await apiPost("/api/simulation/start");
    autoSelectDefaultPiece();
    renderAll();
  } catch (err) {
    alert("Failed to start simulation: " + err.message);
  } finally {
    setBusy(false);
  }
}

async function resetSimulation() {
  if (isBusy) return;
  try {
    setBusy(true);
    appState = await apiPost("/api/simulation/reset");
    autoSelectDefaultPiece();
    renderAll();
  } catch (err) {
    alert("Failed to reset simulation: " + err.message);
  } finally {
    setBusy(false);
  }
}

async function exitSimulation(commit = false) {
  if (isBusy) return;
  try {
    setBusy(true);
    appState = await apiPost("/api/simulation/exit", { commit });
    autoSelectDefaultPiece();
    renderAll();
  } catch (err) {
    alert("Failed to exit simulation: " + err.message);
  } finally {
    setBusy(false);
    scheduleAiIfNeeded();
  }
}

const simToggleBtn = document.getElementById("toggleSimulationBtn");
if (simToggleBtn) simToggleBtn.addEventListener("click", () => toggleSimulationMode());

const simStepAi = document.getElementById("simStepAiBtn");
if (simStepAi) simStepAi.addEventListener("click", () => triggerAiMove());

const simUndo = document.getElementById("simUndoBtn");
if (simUndo) simUndo.addEventListener("click", () => stepTimelineBackward());

const simReset = document.getElementById("simResetBtn");
if (simReset) simReset.addEventListener("click", () => resetSimulation());

const simCommit = document.getElementById("simCommitBtn");
if (simCommit) simCommit.addEventListener("click", () => exitSimulation(true));

const simExit = document.getElementById("simExitBtn");
if (simExit) simExit.addEventListener("click", () => exitSimulation(false));

document.getElementById("toggleArrowsBtn").addEventListener("click", () => {
  showAiArrows = !showAiArrows;
  document.getElementById("toggleArrowsBtn").classList.toggle("active-toggle", showAiArrows);
  drawBoard();
});

const soundBtn = document.getElementById("toggleSoundBtn");
if (soundBtn) {
  soundBtn.addEventListener("click", toggleSoundMute);
  updateSoundButtonState();
}

document.getElementById("elevScaleSlider").addEventListener("input", (e) => {
  elevScale = parseInt(e.target.value, 10) || 12;
  drawBoard();
});

document.getElementById("analyzeBtn").addEventListener("click", runAnalyzeHint);
document.getElementById("stepAiBtn").addEventListener("click", () => triggerAiMove());
document.getElementById("jumpStartBtn").addEventListener("click", jumpToStart);
document.getElementById("stepBackBtn").addEventListener("click", stepTimelineBackward);
document.getElementById("stepFwdBtn").addEventListener("click", stepTimelineForward);
document.getElementById("jumpEndBtn").addEventListener("click", jumpToEnd);
document.getElementById("autoPlayBtn").addEventListener("click", () => toggleAutoPlay());

window.addEventListener("resize", () => {
  drawBoard();
});

// Shortcuts Controls
const toggleHintsBtn = document.getElementById("toggleShortcutsHintsBtn");
if (toggleHintsBtn) {
  toggleHintsBtn.addEventListener("click", () => setShortcutHintsVisible(!showShortcutHints));
}

const openShortcutsBtn = document.getElementById("openShortcutsModalBtn");
if (openShortcutsBtn) {
  openShortcutsBtn.addEventListener("click", openShortcutsModal);
}

const closeShortcutsBtn = document.getElementById("closeShortcutsModalBtn");
if (closeShortcutsBtn) {
  closeShortcutsBtn.addEventListener("click", closeShortcutsModal);
}

const shortcutsModalEl = document.getElementById("shortcutsModal");
if (shortcutsModalEl) {
  shortcutsModalEl.addEventListener("click", (e) => {
    if (e.target === shortcutsModalEl) closeShortcutsModal();
  });
}

const modalHintsToggle = document.getElementById("modalShortcutsHintsToggle");
if (modalHintsToggle) {
  modalHintsToggle.addEventListener("change", (e) => {
    setShortcutHintsVisible(e.target.checked);
  });
}

const canvasViewportEl = document.getElementById("canvasViewport");
if (canvasViewportEl && typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => {
    drawBoard();
  }).observe(canvasViewportEl);
}

// ---------------------------------------------------------------------------
// Elo Arena & Leaderboard System Logic
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------------------------------------------------------------------
// Online Game Import Modal (traversalgame.com)
// ---------------------------------------------------------------------------
function initImportSystem() {
  const openBtn = document.getElementById("openImportModalBtn");
  const closeBtn = document.getElementById("closeImportModalBtn");
  const modal = document.getElementById("importModal");
  const urlInput = document.getElementById("importGameUrlInput");
  const cookieInput = document.getElementById("importCookieInput");
  const rawJsonInput = document.getElementById("importRawJsonInput");
  const fullGameBtn = document.getElementById("importFullGameBtn");
  const terrainOnlyBtn = document.getElementById("importTerrainOnlyBtn");
  const statusMsg = document.getElementById("importStatusMsg");
  const advDetails = document.getElementById("importAdvancedDetails");
  const curlBadge = document.getElementById("importCurlBadge");
  const cookieWarning = document.getElementById("importCookieWarning");

  const savedUrl = localStorage.getItem("traversal_last_import_url");
  if (savedUrl && urlInput) urlInput.value = savedUrl;
  const savedCookie = localStorage.getItem("traversal_import_cookie");
  if (savedCookie && cookieInput) cookieInput.value = savedCookie;

  function checkUrlInput() {
    const val = (urlInput ? urlInput.value : "").trim();
    if (curlBadge) {
      curlBadge.style.display = (val.startsWith("curl") || val.includes("curl 'http")) ? "inline-block" : "none";
    }
  }

  function checkCookieInput() {
    if (!cookieWarning || !cookieInput) return;
    const val = cookieInput.value.trim();
    if (val.length > 20 && val.length < 150 && !val.includes(".") && !val.includes("=")) {
      cookieWarning.style.display = "block";
      cookieWarning.innerHTML = "<strong>Notice:</strong> This token appears to be only a signature (~" + val.length + " chars). Full <code>traversal_session</code> tokens start with <code>eyJ...</code> (~320 chars). In <em>Storage &rarr; Cookies</em>, click <code>traversal_session</code> and copy the full value.";
    } else {
      cookieWarning.style.display = "none";
    }
  }

  if (urlInput) {
    urlInput.addEventListener("input", checkUrlInput);
    urlInput.addEventListener("paste", () => setTimeout(checkUrlInput, 20));
  }
  if (cookieInput) {
    cookieInput.addEventListener("input", checkCookieInput);
    cookieInput.addEventListener("paste", () => setTimeout(checkCookieInput, 20));
  }

  checkUrlInput();
  checkCookieInput();

  function openModal() {
    if (!modal) return;
    modal.classList.add("open");
    if (statusMsg) statusMsg.style.display = "none";
    checkUrlInput();
    checkCookieInput();
    setTimeout(() => urlInput && urlInput.focus(), 50);
  }

  function closeModal() {
    if (!modal) return;
    modal.classList.remove("open");
  }

  if (openBtn) openBtn.addEventListener("click", openModal);
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }

  async function runImport(mode) {
    const url = (urlInput.value || "").trim();
    const cookie = (cookieInput.value || "").trim();
    const rawJson = (rawJsonInput.value || "").trim();

    if (!url && !rawJson) {
      showStatus("Please enter a game URL, UUID, or paste game JSON.", "error");
      return;
    }

    if (url) localStorage.setItem("traversal_last_import_url", url);
    if (cookie) localStorage.setItem("traversal_import_cookie", cookie);

    showStatus("Fetching and importing online game...", "info");
    fullGameBtn.disabled = true;
    terrainOnlyBtn.disabled = true;

    try {
      const payload = {
        game_url_or_id: url,
        mode: mode,
        cookie: cookie || null,
        raw_json: rawJson || null,
      };
      appState = await apiPost("/api/import_online_game", payload);
      toggleAutoPlay(false);
      autoSelectDefaultPiece();
      renderAll();

      const info = appState.import_info || {};
      const msg = info.message || `Successfully imported ${mode === "full" ? "full match" : "terrain"}.`;
      showStatus(msg, "success");

      setTimeout(() => {
        closeModal();
      }, 900);
    } catch (err) {
      const errMsg = err.message || "Import failed";
      showStatus(errMsg, "error");
      if (errMsg.includes("requires login") || errMsg.includes("Authentication")) {
        if (advDetails) advDetails.open = true;
      }
    } finally {
      fullGameBtn.disabled = false;
      terrainOnlyBtn.disabled = false;
    }
  }

  function showStatus(text, type) {
    if (!statusMsg) return;
    statusMsg.style.display = "block";
    statusMsg.className = "status-msg " + (
      type === "error" ? "status-msg-error" :
      type === "success" ? "status-msg-success" : "status-msg-info"
    );
    statusMsg.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
  }

  if (fullGameBtn) fullGameBtn.addEventListener("click", () => runImport("full"));
  if (terrainOnlyBtn) terrainOnlyBtn.addEventListener("click", () => runImport("terrain_only"));
}

let tournamentPollTimer = null;

function initEloSystem() {
  const openBtn = document.getElementById("openLeaderboardBtn");
  const closeBtn = document.getElementById("closeEloModalBtn");
  const modal = document.getElementById("eloModal");

  if (openBtn) openBtn.addEventListener("click", openEloModal);
  if (closeBtn) closeBtn.addEventListener("click", closeEloModal);
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeEloModal();
    });
  }

  document.querySelectorAll(".modal-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".modal-tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => (p.style.display = "none"));
      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      const panel = document.getElementById(tabId);
      if (panel) panel.style.display = "block";

      if (tabId === "tabLeaderboard") loadLeaderboard();
      else if (tabId === "tabMatches") loadMatchHistory();
      else if (tabId === "tabTournament") populateTournamentCheckpoints();
    });
  });

  const saveBtn = document.getElementById("savePlayerNameBtn");
  if (saveBtn) saveBtn.addEventListener("click", saveHumanPlayerName);

  const nameInput = document.getElementById("humanPlayerNameInput");
  if (nameInput) {
    nameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") saveHumanPlayerName();
    });
  }

  const refreshBtn = document.getElementById("refreshLeaderboardBtn");
  if (refreshBtn) refreshBtn.addEventListener("click", loadLeaderboard);

  const resetBtn = document.getElementById("resetRatingsBtn");
  if (resetBtn) resetBtn.addEventListener("click", resetAllRatings);

  const startTournBtn = document.getElementById("startTournamentBtn");
  if (startTournBtn) startTournBtn.addEventListener("click", startTournament);

  const selectAllBtn = document.getElementById("tournSelectAllBtn");
  if (selectAllBtn) {
    selectAllBtn.addEventListener("click", () => {
      const targetId = document.getElementById("tournTargetSelect")?.value || "";
      document.querySelectorAll("#tournCheckpointsList input[type='checkbox']").forEach((cb) => {
        if (cb.value !== targetId) cb.checked = true;
      });
      updateTournamentEstimate();
    });
  }
  const selectNoneBtn = document.getElementById("tournSelectNoneBtn");
  if (selectNoneBtn) {
    selectNoneBtn.addEventListener("click", () => {
      document.querySelectorAll("#tournCheckpointsList input[type='checkbox']").forEach((cb) => (cb.checked = false));
      updateTournamentEstimate();
    });
  }
  const pairsSelect = document.getElementById("tournPairsSelect");
  if (pairsSelect) {
    pairsSelect.addEventListener("change", () => {
      const customInp = document.getElementById("tournCustomPairsInput");
      if (customInp) {
        customInp.style.display = pairsSelect.value === "custom" ? "inline-block" : "none";
      }
      updateTournamentEstimate();
    });
  }
  const customPairsInp = document.getElementById("tournCustomPairsInput");
  if (customPairsInp) {
    customPairsInp.addEventListener("input", updateTournamentEstimate);
  }
  const targetSelect = document.getElementById("tournTargetSelect");
  if (targetSelect) {
    targetSelect.addEventListener("change", onTournamentTargetChange);
  }
}

function openEloModal() {
  const modal = document.getElementById("eloModal");
  if (!modal) return;
  modal.classList.add("open");
  const nameInput = document.getElementById("humanPlayerNameInput");
  if (nameInput && appState && appState.elo) {
    nameInput.value = appState.elo.human_player_name || "Guillaume";
  }
  loadLeaderboard();
  populateTournamentCheckpoints();
  checkTournamentStatus();
}

function closeEloModal() {
  const modal = document.getElementById("eloModal");
  if (modal) modal.classList.remove("open");
}

async function saveHumanPlayerName() {
  const nameInput = document.getElementById("humanPlayerNameInput");
  if (!nameInput) return;
  const newName = nameInput.value.trim() || "Guillaume";
  nameInput.value = newName;
  localStorage.setItem("traversal_player_name", newName);

  try {
    appState = await apiPost("/api/elo/player_name", { name: newName });
    populateAgentDropdowns();
    syncControlsFromState();
    renderAll();
    loadLeaderboard();
  } catch (err) {
    alert("Failed to update player name: " + err.message);
  }
}

async function loadLeaderboard() {
  const tbody = document.getElementById("eloLeaderboardBody");
  if (!tbody) return;

  try {
    const res = await fetch("/api/elo/leaderboard");
    const data = await res.json();
    const lb = data.leaderboard || [];

    tbody.innerHTML = "";
    if (lb.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="muted-hint" style="text-align:center; padding:20px;">No players found.</td></tr>';
      return;
    }

    const curUserName = (appState && appState.elo && appState.elo.human_player_name) || "Guillaume";
    let userFound = null;

    for (const p of lb) {
      const isUser = p.id === curUserName || (p.kind === "human" && p.name === curUserName);
      if (isUser) userFound = p;

      const rankBadge = `#${p.rank}`;
      const rankClass = p.rank <= 3 ? "rank-top" : "";

      let kindBadge = "Human";
      if (p.kind === "alphazero") kindBadge = "AlphaZero";
      else if (p.kind === "heuristic") kindBadge = "Heuristic";

      let eloClass = "elo-pill-mid";
      if (p.rating >= 1650) eloClass = "elo-pill-high";
      else if (p.rating >= 1550) eloClass = "elo-pill-gold";
      else if (p.rating < 1450) eloClass = "elo-pill-low";

      const winPct = p.games_played > 0 ? `${(p.win_rate * 100).toFixed(1)}%` : "-";

      const tr = document.createElement("tr");
      if (isUser) tr.className = "user-row";

      tr.innerHTML = `
        <td class="${rankClass}" style="font-weight:700;">${rankBadge}</td>
        <td>
          <strong>${escapeHtml(p.name)}</strong>
          ${isUser ? '<span class="you-tag">YOU</span>' : ""}
        </td>
        <td><span class="kind-tag">${kindBadge}</span></td>
        <td><span class="elo-pill ${eloClass}">${p.rating.toFixed(1)}</span></td>
        <td class="muted-hint">${p.peak_rating.toFixed(1)}</td>
        <td>${p.wins} - ${p.draws} - ${p.losses}</td>
        <td>${winPct}</td>
        <td>${p.games_played}</td>
      `;
      tbody.appendChild(tr);
    }

    if (userFound) {
      document.getElementById("modalUserElo").textContent = userFound.rating.toFixed(1);
      document.getElementById("modalUserPeak").textContent = userFound.peak_rating.toFixed(1);
      document.getElementById("modalUserRecord").textContent = `${userFound.wins} - ${userFound.draws} - ${userFound.losses} (${userFound.win_rate > 0 ? (userFound.win_rate * 100).toFixed(0) + "%" : "0%"})`;
      const hdrBadge = document.getElementById("headerEloBadge");
      if (hdrBadge) hdrBadge.textContent = `${Math.round(userFound.rating)}`;
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--danger); text-align:center; padding:16px;">Failed to load leaderboard: ${escapeHtml(err.message)}</td></tr>`;
  }
}

async function loadMatchHistory() {
  const container = document.getElementById("matchesContainer");
  if (!container) return;

  try {
    const res = await fetch("/api/elo/matches");
    const data = await res.json();
    const matches = data.matches || [];

    container.innerHTML = "";
    if (matches.length === 0) {
      container.innerHTML = '<div class="muted-hint" style="text-align:center; padding:30px;">No matches recorded yet. Play a game against an AlphaZero model or Heuristic agent to see history here.</div>';
      return;
    }

    for (const m of matches) {
      const div = document.createElement("div");
      div.className = "match-card";

      const d1Sign = m.p1_delta >= 0 ? `+${m.p1_delta.toFixed(1)}` : `${m.p1_delta.toFixed(1)}`;
      const d2Sign = m.p2_delta >= 0 ? `+${m.p2_delta.toFixed(1)}` : `${m.p2_delta.toFixed(1)}`;
      const d1Class = m.p1_delta > 0 ? "delta-badge-pos" : (m.p1_delta < 0 ? "delta-badge-neg" : "delta-badge-zero");
      const d2Class = m.p2_delta > 0 ? "delta-badge-pos" : (m.p2_delta < 0 ? "delta-badge-neg" : "delta-badge-zero");

      let winBadge = '<span style="color:var(--accent-gold); font-weight:700;">Draw</span>';
      if (m.winner === 0) {
        winBadge = `<span style="color:var(--p1-color); font-weight:700;">${escapeHtml(m.player1_name || m.player1_id)} Won</span>`;
      } else if (m.winner === 1) {
        winBadge = `<span style="color:var(--p2-color); font-weight:700;">${escapeHtml(m.player2_name || m.player2_id)} Won</span>`;
      }

      const dateStr = m.timestamp ? new Date(m.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";

      div.innerHTML = `
        <div style="flex:1;">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; flex-wrap:wrap;">
            <span class="muted-hint" style="font-size:11px;">${dateStr}</span>
            <span>${winBadge}</span>
            <span class="badge tabular">${m.ply_count} plies</span>
            <span class="muted-hint" style="font-size:11px;">${escapeHtml(m.termination_reason || "")}</span>
          </div>
          <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
            <div>
              <span style="color:var(--p1-color); font-weight:700;">P1:</span> ${escapeHtml(m.player1_name || m.player1_id)}
              <span class="${d1Class}" style="margin-left:4px;">(${d1Sign} → ${m.p1_rating_after.toFixed(1)})</span>
            </div>
            <div class="muted-hint">vs</div>
            <div>
              <span style="color:var(--p2-color); font-weight:700;">P2:</span> ${escapeHtml(m.player2_name || m.player2_id)}
              <span class="${d2Class}" style="margin-left:4px;">(${d2Sign} → ${m.p2_rating_after.toFixed(1)})</span>
            </div>
          </div>
        </div>
        <div class="tabular" style="text-align:right; font-weight:700; font-size:13px;">
          ${m.scores ? m.scores[0] + " - " + m.scores[1] : ""}
        </div>
      `;
      container.appendChild(div);
    }
  } catch (err) {
    container.innerHTML = `<div style="color:var(--danger); text-align:center; padding:16px;">Failed to load match history: ${escapeHtml(err.message)}</div>`;
  }
}

function getEffectivePairs() {
  const sel = document.getElementById("tournPairsSelect");
  if (!sel) return 1;
  if (sel.value === "custom") {
    const val = parseInt(document.getElementById("tournCustomPairsInput")?.value, 10);
    return isNaN(val) || val < 1 ? 1 : val;
  }
  return parseInt(sel.value, 10) || 1;
}

function populateTournamentCheckpoints() {
  const container = document.getElementById("tournCheckpointsList");
  const targetSelect = document.getElementById("tournTargetSelect");
  if (!container || !appState || !appState.available_agents) return;

  const models = appState.available_agents.filter((ag) => {
    if (!ag || ag.id === "human") return false;
    if (ag.kind === "heuristic") return true;
    const baseId = (ag.base_id || ag.id.split(":")[0] || "").toLowerCase();
    const filename = baseId.split("/").pop();
    return filename === "best.pt";
  });

  if (targetSelect) {
    const prevTarget = targetSelect.value;
    targetSelect.innerHTML = '<option value="">Round-Robin (All selected models play each other)</option>';
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      const gp = m.games_played !== undefined ? `${m.games_played} games` : "";
      opt.textContent = `${m.label}${gp ? " — [" + gp + "]" : ""}`;
      targetSelect.appendChild(opt);
    }
    if (prevTarget && Array.from(targetSelect.options).some((o) => o.value === prevTarget)) {
      targetSelect.value = prevTarget;
    }
  }

  renderTournamentCheckpointsList(models);
  updateTournamentModeUI();
}

function renderTournamentCheckpointsList(models) {
  const container = document.getElementById("tournCheckpointsList");
  if (!container) return;

  if (!models && appState && appState.available_agents) {
    models = appState.available_agents.filter((ag) => {
      if (!ag || ag.id === "human") return false;
      if (ag.kind === "heuristic") return true;
      const baseId = (ag.base_id || ag.id.split(":")[0] || "").toLowerCase();
      const filename = baseId.split("/").pop();
      return filename === "best.pt";
    });
  }
  if (!models) return;

  const targetId = document.getElementById("tournTargetSelect")?.value || "";

  const prevChecked = new Set();
  document.querySelectorAll("#tournCheckpointsList input[type='checkbox']").forEach((cb) => {
    if (cb.checked) prevChecked.add(cb.value);
  });
  const hasPriorState = prevChecked.size > 0;

  container.innerHTML = "";

  for (const m of models) {
    const isTarget = Boolean(targetId && m.id === targetId);

    const label = document.createElement("label");
    label.className = "tourn-row" + (isTarget ? " tourn-row-target" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = m.id;

    if (isTarget) {
      cb.checked = false;
      cb.disabled = true;
      cb.style.opacity = "0.4";
    } else if (hasPriorState) {
      cb.checked = prevChecked.has(m.id);
    } else {
      cb.checked = (m.id.includes("best.pt") && m.variant === "standard") || m.id === "heuristic:standard";
    }
    cb.addEventListener("change", updateTournamentEstimate);

    label.appendChild(cb);

    const nameSpan = document.createElement("span");
    nameSpan.style.flex = "1";
    if (isTarget) {
      nameSpan.innerHTML = `<strong style="color:var(--p1-color);">[TARGET]</strong> ${escapeHtml(m.label)}`;
    } else {
      nameSpan.textContent = m.label;
    }
    label.appendChild(nameSpan);

    if (m.games_played !== undefined) {
      const gamesBadge = document.createElement("span");
      gamesBadge.className = "tourn-games-tag";
      gamesBadge.textContent = `${m.games_played} games`;
      label.appendChild(gamesBadge);
    }

    container.appendChild(label);
  }
  updateTournamentEstimate();
}

function onTournamentTargetChange() {
  renderTournamentCheckpointsList();
  updateTournamentModeUI();
}

function updateTournamentModeUI() {
  const targetSel = document.getElementById("tournTargetSelect");
  const targetId = targetSel ? targetSel.value : "";
  const badge = document.getElementById("tournModeBadge");
  const label = document.getElementById("tournListLabel");
  const btn = document.getElementById("startTournamentBtn");
  const hint = document.getElementById("tournTargetHint");

  if (targetId) {
    if (badge) {
      badge.textContent = "Gauntlet Mode (1 vs Pool)";
      badge.classList.add("gauntlet");
    }
    if (label) label.textContent = "Select Opponents for Target Model to Face:";
    if (btn && !btn.disabled) btn.textContent = "Start Gauntlet";
    if (hint) hint.innerHTML = "Matches are <strong>100% focused</strong> between the target and checked opponents. Opponents do not play each other.";
  } else {
    if (badge) {
      badge.textContent = "Round-Robin Mode";
      badge.classList.remove("gauntlet");
    }
    if (label) label.textContent = "Select Models to Include (Best Only):";
    if (btn && !btn.disabled) btn.textContent = "Start Tournament";
    if (hint) hint.innerHTML = "Choose a specific model to direct 100% of matches to it against the selected opponents below.";
  }
  updateTournamentEstimate();
}

function updateTournamentEstimate() {
  const estimSpan = document.getElementById("tournTotalEstim");
  if (!estimSpan) return;
  const targetSel = document.getElementById("tournTargetSelect");
  const targetId = targetSel ? targetSel.value : "";
  const pairs = getEffectivePairs();

  const checkboxes = document.querySelectorAll("#tournCheckpointsList input[type='checkbox']:checked");
  const selected = Array.from(checkboxes).map((c) => c.value);

  if (targetId) {
    const opponents = selected.filter((id) => id !== targetId);
    const m = opponents.length;
    if (m === 0) {
      estimSpan.textContent = "Select at least 1 opponent for target to face";
      estimSpan.style.color = "var(--text-muted)";
    } else {
      const totalGames = m * pairs * 2;
      estimSpan.textContent = `Target vs ${m} Opponents · ${pairs} pair${pairs > 1 ? "s" : ""} (${totalGames} total games)`;
      estimSpan.style.color = "var(--accent-green)";
    }
  } else {
    const n = selected.length;
    if (n < 2) {
      estimSpan.textContent = `${n} model${n === 1 ? "" : "s"} selected (min. 2 required)`;
      estimSpan.style.color = "var(--text-muted)";
    } else {
      const totalGames = (n * (n - 1) / 2) * pairs * 2;
      estimSpan.textContent = `${n} models · ${pairs} pair${pairs > 1 ? "s" : ""} (${totalGames} total games)`;
      estimSpan.style.color = "var(--accent-gold)";
    }
  }
}

async function startTournament() {
  const targetSel = document.getElementById("tournTargetSelect");
  const targetId = targetSel ? targetSel.value : "";
  const checkboxes = document.querySelectorAll("#tournCheckpointsList input[type='checkbox']:checked");
  let selected = Array.from(checkboxes).map((c) => c.value);

  if (targetId) {
    selected = selected.filter((id) => id !== targetId);
    if (selected.length === 0) {
      alert("Please select at least 1 opponent for the target model to face.");
      return;
    }
    selected.unshift(targetId);
  } else {
    if (selected.length < 2) {
      alert("Please select at least 2 models/agents for the tournament.");
      return;
    }
  }

  const pairs = getEffectivePairs();
  const terrainMode = document.getElementById("tournTerrainSelect")?.value || "random";

  const btn = document.getElementById("startTournamentBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";

  try {
    const res = await apiPost("/api/elo/tournament", {
      checkpoints: selected,
      pairs: pairs,
      terrain_mode: terrainMode,
      target_agent: targetId || null,
    });

    if (res.error) {
      alert("Tournament error: " + res.error);
      btn.disabled = false;
      btn.textContent = targetId ? "Start Gauntlet" : "Start Tournament";
      return;
    }

    checkTournamentStatus();
  } catch (err) {
    alert("Failed to start tournament: " + err.message);
    btn.disabled = false;
    btn.textContent = targetId ? "Start Gauntlet" : "Start Tournament";
  }
}

async function checkTournamentStatus() {
  const pBox = document.getElementById("tournProgressBox");
  const btn = document.getElementById("startTournamentBtn");
  const targetSel = document.getElementById("tournTargetSelect");
  const targetId = targetSel ? targetSel.value : "";
  if (!pBox) return;

  try {
    const res = await fetch("/api/elo/tournament/status");
    const data = await res.json();
    const t = data.tournament || {};

    if (t.running) {
      pBox.style.display = "block";
      if (btn) {
        btn.disabled = true;
        btn.textContent = t.target_agent ? "Gauntlet Running..." : "Tournament Running...";
      }
      const pct = t.total > 0 ? Math.round((t.progress / t.total) * 100) : 0;
      document.getElementById("tournProgressBar").style.width = `${pct}%`;
      document.getElementById("tournProgressText").textContent = `${t.progress} / ${t.total} Games (${pct}%)`;
      document.getElementById("tournCurrentMatchup").textContent = t.current || "Playing matchup...";
      document.getElementById("tournLastResult").textContent = t.last_result ? `Recent: ${t.last_result}` : "";

      if (!tournamentPollTimer) {
        tournamentPollTimer = setTimeout(() => {
          tournamentPollTimer = null;
          checkTournamentStatus();
        }, 1000);
      }
    } else {
      if (btn) {
        btn.disabled = false;
        btn.textContent = targetId ? "Start Gauntlet" : "Start Tournament";
      }
      if (t.progress > 0 && t.progress === t.total) {
        pBox.style.display = "block";
        document.getElementById("tournProgressBar").style.width = "100%";
        document.getElementById("tournProgressText").textContent = `Complete (${t.total}/${t.total})`;
        document.getElementById("tournCurrentMatchup").textContent = "Tournament Finished";
        document.getElementById("tournLastResult").textContent = "Standings updated.";
        loadLeaderboard();
      } else {
        pBox.style.display = "none";
      }
    }
  } catch (_) {}
}

async function resetAllRatings() {
  if (!confirm("Are you sure you want to reset all Elo ratings back to default 1500? This cannot be undone.")) {
    return;
  }
  try {
    await apiPost("/api/elo/reset", {});
    loadLeaderboard();
    loadMatchHistory();
    renderAll();
    alert("All Elo ratings have been reset to 1500.0.");
  } catch (err) {
    alert("Failed to reset ratings: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// Neural Network & Board Analytics System Logic
// ---------------------------------------------------------------------------
let analyticsActive = false;
let analyticsData = null;
let analyticsLayer = "val_Sentinel";
let showAnalyticsPieces = true;
let showAnalyticsValues = true;
let analyticsLoading = false;

const ANALYTICS_LAYER_META = {
  terrain: {
    title: "Terrain Elevation h(r, c)",
    desc: "Base integer terrain height (0 to 6). Orthogonal step cost is |Δh| + 1. High ground gives +1 downhill combat bonus.",
    type: "terrain",
    format: (v) => `${v}`,
  },
  my_dist: {
    title: "P1 Unblocked Dijkstra Cost to Row 9",
    desc: "Minimum movement points required to step off Row 9 across empty terrain. Lower cost is closer to victory.",
    type: "dist",
    format: (v) => `${v.toFixed(0)}`,
  },
  opp_dist: {
    title: "P2 Unblocked Dijkstra Cost to Row 0",
    desc: "Opponent's empty-board Dijkstra cost to Row 0. Lower indicates opponent territorial pressure.",
    type: "dist",
    format: (v) => `${v.toFixed(0)}`,
  },
  topo_adv: {
    title: "Topological Advantage (P2 Dijkstra - P1 Dijkstra)",
    desc: "Spatial distance differential. Positive values (green) favor Player 1; negative values (blue) favor Player 2.",
    type: "bipolar",
    format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}`,
  },
  static_l2_norm: {
    title: "Static Stem Feature L2 Norm (32 latent channels)",
    desc: "Activation magnitude across the 32 static convolutional feature maps. Highlights ridges, choke points, and highways.",
    type: "sequential",
    format: (v) => v.toFixed(1),
  },
  static_pc1: {
    title: "Static Stem Principal Component 1 (PC1)",
    desc: "First principal component of the 32 static feature channels across all tiles. Captures the primary learned terrain topology.",
    type: "bipolar",
    format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`,
  },
  val_Sentinel: {
    title: "Sentinel Positional Value V(s) ∈ [-1, +1]",
    desc: "Model value head expectation if a Sentinel occupies this square. Surges toward +0.98 near Row 9 due to Sentinel Instant Win.",
    type: "value",
    format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`,
  },
  val_Vanguard: {
    title: "Vanguard Positional Value V(s) ∈ [-1, +1]",
    desc: "Model value head expectation with Vanguard on this square. Reflects forward progression and attack threat potential.",
    type: "value",
    format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`,
  },
  val_Explorer: {
    title: "Explorer Positional Value V(s) ∈ [-1, +1]",
    desc: "Model value head expectation with Explorer on this square.",
    type: "value",
    format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`,
  },
  policy_priors: {
    title: "Move 1 Policy Landing Prior P(move)",
    desc: "Sum of probabilities across all active pieces of moving to this tile on this turn.",
    type: "policy",
    format: (v) => (v > 0.0005 ? `${(v * 100).toFixed(0)}%` : "."),
  },
};

function getAnalyticsValue(r, c, layer) {
  if (!analyticsData) return null;
  if (layer === "terrain") return analyticsData.terrain[r][c];
  if (layer === "my_dist") return analyticsData.my_dist[r][c];
  if (layer === "opp_dist") return analyticsData.opp_dist[r][c];
  if (layer === "topo_adv") return analyticsData.topo_adv[r][c];
  if (layer === "static_l2_norm") return analyticsData.static_l2_norm[r][c];
  if (layer === "static_pc1") return analyticsData.static_pc1[r][c];
  if (layer === "policy_priors") return analyticsData.policy_priors[r][c];
  if (layer.startsWith("val_")) {
    const piece = layer.replace("val_", "");
    if (analyticsData.value_maps && analyticsData.value_maps[piece]) {
      return analyticsData.value_maps[piece][r][c];
    }
  }
  return null;
}

function getAnalyticsMinMax(layer) {
  if (!analyticsData) return { min: 0, max: 1 };
  let min = Infinity, max = -Infinity;
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const v = getAnalyticsValue(r, c, layer);
      if (v !== null && v !== undefined) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  if (!isFinite(min)) min = 0;
  if (!isFinite(max)) max = 1;
  return { min, max };
}

function getAnalyticsColor(val, min, max, type) {
  if (type === "policy") {
    if (val < 0.0005) return "rgb(38, 44, 58)";
    const frac = Math.max(0, Math.min(1, val / 0.22));
    const r = Math.round(38 + (217 - 38) * frac);
    const g = Math.round(44 + (62 - 44) * frac);
    const b = Math.round(58 + (82 - 58) * frac);
    return `rgb(${r}, ${g}, ${b})`;
  }

  if (type === "value") {
    if (val >= 0) {
      const frac = Math.max(0, Math.min(1, val));
      const r = Math.round(44 + (22 - 44) * frac);
      const g = Math.round(52 + (148 - 52) * frac);
      const b = Math.round(64 + (104 - 64) * frac);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const frac = Math.max(0, Math.min(1, -val));
      const r = Math.round(44 + (205 - 44) * frac);
      const g = Math.round(52 + (58 - 52) * frac);
      const b = Math.round(64 + (58 - 64) * frac);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  if (type === "bipolar") {
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
    if (val >= 0) {
      const frac = Math.max(0, Math.min(1, val / absMax));
      const r = Math.round(44 + (22 - 44) * frac);
      const g = Math.round(52 + (148 - 52) * frac);
      const b = Math.round(64 + (104 - 64) * frac);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const frac = Math.max(0, Math.min(1, -val / absMax));
      const r = Math.round(44 + (44 - 44) * frac);
      const g = Math.round(52 + (116 - 52) * frac);
      const b = Math.round(64 + (196 - 64) * frac);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  if (type === "dist") {
    const frac = max === min ? 0.5 : (val - min) / (max - min);
    const r = Math.round(24 + (1 - frac) * 32);
    const g = Math.round(46 + (1 - frac) * 140);
    const b = Math.round(80 + frac * 115);
    return `rgb(${r}, ${g}, ${b})`;
  }

  const frac = max === min ? 0.5 : Math.max(0, Math.min(1, (val - min) / (max - min)));
  const r = Math.round(38 + frac * 185);
  const g = Math.round(56 + frac * 145);
  const b = Math.round(76 + (1 - frac) * 90);
  return `rgb(${r}, ${g}, ${b})`;
}

function updateAnalyticsRibbon() {
  if (!analyticsData) return;
  const meta = ANALYTICS_LAYER_META[analyticsLayer];
  const mm = getAnalyticsMinMax(analyticsLayer);

  const minEl = document.getElementById("legendMinLabel");
  const maxEl = document.getElementById("legendMaxLabel");
  if (minEl) minEl.textContent = `Min: ${meta.format(mm.min)}`;
  if (maxEl) maxEl.textContent = `Max: ${meta.format(mm.max)}`;

  const legGrad = document.getElementById("legendGradientBar");
  if (legGrad) {
    if (meta.type === "value") {
      legGrad.style.background = "linear-gradient(to right, rgb(205, 58, 58), rgb(44, 52, 64), rgb(22, 148, 104))";
    } else if (meta.type === "bipolar") {
      legGrad.style.background = "linear-gradient(to right, rgb(44, 116, 196), rgb(44, 52, 64), rgb(22, 148, 104))";
    } else if (meta.type === "dist") {
      legGrad.style.background = "linear-gradient(to right, rgb(56, 186, 80), rgb(24, 46, 195))";
    } else if (meta.type === "policy") {
      legGrad.style.background = "linear-gradient(to right, rgb(38, 44, 58), rgb(217, 62, 82))";
    } else {
      legGrad.style.background = "linear-gradient(to right, rgb(38, 56, 166), rgb(223, 201, 76))";
    }
  }

  const infoEl = document.getElementById("analyticsLayerInfoText");
  if (infoEl) {
    infoEl.innerHTML = `<strong style="color:var(--accent-green);">${escapeHtml(meta.title)}:</strong> ${escapeHtml(meta.desc)}`;
  }

  const ckptSel = document.getElementById("analyticsModelSelect");
  if (ckptSel && analyticsData.available_checkpoints) {
    if (ckptSel.children.length === 0) {
      analyticsData.available_checkpoints.forEach((pt) => {
        const opt = document.createElement("option");
        opt.value = pt;
        opt.textContent = pt;
        if (pt === analyticsData.checkpoint) opt.selected = true;
        ckptSel.appendChild(opt);
      });
    }
  }
}

function isAnalyticsStale() {
  if (!analyticsData || !appState) return true;
  if (analyticsData.cursor !== appState.cursor) return true;
  if (analyticsData.ply_count !== appState.ply_count) return true;
  if (!analyticsData.terrain || !appState.terrain) return true;
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      if (analyticsData.terrain[r][c] !== appState.terrain[r][c]) return true;
    }
  }
  return false;
}

async function setAnalyticsMode(active) {
  analyticsActive = !!active;
  const gameBtn = document.getElementById("viewModeGameBtn");
  const analBtn = document.getElementById("viewModeAnalyticsBtn");
  const ribbon = document.getElementById("analyticsRibbon");

  if (analyticsActive) {
    if (gameBtn) gameBtn.classList.remove("active-toggle");
    if (analBtn) analBtn.classList.add("active-toggle");
    if (ribbon) ribbon.style.display = "flex";

    if (isAnalyticsStale()) {
      await fetchAnalytics();
    } else {
      updateAnalyticsRibbon();
      drawBoard();
      renderHoverInspector();
    }
  } else {
    if (analBtn) analBtn.classList.remove("active-toggle");
    if (gameBtn) gameBtn.classList.add("active-toggle");
    if (ribbon) ribbon.style.display = "none";

    drawBoard();
    renderHoverInspector();
  }
}

async function fetchAnalytics(ckpt) {
  if (analyticsLoading) return;
  analyticsLoading = true;
  const refBtn = document.getElementById("refresh3dAnalyticsBtn");
  const refLabel = refBtn ? refBtn.querySelector(".btn-label") : null;
  if (refLabel) refLabel.textContent = "Loading...";
  else if (refBtn) refBtn.textContent = "Loading...";

  try {
    const payload = ckpt ? { checkpoint: ckpt } : {};
    const data = await apiPost("/api/analytics", payload);
    analyticsData = data;

    updateAnalyticsRibbon();
    drawBoard();
    renderHoverInspector();
  } catch (err) {
    console.error("Failed to load analytics:", err);
    alert("Failed to compute board analytics: " + err.message);
  } finally {
    analyticsLoading = false;
    if (refLabel) refLabel.textContent = "Refresh";
    else if (refBtn) refBtn.textContent = "Refresh";
  }
}

function initAnalyticsSystem() {
  const gameBtn = document.getElementById("viewModeGameBtn");
  const analBtn = document.getElementById("viewModeAnalyticsBtn");

  if (gameBtn) gameBtn.addEventListener("click", () => setAnalyticsMode(false));
  if (analBtn) analBtn.addEventListener("click", () => setAnalyticsMode(true));

  const ckptSel = document.getElementById("analyticsModelSelect");
  if (ckptSel) {
    ckptSel.addEventListener("change", (e) => {
      fetchAnalytics(e.target.value);
    });
  }

  const refBtn = document.getElementById("refresh3dAnalyticsBtn");
  if (refBtn) {
    refBtn.addEventListener("click", () => {
      const ckpt = ckptSel ? ckptSel.value : null;
      fetchAnalytics(ckpt);
    });
  }

  const pBtn = document.getElementById("toggle3dPiecesBtn");
  if (pBtn) {
    pBtn.addEventListener("click", () => {
      showAnalyticsPieces = !showAnalyticsPieces;
      const lbl = pBtn.querySelector(".btn-label");
      const text = showAnalyticsPieces ? "Pieces: ON" : "Pieces: OFF";
      if (lbl) lbl.textContent = text;
      else pBtn.textContent = text;
      pBtn.classList.toggle("active-toggle", showAnalyticsPieces);
      drawBoard();
    });
  }

  const vBtn = document.getElementById("toggle3dValuesBtn");
  if (vBtn) {
    vBtn.addEventListener("click", () => {
      showAnalyticsValues = !showAnalyticsValues;
      const lbl = vBtn.querySelector(".btn-label");
      const text = showAnalyticsValues ? "Values: ON" : "Values: OFF";
      if (lbl) lbl.textContent = text;
      else vBtn.textContent = text;
      vBtn.classList.toggle("active-toggle", showAnalyticsValues);
      drawBoard();
    });
  }

  document.querySelectorAll(".layer-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".layer-pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      analyticsLayer = btn.getAttribute("data-layer");
      updateAnalyticsRibbon();
      drawBoard();
      renderHoverInspector();
    });
  });
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
setShortcutHintsVisible(showShortcutHints);
renderShortcutsModalContent();
initAnalyticsSystem();
fetchInitialState();
