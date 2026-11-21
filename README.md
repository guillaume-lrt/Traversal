# Traversal — AlphaZero AI, Engine & Web Visualizer

A Numba-accelerated Python implementation of the abstract 3D strategy board game **Traversal**, complete with a GPU-batched **AlphaZero** training pipeline (Dual-Stem Residual Network + MCTS), a multi-ply Alpha-Beta heuristic baseline, an automated Elo tournament arena, and an interactive 3D/2D local web visualizer.

---

## Overview

Traversal is a two-player tactical race and capture game played on a procedurally generated $10 \times 10$ grid of variable terrain elevations. This project implements:

- **Headless Game Engine (`traversal_env.py`):** Exact rules implementation accelerated with `numba` `@njit` kernels (Dijkstra movement costs, Explorer special movement, combat modifiers, finishing rules, and threefold repetition detection).
- **AlphaZero Neural Network & Batched MCTS (`encoder.py`, `model.py`, `mcts.py`, `train.py`):** Dual-Stem ResNet separating static terrain features from dynamic board states, paired with lockstep GPU-batched MCTS, static-stem caching, subtree reuse, and dual curriculum annealing.
- **Interactive Web Visualizer (`visualizer.py`, `static/index.html`):** Play as Human vs. AI, AI vs. AI, or inspect live MCTS policy/value overlays in both 3D Isometric and 2D Tactical views, plus a built-in Elo Leaderboard and match history.
- **Elo Arena & Spatial Diagnostics (`elo_system.py`, `elo_arena.py`, `inspect_tiles.py`):** FIDE-style Elo rating system, round-robin/gauntlet tournaments, and neural network spatial activation probes.

---

## Game Rules Summary

- **Board & Terrain:** $10 \times 10$ grid with integer heights $h(r, c) \ge 0$ constrained by $|h(r_1, c_1) - h(r_2, c_2)| \le 2$ across orthogonal neighbors. Supports procedural terrain modes (`random`, `mountains`, `valleys`, `highlands`, `woodlands`, `desert`, `flat`).
- **Movement Cost:** Orthogonal steps cost $|h(r_2, c_2) - h(r_1, c_1)| + 1$ (flat = 1, $\pm 1$ elevation = 2, $\pm 2$ elevation = 3). Pieces cannot pass through occupied squares.
- **Pieces (6 per player):** Each player starts on their baseline (Row 0 or Row 9, Columns 2–7) with 2 of each piece type ($\text{Movement} + \text{Strength} = 6$):
  - **Explorer** ($M=5, S=1, 1\text{ pt}$): May forfeit normal movement to take a standalone 1-square diagonal step or 2-square orthogonal leap to an empty square regardless of height difference.
  - **Vanguard** ($M=4, S=2, 2\text{ pts}$): Gains $+1$ Attack Strength (**Charge Bonus**) when reaching an enemy target via shortest path distance $d_{\min} = 4$.
  - **Sentinel** ($M=3, S=3, 3\text{ pts}$): If a Sentinel is the **very first piece** (of either player) to finish off the opposite edge of the board, its owner wins **immediately**.
- **Combat:** Capturing requires $\text{Attack} \ge \text{Defense}$ (captures via Explorer special moves are illegal):
  - $\text{Attack} = S_{\text{attacker}} + \text{HeightMod} \ (\pm 1 \text{ uphill/downhill}) + \text{VanguardCharge} + \sum S_{\text{allied adj.} \neq \text{attacker}}$
  - $\text{Defense} = S_{\text{defender}} + \text{count}(\text{friendly pieces orthogonally adjacent to defender})$
- **Victory Conditions:**
  1. **Sentinel Instant Win:** First piece to step off Row 9 (canonical) is a Sentinel.
  2. **Two-Piece Finish:** Any 2 pieces total finish off the board; higher score wins (equal scores = draw).
  3. **Elimination / No Legal Moves:** Higher score (or race tiebreak) wins.
  4. **Threefold Repetition & 200-Ply Limit:** Threefold repetition is an immediate draw; 200 plies resolves by score and board progress tiebreakers.

---

## AI Architecture

### 1. Versioned State Encoder (`encoder.py`)
Inputs are canonically oriented from the current player's perspective (moving Row 0 $\to$ Row 9) and split into static and dynamic blocks:
- **`X_static` $(4, 10, 10)$:** Computed once per player per game — normalized terrain height, unblocked Dijkstra distance to Row 9, opponent's unblocked distance, and topological distance advantage.
- **`X_dynamic` v2 $(35, 10, 10)$:** Piece bitplanes, reachability flags, attack/defense threat maps (with friendly-occupied attack cleanup), obstacle-aware blocked Dijkstra distances (`compute_blocked_dist_std_v2`), game progress scalars, and repetition indicator plane (`Dyn 34`). *(Legacy 34-channel v1 checkpoints remain supported automatically.)*
- **Symmetry Augmentation:** Left-right board reflection swapping paired pieces (`Explorer_0 <-> Explorer_1`, `Vanguard_0 <-> Vanguard_1`, `Sentinel_0 <-> Sentinel_1`) and mirroring the 606-action space.

### 2. Dual-Stem Residual Network (`model.py`)
`TraversalDualStemNet` caches the static terrain representation during search:
- **Static Stem:** $(B, 4, 10, 10) \to \text{Conv}_{3\times 3}(32) \to 2\times \text{ResBlock}(32)$
- **Dynamic Stem:** $(B, 35, 10, 10) \to \text{Conv}_{3\times 3}(64) \to 2\times \text{ResBlock}(64)$
- **Fusion Trunk:** Concatenated $(96, 10, 10) \to \text{Conv}_{1\times 1}(96) \to 4\times \text{ResBlock}(96)$
- **Policy Head:** $\text{Conv}_{1\times 1}(96 \to 32) \to \text{Linear}(3200 \to 606)$ with legal-action masking
- **GAP Value Head:** $\text{Conv}_{1\times 1}(96 \to 32) \to \text{Global Mean + Max Pool}(64) \to \text{Linear}(64 \to 64) \to \text{Linear}(64 \to 1) \to \tanh$ (eliminates spatial coordinate memorization; legacy flat value heads are auto-detected on load).

### 3. Batched MCTS & Heuristic Baseline (`mcts.py`, `train.py`)
- **Lockstep GPU-Batched MCTS:** Evaluates leaf nodes across active games in parallel using cached `F_static` tensors, PUCT selection ($c_\text{puct}=1.5$), calibrated root Dirichlet noise ($\alpha=0.18, \epsilon=0.25$), and root subtree reuse across plies.
- **Dual Curriculum Annealing:** Can bootstrap from scratch with heuristic game injection, imitation loss, and MCTS tactical shaping (`shaping_weight`, `v_tac_weight`), then linearly anneal all scaffolding to `0.0` for pure autonomous self-play.
- **Alpha-Beta Heuristic Baseline (`HeuristicAgent`):** Numba `@njit`-compiled negamax alpha-beta search with move ordering, iterative deepening, repetition awareness, and presets from `blitz` (2-ply) up to `grandmaster` (8-ply).

---

## Elo Leaderboard & Training Evolution (`train_v1` $\to$ `train_v8`)

Ratings tracked in [`elo_ratings.json`](elo_ratings.json) across `blitz` (32 sims), `standard` (96 sims), and `master` (256 sims) presets (see [TRAIN_VERSIONS_SUMMARY.md](TRAIN_VERSIONS_SUMMARY.md) for full details):

| Model / Agent | Blitz (32s) | Standard (96s) | Master (256s) | Key Milestone |
| :--- | :---: | :---: | :---: | :--- |
| **`train_v8/best.pt`** | **1599.9** | **1837.5** | **1902.5** *(Peak 2007.5)* | **#1 Overall** — 1,000 iters, GAP + Encoder v2, LR `5e-4 -> 1e-4`, **97.5%** vs Heuristic |
| **`train_v7/best.pt`** | 1578.1 | 1725.4 | 1828.7 *(Peak 1996.3)* | From-scratch Dual Curriculum Annealing (92.5% vs Heuristic) |
| **`train_v6/best.pt`** | 1486.8 | 1487.8 | 1727.0 *(Peak 1818.1)* | Pure zero-curriculum self-play (0% heuristic, 0.0 shaping) |
| **`train_v5/best.pt`** | 1388.2 | 1506.9 | 1616.6 *(Peak 1789.6)* | GAP value head + Encoder v2 + raw targets overhaul |
| **`train_v4/best.pt`** | 1365.4 | 1460.6 | 1540.5 *(Peak 1787.7)* | Peak legacy Flat-head / Encoder v1 model |
| **`train_v3/best.pt`** | 1361.1 | 1380.2 | 1482.8 | Extended legacy horizon (851 iters) |
| **`train_v2/best.pt`** | 1379.6 | 1365.0 | 1311.1 | Scaled self-play buffer & 160 sims |
| **`train_v1/best.pt`** | 1186.8 | 1236.0 | 1265.6 | Initial proof-of-concept |
| *Heuristic Baseline* | 1241.3 *(2-ply)* | 1319.4 *(4-ply)* | 1536.2 *(6-ply)* / 1476.5 *(8-ply)* | Numba Alpha-Beta search |

---

## Project Structure

```text
.
├── traversal_env.py          # Headless Traversal rules engine & Numba movement/combat kernels
├── encoder.py                # Static (4ch) & Dynamic (34/35ch) tensor encoders + mirror augmentation
├── model.py                  # TraversalDualStemNet (ResNet + Policy/GAP Value heads) & loss
├── mcts.py                   # Single-game & GPU-batched MCTS with subtree reuse and tactical shaping
├── train.py                  # AlphaZero training loop, Float16 ring buffer, Aim tracking, HeuristicAgent
├── visualizer.py             # Multi-threaded local HTTP server for the interactive web UI
├── static/index.html         # 3D Isometric & 2D Tactical browser UI + live MCTS overlay + Elo dashboard
├── elo_system.py             # Thread-safe FIDE-style Elo database & variant preset parser
├── elo_arena.py              # CLI for Elo leaderboard, match history, head-to-head & tournaments
├── elo_ratings.json          # Compacted Elo ratings & recent tournament/human match history
├── inspect_tiles.py          # Diagnostic tool to probe static stem features, value maps & priors
├── tests/                    # Pytest suite (engine, encoder, model, MCTS, training, Elo, visualizer)
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── TRAIN_VERSIONS_SUMMARY.md # Technical history & benchmarks for train_v1 through train_v8
└── traversal_alphazero_ai_architecture_specification.md  # Full game & AI architecture spec
```

---

## Quickstart & Usage

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Visualizer

Start the local web server (default `http://127.0.0.1:8000`) to play against checkpoints or Heuristic bots, watch AI vs. AI games, and inspect live MCTS evaluations:

```bash
python visualizer.py --port 8000 --checkpoint-dir checkpoints
```

### 3. Train AlphaZero

Run self-play training with dual curriculum annealing and replay buffer persistence:

```bash
python train.py \
  --iterations 1000 \
  --games-per-iter 12 \
  --mcts-sims 160 \
  --mcts-anneal-iters 300 \
  --heuristic-anneal-iters 300 \
  --target-reuse 3.0 \
  --eval-games 20 \
  --eval-interval 5 \
  --save-buffer
```

Resume or warm-start from an existing checkpoint:

```bash
python train.py --resume checkpoints/train_v8/latest.pt --iterations 1200 --save-buffer
```

### 4. Elo Leaderboard & Automated Arena

View the current leaderboard or recent match history:

```bash
python elo_arena.py leaderboard
python elo_arena.py history --limit 20
```

Run a direct head-to-head match or tournament between checkpoints and `heuristic`:

```bash
python elo_arena.py match checkpoints/train_v8/best.pt:standard heuristic:master --pairs 10
python elo_arena.py tournament --target train_v8/best.pt:master --games-per-pair 10
```

### 5. Spatial Network Inspection

Inspect static stem activations, positional value probes, and Turn 1 policy priors on procedural terrains:

```bash
python inspect_tiles.py --checkpoint checkpoints/train_v8/best.pt --terrain mountains --layer all
```

### 6. Run Unit Tests

```bash
pytest -q
```
