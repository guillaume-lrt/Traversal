# Traversal AlphaZero: Evolution & Advances Across Training Versions

This document provides a comprehensive technical breakdown of each training generation (`train_v1` through `train_v8`) in the Traversal AlphaZero project. It details the architectural changes, search innovations, hyperparameter evolutions, empirical benchmarks, and strategic behaviors learned at each stage.

---

## Executive Summary

| Version | Iterations | Network Architecture | Encoder | Value Target & Reuse | MCTS Shaping / Curriculum | Peak Elo (Master: 256 sims) | Key Advances & Milestone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_v1`** | 184 (best) / 275 | Dual-Stem ResNet, Flat Head (`Linear(800, 128)`) | v1 (34 dyn) | Shaped outcomes, unconstrained reuse | Fixed shaping ($1.0, v_\text{tac}=0.3$), 25% heuristic | **~1265.6** (Peak 1500.0) | Initial baseline proof-of-concept. Rapidly learns basic movement and captures. |
| **`train_v2`** | 674 (best) / 677 | Dual-Stem ResNet, Flat Head | v1 (34 dyn) | Shaped outcomes, unconstrained reuse | Fixed shaping ($1.0, v_\text{tac}=0.3$), 25% heuristic | **~1311.1** (Peak 1500.0) | Scaled self-play batch (12 games/iter, buffer 55k, batch 128, 160 sims). Major jump in tactical competence. |
| **`train_v3`** | 851 (best) / 852 | Dual-Stem ResNet, Flat Head | v1 (34 dyn) | Shaped outcomes, unconstrained reuse | Fixed shaping ($1.0, v_\text{tac}=0.3$), 25% heuristic | **1482.8** (Peak 1627.9) | Extended training horizon. Reliable multi-turn defense and coordinated attacks. |
| **`train_v4`** | 995 (best) / 996 | Dual-Stem ResNet, Flat Head | v1 (34 dyn) | Shaped outcomes, unconstrained reuse | Fixed shaping ($1.0, v_\text{tac}=0.3$), 25% heuristic | **1540.5** (Peak 1787.7) | Peak legacy model. Lowest training loss (1.83), but hit an architectural overfitting wall (fresh MSE 0.525). |
| **`train_v5`** | 893 (best) / 900 | Dual-Stem ResNet, **GAP Head** (`Linear(64, 64)`) | **v2** (35 dyn + rep) | **Raw targets ($\pm 1/0$)**, `--target-reuse 3.0` | **Linear anneal** (shaping $\to 0$, $v_\text{tac} \to 0$ over 300 iters) | **1616.6** (Peak 1789.6) | **The Architectural Overhaul:** GAP value head eliminates spatial memorization; Encoder v2 adds obstacle-aware Dijkstra and repetition plane; MCTS subtree reuse; calibrated Dirichlet noise ($\alpha=0.18$). |
| **`train_v6`** | 1045 (best) / 1100+ | Dual-Stem ResNet, GAP Head | v2 (35 dyn + rep) | Raw targets ($\pm 1/0$), `--target-reuse 3.0` | **Pure Self-Play** (0% heuristic, 0.0 shaping) | **1727.0** (Peak 1818.1) | **Pure Zero-Curriculum Self-Play:** Resumed from v5 with zero heuristic scaffolding and zero synthetic shaping. Surpasses `train_v5/best.pt` head-to-head. |
| **`train_v7`** | 900 (best) / 934 | Dual-Stem ResNet, **GAP Head** (`Linear(64, 64)`) | **v2** (35 dyn + rep) | **Raw targets ($\pm 1/0$)**, `--target-reuse 3.0` | **Dual Curriculum Annealing** (Shaping $\to 0$, $v_\text{tac} \to 0$, Heuristic $\to 0\%$ over 300 iters) | **1828.7** (Peak 1996.3) | **Unified From-Scratch Zero Curriculum:** Bootstraps with heuristic games and tactical MCTS shaping, then anneals all scaffolding to 0.0 by iter 300, reaching 92.5% vs HeuristicAgent. |
| **`train_v8`** | 555 (best) / 1000 | Dual-Stem ResNet, **GAP Head** (`Linear(64, 64)`) | **v2** (35 dyn + rep) | **Raw targets ($\pm 1/0$)**, `--target-reuse 3.0`, fine-tuned LR (`5e-4 -> 1e-4`) | **Dual Curriculum Annealing** (Shaping $\to 0$, $v_\text{tac} \to 0$, Heuristic $\to 0\%$ over 300 iters) | **1902.5** (Peak **2007.5**) | **Peak Grandmaster Generation:** Completes full 1,000 iterations with fine-tuned cosine LR (`5e-4 -> 1e-4`), setting project records: **97.5%** vs Heuristic, **1837.5** Standard Elo, and **1902.5** Master Elo. |

---

## Detailed Version Analysis

---

### `train_v1`: Baseline AlphaZero Proof-of-Concept

#### 1. Core Configuration & Design
* **Architecture:** Dual-stem residual network (Static stem 4ch $\to$ Conv 32; Dynamic stem 34ch $\to$ Conv 64; concatenated to 96ch $\to$ 6 residual blocks $\to$ Policy head + Flat Value head).
* **Value Head:** Fully-connected layer flattening the $96 \times 10 \times 10$ feature map directly into `Linear(800, 128) -> Linear(128, 1)`.
* **Encoder (v1):** 4 static planes (height, unblocked distances) + 34 dynamic planes (piece positions, reachability flags, attack/defense potentials, blocked distances, game progress scalars).
* **Training Hyperparameters:** Batch size 64/128, 8–12 games/iteration, buffer capacity 40k–55k samples, 60–100 training steps/iteration, learning rate $10^{-3} \to 10^{-4}$.
* **Curriculum & MCTS:** 25% of self-play turns injected from `HeuristicAgent`; MCTS guided by tactical prior shaping (`shaping_weight = 1.0`) and heuristic value anchor (`v_tac_weight = 0.30`).

#### 2. Key Observations & Strengths
* Successfully demonstrated that AlphaZero self-play with dual-stem ResNets can master the complex rules of Traversal (variable terrain movement costs, elevation combat modifiers, Explorer leaping, and Sentinel victory conditions).
* Rapidly reached basic tactical proficiency within ~180 iterations, learning to capture undefended pieces and advance toward the opponent's baseline.

#### 3. Limitations
* Small evaluation sample sizes (4 games) led to high variance in reported win rates.
* Flat value head began memorizing specific board coordinates rather than general piece configurations.
* **Peak Ratings:** Standard (96 sims): **1236.0** | Master (256 sims): **1311.0**.

---

### `train_v2`: Scaling Self-Play Data & Search Budget

#### 1. What Changed from `train_v1`
* **Dataset Scale:** Replay buffer enlarged to 55,000 samples.
* **Search Depth:** MCTS search iterations increased to 160 simulations per turn during self-play.
* **Extended Iterations:** Training extended to 677 iterations (checkpointed at `iter_0100` through `iter_0600`).
* **Evaluation:** Increased evaluation games to 8 games per checkpoint at 64 simulations.

#### 2. Empirical Performance
* **Loss Convergence:** Training loss decreased smoothly from ~2.60 down to **2.011** (best at iteration 674).
* **Elo Advancement:** 
  * Master rating leaped from **1311.0 to 1496.2 (+185.2 Elo)**.
  * Standard rating leaped from **1236.0 to 1472.7 (+236.7 Elo)**.
* **Gameplay Advancements:** Developed reliable multi-piece coordination, active Vanguard charges utilizing the $+1$ strength bonus at distance 4, and consistent defense against early Explorer rushes.

#### 3. Limitations
* Value head continued to exhibit spatial memorization.
* Unbounded sample reuse ratio caused the optimizer to overfit to older replay buffer positions.

---

### `train_v3`: Extended Self-Play Horizon & Mid-Game Stability

#### 1. What Changed from `train_v2`
* **Extended Horizon:** Pushed training past 850 iterations (`latest.pt` at iter 852, `best.pt` at iter 851).
* **Checkpoint Cadence:** Saved full model states every 100 iterations (`iter_0100.pt` through `iter_0800.pt`).
* Maintained 160 MCTS simulations and 12 games per iteration to build an extensive database of 850+ iterations of self-play.

#### 2. Empirical Performance
* **Elo Advancement:**
  * Master rating reached **1627.9 (+131.7 Elo over v2)**.
  * Standard rating reached **1603.9 (+131.2 Elo over v2)**.
* **Strategic Evolution:** Demonstrated solid mid-game piece protection and recognition of two-piece race positions. Mastered defensive encirclement of advancing opponent Sentinels.

#### 3. Limitations
* Plateauing returns: Despite adding ~180 iterations beyond v2, Elo gains began tapering off (+131 Elo vs +185 Elo in v2), signalling that structural model deficiencies (value head memorization, biased MCTS targets) were capping further self-play growth.

---

### `train_v4`: Peak Legacy Generation & The Overfitting Diagnostic

#### 1. Characteristics & Milestone
* Trained up to **996 iterations** (`best.pt` at iter 995, `latest.pt` at iter 996).
* Achieved the lowest reported training loss of any legacy model (**1.829**).
* **Peak Ratings:** Blitz (32 sims): **1488.5** | Standard (96 sims): **1615.2** | Master (256 sims): **1653.7**.

#### 2. The Comprehensive Code Review & Root Cause Analysis
Despite achieving a respectable Master Elo of 1653.7, an in-depth empirical audit revealed fundamental bottlenecks holding the model back from true grandmaster strength:

1. **Severe Value Head Overfitting (Spatial Memorization):**
   * While training value MSE appeared deceptively low (~0.05), evaluation on **fresh held-out self-play games yielded a value MSE of 0.525**!
   * Predicting the trivial mean across all games yielded an MSE of 0.542—meaning the value head was barely performing better than random baseline guessing on unfamiliar positions.
   * Cause: The flat value head (`Linear(800, 128)`) possessed 102,400 spatial weights that memorized absolute $(r, c)$ tile indices rather than relative piece constellations.
2. **Excessive Sample Reuse:**
   * Gradient samples per iteration were drawn without tracking the ratio of new-to-old samples, resulting in sample reuse ratios $> 5\times$ that reinforced stale patterns.
3. **MCTS Tactical Shaping Pathology (The Sentinel Rush Bias):**
   * The fixed tactical prior shaping (`shaping_weight=1.0`) and heuristic value blend (`v_tac_weight=0.30`) heavily incentivized immediate forward Sentinel rushes.
   * **Empirical diagnostic:** Sentinel instant wins accounted for **60.0%** of all game terminations! Natural two-piece finish races were crowded out (only 40.0%), stunting the network's understanding of deep endgame race dynamics.
4. **Target Inconsistency:**
   * Value targets inserted into the replay buffer used shaped speed/material rewards rather than strictly matching the $\pm 1 / 0$ leaf terminal values of the MCTS tree.
5. **Encoder Blind Spots (Input v1):**
   * Blocked Dijkstra distances clamped occupied tiles to a distance of 30.0 instead of calculating unblocked traversal around obstacles.
   * No repetition history plane existed, leaving the network blind to threefold repetition draw traps.
6. **MCTS Dirichlet Noise Miscalibration:**
   * Dirichlet noise $\alpha = 0.30$ (standard for chess, average branching ~35) was too concentrated for Traversal's average branching factor (~55), where $\alpha \approx 10 / 55 \approx 0.18$ was theoretically appropriate.

These diagnostics paved the way for the full **Phase 1–6 Architectural Overhaul** in `train_v5`.

---

### `train_v5`: The Architectural & Algorithmic Overhaul

`train_v5` executed the complete transformation specified in `PROGRESS.md`, rebuilding the representation, training targets, MCTS search dynamics, and evaluation pipeline from the ground up.

```
       Dual-Stem Convolutional Features (96 channels, 10x10)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
      Policy Head                         GAP Value Head
   Conv(96->32, 1x1)                     Conv(96->32, 1x1)
            │                                     │
    BN + ReLU + Flat                      Global Pool (Mean + Max)
            │                                     │
   Linear(3200, 606)                       Linear(64, 64) -> ReLU
            │                                     │
      Action Mask                          Linear(64, 1) -> Tanh
            │                                     │
     Softmax(606)                             Value v in [-1, +1]
```

#### 1. Core Architectural & Algorithmic Innovations
1. **Global Average + Max Pooling (GAP) Value Head:**
   * Replaced the 800-dim flat linear layer with combined channel-wise spatial mean and max pooling (`cat(mean_pool, max_pool)` $\in \mathbb{R}^{64}$) followed by a compact two-layer MLP (`Linear(64, 64) -> Linear(64, 1)`).
   * **Impact:** Enforces total spatial translation and layout invariance. Value predictions rely on relational piece density and threats rather than memorized $(r, c)$ slots.
2. **Versioned State Encoder v2 (35 Dynamic Channels):**
   * *Blocked Distance Fix:* Implemented `compute_blocked_dist_std_v2`, routing distances around obstacles without pass-through traversal (fixing the 30.0 clamp bug).
   * *Repetition Plane (Dyn 34):* Added a binary indicator plane ($1.0$ if the board state has appeared $\ge 2$ times, $0.0$ otherwise), enabling the network to steer toward or away from threefold repetition draws.
   * *Friendly Attack Cleanup:* Zeroed out friendly-occupied tiles in attack planes (`Dyn 24` and `Dyn 26`).
3. **Controlled Sample Reuse & Fresh Target Validation:**
   * Introduced `--target-reuse 3.0`: Gradient steps dynamically scale with incoming self-play volume ($R \times N_\text{new} / B$), preventing over-fitting.
   * Instrumented Aim tracking with fresh held-out validation metrics before buffer insertion: `fresh_value_mse`, `fresh_policy_kl`, `fresh_value_sign_acc`, and `fresh_value_corr`.
4. **Target Consistency & Value Smoothing:**
   * Configured `--value-target raw` ($\pm 1 / 0$), perfectly aligning replay buffer targets with MCTS leaf terminations.
   * Added visit-weighted root Q-blend option ($z' = (1-\lambda)z + \lambda \cdot \text{root\_Q}$).
5. **MCTS Shaping & Curriculum Annealing:**
   * Introduced linear annealing of MCTS shaping (`shaping_weight: 1.0 -> 0.0`) and heuristic value anchor (`v_tac_weight: 0.30 -> 0.0`) over 300 iterations.
   * **Result:** Re-measured Sentinel instant wins dropped from **60.0% down to 40.0%**, while rich two-piece finish races increased from **40.0% up to 60.0%**!
6. **MCTS Subtree Reuse & Root Tensor Caching:**
   * Carried forward explored child subtrees (`root = root.children[action]`) between consecutive moves, preserving accumulated visits ($N$) and search evaluations ($W$).
   * Cached root dynamic features (`node.x_dyn`), eliminating redundant CPU Dijkstra encoding passes.
7. **Calibrated Exploration & Temperature Schedule:**
   * Calibrated Dirichlet noise to $\alpha = 0.18$ ($10 / \bar{b} \approx 10 / 55$).
   * Extended exploration temperature: $\tau = 1.0$ for plies 0..14, $\tau = 0.5$ for plies 15..29, with configurable $\tau_\text{floor}$.
8. **Optimized High-Throughput Replay Buffer:**
   * Converted buffer storage to preallocated contiguous NumPy ring arrays with `float16` dynamic tensors, cutting RAM consumption by 50% (~280 MB for 40,000 samples) and enabling $O(1)$ vectorized indexing (`np.take`).
9. **Lockstep Batched Evaluation & Elo Ladder Integration:**
   * Replaced single-game serial evaluations with lockstep 40-game GPU batches featuring paired terrain symmetry (even games AI as P0, odd as P1 on identical terrains), speeding up evaluation by ~50x.
   * Integrated `EloDatabase` (`elo_system.py`) to atomically track Elo progress against frozen checkpoints and heuristic presets.

#### 2. Empirical Performance
* **Checkpoints:** `best.pt` at iteration 893, `latest.pt` at iteration 900.
* **Elo Leap:**
  * **Master (256 sims):** **1786.7** (**+133.0 Elo over `train_v4/best.pt:master`**).
  * **Standard (96 sims):** **1652.0** (+36.8 Elo over v4).
  * **Blitz (32 sims):** **1550.9** (+62.4 Elo over v4).
* **Heuristic Dominance:** Win rate against `HeuristicAgent` reached **88.75%** on 40-game symmetric batched evaluation.
* **Strategic Maturity:** Exhibits sophisticated positional squeeze play, patience in waiting for optimal Vanguard charge angles, accurate calculations of two-piece finish races, and zero susceptibility to coordinate memorization.

---

### `train_v6`: Pure Zero-Curriculum Self-Play

#### 1. What Changed from `train_v5`
`train_v6` represents the graduation of Traversal AlphaZero into **pure, unassisted self-play**:
* **Resumed State:** Resumed from `train_v5/latest.pt` (iteration 900) and inherited `train_v5`'s mature replay buffer (`replay_buffer.pt`).
* **Zero Heuristic Injection:** `--heuristic-fraction 0.0` (no heuristic moves injected at all; 100% pure AlphaZero self-play).
* **Zero MCTS Shaping:** `--shaping-weight 0.0` and `--v-tac-weight 0.0` (all search priors and leaf evaluations originate strictly from the neural network).
* **Direct Benchmark Tracking:** `--eval-opponents checkpoints/train_v5/best.pt`, playing continuous head-to-head matches against the reigning champion `train_v5/best.pt`.

#### 2. Empirical Performance (In-Flight & Benchmarks)
* **Progress:** Advanced from iteration 900 to 940+.
* **Head-to-Head vs `train_v5/best.pt`:**
  * **78 Wins, 64 Losses, 18 Draws** across 160 competitive games (**54.4% win rate** against the peak v5 champion).
* **Dominance vs `heuristic:standard`:**
  * **117 Wins, 37 Losses, 6 Draws** across 160 games (**75.0% win rate**).
* **Strategic Evolution:** Without artificial shaping anchors, the network discovers novel positional sacrifices and long-range Explorer diversion tactics that diverge from hand-crafted heuristic intuitions.

---

### `train_v7`: The Unified From-Scratch Dual-Annealed Generation

#### 1. Motivation & Paradigm Shift
`train_v7` was conceived to address the fundamental structural divide between `train_v5` and `train_v6`:
* **The Structural Divide in Prior Versions:**
  * `train_v5` established the superior GAP value head, Encoder v2, raw targets, and MCTS shaping annealing, but relied on a static 25% `HeuristicAgent` game injection across its entire 900-iteration lifetime.
  * `train_v6` demonstrated 100% pure self-play with 0% heuristic assistance and 0.0 shaping, but had to be warm-started from `train_v5/latest.pt` (iteration 900).
* **The `train_v7` Thesis:** Can a model train **from scratch** (iteration 1, random initialization) using early scaffolding (both heuristic opponents and MCTS tactical shaping), and then systematically transition into **pure, autonomous AlphaZero self-play** through synchronized dual annealing?
* **Dual Curriculum Annealing:**
  * Linearly anneals tactical MCTS prior shaping (`shaping_weight: 1.0 -> 0.0`) and the heuristic value anchor (`v_tac_weight: 0.30 -> 0.0`) over the first 300 iterations (`--mcts-anneal-iters 300`).
  * Simultaneously anneals heuristic game injection (`heuristic_fraction: 0.25 -> 0.0`) and heuristic imitation loss (`heuristic_imitation_weight: 0.50 -> 0.0`) over the same 300 iterations (`--heuristic-anneal-iters 300`).
  * Beyond iteration 300, the network operates in **100% autonomous, zero-curriculum self-play**.

#### 2. Configuration & Setup
* **Architecture:** Dual-Stem Convolutional ResNet with Global Average + Max Pooling (GAP) Value Head (`Conv(96->32) -> Mean+Max Pool(64) -> Linear(64,64) -> Linear(64,1)`) and Convolutional Policy Head (`Linear(3200, 606)`).
* **Encoder:** Versioned State Encoder v2 (35 channels: obstacle-aware Dijkstra v2, repetition indicator plane `Dyn 34`, friendly attack cleanup).
* **Search & Compute:** 12 games per iteration, 160 MCTS simulations per turn, root Dirichlet noise ($\alpha=0.18, \epsilon=0.25$), subtree reuse, temperature $\tau=1.0 \to 0.5$ (ply 30 threshold).
* **Buffer & Optimization:** Contiguous `float16` NumPy ring buffer (40k capacity), controlled sample reuse ratio (`--target-reuse 3.0`, 60 gradient steps/iter), raw terminal targets ($\pm 1.0 / 0.0$), `AdamW(lr=1e-3, weight_decay=1e-4)` with cosine annealing to $10^{-4}$ over 1,000 iterations.
* **Evaluation Protocol:** Symmetric 20-game GPU batches against `HeuristicAgent` every 5 iterations at 48 MCTS simulations with atomic Elo tracking (`elo_ratings.json`).

#### 3. Empirical Training Dynamics & Three-Phase Evolution
1. **Phase 1: Bootstrapped Exploration (Iterations 1–85):**
   * Rapid initial loss reduction from ~3.84 down to ~2.99 as basic board topologies, legal movement patterns, and captures were assimilated.
   * Replay buffer filled from 0 to 40,000 samples within ~80 iterations.
   * *Diagnostic note:* Following parameter realignment at iteration 85 (`--mcts-anneal-iters 300`), shaping and $v_\text{tac}$ were synchronized with the heuristic schedule, starting from $0.72$ and $0.21$ respectively.
2. **Phase 2: The Dual-Annealing Crucible (Iterations 86–300):**
   * As both tactical shaping ($0.72 \to 0.00$) and heuristic opponents ($18\% \to 0\%$) were dialed down, the network was forced to develop independent positional evaluation without hand-crafted crutches.
   * Win rate against `HeuristicAgent` rose steadily: 40.0% (iter 150) $\to$ 50.0% (iter 200) $\to$ 70.0% (iter 300).
   * By iteration 300, all synthetic search shaping and heuristic opponents were exactly 0.0.
3. **Phase 3: Autonomous Pure Self-Play Scaling (Iterations 301–915+):**
   * **Deepening Endgame Complexity:** With search guidance unconstrained by heuristic biases, self-play games deepened dramatically—average plies expanded from ~34 plies during Phase 1 up to **50–61 plies** in Phase 3.
   * **Training Convergence:** Total loss smoothly reached **2.31** (Policy loss: 2.24, Value loss: ~0.098, Entropy: 1.45).
   * **Record Heuristic Dominance:** At iteration 900 (`checkpoints/train_v7/best.pt`), the model achieved a peak evaluation win rate of **92.5%** against `HeuristicAgent` (37 wins out of 40 games in symmetric arena play)—setting a new project record against the baseline.
   * **Extensive Match Record:** Across 3,660 logged evaluation games against `heuristic:standard`, `train_v7` attained **2,144 Wins, 1,351 Losses, 165 Draws** (58.6% overall win rate across all training stages, peaking at Elo **1510.0** at 48 sims).

---

### `train_v8`: Fine-Tuned Grandmaster Generation

#### 1. Configuration & Advances
`train_v8` built on the Dual-Stem GAP + Encoder v2 + Dual-Annealing pipeline established in `train_v7`, extending training through a full **1,000 iterations** with refined optimization and continuous Elo arena benchmarking:
* **Architecture & Encoder:** Dual-Stem Convolutional ResNet with Global Average + Max Pooling (GAP) Value Head and Versioned State Encoder v2 (35 dynamic channels).
* **Schedule & Optimization:** Fine-tuned cosine annealing learning rate schedule (`lr_init=5e-4 -> lr_min=1e-4` over 1,000 iterations, with `AdamW` weight decay `1e-4`), `--target-reuse 3.0`, raw terminal targets ($\pm 1/0$), and 160 MCTS simulations per self-play move (12 games/iteration, 40k `float16` ring replay buffer).
* **Dual Curriculum Annealing:** Linearly annealed tactical MCTS shaping (`1.0 -> 0.0`), heuristic value anchor (`0.30 -> 0.0`), heuristic game injection (`0.25 -> 0.0`), and heuristic imitation weight (`0.50 -> 0.0`) over 300 iterations, followed by 700 iterations of autonomous pure self-play.

#### 2. Empirical Performance
* **Best Checkpoint (`checkpoints/train_v8/best.pt`):** Reached at **iteration 555** (total loss **2.2331**, minimum run loss **2.1746**), setting a new project record of **97.5% win rate** against `HeuristicAgent` (40-game symmetric evaluation).
* **Training Evaluation Record:** Across 4,000 evaluation games against `heuristic:standard`, `train_v8` scored **3,059 Wins, 182 Draws, 759 Losses** (**78.7%** overall score rate across all training stages).
* **Elo Dominance Across All Time Controls (`elo_ratings.json`):**
  * **Master (256 sims):** **1902.5 Elo** (Peak **2007.5**, 381W-98D-67L, **78.8%** score rate over 546 arena games) — **#1 overall** on the leaderboard.
  * **Standard (96 sims):** **1837.5 Elo** (Peak **1873.6**, 269W-114D-163L, **59.7%** over 546 arena games) — **#2 overall**, outrating every previous generation's Master (256-sim) model.
  * **Blitz (32 sims):** **1599.9 Elo** (Peak **1656.8**, 125W-111D-310L over 546 arena games).
* **Head-to-Head Arena Dominance (`train_v8/best.pt:master`):**
  * vs `train_v7/best.pt:master`: **41W - 15D - 28L**
  * vs `train_v6/best.pt:master`: **47W - 10D - 9L**
  * vs `train_v5/best.pt:master`: **23W - 7D - 6L**
  * vs `train_v4/best.pt:master`: **6W - 0D - 0L**

---

## Comprehensive Quantitative Comparison

### Elo Ratings Across Presets & Opponents (from `elo_ratings.json`)

| Model / Agent | Blitz (32 sims) | Standard (96 sims) | Master (256 sims) | Status / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **`train_v8/best.pt`** | **1599.9** (Peak 1656.8) | **1837.5** (Peak 1873.6) | **1902.5** (Peak **2007.5**) | **#1 Overall** (Iter 555 / 1000, **97.5%** vs Heuristic) |
| **`train_v7/best.pt`** | 1578.1 (Peak 1705.0) | 1725.4 (Peak 1814.4) | 1828.7 (Peak 1996.3) | Iter 900, Dual Anneal from scratch (92.5% vs Heuristic) |
| **`train_v6/best.pt`** | 1486.8 (Peak 1520.9) | 1487.8 (Peak 1724.7) | 1727.0 (Peak 1818.1) | Iter 1045, Pure zero-curriculum self-play |
| **`train_v5/best.pt`** | 1388.2 (Peak 1561.4) | 1506.9 (Peak 1660.8) | 1616.6 (Peak 1789.6) | Iter 893, GAP head + Encoder v2 overhaul |
| **`train_v5/latest.pt`** | 1420.5 | 1532.4 | 1656.6 | Iter 900 checkpoint |
| **`train_v4/best.pt`** | 1365.4 (Peak 1575.4) | 1460.6 (Peak 1681.7) | 1540.5 (Peak 1787.7) | Peak legacy model (Iter 995, Flat head, Encoder v1) |
| **`train_v3/best.pt`** | 1361.1 (Peak 1513.2) | 1380.2 (Peak 1619.4) | 1482.8 (Peak 1627.9) | Iter 851 checkpoint |
| **`train_v2/best.pt`** | 1379.6 (Peak 1534.9) | 1365.0 | 1311.1 | Iter 674 checkpoint |
| **`train_v1/best.pt`** | 1186.8 | 1236.0 | 1265.6 | Initial proof-of-concept (Iter 184) |
| *Heuristic Agent (baseline)* | 1241.3 (2-ply) | 1319.4 (4-ply) | 1536.2 (6-ply) / 1476.5 (8-ply GM) | Classical Alpha-Beta search with iterative deepening |

### Head-to-Head Match Matrix (Key Recorded Matches)

```
                       vs train_v7/best:m  vs train_v6/best:m  vs train_v5/best:m  vs heuristic:std
train_v8/best:master   41W - 15D - 28L     47W - 10D - 9L      23W - 7D - 6L       --
train_v8 (Eval Run)    --                  --                  --                  3,059W - 182D - 759L
train_v7 (Eval Run)    --                  --                  --                  2,189W - 165D - 1,366L
train_v6 (Iter 940+)   --                  --                  78W - 18D - 64L     117W - 6D - 37L
train_v5 (Eval Run)    --                  --                  --                  14,861W - 5,517D - 15,622L
```

---

## Architectural & Algorithmic Milestones Summary

```
train_v1 ──► train_v2 ──► train_v3 ──► train_v4 ──► train_v5 ──► train_v6 ──► train_v7 ──► train_v8
(Baseline)   (Data Scale) (Long Horiz) (Peak Legacy)(GAP Overhaul)(Pure Self-Play)(Dual Anneal)(Fine-Tuned GM)
• Flat Head  • Buffer 55k • 850+ iters • 996 iters  • GAP Head   • 0% Heuristic • From-Scratch • 1,000 iters
• v1 Encoder • 160 Sims   • Stability  • Loss 1.83  • Encoder v2 • 0.0 Shaping  • Dual Anneal  • LR 5e-4->1e-4
• 25% Heur.  • 128 Batch  • 1483 Elo   • 1541 Elo   • 1617 Elo   • 1727 Elo     • 1829 Elo     • 1903 Elo
• 1266 Elo   • 1311 Elo                [Overfit Wall] (Pk 1790)    (Pk 1818)      (Pk 1996)      (Peak 2007.5)
```

1. **Spatial Invariance Was Decisive:** The transition from a flattened linear value head (`train_v4`) to Global Pooling (`train_v5`) was the single highest-impact architectural change, turning a memorizing model into a generalising positional evaluator.
2. **Search / Target Consistency:** Shifting value targets from heuristic-shaped scores to raw game outcomes ($\pm 1 / 0$) aligned the policy-value network with the true game-theoretic payoff of MCTS terminal leaves.
3. **Annealing Over Static Anchors:** Hand-crafted tactical priors are effective for early bootstrapping, but must be annealed to 0.0; otherwise, they induce pathological biases (such as the 60% Sentinel rush distortion observed in v4).
4. **MCTS Subtree Reuse & Tensor Caching:** Propagating explored subtrees across plies yielded deeper effective searches at constant compute, while caching root dynamic tensors eliminated redundant Dijkstra graph computations.
5. **Autonomy Achieved:** `train_v6` confirms that once a mature representation is seeded, removing all heuristic scaffolding allows the AlphaZero self-play engine to surpass hand-crafted play.
6. **From-Scratch Autonomy & Peak Grandmaster Scaling:** `train_v7` and `train_v8` validate that AlphaZero can bootstrap from scratch with dual-annealed scaffolding over 300 iterations, then scale cleanly in pure self-play to **1902.5 Master Elo (Peak 2007.5)** and a **97.5%** evaluation win rate against `HeuristicAgent`.
