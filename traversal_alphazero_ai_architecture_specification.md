# Technical Specification: Traversal Game Engine & AlphaZero AI

## 1. Project Overview & System Goals
The objective of this project is to implement a fast, headless Python recreation of the abstract 3D strategy board game **Traversal** (https://www.traversalgame.com/rules) and train a game-playing AI using **AlphaZero-style Monte Carlo Tree Search (MCTS)** paired with a **Two-Block Dual-Stem Policy-Value Residual Network**.

### Core Engineering Constraints for the Coding Agent
1. **Pure Python Headless Core:** Do not use heavy 3D game engines (Unity/Unreal). Implement the core game state, move generator, and state tensor encoder in pure Python/NumPy (optionally accelerated with `numba` `@njit` or `jax` depending on local hardware).
2. **Hardware-Adaptive Stack:** Inspect the local machine (CPU cores, RAM, and available GPU/CUDA/MPS support) to choose appropriate batch sizes, number of MCTS simulations per turn, and parallel self-play workers.
3. **Decoupled UI:** Keep the training and MCTS simulation loop 100% headless. If a UI is requested for human play or debugging, build a lightweight separate module (e.g., `pygame` or a local web UI) that imports the headless `TraversalGame` class.

---

## 2. Exact Rules of Traversal

### 2.1 Board & Terrain Generation
* **Grid:** A $10 \times 10$ grid of integer terrain heights $h(r, c) \ge 0$, where rows and columns are indexed $0 \dots 9$.
* **Slope Constraint:** Orthogonally adjacent squares differ in height by at most $2$ units:
  $$|h(r_1, c_1) - h(r_2, c_2)| \le 2$$
  This guarantees there are no impassable cliffs on an empty board.
* **Terrain Modes:** Every match generates a procedural terrain (e.g., *Random*, *Mountains*, *Valleys*, *Highlands*, *Woodlands*, *Desert*) designed so neither player has a significant structural advantage.

### 2.2 Pieces & Starting Setup
Each player commands $6$ pieces ($2$ of each of the $3$ types). For every piece type, $\text{Movement} + \text{Strength} = 6$:

| Piece Type | Count per Player | Movement ($M_i$) | Base Strength ($S_i$) | Score Value | Special Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Explorer** | $2$ | $5$ | $1$ | $1\text{ pt}$ | **Special Move:** From its starting position, may instead move $1$ square diagonally OR leap $2$ squares orthogonally in a single cardinal direction, ignoring height differences (uses full turn; cannot combine with normal movement). **Note:** Explorer special moves (diagonal step and orthogonal leap) are strictly movement abilities to empty squares and cannot land on an enemy piece to capture it. All captures must be reachable via standard orthogonal movement steps. |
| **Vanguard** | $2$ | $4$ | $2$ | $2\text{ pts}$ | **Charge Bonus:** Gains $+1$ Attack Strength when the shortest path from its starting square to the target uses all $4$ of its Movement ($d_{\min} = 4$). |
| **Sentinel** | $2$ | $3$ | $3$ | $3\text{ pts}$ | **Instant Victory:** If a Sentinel is the **very first piece** (of either side) to finish off the board, its owner wins the game immediately. |

* **Starting Positions:**
  * **Player 1:** Starts on Row $0$, Columns $2 \dots 7$ (moving toward Row $9$).
  * **Player 2:** Starts on Row $9$, Columns $2 \dots 7$ (moving toward Row $0$).
  * Pieces are arranged symmetrically across Columns $2 \dots 7$: `[Sentinel, Vanguard, Explorer, Explorer, Vanguard, Sentinel]`.
  * Maintain a fixed canonical ordering of each player's $6$ pieces throughout the codebase:
    `[Explorer_0, Explorer_1, Vanguard_0, Vanguard_1, Sentinel_0, Sentinel_1]`.

### 2.3 Movement Rules
* On a turn, a player selects **one** of their active pieces and moves it.
* **Standard Movement:** Orthogonal steps (up, down, left, right). Moving from $(r_1, c_1)$ to an adjacent square $(r_2, c_2)$ costs:
  $$\text{Step Cost} = |h(r_2, c_2) - h(r_1, c_1)| + 1$$
  *(Flat ground costs $1$; elevation change of $\pm 1$ costs $2$; elevation change of $\pm 2$ costs $3$).*
* **Blocking:** Pieces **cannot move through** any occupied square (friendly or enemy), except during an Explorer's $2$-square orthogonal leap. A piece can never land on a friendly piece.
* **No Null Move:** A piece **cannot end its turn on the square it started from** ($d = 0$ is illegal).
* **Explorer Special Movement:** Instead of standard orthogonal movement, an Explorer may take a standalone move from its starting square $(r_0, c_0)$:
  1. **Diagonal Step:** Move $1$ square diagonally ($\Delta r = \pm 1, \Delta c = \pm 1$) regardless of height difference.
  2. **Orthogonal Leap:** Jump $2$ squares in a single cardinal direction ($\Delta r = \pm 2, \Delta c = 0$ or $\Delta r = 0, \Delta c = \pm 2$) regardless of height difference.

### 2.4 Combat Rules
* A piece may move onto an enemy-occupied square $(r_t, c_t)$ to capture it if and only if the enemy square is reachable within the piece's movement rules **and** the attacker's effective strength is greater than or equal to the defender's effective strength:
  $$\text{Attack}_i(r_t, c_t) \ge \text{Defense}(r_t, c_t)$$
* **Illegal Failed Attacks:** If $\text{Attack}_i(r_t, c_t) < \text{Defense}(r_t, c_t)$, or if target $(r_t, c_t)$ is reachable only via an Explorer's special move (diagonal step or orthogonal leap), the capture is **illegal** and must be masked out.
* **Effective Attack Formula:** For attacking piece $i$ starting its turn at $(r_0, c_0)$ and attacking target square $(r_t, c_t)$:
  $$\text{Attack}_i(r_t, c_t) = S_i + \text{HeightMod}(r_0, c_0 \to r_t, c_t) + \text{VanguardBonus}_i(r_t, c_t) + \sum_{\substack{\text{friendly } j \text{ adj. to } (r_t, c_t) \\ j \neq i}} S_j$$
  * **Height Modifier:** Determined by comparing the attacker's **starting square at the beginning of the turn** $(r_0, c_0)$ against the target square $(r_t, c_t)$:
    * $+1$ if $h(r_0, c_0) > h(r_t, c_t)$ (attacking downhill).
    * $-1$ if $h(r_0, c_0) < h(r_t, c_t)$ (attacking uphill).
    * $0$ if $h(r_0, c_0) = h(r_t, c_t)$.
  * **Vanguard Charge Bonus:** $+1$ if piece $i$ is a Vanguard and the **shortest valid movement cost** $d_{\min}\big((r_0, c_0) \to (r_t, c_t)\big) = 4$.
  * **Allied Support:** Each friendly piece $j \neq i$ currently orthogonally adjacent to $(r_t, c_t)$ adds its **base strength $S_j$** to the attack. *(Crucial edge case: exclude the attacking piece $i$ itself if $i$ started the turn already adjacent to $(r_t, c_t)$).*
* **Effective Defense Formula:** For a piece at $(r_t, c_t)$:
  $$\text{Defense}(r_t, c_t) = S_{\text{defender}} + \sum_{\text{friendly } k \text{ adj. to } (r_t, c_t)} 1$$
  *(Each friendly piece orthogonally adjacent to the defender adds $+1$ to defense, regardless of piece type).*
* Captured pieces are permanently removed from the board. Attacking ends the turn immediately.

### 2.5 Scoring, Finishing & Victory Conditions
* **Finishing (Stepping Off the Board):**
  * To finish, a piece must be on the final row (Row $9$ for the current canonical player) and have at least $1$ Movement remaining to step off the board.
  * Specifically, a piece starting at $(r_0, c_0)$ can finish this turn if it starts on Row $9$, OR if it can reach any empty square on Row $9$ using at most $M_i - 1$ movement points ($d_{\min} \le M_i - 1$).
  * When a piece finishes, it is removed from the board and its base strength $S_i$ is added to its owner's score.
* **Game Termination & Winner Determination:**
  1. **Sentinel Instant Win:** If the **first piece of the entire game** to finish is a Sentinel, that player wins immediately ($z = +1$).
  2. **Two-Piece Finish:** The game ends immediately as soon as **any $2$ pieces total** (from either or both players) have finished. The player with the higher total score wins; equal scores result in a draw ($z = 0$).
  3. **Elimination / No Legal Moves:** If a player has no remaining pieces on the board (or zero legal moves on their turn), the game ends; compare current scores (or remaining piece progress if tied at $0$).
  4. **Threefold Repetition Draw:** If the exact same position (same piece coordinates, statuses, and player to move) is reached for the third time in a match, the game immediately ends in a draw ($z = 0$, `winner = -1`). In MCTS, this prevents winning players from repeating moves (since expected reward drops to $0.0$) while allowing players at a disadvantage to seek repetitions to secure a draw.
  5. **200-Move Limit Tiebreaker:** If the game reaches $200$ total plies (half-moves):
     * Higher score wins.
     * If neither player has scored, the player whose on-board piece is closest to finishing wins (counting the rows remaining to cross).
     * If both players' closest pieces are equally close, the stronger piece ($S_i$) wins.
     * If also equally strong, the game is a draw ($z = 0$).

---

## 3. Canonical Perspective & Input Tensor Specification

### 3.1 Canonical Board Orientation
Always encode the board from the **current player's perspective**:
* The current player always moves from **Row $0$ (bottom) toward Row $9$ (top)**.
* When it is Player 2's turn, rotate all $10 \times 10$ spatial grids by $180^\circ$ (`np.rot90(grid, 2)` or `grid[::-1, ::-1]`) and swap "my" and "opponent" channels before feeding them to the neural network. Also transform destination square indices $(r, c) \mapsto (9 - r, 9 - c)$ when mapping actions between the policy vector and the physical board.

### 3.2 Two-Block Input Split (Static vs. Dynamic Tensors)
Because the terrain never changes during a game while pieces move every turn, split the neural network inputs into two tensors:
1. **`X_static` of shape $(4, 10, 10)$:** Computed only twice at Move $0$ (once for Player 1's view, once rotated $180^\circ$ for Player 2's view).
2. **`X_dynamic` of shape $(34, 10, 10)$:** Computed at every board state.

#### Block 1: Static Terrain Input (`X_static`, Shape: $4 \times 10 \times 10$)

| Channel | Name | Mathematical Definition & Normalization |
| :--- | :--- | :--- |
| **`Static 0`** | `terrain_height` | **Median-centered, fixed-scale height:** $$h_{\text{norm}}(r, c) = \frac{h(r, c) - \text{median}(h)}{5.0}$$ *Rationale:* Subtracting the board's median makes the representation invariant to isolated corner spikes, while dividing by a universal constant $5.0$ (instead of per-board $h_{\max}$) ensures $\Delta h = 1$ always equals $0.20$ across all boards. |
| **`Static 1`** | `my_unblocked_dist_std` | Empty-board Dijkstra minimum movement cost from $(r, c)$ to step off Row $9$ using standard movement, divided by $30.0$. |
| **`Static 2`** | `my_unblocked_dist_exp` | Empty-board Dijkstra minimum turns from $(r, c)$ to step off Row $9$ using Explorer movement (including diagonal steps and $2$-tile orthogonal leaps), divided by $10.0$. |
| **`Static 3`** | `opp_unblocked_dist_std` | Empty-board Dijkstra minimum movement cost from $(r, c)$ to step off Row $0$ (opponent's finish direction), divided by $30.0$. |

#### Block 2: Dynamic Board Input (`X_dynamic`, Shape: $34 \times 10 \times 10$)

| Channels | Name | Encoding Specification |
| :--- | :--- | :--- |
| **`Dyn 0..5`** | `my_pos[0..5]` | **Pure Binary $\{0.0, 1.0\}$:** $6$ planes (ordered `[E0, E1, V0, V1, S0, S1]`). Set to $1.0$ at Piece $i$'s current square $(r_0, c_0)$, and $0.0$ everywhere else (all $0.0$ if captured or finished). |
| **`Dyn 6..11`** | `opp_pos[0..5]` | **Pure Binary $\{0.0, 1.0\}$:** $6$ planes for the opponent's $6$ pieces. |
| **`Dyn 12..17`** | `my_reach[0..5]` | **Pure Binary $\{0.0, 1.0\}$ Legal Destination + Exit Flag:** For each friendly piece $i$: <br>1. Set $(r, c)$ to $1.0$ if Piece $i$ can **legally move to or legally capture** $(r, c)$ this turn (already filtering out failed attacks where $\text{Attack}_i < \text{Defense}$).<br>2. **Origin Exit Flag:** Because a piece can never end its turn on its starting square $(r_0, c_0)$, pixel $(r_0, c_0)$ is unused for on-board moves. Set `my_reach[i][r_0, c_0] = 1.0` **if and only if** Piece $i$ can legally step off the board to finish this turn! Otherwise set to $0.0$. |
| **`Dyn 18..23`** | `opp_reach[0..5]` | **Pure Binary $\{0.0, 1.0\}$:** Exact same legal destination and origin-exit-flag encoding computed as if it were the opponent's turn to move. |
| **`Dyn 24`** | `my_max_attack` | For every tile $(r, c)$ (both empty and occupied), compute $\max_i \text{Attack}_i(r, c) / 10.0$ across all friendly pieces $i$ that can reach $(r, c)$ this turn (incorporating starting height $\pm 1$, Vanguard $d_{\min}=4$ bonus $+1$, and adjacent allies $j \neq i$). Set to $0.0$ if no friendly piece can reach $(r, c)$. |
| **`Dyn 25`** | `my_defense` | For every tile $(r, c)$ occupied by a friendly piece, $\text{Defense}(r, c) / 7.0$; $0.0$ elsewhere. |
| **`Dyn 26`** | `opp_max_attack` | Opponent's maximum effective attack projected onto every tile $(r, c)$ divided by $10.0$. |
| **`Dyn 27`** | `opp_defense` | Opponent's effective defense on every opponent-occupied tile divided by $7.0$. |
| **`Dyn 28`** | `my_blocked_dist` | Dynamic Dijkstra minimum movement cost from $(r, c)$ to step off Row $9$ treating all currently occupied squares as impassable walls (clamped to $30.0$ and divided by $30.0$). |
| **`Dyn 29`** | `opp_blocked_dist` | Dynamic Dijkstra minimum movement cost from $(r, c)$ to step off Row $0$ treating all currently occupied squares as impassable walls (clamped to $30.0$ and divided by $30.0$). |
| **`Dyn 30..33`** | `scalar_planes` | $4$ constant $10 \times 10$ planes filled with: <br>• `Dyn 30`: `my_score / 6.0`<br>• `Dyn 31`: `opp_score / 6.0`<br>• `Dyn 32`: `total_pieces_finished / 2.0`<br>• `Dyn 33`: `ply_count / 200.0` |

> **Why Pure Binary $\{0, 1\}$ for `pos` and `reach`:**
> Intra-turn movement points spent ($d \in \{1 \dots M_i\}$) do not affect the post-move board state for non-capturing moves, and the Vanguard's $d_{\min} = 4$ charge bonus is already explicitly captured in `my_reach` (via legal capture filtering on occupied squares) and in `my_max_attack` / `opp_max_attack` (on empty squares). Keeping `pos` and `reach` strictly binary creates a sharp $1.0 \to 0.0$ step discontinuity at the movement boundary and avoids any collision with $0.0$ convolutional zero-padding outside the $10 \times 10$ board.

---

## 4. Action Space & Global Masking ($6 \times 101 = 606$ Actions)

### 4.1 Discrete Action Encoding
The action space is a flat discrete vector of size **$606$**, structured conceptually as **$6$ pieces $\times 101$ targets**:
* Given an action index $a \in \{0 \dots 605\}$:
  * **Piece Slot:** $k = a \pmod{101}$ where $k = \lfloor a / 101 \rfloor \in \{0 \dots 5\}$ corresponds to `[E0, E1, V0, V1, S0, S1]`.
  * **Target Index:** $t = a \bmod 101 \in \{0 \dots 100\}$:
    * If $0 \le t \le 99$: Move Piece $k$ to canonical board square $(r, c) = (\lfloor t / 10 \rfloor, t \bmod 10)$.
    * If $t = 100$: Piece $k$ **steps off Row $9$ to finish and score**.

### 4.2 Flattened Action Masking Rule (Critical)
* At any state $S$, the engine produces a boolean legal move mask `legal_mask` of shape $(606,)$ (flattened from $(6, 101)$).
* **Never apply Softmax along the $101$ dimension independently per piece.** Instead:
  1. Take the raw policy logits `logits` of shape $(606,)$.
  2. Mask illegal actions: `masked_logits = torch.where(legal_mask, logits, -1e8)`.
  3. Apply a **single Softmax across all $606$ logits** so that $\sum_{a=0}^{605} P(a) = 1.0$.

---

## 5. Two-Block Dual-Stem Neural Network Architecture

```
X_static (B, 4, 10, 10)                 X_dynamic (B, 34, 10, 10)
          │                                         │
  [Static Conv Stem]                        [Dynamic Conv Stem]
  Conv3x3(4 -> 32) + BN + ReLU              Conv3x3(34 -> 64) + BN + ReLU
  2 × ResBlock(32 channels)                 2 × ResBlock(64 channels)
          │                                         │
   F_static (B, 32, 10, 10)                  F_dyn (B, 64, 10, 10)
          └───────────────────┬─────────────────────┘
                              ▼
                  Concatenate along channel dim
                     F_joint (B, 96, 10, 10)
                              │
                     [Shared ResNet Trunk]
                  Conv1x1(96 -> 96) + BN + ReLU
                  4 × ResBlock(96 channels)
                        /           \
                       ▼             ▼
              Policy Head (π)    Value Head (v)
               (B, 606) logits    (B, 1) in [-1, 1]
```

### 5.1 Architectural Specifications
1. **Static Stem (`encode_static(X_static)`):**
   * `Conv2d(4, 32, kernel_size=3, padding=1, bias=False)` $\to$ `BatchNorm2d(32)` $\to$ `ReLU`
   * $2$ Residual Blocks with $32$ channels.
   * Output: `F_static` of shape $(B, 32, 10, 10)$.
2. **Dynamic Stem (`encode_dynamic(X_dynamic)`):**
   * `Conv2d(34, 64, kernel_size=3, padding=1, bias=False)` $\to$ `BatchNorm2d(64)` $\to$ `ReLU`
   * $2$ Residual Blocks with $64$ channels.
   * Output: `F_dyn` of shape $(B, 64, 10, 10)$.
3. **Fusion & Shared Trunk (`forward_trunk(F_static, F_dyn)`):**
   * Concatenate `F_joint = torch.cat([F_static, F_dyn], dim=1)` $\to$ shape $(B, 96, 10, 10)$.
   * $1 \times 1$ fusion convolution `Conv2d(96, 96, kernel_size=1, bias=False)` $\to$ `BatchNorm2d(96)` $\to$ `ReLU`.
   * $4$ Residual Blocks with $96$ channels.
4. **Policy Head:**
   * `Conv2d(96, 32, kernel_size=1, bias=False)` $\to$ `BatchNorm2d(32)` $\to$ `ReLU`
   * Flatten $(B, 32 \times 10 \times 10 = 3200)$ $\to$ `Linear(3200, 606)` outputting raw logits $\vec{l}$ of shape $(B, 606)$.
5. **Value Head:**
   * `Conv2d(96, 8, kernel_size=1, bias=False)` $\to$ `BatchNorm2d(8)` $\to$ `ReLU`
   * Flatten $(B, 8 \times 10 \times 10 = 800)$ $\to$ `Linear(800, 64)` $\to$ `ReLU` $\to$ `Linear(64, 1)` $\to$ `Tanh()` outputting scalar $v \in [-1, 1]$.

### 5.2 Inference Caching vs. Joint Training Protocol
* **During MCTS Self-Play (Inference Mode):**
  * At the start of each game (Move $0$), call `F_static_p1 = model.encode_static(X_static_p1)` and `F_static_p2 = model.encode_static(X_static_p2)` inside `torch.no_grad()` and cache both tensors on the game/MCTS runner.
  * For every MCTS leaf evaluation during that game, pass the cached `F_static` directly into `model.forward_with_cached_static(F_static, X_dynamic)`, completely bypassing the Static Stem across all $200 \times N_{\text{sims}}$ evaluations.
* **During Gradient Updates (Training Mode):**
  * Store raw `(X_static, X_dynamic, pi_mcts, z)` tuples in the Replay Buffer (since `X_static` is the same for all turns of a given game, you can store `X_static` once per game ID to save RAM).
  * During `loss.backward()`, run the full forward pass (`encode_static` + `encode_dynamic` + `forward_trunk`) so gradients flow end-to-end into **both** the Static Stem and the Dynamic Stem simultaneously.

---

## 6. AlphaZero MCTS & Self-Play Training Pipeline

### 6.1 MCTS Node & PUCT Selection
Each node $s$ in the search tree stores:
* Prior probability $P(s, a)$ from the masked Policy Head output.
* Visit count $N(s, a)$ initialized to $0$.
* Total action value $W(s, a)$ initialized to $0.0$.
* Mean action value $Q(s, a) = W(s, a) / N(s, a)$ (or $0.0$ if $N(s, a) = 0$).

At each simulation step:
1. **Root Dirichlet Noise:** At the root node $s_{\text{root}}$, mix Dirichlet noise into the prior probabilities of legal moves to ensure exploration:
   $$P(s_{\text{root}}, a) \leftarrow (1 - \varepsilon) P(s_{\text{root}}, a) + \varepsilon \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$
   with $\varepsilon = 0.25$ and $\alpha = 0.3$ (scaled inversely with the average number of legal moves, typically $\sim 25\text{--}35$ in Traversal).
2. **PUCT Selection:** Among legal actions $a$, select the action maximizing:
   $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
   with $c_{\text{puct}} \approx 1.5$.
3. **Leaf Expansion & Evaluation:** When reaching an unexpanded state $s_{\text{leaf}}$:
   * If $s_{\text{leaf}}$ is terminal, obtain the exact game outcome $v \in \{-1, 0, +1\}$ from the perspective of the player to move at $s_{\text{leaf}}$.
   * Otherwise, evaluate `(F_static_cached, X_dynamic)` with the neural network to get masked probabilities $\vec{P}(s_{\text{leaf}})$ and value $v \in [-1, 1]$.
4. **Backpropagation:** Walk back up the search path to $s_{\text{root}}$, incrementing $N(s, a) \leftarrow N(s, a) + 1$, updating $W(s, a) \leftarrow W(s, a) + v$, and **flipping the sign of $v \leftarrow -v$** at every parent step because Traversal is a two-player zero-sum alternating-turn game.

### 6.2 Move Selection & Horizontal Symmetry Augmentation
* **Temperature Schedule ($\tau$):**
  * For the first $30$ plies of a self-play game, set temperature $\tau = 1.0$ and sample the move proportionally to visit counts: $\pi_{\text{MCTS}}(a) = \frac{N(s_{\text{root}}, a)^{1/\tau}}{\sum_b N(s_{\text{root}}, b)^{1/\tau}}$.
  * After ply $30$ (and during evaluation matches), set $\tau \to 0$ (pick $\arg\max_a N(s_{\text{root}}, a)$ deterministically).
* **Free $2\times$ Data Augmentation (Left-Right Mirroring):**
  * Because Traversal's rules and starting setup are symmetric under horizontal reflection (flipping columns $c \leftrightarrow 9 - c$ and swapping the pairs of identical pieces `E0 <-> E1`, `V0 <-> V1`, `S0 <-> S1`), every recorded self-play state `(X_static, X_dynamic, pi_mcts, z)` can be horizontally mirrored when added to the replay buffer, doubling sample efficiency for free.

### 6.3 Loss Function & Optimizer
Train the network on mini-batches of $(X_{\text{static}}, X_{\text{dynamic}}, \vec{\pi}_{\text{MCTS}}, z)$ sampled uniformly from the Replay Buffer:
$$\mathcal{L}(\theta) = \underbrace{(z - v_\theta)^2}_{\text{MSE Value Loss}} - \underbrace{\sum_{a \in \text{legal}} \pi_{\text{MCTS}}(a) \log p_\theta(a)}_{\text{Cross-Entropy Policy Loss}} + \underbrace{c \|\theta\|_2^2}_{\text{L2 Weight Decay}}$$
* Use `AdamW` with learning rate $10^{-3}$ (cosine annealed to $10^{-4}$) and weight decay $c = 10^{-4}$.
* Compute the policy log-softmax strictly over legal actions (`masked_logits`) so no numerical underflow or illegal-action gradient leakage occurs.

---

## 7. Recommended Implementation Milestones for the Coding Agent
1. **Milestone 1 (`traversal_env.py`):** Implement the headless `TraversalGame` class with procedural terrain generation, Dijkstra movement/reachability, combat math (including Vanguard $d_{\min}=4$ and self-ally exclusion), finishing rules, and unit tests verifying all edge cases.
2. **Milestone 2 (`encoder.py`):** Implement `encode_static_tensor()` ($4 \times 10 \times 10$), `encode_dynamic_tensor()` ($34 \times 10 \times 10$), and the $606$-element action/mask mapping with canonical $180^\circ$ perspective rotation.
3. **Milestone 3 (`model.py`):** Implement the PyTorch `TraversalDualStemNet` with both full joint forward passes (for training) and cached static forward passes (for MCTS inference).
4. **Milestone 4 (`mcts.py` & `train.py`):** Implement the MCTS tree search, self-play worker loop, horizontal symmetry augmentation, replay buffer, and evaluation arena (gating new checkpoints against previous checkpoints or a heuristic baseline).