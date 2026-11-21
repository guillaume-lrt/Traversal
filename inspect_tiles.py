"""
Inspect Tile Strengths & Static Stem Spatial Analysis for Traversal AlphaZero.

Analyzes how the AlphaZero Dual-Stem neural network evaluates the 10x10 board:
1. Static Stem Inputs: Elevation, Dijkstra distances to finish lines, topological advantage.
2. Static Stem Latent Features (F_static, 32 channels): Activation L2-norm and PC1.
3. Positional Value Probes: Counterfactual value V(s) with piece on square (r, c).
4. Policy Landing Prior: Turn 1 probability distribution across board tiles.

Usage:
    .venv/bin/python inspect_tiles.py --terrain Mountains --layer all
    .venv/bin/python inspect_tiles.py --terrain Highlands --layer sentinel_val
    .venv/bin/python inspect_tiles.py --html /tmp/interactive_board.html
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any
import numpy as np
import torch

from traversal_env import (
    TraversalGame,
    TERRAIN_MODES,
    PIECE_NAMES,
    PIECE_TYPES,
    NUM_TARGETS,
)
from encoder import encode_both_static_tensors, encode_dynamic_tensor
from model import load_model_from_checkpoint, masked_softmax


def analyze_board(
    model: torch.nn.Module,
    terrain_mode: str = "Mountains",
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs complete static stem and value/policy analysis on a procedural board."""
    terrain_mode_lower = terrain_mode.lower().strip()
    game = TraversalGame(terrain_mode=terrain_mode_lower, seed=seed)
    terrain = game.terrain
    x_static_p1, _ = encode_both_static_tensors(terrain)

    model.eval()
    with torch.no_grad():
        t_static = torch.from_numpy(x_static_p1).to(device)
        f_static = model.encode_static(t_static)
        f_static_np = f_static.cpu().numpy()

        # 1. Static Stem L2 Norm
        l2_norm = np.linalg.norm(f_static_np, axis=0)

        # 2. Static Stem PC1 (Dominant learned latent spatial dimension)
        flat = f_static_np.reshape(32, 100).T
        centered = flat - flat.mean(axis=0)
        u, _, _ = np.linalg.svd(centered, full_matrices=False)
        pc1 = u[:, 0].reshape(10, 10)

        # 3. Dynamic start state policy
        x_dyn = encode_dynamic_tensor(game, version=getattr(model, "encoder_version", "v2"))
        legal_mask = game.get_legal_mask()
        logits, base_val = model.forward_with_cached_static(
            f_static, torch.from_numpy(x_dyn).to(device)
        )
        probs = masked_softmax(logits, legal_mask).cpu().numpy()

        # Destination landing distribution
        policy_tiles = np.zeros((10, 10), dtype=np.float32)
        for k in range(6):
            for r in range(10):
                for c in range(10):
                    policy_tiles[r, c] += probs[k * NUM_TARGETS + (r * 10 + c)]

        # 4. Positional Value Probes for Sentinel, Vanguard, Explorer
        piece_slots = {"Sentinel": 4, "Vanguard": 2, "Explorer": 0}
        value_maps = {}
        for pname, pslot in piece_slots.items():
            vgrid = np.zeros((10, 10), dtype=np.float32)
            for r in range(10):
                for c in range(10):
                    tg = TraversalGame(terrain_mode=terrain_mode, seed=seed)
                    tg.pos[0, pslot] = [r, c]
                    xd = encode_dynamic_tensor(
                        tg, version=getattr(model, "encoder_version", "v2")
                    )
                    _, val = model.forward_with_cached_static(
                        f_static, torch.from_numpy(xd).to(device)
                    )
                    vgrid[r, c] = float(val.item())
            value_maps[pname] = vgrid

    return {
        "terrain_mode": terrain_mode,
        "seed": seed,
        "terrain": terrain,
        "my_dist": x_static_p1[1] * 30.0,
        "opp_dist": x_static_p1[3] * 30.0,
        "topo_adv": (x_static_p1[3] - x_static_p1[1]) * 30.0,
        "static_l2_norm": l2_norm,
        "static_pc1": pc1,
        "policy_priors": policy_tiles,
        "value_maps": value_maps,
        "base_val": float(base_val.item()),
    }


def print_grid(
    title: str, grid: np.ndarray, fmt: str = "{:+5.2f}", row_order: str = "descending"
) -> None:
    """Prints a 10x10 grid with clear row indices and orientation markers."""
    print(f"\n=== {title} ===")
    print("    " + " ".join(f" c{c:<3}" for c in range(10)))
    print("    " + "-" * 49)
    rows = range(9, -1, -1) if row_order == "descending" else range(10)
    for r in rows:
        row_str = " ".join(fmt.format(grid[r, c]) for c in range(10))
        label = ""
        if r == 9:
            label = " <-- P1 Goal (Exit)"
        elif r == 0:
            label = " <-- P1 Start Row"
        print(f"r{r}: {row_str}{label}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tile strength evaluations in Traversal AlphaZero."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/train_v5/best.pt",
        help="Path to model checkpoint (default: checkpoints/train_v5/best.pt)",
    )
    parser.add_argument(
        "--terrain",
        type=str.lower,
        default="mountains",
        choices=[m.lower() for m in TERRAIN_MODES],
        help="Terrain mode (default: mountains)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Procedural terrain seed (default: 42)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all",
        choices=[
            "all",
            "terrain",
            "dijkstra",
            "topo_adv",
            "static_norm",
            "static_pc1",
            "sentinel_val",
            "vanguard_val",
            "explorer_val",
            "policy_prior",
        ],
        help="Which analytical layer to display",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device ('cpu' or 'cuda')"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        # Look for fallback checkpoints
        fallbacks = [
            "checkpoints/train_v5/latest.pt",
            "checkpoints/train_v4/best.pt",
            "checkpoints/train_v3/best.pt",
        ]
        found = False
        for fb in fallbacks:
            if os.path.isfile(fb):
                print(f"[inspect_tiles] '{args.checkpoint}' not found, using fallback: {fb}")
                args.checkpoint = fb
                found = True
                break
        if not found:
            sys.exit(f"Error: checkpoint '{args.checkpoint}' not found.")

    print(f"Loading checkpoint '{args.checkpoint}' on {args.device}...")
    model, meta = load_model_from_checkpoint(args.checkpoint, device=args.device)
    print(
        f"Model loaded: version={model.encoder_version}, value_head={model.value_head_type}, dynamic_channels={model.dynamic_channels}"
    )

    data = analyze_board(
        model, terrain_mode=args.terrain, seed=args.seed, device=args.device
    )
    print(f"Board analyzed: mode='{args.terrain}', seed={args.seed}")
    print(f"Move 0 Root Position Evaluation V(s): {data['base_val']:+.4f}")

    if args.layer in ("all", "terrain"):
        print_grid(
            f"Terrain Heights h(r, c) [{args.terrain}]",
            data["terrain"],
            fmt="{:4d}",
            row_order="descending",
        )

    if args.layer in ("all", "dijkstra"):
        print_grid(
            "P1 Unblocked Standard Dijkstra Steps to Row 9",
            data["my_dist"],
            fmt="{:5.1f}",
            row_order="descending",
        )
        print_grid(
            "P2 Unblocked Standard Dijkstra Steps to Row 0",
            data["opp_dist"],
            fmt="{:5.1f}",
            row_order="descending",
        )

    if args.layer in ("all", "topo_adv"):
        print_grid(
            "Topological Distance Advantage (P2 Dijkstra - P1 Dijkstra)",
            data["topo_adv"],
            fmt="{:+5.1f}",
            row_order="descending",
        )

    if args.layer in ("all", "static_norm"):
        print_grid(
            "Static Stem Feature L2 Norm (32 latent channels per tile)",
            data["static_l2_norm"],
            fmt="{:5.1f}",
            row_order="descending",
        )

    if args.layer in ("all", "static_pc1"):
        print_grid(
            "Static Stem Principal Component 1 (Dominant learned terrain feature)",
            data["static_pc1"],
            fmt="{:+5.2f}",
            row_order="descending",
        )

    if args.layer in ("all", "sentinel_val"):
        print_grid(
            "Positional Value V(s) Probe: Sentinel on Tile (Instant-Win Threat)",
            data["value_maps"]["Sentinel"],
            fmt="{:+5.2f}",
            row_order="descending",
        )

    if args.layer in ("all", "vanguard_val"):
        print_grid(
            "Positional Value V(s) Probe: Vanguard on Tile",
            data["value_maps"]["Vanguard"],
            fmt="{:+5.2f}",
            row_order="descending",
        )

    if args.layer in ("all", "explorer_val"):
        print_grid(
            "Positional Value V(s) Probe: Explorer on Tile",
            data["value_maps"]["Explorer"],
            fmt="{:+5.2f}",
            row_order="descending",
        )

    if args.layer in ("all", "policy_prior"):
        print_grid(
            "Move 1 Policy Landing Prior (Sum of probabilities of landing on tile)",
            data["policy_priors"] * 100.0,
            fmt="{:4.1f}%",
            row_order="descending",
        )


if __name__ == "__main__":
    main()
