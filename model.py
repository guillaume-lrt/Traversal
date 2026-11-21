"""
Two-Block Dual-Stem Policy-Value Residual Network for Traversal (Milestone 3).

Implements `TraversalDualStemNet` following Section 5 and Section 6.3 of
`traversal_alphazero_ai_architecture_specification.md`:
  - Static Stem:  (B, 4, 10, 10)  -> Conv3x3(4->32) + BN + ReLU -> 2x ResBlock(32) -> (B, 32, 10, 10)
  - Dynamic Stem: (B, 34, 10, 10) -> Conv3x3(34->64) + BN + ReLU -> 2x ResBlock(64) -> (B, 64, 10, 10)
  - Fusion Trunk: Cat(96) -> Conv1x1(96->96) + BN + ReLU -> 4x ResBlock(96)
  - Policy Head:  Conv1x1(96->32) + BN + ReLU -> Flatten(3200) -> Linear(3200->606)
  - Value Head:   Conv1x1(96->8)  + BN + ReLU -> Flatten(800)  -> Linear(800->64) -> ReLU -> Linear(64->1) -> Tanh
"""

from __future__ import annotations

import io
import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from traversal_env import ACTION_SPACE_SIZE
from encoder import STATIC_CHANNELS, DYNAMIC_CHANNELS


class ResBlock(nn.Module):
    """Standard 2-layer 3x3 Residual Block with BatchNorm and ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class TraversalDualStemNet(nn.Module):
    """
    Two-Block Dual-Stem Policy-Value Residual Network for Traversal.
    Supports both:
      - Joint forward pass `forward(x_static, x_dynamic)` for end-to-end gradient updates
      - Cached static inference `forward_with_cached_static(f_static, x_dynamic)` for MCTS
    """

    def __init__(
        self,
        value_head_type: str = "gap",
        dynamic_channels: Optional[int] = None,
        encoder_version: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.value_head_type = value_head_type.lower().strip()
        if self.value_head_type not in ("gap", "flat"):
            raise ValueError(f"Unknown value_head_type: '{value_head_type}', choose 'gap' or 'flat'")

        if encoder_version is not None:
            self.encoder_version = encoder_version.lower().strip()
            self.dynamic_channels = 35 if self.encoder_version == "v2" else 34
        elif dynamic_channels is not None:
            self.dynamic_channels = int(dynamic_channels)
            self.encoder_version = "v2" if self.dynamic_channels == 35 else "v1"
        else:
            self.dynamic_channels = 34
            self.encoder_version = "v1"

        # 1. Static Stem: (B, 4, 10, 10) -> (B, 32, 10, 10)
        self.static_stem = nn.Sequential(
            nn.Conv2d(STATIC_CHANNELS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            ResBlock(32),
        )

        # 2. Dynamic Stem: (B, dynamic_channels, 10, 10) -> (B, 64, 10, 10)
        self.dynamic_stem = nn.Sequential(
            nn.Conv2d(self.dynamic_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
        )

        # 3. Fusion & Shared Trunk: (B, 96, 10, 10) -> (B, 96, 10, 10)
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
        )

        # 4. Policy Head: (B, 96, 10, 10) -> (B, 606)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 10 * 10, ACTION_SPACE_SIZE)

        # 5. Value Head:
        # 'gap': Global (Average + Max) Pooling head to eliminate spatial layout memorisation (review #1.3)
        # 'flat': Legacy (8 * 10 * 10 = 800 -> 64) flattened head for older checkpoints
        if self.value_head_type == "gap":
            self.value_conv = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.value_fc1 = nn.Linear(32 * 2, 64)  # 32 mean-pool + 32 max-pool
            self.value_fc2 = nn.Linear(64, 1)
        else:
            self.value_conv = nn.Sequential(
                nn.Conv2d(96, 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            )
            self.value_fc1 = nn.Linear(8 * 10 * 10, 64)
            self.value_fc2 = nn.Linear(64, 1)

    def get_config(self) -> Dict[str, Any]:
        return {
            "value_head_type": self.value_head_type,
            "encoder_version": self.encoder_version,
            "dynamic_channels": self.dynamic_channels,
        }

    def encode_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Encodes `X_static` of shape (B, 4, 10, 10) or unbatched (4, 10, 10) into
        `F_static` of shape (B, 32, 10, 10) or (32, 10, 10).
        """
        unbatched = x_static.ndim == 3
        if unbatched:
            x_static = x_static.unsqueeze(0)
        f_static = self.static_stem(x_static)
        return f_static.squeeze(0) if unbatched else f_static

    def encode_dynamic(self, x_dynamic: torch.Tensor) -> torch.Tensor:
        """
        Encodes `X_dynamic` of shape (B, 34, 10, 10) or unbatched (34, 10, 10) into
        `F_dyn` of shape (B, 64, 10, 10) or (64, 10, 10).
        """
        unbatched = x_dynamic.ndim == 3
        if unbatched:
            x_dynamic = x_dynamic.unsqueeze(0)
        f_dyn = self.dynamic_stem(x_dynamic)
        return f_dyn.squeeze(0) if unbatched else f_dyn

    def forward_trunk(
        self, f_static: torch.Tensor, f_dyn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses `f_static` (B, 32, 10, 10) and `f_dyn` (B, 64, 10, 10) through the
        shared trunk and outputs `(logits, value)` of shapes `(B, 606)` and `(B, 1)`.
        Also supports unbatched inputs (32, 10, 10) and (64, 10, 10), and broadcasts
        when one stem feature has batch size 1 and the other has batch size > 1.
        """
        unbatched = f_static.ndim == 3 and f_dyn.ndim == 3
        if f_static.ndim == 3:
            f_static = f_static.unsqueeze(0)
        if f_dyn.ndim == 3:
            f_dyn = f_dyn.unsqueeze(0)

        if f_static.device != f_dyn.device:
            f_static = f_static.to(f_dyn.device)

        if f_static.shape[0] == 1 and f_dyn.shape[0] > 1:
            f_static = f_static.expand(f_dyn.shape[0], -1, -1, -1)
        elif f_dyn.shape[0] == 1 and f_static.shape[0] > 1:
            f_dyn = f_dyn.expand(f_static.shape[0], -1, -1, -1)

        f_joint = torch.cat([f_static, f_dyn], dim=1)
        trunk_feat = self.shared_trunk(f_joint)

        p_feat = self.policy_conv(trunk_feat).flatten(1)
        logits = self.policy_fc(p_feat)

        v_feat = self.value_conv(trunk_feat)
        if self.value_head_type == "gap":
            avg_p = v_feat.mean(dim=[-2, -1])
            max_p = v_feat.amax(dim=[-2, -1])
            v_pooled = torch.cat([avg_p, max_p], dim=-1)
            v_hid = F.relu(self.value_fc1(v_pooled), inplace=True)
        else:
            v_flat = v_feat.flatten(1)
            v_hid = F.relu(self.value_fc1(v_flat), inplace=True)
        value = torch.tanh(self.value_fc2(v_hid))

        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    def forward_with_cached_static(
        self, f_static: torch.Tensor, x_dynamic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast MCTS leaf evaluation using pre-computed `f_static` (Section 5.2).
        Supports both batched and unbatched inputs.
        """
        unbatched = f_static.ndim == 3 and x_dynamic.ndim == 3
        if f_static.ndim == 3:
            f_static = f_static.unsqueeze(0)
        if x_dynamic.ndim == 3:
            x_dynamic = x_dynamic.unsqueeze(0)

        f_dyn = self.dynamic_stem(x_dynamic)
        logits, value = self.forward_trunk(f_static, f_dyn)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    def forward(
        self, x_static: torch.Tensor, x_dynamic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full joint forward pass for training so gradients flow into both stems simultaneously.
        Supports both batched and unbatched inputs.
        """
        unbatched = x_static.ndim == 3 and x_dynamic.ndim == 3
        if x_static.ndim == 3:
            x_static = x_static.unsqueeze(0)
        if x_dynamic.ndim == 3:
            x_dynamic = x_dynamic.unsqueeze(0)

        f_static = self.static_stem(x_static)
        f_dyn = self.dynamic_stem(x_dynamic)
        logits, value = self.forward_trunk(f_static, f_dyn)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def masked_softmax(
    logits: torch.Tensor, legal_mask: torch.Tensor | np.ndarray
) -> torch.Tensor:
    """
    Applies a single global Softmax across all 606 logits after masking illegal actions to -1e8
    (Section 4.2). Guarantees exact 0.0 on illegal actions and sum(P) = 1.0 over legal actions.
    Supports Tensor, NumPy, and list masks, boolean/integer dtypes, mismatched devices, and 1D/2D shapes.
    """
    legal_mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)

    if legal_mask.ndim < logits.ndim:
        legal_mask = legal_mask.unsqueeze(0)
    elif logits.ndim < legal_mask.ndim:
        logits = logits.unsqueeze(0)

    mask_val = -1e4 if logits.dtype == torch.float16 else -1e8
    neg_inf = torch.as_tensor(mask_val, dtype=logits.dtype, device=logits.device)
    masked_logits = torch.where(legal_mask, logits, neg_inf)
    probs = F.softmax(masked_logits, dim=-1)
    probs = torch.where(legal_mask, probs, torch.zeros_like(probs))
    norm = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / norm


def compute_alphazero_loss(
    logits: torch.Tensor,
    value_pred: torch.Tensor,
    pi_target: torch.Tensor | np.ndarray,
    z_target: torch.Tensor | np.ndarray | float | int,
    legal_mask: torch.Tensor | np.ndarray,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the combined AlphaZero loss (Section 6.3):
      L = (z - v)^2 - sum_{a in legal} pi_MCTS(a) * log p_theta(a)
    (L2 weight decay c=1e-4 is applied via AdamW weight_decay).
    Supports Tensor, NumPy, and Python scalar targets/masks, device synchronization, and unbatched inputs.
    """
    pi_target = torch.as_tensor(pi_target, dtype=logits.dtype, device=logits.device)
    legal_mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    z_target = torch.as_tensor(z_target, dtype=value_pred.dtype, device=value_pred.device)

    v_flat = value_pred.view(-1)
    z_flat = z_target.view(-1)
    value_loss = F.mse_loss(v_flat, z_flat)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        legal_mask = legal_mask.unsqueeze(0)
        pi_target = pi_target.unsqueeze(0)

    mask_val = -1e4 if logits.dtype == torch.float16 else -1e8
    neg_inf = torch.as_tensor(mask_val, dtype=logits.dtype, device=logits.device)
    masked_logits = torch.where(legal_mask, logits, neg_inf)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # Zero out illegal terms before multiplying by pi_target to prevent any gradient leakage
    safe_log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(pi_target * safe_log_probs).sum(dim=-1).mean()

    # Exact target entropy and KL divergence D_KL(pi_target || p_theta) = CE - H(pi_target)
    target_ent = -(
        torch.where(
            pi_target > 0,
            pi_target * torch.log(pi_target.clamp_min(1e-12)),
            torch.zeros_like(pi_target),
        )
    ).sum(dim=-1).mean()
    policy_kl = (policy_loss - target_ent).clamp_min(0.0)

    total_loss = value_loss + policy_loss
    metrics = {
        "loss_total": float(total_loss.detach().item()),
        "loss_value": float(value_loss.detach().item()),
        "loss_policy": float(policy_loss.detach().item()),
        "policy_entropy": float(target_ent.detach().item()),
        "policy_kl": float(policy_kl.detach().item()),
    }
    return total_loss, metrics


def load_model_from_checkpoint(
    checkpoint_or_path: str | Dict[str, Any],
    device: torch.device | str = "cpu",
) -> Tuple[TraversalDualStemNet, Dict[str, Any]]:
    """
    Loads a TraversalDualStemNet from a file path or loaded checkpoint dict,
    auto-detecting architecture configuration (such as 'gap' vs legacy 'flat' value head)
    to ensure full backward compatibility with older checkpoints (review #1.3).
    Returns (model, metadata_dict).
    """
    if isinstance(checkpoint_or_path, (str, os.PathLike, io.IOBase)):
        ckpt = torch.load(checkpoint_or_path, map_location=device, weights_only=False)
    else:
        ckpt = checkpoint_or_path

    metadata: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        for k, v in ckpt.items():
            if k != "model_state_dict":
                metadata[k] = v
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    if not isinstance(state_dict, dict) or not any(
        k.startswith(("static_stem", "dynamic_stem", "shared_trunk", "policy_", "value_"))
        for k in state_dict.keys()
    ):
        target_name = (
            checkpoint_or_path
            if isinstance(checkpoint_or_path, (str, os.PathLike))
            else type(checkpoint_or_path).__name__
        )
        found_keys = list(state_dict.keys())[:8] if isinstance(state_dict, dict) else type(state_dict)
        raise ValueError(
            f"Checkpoint at '{target_name}' is not a valid Traversal model checkpoint "
            f"(expected neural network parameters, found keys: {found_keys})."
        )

    value_head_type = None
    encoder_version = None
    dynamic_channels = None

    if "model_config" in metadata and isinstance(metadata["model_config"], dict):
        cfg = metadata["model_config"]
        value_head_type = cfg.get("value_head_type", None)
        encoder_version = cfg.get("encoder_version", None)
        dynamic_channels = cfg.get("dynamic_channels", None)

    if value_head_type is None and "value_fc1.weight" in state_dict:
        # Auto-detect from weight shape: (64, 800) is legacy flat head, (64, 64) is GAP head
        fc1_shape = state_dict["value_fc1.weight"].shape
        if len(fc1_shape) >= 2 and fc1_shape[1] == 800:
            value_head_type = "flat"
        elif len(fc1_shape) >= 2 and fc1_shape[1] == 64:
            value_head_type = "gap"
    if value_head_type is None:
        value_head_type = "gap"

    if dynamic_channels is None and "dynamic_stem.0.weight" in state_dict:
        stem_weight_shape = state_dict["dynamic_stem.0.weight"].shape
        if len(stem_weight_shape) >= 2:
            dynamic_channels = int(stem_weight_shape[1])
            if encoder_version is None:
                encoder_version = "v2" if dynamic_channels == 35 else "v1"

    model = TraversalDualStemNet(
        value_head_type=value_head_type,
        dynamic_channels=dynamic_channels,
        encoder_version=encoder_version,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, metadata
