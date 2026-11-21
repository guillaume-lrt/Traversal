"""
Unit tests for Milestone 3 (`model.py`).
Verifies Dual-Stem architecture shapes, cached static inference equivalence,
masked global softmax, and end-to-end gradient flow without illegal-action leakage.
"""

import io
import os
import numpy as np
import torch
import pytest

from model import (
    TraversalDualStemNet,
    masked_softmax,
    compute_alphazero_loss,
    load_model_from_checkpoint,
)


def test_dual_stem_shapes_and_cached_static_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    b = 4
    x_static = torch.randn(b, 4, 10, 10, device=device)
    x_dynamic = torch.randn(b, 34, 10, 10, device=device)

    with torch.no_grad():
        f_static = model.encode_static(x_static)
        f_dyn = model.encode_dynamic(x_dynamic)
        assert f_static.shape == (b, 32, 10, 10)
        assert f_dyn.shape == (b, 64, 10, 10)

        logits_full, val_full = model(x_static, x_dynamic)
        logits_cached, val_cached = model.forward_with_cached_static(f_static, x_dynamic)

        assert logits_full.shape == (b, 606)
        assert val_full.shape == (b, 1)
        assert torch.all(val_full >= -1.0) and torch.all(val_full <= 1.0)
        assert torch.allclose(logits_full, logits_cached, atol=1e-6)
        assert torch.allclose(val_full, val_cached, atol=1e-6)


def test_masked_softmax_and_joint_gradient_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.train()

    b = 4
    x_static = torch.randn(b, 4, 10, 10, device=device)
    x_dynamic = torch.randn(b, 34, 10, 10, device=device)

    # Create a sparse boolean legal mask with 20 legal actions per batch item
    legal_mask = torch.zeros(b, 606, dtype=torch.bool, device=device)
    legal_mask[:, :20] = True

    logits, val = model(x_static, x_dynamic)
    probs = masked_softmax(logits, legal_mask)

    # Illegal actions must have exact 0.0 probability, and legal probabilities sum to 1.0
    assert torch.all(probs[:, 20:] == 0.0)
    assert torch.allclose(probs[:, :20].sum(dim=-1), torch.ones(b, device=device), atol=1e-5)

    # Create target MCTS distribution over legal actions and target value z
    pi_target = torch.zeros(b, 606, device=device)
    pi_target[:, :20] = 1.0 / 20.0
    z_target = torch.tensor([1.0, -1.0, 0.0, 1.0], device=device)

    loss, metrics = compute_alphazero_loss(logits, val, pi_target, z_target, legal_mask)
    loss.backward()

    # Verify non-zero gradients flow into BOTH Static Stem and Dynamic Stem
    static_grad_norm = sum(
        p.grad.norm().item() for p in model.static_stem.parameters() if p.grad is not None
    )
    dynamic_grad_norm = sum(
        p.grad.norm().item() for p in model.dynamic_stem.parameters() if p.grad is not None
    )
    assert static_grad_norm > 0.0
    assert dynamic_grad_norm > 0.0
    assert metrics["loss_total"] > 0.0


def test_unbatched_3d_inputs_and_stem_broadcasting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.eval()

    # 1. Unbatched 3D inputs
    xs_3d = torch.randn(4, 10, 10, device=device)
    xd_3d = torch.randn(34, 10, 10, device=device)

    with torch.no_grad():
        fs_3d = model.encode_static(xs_3d)
        fd_3d = model.encode_dynamic(xd_3d)
        assert fs_3d.shape == (32, 10, 10)
        assert fd_3d.shape == (64, 10, 10)

        l_3d, v_3d = model(xs_3d, xd_3d)
        assert l_3d.shape == (606,)
        assert v_3d.shape == (1,)

        l_cached_3d, v_cached_3d = model.forward_with_cached_static(fs_3d, xd_3d)
        assert l_cached_3d.shape == (606,)
        assert v_cached_3d.shape == (1,)
        assert torch.allclose(l_3d, l_cached_3d, atol=1e-6)
        assert torch.allclose(v_3d, v_cached_3d, atol=1e-6)

        # 2. Broadcasting: 1 static feature paired with batch of 4 dynamic states
        fs_1 = fs_3d.unsqueeze(0)  # (1, 32, 10, 10)
        xd_b = torch.randn(4, 34, 10, 10, device=device)
        l_b, v_b = model.forward_with_cached_static(fs_1, xd_b)
        assert l_b.shape == (4, 606)
        assert v_b.shape == (4, 1)


def test_masked_softmax_and_loss_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraversalDualStemNet().to(device)
    model.train()

    import numpy as np

    # 1. NumPy mask and uint8 mask with masked_softmax
    logits = torch.randn(2, 606, device=device)
    mask_np = np.zeros((2, 606), dtype=np.uint8)
    mask_np[0, :10] = 1
    mask_np[1, 10:20] = 1

    probs = masked_softmax(logits, mask_np)
    assert probs.shape == (2, 606)
    assert torch.all(probs[0, 10:] == 0.0)
    assert torch.all(probs[1, :10] == 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, device=device), atol=1e-5)

    # 2. All-False mask (no legal moves) produces zeros without NaN
    mask_all_false = np.zeros((2, 606), dtype=bool)
    probs_false = masked_softmax(logits, mask_all_false)
    assert not torch.isnan(probs_false).any()
    assert torch.all(probs_false == 0.0)

    # 3. compute_alphazero_loss with NumPy targets, mismatched 1D vs 2D z, and unbatched inputs
    logits_b = torch.randn(3, 606, device=device, requires_grad=True)
    val_b = torch.randn(3, 1, device=device, requires_grad=True)
    pi_np = np.zeros((3, 606), dtype=np.float32)
    pi_np[:, :5] = 0.2
    z_1d_np = np.array([1.0, -1.0, 0.0], dtype=np.float32)  # (3,) vs (3, 1)
    mask_b = np.zeros((3, 606), dtype=bool)
    mask_b[:, :5] = True

    loss, metrics = compute_alphazero_loss(logits_b, val_b, pi_np, z_1d_np, mask_b)
    loss.backward()
    assert not torch.isnan(logits_b.grad).any()
    assert not torch.isnan(val_b.grad).any()
    assert metrics["loss_total"] > 0.0

    # 4. Unbatched 1D compute_alphazero_loss
    l_1d = torch.randn(606, device=device, requires_grad=True)
    v_1d = torch.randn(1, device=device, requires_grad=True)
    pi_1d = np.zeros(606, dtype=np.float32)
    pi_1d[:5] = 0.2
    z_scalar = 1.0
    mask_1d = np.zeros(606, dtype=bool)
    mask_1d[:5] = True

    loss_1d, metrics_1d = compute_alphazero_loss(l_1d, v_1d, pi_1d, z_scalar, mask_1d)
    loss_1d.backward()
    assert not torch.isnan(l_1d.grad).any()
    assert not torch.isnan(v_1d.grad).any()
    assert metrics_1d["loss_total"] > 0.0


def test_value_head_gap_and_flat_architectures():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_gap = TraversalDualStemNet(value_head_type="gap").to(device)
    model_flat = TraversalDualStemNet(value_head_type="flat").to(device)

    assert model_gap.value_head_type == "gap"
    assert model_flat.value_head_type == "flat"
    assert model_gap.get_config()["value_head_type"] == "gap"
    assert model_flat.get_config()["value_head_type"] == "flat"

    # Verify parameter counts reflect global pooling (GAP has ~4.2k in MLP vs ~51.2k in flat)
    gap_params = sum(p.numel() for p in model_gap.value_fc1.parameters())
    flat_params = sum(p.numel() for p in model_flat.value_fc1.parameters())
    assert gap_params < flat_params
    assert gap_params == (32 * 2 * 64 + 64)  # 4160
    assert flat_params == (800 * 64 + 64)     # 51264

    b = 2
    xs = torch.randn(b, 4, 10, 10, device=device)
    xd = torch.randn(b, 34, 10, 10, device=device)

    with torch.no_grad():
        lg_gap, val_gap = model_gap(xs, xd)
        lg_flat, val_flat = model_flat(xs, xd)

        assert lg_gap.shape == (b, 606)
        assert val_gap.shape == (b, 1)
        assert lg_flat.shape == (b, 606)
        assert val_flat.shape == (b, 1)


def test_load_model_from_checkpoint_autodetect_and_legacy_compatibility():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. GAP checkpoint with model_config metadata
    gap_net = TraversalDualStemNet(value_head_type="gap")
    buf_gap = io.BytesIO()
    torch.save(
        {
            "iteration": 10,
            "model_state_dict": gap_net.state_dict(),
            "model_config": gap_net.get_config(),
        },
        buf_gap,
    )
    buf_gap.seek(0)
    loaded_gap, meta_gap = load_model_from_checkpoint(buf_gap, device=device)
    assert loaded_gap.value_head_type == "gap"
    assert meta_gap.get("iteration") == 10

    # 2. Legacy flat checkpoint WITHOUT model_config (auto-detection from weight shape)
    flat_net = TraversalDualStemNet(value_head_type="flat")
    buf_flat = io.BytesIO()
    torch.save(
        {
            "iteration": 5,
            "model_state_dict": flat_net.state_dict(),
        },
        buf_flat,
    )
    buf_flat.seek(0)
    loaded_flat, meta_flat = load_model_from_checkpoint(buf_flat, device=device)
    assert loaded_flat.value_head_type == "flat"
    assert meta_flat.get("iteration") == 5

    # 3. Direct state dict without outer container
    buf_raw = io.BytesIO()
    torch.save(flat_net.state_dict(), buf_raw)
    buf_raw.seek(0)
    loaded_raw, _ = load_model_from_checkpoint(buf_raw, device=device)
    assert loaded_raw.value_head_type == "flat"

    # 4. Actual disk checkpoint from train_v4/best.pt if present
    legacy_ckpt = "checkpoints/train_v4/best.pt"
    if os.path.isfile(legacy_ckpt):
        real_legacy, real_meta = load_model_from_checkpoint(legacy_ckpt, device=device)
        assert real_legacy.value_head_type == "flat"
        assert real_legacy.encoder_version == "v1"
        assert real_legacy.dynamic_channels == 34
        assert "iteration" in real_meta
        # Verify inference runs properly
        xs = torch.randn(1, 4, 10, 10, device=device)
        xd = torch.randn(1, 34, 10, 10, device=device)
        with torch.no_grad():
            l, v = real_legacy(xs, xd)
            assert l.shape == (1, 606)
            assert v.shape == (1, 1)

    # 5. V2 checkpoint with 35 dynamic channels and GAP value head
    v2_net = TraversalDualStemNet(value_head_type="gap", encoder_version="v2")
    buf_v2 = io.BytesIO()
    torch.save(
        {
            "iteration": 15,
            "model_state_dict": v2_net.state_dict(),
            "model_config": v2_net.get_config(),
        },
        buf_v2,
    )
    buf_v2.seek(0)
    loaded_v2, meta_v2 = load_model_from_checkpoint(buf_v2, device=device)
    assert loaded_v2.value_head_type == "gap"
    assert loaded_v2.encoder_version == "v2"
    assert loaded_v2.dynamic_channels == 35
    assert meta_v2.get("iteration") == 15

    # 6. V2 checkpoint WITHOUT model_config (auto-detection of 35 channels from dynamic_stem.0.weight)
    buf_v2_raw = io.BytesIO()
    torch.save(
        {
            "model_state_dict": v2_net.state_dict(),
        },
        buf_v2_raw,
    )
    buf_v2_raw.seek(0)
    loaded_v2_auto, _ = load_model_from_checkpoint(buf_v2_raw, device=device)
    assert loaded_v2_auto.dynamic_channels == 35
    assert loaded_v2_auto.encoder_version == "v2"

    # 7. Non-model checkpoint (e.g. replay buffer dict) raises ValueError
    buf_invalid = io.BytesIO()
    torch.save(
        {
            "capacity": 40000,
            "size": 100,
            "head": 100,
            "buf_static_key": np.zeros(100),
        },
        buf_invalid,
    )
    buf_invalid.seek(0)
    with pytest.raises(ValueError, match="not a valid Traversal model checkpoint"):
        load_model_from_checkpoint(buf_invalid, device=device)


def test_model_encoder_version_v2_and_dynamic_channels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v1 = TraversalDualStemNet(encoder_version="v1").to(device)
    model_v2 = TraversalDualStemNet(encoder_version="v2").to(device)

    assert model_v1.encoder_version == "v1"
    assert model_v1.dynamic_channels == 34
    assert model_v2.encoder_version == "v2"
    assert model_v2.dynamic_channels == 35

    assert model_v1.get_config()["encoder_version"] == "v1"
    assert model_v1.get_config()["dynamic_channels"] == 34
    assert model_v2.get_config()["encoder_version"] == "v2"
    assert model_v2.get_config()["dynamic_channels"] == 35

    # Check forward pass with 35 channels
    b = 3
    xs = torch.randn(b, 4, 10, 10, device=device)
    xd_v2 = torch.randn(b, 35, 10, 10, device=device)

    logits, val = model_v2(xs, xd_v2)
    assert logits.shape == (b, 606)
    assert val.shape == (b, 1)

    # Cached forward with 35 channels
    fs = model_v2.encode_static(xs)
    l_cached, v_cached = model_v2.forward_with_cached_static(fs, xd_v2)
    assert torch.allclose(logits, l_cached, atol=1e-6)
    assert torch.allclose(val, v_cached, atol=1e-6)


