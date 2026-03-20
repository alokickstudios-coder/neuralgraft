"""Tests for the WeightSurgeon."""

import json
import tempfile
import torch
import pytest
from pathlib import Path
from safetensors.torch import save_file, load_file
from neuralgraft.surgeon import WeightSurgeon
from neuralgraft.prober import CapabilityDirection


@pytest.fixture
def tmp_model(tmp_path):
    """Create a minimal fake model checkpoint."""
    state_dict = {
        "transformer_blocks.0.attn1.to_out.0.weight": torch.randn(64, 64),
        "transformer_blocks.0.ff.net.2.weight": torch.randn(64, 64),
        "transformer_blocks.1.attn1.to_out.0.weight": torch.randn(64, 64),
        "norm.weight": torch.randn(64),
    }
    path = tmp_path / "model.safetensors"
    save_file(state_dict, str(path))
    return path


@pytest.fixture
def tmp_lora(tmp_path):
    """Create a minimal fake LoRA checkpoint."""
    rank = 4
    lora_sd = {
        "transformer_blocks.0.attn1.to_out.0.lora_A.weight": torch.randn(rank, 64),
        "transformer_blocks.0.attn1.to_out.0.lora_B.weight": torch.randn(64, rank),
        "transformer_blocks.0.attn1.to_out.0.alpha": torch.tensor(float(rank)),
    }
    path = tmp_path / "lora.safetensors"
    save_file(lora_sd, str(path))
    return path


def test_bake_lora(tmp_model, tmp_lora, tmp_path):
    """LoRA baking should modify weights and produce manifest."""
    output = tmp_path / "baked.safetensors"

    surgeon = WeightSurgeon()
    result = surgeon.bake_loras(tmp_model, output, [(tmp_lora, 0.5)])

    assert result.exists()
    assert output.stat().st_size > 0
    assert output.with_suffix(".manifest.json").exists()

    # Verify weights changed
    original = load_file(str(tmp_model))
    baked = load_file(str(output))

    key = "transformer_blocks.0.attn1.to_out.0.weight"
    assert not torch.allclose(original[key], baked[key])

    # Verify manifest
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["operation"] == "lora_bake"
    assert manifest["baked_keys"] > 0


def test_bake_preserves_unmodified(tmp_model, tmp_lora, tmp_path):
    """LoRA baking should not modify weights without matching LoRA keys."""
    output = tmp_path / "baked.safetensors"

    surgeon = WeightSurgeon()
    surgeon.bake_loras(tmp_model, output, [(tmp_lora, 0.5)])

    original = load_file(str(tmp_model))
    baked = load_file(str(output))

    # Block 1 has no LoRA, should be unchanged
    key = "transformer_blocks.1.attn1.to_out.0.weight"
    assert torch.allclose(original[key], baked[key])


def test_spectral_steering(tmp_model, tmp_path):
    """Spectral steering should modify targeted weights."""
    output = tmp_path / "grafted.safetensors"

    direction = torch.randn(64)
    direction = direction / direction.norm()

    directions = {
        "test_cap": [
            CapabilityDirection(
                layer_name="transformer_blocks.0",
                direction=direction,
                coefficient=direction,
                r_squared=0.8,
                strength=0.15,
            )
        ]
    }

    surgeon = WeightSurgeon(max_delta_norm=0.15)
    result = surgeon.operate(
        tmp_model, output, directions,
        layer_names=["transformer_blocks.0"],
    )

    assert result.exists()

    original = load_file(str(tmp_model))
    grafted = load_file(str(output))

    key = "transformer_blocks.0.attn1.to_out.0.weight"
    assert not torch.allclose(original[key], grafted[key])

    # Verify manifest
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["operation"] == "spectral_steering"
    assert "test_cap" in manifest["capabilities"]


def test_safety_clamp(tmp_model, tmp_path):
    """Large directions should be clamped to max_delta_norm."""
    output = tmp_path / "grafted.safetensors"

    # Very strong direction
    direction = torch.randn(64)
    direction = direction / direction.norm()

    directions = {
        "strong": [
            CapabilityDirection(
                layer_name="transformer_blocks.0",
                direction=direction,
                coefficient=direction,
                r_squared=1.0,
                strength=10.0,  # Very high strength
            )
        ]
    }

    surgeon = WeightSurgeon(max_delta_norm=0.05)  # Tight clamp
    result = surgeon.operate(
        tmp_model, output, directions,
        layer_names=["transformer_blocks.0"],
    )

    original = load_file(str(tmp_model))
    grafted = load_file(str(output))

    key = "transformer_blocks.0.attn1.to_out.0.weight"
    W_orig = original[key].float()
    W_graft = grafted[key].float()

    rel_change = (W_graft - W_orig).norm() / (W_orig.norm() + 1e-8)
    assert rel_change <= 0.06  # Small margin for floating point


def test_protected_keys_not_modified(tmp_model, tmp_path):
    """Protected keys (norms, embeds) should never be modified."""
    output = tmp_path / "grafted.safetensors"

    direction = torch.randn(64)
    direction = direction / direction.norm()

    directions = {
        "test": [
            CapabilityDirection(
                layer_name="norm",
                direction=direction,
                coefficient=direction,
                r_squared=1.0,
                strength=0.5,
            )
        ]
    }

    surgeon = WeightSurgeon()
    result = surgeon.operate(
        tmp_model, output, directions,
        layer_names=["norm"],
    )

    original = load_file(str(tmp_model))
    grafted = load_file(str(output))

    # norm.weight is protected, should be unchanged
    assert torch.allclose(original["norm.weight"], grafted["norm.weight"])
