"""Tests for the CapabilityProber."""

import torch
import pytest
from neuralgraft.prober import CapabilityProber, CapabilityDirection


def test_probe_finds_directions():
    """Prober should find directions when activations correlate with scores."""
    torch.manual_seed(42)
    N, d = 200, 8  # N >> d, and d <= rank so full signal is captured

    # Create activations where a specific direction correlates with scores
    H = torch.randn(N, d)
    true_direction = torch.randn(d)
    true_direction = true_direction / true_direction.norm()
    scores = (H @ true_direction) + torch.randn(N) * 0.1  # Linear + noise

    activations = {"block.0": H}
    prober = CapabilityProber(rank=8, min_r_squared=0.01)
    directions = prober.probe(activations, scores, ["block.0"], strength=0.15)

    assert len(directions) == 1
    assert directions[0].r_squared > 0.5
    assert directions[0].layer_name == "block.0"
    assert directions[0].direction.shape == (d,)
    assert abs(directions[0].direction.norm().item() - 1.0) < 1e-5


def test_probe_rejects_noise():
    """Prober should reject layers with no capability signal."""
    torch.manual_seed(42)
    N, d = 200, 8  # Overdetermined, random scores

    H = torch.randn(N, d)
    # Scores completely independent of activations
    torch.manual_seed(999)
    scores = torch.randn(N)

    activations = {"block.0": H}
    prober = CapabilityProber(rank=8, min_r_squared=0.5)
    directions = prober.probe(activations, scores, ["block.0"], strength=0.15)

    # With high R^2 threshold, random data should be rejected
    assert len(directions) == 0


def test_probe_multiple_layers():
    """Prober should sort results by R-squared."""
    torch.manual_seed(42)
    N, d = 200, 8  # N >> d, d <= rank

    true_dir = torch.randn(d)
    true_dir = true_dir / true_dir.norm()

    H_good = torch.randn(N, d)
    scores = H_good @ true_dir + torch.randn(N) * 0.1  # Strong linear relationship

    H_noisy = torch.randn(N, d)  # No relationship

    activations = {"block.0": H_good, "block.1": H_noisy}
    prober = CapabilityProber(rank=8, min_r_squared=0.01)
    directions = prober.probe(activations, scores, ["block.0", "block.1"], strength=0.15)

    assert len(directions) >= 1
    # First direction should be from the correlated block
    assert directions[0].layer_name == "block.0"
    assert directions[0].r_squared > 0.5


def test_probe_multi_target():
    """Multi-target probing with PCA decomposition."""
    torch.manual_seed(42)
    N, d, d_score = 36, 64, 4

    H = torch.randn(N, d)
    score_matrix = torch.randn(N, d_score)

    activations = {"block.0": H}
    prober = CapabilityProber(rank=4, min_r_squared=0.01)
    directions = prober.probe_multi_target(activations, score_matrix, ["block.0"], strength=0.15)

    # Should attempt to probe for each principal component
    assert isinstance(directions, list)


def test_capability_direction_dataclass():
    """CapabilityDirection should store all fields."""
    d = CapabilityDirection(
        layer_name="transformer_blocks.5",
        direction=torch.randn(128),
        coefficient=torch.randn(128),
        r_squared=0.75,
        strength=0.15,
    )
    assert d.layer_name == "transformer_blocks.5"
    assert d.r_squared == 0.75
    assert d.strength == 0.15
    assert d.direction.shape == (128,)
