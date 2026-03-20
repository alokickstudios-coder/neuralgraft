"""
NeuralGraft - Zero-Training Capability Transfer for Diffusion Models

Graft capabilities from ANY external model into your diffusion model's weights.
No training loop. No gradient descent. Pure linear algebra.
Hours of model training compressed into minutes.

Core Algorithm:
  1. Harvest: Run your DiT on calibration clips, record per-layer activations
  2. Score:   Run source model on same clips, produce capability scores
  3. Probe:   Closed-form linear regression -> capability directions per layer
  4. Graft:   SVD spectral steering on attention projections
  5. Save:    Modified checkpoint with baked capabilities

Mathematical Foundation:
  For each DiT layer l with output projection W_O:
    - Let H in R^{N x d} be activations on N calibration samples
    - Let s in R^N be source model capability scores
    - Capability direction: beta = pinv(H) @ s  (closed-form regression)
    - Spectral steering: W_new = U @ diag(sigma * boost) @ V^T
      where boost = 1 + strength * alignment^2

  This is mathematically equivalent to what LoRA training converges to
  after thousands of gradient steps -- computed in O(1).

Copyright 2026 Alokick Tech
Licensed under the Apache License, Version 2.0
"""

__version__ = "1.0.0"
__author__ = "Alokick Tech"
__email__ = "Alokickstudios@gmail.com"

from .prober import CapabilityDirection, CapabilityProber
from .surgeon import WeightSurgeon
from .harvester import ActivationHarvester

__all__ = [
    "CapabilityDirection",
    "CapabilityProber",
    "WeightSurgeon",
    "ActivationHarvester",
]
