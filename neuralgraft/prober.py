"""
Capability Prober - Discovers capability directions in DiT weight space.

Uses closed-form linear regression (SVD-based pseudoinverse) to find which
directions in each DiT layer's activation space correspond to a source
model's capability.

This is the mathematical equivalent of what LoRA training converges to
after thousands of gradient steps -- computed in O(1).

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch

logger = logging.getLogger("neuralgraft.prober")


@dataclass
class CapabilityDirection:
    """A discovered capability direction for one DiT layer.

    Attributes:
        layer_name: Name of the transformer block (e.g. "transformer_blocks.5")
        direction: [d_hidden] unit vector pointing in the capability's direction
        coefficient: [d_hidden] raw regression weights (unnormalized)
        r_squared: How well this layer predicts the capability (0-1)
        strength: Grafting strength, scaled by R-squared
    """
    layer_name: str
    direction: torch.Tensor
    coefficient: torch.Tensor
    r_squared: float
    strength: float


class CapabilityProber:
    """Discovers capability directions via closed-form linear regression.

    Given activations H and capability scores s, finds the direction beta
    such that s ~ H @ beta using SVD-based pseudoinverse:

        H = U Sigma V^T
        beta = V @ diag(1/Sigma[:k]) @ U^T @ s

    R-squared measures how well the layer predicts the capability.
    Only layers with R^2 > min_r_squared are kept.

    Args:
        rank: Number of singular values to use in pseudoinverse (default: 8)
        min_r_squared: Minimum R^2 threshold for keeping a direction (default: 0.01)
    """

    def __init__(self, rank: int = 8, min_r_squared: float = 0.01):
        self.rank = rank
        self.min_r2 = min_r_squared

    def probe(
        self,
        activations: Dict[str, torch.Tensor],
        scores: torch.Tensor,
        layer_names: List[str],
        strength: float = 0.25,
    ) -> List[CapabilityDirection]:
        """
        Find capability directions in each layer's activation space.

        Args:
            activations: {layer_name: [N, d_hidden]} activation matrices
            scores: [N] capability scores from source model (higher = more capable)
            layer_names: Ordered layer names to probe
            strength: Base grafting strength (modulated by R-squared per layer)

        Returns:
            List of CapabilityDirection sorted by R-squared (strongest first)
        """
        directions = []

        # Standardize scores (zero mean, unit variance) for stable regression
        s = scores.float()
        if s.std() < 1e-8:
            logger.warning("Source model scores have zero variance -- cannot probe")
            return directions
        s = (s - s.mean()) / (s.std() + 1e-8)

        for layer_name in layer_names:
            if layer_name not in activations:
                continue

            H = activations[layer_name].float()  # [N, d]
            N, d = H.shape

            if N != s.shape[0]:
                if N < len(s):
                    s_resized = s[:N]
                else:
                    # Tile scores to match activation count (handles N >> len(s))
                    repeats = (N // len(s)) + 1
                    s_resized = s.repeat(repeats)[:N]
            else:
                s_resized = s

            # --- Closed-form linear regression ---
            # We want: s ~ H @ beta
            # Solution: beta = pinv(H) @ s = V @ diag(1/Sigma[:k]) @ U^T @ s
            try:
                U, S_vals, Vt = torch.linalg.svd(H, full_matrices=False)

                # Truncated pseudoinverse using top-k singular values
                k = min(self.rank, len(S_vals), N, d)
                S_inv = torch.zeros_like(S_vals)
                S_inv[:k] = 1.0 / (S_vals[:k] + 1e-8)

                # beta = V @ S^{-1} @ U^T @ s
                beta = Vt.T @ torch.diag(S_inv) @ U.T @ s_resized  # [d]

                # Compute R-squared (coefficient of determination)
                s_pred = H @ beta
                ss_res = ((s_resized - s_pred) ** 2).sum()
                ss_tot = ((s_resized - s_resized.mean()) ** 2).sum()
                r_squared = 1.0 - (ss_res / (ss_tot + 1e-8))
                r_squared = max(0.0, float(r_squared))

                if r_squared < self.min_r2:
                    continue

                # Capability direction = normalized regression coefficient
                direction = beta / (beta.norm() + 1e-8)

                # Adaptive strength: stronger where R-squared is higher
                effective_strength = strength * r_squared

                cap_dir = CapabilityDirection(
                    layer_name=layer_name,
                    direction=direction,
                    coefficient=beta,
                    r_squared=r_squared,
                    strength=effective_strength,
                )
                directions.append(cap_dir)

            except Exception as e:
                logger.warning(f"Probe failed for {layer_name}: {e}")
                continue

        # Sort by R-squared (most predictive layers first)
        directions.sort(key=lambda d: d.r_squared, reverse=True)

        if directions:
            top = directions[0]
            logger.info(f"  Best layer: {top.layer_name} (R^2={top.r_squared:.4f})")
            logger.info(f"  {len(directions)}/{len(layer_names)} layers have significant signal")
        else:
            logger.warning("  No layers found with significant capability signal")

        return directions

    def probe_multi_target(
        self,
        activations: Dict[str, torch.Tensor],
        score_matrix: torch.Tensor,
        layer_names: List[str],
        strength: float = 0.25,
    ) -> List[CapabilityDirection]:
        """
        Multi-target probing: source model produces vector output per sample.

        Uses PCA to decompose the score matrix into principal components,
        then probes for each component independently.

        Args:
            activations: {layer_name: [N, d_hidden]} activation matrices
            score_matrix: [N, d_score] multi-dimensional capability scores
            layer_names: Ordered layer names
            strength: Base grafting strength

        Returns:
            List of CapabilityDirection across all principal components
        """
        directions = []

        S_centered = score_matrix.float() - score_matrix.float().mean(0)
        U, S_vals, Vt = torch.linalg.svd(S_centered, full_matrices=False)

        k = min(self.rank, S_vals.shape[0])
        principal_scores = U[:, :k] * S_vals[:k]  # [N, k]

        for i in range(k):
            dirs = self.probe(
                activations, principal_scores[:, i], layer_names,
                strength * float(S_vals[i] / S_vals[0])
            )
            directions.extend(dirs)

        return directions
