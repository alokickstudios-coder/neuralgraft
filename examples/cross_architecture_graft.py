#!/usr/bin/env python3
"""
Cross-Architecture Grafting Example

This demonstrates NeuralGraft's most powerful feature: transferring capabilities
from one model architecture to another WITHOUT any training.

Example: Graft WAN 2.2's motion quality into LTX 2.3's weights.

You cannot directly merge weights from different architectures (different shapes,
different layer counts, different attention patterns). But you CAN:
  1. Score what one model does well (WAN's motion quality)
  2. Find where in your model those capabilities live
  3. Amplify those directions in your model's weights

This is what NeuralGraft does.
"""

from pathlib import Path
import torch
import numpy as np
from neuralgraft import WeightSurgeon, ActivationHarvester, CapabilityProber
from neuralgraft.codecs import BaseCodec, register_codec


# -----------------------------------------------------------------------
# Step 1: Define a codec that scores clips using the SOURCE model
# -----------------------------------------------------------------------

@register_codec
class WANMotionCodec(BaseCodec):
    """Score motion quality by running WAN 2.2 and measuring output quality.

    In practice, you'd score the SOURCE model's outputs using automated
    quality metrics. Here we use optical flow smoothness as a proxy.
    """
    name = "wan_motion"
    description = "Motion quality scored via optical flow analysis"

    def _load_model(self):
        import cv2
        self._cv2 = cv2

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Score temporal motion quality of frames."""
        cv2 = self._cv2
        frames_np = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        if len(frames_np) < 3:
            return torch.tensor([0.5] * frames.shape[0])

        # Compute optical flow smoothness
        flow_smoothness_scores = []
        prev_flow = None

        for i in range(len(frames_np) - 1):
            gray_a = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(frames_np[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            if prev_flow is not None:
                # Smooth motion = low acceleration (low diff between consecutive flows)
                acceleration = np.mean(np.abs(flow - prev_flow))
                smoothness = 1.0 / (1.0 + acceleration * 10)
                flow_smoothness_scores.append(smoothness)

            prev_flow = flow

        avg = np.mean(flow_smoothness_scores) if flow_smoothness_scores else 0.5
        return torch.tensor([avg] * frames.shape[0])


# -----------------------------------------------------------------------
# Step 2: Harvest, probe, and graft
# -----------------------------------------------------------------------

def graft_motion_quality(
    base_model: str,
    calibration_dir: str,
    output: str = "motion-grafted.safetensors",
    strength: float = 0.15,
):
    """Graft motion quality into a diffusion model checkpoint."""

    base = Path(base_model)
    cal_dir = Path(calibration_dir)
    out = Path(output)

    print(f"Base model:  {base.name}")
    print(f"Calibration: {cal_dir}")
    print(f"Output:      {out}")
    print(f"Strength:    {strength}")
    print()

    # Phase 1: Harvest
    print("Phase 1: Harvesting activations...")
    harvester = ActivationHarvester(device="cuda:0")
    activations, layer_names = harvester.harvest(
        model_path=base,
        calibration_dir=cal_dir,
        n_clips=12,
        n_frames=9,
    )
    print(f"  {len(layer_names)} blocks harvested")

    # Phase 2: Score with our WAN motion codec
    print("\nPhase 2: Scoring with WAN motion codec...")
    from neuralgraft.codecs import get_codec
    codec = get_codec("wan_motion", device="cuda:0")
    scores = codec.score(cal_dir, n_clips=12, n_frames=9)
    print(f"  Scores range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Phase 3: Probe for motion-controlling layers
    print("\nPhase 3: Probing for motion directions...")
    prober = CapabilityProber(rank=8)
    directions = prober.probe(activations, scores, layer_names, strength)
    print(f"  Found {len(directions)} layers with motion signal")
    if directions:
        print(f"  Best layer: {directions[0].layer_name} (R^2={directions[0].r_squared:.4f})")

    # Phase 4: Spectral steering
    print("\nPhase 4: Applying spectral steering...")
    surgeon = WeightSurgeon(max_delta_norm=0.10)
    result = surgeon.operate(
        model_path=base,
        output_path=out,
        directions={"motion_quality": directions},
        layer_names=layer_names,
    )
    print(f"\nDone! Motion-grafted model saved to: {result}")
    print(f"Size: {result.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graft motion quality into a model")
    parser.add_argument("--base", required=True, help="Base model checkpoint")
    parser.add_argument("--calibration", required=True, help="Calibration clips directory")
    parser.add_argument("--output", default="motion-grafted.safetensors")
    parser.add_argument("--strength", type=float, default=0.15)
    args = parser.parse_args()

    graft_motion_quality(args.base, args.calibration, args.output, args.strength)
