#!/usr/bin/env python3
"""
NeuralGraft Quickstart - Graft capabilities into your model in 3 steps.

This example demonstrates:
  1. Baking a LoRA into base model weights (permanent, no runtime overhead)
  2. Grafting quality capabilities from calibration clips
  3. LoRA-derived spectral amplification

Requirements:
  - A safetensors model checkpoint (any DiT-based diffusion model)
  - A directory of calibration video clips (8-12 clips, any resolution)
  - Optional: LoRA weights to bake or use as amplification directions
"""

from pathlib import Path
from neuralgraft import WeightSurgeon, ActivationHarvester, CapabilityProber
from neuralgraft.codecs import get_codec

# -----------------------------------------------------------------------
# Configuration - EDIT THESE PATHS
# -----------------------------------------------------------------------
BASE_MODEL = Path("path/to/your/model.safetensors")
LORA_PATH = Path("path/to/your/lora.safetensors")
CALIBRATION_DIR = Path("path/to/calibration/clips/")  # 8-12 short video clips
OUTPUT = Path("my-neuralgrafted-model.safetensors")


def example_1_bake_lora():
    """Step 1: Bake a LoRA directly into model weights.

    This permanently merges the LoRA's improvements:
      W_new = W + strength * (alpha/rank) * (B @ A)

    No runtime LoRA loading needed after this.
    """
    print("\n=== Example 1: LoRA Baking ===\n")

    surgeon = WeightSurgeon()
    result = surgeon.bake_loras(
        model_path=BASE_MODEL,
        output_path=OUTPUT,
        lora_paths=[
            (LORA_PATH, 0.5),  # (path, strength)
            # Add more LoRAs here:
            # (Path("another-lora.safetensors"), 0.3),
        ],
    )
    print(f"Baked model saved to: {result}")


def example_2_graft_capabilities():
    """Step 2: Graft quality capabilities using calibration clips.

    This:
      1. Harvests activation proxies from the model on calibration data
      2. Scores those clips with quality codecs (sharpness, SSIM, etc.)
      3. Probes for which layers control each quality dimension
      4. Applies spectral steering to amplify those directions
    """
    print("\n=== Example 2: Capability Grafting ===\n")

    # Phase 1: Harvest activations
    harvester = ActivationHarvester(device="cuda:0")
    activations, layer_names = harvester.harvest(
        model_path=BASE_MODEL,
        calibration_dir=CALIBRATION_DIR,
        n_clips=12,
        n_frames=9,
    )
    print(f"Harvested {len(layer_names)} blocks")

    # Phase 2: Score with codecs and probe for directions
    prober = CapabilityProber(rank=8)
    all_directions = {}

    capabilities = {
        "sharpness":       {"codec": "sharpness",       "strength": 0.15},
        "temporal_ssim":   {"codec": "temporal_ssim",    "strength": 0.15},
        "face_stability":  {"codec": "face_stability",   "strength": 0.18},
    }

    for cap_name, config in capabilities.items():
        codec = get_codec(config["codec"], device="cuda:0")
        scores = codec.score(CALIBRATION_DIR, n_clips=12, n_frames=9)
        directions = prober.probe(activations, scores, layer_names, config["strength"])
        all_directions[cap_name] = directions
        print(f"  {cap_name}: {len(directions)} layers, "
              f"best R^2={directions[0].r_squared:.4f}" if directions else "  no signal")

    # Phase 3: Apply spectral steering
    surgeon = WeightSurgeon(max_delta_norm=0.15)
    result = surgeon.operate(
        model_path=BASE_MODEL,
        output_path=OUTPUT,
        directions=all_directions,
        layer_names=layer_names,
    )
    print(f"Grafted model saved to: {result}")


def example_3_custom_codec():
    """Step 3: Write your own codec to graft ANY capability.

    A codec just needs to score calibration clips. NeuralGraft discovers
    which model layers control that capability and amplifies them.
    """
    print("\n=== Example 3: Custom Codec ===\n")

    from neuralgraft.codecs import BaseCodec, register_codec
    import torch
    import cv2
    import numpy as np

    @register_codec
    class BrightnessCodec(BaseCodec):
        """Example: score clips by average brightness."""
        name = "brightness"
        description = "Average frame brightness"

        def _load_model(self):
            pass  # No model needed

        def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
            scores = []
            for i in range(frames.shape[0]):
                frame = frames[i].cpu().mean().item()  # Average pixel value
                scores.append(frame)
            return torch.tensor(scores)

    # Now use it like any other codec:
    codec = get_codec("brightness", device="cpu")
    print(f"Custom codec registered: {codec.name} - {codec.description}")
    # scores = codec.score(CALIBRATION_DIR)


if __name__ == "__main__":
    print("NeuralGraft Quickstart Examples")
    print("=" * 50)
    print()
    print("Edit the paths at the top of this file, then uncomment the example to run:")
    print()
    print("  example_1_bake_lora()        # Bake LoRAs into weights")
    print("  example_2_graft_capabilities()  # Graft quality capabilities")
    print("  example_3_custom_codec()     # Write your own codec")
    print()

    # Uncomment one:
    # example_1_bake_lora()
    # example_2_graft_capabilities()
    example_3_custom_codec()
