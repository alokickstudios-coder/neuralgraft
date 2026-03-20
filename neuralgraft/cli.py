"""
NeuralGraft CLI - Command-line interface for all NeuralGraft operations.

Usage:
    neuralgraft bake     --base model.safetensors --loras lora.safetensors:0.5
    neuralgraft graft    --base model.safetensors --calibration ./clips/
    neuralgraft amplify  --base model.safetensors --loras lora.safetensors:0.15
    neuralgraft full     --base model.safetensors --loras lora.safetensors:0.5
    neuralgraft list     # List available codecs

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import argparse
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuralgraft")


def _default_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _banner(title: str, details: Dict = None):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    if details:
        for k, v in details.items():
            print(f" {k:14s}: {v}")
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------

def cmd_bake(args):
    """Bake trained LoRAs into base model weights."""
    from .surgeon import WeightSurgeon

    base = Path(args.base)
    output = Path(args.output)

    loras = []
    if args.loras:
        for spec in args.loras:
            parts = spec.rsplit(":", 1)
            path = Path(parts[0]).expanduser()
            strength = float(parts[1]) if len(parts) > 1 else 0.5
            loras.append((path, strength))

    if not loras:
        logger.error("No LoRA files specified. Use --loras path:strength")
        raise SystemExit(1)

    _banner("NeuralGraft -- LoRA Baking", {
        "Base model": base.name,
        "Output": output.name,
        "LoRAs": len(loras),
    })

    for p, s in loras:
        logger.info(f"  {p.name} (strength={s:.2f})")

    t0 = time.time()
    surgeon = WeightSurgeon()
    result = surgeon.bake_loras(base, output, loras)
    dt = time.time() - t0

    _banner("Bake Complete", {
        "Time": f"{dt:.1f}s ({dt/60:.1f} min)",
        "Output": str(result),
        "Size": f"{result.stat().st_size / 1e9:.1f} GB",
    })
    return result


def cmd_graft(args):
    """Score quality gaps + spectral steering (uses external codecs)."""
    from .harvester import ActivationHarvester
    from .prober import CapabilityProber
    from .surgeon import WeightSurgeon
    from .codecs import get_codec

    base = Path(args.base)
    output = Path(args.output)
    cal_dir = Path(args.calibration).expanduser()
    strength = args.strength
    device = args.device or _default_device()
    logger.info(f"Using device: {device}")

    # Default capabilities (all lightweight, no external models needed)
    default_caps = {
        "sharpness":         {"codec": "sharpness",         "strength": 0.15},
        "edges":             {"codec": "edges",             "strength": 0.12},
        "temporal_ssim":     {"codec": "temporal_ssim",     "strength": 0.15},
        "flow_smoothness":   {"codec": "flow_smoothness",   "strength": 0.12},
        "color_consistency": {"codec": "color_consistency", "strength": 0.10},
        "face_stability":    {"codec": "face_stability",    "strength": 0.18},
        "texture":           {"codec": "texture",           "strength": 0.10},
    }

    caps = default_caps
    if args.capabilities:
        caps = {k: v for k, v in default_caps.items()
                if k in args.capabilities or v["codec"] in args.capabilities}
        for name in args.capabilities:
            if name not in caps and name not in [v["codec"] for v in caps.values()]:
                caps[name] = {"codec": name, "strength": 0.15}

    _banner("NeuralGraft -- Spectral Steering", {
        "Base model": base.name,
        "Output": output.name,
        "Calibration": str(cal_dir),
        "Capabilities": ", ".join(caps.keys()),
        "Strength": f"{strength}x",
    })

    t0 = time.time()

    # Phase 1: Harvest activations
    logger.info("Phase 1: Harvesting activation proxies...")
    harvester = ActivationHarvester(device=device)
    activations, layer_names = harvester.harvest(
        model_path=base,
        calibration_dir=cal_dir,
        n_clips=args.clips,
        n_frames=args.frames,
    )

    # Phase 2+3: Score + probe per capability
    prober = CapabilityProber(rank=args.probe_rank)
    all_directions = {}

    for cap_name, cap_config in caps.items():
        codec_name = cap_config["codec"]
        cap_strength = cap_config["strength"] * strength

        logger.info(f"\n[{cap_name}] Scoring with {codec_name}...")
        try:
            codec = get_codec(codec_name, device=device)
            # Pass the harvester's resolved video files to ensure alignment
            scores = codec.score(cal_dir, args.clips, args.frames,
                                 video_files=harvester.calibration_video_files)
            directions = prober.probe(activations, scores, layer_names, cap_strength)
            all_directions[cap_name] = directions
            n = len(directions)
            r2 = directions[0].r_squared if directions else 0.0
            logger.info(f"  [{cap_name}] {n} layers, best R^2={r2:.4f}")
        except Exception as e:
            logger.error(f"  [{cap_name}] FAILED: {e}")
            all_directions[cap_name] = []

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Phase 4: Weight surgery
    logger.info("\nPhase 4: Applying spectral steering...")
    surgeon = WeightSurgeon()
    result = surgeon.operate(base, output, all_directions, layer_names)

    dt = time.time() - t0
    total_dirs = sum(len(d) for d in all_directions.values())

    _banner("Graft Complete", {
        "Time": f"{dt:.1f}s ({dt/60:.1f} min)",
        "Directions": f"{total_dirs} across {len(layer_names)} blocks",
        "Output": str(result),
    })
    return result


def cmd_amplify(args):
    """LoRA-derived spectral amplification.

    Uses LoRA deltas (B@A) as ground-truth capability directions, then
    amplifies those directions in the base model via spectral steering.

    Complementary to baking:
      - Baking adds DeltaW = B@A (additive shift)
      - Amplify boosts singular values aligned with DeltaW (multiplicative emphasis)
    """
    from .surgeon import WeightSurgeon
    from .prober import CapabilityDirection
    from safetensors.torch import load_file

    base = Path(args.base)
    output = Path(args.output)
    boost_factor = args.strength

    lora_sources = []
    if args.loras:
        for spec in args.loras:
            parts = spec.rsplit(":", 1)
            path = Path(parts[0]).expanduser()
            strength = float(parts[1]) if len(parts) > 1 else 0.15
            lora_sources.append((path, strength))

    if not lora_sources:
        logger.error("No LoRA files specified. Use --loras path:strength")
        raise SystemExit(1)

    _banner("NeuralGraft -- LoRA-Derived Spectral Amplification", {
        "Base model": base.name,
        "Output": output.name,
        "LoRA sources": len(lora_sources),
        "Boost factor": f"{boost_factor}x",
    })

    t0 = time.time()

    logger.info("Phase 1: Extracting capability directions from LoRA deltas...")

    all_directions: Dict[str, List[CapabilityDirection]] = {}

    for lora_path, amp_strength in lora_sources:
        if not lora_path.exists():
            logger.warning(f"  Skipping {lora_path.name}: not found")
            continue

        lora_sd = load_file(str(lora_path))
        cap_name = lora_path.stem.replace("-", "_")
        directions = []

        logger.info(f"  Extracting from {lora_path.name} (amp={amp_strength:.3f})...")

        # Group LoRA weights by layer (supports standard, Kohya, PEFT formats)
        lora_pairs = {}
        for key in lora_sd:
            if "lora_A" in key or "lora_down" in key:
                base_key = key
                for pat in [".lora_A.weight", "lora_A.weight",
                            ".lora_down.weight", "lora_down.weight"]:
                    base_key = base_key.replace(pat, "")
                for prefix in ["base_model.model.", "base_model."]:
                    if base_key.startswith(prefix):
                        base_key = base_key[len(prefix):]
                lora_pairs.setdefault(base_key, {})["A"] = lora_sd[key]
            elif "lora_B" in key or "lora_up" in key:
                base_key = key
                for pat in [".lora_B.weight", "lora_B.weight",
                            ".lora_up.weight", "lora_up.weight"]:
                    base_key = base_key.replace(pat, "")
                for prefix in ["base_model.model.", "base_model."]:
                    if base_key.startswith(prefix):
                        base_key = base_key[len(prefix):]
                lora_pairs.setdefault(base_key, {})["B"] = lora_sd[key]
            elif key.endswith(".alpha"):
                base_key = key.replace(".alpha", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if base_key.startswith(prefix):
                        base_key = base_key[len(prefix):]
                alpha_val = lora_sd[key]
                lora_pairs.setdefault(base_key, {})["alpha"] = (
                    alpha_val.item() if alpha_val.numel() == 1 else float(alpha_val.flatten()[0])
                )

        eligible_pairs = [(k, p) for k, p in lora_pairs.items() if "A" in p and "B" in p]
        logger.info(f"    {len(eligible_pairs)} LoRA pairs to process")

        for base_key, pair in eligible_pairs:
            A = pair["A"].float()
            B = pair["B"].float()

            # Only amplify attention and FFN projections
            if not any(target in base_key for target in [
                "attn1.to_out.0", "attn2.to_out.0", "to_out.0",
                "ff.net.0.proj", "ff.net.2",
            ]):
                continue

            try:
                U_B, S_B, _ = torch.linalg.svd(B, full_matrices=False)
            except Exception:
                continue

            if S_B.shape[0] == 0 or S_B[0] < 1e-8:
                continue

            direction = U_B[:, 0]  # Principal output direction
            effective_strength = amp_strength * boost_factor

            # Extract block name
            block_name = base_key
            for pattern in ["attn1.to_out.0", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"]:
                if pattern in base_key:
                    block_name = base_key[:base_key.index(pattern)].rstrip(".")
                    break

            cap_dir = CapabilityDirection(
                layer_name=block_name,
                direction=direction,
                coefficient=direction,
                r_squared=1.0,  # Real direction, not probed
                strength=effective_strength,
            )
            directions.append(cap_dir)

        all_directions[cap_name] = directions
        logger.info(f"    {cap_name}: {len(directions)} directions extracted")

        del lora_sd
        gc.collect()

    total_dirs = sum(len(d) for d in all_directions.values())
    if total_dirs == 0:
        logger.error("No capability directions found")
        raise SystemExit(1)

    logger.info(f"\nPhase 2: Applying spectral steering ({total_dirs} directions)...")

    layer_names = sorted(set(
        d.layer_name for dirs in all_directions.values() for d in dirs
    ))

    surgeon = WeightSurgeon(max_delta_norm=0.10)
    result = surgeon.operate(base, output, all_directions, layer_names)

    dt = time.time() - t0

    _banner("Amplify Complete", {
        "Time": f"{dt:.1f}s ({dt/60:.1f} min)",
        "Directions": f"{total_dirs} from {len(lora_sources)} LoRAs",
        "Output": str(result),
        "Size": f"{result.stat().st_size / 1e9:.1f} GB",
    })
    return result


def cmd_full(args):
    """Full pipeline: bake LoRAs + spectral amplification."""
    _banner("NeuralGraft -- Full Pipeline", {
        "Mode": "LoRA Bake + Spectral Amplification",
    })

    t0 = time.time()

    # Step 1: Bake LoRAs
    import os
    pid = os.getpid()
    intermediate = Path(args.output).with_name(f"_neuralgraft_intermediate_{pid}.safetensors")
    args_bake = argparse.Namespace(**vars(args))
    args_bake.output = str(intermediate)
    baked = cmd_bake(args_bake)

    if baked is None:
        logger.warning("No LoRAs to bake, skipping to amplification")
        intermediate = Path(args.base)

    # Step 2: Amplify
    args_amp = argparse.Namespace(**vars(args))
    args_amp.base = str(intermediate)
    result = cmd_amplify(args_amp)

    # Clean up intermediate
    if intermediate.exists() and intermediate != Path(args.base):
        intermediate.unlink()
        intermediate.with_suffix(".manifest.json").unlink(missing_ok=True)
        logger.info(f"Cleaned up intermediate: {intermediate.name}")

    dt = time.time() - t0
    _banner("Full Pipeline Complete", {
        "Total time": f"{dt:.1f}s ({dt/60:.1f} min)",
        "Output": str(result) if result else "FAILED",
    })
    return result


def cmd_forge(args):
    """Construct a LoRA from images without training."""
    from .forge import LoRAForge

    base = Path(args.base)
    image_dir = Path(args.images)
    output = Path(args.output)
    device = args.device or _default_device()

    _banner("NeuralGraft -- LoRA Forge (Zero-Training LoRA Construction)", {
        "Base model": base.name,
        "Dataset": str(image_dir),
        "Output": output.name,
        "Rank": str(args.rank),
        "Strength": f"{args.strength}x",
        "Device": device,
        "Vision model": "DINOv2" if args.use_vision_model else "OpenCV (lightweight)",
    })

    t0 = time.time()

    forge = LoRAForge(
        rank=args.rank,
        strength=args.strength,
        min_r_squared=args.min_r2,
        use_vision_model=args.use_vision_model,
    )

    result = forge.forge(
        model_path=base,
        image_dir=image_dir,
        output_path=output,
        max_images=args.max_images,
        device=device,
        trigger_word=args.trigger_word,
    )

    dt = time.time() - t0

    if result is None:
        logger.error("Forge failed -- no concept signal found in dataset")
        raise SystemExit(1)

    _banner("Forge Complete", {
        "Time": f"{dt:.1f}s ({dt/60:.1f} min)",
        "Output": str(result),
        "Size": f"{result.stat().st_size / 1e6:.1f} MB",
    })

    if args.trigger_word:
        print(f"\n  Use trigger word '{args.trigger_word}' in your prompts to activate the LoRA.\n")

    return result


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="neuralgraft",
        description=(
            "NeuralGraft: Zero-training capability transfer for diffusion models. "
            "Hours of model training in minutes."
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Operation mode")

    # Shared arguments for all commands
    for name, help_text in [
        ("bake",    "Merge trained LoRAs into base weights"),
        ("graft",   "Score quality gaps + spectral steering (codec-based)"),
        ("amplify", "LoRA-derived spectral amplification"),
        ("full",    "Bake + amplify in one pass"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--base", type=str, required=True,
                        help="Path to base model checkpoint (safetensors)")
        p.add_argument("--output", "-o", type=str, default="neuralgrafted.safetensors",
                        help="Output path (default: neuralgrafted.safetensors)")
        p.add_argument("--calibration", type=str, default="./calibration_clips",
                        help="Directory with calibration video clips")
        p.add_argument("--strength", type=float, default=1.0,
                        help="Global strength multiplier (default: 1.0)")
        p.add_argument("--device", default=None,
                        help="Torch device (default: auto-detect cuda/mps/cpu)")
        p.add_argument("--clips", type=int, default=12,
                        help="Number of calibration clips (default: 12)")
        p.add_argument("--frames", type=int, default=9,
                        help="Frames per clip (default: 9)")
        p.add_argument("--probe-rank", type=int, default=8,
                        help="SVD truncation rank for probing (default: 8)")
        p.add_argument("--capabilities", nargs="+", default=None,
                        help="Specific capabilities to graft (default: all)")
        p.add_argument("--loras", nargs="+", default=None,
                        help="LoRA files as path:strength (e.g. lora.safetensors:0.6)")

    # Forge command (separate args)
    p_forge = sub.add_parser("forge",
        help="Construct a LoRA from images WITHOUT training (activation regression)")
    p_forge.add_argument("--base", type=str, required=True,
                         help="Path to base model checkpoint (safetensors)")
    p_forge.add_argument("--images", type=str, required=True,
                         help="Directory containing dataset images (10-100 recommended)")
    p_forge.add_argument("--output", "-o", type=str, default="forged-lora.safetensors",
                         help="Output LoRA path (default: forged-lora.safetensors)")
    p_forge.add_argument("--rank", type=int, default=16,
                         help="LoRA rank (default: 16)")
    p_forge.add_argument("--strength", type=float, default=1.0,
                         help="LoRA strength multiplier (default: 1.0)")
    p_forge.add_argument("--trigger-word", type=str, default=None,
                         help="Trigger word for the concept (e.g. 'mystyle')")
    p_forge.add_argument("--max-images", type=int, default=100,
                         help="Maximum images to use (default: 100)")
    p_forge.add_argument("--device", default=None,
                         help="Torch device for vision model (default: auto-detect)")
    p_forge.add_argument("--use-vision-model", action="store_true", default=False,
                         help="Use DINOv2 for richer features (requires internet on first run)")
    p_forge.add_argument("--min-r2", type=float, default=0.01,
                         help="Minimum R^2 to include a layer (default: 0.01)")

    # List codecs
    sub.add_parser("list", help="List available codecs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "list":
        from .codecs import get_codec, list_codecs
        print("\nAvailable codecs:")
        print("-" * 50)
        for name in list_codecs():
            codec = get_codec(name, device="cpu")
            print(f"  {name:22s}  {codec.description}")
        print()
        return

    if args.command == "forge":
        cmd_forge(args)
        return

    commands = {
        "bake": cmd_bake,
        "graft": cmd_graft,
        "amplify": cmd_amplify,
        "full": cmd_full,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
