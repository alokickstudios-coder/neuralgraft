"""
Weight Surgeon - Applies capability directions as permanent weight modifications.

Two modes of operation:
  1. LoRA Baking: Merges trained LoRA weights directly into the base model
     checkpoint, eliminating runtime overhead.
  2. Capability Grafting: Modifies attention weights to amplify discovered
     capability directions using SVD-based spectral steering.

Both produce a single checkpoint with capabilities permanently baked in.

The spectral steering algorithm:
  For each weight matrix W = U Sigma V^T:
    1. Project capability direction v into U's space
    2. Compute alignment: a = |U^T @ v|
    3. Boost singular values: sigma_new = sigma * (1 + strength * a^2)
    4. Reconstruct: W_new = U @ diag(sigma_new) @ V^T

This modifies WHAT the model attends to without changing HOW it attends.

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import logging
import hashlib
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .prober import CapabilityDirection

logger = logging.getLogger("neuralgraft.surgeon")

# FP8 format max representable values
_FP8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}

# Keys to target for weight surgery (attention output projections + FFN)
DEFAULT_SURGERY_TARGETS = [
    "attn1.to_out.0.weight",
    "attn2.to_out.0.weight",
    "attn.to_out.0.weight",
    "to_out.0.weight",
    "ff.net.2.weight",
]

# Keys to NEVER modify (normalization, embeddings, quantization metadata)
DEFAULT_PROTECTED_PATTERNS = [
    "per_channel_statistics",
    "scale", "zero_point",
    "norm", "ln_", "layer_norm",
    "embed", "pos_embed",
]


def _is_protected(key: str, protected_patterns: List[str] = None) -> bool:
    """Check if a key should be protected from modification."""
    patterns = protected_patterns or DEFAULT_PROTECTED_PATTERNS
    return any(p in key for p in patterns)


def _file_checksum(path: Path, nbytes: int = 4096) -> str:
    """Compute a short checksum of the first nbytes of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(nbytes)).hexdigest()[:16]


def _read_metadata(path: Path) -> Optional[Dict[str, str]]:
    """Read safetensors metadata (preserves model config for inference frameworks)."""
    try:
        with safe_open(str(path), framework="pt") as f:
            return f.metadata()
    except Exception as e:
        logger.warning(f"Could not read metadata from {path}: {e}")
        return None


class WeightSurgeon:
    """Applies capability directions and LoRA weights to model checkpoints.

    Args:
        safety_checks: Enable NaN/Inf detection and delta clamping (default: True)
        max_delta_norm: Maximum relative weight change per tensor (default: 0.15)
        surgery_targets: Weight name suffixes to target (default: attention + FFN)
        protected_patterns: Key patterns to never modify
    """

    def __init__(
        self,
        safety_checks: bool = True,
        max_delta_norm: float = 0.15,
        surgery_targets: List[str] = None,
        protected_patterns: List[str] = None,
    ):
        self.safety_checks = safety_checks
        self.max_delta_norm = max_delta_norm
        self.surgery_targets = surgery_targets or DEFAULT_SURGERY_TARGETS
        self.protected_patterns = protected_patterns or DEFAULT_PROTECTED_PATTERNS

    # ------------------------------------------------------------------
    # LoRA Baking
    # ------------------------------------------------------------------

    def bake_loras(
        self,
        model_path: Path,
        output_path: Path,
        lora_paths: List[Tuple[Path, float]],
    ) -> Path:
        """
        Merge LoRA weight deltas directly into base model weights.

        Formula: W_new = W + strength * (alpha/rank) * (B @ A)

        Handles FP8 quantized models: dequantizes to BF16, applies delta,
        re-quantizes with recomputed scale factor.

        Args:
            model_path: Base model checkpoint (safetensors format)
            output_path: Where to save the merged checkpoint
            lora_paths: List of (lora_path, strength) tuples

        Returns:
            Path to the merged checkpoint
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        logger.info(f"Baking {len(lora_paths)} LoRAs into {model_path.name}")

        # Preserve original metadata (critical for framework model detection)
        original_metadata = _read_metadata(model_path)

        state_dict = load_file(str(model_path))
        baked_keys = []

        for lora_path, strength in lora_paths:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                logger.warning(f"  LoRA not found: {lora_path}")
                continue

            lora_sd = load_file(str(lora_path))
            logger.info(f"  Baking {lora_path.name} (strength={strength:.2f}, {len(lora_sd)} keys)")

            # Group LoRA weights by layer
            # Supports multiple LoRA formats:
            #   - Standard: lora_A/lora_B (diffusers, HuggingFace)
            #   - Kohya/CivitAI: lora_down/lora_up
            #   - PEFT: base_model.model.*.lora_A
            lora_pairs = {}
            for key in lora_sd:
                # Detect A matrix (lora_A or lora_down)
                if "lora_A" in key or "lora_down" in key:
                    base_key = key
                    for pat in [".lora_A.weight", "lora_A.weight",
                                ".lora_down.weight", "lora_down.weight"]:
                        base_key = base_key.replace(pat, "")
                    # Strip PEFT prefix
                    for prefix in ["base_model.model.", "base_model."]:
                        if base_key.startswith(prefix):
                            base_key = base_key[len(prefix):]
                    lora_pairs.setdefault(base_key, {})["A"] = lora_sd[key]
                # Detect B matrix (lora_B or lora_up)
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

            # Build lookup for fast key matching
            base_key_set = set(state_dict.keys())

            def _find_target(lora_base_key: str) -> Optional[str]:
                # Try multiple prefix conventions to match LoRA key → model key
                stripped = lora_base_key.rstrip(".")
                # Also try stripping diffusion_model. prefix (ComfyUI LoRA format)
                stripped_no_dm = stripped
                if stripped_no_dm.startswith("diffusion_model."):
                    stripped_no_dm = stripped_no_dm[len("diffusion_model."):]
                candidates = [
                    stripped + ".weight",
                    "model." + stripped + ".weight",
                    "diffusion_model." + stripped + ".weight",
                    stripped_no_dm + ".weight",
                    "model." + stripped_no_dm + ".weight",
                ]
                for c in candidates:
                    if c in base_key_set:
                        return c
                return None

            for base_key, pair in lora_pairs.items():
                if "A" not in pair or "B" not in pair:
                    continue

                A = pair["A"].float()  # [rank, d_in]
                B = pair["B"].float()  # [d_out, rank]

                target_key = _find_target(base_key)
                if target_key is None:
                    continue

                W = state_dict[target_key]
                orig_dtype = W.dtype

                # Handle FP8 scaled format
                scale_key = target_key.replace(".weight", ".weight_scale")
                has_scale = scale_key in state_dict

                if orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    W_float = W.to(torch.bfloat16).float()
                    if has_scale:
                        W_float = W_float * state_dict[scale_key].float()
                else:
                    W_float = W.float()

                # Delta W = strength * (alpha/rank) * B @ A
                rank = A.shape[0]
                alpha = pair.get("alpha", float(rank))
                delta = strength * (alpha / rank) * (B @ A)

                if delta.shape != W_float.shape:
                    logger.warning(f"    Shape mismatch {target_key}: {delta.shape} vs {W_float.shape}")
                    continue

                W_new = W_float + delta

                # Safety checks
                if self.safety_checks:
                    if torch.isnan(W_new).any() or torch.isinf(W_new).any():
                        logger.error(f"    NaN/Inf after merge: {target_key} -- SKIPPED")
                        continue
                    rel_change = delta.norm() / (W_float.norm() + 1e-8)
                    if rel_change > 0.5:
                        logger.warning(f"    Large delta {target_key}: {rel_change:.4f}")

                # Re-quantize to FP8 if needed
                if orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    fp8_max = _FP8_MAX.get(orig_dtype, 448.0)
                    abs_max = W_new.abs().max().float()
                    new_scale = abs_max / fp8_max
                    if new_scale < 1e-12:
                        new_scale = torch.tensor(1.0)
                    state_dict[target_key] = (W_new / new_scale).clamp(-fp8_max, fp8_max).to(orig_dtype)
                    if has_scale:
                        state_dict[scale_key] = new_scale
                else:
                    state_dict[target_key] = W_new.to(orig_dtype)

                baked_keys.append(target_key)

            del lora_sd
            gc.collect()

        logger.info(f"  Baked {len(baked_keys)} weight tensors total")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(output_path), metadata=original_metadata)

        # Save manifest
        manifest = {
            "operation": "lora_bake",
            "source": str(model_path),
            "loras": [{"path": str(p), "strength": s} for p, s in lora_paths],
            "baked_keys": len(baked_keys),
            "checksum": _file_checksum(output_path),
        }
        output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

        del state_dict
        gc.collect()

        logger.info(f"  Saved -> {output_path} ({output_path.stat().st_size / 1e9:.1f} GB)")
        return output_path

    # ------------------------------------------------------------------
    # Capability Grafting -- SVD-based spectral steering
    # ------------------------------------------------------------------

    def operate(
        self,
        model_path: Path,
        output_path: Path,
        directions: Dict[str, List[CapabilityDirection]],
        layer_names: List[str],
    ) -> Path:
        """
        Apply capability directions to model weights via spectral steering.

        For each targeted weight matrix W = U Sigma V^T:
          1. Project capability direction v into U's singular vector space
          2. Compute alignment: a = |U^T @ v|
          3. Boost aligned singular values: sigma_new = sigma * (1 + strength * a^2)
          4. Safety cap: max 2x boost per singular value
          5. Clamp total relative change to max_delta_norm
          6. Reconstruct: W_new = U @ diag(sigma_new) @ V^T

        Args:
            model_path: Source checkpoint
            output_path: Where to save the grafted checkpoint
            directions: {capability_name: [CapabilityDirection, ...]}
            layer_names: Ordered list of layer names

        Returns:
            Path to grafted checkpoint
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        logger.info(f"Spectral steering: {model_path.name} -> {output_path.name}")

        original_metadata = _read_metadata(model_path)

        # Load from output if it exists (preserves accumulated grafts)
        load_path = output_path if output_path.exists() and output_path != model_path else model_path
        logger.info(f"  Loading weights from: {load_path.name}")
        state_dict = load_file(str(load_path))
        modified_keys = []

        # Index capability directions by block name
        block_directions: Dict[str, List[CapabilityDirection]] = {}
        for cap_name, cap_dirs in directions.items():
            for d in cap_dirs:
                block_directions.setdefault(d.layer_name, []).append(d)

        # Filter eligible keys
        eligible_keys = [
            key for key in sorted(state_dict.keys())
            if any(key.endswith(target) for target in self.surgery_targets)
            and not _is_protected(key, self.protected_patterns)
            and state_dict[key].dim() == 2
        ]
        logger.info(f"  {len(eligible_keys)} eligible weight matrices")

        for idx, key in enumerate(eligible_keys):
            W = state_dict[key]
            if idx % 50 == 0:
                gc.collect()
                logger.info(f"  Processing {idx+1}/{len(eligible_keys)}...")

            # Find matching capability directions (boundary-aware)
            matching_dirs = []
            for bn, dirs in block_directions.items():
                if (bn + ".") in key or key.startswith(bn + "."):
                    matching_dirs.extend(dirs)

            if not matching_dirs:
                continue

            orig_dtype = W.dtype

            # Handle FP8
            scale_key = key.replace(".weight", ".weight_scale")
            has_scale = scale_key in state_dict

            if orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                W_float = W.to(torch.bfloat16).float()
                if has_scale:
                    W_float = W_float * state_dict[scale_key].float()
                was_fp8 = True
            else:
                W_float = W.float()
                was_fp8 = False

            d_out, d_in = W_float.shape
            min_dim = min(d_out, d_in)

            # SVD -- use low-rank for large matrices (>5x faster)
            use_lowrank = min_dim > 512
            svd_rank = min(256, min_dim)

            try:
                if use_lowrank:
                    U, sigma, V = torch.svd_lowrank(W_float, q=svd_rank)
                    Vt = V.T
                else:
                    U, sigma, Vt = torch.linalg.svd(W_float, full_matrices=False)
            except Exception as e:
                logger.warning(f"  SVD failed for {key}: {e}")
                continue

            k = sigma.shape[0]

            # Compute spectral boost from all matching directions
            boost = torch.ones(k)

            for cap_dir in matching_dirs:
                v = cap_dir.direction.float()
                cap_strength = cap_dir.strength

                # Project into U's output singular vector space
                v_proj = torch.zeros(d_out)
                overlap = min(v.shape[0], d_out)
                v_proj[:overlap] = v[:overlap]
                v_proj = v_proj / (v_proj.norm() + 1e-8)

                # Alignment: how much each singular vector aligns
                alignment = (U.T @ v_proj).abs()  # [k]

                # Boost proportional to alignment^2
                boost = boost * (1.0 + cap_strength * alignment[:k] ** 2)

            # Safety: cap max boost at 2x
            max_ratio = boost.max()
            if max_ratio > 2.0:
                excess = (boost - 1.0).clamp(min=0)
                excess = excess * (1.0 / (max_ratio / 2.0))
                boost = 1.0 + excess

            if use_lowrank:
                # Delta approach: only modify top-k components
                delta_sigma = sigma * (boost - 1.0)
                W_new = W_float + U @ torch.diag(delta_sigma) @ Vt
            else:
                sigma_new = sigma * boost.to(sigma.device)
                W_new = U @ torch.diag(sigma_new) @ Vt

            # Safety checks
            if self.safety_checks:
                rel_change = (W_new - W_float).norm() / (W_float.norm() + 1e-8)
                if rel_change > self.max_delta_norm:
                    scale = self.max_delta_norm / rel_change
                    W_new = W_float + scale * (W_new - W_float)

                if torch.isnan(W_new).any() or torch.isinf(W_new).any():
                    logger.error(f"  NaN/Inf in {key} -- SKIPPED")
                    continue

            # Re-quantize
            if was_fp8:
                fp8_max = _FP8_MAX.get(orig_dtype, 448.0)
                abs_max = W_new.abs().max().float()
                new_scale = abs_max / fp8_max
                if new_scale < 1e-12:
                    new_scale = torch.tensor(1.0)
                state_dict[key] = (W_new / new_scale).clamp(-fp8_max, fp8_max).to(orig_dtype)
                if has_scale:
                    state_dict[scale_key] = new_scale
            else:
                state_dict[key] = W_new.to(orig_dtype)
            modified_keys.append(key)

            del W_float, W_new, U, sigma, Vt, boost
            if 'V' in dir():
                del V

        if len(modified_keys) == 0:
            logger.warning(
                "WARNING: Zero weight matrices were modified! This likely means your model "
                "uses different weight naming than the default surgery_targets. "
                f"Current targets: {self.surgery_targets}. "
                "Check your model's key names and pass custom surgery_targets to WeightSurgeon()."
            )

        logger.info(f"Modified {len(modified_keys)} weight tensors via spectral steering")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(output_path), metadata=original_metadata)

        manifest = {
            "operation": "spectral_steering",
            "source": str(model_path),
            "capabilities": list(directions.keys()),
            "modified_keys": len(modified_keys),
            "max_delta_norm": self.max_delta_norm,
            "checksum": _file_checksum(output_path),
        }
        output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

        del state_dict
        gc.collect()

        logger.info(f"Saved -> {output_path}")
        return output_path
