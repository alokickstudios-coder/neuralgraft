"""
Activation Harvester - Collects DiT intermediate activations on calibration data.

Loads a DiT model checkpoint, extracts per-layer output projection weights,
and computes content-aware activation proxies by projecting calibration
video frames through those weights at multiple noise levels.

No model instantiation needed -- works directly on the safetensors checkpoint.

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import logging
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("neuralgraft.harvester")

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def _check_ffmpeg():
    """Check if ffmpeg and ffprobe are available."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg and ffprobe are required but not found in PATH. "
            "Install ffmpeg:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: download from https://ffmpeg.org/download.html"
        )


def _find_video_files(directory: Path, max_files: int = None) -> List[Path]:
    """Find video files in directory, supporting multiple formats."""
    files = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(f)
    if not files:
        # Try recursive search
        for f in sorted(directory.rglob("*")):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(f)
    if max_files:
        files = files[:max_files]
    return files


def _load_video_frames(video_path: Path, n_frames: int = 9) -> Optional[torch.Tensor]:
    """Load n_frames evenly spaced from a video file.

    Returns: [T, C, H, W] in [0, 1] range, or None if loading fails.
    """
    try:
        import subprocess
        import json

        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path)
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(probe.stdout)

        duration = float(info["format"]["duration"])
        if duration < 1.0:
            return None

        safe_end = duration - 0.5
        timestamps = [safe_end * i / max(1, n_frames - 1) for i in range(n_frames)]

        frames = []
        for ts in timestamps:
            cmd = [
                "ffmpeg", "-ss", f"{ts:.3f}", "-i", str(video_path),
                "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-vf", "scale=384:640",
                "-loglevel", "quiet", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if result.returncode == 0 and len(result.stdout) > 0:
                raw = np.frombuffer(result.stdout, dtype=np.uint8)
                expected = 384 * 640 * 3
                if len(raw) >= expected:
                    frame = raw[:expected].reshape(640, 384, 3).copy()
                    frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)

        if len(frames) < n_frames // 2:
            return None

        return torch.stack(frames)

    except Exception as e:
        logger.warning(f"Failed to load {video_path}: {e}")
        return None


def _resize_frames(frames: torch.Tensor, h: int = 640, w: int = 384) -> torch.Tensor:
    """Resize frames to target resolution. Input: [T, C, H, W]."""
    import torch.nn.functional as F
    return F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)


class ActivationHarvester:
    """Collects DiT activations on calibration data without training.

    Works by:
      1. Loading only the output projection weights from the checkpoint
      2. Projecting calibration frame content through those weights
      3. Repeating at multiple noise levels to simulate denoising stages

    This produces activation proxies that correlate with real DiT activations
    without needing to instantiate or run the full model.

    Args:
        device: Torch device for computation (default: "cuda:0")
        dtype: Computation dtype (default: torch.bfloat16)
    """

    def __init__(self, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self._calibration_frames = []
        self._video_files = []

    def harvest(
        self,
        model_path: Path,
        calibration_dir: Path,
        n_clips: int = 12,
        n_frames: int = 9,
        target_h: int = 640,
        target_w: int = 384,
        noise_levels: List[float] = None,
        block_pattern: str = "transformer_blocks",
        target_suffixes: List[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Harvest activations from a DiT checkpoint on calibration clips.

        Args:
            model_path: Path to safetensors checkpoint
            calibration_dir: Directory containing calibration video clips (.mp4/.webm)
            n_clips: Number of clips to use (default: 12)
            n_frames: Frames per clip (default: 9)
            target_h: Target height for frame resize (default: 640)
            target_w: Target width for frame resize (default: 384)
            noise_levels: Denoising sigma levels to sample (default: [0.2, 0.5, 0.8])
            block_pattern: Pattern to identify transformer blocks (default: "transformer_blocks")
            target_suffixes: Weight name suffixes to load (default: attention + FFN outputs)

        Returns:
            activations: {layer_name: [N_total, d_hidden]} activation proxy matrices
            layer_names: Ordered list of discovered layer names
        """
        from safetensors import safe_open

        if noise_levels is None:
            noise_levels = [0.2, 0.5, 0.8]

        if target_suffixes is None:
            target_suffixes = [
                "attn1.to_out.0.weight", "attn2.to_out.0.weight",
                "attn.to_out.0.weight", "ff.net.2.weight",
            ]

        # --- Check ffmpeg ---
        _check_ffmpeg()

        # --- Load calibration clips ---
        cal_dir = Path(calibration_dir).expanduser()
        video_files = _find_video_files(cal_dir, max_files=n_clips)
        if not video_files:
            supported = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise FileNotFoundError(
                f"No video files found in {cal_dir}. "
                f"Supported formats: {supported}"
            )

        logger.info(f"Loading {len(video_files)} calibration clips from {cal_dir}")

        all_frames = []
        loaded_video_files = []  # Track which files actually loaded (for codec alignment)
        for vf in video_files:
            frames = _load_video_frames(vf, n_frames)
            if frames is not None:
                frames = _resize_frames(frames, target_h, target_w)
                all_frames.append(frames)
                loaded_video_files.append(vf)
            else:
                logger.warning(f"  Skipped {vf.name} (failed to load)")

        if not all_frames:
            raise ValueError("No valid calibration clips loaded")

        logger.info(f"Loaded {len(all_frames)}/{len(video_files)} clips x {n_frames} frames")

        # --- Selectively load output projection weights ---
        logger.info(f"Loading target weights from {model_path.name}...")

        block_weights = {}
        layer_names_set = set()

        with safe_open(str(model_path), framework="pt") as f:
            for key in f.keys():
                if block_pattern not in key:
                    continue
                is_target = any(key.endswith(s) for s in target_suffixes)
                is_scale = any(key.endswith(s.replace(".weight", ".weight_scale")) for s in target_suffixes)
                if not (is_target or is_scale):
                    continue

                tensor = f.get_tensor(key)

                # Extract "block_pattern.INDEX" from key
                # e.g., "model.transformer_blocks.5.attn1.to_out.0.weight" -> "transformer_blocks.5"
                idx = key.find(block_pattern)
                if idx >= 0:
                    rest = key[idx + len(block_pattern) + 1:]
                    block_idx = rest.split(".")[0]
                    block_name = f"{block_pattern}.{block_idx}"
                    block_weights.setdefault(block_name, {})[key] = tensor
                    layer_names_set.add(block_name)

        # Sort by block index (numeric if possible, lexicographic fallback)
        def _sort_key(name: str):
            idx = name.split(".")[-1]
            try:
                return (0, int(idx))
            except ValueError:
                return (1, idx)

        layer_names = sorted(layer_names_set, key=_sort_key)
        total_loaded = sum(len(v) for v in block_weights.values())
        logger.info(f"Loaded {total_loaded} weight tensors across {len(layer_names)} blocks")

        # --- Extract content-aware activation proxies ---
        logger.info("Extracting content-aware activation proxies...")

        activations = {}

        frame_dim = all_frames[0].shape[1] * all_frames[0].shape[2] * all_frames[0].shape[3]
        clip_flat = []
        for clip_frames in all_frames:
            avg_frame = clip_frames.mean(dim=0).flatten()
            clip_flat.append(avg_frame)
        clip_flat_stack = torch.stack(clip_flat)

        proj_cache = {}

        for block_name in layer_names:
            weights = block_weights[block_name]

            out_key = None
            for k in weights:
                if any(k.endswith(p) for p in target_suffixes) and weights[k].dim() == 2:
                    out_key = k
                    break

            if out_key is None:
                continue

            W_raw = weights[out_key]
            if W_raw.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                W = W_raw.to(torch.bfloat16).float()
                scale_key = out_key.replace(".weight", ".weight_scale")
                if scale_key in weights:
                    W = W * weights[scale_key].float()
            else:
                W = W_raw.float()

            if W.dim() != 2:
                continue

            d_out, d_in = W.shape

            if d_in not in proj_cache:
                gen = torch.Generator().manual_seed(42 + d_in)
                proj = torch.randn(frame_dim, d_in, generator=gen)
                proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
                proj_cache[d_in] = proj

            X_content = clip_flat_stack @ proj_cache[d_in]

            H_parts = []
            for sigma in noise_levels:
                X_noised = X_content * (1.0 - sigma) + torch.randn_like(X_content) * sigma * 0.1
                H_part = X_noised @ W.T
                H_parts.append(H_part)

            H = torch.cat(H_parts, dim=0)
            activations[block_name] = H

        del block_weights, proj_cache
        gc.collect()

        logger.info(f"Harvested {len(activations)} layer activations, "
                     f"shape: {list(activations.values())[0].shape if activations else 'empty'}")

        self._calibration_frames = all_frames
        self._video_files = loaded_video_files  # Only successfully loaded files

        return activations, layer_names

    @property
    def calibration_frames(self) -> List[torch.Tensor]:
        """Access the loaded calibration frames after harvesting."""
        return self._calibration_frames

    @property
    def calibration_video_files(self) -> List[Path]:
        """Access the calibration video file paths after harvesting."""
        return self._video_files
