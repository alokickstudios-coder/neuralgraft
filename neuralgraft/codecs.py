"""
Source Model Codecs - Universal adapters that extract capability scores
from ANY external model.

Each codec:
  1. Loads a source model (or uses OpenCV/numpy for lightweight scoring)
  2. Runs it on calibration video frames
  3. Returns a score vector: [N_samples] or [N_samples, d_score]

Adding a new capability is as simple as writing a new codec class (~30 lines).
The codec just needs to implement `_load_model()` and `_score_frames()`.
NeuralGraft handles everything else.

Example:
    class MyCodec(BaseCodec):
        name = "my_capability"
        description = "Scores my custom quality metric"

        def _load_model(self):
            self.model = load_my_model()

        def _score_frames(self, frames):
            return self.model(frames)  # Returns [T] scores

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
import numpy as np

logger = logging.getLogger("neuralgraft.codecs")

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV (cv2) not found. Most codecs require it. Install with: pip install opencv-python")


class BaseCodec(ABC):
    """Base class for source model codecs.

    Subclass this to add new capabilities to NeuralGraft.
    You only need to implement `_load_model()` and `_score_frames()`.

    Attributes:
        name: Short identifier for this codec
        description: Human-readable description
    """

    name: str = "base"
    description: str = "Base codec"

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model = None

    @abstractmethod
    def _load_model(self):
        """Load the source model. Called once before scoring."""
        pass

    @abstractmethod
    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of frames.

        Args:
            frames: [T, C, H, W] in [0, 1] range

        Returns:
            scores: [T] capability scores (higher = more of this capability)
        """
        pass

    def _unload(self):
        """Release model memory."""
        self.model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(
        self,
        calibration_dir: Path,
        n_clips: int = 12,
        n_frames: int = 9,
        video_files: List[Path] = None,
    ) -> torch.Tensor:
        """Score calibration clips. Returns [n_clips * 3] scores (matching harvester).

        The x3 is because the harvester samples at 3 noise levels per clip.

        Args:
            calibration_dir: Directory containing calibration clips
            n_clips: Maximum number of clips to score
            n_frames: Frames per clip to extract
            video_files: Pre-resolved list of video files (ensures alignment
                         with harvester). If None, discovers files automatically.
        """
        logger.info(f"  [{self.name}] Loading model...")
        self._load_model()

        from .harvester import _load_video_frames, _resize_frames, _find_video_files

        if video_files is None:
            cal_dir = Path(calibration_dir).expanduser()
            video_files = _find_video_files(cal_dir, max_files=n_clips)

        clip_scores = []
        for vf in video_files:
            frames = _load_video_frames(vf, n_frames)
            if frames is None:
                clip_scores.append(0.5)
                continue
            frames = _resize_frames(frames, 640, 384)

            with torch.no_grad():
                s = self._score_frames(frames.to(self.device))
            clip_scores.append(float(s.cpu().mean()))

        self._unload()

        if not clip_scores:
            return torch.zeros(n_clips * 3)

        scores = torch.tensor(clip_scores, dtype=torch.float32)
        scores = scores.repeat(3)  # Match harvester's 3 noise levels

        logger.info(f"  [{self.name}] Scored {len(scores)} samples, "
                     f"range [{scores.min():.3f}, {scores.max():.3f}]")
        return scores


# -----------------------------------------------------------------------
# Built-in Lightweight Codecs (OpenCV + numpy only, no external models)
# -----------------------------------------------------------------------

class LaplacianSharpnessCodec(BaseCodec):
    """Scores frame sharpness using Laplacian variance."""
    name = "sharpness"
    description = "Frame sharpness via Laplacian variance (no model needed)"

    def _load_model(self):
        pass  # No model needed

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
            scores.append(lap_var / (lap_var + 200.0))  # Normalize to ~[0, 1]
        return torch.tensor(scores)


class EdgeDensityCodec(BaseCodec):
    """Scores edge detail density using Canny edge detection."""
    name = "edges"
    description = "Edge density via Canny detection (no model needed)"

    def _load_model(self):
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            median_val = np.median(gray)
            edges = cv2.Canny(gray, int(0.66 * median_val), int(1.33 * median_val))
            ratio = np.count_nonzero(edges) / edges.size
            scores.append(min(1.0, ratio / 0.15))
        return torch.tensor(scores)


class TemporalSSIMCodec(BaseCodec):
    """Scores temporal consistency via frame-to-frame SSIM."""
    name = "temporal_ssim"
    description = "Temporal consistency via inter-frame SSIM (no model needed)"

    def _load_model(self):
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            # Fallback: simple pixel-level mean absolute difference
            logger.warning("scikit-image not installed, using MAD fallback for SSIM. "
                           "Install with: pip install neuralgraft[full]")
            frames_np = frames.cpu().numpy()
            diffs = []
            for i in range(frames_np.shape[0] - 1):
                mad = np.mean(np.abs(frames_np[i] - frames_np[i + 1]))
                diffs.append(1.0 - min(1.0, mad * 5))  # Invert: low diff = high consistency
            avg = sum(diffs) / len(diffs) if diffs else 0.5
            return torch.tensor([avg] * frames.shape[0])

        scores = []
        frames_np = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for i in range(len(frames_np) - 1):
            gray_a = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(frames_np[i + 1], cv2.COLOR_RGB2GRAY)
            s = ssim(gray_a, gray_b)
            scores.append(s)
        if not scores:
            return torch.tensor([0.5] * frames.shape[0])
        avg = sum(scores) / len(scores)
        return torch.tensor([avg] * frames.shape[0])


class OpticalFlowCodec(BaseCodec):
    """Scores motion smoothness via optical flow variance."""
    name = "flow_smoothness"
    description = "Motion smoothness via optical flow (Farneback, no model needed)"

    def _load_model(self):
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames_np = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        flow_diffs = []
        prev_flow = None
        for i in range(len(frames_np) - 1):
            gray_a = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(frames_np[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if prev_flow is not None:
                diff = np.mean(np.abs(flow - prev_flow))
                flow_diffs.append(diff)
            prev_flow = flow
        if not flow_diffs:
            return torch.tensor([0.5] * frames.shape[0])
        # Lower variance = smoother = higher score
        smoothness = 1.0 / (1.0 + np.mean(flow_diffs))
        return torch.tensor([smoothness] * frames.shape[0])


class ColorConsistencyCodec(BaseCodec):
    """Scores color consistency across frames."""
    name = "color_consistency"
    description = "Color histogram consistency across frames (no model needed)"

    def _load_model(self):
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames_np = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        hists = []
        for f in frames_np:
            hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            h = h.flatten() / (h.sum() + 1e-8)
            hists.append(h)
        if len(hists) < 2:
            return torch.tensor([0.5] * frames.shape[0])
        diffs = []
        for i in range(len(hists) - 1):
            d = cv2.compareHist(hists[i].astype(np.float32), hists[i+1].astype(np.float32),
                                cv2.HISTCMP_CORREL)
            diffs.append(d)
        avg = sum(diffs) / len(diffs)
        return torch.tensor([max(0, avg)] * frames.shape[0])


class FaceStabilityCodec(BaseCodec):
    """Scores face region stability across frames (anti-morphing)."""
    name = "face_stability"
    description = "Face region stability via template matching (no model needed)"

    def _load_model(self):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames_np = (frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        face_sizes = []
        for f in frames_np:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) > 0:
                areas = [w * h for (_, _, w, h) in faces]
                face_sizes.append(max(areas))
            else:
                face_sizes.append(0)
        if not any(face_sizes):
            return torch.tensor([0.5] * frames.shape[0])
        # Stability = low variance in face sizes
        arr = np.array([s for s in face_sizes if s > 0], dtype=np.float32)
        if len(arr) < 2:
            return torch.tensor([0.5] * frames.shape[0])
        cv_coeff = arr.std() / (arr.mean() + 1e-8)
        stability = 1.0 / (1.0 + cv_coeff * 5)
        return torch.tensor([stability] * frames.shape[0])


class TextureEntropyCodec(BaseCodec):
    """Scores texture richness via patch histogram entropy."""
    name = "texture"
    description = "Texture richness via patch entropy (no model needed)"

    def _load_model(self):
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        scores = []
        for i in range(frames.shape[0]):
            frame = (frames[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Compute entropy over 32x32 patches
            h, w = gray.shape
            patch_entropies = []
            for y in range(0, h - 31, 32):
                for x in range(0, w - 31, 32):
                    patch = gray[y:y+32, x:x+32]
                    hist = cv2.calcHist([patch], [0], None, [64], [0, 256]).flatten()
                    hist = hist / (hist.sum() + 1e-8)
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    patch_entropies.append(entropy)
            avg_entropy = np.mean(patch_entropies) if patch_entropies else 3.0
            scores.append(avg_entropy / 6.0)  # Normalize: max entropy ~6 bits
        return torch.tensor(scores)


# -----------------------------------------------------------------------
# Codec Registry
# -----------------------------------------------------------------------

_CODEC_REGISTRY: Dict[str, Type[BaseCodec]] = {}


def register_codec(cls: Type[BaseCodec]) -> Type[BaseCodec]:
    """Decorator to register a codec in the global registry.

    Example:
        @register_codec
        class MyCodec(BaseCodec):
            name = "my_codec"
            ...
    """
    _CODEC_REGISTRY[cls.name] = cls
    return cls


def get_codec(name: str, device: str = "cuda:0") -> BaseCodec:
    """Get a codec by name. Falls back to built-in codecs."""
    if name in _CODEC_REGISTRY:
        return _CODEC_REGISTRY[name](device=device)

    # Built-in codecs
    builtins = {
        "sharpness": LaplacianSharpnessCodec,
        "edges": EdgeDensityCodec,
        "temporal_ssim": TemporalSSIMCodec,
        "flow_smoothness": OpticalFlowCodec,
        "color_consistency": ColorConsistencyCodec,
        "face_stability": FaceStabilityCodec,
        "texture": TextureEntropyCodec,
    }

    if name in builtins:
        return builtins[name](device=device)

    raise ValueError(
        f"Unknown codec: '{name}'. Available: {list(builtins.keys()) + list(_CODEC_REGISTRY.keys())}"
    )


def list_codecs() -> List[str]:
    """List all available codec names."""
    builtins = ["sharpness", "edges", "temporal_ssim", "flow_smoothness",
                "color_consistency", "face_stability", "texture"]
    return builtins + list(_CODEC_REGISTRY.keys())
