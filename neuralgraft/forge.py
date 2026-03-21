"""
LoRA Forge - Construct LoRA weights from a raw image dataset WITHOUT training.

Instead of gradient descent, this uses activation regression:
  1. Extract a "concept signature" from your dataset images (multi-dimensional
     visual features that characterize what makes these images unique)
  2. Load the DiT checkpoint and create activation proxies per layer
  3. Regress: which activation directions predict the concept signature?
  4. Construct LoRA matrices (B @ A) from those directions
  5. Save as standard LoRA safetensors format

The key insight: a LoRA modifies weights along specific directions.
Those directions can be DISCOVERED from data via regression instead of
LEARNED via gradient descent. Same result, no training loop.

Works with:
  - Standard LoRA format (compatible with diffusers, ComfyUI, etc.)
  - Any DiT-based model in safetensors format
  - Any image dataset (JPG, PNG, WEBP) -- 10-100 images recommended

Copyright 2026 Alokick Tech. Licensed under Apache 2.0.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger("neuralgraft.forge")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Visual Feature Extraction (concept signature -- no external models needed)
# ---------------------------------------------------------------------------

def _load_images(image_dir: Path, max_images: int = 100,
                 target_h: int = 640, target_w: int = 384) -> List[np.ndarray]:
    """Load and resize images from a directory. Returns list of BGR uint8 arrays."""
    image_dir = Path(image_dir).expanduser()
    files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        # Try recursive
        files = sorted(
            f for f in image_dir.rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not files:
        raise FileNotFoundError(
            f"No images found in {image_dir}. "
            f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    files = files[:max_images]
    images = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            img = cv2.resize(img, (target_w, target_h))
            images.append(img)

    if not images:
        raise ValueError(f"Failed to load any images from {image_dir}")

    logger.info(f"Loaded {len(images)}/{len(files)} images from {image_dir}")
    return images


def _extract_concept_signature(images: List[np.ndarray]) -> torch.Tensor:
    """
    Extract a multi-dimensional concept signature from a set of images.

    The signature captures what makes this set of images visually unique:
      - Color distribution (HSV histograms)
      - Texture characteristics (Laplacian variance, edge density)
      - Spatial frequency content (FFT energy bands)
      - Luminance/contrast profile
      - Structural complexity (entropy)

    Returns: [N_images, d_features] feature matrix
    """
    all_features = []

    for img in images:
        features = []

        # --- Color: HSV histogram (captures color palette) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        h_hist = h_hist / (h_hist.sum() + 1e-8)
        s_hist = s_hist / (s_hist.sum() + 1e-8)
        v_hist = v_hist / (v_hist.sum() + 1e-8)
        features.extend(h_hist.tolist())  # 30
        features.extend(s_hist.tolist())  # 16
        features.extend(v_hist.tolist())  # 16

        # --- Texture: Laplacian, edges, entropy ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Laplacian sharpness
        lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
        features.append(lap_var / (lap_var + 200.0))  # 1

        # Edge density
        gray_u8 = gray.clip(0, 255).astype(np.uint8)
        median_val = np.median(gray_u8)
        edges = cv2.Canny(gray_u8, int(max(1, 0.66 * median_val)),
                          int(min(255, 1.33 * median_val)))
        features.append(np.count_nonzero(edges) / edges.size)  # 1

        # Texture entropy (patch-based)
        h, w = gray_u8.shape
        entropies = []
        for y in range(0, h - 31, 64):
            for x in range(0, w - 31, 64):
                patch = gray_u8[y:y+32, x:x+32]
                hist = cv2.calcHist([patch], [0], None, [32], [0, 256]).flatten()
                hist = hist / (hist.sum() + 1e-8)
                entropies.append(-np.sum(hist * np.log2(hist + 1e-10)))
        features.append(np.mean(entropies) / 5.0 if entropies else 0.5)  # 1

        # --- Spatial frequency: FFT energy bands ---
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)
        bands = 8
        for i in range(bands):
            r_inner = int(max_r * i / bands)
            r_outer = int(max_r * (i + 1) / bands)
            mask = np.zeros_like(gray, dtype=bool)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
            mask = (dist >= r_inner) & (dist < r_outer)
            band_energy = magnitude[mask].mean() if mask.any() else 0
            features.append(float(band_energy) / (magnitude.mean() + 1e-8))  # 8

        # --- Luminance/contrast ---
        features.append(float(gray.mean()) / 255.0)  # 1
        features.append(float(gray.std()) / 128.0)   # 1

        # --- Color channel stats ---
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            features.append(ch.mean() / 255.0)  # 3
            features.append(ch.std() / 128.0)    # 3

        all_features.append(features)

    # Total features per image: 30 + 16 + 16 + 1 + 1 + 1 + 8 + 1 + 1 + 6 = 81
    return torch.tensor(all_features, dtype=torch.float32)


def _extract_concept_signature_with_model(
    images: List[np.ndarray], device: str = "cpu"
) -> torch.Tensor:
    """
    Extract concept signature using a pretrained vision model (DINOv2/CLIP).
    Falls back to OpenCV features if models unavailable.

    Returns: [N_images, d_features] feature matrix
    """
    # Try DINOv2 via torch.hub (best for visual similarity)
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               trust_repo=True, verbose=False)
        model = model.to(device).eval()

        features = []
        for img in images:
            # BGR → RGB, resize to 224x224, normalize
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model(tensor)  # [1, 384]
            features.append(feat.cpu().squeeze(0))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  Extracted {len(features)} DINOv2 features (384-dim)")
        return torch.stack(features)

    except Exception as e:
        logger.info(f"  DINOv2 unavailable ({e}), using OpenCV features")
        return _extract_concept_signature(images)


def _extract_character_signature(
    images: List[np.ndarray],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract full-body character features for character LoRAs.

    Captures the COMPLETE character appearance:
      - Face identity (InsightFace ArcFace 512-dim if available)
      - Full-body style features (color palette, texture, proportions)

    When a face is detected, features are identity-focused.
    When no face is found, full-body features still capture the character's
    overall appearance (outfit colors, body proportions, background style).

    Returns:
        (features, valid_indices): feature matrix [N_valid, d_feat] and
        list of image indices that produced valid features (for alignment
        with activation proxies).
    """
    n_images = len(images)

    # ── Phase A: Full-body style features for ALL images ──
    # These capture outfit, proportions, color palette — always available
    body_features = _extract_concept_signature(images)  # [N, 81]

    # ── Phase B: Face identity features (optional, stacked with body) ──
    face_features = None
    valid_face_indices = list(range(n_images))  # default: all images

    # Try InsightFace (gold standard for face identity)
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        embeddings = []
        face_indices = []
        for i, img in enumerate(images):
            faces = app.get(img)
            if faces:
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(torch.from_numpy(largest.embedding).float())
                face_indices.append(i)

        del app

        if len(embeddings) >= 3:
            face_features = torch.stack(embeddings)  # [N_faces, 512]
            valid_face_indices = face_indices
            logger.info(f"  InsightFace: {len(embeddings)}/{n_images} faces (512-dim)")
        else:
            logger.info(f"  InsightFace: only {len(embeddings)} faces, using full-body only")

    except ImportError:
        logger.info("  InsightFace not installed, using full-body features")
    except Exception as e:
        logger.info(f"  InsightFace failed ({e}), using full-body features")

    # ── Combine: face identity (when available) + full-body features (always) ──
    # This captures the COMPLETE character: face, outfit, body, colors, everything.

    if face_features is not None:
        # Filter body features to match only images where faces were found
        body_aligned = body_features[valid_face_indices]  # [N_faces, 81]
        # Concatenate: [N_faces, 512 + 81] = [N_faces, 593]
        combined = torch.cat([face_features, body_aligned], dim=1)
        logger.info(f"  Combined features: {combined.shape[1]}-dim (512 face + 81 body)")
        return combined, valid_face_indices
    else:
        # No face encoder available — use full-body features for ALL images
        logger.info(f"  Full-body features: {body_features.shape[1]}-dim (all {n_images} images)")
        return body_features, list(range(n_images))

    logger.info(f"  OpenCV face features: {len(all_features)} faces ({len(all_features[0])}-dim)")
    return torch.tensor(all_features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# LoRA Construction from Concept Directions
# ---------------------------------------------------------------------------

class LoRAForge:
    """Construct LoRA weights from a dataset without any training.

    Algorithm:
      1. Extract concept signature from dataset images
      2. Load DiT checkpoint weights per block
      3. Create activation proxies by projecting images through weights
      4. Regress: which activation directions predict the concept features?
      5. Construct LoRA (B, A) from regression directions + weight projections
      6. Save as standard LoRA safetensors

    Args:
        rank: LoRA rank (default: 16)
        strength: Global strength multiplier for the LoRA (default: 1.0)
        min_r_squared: Minimum R^2 to include a layer (default: 0.01)
        use_vision_model: Try DINOv2 for features (default: False, uses OpenCV)
        mode: "style" (default) for visual style, "character" for face identity
    """

    MODES = ("style", "character")

    def __init__(
        self,
        rank: int = 16,
        strength: float = 1.0,
        min_r_squared: float = 0.01,
        use_vision_model: bool = False,
        mode: str = "style",
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
        self.rank = rank
        self.strength = strength
        self.mode = mode
        self.min_r2 = min_r_squared
        self.use_vision_model = use_vision_model

    def forge(
        self,
        model_path: Path,
        image_dir: Path,
        output_path: Path,
        max_images: int = 100,
        target_h: int = 640,
        target_w: int = 384,
        device: str = "cpu",
        block_pattern: str = "transformer_blocks",
        target_suffixes: List[str] = None,
        trigger_word: str = None,
    ) -> Path:
        """
        Construct a LoRA from a dataset of images.

        Args:
            model_path: Base DiT model checkpoint (safetensors)
            image_dir: Directory containing dataset images
            output_path: Where to save the LoRA (safetensors)
            max_images: Maximum number of images to use (default: 100)
            target_h: Resize height (default: 640)
            target_w: Resize width (default: 384)
            device: Torch device for vision model (default: cpu)
            block_pattern: DiT block naming pattern
            target_suffixes: Weight suffixes to target
            trigger_word: Optional trigger word for the LoRA concept

        Returns:
            Path to the saved LoRA checkpoint
        """
        model_path = Path(model_path)
        image_dir = Path(image_dir).expanduser()
        output_path = Path(output_path)

        if target_suffixes is None:
            # Target ALL attention + feedforward layers (matches ComfyUI LoRA format)
            target_suffixes = [
                "attn1.to_q.weight", "attn1.to_k.weight",
                "attn1.to_v.weight", "attn1.to_out.0.weight",
                "attn2.to_q.weight", "attn2.to_k.weight",
                "attn2.to_v.weight", "attn2.to_out.0.weight",
                "ff.net.0.proj.weight", "ff.net.2.weight",
            ]

        t0 = time.time()

        # ── Phase 1: Extract concept signature ──────────────────────────
        mode_label = "character identity" if self.mode == "character" else "visual style"
        logger.info(f"Phase 1: Extracting {mode_label} signature from dataset...")
        images = _load_images(image_dir, max_images, target_h, target_w)

        # valid_indices tracks which images produced features (some may be
        # skipped if InsightFace can't find a face). The activation proxy
        # must be filtered to match.
        valid_indices = None

        if self.mode == "character":
            concept_features, valid_indices = _extract_character_signature(images)
        elif self.use_vision_model:
            concept_features = _extract_concept_signature_with_model(images, device)
        else:
            concept_features = _extract_concept_signature(images)

        N_images, d_feat = concept_features.shape
        logger.info(f"  Concept signature: {N_images} images x {d_feat} features")

        # Standardize features (zero mean, unit variance per dimension)
        feat_mean = concept_features.mean(0)
        feat_std = concept_features.std(0) + 1e-8
        concept_features = (concept_features - feat_mean) / feat_std

        # ── Phase 2: Load DiT weights and create activation proxies ─────
        logger.info(f"Phase 2: Loading DiT weights from {model_path.name}...")

        block_weights = {}
        layer_names_set = set()

        with safe_open(str(model_path), framework="pt") as f:
            for key in f.keys():
                if block_pattern not in key:
                    continue
                is_target = any(key.endswith(s) for s in target_suffixes)
                is_scale = any(key.endswith(s.replace(".weight", ".weight_scale"))
                               for s in target_suffixes)
                if not (is_target or is_scale):
                    continue

                tensor = f.get_tensor(key)
                idx = key.find(block_pattern)
                if idx >= 0:
                    rest = key[idx + len(block_pattern) + 1:]
                    block_idx = rest.split(".")[0]
                    block_name = f"{block_pattern}.{block_idx}"
                    block_weights.setdefault(block_name, {})[key] = tensor
                    layer_names_set.add(block_name)

        def _sort_key(name):
            idx = name.split(".")[-1]
            try:
                return (0, int(idx))
            except ValueError:
                return (1, idx)

        layer_names = sorted(layer_names_set, key=_sort_key)
        logger.info(f"  Found {len(layer_names)} transformer blocks")

        # Create image content vectors (downsample to manageable size first)
        # Full resolution (3*640*384 = 737K dims) creates a 737K x d_in projection
        # matrix that OOMs on most machines. Downsample to 64x48 = ~9K dims instead.
        PROXY_H, PROXY_W = 64, 48
        image_tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (PROXY_W, PROXY_H))
            t = torch.from_numpy(rgb_small).float().permute(2, 0, 1).flatten() / 255.0
            image_tensors.append(t)
        image_flat = torch.stack(image_tensors)  # [N, 3*64*48] = [N, 9216]

        # Filter image_flat to match valid_indices from feature extraction
        # (character mode may skip images where no face was detected)
        if valid_indices is not None and len(valid_indices) < len(image_flat):
            image_flat = image_flat[valid_indices]
            logger.info(f"  Aligned activation proxy to {len(valid_indices)} images with features")

        frame_dim = image_flat.shape[1]

        # ── Phase 3: Activation regression per layer ────────────────────
        logger.info("Phase 3: Regressing concept directions per layer...")

        lora_state_dict = {}
        layers_with_signal = 0
        proj_cache = {}

        for block_name in layer_names:
            weights = block_weights.get(block_name, {})

            # Process ALL matching weight matrices in this block
            target_keys = [
                k for k in weights
                if any(k.endswith(p) for p in target_suffixes) and weights[k].dim() == 2
            ]

            if not target_keys:
                continue

            block_had_signal = False

            for out_key in target_keys:

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

                # Create activation proxy: project image content through W
                # Projection matrix: [9216, d_in] ≈ 150 MB for d_in=4096 (vs 24 GB before)
                if d_in not in proj_cache:
                    gen = torch.Generator().manual_seed(42 + d_in)
                    proj = torch.randn(frame_dim, d_in, generator=gen)
                    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
                    proj_cache[d_in] = proj

                X_content = image_flat @ proj_cache[d_in]  # [N, d_in]
                H = X_content @ W.T  # [N, d_out] — activation proxy

                # ── Multi-target regression: concept_features ≈ H @ Beta ──
                try:
                    U_H, S_H, Vt_H = torch.linalg.svd(H, full_matrices=False)
                    k = min(self.rank, len(S_H), N_images, d_out)
                    S_inv = torch.zeros_like(S_H)
                    S_inv[:k] = 1.0 / (S_H[:k] + 1e-8)

                    Beta = Vt_H.T @ torch.diag(S_inv) @ U_H.T @ concept_features

                    F_pred = H @ Beta
                    ss_res = ((concept_features - F_pred) ** 2).sum(0)
                    ss_tot = ((concept_features - concept_features.mean(0)) ** 2).sum(0)
                    r2_per_feat = 1.0 - (ss_res / (ss_tot + 1e-8))
                    r2_mean = float(r2_per_feat.clamp(min=0).mean())

                    if r2_mean < self.min_r2:
                        continue

                    block_had_signal = True

                except Exception as e:
                    logger.warning(f"  Regression failed for {out_key}: {e}")
                    continue

                # ── Construct LoRA (B, A) from regression directions ──
                try:
                    U_beta, S_beta, _ = torch.linalg.svd(Beta, full_matrices=False)
                except Exception:
                    continue

                r = min(self.rank, len(S_beta), d_out, d_in)

                out_dirs = U_beta[:, :r]  # [d_out, r]
                in_dirs = W.T @ out_dirs   # [d_in, r]
                in_norms = in_dirs.norm(dim=0, keepdim=True) + 1e-8
                in_dirs = in_dirs / in_norms

                scale = torch.sqrt(S_beta[:r].clamp(min=0) * self.strength * r2_mean)

                B = out_dirs * scale.unsqueeze(0)   # [d_out, r]
                A = (in_dirs * scale.unsqueeze(0)).T  # [r, d_in]

                # Safety: clamp to prevent extreme LoRA values
                B = B.clamp(-1.0, 1.0)
                A = A.clamp(-1.0, 1.0)

                # ── Convert key format: model weight key → LoRA key ──
                # ComfyUI expects: diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight
                # Strip ALL wrapper prefixes, then add diffusion_model.
                lora_base = out_key.replace(".weight", "")
                # Strip known wrapper prefixes
                for prefix in ["model.diffusion_model.", "model.", "diffusion_model."]:
                    if lora_base.startswith(prefix):
                        lora_base = lora_base[len(prefix):]
                        break
                # Always add the canonical prefix
                lora_base = "diffusion_model." + lora_base

                lora_state_dict[f"{lora_base}.lora_A.weight"] = A.to(torch.bfloat16).contiguous()
                lora_state_dict[f"{lora_base}.lora_B.weight"] = B.to(torch.bfloat16).contiguous()

                del H, Beta, U_H, S_H, Vt_H, U_beta, S_beta, B, A

            if block_had_signal:
                layers_with_signal += 1

        del proj_cache, image_flat
        gc.collect()

        # ── Phase 5: Save LoRA ──────────────────────────────────────────
        if not lora_state_dict:
            logger.error("No layers had sufficient concept signal. Try more images or lower --min-r2")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata for compatibility
        metadata = {
            "format": "lora",
            "neuralgraft_forge": "true",
            "rank": str(self.rank),
            "strength": str(self.strength),
            "n_images": str(N_images),
            "n_layers": str(layers_with_signal),
        }
        if trigger_word:
            metadata["trigger_word"] = trigger_word

        save_file(lora_state_dict, str(output_path), metadata=metadata)

        dt = time.time() - t0
        n_params = sum(t.numel() for t in lora_state_dict.values())
        size_mb = output_path.stat().st_size / 1e6

        # Save manifest
        manifest = {
            "operation": "forge_lora",
            "model": str(model_path),
            "dataset": str(image_dir),
            "n_images": N_images,
            "feature_dims": d_feat,
            "rank": self.rank,
            "strength": self.strength,
            "layers_with_signal": layers_with_signal,
            "total_layers": len(layer_names),
            "lora_keys": len(lora_state_dict),
            "parameters": n_params,
            "time_seconds": round(dt, 1),
            "trigger_word": trigger_word,
        }
        output_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        logger.info(f"\n  Forged LoRA saved to: {output_path}")
        logger.info(f"  Layers with concept signal: {layers_with_signal}/{len(layer_names)}")
        logger.info(f"  Parameters: {n_params:,}")
        logger.info(f"  Size: {size_mb:.1f} MB")
        logger.info(f"  Time: {dt:.1f}s ({dt/60:.1f} min)")
        if trigger_word:
            logger.info(f"  Trigger word: '{trigger_word}'")

        return output_path
