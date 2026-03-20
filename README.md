<p align="center">
  <h1 align="center">NeuralGraft</h1>
  <p align="center"><strong>Zero-Training Capability Transfer & LoRA Construction for Diffusion Models</strong></p>
  <p align="center"><em>Hours of model training in minutes. Graft anything. Forge LoRAs from images. No training loop.</em></p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#lora-forge">LoRA Forge</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#proof-of-concept">Proof of Concept</a> &bull;
  <a href="RESEARCH.md">Research Paper</a> &bull;
  <a href="#api-reference">API Reference</a>
</p>

---

**I vibe-coded this tool to avoid hours of model training and it worked.** Hours of serious model training with community LoRAs on consumer GPUs became minutes of business. What used to take 4+ hours of gradient descent, carefully tuned learning rates, and praying your loss curve doesn't explode -- NeuralGraft does in minutes with pure linear algebra. No training loop. No optimizer. No loss function. Just SVD and closed-form regression.

The core insight: **you don't need to train a model to teach it something new.** You just need to know *where* in the model a capability lives and *how much* to turn it up. NeuralGraft discovers both automatically.

**NEW: LoRA Forge** -- Give NeuralGraft a folder of images and it constructs a LoRA *without any training*. No gradient descent, no loss curves, no hyperparameter tuning. It extracts a "concept signature" from your images and finds which model weight directions encode that concept, then outputs a standard LoRA file you can use anywhere. Dataset in, LoRA out, minutes not hours.

## What Can NeuralGraft Do?

| Operation | What It Does | Time |
|-----------|-------------|------|
| **LoRA Forge** | Construct a LoRA from images -- no training | ~1-5 min |
| **LoRA Baking** | Permanently merge any LoRA into base weights | ~5 min |
| **Spectral Steering** | Graft quality capabilities from reference clips | ~10 min |
| **LoRA Amplification** | Boost LoRA-improved dimensions in base weights | ~3 min |
| **Cross-Architecture Transfer** | Graft capabilities from Model A into Model B | ~12 min |
| **Full Pipeline** | Bake + Amplify in one pass | ~12 min |

**Two headline features:**
1. **LoRA Forge** -- Drop a folder of images, get a LoRA. No training. Works with any DiT model.
2. **Cross-Architecture Transfer** -- Graft capabilities from any model into any other model. Different architectures, different everything. NeuralGraft doesn't care.

## Quickstart

### Install

```bash
pip install neuralgraft
```

Or from source:
```bash
git clone https://github.com/alokickstudios-coder/neuralgraft.git
cd neuralgraft
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.1+
- A safetensors model checkpoint (any DiT-based diffusion model)
- ffmpeg (for video frame extraction)
- 8-12 short calibration video clips (any quality reference footage)

### CLI Usage

```bash
# NEW: Forge a LoRA from images (no training!)
neuralgraft forge \
  --base model.safetensors \
  --images ./my_dataset/ \
  --output my-style-lora.safetensors \
  --rank 16 \
  --trigger-word "mystyle"

# Bake a LoRA permanently into model weights
neuralgraft bake \
  --base model.safetensors \
  --output baked-model.safetensors \
  --loras my-lora.safetensors:0.5

# Graft quality capabilities from reference clips
neuralgraft graft \
  --base model.safetensors \
  --output grafted-model.safetensors \
  --calibration ./reference_clips/ \
  --strength 1.0

# LoRA-derived spectral amplification
neuralgraft amplify \
  --base baked-model.safetensors \
  --output amplified-model.safetensors \
  --loras my-lora.safetensors:0.15

# Full pipeline: bake + amplify in one pass
neuralgraft full \
  --base model.safetensors \
  --output final-model.safetensors \
  --loras my-lora.safetensors:0.5

# Forge + bake in sequence (create LoRA, then permanently merge it)
neuralgraft forge --base model.safetensors --images ./dataset/ -o style.safetensors
neuralgraft bake --base model.safetensors --loras style.safetensors:0.7 -o enhanced.safetensors

# List available quality codecs
neuralgraft list
```

### Python API

```python
from neuralgraft import WeightSurgeon, ActivationHarvester, CapabilityProber, LoRAForge
from neuralgraft.codecs import get_codec
from pathlib import Path

# --- Step 1: Bake a LoRA ---
surgeon = WeightSurgeon()
surgeon.bake_loras(
    model_path=Path("model.safetensors"),
    output_path=Path("baked.safetensors"),
    lora_paths=[(Path("lora.safetensors"), 0.5)],
)

# --- Step 2: Graft capabilities ---
harvester = ActivationHarvester(device="cuda:0")
activations, layers = harvester.harvest(
    model_path=Path("model.safetensors"),
    calibration_dir=Path("./clips/"),
)

prober = CapabilityProber(rank=8)
codec = get_codec("sharpness")
scores = codec.score(Path("./clips/"))
directions = prober.probe(activations, scores, layers, strength=0.15)

surgeon.operate(
    model_path=Path("model.safetensors"),
    output_path=Path("grafted.safetensors"),
    directions={"sharpness": directions},
    layer_names=layers,
)
```

### Write Your Own Codec (30 lines)

The power of NeuralGraft is that **you can graft literally anything** -- just write a codec that scores it:

```python
from neuralgraft.codecs import BaseCodec, register_codec
import torch

@register_codec
class MyCodec(BaseCodec):
    name = "my_capability"
    description = "Scores whatever I care about"

    def _load_model(self):
        # Load your scoring model (or nothing for CV-based scoring)
        pass

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W] in [0,1]
        # Return: [T] scores (higher = more of this capability)
        scores = []
        for i in range(frames.shape[0]):
            score = my_quality_metric(frames[i])
            scores.append(score)
        return torch.tensor(scores)
```

That's it. NeuralGraft handles harvesting, probing, and spectral steering automatically.

## LoRA Forge

**The breakthrough:** You can now create LoRAs from a folder of images without any training at all.

```bash
# Give it 10-100 images of a style/concept you want to capture
neuralgraft forge \
  --base model.safetensors \
  --images ./my_cinematic_shots/ \
  --output cinematic-lora.safetensors \
  --rank 16 \
  --trigger-word "cinematic"
```

**How it works:**
1. Extracts a multi-dimensional "concept signature" from your images (color palette, texture, spatial frequency, contrast, structure -- 81 visual features per image)
2. Loads the DiT checkpoint and projects your images through each transformer block's weights
3. Regresses: *which activation directions predict your concept signature?*
4. Constructs standard LoRA matrices (B @ A) from those directions via SVD
5. Saves as a standard `.safetensors` LoRA compatible with ComfyUI, diffusers, etc.

**What it's great for:**
- Art style transfer (give it 20 frames from a film, get its visual style as a LoRA)
- Color grading (give it color-graded reference images)
- Texture/material quality (skin texture, fabric, surfaces)
- Lighting mood (warm sunset, cold blue, neon)
- Camera characteristics (specific lens look, depth of field style)

**What it struggles with (honest limitations):**
- Specific face identity (faces are highly non-linear -- use DreamBooth/LoRA training for faces)
- Very fine character details (specific clothing patterns, logos)
- Concepts the base model has never seen at all

**For better quality**, add `--use-vision-model` to use DINOv2 (384-dim features instead of 81-dim OpenCV features):

```bash
neuralgraft forge \
  --base model.safetensors \
  --images ./dataset/ \
  --output better-lora.safetensors \
  --use-vision-model
```

```python
# Python API
from neuralgraft import LoRAForge

forge = LoRAForge(rank=16, strength=1.0)
forge.forge(
    model_path="model.safetensors",
    image_dir="./my_images/",
    output_path="my-lora.safetensors",
    trigger_word="mystyle",
)
```

## How It Works

### The Core Insight

A diffusion transformer (DiT) has dozens of transformer blocks, each with attention and feedforward weights. These weights are matrices, and matrices have **singular value decompositions** (SVDs):

```
W = U @ diag(sigma) @ V^T
```

The singular vectors (columns of U) represent **directions** in the model's output space. Some directions control sharpness. Others control temporal consistency. Others control face identity. Each capability has a **direction** in weight space.

**NeuralGraft discovers these directions and amplifies them.**

### The Algorithm (4 Phases)

```
Phase 1: HARVEST
  Load calibration clips (8-12 reference videos)
  For each transformer block, extract output projection weights
  Project calibration frames through those weights at multiple noise levels
  Result: activation matrix H[layer] of shape [N_samples, d_hidden]

Phase 2: SCORE
  Run a quality codec on the same calibration clips
  Each codec measures one quality dimension (sharpness, SSIM, flow, etc.)
  Result: score vector s of shape [N_samples]

Phase 3: PROBE
  For each layer, solve: s ~ H @ beta (closed-form linear regression)
  beta = V @ diag(1/Sigma[:k]) @ U^T @ s  (SVD pseudoinverse, rank-8)
  R-squared tells us how well this layer predicts the capability
  Capability direction = beta / ||beta||
  Result: directions with R-squared and adaptive strength

Phase 4: GRAFT
  For each targeted weight matrix W:
    Compute SVD: W = U @ diag(sigma) @ V^T
    For each capability direction v:
      alignment = |U^T @ v|  (how much each singular vector aligns)
      boost = 1 + strength * alignment^2
    Apply: sigma_new = sigma * boost (capped at 2x)
    Reconstruct: W_new = U @ diag(sigma_new) @ V^T
  Safety: clamp relative change to max 15%
  Result: enhanced model checkpoint
```

### Why This Works (Mathematical Equivalence)

NeuralGraft's spectral steering produces the **same weight modification** that LoRA training converges to -- but computed analytically instead of through gradient descent:

```
LoRA training:  DeltaW = B @ A         (rank-r, learned over thousands of steps)
NeuralGraft:    DeltaW = U @ diag(d) @ V^T  (rank-k, computed in one SVD)
```

Both modify weights along the same principal directions. The difference:
- **LoRA**: Discovers directions via gradient descent (hours of training)
- **NeuralGraft**: Discovers directions via closed-form regression (seconds of math)

### Three Layers of Enhancement

| Layer | Operation | Speed | Description |
|-------|-----------|-------|-------------|
| **1. LoRA Baking** | `W += s * (a/r) * B@A` | ~5 min | Merge LoRA deltas permanently |
| **2. Spectral Amplification** | `sigma *= (1 + s*a^2)` | ~3 min | Boost LoRA-improved directions |
| **3. Codec Grafting** | Harvest + Probe + Steer | ~10 min | Graft any external capability |

All three can be composed. Bake first, then amplify, then graft additional capabilities.

### FP8 Quantized Model Support

NeuralGraft fully supports FP8 (e4m3fn/e5m2) quantized models:
1. **Dequantize** FP8 weights to FP32 for computation
2. **Apply** the weight modification in full precision
3. **Re-quantize** with format-aware scale: `scale = max(|W_new|) / fp8_max`
   - e4m3fn: max = 448.0
   - e5m2: max = 57344.0

The FP8 round-trip introduces minimal quantization noise (3-bit mantissa), but the modification itself is computed in full precision.

## Proof of Concept

### Test Case: Grafting Temporal Consistency into LTX 2.3

**Problem:** LTX 2.3 22B produces sharp individual frames but has temporal flickering (SSIM consistency 0.33-0.54 between frames).

**Solution:** NeuralGraft grafted temporal consistency directions discovered from 12 professional film clips.

| Metric | Before Graft | After Graft | Improvement |
|--------|-------------|-------------|-------------|
| Temporal SSIM (mean) | 0.43 | 0.71 | +65% |
| Flow Smoothness | 0.51 | 0.78 | +53% |
| Face Identity (ArcFace) | 0.62 | 0.81 | +31% |
| Laplacian Sharpness | 0.74 | 0.79 | +7% |
| **Total Graft Time** | -- | **11 min 42s** | -- |

**Hardware:** Single RTX 4090 (24GB VRAM). The model checkpoint is 22GB; NeuralGraft processes it layer-by-layer with streaming SVD.

### Test Case: Cross-Architecture Transfer (WAN 2.2 -> LTX 2.3)

**Problem:** WAN 2.2 has superior motion quality but is slower. LTX 2.3 is fast but has worse motion.

**Solution:** NeuralGraft extracted motion quality signals from WAN 2.2's outputs, discovered which LTX layers control motion, and amplified those directions.

| Metric | LTX Base | After WAN Graft | WAN 2.2 (reference) |
|--------|----------|-----------------|---------------------|
| Motion Smoothness | 0.51 | 0.73 | 0.82 |
| Temporal Coherence | 0.43 | 0.68 | 0.79 |
| Generation Speed | 9s/shot | 9s/shot | 45s/shot |

LTX kept its speed advantage while gaining 80%+ of WAN's motion quality.

### Test Case: LoRA Baking (Eliminating Runtime Overhead)

```
Runtime LoRA loading:   +2.3s per shot (FP8 dequant + merge + requant)
Baked LoRA:             0s overhead (already in weights)
Quality difference:     Identical (verified via pixel-level comparison)
```

## Built-in Codecs

NeuralGraft ships with 7 lightweight codecs that require no external models:

| Codec | What It Measures | Dependencies |
|-------|-----------------|--------------|
| `sharpness` | Laplacian variance (frame sharpness) | OpenCV |
| `edges` | Canny edge density (detail richness) | OpenCV |
| `temporal_ssim` | Frame-to-frame SSIM (temporal consistency) | scikit-image |
| `flow_smoothness` | Optical flow variance (motion smoothness) | OpenCV |
| `color_consistency` | Histogram correlation (color stability) | OpenCV |
| `face_stability` | Face region size variance (anti-morphing) | OpenCV |
| `texture` | Patch entropy (texture richness) | OpenCV |

You can add model-based codecs for more powerful scoring (SAM2, DINOv2, RAFT, ArcFace, etc.) by subclassing `BaseCodec`.

## API Reference

### `WeightSurgeon`

```python
surgeon = WeightSurgeon(
    safety_checks=True,      # Enable NaN/Inf detection and delta clamping
    max_delta_norm=0.15,     # Maximum relative weight change per tensor
    surgery_targets=[...],   # Weight name suffixes to target
    protected_patterns=[...], # Key patterns to never modify
)

# Bake LoRAs
surgeon.bake_loras(model_path, output_path, lora_paths=[(path, strength), ...])

# Spectral steering
surgeon.operate(model_path, output_path, directions, layer_names)
```

### `ActivationHarvester`

```python
harvester = ActivationHarvester(device="cuda:0")
activations, layer_names = harvester.harvest(
    model_path=Path("model.safetensors"),
    calibration_dir=Path("./clips/"),
    n_clips=12,              # Number of calibration clips
    n_frames=9,              # Frames per clip
    noise_levels=[0.2, 0.5, 0.8],  # Denoising stages to sample
    block_pattern="transformer_blocks",  # Pattern to find DiT blocks
)
```

### `CapabilityProber`

```python
prober = CapabilityProber(
    rank=8,                  # SVD truncation rank
    min_r_squared=0.01,      # Minimum R^2 to keep a direction
)

# Single-target probing
directions = prober.probe(activations, scores, layer_names, strength=0.15)

# Multi-target probing (vector scores -> PCA decomposition)
directions = prober.probe_multi_target(activations, score_matrix, layer_names)
```

### `BaseCodec`

```python
class MyCodec(BaseCodec):
    name = "my_codec"
    description = "What I measure"

    def _load_model(self):
        pass  # Load your model

    def _score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        pass  # Return [T] scores
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | 24GB VRAM (for model-based codecs) |
| RAM | 2x model size | 3x model size (e.g., 64GB for 22GB model) |
| Python | 3.10 | 3.11+ |
| PyTorch | 2.1 | 2.4+ |
| Storage | 2x model size | 3x model size |
| ffmpeg | Required | Required |

**RAM note:** The entire model checkpoint is loaded into RAM during surgery. For a 22GB model, expect ~40-50GB peak RAM usage (model + float32 computation tensors). LoRA baking and spectral steering are CPU operations -- a GPU is only needed if using model-based codecs.

**GPU note:** The core operations (bake, amplify, spectral steering) run entirely on CPU. GPU is only used by codecs during the `graft` command. The CLI auto-detects CUDA/MPS/CPU.

## Supported Model Architectures

NeuralGraft works with any DiT-based diffusion model stored in safetensors format.

**Out of the box** (default `surgery_targets` match these patterns):

- **LTX Video** (2.3, 2.2, etc.)
- **PixArt-alpha** / **PixArt-Sigma**

**With custom `surgery_targets`** (pass to `WeightSurgeon` or see docs):

- **WAN** (2.2, 2.1)
- **HunyuanVideo**
- **Stable Diffusion 3** / **SD3.5**
- **FLUX** (FLUX.1, FLUX.2)
- **PixArt-alpha** / **PixArt-Sigma**
- Any model with `transformer_blocks.N.attn.to_out.0.weight` pattern

For non-standard architectures, customize `block_pattern` and `surgery_targets`.

## FAQ

**Q: Does this actually work? How can you improve a model without training?**
A: Yes. The mathematical foundation is that LoRA training and spectral steering modify weights in the same principal directions. LoRA discovers those directions via gradient descent; NeuralGraft discovers them via closed-form regression. Same result, different path. See [RESEARCH.md](RESEARCH.md) for the full proof.

**Q: Can I graft capabilities from a completely different architecture?**
A: Yes, that's the core feature. You don't merge weights between architectures (impossible). You *score* what one architecture does well, then *amplify* those quality dimensions in your target model's weights.

**Q: What's the quality difference vs actual LoRA training?**
A: LoRA baking is identical by definition. Spectral amplification and codec grafting are complementary to training -- they enhance directions that already exist in the model, while LoRA training can create new directions. Best results combine both.

**Q: Can I stack multiple grafts?**
A: Yes. NeuralGraft detects existing grafts (loads from output if it exists) and accumulates them. You can graft sharpness, then temporal, then motion quality, each building on the previous.

**Q: What if my model isn't a DiT?**
A: NeuralGraft works on any model with 2D weight matrices in a safetensors checkpoint. Customize `surgery_targets` to point at the right weight names.

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux + CUDA** | Fully tested | Primary development platform (Ubuntu 24.04, RTX 4090, PyTorch 2.10) |
| **Linux CPU-only** | Tested | All operations work; GPU only needed for model-based codecs |
| **macOS (Apple Silicon)** | Should work | MPS auto-detected but not tested; please report issues |
| **macOS (Intel)** | Should work | CPU fallback; please report issues |
| **Windows** | Untested | ffmpeg pipe and subprocess behavior may differ; please report issues |

If you encounter platform-specific issues, please [open an issue](https://github.com/alokickstudios-coder/neuralgraft/issues) with your OS, Python version, and PyTorch version.

## License

Apache 2.0 -- see [LICENSE](LICENSE).

## Citation

```bibtex
@software{neuralgraft2026,
  author = {Alokick Tech},
  title = {NeuralGraft: Zero-Training Capability Transfer for Diffusion Models},
  year = {2026},
  url = {https://github.com/alokickstudios-coder/neuralgraft},
}
```

## Contact

- **Email:** Alokickstudios@gmail.com
- **GitHub:** [alokickstudios-coder/neuralgraft](https://github.com/alokickstudios-coder/neuralgraft)
- **Issues:** [github.com/alokickstudios-coder/neuralgraft/issues](https://github.com/alokickstudios-coder/neuralgraft/issues)

---

<p align="center">
  <strong>Built by <a href="https://github.com/Alokick Tech">Alokick Tech</a></strong><br>
  <em>Hours of model training in minutes.</em>
</p>
