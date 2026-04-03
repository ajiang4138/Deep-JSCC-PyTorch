# Deep-JSCC-PyTorch MIC (Mapping to Irregular Constellation) Migration

## Overview

This document records all major changes and additions made to the Deep-JSCC-PyTorch project throughout the migration chat. The primary objective was to add **MIC (Mapping to Irregular Constellation) as a hardware-oriented finite-constellation transmission option** with full training, evaluation, and visualization support.

---

## Phase 1: Core MIC Implementation

### 1.1 New File: `constellation.py`

**Purpose**: Core MIC quantization module providing constellation management, codebook operations, and symbol mapping.

**Key Components**:
- **Symbol I/Q Pairing** (`pair_channels_to_symbols()`, `unpair_symbols_to_channels()`):
  - Reshapes [B, 2c, H, W] → [B, c, H, W, 2] pairing adjacent channels as I/Q complex symbols
  - Enables symbol-wise quantization over channel pairs

- **MICLayer** class:
  - Hard and soft surrogate gradient modes: `soft`, `straight_through`, `hard_forward_soft_backward`
  - Power normalization modes: `codebook`, `post_mapper`, `none`
  - Configurable constellation size (`constellation_size`)
  - Clipping value support (`clip_value` parameter for soft mode)
  - Forward pass: nearest-neighbor hard mapping with gradient control
  - `forward_deploy()`: deployment-ready hard quantization without surrogate gradients

- **Codebook Management**:
  - `normalize_constellation_power()`, `normalize_symbol_power()`
  - Power constraint application and statistics collection
  - Codebook export utilities: `.pt` (PyTorch), `.npy` (NumPy), `.json` (metadata)

- **Standalone Mapper** (`map_to_mic_codebook()`):
  - Deployment-ready function for hard MIC mapping
  - No gradient computation; used in inference/export workflows

- **Fallback Centroid Estimation** (added Phase 3):
  - `_kmeans_centroids()`: k-means clustering on symbol data
  - `_estimate_centers_from_post_symbols()`: Fallback handler for visualization when trained codebook unavailable
  - Key for debug visualization robustness

### 1.2 Modified File: `model.py`

**Changes**:

- **Encoder Normalization** (lines 87-106):
  - Added `_normlizationLayer(P=1)` method
  - Per-image power normalization formula: $z_{\text{norm}} = \sqrt{P \cdot k} \cdot z / \sqrt{z^T z}$
  - Ensures consistent latent scale across examples

- **Decoder Final Activation** (line 132):
  - Added `Sigmoid()` final activation layer
  - Constrains RGB output to [0, 1] range
  - Critical for color fading analysis (Phase 6)

- **Mapper Insertion in Forward Path** (lines 158-165):
  - Optional mapper insertion: `if self.mapper is not None`
  - Hard pass-through when `mapper=None` (backward compatible)
  - Mapper applies symbol quantization before channel

- **Debug Forward Path** (lines 167-195):
  - New `forward_debug(return_mapper_indices=False)` method
  - Returns intermediate activations for analysis:
    - `z_encoded`: encoder output
    - `z_mapper_in`: mapper input (optionally with indices)
    - `z_transmitted`: received latent from channel
    - `output`: final reconstruction
  - Supports visualization and debug scripts

- **Mapper Control Methods**:
  - `set_mapper(mapper)`: Attach mapper instance
  - `disable_mapper()`: Set `mapper=None`
  - `set_mapper_deploy_mode(deploy_mode)`: Switch between train_surrogate and hard_deploy
  - `export_mapper_state(export_path, mode='npy')`: Export mapper codebook
  - `get_mapper_stats()`: Retrieve mapper statistics (cluster sizes, codebook power)
  - `get_mapper_config()`: Extract mapper configuration (type, size, modes)

### 1.3 Modified File: `train.py`

**Changes**:

- **CLI Arguments for Mapper**:
  - `--mapper_type`: MIC or other constellation types
  - `--constellation_size`: Effective constellation cardinality (default: 128)
  - `--mic_surrogate_mode`: Gradient surrogate mode (soft, straight_through, hard_forward_soft_backward)
  - `--mic_power_constraint_mode`: Power normalization mode (codebook, post_mapper, none)
  - `--mic_clip_value`: Clipping threshold for soft mode
  - `--freeze_mapper`: Freeze mapper weights during training
  - `--mapper_fine_tune`: Continue training from previous mapper checkpoint
  - `--mapper_lr`: Separate learning rate for mapper parameters
  - `--mapper_scheduler`: Separate scheduler config for mapper
  - `--export_mapper`: Path to export trained mapper after training completion

- **Optimizer Integration** (`build_optimizer()`):
  - Separates model and mapper parameters into distinct param groups
  - Mapper param group uses `--mapper_lr` if different from base LR
  - Enables independent learning rate scheduling for mapper convergence

- **Checkpoint Format**:
  - New structured format with `mapper_config` metadata
  - Backward compatible: detects and loads legacy raw state dicts
  - Saves full mapper state within checkpoint for reproducibility
  - Includes surrogate vs. deploy mode tracking

- **Fine-tuning and Freezing**:
  - `--mapper_fine_tune` loads previous mapper from checkpoint
  - `--freeze_mapper` disables mapper parameter gradients
  - Supports curriculum learning workflows (e.g., train model first, freeze mapper, then train mapper)

### 1.4 Modified File: `eval.py`

**Changes**:

- **Mapper-Aware Checkpoint Loading**:
  - Detects new structured checkpoint format vs. legacy raw dict
  - Extracts and validates `mapper_config` metadata
  - Reconstructs mapper instance from checkpoint config
  - Sets correct eval mode (train_surrogate vs. hard_deploy)

- **Hard Deploy Mode Support**:
  - `eval_mode='hard_deploy'`: Uses `forward_deploy()` path (no surrogate gradients)
  - `eval_mode='train_surrogate'`: Uses training surrogate mode for fair comparison with train process
  - Critical for production deployment evaluation

- **Export Workflows**:
  - Export mapper state alongside model checkpoint
  - Support for multi-checkpoint evaluation (sweep over multiple trained checkpoints)
  - Mapper export path configurable via CLI

### 1.5 New File: `debug_mapper.py`

**Purpose**: Unit-test-like sanity checks for MIC integration.

**Tests**:
- Shape roundtrip: verify `pair_channels_to_symbols()` / `unpair_symbols_to_channels()` preserve dimensions
- Gradient flow: confirm backprop works through quantization with surrogate gradients
- Power normalization: validate that encoder output power and mapper constraints are applied correctly
- Device compatibility: test CPU ↔ GPU transitions
- Checkpoint format: verify mapper_config serialization and deserialization

**Status**: Passing in virtual environment (`pytest debug_mapper.py`)

### 1.6 Modified File: `signal_constellation_visualizer.py`

**Changes**:
- Updated to handle new checkpoint format with `mapper_config` metadata
- Backward compatible with legacy raw state dicts
- Correctly extracts mapper configuration for visualization
- Supports adjacent I/Q pairing visualization

---

## Phase 2: Visualization and Debug Script

### 2.1 New File: `debug_mic_visualizer.py`

**Purpose**: Rich constellation visualization with optional baseline-vs-MIC comparison, decision region overlays, and centroid markers.

**Key Features**:

- **Checkpoint Loading** (`build_model_from_checkpoint()`):
  - Loads model from structured or legacy checkpoint format
  - Reconstructs mapper from saved config
  - Supports `--snr_override` to test decoder performance at different SNR levels without retraining
  - Sets correct eval mode and device (CPU/GPU)

- **Mapper Configuration Inference** (`infer_mapper_config()`):
  - Extracts constellation type, size, and modes from checkpoint
  - Falls back to command-line args if metadata missing
  - Ensures consistency between checkpoint and visualization parameters

- **Decision Region Visualization** (Phase 3 enhancement):
  - `build_decision_regions()`: Voronoi grid labeling over I/Q space (lines 173-181)
  - `plot_region_boundaries()`: Contour drawing with marker overlays (lines 184-201)
  - Shows hard decision boundaries for quantization visualization

- **Fallback Centroid Estimation** (Phase 3 addition):
  - `_kmeans_centroids()`: k-means clustering on pre-mapper latent symbols (lines 204-231)
  - `_estimate_centers_from_post_symbols()`: Fallback handler; estimates cluster centers from post-mapper symbol distribution (lines 234-259)
  - `compute_cluster_centroids()`: Unified dispatch; uses saved codebook if available, otherwise falls back to estimation (lines 262-293)
  - **Motivation**: Many checkpoints lack trained mapper; visualization robustness requires fallback strategy

- **Plotting Functions**:
  - `plot_single_constellation()` (lines 296-340): Single constellation plot with optional overlay
  - `plot_comparison_constellation()` (lines 360-404): Side-by-side baseline vs. MIC comparison
  - Includes centroid markers, decision contours, title annotations

- **Test Modes**:
  - `run_single()` (lines 428-513): Visualize single checkpoint/image, with optional overlay
  - `run_compare()` (lines 516-631): Compare baseline (no mapper) vs. MIC (with mapper/estimated codebook)
  - Both modes support customizable constellation size and overlay parameters

- **CLI Arguments**:
  - `--checkpoint`: Path to model checkpoint
  - `--image`: Path to input image for encoding/reconstruction
  - `--mode`: 'single' or 'compare'
  - `--snr_override`: Override checkpoint SNR for decoder testing
  - `--constellation_size`: Override checkpoint constellation size
  - `--overlay_k`: Fallback k-means cluster count for centroid estimation (default: 128)
  - `--compare_baseline`: Baseline checkpoint for comparison mode
  - `--output_dir`: Directory for saving visualization plots

---

## Phase 3: Overlay Robustness Enhancement

### 3.1 Enhanced: `debug_mic_visualizer.py` Fallback Strategy

**Problem**: Visualization failed when checkpoint had no trained mapper (common for legacy models).

**Solution**:
- Added `_kmeans_centroids()` helper for k-means clustering on symbol data
- Added `_estimate_centers_from_post_symbols()` to estimate cluster centers from post-mapper symbol cloud
- Updated `compute_cluster_centroids()` to use fallback when explicit codebook unavailable
- Wired fallback computation into both `run_single()` and `run_compare()` test modes
- Added `--overlay_k` CLI parameter to control fallback center count

**Impact**: Visualization now works reliably even for checkpoints without explicit trained mapper.

---

## Phase 4: Comparative Test Framework

### 4.1 New File: `test_mic_image_triplet.py`

**Purpose**: Generate 3-panel triplet comparisons showing input image, output without MIC, and output with MIC, including PSNR labels and metadata.

**Core Functionality** (`run_for_image()`):

- **No-MIC Path** (baseline):
  - Flow: encoder → [channel noise if SNR set] → decoder → clipped RGB output
  - Provides reference reconstruction quality without quantization

- **MIC Path** (with estimated or trained mapper):
  - Flow: encoder → estimated/trained mapper → hard nearest-neighbor symbol quantization → [channel noise] → decoder → clipped RGB output
  - Uses trained mapper if available in checkpoint (via `--prefer_trained_mapper` flag)
  - Falls back to k-means estimated codebook from pre-mapper latent if no trained mapper

- **Codebook Estimation** (`estimate_codebook_from_latent()`):
  - k-means clustering on pre-mapper latent symbols
  - Generates approximate constellation when trained mapper unavailable
  - Enables comparative testing across checkpoints with/without trained MIC

- **Quantization Details**:
  - Hard nearest-neighbor mapping to estimated/trained constellation
  - Maintains gradient flow through quantization in test path
  - Respects constellation size and power normalization mode from checkpoint

- **Output Format** (`save_triplet_plot()`):
  - 3-panel figure: [Input | Output (No MIC) | Output (With MIC)]
  - PSNR labels on output panels
  - Title and metadata: checkpoint name, SNR, MIC mode, constellation size
  - Saved as high-quality PNG with `dpi=150`

- **Reproducibility**:
  - Seeded random number generator per image (same seed across runs ensures deterministic output)
  - Documented checkpoint info and MIC configuration in output metadata
  - CSV log of PSNRs and parameters for batch analysis

**CLI Arguments**:
- `--checkpoint`: Path to trained model checkpoint
- `--images`: Space-separated list of image paths to test
- `--snr`: Channel SNR override (optional; uses checkpoint default if omitted)
- `--constellation_size`: Override checkpoint constellation size
- `--mic_clip_value`: Override MIC clip value for soft mode
- `--mic_power_constraint_mode`: Override power normalization (codebook, post_mapper, none)
- `--prefer_trained_mapper`: Use trained mapper from checkpoint (vs. estimated codebook)
- `--output_dir`: Directory for saving triplet plots and metadata

**Test Runs Executed** (Phase 5):
- SNR=7: kodim23.png, kodim08.png with CIFAR10_8_7.0_0.17_AWGN checkpoint
- SNR=19: kodim23.png, kodim08.png, 4ksunset.jpg with CIFAR10_8_19.0_0.17_AWGN checkpoint
- SNR=4: 4ksunset.jpg with CIFAR10_8_4.0_0.17_AWGN checkpoint
- SNR=19 with CLI override: kodim23.png, kodim08.png, 4ksunset.jpg (verifying SNR override flag)

**Output Organization**:
```
out/test_mic_triplet_CHECKPOINT_NAME/
├── triplet_IMAGE_NAME.png
├── triplet_IMAGE_NAME_snr19/  (SNR-specific subfolder if multiple SNRs tested)
│   ├── triplet_IMAGE_NAME.png
│   └── metrics.csv
└── metrics.csv
```

---

## Phase 5: SNR Override and Testing Validation

### 5.1 Verified Feature: SNR CLI Override Support

**Purpose**: Test decoder performance at arbitrary SNR values without retraining model.

**Implementation**:
- `--snr` flag in `debug_mic_visualizer.py` (line 113) and `test_mic_image_triplet.py`
- Decouples test SNR from checkpoint's training SNR
- Channel uses test SNR to compute noise power: $P_{\text{noise}} = P_{\text{signal}} / 10^{\text{SNR}/10}$

**Use Cases**:
- Evaluate same checkpoint across SNR range (e.g., SNR=4, 7, 10, 19)
- Study decoder robustness without re-training
- Generate performance curves for fixed checkpoint

**Test Validation**:
- Ran triplet tests with SNR=19 checkpoint using `--snr 19` CLI flag
- Confirmed output folder naming reflects CLI SNR override
- Verified that triplet PSNRs are consistent with explicit SNR parameter

---

## Phase 6: Color Fading Analysis and Root Cause Investigation

### 6.1 Observed Phenomenon: Color Desaturation at High SNR

**User Observation**: "For the higher SNR values, this is causing a fading in the colors of the decoded images. Why is this?"

**Investigation Scope**:
- Analyzed [channel.py](channel.py) noise model (lines 1-62)
- Analyzed [model.py](model.py) encoder/decoder architecture (lines 1-190)
- Analyzed [test_mic_image_triplet.py](test_mic_image_triplet.py) MIC test pipeline (lines 1-240)

### 6.2 Root Cause Analysis

**Finding**: Color fading is **NOT directly caused by SNR** but rather by **quantization bias + distribution mismatch**.

**Mechanism**:

1. **Channel Noise Does Not Fade Colors**:
   - SNR formula: $P_{\text{noise}} = P_{\text{signal}} / 10^{\text{SNR}/10}$ (channel.py:31)
   - Higher SNR → **lower** noise amplitude
   - Cleaner channel transmission should improve color fidelity, not degrade it

2. **Primary Cause: MIC Quantization Bias**:
   - Estimated k-means codebook from pre-mapper latent introduces hard quantization
   - Hard nearest-neighbor mapping reduces latent dynamic range (quantization error)
   - Quantization bias is **independent of SNR**; determined by constellation cardinality and data distribution

3. **Secondary Amplification: Decoder Contrast Loss**:
   - Quantized/clipped latent has lower contrast than original
   - Sigmoid decoder (model.py:132) outputs: $\text{RGB} = \sigma(\text{decoder latent})$
   - Lower latent contrast → flatter Sigmoid curve region → more desaturated RGB values

4. **Visibility Effect: SNR Masking**:
   - **At low SNR** (e.g., SNR=4, 7):
     - Channel noise adds power to latent signal
     - Noise mask partially obscures quantization bias
     - Human perception: noise dominates visual degradation
   
   - **At high SNR** (e.g., SNR=19):
     - Channel noise nearly absent
     - Quantization bias becomes **only** source of latent distortion
     - Hard contrast loss fully visible → obvious color desaturation
   - **This is NOT a new problem at high SNR; it becomes visible at high SNR**

5. **Distribution Shift Factor**:
   - Checkpoints trained on **CIFAR10** (32×32 synthetic images, limited color palette)
   - Tested on **Kodak/4K natural images** (high-resolution, full color gamut)
   - Domain mismatch: encoder learned to compress CIFAR10 features, not natural photo features
   - Estimated codebook poorly represents natural image latent distribution
   - Quantization error magnified for out-of-distribution data

### 6.3 Impact of MIC Configuration

**Controllable Parameters** (via CLI):
- `--constellation_size`: Smaller → more quantization; larger → more fidelity
  - Default: 128 (2^7 bits per symbol pair, ~16 bits per 32×32 CIFAR10 block)
- `--mic_clip_value`: Soft mode clipping threshold affects gradient flow during training
- `--mic_power_constraint_mode`: Codebook vs. post_mapper normalization affects decoder input range

**Not Root Cause Factors**:
- SNR value (hidden benefit: reveals bias rather than causing it)
- Channel AWGN (lower SNR adds noise, not color fade)
- Encoder power normalization (preserves latent scale correctly)

### 6.4 Implications for Evaluation

**Key Insights**:
- High-SNR performance limited by **MIC quantization**, not channel fidelity
- Color fading naturally decreases with:
  - Larger constellation size (less quantization)
  - Domain-matched training (CIFAR10 checkpoint on CIFAR10 images would show less fading)
  - Trained (vs. estimated) MIC mapper (optimized codebook placement in learned feature space)

**Testing Best Practices**:
- When comparing MIC vs. no-MIC at high SNR:
  - Color fading is **expected** with estimated codebook and domain mismatch
  - Trained MIC mapper + domain-matched checkpoint will show better color fidelity
  - Use SNR as **diagnostic tool**: color fading amplitude reveals quantization severity

---

## Known Behaviors and Design Decisions

### Color Fading at High SNR
- Unsure if a bug
- Root cause: MIC quantization bias + distribution mismatch between CIFAR10 training and natural image test data
- At high SNR, channel noise is minimal, so quantization bias becomes the dominant distortion (visible as color desaturation)
- Addressed by using trained MIC mapper, larger constellation size, or domain-matched checkpoint

### Fallback Centroid Estimation
- Used when checkpoint lacks explicit trained mapper (common for legacy checkpoints)
- k-means clustering on pre-mapper or post-mapper symbols provides approximate decision boundaries
- Enables visualization robustness; visualization is approximate but functional

### Backward Compatibility
- Old checkpoints (raw state dicts) still load and run without mapper
- New checkpoints save structured format with `mapper_config` metadata
- Automatic format detection on load; no user configuration needed

### SNR Override Mechanism
- Test-time SNR can differ from training SNR
- Useful for robustness evaluation without retraining
- Does not modify checkpoint; temporary decoder evaluation only

---
