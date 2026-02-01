# Technical Documentation - 360° Panorama Stitching System

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Processing Pipeline](#processing-pipeline)
4. [Rotation Techniques](#rotation-techniques)
5. [Optical Flow](#optical-flow)
6. [Blending Techniques](#blending-techniques)
7. [Memory Optimization](#memory-optimization)
8. [Feature Configuration](#feature-configuration)
9. [References and Theory](#references-and-theory)

---

## Overview

This system performs 360° panorama stitching from 6 fisheye cameras arranged in a circular configuration. The process combines multiple advanced computer vision techniques:

- **Fisheye to equirectangular projection**: Converting fisheye images to spherical format
- **Global rotation computation**: Using SLERP or Bundle Adjustment
- **Optical flow refinement**: RAFT for fine alignment
- **Multi-band blending**: Laplacian pyramids for smooth transitions
- **Seam carving**: Dynamic programming for optimal seams

---

## System Architecture

### Main Components

```
runstich.py
├── PanoramaStitcher (main class)
│   ├── Stage 1: Rotation Computation
│   │   ├── Fisheye → Equirectangular
│   │   ├── Feature Matching (XFeat)
│   │   └── Global Rotation (SLERP/Bundle)
│   │
│   └── Stage 2: Stitching and Blending
│       ├── Spherical Rotation
│       ├── Optical Flow (RAFT)
│       ├── Flow Warping
│       └── Blending (Simple/Multi-band/Seam)
│
config_features.py (Feature toggles)
│   ├── USE_MULTISCALE_FLOW
│   ├── USE_MULTIBAND_BLENDING
│   ├── USE_SEAM_CARVING
│   └── USE_SMOOTH_BOUNDARIES
│
GUI (PySide6)
    ├── Stitcher View
    ├── Calibration View
    └── Settings View
```

### Models and Dependencies

- **XFeat**: GPU-accelerated feature matching
- **RAFT**: State-of-the-art optical flow (torch)
- **OpenCV**: Image operations and remapping
- **SciPy**: SLERP for rotation interpolation

---

## Processing Pipeline

### Stage 1: Rotation Computation

#### 1.1 Fisheye → Equirectangular Conversion

**Objective**: Transform fisheye images to uniform spherical projection.

**Process**:
```python
# Equirectangular coordinates (θ, φ)
theta = (x - center_x) / center_x * (π/2)  # Azimuth
phi = (y - center_y) / center_y * (π/2)    # Elevation

# Sphere → Cartesian
X = sin(θ) * cos(φ)
Y = sin(φ)
Z = cos(θ) * cos(φ)

# Cartesian → Fisheye polar
θ_fisheye = atan2(Y, X)
φ_fisheye = acos(Z)
ρ = φ_fisheye / (π/2)  # Normalized radius

# Apply distortion (polynomial model)
ρ' = ρ + D₀*ρ² + D₁*ρ⁴

# Fisheye pixel coordinates
x_fisheye = ρ' * r * cos(θ_fisheye) + cx
y_fisheye = ρ' * r * sin(θ_fisheye) + cy
```

**Optimization**: 
- Remapping cache by (height, width, camera)
- Using `cv2.remap` with linear interpolation
- Pre-computed maps as float32

#### 1.2 Feature Matching with XFeat

**XFeat** is a deep learning-based accelerated feature extractor.

**Advantages**:
- ~4096 keypoints per image
- Robust matching with learned descriptors
- GPU-optimized speed

**Usage**:
```python
mkpts_0, mkpts_1 = xfeat.match_xfeat(img1, img2, top_k=4096)
```

#### 1.3 Pairwise Rotation Computation

Converts pixel matches to unit sphere points and solves for optimal rotation.

**Pixel → Sphere Transformation**:
```python
# Normalize pixels to [-π/2, π/2]
theta = (x - center_x) / center_x * (π/2)
phi = (y - center_y) / center_y * (π/2)

# Spherical projection
v = [sin(θ)*cos(φ), sin(φ), cos(θ)*cos(φ)]
```

**RANSAC Optimization**:
- 100 iterations with random samples
- Minimum 10 matches required
- Least squares with bounds [-π, π]

**Euler Rotation Matrices**:
```python
Rx = [[1,      0,         0     ],
      [0, cos(θ), -sin(θ)],
      [0, sin(θ),  cos(θ)]]

Ry = [[cos(φ),  0, sin(φ)],
      [0,       1,      0 ],
      [-sin(φ), 0, cos(φ)]]

Rz = [[cos(γ), -sin(γ), 0],
      [sin(γ),  cos(γ), 0],
      [0,            0, 1]]

R = Rz @ Ry @ Rx
```

---

## Rotation Techniques

### Method 1: SLERP with Loop Closure Correction (Default)

**Problem**: Sequential rotations accumulate error, causing the chain 1→2→3→4→5→6 to mismatch with 6→1.

**Solution**:

1. **Compute sequential rotations** (1→2, 2→3, ..., 5→6)
2. **Measure direct closure rotation** (6→1)
3. **Compute closure error**: R_error = R_closure @ R_chain^T
4. **Distribute error with SLERP**:

```python
# Create SLERP interpolator
slerp = Slerp([0.0, 1.0], [R_identity, R_error])

# Apply fractional correction
for i in range(n_cameras - 1):
    alpha = (i + 1) / n_cameras
    R_correction = slerp(alpha)
    R_corrected[i] = R_correction @ R_chain[i]
```

**Advantages**:
- Guarantees uniform 360° coverage
- Equitable error distribution
- ~60° spacing per camera (6 cameras)

**Expected result**: θ ≈ [0°, 60°, 120°, 180°, 240°, 300°]

### Method 2: Bundle Adjustment

**Objective**: Global optimization of all rotations simultaneously.

**Formulation**:

Minimize:
```
E = Σ_pairs Σ_points ||R_predicted @ v_src - v_dst||²
```

Where:
- R_predicted = R_j @ R_i^T (predicted relative rotation)
- v_src, v_dst: points on unit sphere

**Representation**: Axis-angle (rotvec) for singularity-free optimization

**Constraints**:
- Camera 1 fixed (identity)
- 5 cameras × 3 parameters = 15 variables
- Sequential pairs (weight 1.0) + skip-one (weight 0.5)

**Optimizer**: Trust Region Reflective (TRF)
- max_nfev: 500 iterations
- ftol, xtol, gtol: 1e-6, 1e-8, 1e-8

**Limitation**: Does not guarantee uniform 360° distribution, may converge to local minima.

---

## Optical Flow

### RAFT (Recurrent All-Pairs Field Transforms)

**Purpose**: Fine alignment refinement after geometric rotations.

**Architecture**:
- CNN encoder for features
- All-pairs correlation between frames
- Recurrent GRU for iterative refinement
- Output: dense displacement field (H×W×2)

### Resolution Configuration

**Base** (cfg.USE_MULTISCALE_FLOW = False):
```python
work_h, work_w = 400, 720  # Reduced for memory
```

**Multi-scale** (cfg.USE_MULTISCALE_FLOW = True):
```python
scales = [1.0, 0.75, 0.5]  # cfg.FLOW_SCALES
# Resolutions: 520×960, 390×720, 260×480
# Weighted combination: weights = scales / sum(scales)
```

**Multi-scale Advantages**:
- Captures large motions (low scale)
- Fine details (high scale)
- More robust to large displacements

### Edge-Weighted Flow

**Problem**: Flow should move seams/edges but not distort central content.

**Solution**:

1. **Distance Transform** from valid edges:
```python
valid_mask = any(target > 0, axis=2)
dist = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
```

2. **Inverse weight** (1 at edges, 0 at center):
```python
flow_weight = 1.0 - (dist / dist.max())
flow *= flow_weight[:,:,None]
```

3. **Validity mask**:
```python
valid_flow = (target_gray > 10) & (stitched_gray > 10)
flow[~valid_flow] = 0
```

**Result**: Flow aligns seams without deforming internal content.

---

## Blending Techniques

### 1. Simple Blending (Default)

**Method**: Alpha blending with distance mask.

```python
# Distance mask from edges
mask_blend = cv2.distanceTransform(valid_mask, cv2.DIST_L2)
mask_blend = mask_blend / 255.0

# Linear blending
result = mask_blend * target + (1 - mask_blend) * stitched
```

**Advantages**: Fast, low memory consumption
**Disadvantages**: Visible seams in lighting changes

### 2. Smooth Boundaries (cfg.USE_SMOOTH_BOUNDARIES)

**Extension** of simple blending with smoothing.

```python
# Gaussian blur for smooth transition
mask = cv2.GaussianBlur(mask, (21, 21), 0)

# Sigmoid falloff (non-linear transition)
k = 10  # Slope
mask = 1.0 / (1.0 + exp(-k * (mask - 0.5)))
```

**Effect**: More gradual transitions, reduces "banding" in seams.

### 3. Multi-band Blending (cfg.USE_MULTIBAND_BLENDING)

**Theory**: Blending in separate frequencies hides seams better than direct alpha blending.

**Algorithm**:

1. **Build Laplacian pyramids** of both images:
```python
# Gaussian pyramid
G[0] = img
G[i+1] = pyrDown(G[i])

# Laplacian pyramid
L[i] = G[i] - pyrUp(G[i+1])
L[n] = G[n]  # Lowest level
```

2. **Gaussian pyramid of mask**:
```python
M[i] = pyrDown(M[i-1])
```

3. **Per-level blending**:
```python
LS[i] = (1 - M[i]) * L1[i] + M[i] * L2[i]
```

4. **Reconstruction**:
```python
result = LS[n]
for i in n-1 → 0:
    result = pyrUp(result) + LS[i]
```

**Advantages**:
- High frequencies (details) mixed in narrow band
- Low frequencies (color) blended smoothly
- Imperceptible seams

**Configuration**:
- `MULTIBAND_LEVELS = 3` (default)
- Auto-reduce for images >8MP
- Memory: ~3× image size

### 4. Seam Carving (cfg.USE_SEAM_CARVING)

**Objective**: Find optimal seam that minimizes color differences.

**Algorithm - Dynamic Programming**:

1. **Cost map**:
```python
diff = ||img1 - img2||₂  # L2 norm per pixel
diff_smooth = GaussianBlur(diff, (15,15))
```

2. **DP Forward Pass**:
```python
dp[0, :] = cost[0, :]
for i in 1→H:
    for j in 0→W:
        dp[i,j] = cost[i,j] + min(dp[i-1, j-1:j+2])
```

3. **Backtracking**:
```python
seam[H-1] = argmin(dp[H-1, :])
for i in H-2→0:
    seam[i] = path[i+1, seam[i+1]]
```

4. **Binary mask**:
```python
mask[i, :seam[i]] = 0  # Use img1
mask[i, seam[i]:] = 1  # Use img2
```

**Complexity**: O(H×W×3) ≈ linear in pixels

**Advantages**: 
- Seams in low-contrast areas
- Avoids crossing prominent objects

---

## Memory Optimization

### Original Problem

With 4000×4000 images per camera:
- 6 images: ~288 MB
- Equirectangular: ~192 MB
- Panorama 8000×4000: ~96 MB
- Optical flow: ~150 MB
- **Total**: >700 MB → Killed (Exit 137)

### Implemented Strategies

#### 1. On-Demand Loading (Stage 2)
```python
# DO NOT load all images at startup
for i in 2→6:
    img = cv2.imread(path)  # Load only when needed
    process(img)
    del img
    gc.collect()
```

#### 2. Flow Resolution Reduction
```python
# Before: 520×960
# Now: 400×720  (-44% pixels)
work_h, work_w = 400, 720
```

#### 3. rotate_image_spherical Optimization
```python
# Before: np.stack([x, y, z], axis=-1)  # Memory copy
# Now: component-wise multiplication
xyz_x = x_d * R[0,0] + y_d * R[1,0] + z_d * R[2,0]
xyz_y = x_d * R[0,1] + y_d * R[1,1] + z_d * R[2,1]
xyz_z = x_d * R[0,2] + y_d * R[1,2] + z_d * R[2,2]

# Free intermediates immediately
del x_d, y_d, z_d
gc.collect()
```

**Reduction**: ~73% memory in spherical rotation

#### 4. Universal Float32
```python
# float64 (8 bytes) → float32 (4 bytes)
arrays = arrays.astype(np.float32)
```

#### 5. Aggressive Garbage Collection
```python
if cfg.FORCE_GC:
    gc.collect()
```

#### 6. GPU Memory Management
```python
if device == 'cuda':
    torch.cuda.empty_cache()
```

#### 7. Disable Visualizations
```python
# Comment out all debug cv2.imwrite
# Saves ~150 MB in flow_vis, flow_magnitude, etc.
```

**Total Result**: ~800 MB → ~250 MB (~69% reduction)

---

## Feature Configuration

### File: config_features.py

```python
# Optical Flow
USE_MULTISCALE_FLOW = False
FLOW_SCALES = [1.0, 0.75, 0.5]

# Blending
USE_MULTIBAND_BLENDING = False
MULTIBAND_LEVELS = 3
USE_SEAM_CARVING = False

# Boundaries
USE_SMOOTH_BOUNDARIES = False
BOUNDARY_BLUR_KERNEL = (21, 21)
BOUNDARY_SIGMOID_K = 10

# Memory
AUTO_REDUCE_LEVELS = True
LARGE_IMAGE_THRESHOLD = 8000000
FORCE_GC = True
```

### Available Presets

#### 1. Fast (Default)
```python
CURRENT_PRESET = "fast"
# Everything disabled
# Time: ~30s
# Memory: ~250 MB
```

#### 2. Balanced
```python
CURRENT_PRESET = "balanced"
USE_SMOOTH_BOUNDARIES = True
# Time: ~35s
# Memory: ~270 MB
```

#### 3. Quality
```python
CURRENT_PRESET = "quality"
USE_MULTISCALE_FLOW = True
USE_MULTIBAND_BLENDING = True
USE_SEAM_CARVING = True
USE_SMOOTH_BOUNDARIES = True
# Time: ~90s
# Memory: ~600 MB
```

#### 4. Memory Efficient
```python
CURRENT_PRESET = "memory_efficient"
USE_MULTIBAND_BLENDING = True
MULTIBAND_LEVELS = 2  # Reduced
# Time: ~45s
# Memory: ~300 MB
```

### Change Preset

```python
# In config_features.py
apply_preset("quality")
```

Or edit variables directly.

---

## References and Theory

### Key Papers

1. **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**
   - Teed & Deng, ECCV 2020
   - https://arxiv.org/abs/2003.12039

2. **XFeat: Accelerated Features for Lightweight Image Matching**
   - Guilherme Potje et al., CVPR 2024
   - https://github.com/verlab/accelerated_features

3. **Multi-band Blending**
   - Burt & Adelson, "A Multiresolution Spline With Application to Image Mosaics", ACM TOG 1983

4. **Graph Cuts for Seam Finding**
   - Agarwala et al., "Interactive Digital Photomontage", SIGGRAPH 2004

### Mathematical Models

#### Omnidirectional Projection

UCM Model (Unified Camera Model):
```
x_cam = (X/Z_s, Y/Z_s)
Z_s = ξ*d + Z
d = sqrt(X² + Y² + Z²)
```

Where ξ is the omnidirectional projection parameter.

#### SLERP (Spherical Linear Interpolation)

For rotations R₀, R₁ with parameter t ∈ [0,1]:

```
Slerp(R₀, R₁, t) = R₀ * (R₀⁻¹ * R₁)^t
```

In axis-angle:
```
ω = log(R₀⁻¹ * R₁)  # Axis-angle
R(t) = R₀ * exp(t * ω)
```

#### Optical Flow

Brightness constancy equation:
```
I(x, y, t) = I(x+u, y+v, t+1)
```

Linearization (Horn-Schunck):
```
Iₓ*u + Iᵧ*v + Iₜ = 0
```

RAFT uses correlation volume instead:
```
C(x₁, x₂) = ⟨f₁(x₁), f₂(x₂)⟩ / √(||f₁|| * ||f₂||)
```

---

## Troubleshooting

### Error: Killed (Exit 137)
**Cause**: Out of memory
**Solution**: 
- Apply "memory_efficient" preset
- Reduce MULTIBAND_LEVELS to 2
- Disable MULTISCALE_FLOW

### Error: IndexError in process_stage_2
**Cause**: Mismatch between loop range and array size
**Solution**: Verify that `rot_idx = i - 2` (fixed)

### Panorama with Visible Seams
**Cause**: Simple blending in high-contrast areas
**Solution**: Enable USE_MULTIBAND_BLENDING or USE_SEAM_CARVING

### Rotations Don't Cover 360°
**Cause**: Bundle Adjustment converged to local minimum
**Solution**: Use rotation_method='slerp' (default)

### Distorted Features
**Cause**: Advanced features causing excessive warping
**Solution**: Keep "fast" preset as baseline

---

## File Structure

```
stich_old/
├── runstich.py              # Main pipeline
├── config_features.py       # Feature configuration
├── mycalibrate.py          # Omnidirectional calibration
├── kandao.json             # Camera parameters
├── insta360.json
├── doc/
│   ├── TECHNICAL_DOCUMENTATION.md  # This file
│   ├── ROTATION_METHODS.md
│   ├── BUNDLE_ADJUSTMENT_IMPROVEMENTS.md
│   └── calibration.md
├── dataset/
│   └── origin_{id}_1.jpg   # Input images
└── calib/
    └── filtered/           # Calibration images

sticher_gui_qt/
├── views/
│   ├── stitcher_view.py    # Stitching view
│   ├── calibration_view.py
│   └── settings_view.py
└── widgets/
    └── panorama_viewer_360.py  # Interactive viewer
```

---

## Usage Commands

### CLI

```bash
# Basic (fast preset)
python stich_old/runstich.py \
  --config stich_old/kandao.json \
  --input_template "stich_old/dataset/origin_{id}_1.jpg" \
  --output panorama.jpg

# With visualization
python stich_old/runstich.py \
  --config stich_old/kandao.json \
  --input_template "stich_old/dataset/origin_{id}_1.jpg" \
  --output panorama.jpg \
  --show

# Bundle Adjustment method (add parameter to code)
stitcher = PanoramaStitcher(config, rotation_method='bundle')
```

### GUI

```bash
python sticher_gui_qt/main.py
```

Interface with:
- Stitcher Tab: Load images, execute pipeline, visualize
- Calibration Tab: Camera calibration
- Settings Tab: Per-camera parameters
- 360° Viewer: Mouse drag to rotate, scroll to zoom

---

## Performance Benchmarks

Hardware: NVIDIA GPU, 16GB RAM

| Preset | Time | Memory | Seam Quality |
|--------|------|--------|--------------|
| Fast | 30s | 250 MB | Acceptable |
| Balanced | 35s | 270 MB | Good |
| Quality | 90s | 600 MB | Excellent |
| Memory Efficient | 45s | 300 MB | Very Good |

Resolution: 6× images 4000×4000 → Panorama 8000×4000

---

## Contributions and Future Improvements

### Possible Optimizations

1. **Tiling**: Process panorama in tiles to reduce memory
2. **Sparse Flow**: Calculate flow only in overlap regions
3. **GPU Blending**: Implement pyramids in CUDA
4. **Dynamic Scaling**: Adjust resolution based on available memory

### Experimental Features

1. **HDR Stitching**: Multi-exposure fusion
2. **Video Stitching**: Pipeline for temporal sequences
3. **Real-time**: Optimization for live stitching
4. **Deep Learning Blending**: CNN to predict optimal masks

---

## License and Credits

**Developed by**: [Your Name/Organization]

**Main Libraries**:
- PyTorch (BSD License)
- OpenCV (Apache 2.0)
- PySide6 (LGPL)
- SciPy (BSD)

**Pre-trained Models**:
- RAFT: BSD License
- XFeat: MIT License

---

**Last update**: February 2026
**Version**: 2.0

1. **HDR Stitching**: Fusión de exposiciones múltiples
2. **Video Stitching**: Pipeline para secuencias temporales
3. **Real-time**: Optimización para stitching en vivo
4. **Deep Learning Blending**: CNN para predecir máscaras óptimas

---

## Licencia y Créditos

**Desarrollado por**: [Tu Nombre/Organización]

**Librerías Principales**:
- PyTorch (BSD License)
- OpenCV (Apache 2.0)
- PySide6 (LGPL)
- SciPy (BSD)

**Modelos Pre-entrenados**:
- RAFT: BSD License
- XFeat: MIT License

---

**Última actualización**: Febrero 2026
**Versión**: 2.0
