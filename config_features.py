#!/usr/bin/env python3
"""
Configuration file for stitching features.
Adjust these settings to enable/disable advanced features.
"""

# ========================================
# OPTICAL FLOW SETTINGS
# ========================================

# Use multi-scale optical flow (slower, more robust)
USE_MULTISCALE_FLOW = True  # Set to True to enable

# Multi-scale pyramid scales
FLOW_SCALES = [1.0, 0.75, 0.5]  # Reduce for faster processing


# ========================================
# BLENDING SETTINGS
# ========================================

# Use advanced multi-band blending (memory intensive)
USE_MULTIBAND_BLENDING = True  # Set to True to enable

# Number of pyramid levels for multi-band blending
MULTIBAND_LEVELS = 3  # Reduce to 2 or 1 if memory issues

# Use seam carving for optimal blend boundary
USE_SEAM_CARVING = True  # Set to True to enable


# ========================================
# BOUNDARY WEIGHTING SETTINGS
# ========================================

# Apply smooth gradients to boundary weights
USE_SMOOTH_BOUNDARIES = True  # Set to True to enable

# Gaussian blur kernel size for smooth boundaries
BOUNDARY_BLUR_KERNEL = (21, 21)

# Sigmoid steepness for boundary falloff
BOUNDARY_SIGMOID_K = 10


# ========================================
# MEMORY OPTIMIZATION SETTINGS
# ========================================

# Automatically reduce pyramid levels for large images
AUTO_REDUCE_LEVELS = True

# Threshold for large image detection (pixels)
LARGE_IMAGE_THRESHOLD = 8000000  # 8 megapixels

# Force garbage collection after heavy operations
FORCE_GC = True


# ========================================
# DEBUG SETTINGS
# ========================================

# Save flow visualizations
SAVE_FLOW_VIS = True

# Save intermediate results
SAVE_INTERMEDIATE = True

# Verbose output
VERBOSE = True


# ========================================
# PRESETS
# ========================================

def apply_preset(preset_name):
    """Apply a predefined configuration preset."""
    global USE_MULTISCALE_FLOW, USE_MULTIBAND_BLENDING, USE_SEAM_CARVING
    global USE_SMOOTH_BOUNDARIES, MULTIBAND_LEVELS
    
    if preset_name == "fast":
        # Fast processing, basic quality
        USE_MULTISCALE_FLOW = False
        USE_MULTIBAND_BLENDING = False
        USE_SEAM_CARVING = False
        USE_SMOOTH_BOUNDARIES = False
        print("✓ Applied FAST preset")
        
    elif preset_name == "balanced":
        # Good balance of quality and speed
        USE_MULTISCALE_FLOW = False
        USE_MULTIBAND_BLENDING = False
        USE_SEAM_CARVING = False
        USE_SMOOTH_BOUNDARIES = True
        MULTIBAND_LEVELS = 3
        print("✓ Applied BALANCED preset")
        
    elif preset_name == "quality":
        # Best quality, slower
        USE_MULTISCALE_FLOW = True
        USE_MULTIBAND_BLENDING = True
        USE_SEAM_CARVING = True
        USE_SMOOTH_BOUNDARIES = True
        MULTIBAND_LEVELS = 3
        print("✓ Applied QUALITY preset")
        
    elif preset_name == "memory_efficient":
        # Reduced memory usage
        USE_MULTISCALE_FLOW = False
        USE_MULTIBAND_BLENDING = True
        USE_SEAM_CARVING = False
        USE_SMOOTH_BOUNDARIES = False
        MULTIBAND_LEVELS = 2
        print("✓ Applied MEMORY_EFFICIENT preset")
        
    else:
        print(f"⚠️  Unknown preset: {preset_name}")
        print("Available presets: fast, balanced, quality, memory_efficient")


# ========================================
# CURRENT CONFIGURATION
# ========================================

# Apply default preset (change this to switch presets)
CURRENT_PRESET = "fast"  # Options: fast, balanced, quality, memory_efficient

# Uncomment to apply preset on import
# apply_preset(CURRENT_PRESET)


if __name__ == "__main__":
    print("="*60)
    print("STITCHING CONFIGURATION")
    print("="*60)
    print(f"\nCurrent Preset: {CURRENT_PRESET}")
    print(f"\nOptical Flow:")
    print(f"  Multi-scale: {USE_MULTISCALE_FLOW}")
    print(f"  Scales: {FLOW_SCALES}")
    print(f"\nBlending:")
    print(f"  Multi-band: {USE_MULTIBAND_BLENDING}")
    print(f"  Levels: {MULTIBAND_LEVELS}")
    print(f"  Seam carving: {USE_SEAM_CARVING}")
    print(f"\nBoundary:")
    print(f"  Smooth: {USE_SMOOTH_BOUNDARIES}")
    print(f"\nMemory:")
    print(f"  Auto reduce: {AUTO_REDUCE_LEVELS}")
    print(f"\nDebug:")
    print(f"  Save visualizations: {SAVE_FLOW_VIS}")
    print(f"  Save intermediate: {SAVE_INTERMEDIATE}")
    print("="*60)
