# Copiar toda la lógica de stitcher desde stich_old
import sys
import os

# Add stich_old to path to reuse code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from runstich import PanoramaStitcher
except ImportError:
    # Fallback: define a stub
    print("Warning: Could not import PanoramaStitcher from stich_old")
    PanoramaStitcher = None
