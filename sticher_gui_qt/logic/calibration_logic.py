# Copiar lógica de calibración
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stich_old'))

try:
    from mycalibrate import *
except ImportError:
    print("Warning: Could not import from mycalibrate")
