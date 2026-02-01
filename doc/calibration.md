# Omnidirectional Camera Calibration Script

This script (`mycalibrate.py`) is designed to calibrate fisheye or omnidirectional cameras using a set of checkerboard images. It uses OpenCV's `cv2.omnidir` module to calculate intrinsic parameters.

## Usage

```bash
python mycalibrate.py --image_dir <path_to_images> [options]
```

### Arguments

- `--image_dir`: (Required) Path to the directory containing `.jpg` calibration images.
- `--pattern_rows`: Number of inner corners per row in the checkerboard pattern (default: 7).
- `--pattern_cols`: Number of inner corners per column in the checkerboard pattern (default: 9).
- `--show`: If established, it displays the images with the corners detected during processing (it can slow down execution).

### Example

```bash
python mycalibrate.py --image_dir ./calib/images --show
```

## Functionality

1.  **Image Loading**: Iterates through all `.jpg` files in the specified directory.
2.  **Corner Detection**: Uses `cv2.findChessboardCorners` to locate the checkerboard pattern.
3.  **Subpixel Refinement**: Refines corner positions using `cv2.cornerSubPix` for higher accuracy.
4.  **Circle Detection**: Attempts to detect the circular boundary of the fisheye lens using Hough Transforms (`findcircle_on_image`).
5.  **Calibration**: Uses `cv2.omnidir.calibrate` to compute:
    - `K`: Camera matrix.
    - `D`: Distortion coefficients.
    - `xi`: Projection parameter for omnidirectional model.

## Output

The script prints the calibrated parameters to the console:
- `K`: Intrinsic matrix.
- `D`: Distortion coefficients.
- `xi`: Xi parameter.
- `RMS`: Root Mean Square error (calibration quality).

## Requirements

- Python 3.x
- OpenCV (`opencv-contrib-python` for `cv2.omnidir`)
- NumPy
