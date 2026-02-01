
Here is the documentation for the runstich.py script:
## runstich.py Documentation

### Overview
The `runstich.py` script processes images using the `mosaic_dualstage_1` and `mosaic_dualstage_2` functions. It reads parameters from JSON files and processes images in batches.

### Dependencies
- `cv2`
- `json`
- `mosaic_dualstage_1`
- `mosaic_dualstage_2`

### Script Structure

#### Reading Parameters from JSON Files
The script reads parameters from two JSON files: `kandao.json` and `insta360.json`.

#### Processing Images with `mosaic_dualstage_1`
The script processes a single image using the `mosaic_dualstage_1` function.

#### Batch Processing with `mosaic_dualstage_2`
The script processes multiple images in a loop using the `mosaic_dualstage_2` function.

#### Cleanup
The script includes cleanup commands to close OpenCV windows and save the final image.

### File Paths
- **Input images** are located in:
    - `C:/soft/manual_oracle/stich/dataset/tuneladora/`
    - `C:/soft/manual_oracle/stich/dataset/prueba_time_lapse/`
- **Output images** are saved in:
    - `C:/soft/manual_oracle/stich/out2/`
    - `C:/soft/manual_oracle/stich/out/`

### JSON Files
- `kandao.json`: Contains parameters for the first set of images.
- `insta360.json`: Contains parameters for the second set of images.

### Functions
- `mosaic_dualstage_1(input_image_path, output_image_path)`: Processes a single image.
- `mosaic_dualstage_2(input_image_path, acc_rots, output_image_path)`: Processes multiple images in a loop.

### Usage
Run the script to process images as described above. Ensure that the JSON files and image directories are correctly set up.

For more details, refer to the script itself: `runstich.py`.

## Calibration Tool

For information on how to use the camera calibration script, see [Calibration Documentation](doc/calibration.md).
