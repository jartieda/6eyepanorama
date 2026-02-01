import cv2
import numpy as np
import os
import glob
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate fisheye/omnidirectional camera using chessboard images.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the calibration images')
    parser.add_argument('--pattern_rows', type=int, default=7, help='Number of inner corners per row')
    parser.add_argument('--pattern_cols', type=int, default=9, help='Number of inner corners per column')
    parser.add_argument('--show', action='store_true', help='Show processing images')
    return parser.parse_args()

def findcircle_on_image(img, gray, show=False):
    #threshold image 
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    
    if show:
        cv2.namedWindow('binary', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('binary', binary)
        cv2.waitKey(1)

    circles = cv2.HoughCircles(
        binary, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=10, 
        param1=50, 
        param2=100, 
        minRadius=1700,
        maxRadius=2200
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Optional: draw circles if showing
        # for (x, y, r) in circles:
        #     cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        return circles
    return None

def main():
    args = parse_args()
    
    CHECKERBOARD = (args.pattern_rows, args.pattern_cols)
    # print cv version
    print(f"OpenCV Version: {cv2.__version__}")

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:2*CHECKERBOARD[0]:2, 0:2*CHECKERBOARD[1]:2].T.reshape(-1, 2)
    
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Search for images in the provided directory
    search_pattern = os.path.join(args.image_dir, '*.jpg') 
    images = glob.glob(search_pattern)
    
    if not images:
        print(f"No images found in {args.image_dir}")
        return

    # Optional: Filter directory if needed, or just run. 
    # For now, we removed the hardcoded filter logic but kept the structure
    
    print(f"objpoints shape template: {objp.shape}")
    
    circles = []

    for fname in images:
        print(f"Processing {fname}")
        
        # Original filter: if not "(2)" in fname: continue
        # You can re-enable this or make it an argument if strictly needed.
        # For generality, I'm verifying it's a file.
        if not os.path.isfile(fname):
            continue

        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue

        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            if _img_shape != img.shape[:2]:
                print(f"Skipping {fname} due to size mismatch.")
                continue

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        c = findcircle_on_image(img, gray, show=args.show)
        if c is not None:
            circles.extend(c)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("chessboard found")
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            
            if args.show:
                # Draw and display the corners
                cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(50)
        else:
            print("chessboard NOT found")

    # get avg x, y , radius from circles
    if circles:
        avg_x = np.mean([circle[0] for circle in circles])
        avg_y = np.mean([circle[1] for circle in circles])
        avg_radius = np.mean([circle[2] for circle in circles])
        print(f"Average Circle - X: {avg_x}, Y: {avg_y}, Radius: {avg_radius}")
    else:
        print("No circles found.")


    N_OK = len(objpoints)
    print("N_OK=" + str(N_OK))
    
    if N_OK == 0:
        print("Not enough valid images for calibration.")
        return

    print("start calibr")
    print(f"objpoints: {len(objpoints)}")
    
    dims = _img_shape[::-1] 
    
    # Run calibration
    # Note: If no initial guess is provided, CALIB_USE_GUESS should not be used or K/D must be provided.
    # The original code used CALIB_USE_GUESS with None, which might default inside OpenCV or be incorrect usage.
    # We will try to run without GUESS first or remove the None params if we use the flag.
    # Ideally for omnidir:
    flags = cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
    # If you want to use guess, you need to provide K and D. 
    # Let's trust opencv to find it without guess for now, or match original "careless" usage if it worked by luck.
    # Original: flags=cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
    # but passed K=None. Let's start clean.
    
    try:
        rms, k, xi, d, rvecs, tvecs, idx  =  cv2.omnidir.calibrate(
                objectPoints=objpoints, 
                imagePoints=imgpoints, 
                size=dims, 
                K=None, xi=None, D=None,
                flags=flags,
                criteria=subpix_criteria)

        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(k.tolist()) + ")")
        print("D=np.array(" + str(d.tolist()) + ")")
        print(f"RMS: {rms}")
        print(f"xi: {xi}")

    except Exception as e:
        print(f"Calibration failed: {e}")

if __name__ == "__main__":
    main()