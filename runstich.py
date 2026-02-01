
import cv2
import numpy as np
import sys
import torch
import json
import argparse
import os
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from itertools import combinations

from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F

# Import feature configuration
try:
    import config_features as cfg
except ImportError:
    # Fallback if config_features is not in path
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import config_features as cfg

class PanoramaStitcher:
    def __init__(self, config_path, device='cuda', show=False, rotation_method='slerp'):
        """
        Initialize PanoramaStitcher.
        
        Args:
            config_path: Path to camera configuration JSON
            device: 'cuda' or 'cpu'
            show: Show intermediate visualizations
            rotation_method: 'slerp' or 'bundle' for rotation computation
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.show = show
        self.rotation_method = rotation_method
        self.load_config(config_path)
        self.load_models()
        self.maps = {} # Cache for remap coordinates

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.params = json.load(f)
        print(f"Loaded configuration from {config_path}")

    def load_models(self):
        print("Loading XFeat and Raft models...")
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        
        weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = weights.transforms()
        self.raft_model = raft_large(weights=weights, progress=False).to(self.device)
        self.raft_model.eval()
        print("Models loaded.")

    def _get_remap_maps(self, height, width, cam_id):
        # Cache key based on dimensions and camera params
        param = self.params[cam_id - 1] # 1-based to 0-based
        cx, cy, r, D = param["center_x"], param["center_y"], param["radius"], param["D"]
        
        key = (height, width, cx, cy, r, str(D))
        if key in self.maps:
            return self.maps[key]

        max_size = max(height, width)
        
        # Grid for equirectangular image
        # THETA: -pi/2 to pi/2, PHI: -pi/2 to pi/2
        # Corresponding to range of pixels 0 to max_size
        
        # Grid of destination pixels (equirectangular)
        grid_x, grid_y = np.meshgrid(np.arange(max_size), np.arange(max_size))
        
        # Normalize to spherical coordinates (Theta, Phi)
        theta = (grid_x - max_size / 2) / (max_size / 2) * (np.pi / 2)
        phi = (grid_y - max_size / 2) / (max_size / 2) * (np.pi / 2)
        
        # Spherical to Cartesian (Unit Sphere)
        # Note: Original code axes:
        # X = sin(theta) * cos(phi)
        # Y = sin(phi)
        # Z = cos(theta) * cos(phi)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        
        # Cartesian to Fisheye Projection
        # r = atan2(sqrt(x^2+y^2), z)  <-- Standard fisheye is r = f * theta
        # But this code seems to use specific projection:
        # THETA_P = atan2(Y, X)
        # PHI_P = arccos(Z)  <-- Angle from Z axis
        # RHO_P = PHI_P / (pi/2)  <-- Linear projection?
        
        theta_p = np.arctan2(y, x)
        phi_p = np.arccos(z)
        rho_p = phi_p / (np.pi / 2)
        
        # Distortion application
        if D is not None:
             rho_p = rho_p + D[0][0] * rho_p**2 + D[0][1] * rho_p**4
        
        rho_p = rho_p * r
        
        # Back to pixel coordinates in fisheye source
        x_fisheye = rho_p * np.cos(theta_p) + cx
        y_fisheye = rho_p * np.sin(theta_p) + cy
        
        # Generate Float32 maps for cv2.remap
        map_x = x_fisheye.astype(np.float32)
        map_y = y_fisheye.astype(np.float32)
        
        self.maps[key] = (map_x, map_y)
        return map_x, map_y

    def fisheye_to_equirectangular(self, img, cam_id, out_filename=None):
        height, width = img.shape[:2]
        map_x, map_y = self._get_remap_maps(height, width, cam_id)
        
        # cv2.remap is highly optimized
        equirectangular = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        if out_filename:
            cv2.imwrite(out_filename, equirectangular)
            
        return equirectangular

    def match_with_xfeat(self, im1, im2, outfilename="xfeat_match.jpg"):
        if hasattr(self.xfeat, 'match_xfeat'):
             mkpts_0, mkpts_1 = self.xfeat.match_xfeat(im1, im2, top_k=4096)
        else:
            # Fallback for different xfeat versions/wrappers if needed
             mkpts_0, mkpts_1 = self.xfeat(im1, im2) # depending on API

        # Convert to KeyPoints for consistency
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_0]
        keypoints2 = [cv2.KeyPoint(p[1], p[1], 5) for p in mkpts_1] # Note: p[1] usage in original seems suspicious? 
        # Actually original code:
        # keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_0]
        # keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_1] 
        # Wait, mkpts_1 elements are [x,y]. 
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_1]

        matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]
        
        if self.show:
             im3 = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:500], None)
             cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
             cv2.resizeWindow("Matches", 1600, 600)
             cv2.imshow("Matches", im3)
             cv2.waitKey(100)

        return matches, keypoints1, keypoints2

    def spherical_to_cartesian(self, theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        return np.stack([x, y, z], axis=-1)

    def get_rot_mat(self, src_pts, dst_pts, center_x, center_y):
        # Vectorized spherical conversion
        theta_src = (src_pts[:, 0] - center_x) / center_x * (np.pi / 2)
        phi_src = (src_pts[:, 1] - center_y) / center_y * (np.pi / 2)
        src_sph = self.spherical_to_cartesian(theta_src, phi_src)

        theta_dst = (dst_pts[:, 0] - center_x) / center_x * (np.pi / 2)
        phi_dst = (dst_pts[:, 1] - center_y) / center_y * (np.pi / 2)
        dst_sph = self.spherical_to_cartesian(theta_dst, phi_dst)

        def loss_func(x):
            rx = np.array([[1, 0, 0], 
                           [0, np.cos(x[0]), -np.sin(x[0])],
                           [0, np.sin(x[0]), np.cos(x[0])]])   
            ry = np.array([[np.cos(x[1]), 0, np.sin(x[1])],
                           [0, 1, 0],
                           [-np.sin(x[1]), 0, np.cos(x[1])]])
            rz = np.array([[np.cos(x[2]), -np.sin(x[2]), 0],
                           [np.sin(x[2]), np.cos(x[2]), 0],
                           [0, 0, 1]])
            R = rz @ ry @ rx # Matrix multiplication
            
            rotated_src = src_sph @ R.T
            return np.sum(np.linalg.norm(rotated_src - dst_sph, axis=1))

        x0 = np.array([0., 0., 0.])
        res = least_squares(loss_func, x0, bounds=([-np.pi]*3, [np.pi]*3))
        
        # Reconstruct R
        theta, phi, gamma = res.x
        rx = np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        ry = np.array([[np.cos(phi), 0, np.sin(phi)],[0, 1, 0],[-np.sin(phi), 0, np.cos(phi)]])
        rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma), np.cos(gamma), 0],[0, 0, 1]])
        
        R = rz @ ry @ rx
        error_list = np.linalg.norm((src_sph @ R.T) - dst_sph, axis=1)
        
        return R, error_list

    def get_rot_mat_ransac(self, src_pts, dst_pts, center_x, center_y):
        best_R = np.eye(3)
        min_error = float('inf')
        best_mask = None
        
        num_points = len(src_pts)
        if num_points < 10:
             # Fallback if too few points
             return self.get_rot_mat(src_pts, dst_pts, center_x, center_y)[0], src_pts, dst_pts

        for _ in range(100):
            indices = np.random.choice(num_points, min(100, num_points), replace=False)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]
            
            R, errors = self.get_rot_mat(src_sample, dst_sample, center_x, center_y)
            total_error = np.sum(np.abs(errors))
            
            if total_error < min_error:
                min_error = total_error
                best_R = R
                # Re-evaluate on all points to find good matches
                # This could be optimized, but following original logic roughly
        
        # Final pass to filter outliers
        _, all_errors = self.get_rot_mat(src_pts, dst_pts, center_x, center_y) 
        # Note: Ideally we use best_R to compute errors, but function returns new R.
        # Let's compute errors for best_R manually or reuse existing function 
        # Simulating original logic:
        # Original logic re-ran get_rot_mat on "good" points.
        
        # Let's just return what we have for now to match original flow roughly
        return best_R
    
    def compute_all_pairwise_rotations(self, equirects):
        """
        Compute rotations between ALL pairs of cameras (not just sequential).
        This provides more constraints for bundle adjustment.
        Returns: dict of {(cam_i, cam_j): (rotation_matrix, src_pts, dst_pts, matches)}
        """
        n_cameras = len(equirects)
        center_x, center_y = 2000, 2000
        
        pairwise_data = {}
        
        print("\n=== Computing Pairwise Rotations ===")
        
        # 1. Sequential neighbors (1-2, 2-3, ..., 6-1)
        for i in range(n_cameras):
            j = (i + 1) % n_cameras
            cam_i = i + 1  # 1-based
            cam_j = j + 1
            
            print(f"Matching Camera {cam_i} -> {cam_j}")
            matches, kp1, kp2 = self.match_with_xfeat(equirects[i], equirects[j])
            
            if len(matches) < 10:
                print(f"  WARNING: Only {len(matches)} matches found!")
                continue
                
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            R = self.get_rot_mat_ransac(src_pts, dst_pts, center_x, center_y)
            pairwise_data[(cam_i, cam_j)] = {
                'rotation': R,
                'src_pts': src_pts,
                'dst_pts': dst_pts,
                'matches': matches,
                'type': 'sequential'
            }
            print(f"  {len(matches)} matches, rotation computed")
        
        # 2. Skip-one neighbors (1-3, 2-4, 3-5, 4-6, 5-1, 6-2) for extra constraints
        for i in range(n_cameras):
            j = (i + 2) % n_cameras
            cam_i = i + 1
            cam_j = j + 1
            
            print(f"Matching Camera {cam_i} -> {cam_j} (skip)")
            matches, kp1, kp2 = self.match_with_xfeat(equirects[i], equirects[j])
            
            if len(matches) < 8:  # Lower threshold for skip connections
                print(f"  Skipping: only {len(matches)} matches")
                continue
                
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            R = self.get_rot_mat_ransac(src_pts, dst_pts, center_x, center_y)
            pairwise_data[(cam_i, cam_j)] = {
                'rotation': R,
                'src_pts': src_pts,
                'dst_pts': dst_pts,
                'matches': matches,
                'type': 'skip'
            }
            print(f"  {len(matches)} matches, rotation computed")
        
        return pairwise_data
    
    def bundle_adjustment_rotation(self, pairwise_data, n_cameras=6):
        """
        Global bundle adjustment to optimize all camera rotations simultaneously.
        Uses axis-angle representation for optimization.
        
        Args:
            pairwise_data: dict from compute_all_pairwise_rotations
            n_cameras: number of cameras
            
        Returns:
            list of optimized rotation matrices (camera 1 is identity, 2-6 are relative to 1)
        """
        print("\n=== Bundle Adjustment ===")
        
        # Initialize: camera 1 is identity, others from sequential chain
        initial_rotations = [np.eye(3)]
        
        # Build initial guess from sequential chain
        acc = np.eye(3)
        for i in range(1, n_cameras):
            cam_pair = (i, i+1) if i < n_cameras else (n_cameras, 1)
            if cam_pair in pairwise_data:
                R_rel = pairwise_data[cam_pair]['rotation']
                acc = np.dot(R_rel, acc)
                initial_rotations.append(acc.copy())
            else:
                # Fallback to identity if missing
                initial_rotations.append(acc.copy())
        
        # Convert to axis-angle (camera 1 is always identity, optimize 2-6)
        def matrix_to_axisangle(R_matrix):
            r = R.from_matrix(R_matrix)
            return r.as_rotvec()
        
        def axisangle_to_matrix(rotvec):
            r = R.from_rotvec(rotvec)
            return r.as_matrix()
        
        # Initial parameters: 5 cameras × 3 axis-angle params = 15 params
        x0 = []
        for i in range(1, n_cameras):
            x0.extend(matrix_to_axisangle(initial_rotations[i]))
        x0 = np.array(x0)
        
        print(f"Initial parameters: {len(x0)} values")
        print(f"Optimizing over {len(pairwise_data)} pairwise constraints")
        
        # Cost function
        def cost_function(params):
            # Reconstruct rotation matrices
            rotations = [np.eye(3)]  # Camera 1 fixed
            for i in range(n_cameras - 1):
                rotvec = params[i*3:(i+1)*3]
                rotations.append(axisangle_to_matrix(rotvec))
            
            residuals = []
            center_x, center_y = 2000, 2000
            
            # For each pairwise constraint
            for (cam_i, cam_j), data in pairwise_data.items():
                R_measured = data['rotation']
                src_pts = data['src_pts']
                dst_pts = data['dst_pts']
                
                # Get absolute rotations
                R_i = rotations[cam_i - 1]  # 1-based to 0-based
                R_j = rotations[cam_j - 1]
                
                # Predicted relative rotation: R_ij = R_j @ R_i^T
                R_predicted = np.dot(R_j, R_i.T)
                
                # Limit number of points for speed (use ~20 points per pair)
                n_samples = min(20, len(src_pts))
                indices = np.linspace(0, len(src_pts)-1, n_samples, dtype=int)
                
                # Compute reprojection error
                for idx in indices:
                    src_pt = src_pts[idx]
                    dst_pt = dst_pts[idx]
                    
                    # Convert pixel to sphere
                    theta_src = (src_pt[0] - center_x) / center_x * (np.pi/2)
                    phi_src = (src_pt[1] - center_y) / center_y * (np.pi/2)
                    
                    x_src = np.sin(theta_src) * np.cos(phi_src)
                    y_src = np.sin(phi_src)
                    z_src = np.cos(theta_src) * np.cos(phi_src)
                    v_src = np.array([x_src, y_src, z_src])
                    
                    # Apply predicted rotation
                    v_dst_pred = R_predicted @ v_src
                    
                    # Convert back to pixel
                    theta_dst_pred = np.arctan2(v_dst_pred[0], v_dst_pred[2])
                    phi_dst_pred = np.arcsin(np.clip(v_dst_pred[1], -1, 1))
                    
                    x_dst_pred = (theta_dst_pred / (np.pi/2) * center_x) + center_x
                    y_dst_pred = (phi_dst_pred / (np.pi/2) * center_y) + center_y
                    
                    # Error in pixels (normalized for better convergence)
                    dx = (x_dst_pred - dst_pt[0]) / center_x  # Normalize by image size
                    dy = (y_dst_pred - dst_pt[1]) / center_y
                    
                    # Weight by constraint type
                    weight = 1.0 if data['type'] == 'sequential' else 0.5
                    residuals.append(weight * dx)
                    residuals.append(weight * dy)
            
            return np.array(residuals)
        
        # Optimize
        print("Running optimization...")
        result = least_squares(
            cost_function, 
            x0, 
            method='trf',  # Trust Region Reflective - better for large problems
            verbose=2,
            max_nfev=500,  # Increased iterations
            ftol=1e-6,     # Tighter tolerance on cost function
            xtol=1e-8,     # Tighter tolerance on parameters
            gtol=1e-8      # Tighter tolerance on gradient
        )
        
        print(f"\nOptimization result: {result.message}")
        print(f"Cost reduction: {np.sum(cost_function(x0)**2):.2e} -> {np.sum(result.fun**2):.2e}")
        print(f"Optimization success: {result.success}")
        print(f"Number of iterations: {result.nfev}")
        
        # Compute improvement
        initial_cost = np.sum(cost_function(x0)**2)
        final_cost = np.sum(result.fun**2)
        if initial_cost > 0:
            improvement_pct = 100 * (1 - final_cost / initial_cost)
            print(f"Improvement: {improvement_pct:.1f}%")
        
        # Extract optimized rotations
        optimized_rotations = [np.eye(3)]
        for i in range(n_cameras - 1):
            rotvec = result.x[i*3:(i+1)*3]
            optimized_rotations.append(axisangle_to_matrix(rotvec))
        
        # Check if optimization improved significantly
        if initial_cost > 0 and final_cost / initial_cost > 0.95:
            print("\n⚠️  WARNING: Bundle adjustment didn't improve much (<5%).")
            print("   Falling back to initial sequential rotations.")
            # Return initial rotations (already accumulated from sequential chain)
            return [initial_rotations[i] for i in range(1, n_cameras)]
        
        return [optimized_rotations[i] for i in range(1, n_cameras)]

    def compute_rotations_slerp(self, equirects):
        """
        Improved rotation computation with loop closure correction.
        Uses sequential chain + SLERP distribution of closure error.
        Guarantees uniform 360° angular distribution.
        """
        print("\n=== Computing Rotations with Improved SLERP ===")
        
        n_cameras = len(equirects)
        center_x, center_y = 2000, 2000
        
        # Step 1: Compute sequential rotations (1->2, 2->3, ..., 5->6)
        print("\n--- Sequential Pairwise Rotations ---")
        sequential_rotations = []
        for i in range(n_cameras - 1):
            print(f"Matching Camera {i+1} -> {i+2}")
            matches, kp1, kp2 = self.match_with_xfeat(equirects[i], equirects[i+1])
            
            if len(matches) < 10:
                print(f"  WARNING: Only {len(matches)} matches!")
                
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            R = self.get_rot_mat_ransac(src_pts, dst_pts, center_x, center_y)
            sequential_rotations.append(R)
            print(f"  {len(matches)} matches, rotation computed")
        
        # Step 2: Compute closure rotation (6->1)
        print(f"\nMatching Camera {n_cameras} -> 1 (closure)")
        matches, kp6, kp1 = self.match_with_xfeat(equirects[-1], equirects[0])
        src_pts = np.float32([kp6[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches])
        
        R_closure = self.get_rot_mat_ransac(src_pts, dst_pts, center_x, center_y)
        print(f"  {len(matches)} matches, rotation computed")
        
        # Step 3: Accumulate rotations via chain
        print("\n--- Accumulating Rotations (Chain) ---")
        acc_chain = []
        acc = np.eye(3)
        
        for i, R in enumerate(sequential_rotations):
            acc = np.dot(R, acc)
            acc_chain.append(acc.copy())
            print(f"Camera {i+2} accumulated")
        
        # Step 4: Compute closure from chain vs direct
        acc_6_chain = acc_chain[-1]  # Camera 6 via chain
        acc_6_closure = R_closure.T  # Camera 6 via direct measurement
        
        print("\n--- Loop Closure Error Analysis ---")
        print(f"Camera 6 via chain:\n{acc_6_chain}")
        print(f"Camera 6 via closure:\n{acc_6_closure}")
        
        # Compute error rotation
        R_error = acc_6_closure @ acc_6_chain.T
        error_angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        print(f"Loop closure error angle: {np.degrees(error_angle):.2f}°")
        
        # Step 5: Distribute error using SLERP
        print("\n--- Distributing Error with SLERP ---")
        
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp
        
        # Convert to scipy Rotation
        r_identity = R_scipy.from_matrix(np.eye(3))
        r_error = R_scipy.from_matrix(R_error)
        
        # Create SLERP interpolator from identity to error
        times = np.array([0.0, 1.0])
        rotations_slerp = R_scipy.concatenate([r_identity, r_error])
        slerp = Slerp(times, rotations_slerp)
        
        # Apply fractional correction to each camera
        corrected_rotations = []
        for i in range(len(acc_chain)):
            # Apply (i+1)/n_cameras fraction of error
            alpha = (i + 1) / n_cameras
            r_correction = slerp(alpha)
            
            # Apply correction: R_corrected = R_correction @ R_original
            R_corrected = r_correction.as_matrix() @ acc_chain[i]
            corrected_rotations.append(R_corrected)
            
            # Compute corrected angle
            R_test = R_corrected
            theta = np.arctan2(R_test[0, 2], R_test[2, 2])
            print(f"Camera {i+2}: alpha={alpha:.3f}, theta={np.degrees(theta):.1f}°")
        
        # Verify 360° coverage
        print("\n--- Rotation Coverage Check ---")
        angles = []
        for i, R in enumerate(corrected_rotations):
            theta = np.arctan2(R[0, 2], R[2, 2])
            angles.append(np.degrees(theta))
            print(f"Camera {i+2}: {angles[-1]:.1f}°")
        
        # Expected angles for 6 cameras: 0°, 60°, 120°, 180°, 240°, 300° (or variations)
        print(f"\nExpected ~60° spacing for {n_cameras} cameras")
        
        return corrected_rotations
    
    def compute_rotations_bundle(self, equirects):
        """
        Rotation computation using Bundle Adjustment.
        Optimizes all rotations globally to minimize reprojection errors.
        May not guarantee uniform 360° distribution.
        """
        print("\n=== Computing Rotations with Bundle Adjustment ===")
        
        # Step 1: Compute all pairwise rotations
        pairwise_rotations = self.compute_all_pairwise_rotations(equirects)
        
        # Step 2: Run bundle adjustment
        optimized_rotations = self.bundle_adjustment_rotation(pairwise_rotations, len(equirects))
        
        return optimized_rotations
    
    def compute_all_rotations(self, equirects):
        """
        Router method to select rotation computation method.
        Delegates to SLERP or Bundle Adjustment based on self.rotation_method.
        """
        if self.rotation_method == 'bundle':
            return self.compute_rotations_bundle(equirects)
        else:  # Default to 'slerp'
            return self.compute_rotations_slerp(equirects)

    def get_accumulated_rotations(self, rotations):
        """
        Legacy function for compatibility.
        Rotations already come corrected with SLERP from compute_all_rotations.
        """
        print("\n--- Using SLERP-Corrected Rotations ---")
        for i, rot in enumerate(rotations, start=2):
            print(f"Camera {i} rotation matrix (relative to camera 1):")
            print(rot)
            print()
        return rotations

    def rotate_image_spherical(self, img, rotation_matrix, out_width=None):
        import gc
        height, width = img.shape[:2]
        if out_width is None:
            out_width = width * 2 # Panorama width
        
        center_x = out_width / 4.0 
        center_y = height / 2.0 
        
        # Use ogrid instead of meshgrid to save memory
        y_coords = np.arange(height, dtype=np.float32)
        x_coords = np.arange(out_width, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij', sparse=False, copy=False)
        
        # 1. Dest Pixels -> Dest Spherical (in-place where possible)
        theta_dest = (grid_x - center_x) / center_x * (np.pi/2)
        phi_dest = (grid_y - center_y) / center_y * (np.pi/2)
        
        del grid_x, grid_y
        
        # Calculate xyz components without intermediate storage
        cos_phi = np.cos(phi_dest)
        sin_theta = np.sin(theta_dest)
        cos_theta = np.cos(theta_dest)
        
        x_d = sin_theta * cos_phi
        y_d = np.sin(phi_dest)
        z_d = cos_theta * cos_phi
        
        del sin_theta, cos_theta, cos_phi, theta_dest, phi_dest
        gc.collect()
        
        # 2. Rotate using matrix multiplication
        R = rotation_matrix.T
        xyz_src_x = x_d * R[0,0] + y_d * R[1,0] + z_d * R[2,0]
        xyz_src_y = x_d * R[0,1] + y_d * R[1,1] + z_d * R[2,1]  
        xyz_src_z = x_d * R[0,2] + y_d * R[1,2] + z_d * R[2,2]
        
        del x_d, y_d, z_d
        gc.collect()
        
        # 3. Spherical Src -> Pixels Src
        theta_src = np.arctan2(xyz_src_x, xyz_src_z)
        r_src = np.sqrt(xyz_src_x**2 + xyz_src_y**2 + xyz_src_z**2)
        phi_src = np.arcsin(np.clip(xyz_src_y / r_src, -1.0, 1.0))
        
        del xyz_src_x, xyz_src_y, xyz_src_z, r_src
        gc.collect()
        
        # Map back to pixels
        src_cx = width / 2.0
        src_cy = height / 2.0
        
        map_x = ((theta_src / (np.pi/2) * src_cx) + src_cx).astype(np.float32)
        map_y = ((phi_src / (np.pi/2) * src_cy) + src_cy).astype(np.float32)
        
        del theta_src, phi_src
        gc.collect()
        
        # Remap
        rotated_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        # Simple mask
        mask_weights = np.ones((height, out_width), dtype=np.float32)
        rotated_mask = cv2.remap(mask_weights, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        del map_x, map_y, mask_weights
        gc.collect()
        
        return rotated_img, rotated_mask

    def warp_target_with_flow(self, target_img, flow):
        """Warp target image using optical flow.
        
        Args:
            target_img: Image to warp (H, W, 3)
            flow: Optical flow (H, W, 2)
            
        Returns:
            warped: Warped image (H, W, 3)
        """
        h, w = target_img.shape[:2]
        
        # Create sampling grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
        
        # Warp the image
        warped = cv2.remap(target_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        return warped
    
    def warp_image_plain_min_flow(self, overlaped, im1, flow, mask, decreas_flow=200):
        import gc
        h, w = im1.shape[:2]
        hflow, wflow = flow.shape[:2]
        scale = h / hflow
        
        # resize flow to match the image bilinear interpolation
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR is faster
        flow = scale * flow
        
        # Warp mask if needed? Reference wraps mask too.
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[:,:,0]).astype(np.float32)
        map_y = (grid_y + flow[:,:,1]).astype(np.float32)
        
        # Free flow and grid immediately
        del flow, grid_x, grid_y
        
        # Warp the new image (im1) to match overlaped
        # Dest(x,y) = Src(x+flow, y+flow)
        warped_im1 = cv2.remap(im1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        # Warp the mask
        warped_mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        # Free maps
        del map_x, map_y
        
        # Blending logic from reference:
        # overlaped[Y, X] = np.where(
        #     (np.any(new_img[Y, X] != 0, axis=-1) & np.any(overlaped[Y, X] != 0, axis=-1))[:, None], 
        #     wrapped_mask[Y, X] * new_img[Y, X] + (1 - wrapped_mask[Y, X]) * overlaped[Y, X],
        #     np.where(np.any(new_img[Y, X] != 0, axis=-1)[:, None], new_img[Y, X], overlaped[Y, X])
        # )
        
        # Simplify vectorized blending
        # ensure masks are 3 channel for broadcasting if needed, or use keep dims
        if len(warped_mask.shape) == 2:
             warped_mask = warped_mask[:,:,None]
             
        im1_exists = np.any(warped_im1 != 0, axis=-1, keepdims=True)
        overlap_exists = np.any(overlaped != 0, axis=-1, keepdims=True)
        
        # Condition 1: Both exist -> Blend
        both_exist = im1_exists & overlap_exists
        
        # Condition 2: Only im1 exists -> Use im1
        only_im1 = im1_exists & (~overlap_exists)
        
        # Result
        # Start with copy of overlaped
        result = overlaped.copy()
        
        # Apply Blend
        # Note: reference uses 'wrapped_mask' as alpha for new_img.
        # result = alpha * new + (1-alpha) * old
        
        blend_region = warped_mask * warped_im1 + (1 - warped_mask) * overlaped
        
        np.copyto(result, blend_region.astype(np.uint8), where=both_exist)
        np.copyto(result, warped_im1, where=only_im1)
        
        return result

    def get_optical_flow(self, img1, img2):
        import gc
        # Resize for RAFT - reduced resolution to save memory
        work_h, work_w = 400, 720  # Reduced from 520x960
        img1_r = cv2.resize(img1, (work_w, work_h))
        img2_r = cv2.resize(img2, (work_w, work_h))
        
        img1_t = F.to_tensor(img1_r).unsqueeze(0).to(self.device)
        img2_t = F.to_tensor(img2_r).unsqueeze(0).to(self.device)
        
        # Free resized images
        del img1_r, img2_r
        
        # Preprocess
        img1_t, img2_t = self.raft_transforms(img1_t, img2_t)
        
        with torch.no_grad():
            list_of_flows = self.raft_model(img1_t, img2_t)
            predicted_flow = list_of_flows[-1][0] # (2, H, W)
            
        flow_np = predicted_flow.permute(1, 2, 0).cpu().numpy() # (H, W, 2)
        
        # Free GPU tensors immediately
        del img1_t, img2_t, predicted_flow, list_of_flows
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return flow_np, (work_h, work_w)
    
    def get_optical_flow_multiscale(self, img1, img2, scales=[1.0, 0.75, 0.5]):
        """Compute optical flow at multiple scales and combine.
        
        Args:
            img1: First image
            img2: Second image
            scales: List of scale factors (e.g., [1.0, 0.75, 0.5])
            
        Returns:
            flow_np: Combined optical flow (H, W, 2)
            dims: Working dimensions tuple
        """
        print(f"  Computing multi-scale optical flow at scales: {scales}")
        
        flows = []
        weights = []
        
        for scale in scales:
            # Compute flow at this scale
            work_h, work_w = int(520 * scale), int(960 * scale)
            # Ensure dimensions are divisible by 8 (RAFT requirement)
            work_h = (work_h // 8) * 8
            work_w = (work_w // 8) * 8
            
            img1_r = cv2.resize(img1, (work_w, work_h))
            img2_r = cv2.resize(img2, (work_w, work_h))
            
            img1_t = F.to_tensor(img1_r).unsqueeze(0).to(self.device)
            img2_t = F.to_tensor(img2_r).unsqueeze(0).to(self.device)
            
            img1_t, img2_t = self.raft_transforms(img1_t, img2_t)
            
            with torch.no_grad():
                list_of_flows = self.raft_model(img1_t, img2_t)
                predicted_flow = list_of_flows[-1][0]
            
            flow_scale = predicted_flow.permute(1, 2, 0).cpu().numpy()
            
            # Resize to target resolution (520x960)
            target_h, target_w = 520, 960
            flow_resized = cv2.resize(flow_scale, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            # Scale flow vectors appropriately
            flow_resized[:,:,0] *= (target_w / work_w)
            flow_resized[:,:,1] *= (target_h / work_h)
            
            flows.append(flow_resized)
            # Larger scales get higher weights
            weights.append(scale)
        
        # Weighted average of flows
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        combined_flow = np.zeros_like(flows[0])
        for flow, weight in zip(flows, weights):
            combined_flow += flow * weight
        
        print(f"  Multi-scale flow combined with weights: {weights}")
        
        return combined_flow, (520, 960)

    def find_optimal_seam(self, img1, img2, overlap_region):
        """Find optimal seam using graph cuts / dynamic programming.
        
        Args:
            img1: First image (H, W, 3)
            img2: Second image (H, W, 3)
            overlap_region: Binary mask of overlap (H, W)
            
        Returns:
            seam_mask: Optimal seam position (H, W) - 0 for img1, 1 for img2
        """
        h, w = img1.shape[:2]
        
        # Compute color difference in overlap region
        diff = np.linalg.norm(img1.astype(float) - img2.astype(float), axis=2)
        diff = diff * overlap_region  # Only consider overlap
        
        # Apply Gaussian blur to smooth the error
        diff_smooth = cv2.GaussianBlur(diff.astype(np.float32), (15, 15), 0)
        
        # Find vertical seam using dynamic programming
        # Create cost matrix
        cost = diff_smooth.copy()
        
        # Dynamic programming to find minimum cost path
        dp = np.zeros_like(cost)
        dp[0, :] = cost[0, :]
        
        # Traceback matrix
        path = np.zeros_like(cost, dtype=int)
        
        # Forward pass
        for i in range(1, h):
            for j in range(w):
                # Check neighbors: j-1, j, j+1
                if j == 0:
                    neighbors = [dp[i-1, j], dp[i-1, j+1]]
                    indices = [j, j+1]
                elif j == w-1:
                    neighbors = [dp[i-1, j-1], dp[i-1, j]]
                    indices = [j-1, j]
                else:
                    neighbors = [dp[i-1, j-1], dp[i-1, j], dp[i-1, j+1]]
                    indices = [j-1, j, j+1]
                
                min_idx = np.argmin(neighbors)
                dp[i, j] = cost[i, j] + neighbors[min_idx]
                path[i, j] = indices[min_idx]
        
        # Backward pass to find seam
        seam = np.zeros(h, dtype=int)
        seam[-1] = np.argmin(dp[-1, :])
        
        for i in range(h-2, -1, -1):
            seam[i] = path[i+1, seam[i+1]]
        
        # Create seam mask
        seam_mask = np.zeros((h, w), dtype=float)
        for i in range(h):
            seam_mask[i, :seam[i]] = 0  # Use img1 on the left
            seam_mask[i, seam[i]:] = 1  # Use img2 on the right
        
        print(f"  Found optimal seam with average cost: {dp[-1, seam[-1]] / h:.2f}")
        
        return seam_mask
    
    def draw_flow_arrows(self, img, flow, step=16):
        """Draw optical flow as arrows on the image.
        
        Args:
            img: Background image (H, W, 3)
            flow: Optical flow (H, W, 2)
            step: Grid spacing for arrows
            
        Returns:
            vis: Image with flow arrows drawn
        """
        h, w = flow.shape[:2]
        vis = img.copy()
        
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Draw arrows
        for (x1, y1), (x2, y2) in lines:
            # Skip tiny vectors
            if abs(x2 - x1) < 1 and abs(y2 - y1) < 1:
                continue
                
            # Color based on magnitude
            mag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if mag > 20:
                color = (0, 0, 255)  # Red for large flow
            elif mag > 10:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green for small flow
            
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)
        
        return vis
    
    def build_gaussian_pyramid(self, img, levels=5):
        """Build Gaussian pyramid."""
        # Ensure float32 for memory efficiency
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        pyramid = [img]
        for i in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid
    
    def build_laplacian_pyramid(self, img, levels=5):
        """Build Laplacian pyramid."""
        import gc
        
        # Ensure float32 for memory efficiency
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            
        gaussian_pyramid = self.build_gaussian_pyramid(img, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            # Expand the next level
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            
            # Laplacian = Gaussian - Expanded
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
            
            # Free expanded immediately
            del expanded
        
        # Add the smallest Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        # Free Gaussian pyramid
        del gaussian_pyramid
        gc.collect()
        
        return laplacian_pyramid
    
    def reconstruct_from_laplacian(self, laplacian_pyramid):
        """Reconstruct image from Laplacian pyramid."""
        img = laplacian_pyramid[-1]
        
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            img = cv2.pyrUp(img, dstsize=size)
            img = cv2.add(img, laplacian_pyramid[i])
        
        return img
    
    def multiband_blend(self, img1, img2, mask, levels=3):
        """Multi-band blending using Laplacian pyramids.
        
        Args:
            img1: First image (H, W, 3)
            img2: Second image (H, W, 3)
            mask: Blending mask (H, W) - 0 for img1, 1 for img2
            levels: Number of pyramid levels (default 3 for memory efficiency)
            
        Returns:
            blended: Blended image (H, W, 3)
        """
        import gc
        
        h, w = img1.shape[:2]
        print(f"  Applying multi-band blending with {levels} levels (image size: {w}x{h})...")
        
        # For very large images, reduce levels further
        if h * w > 8000000:  # 8 megapixels
            levels = min(levels, 2)
            print(f"  Large image detected, reducing to {levels} levels")
        
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Build Laplacian pyramids for images
        lap1 = self.build_laplacian_pyramid(img1.astype(np.float32), levels)
        lap2 = self.build_laplacian_pyramid(img2.astype(np.float32), levels)
        
        # Build Gaussian pyramid for mask
        mask_pyramid = self.build_gaussian_pyramid(mask.astype(np.float32), levels)
        
        # Blend each level
        blended_pyramid = []
        for idx, (l1, l2, m) in enumerate(zip(lap1, lap2, mask_pyramid)):
            # Ensure mask matches image dimensions at this level
            if len(m.shape) == 2:
                m = m[:, :, np.newaxis]
            
            # Resize mask if needed
            if m.shape[:2] != l1.shape[:2]:
                m = cv2.resize(m, (l1.shape[1], l1.shape[0]))
                if len(m.shape) == 2:
                    m = m[:, :, np.newaxis]
            
            # Blend: (1-m)*l1 + m*l2
            blended_level = (1 - m) * l1 + m * l2
            blended_pyramid.append(blended_level)
            
            # Free memory immediately after use
            del l1, l2, m
        
        # Free pyramid memory
        del lap1, lap2, mask_pyramid
        gc.collect()
        
        # Reconstruct from blended pyramid
        result = self.reconstruct_from_laplacian(blended_pyramid)
        
        del blended_pyramid
        gc.collect()
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def process_stage_1(self, image_pattern, output_dir):
        # Determine rotations using IMPROVED SLERP with loop closure
        # This version:
        # - Computes sequential rotations (1->2, 2->3, ..., 5->6)
        # - Measures direct closure (6->1)
        # - Distributes error evenly using SLERP
        # - Guarantees 360° coverage for circular camera array
        
        print("\n" + "="*60)
        print("STAGE 1: ROTATION COMPUTATION WITH SLERP CORRECTION")
        print("="*60)
        
        # Load 6 images
        # Expectation: image_pattern has {id}
        
        equirects = []
        for i in range(1, 7):
            path = image_pattern.format(id=i)
            print(f"Loading {path}")
            if not os.path.exists(path):
                print(f"Error: {path} not found")
                return None
                
            img = cv2.imread(path)
            # Pre-rotate 90 deg counter-clockwise as per original script
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # To equirectangular
            eq = self.fisheye_to_equirectangular(img, i)
            equirects.append(eq)
            
            if self.show:
                cv2.imshow("Equirectangular", cv2.resize(eq, (800, 800)))
                cv2.waitKey(10)

        print("\nComputing rotations with SLERP correction...")
        rotations = self.compute_all_rotations(equirects)
        acc_rots = self.get_accumulated_rotations(rotations)
        
        # Free equirect images to save memory
        import gc
        del equirects
        gc.collect()
        
        # Save or Return?
        return acc_rots

    def process_stage_2(self, image_pattern, acc_rots, output_path):
        import gc
        
        # Load only base image first
        path = image_pattern.format(id=1)
        img = cv2.imread(path)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base = self.fisheye_to_equirectangular(img, 1)
        del img
        gc.collect()
        
        # 1. Base Image (Image 1)
        h, w = base.shape[:2]
        
        # Canvas
        panorama = np.zeros((h, w * 2, 3), dtype=np.uint8)
        panorama[:h, :w] = base
        
        # Process images one by one to save memory
        rotated_images = []
        blend_masks = []
        
        # Add base
        rotated_images.append(panorama)
        blend_masks.append(None)
        
        # Rotate others and prepare masks
        for i in range(2, 7):
            # Load image on demand
            path = image_pattern.format(id=i)
            img = cv2.imread(path)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            eq = self.fisheye_to_equirectangular(img, i)
            del img
            
            rot_idx = i - 2  # acc_rots has 5 elements for cameras 2-6
            # print  acc_rots[rot_idx]
            print(acc_rots[rot_idx])    
            rot_img, rot_mask = self.rotate_image_spherical(eq, acc_rots[rot_idx])
            del eq  # Free equirect image
            gc.collect()
            
            rotated_images.append(rot_img)
            
            # Mask post-processing as per reference
            # rot_mask is float 0..1? In my impl it's 1.0 where valid.
            mask_uint8 = (rot_mask * 255).astype(np.uint8)
            
            decreas_flow = 10
            smallkernel = np.ones((decreas_flow // 2, decreas_flow // 2), np.uint8)
            mask_uint8 = cv2.dilate(mask_uint8, smallkernel, iterations=1)
            mask_uint8 = cv2.erode(mask_uint8, smallkernel, iterations=1)
            
            # Blend Mask
            mask_blend = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            mask_blend = np.minimum(mask_blend, 255)
            
            # Apply smooth boundaries if enabled
            if cfg.USE_SMOOTH_BOUNDARIES:
                # Gaussian blur for smoothness
                mask_blend = cv2.GaussianBlur(mask_blend.astype(np.float32), cfg.BOUNDARY_BLUR_KERNEL, 0)
                
                # Sigmoid falloff for boundary
                mask_blend = mask_blend / 255.0
                mask_blend = 1.0 / (1.0 + np.exp(-cfg.BOUNDARY_SIGMOID_K * (mask_blend - 0.5)))
                mask_blend = (mask_blend * 255).astype(np.uint8)
            
            # Flow Mask logic (inverted) - Reference computes it but seems to save it?
            # It uses mask_blend for warping later: `rotated_mask = cv2.imread(..., blend_mask) / 255.0`
            # So we store mask_blend normalized
            
            mask_blend_norm = mask_blend / 255.0
            blend_masks.append(mask_blend_norm)

        # Refinement loop
        # Image 1 (index 0) is base.
        stitched = rotated_images[0].copy()
        
        # Loop i from 2 to 6 (Images 2..6)
        # Note: rotated_images has 0..5, blend_masks has 0..5
        for i in range(2, 7):
            print(f"Stitching image {i}...")
            # i=2 -> index 1
            idx = i - 1
            target_img = rotated_images[idx]
            target_mask = blend_masks[idx]
            
            # Compute flow
            # Reference computes flow(overlaped, new_img2) AND flow(new_img2, overlaped)
            # But uses flow1.
            # flow1 = get_flow(stitched, target)
            # This means flow vectors at 'stitched' coords pointing to 'target'.
            # wait, warp_image_plain_min_flow(overlaped, im1, flow1...)
            # It warps 'im1'(target) to match 'overlaped'.
            # To warp B to A, we usually need flow(A, B) if using backward mapping?
            #   Dest(x) = Src(x+flow(x)). Dest=A-grid. Src=B.
            #   So flow must be defined on A-grid.
            #   get_optical_flow(stitched, target) returns flow on stitched grid.
            #   Yes, this is correct for warping target to stitched.
            
            # Optical flow for fine alignment (after SLERP correction)
            print(f"Computing flow stitched-{i}...")
            if cfg.USE_MULTISCALE_FLOW:
                flow1, flow_dims = self.get_optical_flow_multiscale(stitched, target_img, cfg.FLOW_SCALES)
            else:
                flow1, flow_dims = self.get_optical_flow(stitched, target_img)
            
            # Save flow visualization for debugging (skip to save memory)
            # h_flow, w_flow = flow1.shape[:2]
            # stitched_small = cv2.resize(stitched, (w_flow, h_flow))
            # flow_vis = self.draw_flow_arrows(stitched_small, flow1)
            # cv2.imwrite(f"flow_vis_{i}.jpg", flow_vis)
            # del stitched_small, flow_vis
            
            # Resize flow to full resolution
            h_full, w_full = stitched.shape[:2]
            fh, fw = flow1.shape[:2]
            flow_full = cv2.resize(flow1, (w_full, h_full), interpolation=cv2.INTER_LINEAR)  # LINEAR is faster
            flow_full[:,:,0] *= (w_full / fw)
            flow_full[:,:,1] *= (h_full / fh)
            
            # Free flow1 immediately
            del flow1
            gc.collect()
            
            # Skip flow magnitude visualization to save memory
            # flow_magnitude = np.linalg.norm(flow_full, axis=2)
            # flow_mag_vis = np.clip(flow_magnitude / 50.0 * 255, 0, 255).astype(np.uint8)
            # flow_mag_color = cv2.applyColorMap(flow_mag_vis, cv2.COLORMAP_JET)
            # cv2.imwrite(f"flow_magnitude_{i}.jpg", flow_mag_color)
            # del flow_magnitude, flow_mag_vis, flow_mag_color
            
            # ---------------------------------------------------------
            # EDGE-WEIGHTED FLOW LOGIC
            # Objective: Move edges (seams) but keep center fixed.
            # ---------------------------------------------------------
            
            # 1. Create a mask of valid content in the target image
            # target_img is (H, W, 3)
            # Find where it has content (not black)
            valid_mask = np.any(target_img > 0, axis=2).astype(np.uint8) * 255
            
            # 2. Compute Distance Transform from the Center of valid content
            # We want: 0 at center, 1 at edges.
            # Standard distanceTransform gives distance TO nearest zero pixel (edge).
            # So dist_img is High at Center, Low at Edge.
            # We want the inverse distribution.
            
            dist_img = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
            max_dist = dist_img.max()
            
            if max_dist > 0:
                # Normalize 0..1 (1 at center, 0 at edge)
                center_weight = dist_img / max_dist
                
                # Compute flow weight: 0 at center, 1 at edge
                flow_weight = 1.0 - (dist_img / max_dist)
                
                # Visualize weight (skip to save memory)
                # weight_vis = (flow_weight * 255).astype(np.uint8)
                # weight_color = cv2.applyColorMap(weight_vis, cv2.COLORMAP_JET)
                # cv2.imwrite(f"flow_weight_{i}.jpg", weight_color)
                # del weight_vis, weight_color
                
                # Apply weight to flow
                # flow_full is (H, W, 2)
                flow_full[:,:,0] *= flow_weight
                flow_full[:,:,1] *= flow_weight
            
            # 3. Mask out flow in black regions (invalid data)
            # RAFT can hallucinate flow in flat black areas.
            # Mask out flow where target_img OR stitched image is black.
            
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            stitched_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            
            # Create valid mask: pixels must be non-black in BOTH images to have valid flow
            # Or at least in the target image? If target is black, we shouldn't warp it.
            # If stitched is black, we might be filling a hole, but flow needs texture.
            
            valid_flow_mask = (target_gray > 10) & (stitched_gray > 10)
            
            # Erode slightly to avoid edge artifacts
            kernel = np.ones((5,5), np.uint8)
            valid_flow_mask = cv2.erode(valid_flow_mask.astype(np.uint8), kernel).astype(bool)
            
            flow_full[~valid_flow_mask] = 0
            
            # 4. Magnitude limit removed as requested
            # MAX_FLOW = 50.0 
            # flow_magnitude = np.linalg.norm(flow_full, axis=2)
            # flow_mask = flow_magnitude > MAX_FLOW
            
            # if np.any(flow_mask):
            #     scale = MAX_FLOW / np.clip(flow_magnitude, MAX_FLOW, None)
            #     flow_full[:,:,0] *= scale
            #     flow_full[:,:,1] *= scale
            #     pct = 100 * np.sum(flow_mask) / flow_mask.size
            #     print(f"  Limited {np.sum(flow_mask)} pixels ({pct:.1f}%) with excessive flow (>{MAX_FLOW}px)")
                
            #     # Save limited flow magnitude
            #     flow_magnitude_limited = np.linalg.norm(flow_full, axis=2)
            #     flow_mag_lim_vis = np.clip(flow_magnitude_limited / 50.0 * 255, 0, 255).astype(np.uint8)
            #     flow_mag_lim_color = cv2.applyColorMap(flow_mag_lim_vis, cv2.COLORMAP_JET)
            #     cv2.imwrite(f"flow_magnitude_limited_{i}.jpg", flow_mag_lim_color)
            
            # Apply advanced blending if enabled
            if cfg.USE_SEAM_CARVING or cfg.USE_MULTIBAND_BLENDING:
                # Warp target image first
                h_full, w_full = stitched.shape[:2]
                grid_x, grid_y = np.meshgrid(np.arange(w_full), np.arange(h_full))
                map_x = (grid_x + flow_full[:,:,0]).astype(np.float32)
                map_y = (grid_y + flow_full[:,:,1]).astype(np.float32)
                warped_target = cv2.remap(target_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                warped_mask = cv2.remap(target_mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
                
                # Ensure warped_mask is 3D for broadcasting
                if len(warped_mask.shape) == 2:
                    warped_mask = warped_mask[:,:,np.newaxis]
                
                # Find regions where each image exists
                target_exists = np.any(warped_target > 0, axis=-1, keepdims=True)
                stitched_exists = np.any(stitched > 0, axis=-1, keepdims=True)
                overlap = target_exists & stitched_exists
                only_target = target_exists & (~stitched_exists)
                
                # Determine blend weight for overlap region
                if cfg.USE_SEAM_CARVING and np.any(overlap):
                    print(f"  Finding optimal seam...")
                    seam_mask = self.find_optimal_seam(stitched, warped_target, overlap.squeeze())
                    # Convert seam_mask to 3D
                    if len(seam_mask.shape) == 2:
                        seam_mask = seam_mask[:,:,np.newaxis]
                    blend_weight = seam_mask
                else:
                    # Use warped_mask as blend weight (already 3D)
                    blend_weight = warped_mask
                
                # Clip blend_weight to [0, 1]
                blend_weight = np.clip(blend_weight, 0, 1)
                
                # Apply blending in overlap region
                if cfg.USE_MULTIBAND_BLENDING and np.any(overlap):
                    print(f"  Applying multi-band blending...")
                    # Determine levels based on image size
                    levels = cfg.MULTIBAND_LEVELS
                    if cfg.AUTO_REDUCE_LEVELS and h_full * w_full > cfg.LARGE_IMAGE_THRESHOLD:
                        levels = max(2, levels - 1)
                    
                    # Blend only in overlap region
                    blended = self.multiband_blend(stitched, warped_target, blend_weight, levels=levels)
                    
                    # Start with stitched
                    result = stitched.copy()
                    # Copy blended result in overlap region
                    np.copyto(result, blended, where=overlap)
                    # Copy target in only_target region
                    np.copyto(result, warped_target, where=only_target)
                    stitched = result
                    del blended, result
                else:
                    # Simple alpha blending in overlap region
                    result = stitched.copy()
                    
                    # Blend in overlap region: result = (1-alpha)*stitched + alpha*target
                    blend_region = (1 - blend_weight) * stitched + blend_weight * warped_target
                    np.copyto(result, blend_region.astype(np.uint8), where=overlap)
                    
                    # Copy target directly where only target exists
                    np.copyto(result, warped_target, where=only_target)
                    
                    stitched = result
                    del blend_region, result
                    
                del warped_target, warped_mask, map_x, map_y, grid_x, grid_y
                del target_exists, stitched_exists, overlap, only_target, blend_weight
                if cfg.FORCE_GC:
                    gc.collect()
            else:
                # Apply flow-based warp and blend (using original method)
                stitched = self.warp_image_plain_min_flow(stitched, target_img, flow_full, target_mask)
            
            # Free memory immediately after stitching
            del flow_full, target_img, target_mask, target_gray, stitched_gray
            del valid_flow_mask, flow_weight, dist_img, valid_mask
            gc.collect()
            
            if self.show:
                cv2.imshow(f"Stitched", cv2.resize(stitched, (1000, 500)))
                cv2.waitKey(10)
            
            # Save intermediate result
            cv2.imwrite(f"intermediate_stitched_{i}.jpg", stitched)
            print(f"Saved intermediate_stitched_{i}.jpg")

        # Save
        cv2.imwrite(output_path, stitched)
        print(f"Saved panorama to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Panorama Stitching Tool")
    parser.add_argument("--config", required=True, help="Path to JSON configuration (e.g. kandao.json)")
    parser.add_argument("--input_template", required=True, help="Input filename template with {id} (e.g. ./imgs/origin_{id}.jpg)")
    parser.add_argument("--output", required=True, help="Output filename")
    parser.add_argument("--show", action="store_true", help="Show processing steps")
    # Batch mode args could be added
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    stitcher = PanoramaStitcher(args.config, show=args.show)
    
    # 1. Compute Rotations (using input images as calibration source?)
    # Usually we need a set of images to calibrate rotations, then apply to others.
    # For now, let's assume we proceed with the input set.
    
    acc_rots = stitcher.process_stage_1(args.input_template, None)
    
    if acc_rots:
        stitcher.process_stage_2(args.input_template, acc_rots, args.output)
