from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QSpinBox, QTextEdit, 
                               QScrollArea, QSizePolicy)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stich_old'))

from logic.calibration_logic import findcircle_on_image

class CalibrationThread(QThread):
    status_update = Signal(str)
    image_update = Signal(np.ndarray)
    result_update = Signal(str)
    finished_signal = Signal()
    
    def __init__(self, image_dir, rows, cols):
        super().__init__()
        self.image_dir = image_dir
        self.rows = rows
        self.cols = cols
        
    def run(self):
        try:
            import glob
            
            CHECKERBOARD = (self.rows, self.cols)
            
            self.status_update.emit(f"Searching images in {self.image_dir}...")
            
            subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            
            # Prepare object points
            objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:2*CHECKERBOARD[0]:2, 0:2*CHECKERBOARD[1]:2].T.reshape(-1, 2)
            
            _img_shape = None
            objpoints = []
            imgpoints = []
            
            search_pattern = os.path.join(self.image_dir, '*.jpg')
            images = glob.glob(search_pattern)
            
            if not images:
                self.status_update.emit("No images found!")
                return
                
            self.status_update.emit(f"Found {len(images)} images. Processing...")
            
            circles = []
            found_count = 0
            
            for idx, fname in enumerate(images):
                self.status_update.emit(f"Processing {os.path.basename(fname)} ({idx+1}/{len(images)})")
                
                img = cv2.imread(fname)
                if img is None:
                    continue
                
                if _img_shape is None:
                    _img_shape = img.shape[:2]
                else:
                    if _img_shape != img.shape[:2]:
                        continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard
                ret, corners = cv2.findChessboardCorners(
                    gray, CHECKERBOARD, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                # Find circles
                c = findcircle_on_image(img, gray, show=False)
                if c is not None:
                    circles.extend(c)
                
                # Draw results
                vis_img = img.copy()
                if ret:
                    found_count += 1
                    objpoints.append(objp)
                    cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(vis_img, CHECKERBOARD, corners, ret)
                    
                # Resize for display
                h, w = vis_img.shape[:2]
                if h > 600:
                    scale = 600 / h
                    vis_img = cv2.resize(vis_img, (int(w*scale), int(h*scale)))
                
                self.image_update.emit(vis_img)
            
            N_OK = len(objpoints)
            self.status_update.emit(f"Found {N_OK} valid images for calibration")
            
            if N_OK == 0:
                self.result_update.emit("Not enough valid images for calibration.")
                return
            
            # Run calibration
            self.status_update.emit("Running calibration...")
            dims = _img_shape[::-1]
            flags = cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER
            
            try:
                rms, k, xi, d, rvecs, tvecs, idx = cv2.omnidir.calibrate(
                    objectPoints=objpoints, 
                    imagePoints=imgpoints, 
                    size=dims, 
                    K=None, xi=None, D=None,
                    flags=flags,
                    criteria=subpix_criteria
                )
                
                result_text = f"Calibration Results:\n\n"
                result_text += f"Found {N_OK} valid images\n"
                result_text += f"Image size: {dims}\n"
                result_text += f"Chessboard: {self.rows}x{self.cols}\n\n"
                result_text += f"RMS Error: {rms:.4f}\n\n"
                result_text += f"Camera Matrix K:\n{k}\n\n"
                result_text += f"Distortion D:\n{d}\n\n"
                result_text += f"Xi (omnidirectional parameter): {xi}\n\n"
                
                if circles:
                    avg_x = np.mean([cir[0] for cir in circles])
                    avg_y = np.mean([cir[1] for cir in circles])
                    avg_r = np.mean([cir[2] for cir in circles])
                    result_text += f"Average Circle:\n"
                    result_text += f"  Center X: {avg_x:.2f}\n"
                    result_text += f"  Center Y: {avg_y:.2f}\n"
                    result_text += f"  Radius: {avg_r:.2f}\n"
                
                self.result_update.emit(result_text)
                self.status_update.emit("Calibration completed successfully!")
                
            except Exception as e:
                self.result_update.emit(f"Calibration failed: {str(e)}")
                self.status_update.emit(f"Error during calibration: {str(e)}")
            
        except Exception as e:
            import traceback
            self.status_update.emit(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.finished_signal.emit()

class CalibrationView(QWidget):
    def __init__(self):
        super().__init__()
        self.calib_thread = None
        self.init_ui()
        
    def init_ui(self):
        # Main scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout()
        
        # Image directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Images Directory:"), 0)
        self.dir_input = QLineEdit("../stich_old/calib")
        dir_layout.addWidget(self.dir_input, 1)
        self.dir_btn = QPushButton("Browse")
        self.dir_btn.setMaximumWidth(100)
        self.dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_btn, 0)
        layout.addLayout(dir_layout)
        
        # Chessboard size
        chess_layout = QHBoxLayout()
        chess_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setValue(7)
        self.rows_spin.setMinimum(3)
        self.rows_spin.setMaximum(20)
        chess_layout.addWidget(self.rows_spin)
        
        chess_layout.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setValue(9)
        self.cols_spin.setMinimum(3)
        self.cols_spin.setMaximum(20)
        chess_layout.addWidget(self.cols_spin)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        chess_layout.addWidget(self.calibrate_btn)
        chess_layout.addStretch()
        
        layout.addLayout(chess_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Preview image
        preview_label = QLabel("<b>Preview:</b>")
        layout.addWidget(preview_label)
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setScaledContents(True)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_label, 1)
        
        # Results
        result_label = QLabel("<b>Results:</b>")
        layout.addWidget(result_label)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(150)
        self.result_text.setMaximumHeight(250)
        layout.addWidget(self.result_text, 0)
        
        layout.addStretch()
        content.setLayout(layout)
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
    
    def start_calibration(self):
        self.calibrate_btn.setEnabled(False)
        self.result_text.clear()
        self.status_label.setText("Starting calibration...")
        
        self.calib_thread = CalibrationThread(
            self.dir_input.text(),
            self.rows_spin.value(),
            self.cols_spin.value()
        )
        self.calib_thread.status_update.connect(self.update_status)
        self.calib_thread.image_update.connect(self.update_image)
        self.calib_thread.result_update.connect(self.update_result)
        self.calib_thread.finished_signal.connect(self.calibration_finished)
        self.calib_thread.start()
    
    def calibration_finished(self):
        self.calibrate_btn.setEnabled(True)
    
    def update_status(self, status):
        self.status_label.setText(status)
    
    def update_image(self, img_array):
        if img_array is None:
            return
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.preview_label.setPixmap(pixmap)
    
    def update_result(self, result):
        self.result_text.setText(result)
