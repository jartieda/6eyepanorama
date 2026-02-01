from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QFileDialog, QCheckBox, 
                               QScrollArea, QGridLayout, QGroupBox, QTextEdit, 
                               QSizePolicy, QSplitter, QFrame, QTabWidget,
                               QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QTransform
import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'stich_old'))

from logic.stitcher_logic import PanoramaStitcher
from .panorama_viewer_360 import PanoramaViewer360

class ProcessThread(QThread):
    status_update = Signal(str)
    image_update = Signal(str, np.ndarray)  # (name, image)
    finished_signal = Signal()
    
    def __init__(self, config_path, input_template, output_path, video_mode, rotation_method='slerp'):
        super().__init__()
        self.config_path = config_path
        self.input_template = input_template
        self.output_path = output_path
        self.video_mode = video_mode
        self.rotation_method = rotation_method
        self.running = True
        
    def run(self):
        try:
            if PanoramaStitcher is None:
                self.status_update.emit("Error: PanoramaStitcher not available")
                return
                
            method_name = "SLERP" if self.rotation_method == 'slerp' else "Bundle Adjustment"
            self.status_update.emit(f"Initializing stitcher (Method: {method_name})...")
            stitcher = PanoramaStitcher(self.config_path, show=False, rotation_method=self.rotation_method)
            
            # Load and show input images
            self.status_update.emit("Loading input images...")
            for i in range(1, 7):
                if not self.running:
                    return
                path = self.input_template.format(id=i)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        self.image_update.emit(f"input_{i}", img)
            
            # Stage 1: Compute rotations
            self.status_update.emit("Stage 1: Computing camera rotations...")
            acc_rots = stitcher.process_stage_1(self.input_template, None)
            
            if acc_rots is None:
                self.status_update.emit("Error: Could not compute rotations")
                return
            
            self.status_update.emit(f"Computed {len(acc_rots)} rotation matrices")
            
            # Stage 2: Stitch panorama
            self.status_update.emit("Stage 2: Stitching panorama...")
            stitcher.process_stage_2(self.input_template, acc_rots, self.output_path)
            
            # Load and show result
            if os.path.exists(self.output_path):
                result = cv2.imread(self.output_path)
                if result is not None:
                    self.image_update.emit("output", result)
                    self.status_update.emit(f"Done! Saved to {self.output_path}")
            else:
                self.status_update.emit("Done but output file not found")
                
            # Load intermediate images if available
            for i in range(2, 7):
                intermediate_path = f"intermediate_stitched_{i}.jpg"
                if os.path.exists(intermediate_path):
                    img = cv2.imread(intermediate_path)
                    if img is not None:
                        self.image_update.emit(f"intermediate_{i}", img)
            
        except Exception as e:
            import traceback
            self.status_update.emit(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False

class StitcherView(QWidget):
    def __init__(self):
        super().__init__()
        self.process_thread = None
        self.output_image = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # === TOP CONTROLS ===
        controls_frame = QFrame()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(3)
        
        # Config inputs - compact
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Config:"), 0)
        self.config_input = QLineEdit("./stich_old/kandao.json")
        config_layout.addWidget(self.config_input, 1)
        self.config_btn = QPushButton("...")
        self.config_btn.setMaximumWidth(40)
        self.config_btn.clicked.connect(lambda: self.browse_file(self.config_input, "JSON Files (*.json)"))
        config_layout.addWidget(self.config_btn, 0)
        controls_layout.addLayout(config_layout)
        
        # Input template - compact
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"), 0)
        self.input_input = QLineEdit("./stich_old/dataset/origin_{id}_1.jpg")
        input_layout.addWidget(self.input_input, 1)
        self.input_btn = QPushButton("...")
        self.input_btn.setMaximumWidth(40)
        self.input_btn.clicked.connect(lambda: self.browse_file(self.input_input, "Images (*.jpg *.png)"))
        input_layout.addWidget(self.input_btn, 0)
        controls_layout.addLayout(input_layout)
        
        # Output - compact
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"), 0)
        self.output_input = QLineEdit("output_panorama.jpg")
        output_layout.addWidget(self.output_input, 1)
        controls_layout.addLayout(output_layout)
        
        # Rotation Method - compact
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Rotation Method:"), 0)
        self.rotation_method_combo = QComboBox()
        self.rotation_method_combo.addItem("SLERP (Uniform 360°)", "slerp")
        self.rotation_method_combo.addItem("Bundle Adjustment (Optimal)", "bundle")
        self.rotation_method_combo.setToolTip(
            "SLERP: Guarantees uniform angular distribution\n"
            "Bundle Adjustment: Minimizes global reprojection error"
        )
        method_layout.addWidget(self.rotation_method_combo, 1)
        controls_layout.addLayout(method_layout)
        
        # Buttons and status in one line
        btn_status_layout = QHBoxLayout()
        self.video_mode_check = QCheckBox("Video Mode")
        btn_status_layout.addWidget(self.video_mode_check)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        btn_status_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_status_layout.addWidget(self.stop_btn)
        
        self.status_label = QLabel("Ready")
        btn_status_layout.addWidget(self.status_label, 1)
        
        controls_layout.addLayout(btn_status_layout)
        controls_frame.setLayout(controls_layout)
        controls_frame.setMaximumHeight(145)
        main_layout.addWidget(controls_frame)
        
        # === SPLITTER FOR CONTENT ===
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT: Input images (compact grid)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(2, 2, 2, 2)
        
        input_label = QLabel("<b>Inputs</b>")
        left_layout.addWidget(input_label)
        
        input_grid = QGridLayout()
        input_grid.setSpacing(2)
        self.input_labels = []
        for i in range(6):
            label = QLabel(f"{i+1}")
            label.setMinimumSize(120, 120)
            label.setMaximumSize(300, 300)
            label.setScaledContents(True)
            label.setStyleSheet("border: 1px solid #3d3d3d; background-color: #000;")
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.input_labels.append(label)
            input_grid.addWidget(label, i // 3, i % 3)
        left_layout.addLayout(input_grid)
        left_widget.setLayout(left_layout)
        
        # RIGHT: Output and debug
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(5)
        
        output_label_title = QLabel("<b>Output Panorama</b>")
        right_layout.addWidget(output_label_title)
        
        # Tabs for different views
        self.view_tabs = QTabWidget()
        
        # TAB 1: Flat View with zoom
        flat_view_widget = QWidget()
        flat_view_layout = QVBoxLayout()
        flat_view_layout.setContentsMargins(2, 2, 2, 2)
        
        # Zoom controls
        zoom_controls = QHBoxLayout()
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_controls.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_controls.addWidget(self.zoom_out_btn)
        
        self.zoom_reset_btn = QPushButton("Reset")
        self.zoom_reset_btn.setMaximumWidth(60)
        self.zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_controls.addWidget(self.zoom_reset_btn)
        
        self.zoom_label = QLabel("100%")
        zoom_controls.addWidget(self.zoom_label)
        zoom_controls.addStretch()
        
        flat_view_layout.addLayout(zoom_controls)
        
        # Graphics view with scroll
        self.flat_graphics_view = QGraphicsView()
        self.flat_scene = QGraphicsScene()
        self.flat_graphics_view.setScene(self.flat_scene)
        self.flat_pixmap_item = None
        self.zoom_factor = 1.0
        self.flat_graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.flat_graphics_view.setStyleSheet("border: 2px solid #007acc; background-color: #000;")
        flat_view_layout.addWidget(self.flat_graphics_view)
        
        flat_view_widget.setLayout(flat_view_layout)
        
        # TAB 2: 360 View
        self.viewer_360 = PanoramaViewer360()
        
        # Add tabs
        self.view_tabs.addTab(flat_view_widget, "Flat View")
        self.view_tabs.addTab(self.viewer_360, "360° View")
        
        right_layout.addWidget(self.view_tabs, 2)
        
        # Intermediate steps (compact, collapsible)
        self.debug_group = QGroupBox("Intermediate Steps")
        self.debug_group.setCheckable(True)
        self.debug_group.setChecked(False)
        self.debug_group.setMaximumHeight(200)
        debug_layout = QVBoxLayout()
        debug_layout.setContentsMargins(2, 2, 2, 2)
        self.debug_scroll = QScrollArea()
        self.debug_scroll.setWidgetResizable(True)
        self.debug_scroll.setMinimumHeight(80)
        self.debug_container = QWidget()
        self.debug_container_layout = QHBoxLayout()
        self.debug_container_layout.setSpacing(2)
        self.debug_container.setLayout(self.debug_container_layout)
        self.debug_scroll.setWidget(self.debug_container)
        debug_layout.addWidget(self.debug_scroll)
        self.debug_group.setLayout(debug_layout)
        right_layout.addWidget(self.debug_group, 1)
        
        right_widget.setLayout(right_layout)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)  # Left 30%
        splitter.setStretchFactor(1, 2)  # Right 70%
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)
    
    def browse_file(self, line_edit, file_filter):
        filename, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if filename:
            line_edit.setText(filename)
    
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.apply_zoom()
    
    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.apply_zoom()
    
    def zoom_reset(self):
        self.zoom_factor = 1.0
        self.apply_zoom()
    
    def apply_zoom(self):
        if self.flat_pixmap_item is not None:
            transform = QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            self.flat_graphics_view.setTransform(transform)
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
    
    def start_processing(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Starting...")
        
        rotation_method = self.rotation_method_combo.currentData()
        
        self.process_thread = ProcessThread(
            self.config_input.text(),
            self.input_input.text(),
            self.output_input.text(),
            self.video_mode_check.isChecked(),
            rotation_method
        )
        self.process_thread.status_update.connect(self.update_status)
        self.process_thread.image_update.connect(self.update_image)
        self.process_thread.finished_signal.connect(self.processing_finished)
        self.process_thread.start()
    
    def stop_processing(self):
        if self.process_thread:
            self.process_thread.stop()
            self.status_label.setText("Stopping...")
    
    def processing_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def update_status(self, status):
        self.status_label.setText(status)
    
    def update_image(self, name, img_array):
        # Convert cv2 image to QPixmap
        if img_array is None:
            return
        
        # Update appropriate label
        if name.startswith("input_"):
            # Resize for display
            h, w = img_array.shape[:2]
            if h > 400 or w > 400:
                scale = min(400/h, 400/w)
                img_array_resized = cv2.resize(img_array, (int(w*scale), int(h*scale)))
            else:
                img_array_resized = img_array
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            idx = int(name.split("_")[1]) - 1
            if 0 <= idx < 6:
                self.input_labels[idx].setPixmap(pixmap)
                
        elif name == "output":
            # Store original for 360 viewer
            self.output_image = img_array.copy()
            
            # Update flat view
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Update flat graphics view
            self.flat_scene.clear()
            self.flat_pixmap_item = self.flat_scene.addPixmap(pixmap)
            self.flat_scene.setSceneRect(pixmap.rect())
            self.flat_graphics_view.fitInView(self.flat_pixmap_item, Qt.KeepAspectRatio)
            self.zoom_reset()
            
            # Update 360 viewer
            self.viewer_360.set_panorama(img_array)
            
        else:
            # Resize for display
            h, w = img_array.shape[:2]
            if h > 400 or w > 400:
                scale = min(400/h, 400/w)
                img_array = cv2.resize(img_array, (int(w*scale), int(h*scale)))
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Add to debug area (horizontal)
            container = QWidget()
            vlayout = QVBoxLayout()
            vlayout.setContentsMargins(0, 0, 0, 0)
            vlayout.setSpacing(2)
            
            title = QLabel(f"<small>{name}</small>")
            title.setAlignment(Qt.AlignCenter)
            vlayout.addWidget(title)
            
            debug_label = QLabel()
            debug_label.setPixmap(pixmap)
            debug_label.setMaximumSize(150, 100)
            debug_label.setScaledContents(True)
            debug_label.setStyleSheet("border: 1px solid #3d3d3d;")
            vlayout.addWidget(debug_label)
            
            container.setLayout(vlayout)
            self.debug_container_layout.addWidget(container)
