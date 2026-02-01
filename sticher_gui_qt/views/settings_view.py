from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QTextEdit, QMessageBox,
                               QScrollArea, QGroupBox, QGridLayout, QDoubleSpinBox,
                               QSpinBox, QComboBox, QFrame, QTabWidget)
from PySide6.QtCore import Qt
import json


class CameraConfigForm(QWidget):
    """Form for a single camera configuration"""
    
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Title
        title = QLabel(f"<b>Camera {self.camera_index + 1}</b>")
        layout.addWidget(title)
        
        # Scroll for the form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_content = QWidget()
        form_layout = QGridLayout()
        form_layout.setSpacing(5)
        row = 0
        
        # === Image Dimensions ===
        dims_label = QLabel("<b>Image Dimensions</b>")
        form_layout.addWidget(dims_label, row, 0, 1, 4)
        row += 1
        
        form_layout.addWidget(QLabel("Width:"), row, 0)
        self.width = QSpinBox()
        self.width.setRange(0, 10000)
        self.width.setValue(3000)
        form_layout.addWidget(self.width, row, 1)
        
        form_layout.addWidget(QLabel("Height:"), row, 2)
        self.height = QSpinBox()
        self.height.setRange(0, 10000)
        self.height.setValue(4000)
        form_layout.addWidget(self.height, row, 3)
        row += 1
        
        # === Fisheye Circle ===
        circle_label = QLabel("<b>Fisheye Circle</b>")
        form_layout.addWidget(circle_label, row, 0, 1, 4)
        row += 1
        
        form_layout.addWidget(QLabel("Center X:"), row, 0)
        self.center_x = QDoubleSpinBox()
        self.center_x.setRange(0, 5000)
        self.center_x.setDecimals(3)
        self.center_x.setValue(1500)
        form_layout.addWidget(self.center_x, row, 1)
        
        form_layout.addWidget(QLabel("Center Y:"), row, 2)
        self.center_y = QDoubleSpinBox()
        self.center_y.setRange(0, 5000)
        self.center_y.setDecimals(3)
        self.center_y.setValue(2000)
        form_layout.addWidget(self.center_y, row, 3)
        row += 1
        
        form_layout.addWidget(QLabel("Radius:"), row, 0)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0, 5000)
        self.radius.setDecimals(3)
        self.radius.setValue(1900)
        form_layout.addWidget(self.radius, row, 1)
        
        form_layout.addWidget(QLabel("UK:"), row, 2)
        self.uk = QSpinBox()
        self.uk.setRange(0, 100)
        self.uk.setValue(25)
        form_layout.addWidget(self.uk, row, 3)
        row += 1
        
        # === Rotation Parameters (d1, d2, d3) ===
        rot_label = QLabel("<b>Rotation (degrees)</b>")
        form_layout.addWidget(rot_label, row, 0, 1, 4)
        row += 1
        
        form_layout.addWidget(QLabel("d1 (Roll):"), row, 0)
        self.d1 = QDoubleSpinBox()
        self.d1.setRange(-180, 180)
        self.d1.setDecimals(3)
        self.d1.setValue(0)
        form_layout.addWidget(self.d1, row, 1)
        
        form_layout.addWidget(QLabel("d2 (Pitch):"), row, 2)
        self.d2 = QDoubleSpinBox()
        self.d2.setRange(-180, 180)
        self.d2.setDecimals(3)
        self.d2.setValue(0)
        form_layout.addWidget(self.d2, row, 3)
        row += 1
        
        form_layout.addWidget(QLabel("d3 (Yaw):"), row, 0)
        self.d3 = QDoubleSpinBox()
        self.d3.setRange(-180, 180)
        self.d3.setDecimals(3)
        self.d3.setValue(0)
        form_layout.addWidget(self.d3, row, 1)
        row += 1
        
        # === K matrix (3x3) - Intrinsic Camera Matrix ===
        k_label = QLabel("<b>K Matrix (Intrinsic Camera Matrix 3×3)</b>")
        form_layout.addWidget(k_label, row, 0, 1, 4)
        row += 1
        
        k_header = QLabel("<i>fx, skew, cx | 0, fy, cy | 0, 0, 1</i>")
        k_header.setStyleSheet("color: #888;")
        form_layout.addWidget(k_header, row, 0, 1, 4)
        row += 1
        
        self.k_matrix = []
        for i in range(3):
            k_row = []
            for j in range(3):
                k_spin = QDoubleSpinBox()
                k_spin.setRange(-10000, 10000)
                k_spin.setDecimals(6)
                k_spin.setMinimumWidth(100)
                if i == j:
                    k_spin.setValue(1800 if i < 2 else 1)
                elif j == 2:
                    k_spin.setValue(2000 if i == 0 else 1500 if i == 1 else 1)
                else:
                    k_spin.setValue(0)
                form_layout.addWidget(k_spin, row, j)
                k_row.append(k_spin)
            if j < 2:
                form_layout.addWidget(QLabel(""), row, 3)  # Empty spacer
            self.k_matrix.append(k_row)
            row += 1
        
        # === D distortion (1x4) - Distortion Coefficients ===
        d_label = QLabel("<b>D Distortion Coefficients (4 values)</b>")
        form_layout.addWidget(d_label, row, 0, 1, 4)
        row += 1
        
        d_header = QLabel("<i>k1, k2, p1, p2</i>")
        d_header.setStyleSheet("color: #888;")
        form_layout.addWidget(d_header, row, 0, 1, 4)
        row += 1
        
        self.d_distortion = []
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                d_spin = QDoubleSpinBox()
                d_spin.setRange(-1, 1)
                d_spin.setDecimals(10)
                d_spin.setValue(0)
                d_spin.setSingleStep(0.0001)
                d_spin.setMinimumWidth(120)
                form_layout.addWidget(QLabel(f"D[{idx}]:"), row, j * 2)
                form_layout.addWidget(d_spin, row, j * 2 + 1)
                self.d_distortion.append(d_spin)
            row += 1
        
        form_layout.setRowStretch(row, 1)
        scroll_content.setLayout(form_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.setLayout(layout)
    
    def get_config(self):
        """Get configuration as dictionary"""
        k_matrix = [[self.k_matrix[i][j].value() for j in range(3)] for i in range(3)]
        d_distortion = [[self.d_distortion[i].value() for i in range(4)]]
        
        return {
            "radius": self.radius.value(),
            "center_x": self.center_x.value(),
            "center_y": self.center_y.value(),
            "width": self.width.value(),
            "height": self.height.value(),
            "uk": self.uk.value(),
            "d1": self.d1.value(),
            "d2": self.d2.value(),
            "d3": self.d3.value(),
            "K": k_matrix,
            "D": d_distortion
        }
    
    def set_config(self, config):
        """Set configuration from dictionary"""
        self.radius.setValue(config.get("radius", 1900))
        self.center_x.setValue(config.get("center_x", 1500))
        self.center_y.setValue(config.get("center_y", 2000))
        self.width.setValue(config.get("width", 3000))
        self.height.setValue(config.get("height", 4000))
        self.uk.setValue(config.get("uk", 25))
        self.d1.setValue(config.get("d1", 0))
        self.d2.setValue(config.get("d2", 0))
        self.d3.setValue(config.get("d3", 0))
        
        # K matrix
        k_matrix = config.get("K", [[1800, 0, 2000], [0, 1800, 1500], [0, 0, 1]])
        for i in range(3):
            for j in range(3):
                self.k_matrix[i][j].setValue(k_matrix[i][j])
        
        # D distortion
        d_distortion = config.get("D", [[0, 0, 0, 0]])[0]
        for i in range(4):
            self.d_distortion[i].setValue(d_distortion[i])


class SettingsView(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_forms = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Top controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("JSON File:"), 0)
        self.file_input = QLineEdit("../stich_old/kandao.json")
        controls_layout.addWidget(self.file_input, 1)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        controls_layout.addWidget(self.browse_btn)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_json)
        controls_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_json)
        controls_layout.addWidget(self.save_btn)
        
        layout.addLayout(controls_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Number of cameras selector
        num_cameras_layout = QHBoxLayout()
        num_cameras_layout.addWidget(QLabel("Number of Cameras:"))
        self.num_cameras_combo = QComboBox()
        for i in range(1, 13):
            self.num_cameras_combo.addItem(str(i), i)
        self.num_cameras_combo.setCurrentText("6")
        self.num_cameras_combo.currentIndexChanged.connect(self.update_camera_forms)
        num_cameras_layout.addWidget(self.num_cameras_combo)
        num_cameras_layout.addStretch()
        layout.addLayout(num_cameras_layout)
        
        # Tab widget for cameras
        self.camera_tabs = QTabWidget()
        self.camera_tabs.setTabPosition(QTabWidget.North)
        layout.addWidget(self.camera_tabs, 1)
        
        self.setLayout(layout)
        
        # Initialize with 6 cameras
        self.update_camera_forms()
        
        # Auto-load default file if exists
        self.load_json()
    
    def update_camera_forms(self):
        """Update the number of camera forms displayed"""
        # Clear existing tabs
        self.camera_tabs.clear()
        self.camera_forms.clear()
        
        # Create new tabs
        num_cameras = self.num_cameras_combo.currentData()
        for i in range(num_cameras):
            form = CameraConfigForm(i)
            self.camera_forms.append(form)
            self.camera_tabs.addTab(form, f"Camera {i + 1}")
    
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Select JSON File", 
            "", 
            "JSON Files (*.json)"
        )
        if filename:
            self.file_input.setText(filename)
            self.load_json()
    
    def load_json(self):
        try:
            filepath = self.file_input.text()
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check if data is a list (array of cameras)
            if isinstance(data, list):
                cameras_data = data
            else:
                cameras_data = [data]
            
            # Update number of cameras
            num_cameras = len(cameras_data)
            index = self.num_cameras_combo.findData(num_cameras)
            if index >= 0:
                self.num_cameras_combo.setCurrentIndex(index)
            else:
                # If not in combo, create forms manually
                self.update_camera_forms()
            
            # Load data into forms
            for i, camera_config in enumerate(cameras_data):
                if i < len(self.camera_forms):
                    self.camera_forms[i].set_config(camera_config)
            
            self.status_label.setText(f"Loaded: {filepath} ({num_cameras} cameras)")
            
        except FileNotFoundError:
            self.status_label.setText("File not found - using defaults")
        except json.JSONDecodeError as e:
            self.status_label.setText(f"JSON Error: {e}")
            QMessageBox.critical(self, "Error", f"Invalid JSON:\n{e}")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Error loading file:\n{e}")
    
    def save_json(self):
        try:
            filepath = self.file_input.text()
            
            # Collect all camera configurations
            cameras_data = []
            for form in self.camera_forms:
                cameras_data.append(form.get_config())
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(cameras_data, f, indent=4)
            
            self.status_label.setText(f"Saved: {filepath} ({len(cameras_data)} cameras)")
            QMessageBox.information(self, "Success", 
                                   f"Configuration saved successfully!\n{len(cameras_data)} cameras")
            
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Error saving file:\n{e}")
