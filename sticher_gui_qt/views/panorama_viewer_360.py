from PySide6.QtWidgets import QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QPainter, QTransform
import numpy as np
import cv2
import math


class PanoramaViewer360(QGraphicsView):
    """Interactive 360° panorama viewer with mouse drag navigation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Viewer state
        self.panorama_pixmap = None
        self.panorama_item = None
        self.panorama_img = None  # Original cv2 image
        
        # Navigation state
        self.yaw = 0.0  # Horizontal rotation (0 to 360)
        self.pitch = 0.0  # Vertical rotation (-90 to 90)
        self.fov = 90.0  # Field of view
        self.last_mouse_pos = None
        self.is_dragging = False
        
        # View dimensions
        self.view_width = 800
        self.view_height = 600
        
        # Setup
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("border: 2px solid #007acc; background-color: #000;")
        
    def set_panorama(self, img_array):
        """Set the equirectangular panorama image"""
        if img_array is None:
            return
            
        # Store original image
        self.panorama_img = img_array.copy()
        
        # Render initial view
        self.render_view()
    
    def render_view(self):
        """Render the current perspective view from the equirectangular panorama"""
        if self.panorama_img is None:
            return
        
        # Get viewport dimensions
        viewport_rect = self.viewport().rect()
        self.view_width = max(viewport_rect.width(), 400)
        self.view_height = max(viewport_rect.height(), 300)
        
        # Generate perspective view from equirectangular
        view_img = self.equirectangular_to_perspective(
            self.panorama_img,
            self.yaw,
            self.pitch,
            self.fov,
            self.view_width,
            self.view_height
        )
        
        # Convert to QPixmap
        img_rgb = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.panorama_pixmap = QPixmap.fromImage(qt_image)
        
        # Update scene
        self.scene.clear()
        self.panorama_item = self.scene.addPixmap(self.panorama_pixmap)
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.panorama_item, Qt.KeepAspectRatio)
    
    def equirectangular_to_perspective(self, img, yaw, pitch, fov, width, height):
        """Convert equirectangular panorama to perspective view"""
        
        # Convert angles to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        fov_rad = math.radians(fov)
        
        # Calculate focal length
        f = width / (2 * math.tan(fov_rad / 2))
        
        # Create output image
        out_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        pano_height, pano_width = img.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Convert pixel coordinates to normalized coordinates
        x_norm = (x_coords - width / 2) / f
        y_norm = (y_coords - height / 2) / f
        
        # Calculate ray directions
        # Forward direction (camera looking direction)
        z = np.ones_like(x_norm)
        
        # Apply pitch rotation
        x_pitch = x_norm
        y_pitch = y_norm * math.cos(pitch_rad) + z * math.sin(pitch_rad)
        z_pitch = -y_norm * math.sin(pitch_rad) + z * math.cos(pitch_rad)
        
        # Apply yaw rotation
        x_yaw = x_pitch * math.cos(yaw_rad) - z_pitch * math.sin(yaw_rad)
        y_yaw = y_pitch
        z_yaw = x_pitch * math.sin(yaw_rad) + z_pitch * math.cos(yaw_rad)
        
        # Convert to spherical coordinates
        theta = np.arctan2(x_yaw, z_yaw)  # Longitude (-pi to pi)
        r = np.sqrt(x_yaw**2 + y_yaw**2 + z_yaw**2)
        phi = np.arcsin(np.clip(y_yaw / r, -1, 1))  # Latitude (-pi/2 to pi/2)
        
        # Convert to equirectangular pixel coordinates
        u = ((theta / (2 * math.pi) + 0.5) * pano_width).astype(np.float32)
        v = ((0.5 - phi / math.pi) * pano_height).astype(np.float32)
        
        # Sample from panorama
        u = np.clip(u, 0, pano_width - 1)
        v = np.clip(v, 0, pano_height - 1)
        
        # Use remap for efficient sampling
        out_img = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        return out_img
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        if self.is_dragging and self.last_mouse_pos is not None:
            # Calculate delta
            delta = event.pos() - self.last_mouse_pos
            
            # Update rotation (sensitivity factor)
            sensitivity = 0.2
            self.yaw -= delta.x() * sensitivity
            self.pitch += delta.y() * sensitivity
            
            # Clamp pitch
            self.pitch = max(-89, min(89, self.pitch))
            
            # Normalize yaw
            self.yaw = self.yaw % 360
            
            # Re-render
            self.render_view()
            
            self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        # Adjust FOV
        delta = event.angleDelta().y()
        if delta > 0:
            self.fov = max(30, self.fov - 5)  # Zoom in
        else:
            self.fov = min(120, self.fov + 5)  # Zoom out
        
        self.render_view()
    
    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.panorama_img is not None:
            self.render_view()
