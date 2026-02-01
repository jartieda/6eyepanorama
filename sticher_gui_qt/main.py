import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QScreen
from views.stitcher_view import StitcherView
from views.calibration_view import CalibrationView
from views.settings_view import SettingsView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panorama Stitcher - Professional Edition")
        
        # Get screen geometry for better initial sizing
        screen = QApplication.primaryScreen().geometry()
        width = min(1600, int(screen.width() * 0.9))
        height = min(1000, int(screen.height() * 0.9))
        self.resize(width, height)
        
        # Center window on screen
        self.move(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2
        )
        
        # Make window fully resizable - NO restrictions
        self.setMinimumSize(800, 600)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)
        
        # Add tabs
        self.tabs.addTab(StitcherView(), "📷 Stitcher")
        self.tabs.addTab(CalibrationView(), "🎯 Calibration")
        self.tabs.addTab(SettingsView(), "⚙️  Settings")
        
        # Apply dark theme with better contrast
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #3d3d3d;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
                color: #ffffff;
                border-bottom: 2px solid #007acc;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QLineEdit, QTextEdit, QSpinBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                padding: 5px;
                selection-background-color: #007acc;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
                border: 1px solid #007acc;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #808080;
            }
            QGroupBox {
                color: #e0e0e0;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 15px;
                font-weight: bold;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #1e1e1e;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border: 1px solid #007acc;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d3d3d;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4d4d4d;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Panorama Stitcher")
    app.setOrganizationName("AAD DataScience")
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
