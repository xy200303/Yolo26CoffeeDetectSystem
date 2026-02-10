import sys
import cv2
import numpy as np
import os
import time

# Fix for local ultralytics folder shadowing installed package

from ultralytics import YOLO

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QFileDialog, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor

# Import Fluent Widgets
from qfluentwidgets import (FluentWindow, NavigationItemPosition, NavigationInterface,
                            SubtitleLabel, PushButton, PrimaryPushButton, Slider,
                            ComboBox, CardWidget, ImageLabel, InfoBar, InfoBarPosition,
                            setTheme, Theme, FluentIcon as FIF)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        
        self.model = None
        # Load YOLO model
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"Model loaded: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found: {model_path}")
            
        # Define colors (will be populated from model names if available)
        self.colors = []

    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold

    def detect(self, image):
        if self.model is None:
            return image

        # Run inference
        try:
            results = self.model(image, conf=self.conf_threshold)
            
            # Plot results on the image
            # results[0].plot() returns the image with boxes drawn
            annotated_frame = results[0].plot()
            return annotated_frame
            
        except Exception as e:
            print(f"Inference error: {e}")
            return image


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detector, camera_index=0):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                frame = self.detector.detect(frame)
                self.change_pixmap_signal.emit(frame)
            time.sleep(0.01)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class ImageInterface(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent=parent)
        self.detector = detector
        self.current_image = None
        self.setObjectName("ImageInterface")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = SubtitleLabel("图像目标检测", self)
        layout.addWidget(header)
        
        # Controls Card
        control_card = CardWidget(self)
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(20, 10, 20, 10)
        
        self.btn_load = PrimaryPushButton("打开图像", self)
        self.btn_load.setIcon(FIF.FOLDER)
        self.btn_load.clicked.connect(self.select_image)
        
        self.slider_conf = Slider(Qt.Orientation.Horizontal, self)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(25)
        self.slider_conf.setFixedWidth(200)
        self.slider_conf.valueChanged.connect(self.update_threshold)
        
        self.label_conf = QLabel("置信度: 0.25", self)
        
        control_layout.addWidget(self.btn_load)
        control_layout.addStretch(1)
        control_layout.addWidget(QLabel("阈值调节:", self))
        control_layout.addWidget(self.slider_conf)
        control_layout.addWidget(self.label_conf)
        
        layout.addWidget(control_card)
        
        # Image Display
        self.image_label = QLabel("请选择一张图像", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #d0d0d0;
                border-radius: 10px;
                color: #606060;
                font-size: 16px;
                background-color: #f9f9f9;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.image_label, 1)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image.copy()
                result_image = self.detector.detect(image)
                self.display_image(result_image)
                
    def update_threshold(self, value):
        threshold = value / 100.0
        self.label_conf.setText(f"置信度: {threshold:.2f}")
        self.detector.set_conf_threshold(threshold)
        
        if self.current_image is not None:
            result_image = self.detector.detect(self.current_image.copy())
            self.display_image(result_image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        
        target_size = self.image_label.size()
        if target_size.width() < 10 or target_size.height() < 10:
             target_size = pixmap.size()
             
        scaled_pixmap = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


class VideoInterface(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent=parent)
        self.detector = detector
        self.video_thread = None
        self.setObjectName("VideoInterface")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = SubtitleLabel("实时视频检测", self)
        layout.addWidget(header)
        
        # Controls Card
        control_card = CardWidget(self)
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(20, 10, 20, 10)
        
        self.combo_camera = ComboBox(self)
        self.combo_camera.setFixedWidth(150)
        
        self.btn_refresh = PushButton("刷新设备", self)
        self.btn_refresh.setIcon(FIF.SYNC)
        self.btn_refresh.clicked.connect(self.detect_cameras)
        
        self.btn_start = PrimaryPushButton("开始检测", self)
        self.btn_start.setIcon(FIF.PLAY)
        self.btn_start.clicked.connect(self.start_video)
        
        self.btn_stop = PushButton("停止检测", self)
        self.btn_stop.setIcon(FIF.PAUSE)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        
        self.slider_conf = Slider(Qt.Orientation.Horizontal, self)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(25)
        self.slider_conf.setFixedWidth(150)
        self.slider_conf.valueChanged.connect(self.update_threshold)
        
        self.label_conf = QLabel("置信度: 0.25", self)
        
        control_layout.addWidget(QLabel("选择摄像头:", self))
        control_layout.addWidget(self.combo_camera)
        control_layout.addWidget(self.btn_refresh)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)
        control_layout.addWidget(self.slider_conf)
        control_layout.addWidget(self.label_conf)
        
        layout.addWidget(control_card)
        
        # Video Display
        self.video_label = QLabel("摄像头未启动", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border-radius: 10px;
                color: #808080;
                font-size: 16px;
            }
        """)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.video_label, 1)
        
        # Init
        self.detect_cameras()

    def detect_cameras(self):
        self.combo_camera.clear()
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(f"摄像头 {i}")
                cap.release()
        
        if not available_cameras:
            self.combo_camera.addItem("未找到摄像头")
            self.btn_start.setEnabled(False)
        else:
            self.combo_camera.addItems(available_cameras)
            self.btn_start.setEnabled(True)

    def update_threshold(self, value):
        threshold = value / 100.0
        self.label_conf.setText(f"置信度: {threshold:.2f}")
        self.detector.set_conf_threshold(threshold)

    def start_video(self):
        camera_idx = self.combo_camera.currentIndex()
        if camera_idx < 0: return
        
        self.video_thread = VideoThread(self.detector, camera_idx)
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_camera.setEnabled(False)
        self.btn_refresh.setEnabled(False)

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_camera.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.video_label.setText("摄像头已停止")

    def update_video_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        
        target_size = self.video_label.size()
        if target_size.width() < 10: return
            
        scaled_pixmap = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def closeEvent(self, event):
        self.stop_video()
        event.accept()


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv26 咖啡豆SACC烘培程度检测系统")
        self.resize(1000, 700)
        
        # Initialize detector
        self.detector = YOLODetector(MODEL_PATH, conf_threshold=0.25)
        
        # Create sub-interfaces
        self.image_interface = ImageInterface(self.detector, self)
        self.video_interface = VideoInterface(self.detector, self)
        
        # Add navigation items
        self.addSubInterface(self.image_interface, FIF.PHOTO, "图像检测")
        self.addSubInterface(self.video_interface, FIF.VIDEO, "实时视频")
        
        # Set theme (Auto, Light, Dark)
        setTheme(Theme.LIGHT) # Or Theme.DARK

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
