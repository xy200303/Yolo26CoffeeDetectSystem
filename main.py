import sys
import cv2
import numpy as np
import onnxruntime as ort
import os
import time

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
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.onnx')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'classes.txt')

class YOLODetector:
    def __init__(self, model_path, classes_path, conf_threshold=0.25):
        self.classes = self.load_classes(classes_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.conf_threshold = conf_threshold
        
        self.session = None
        # Load ONNX model using onnxruntime
        if os.path.exists(model_path):
            try:
                # Use CPU provider
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape # [batch, channels, height, width]
                
                print(f"Model loaded: {model_path}")
                print(f"Input shape: {self.input_shape}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found: {model_path}")

    def load_classes(self, path):
        classes = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if '→' in line:
                        classes.append(line.split('→')[1])
                    else:
                        classes.append(line)
        else:
            print(f"Classes file not found: {path}")
        return classes

    def preprocess(self, image):
        # Save original dimensions
        original_height, original_width = image.shape[:2]
        
        # Get model input size
        # Assuming input_shape is [1, 3, 640, 640]
        input_height = self.input_shape[2]
        input_width = self.input_shape[3]
        
        # Letterbox resize
        scale = min(input_width / original_width, input_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas and pad
        # Use 114 as padding value (YOLO standard)
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        
        # Center the image
        dx = (input_width - new_width) // 2
        dy = (input_height - new_height) // 2
        canvas[dy:dy+new_height, dx:dx+new_width] = resized
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # HWC -> CHW, Normalize to [0,1]
        img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input, original_width, original_height, scale, dx, dy

    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold

    def detect(self, image):
        if self.session is None:
            return image

        # Preprocess
        input_tensor, orig_w, orig_h, scale, dx, dy = self.preprocess(image)
        
        try:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        except Exception as e:
            print(f"Inference error: {e}")
            return image

        # Post-process
        # Output shape: (1, 300, 6) or similar
        detections = outputs[0][0] # First batch
        
        detected_count = 0
        
        for det in detections:
            x1, y1, x2, y2, score, class_id = det[:6]
            
            if score < self.conf_threshold:
                continue
                
            detected_count += 1
            
            # Restore coordinates
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                continue
                
            self.draw_box(image, x1, y1, w, h, score, int(class_id))
            
        return image

    def draw_box(self, image, x, y, w, h, score, class_id):
        if class_id < 0 or class_id >= len(self.classes):
            class_name = f"Class {class_id}"
        else:
            class_name = self.classes[class_id]

        color = self.colors[class_id % len(self.colors)] if len(self.colors) > 0 else (0, 255, 0)
        label = f"{class_name} {score:.2f}"
        
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = x + t_size[0], y - t_size[1] - 3
        cv2.rectangle(image, (x, y), c2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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
        self.detector = YOLODetector(MODEL_PATH, CLASSES_PATH, conf_threshold=0.25)
        
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
