# YOLOv26 咖啡豆SACC烘培程度检测系统

## 项目简介

**YOLOv26 咖啡豆SACC烘培程度检测系统** 是一款基于深度学习的智能检测软件，旨在通过计算机视觉技术自动识别咖啡豆的烘培程度（如极浅烘焙、中度烘焙、深度烘焙等）。系统采用 YOLOv26 模型（导出为 ONNX 格式）作为核心推理引擎，结合 PyQt6 和 PyQt-Fluent-Widgets 构建了现代化、美观且易用的图形用户界面。

## 主要功能

*   **多模式检测**：
    *   **图像检测**：支持导入本地图片（JPG, PNG, BMP等）进行静态分析，并实时显示检测框和置信度。
    *   **实时视频检测**：支持连接 USB 摄像头进行实时流检测，适用于生产线或实验室场景。
*   **现代化 UI 设计**：
    *   采用 **PyQt-Fluent-Widgets** 组件库，提供类似 Windows 11 / Flutter 的现代化视觉体验。
    *   支持侧边栏导航、卡片式布局和流畅的动画效果。
    *   全中文界面，操作直观。
*   **实时参数调节**：
    *   提供置信度阈值（Confidence Threshold）滑块，用户可根据环境光线和识别需求实时调整过滤灵敏度。
*   **智能设备管理**：
    *   自动扫描并列出可用的摄像头设备。
    *   支持热插拔刷新设备列表。

## 环境要求

*   **操作系统**：Windows 10/11 (推荐), Linux, macOS
*   **Python 版本**：Python 3.8 - 3.11
*   **核心依赖库**：
    *   `numpy`
    *   `opencv-python`
    *   `onnxruntime` (或 `onnxruntime-gpu`)
    *   `PyQt6`
    *   `PyQt6-Fluent-Widgets`

## 安装说明

1.  **克隆或下载项目**：
    确保您已获取项目源码，并进入项目根目录。

2.  **安装依赖**：
    建议使用虚拟环境（venv 或 conda）。
    ```bash
    cd app
    pip install -r requirements.txt
    ```

    *注意：如果遇到 `numpy` 版本冲突，请确保安装兼容 `onnxruntime` 的版本（推荐 `<2.0`）。*

## 使用指南

### 启动软件

在 `app` 目录下运行主程序：

```bash
python main.py
```

### 功能操作

1.  **图像检测**：
    *   点击侧边栏的“图像检测”。
    *   点击“打开图像”按钮选择咖啡豆图片。
    *   调整右下角的“阈值调节”滑块以过滤误检框。
2.  **实时视频**：
    *   点击侧边栏的“实时视频”。
    *   在“选择摄像头”下拉框中选择设备（如未找到，请检查连接并点击“刷新设备”）。
    *   点击“开始检测”启动实时流。
    *   点击“停止检测”结束会话。

## 软件打包 (EXE)

本项目支持使用 `PyInstaller` 打包为独立的可执行文件。

1.  **安装 PyInstaller**：
    ```bash
    pip install pyinstaller
    ```

2.  **执行打包**：
    ```bash
    cd app
    pyinstaller build.spec
    ```

3.  **获取文件**：
    打包完成后，可执行文件将生成在 `app/dist/CoffeeDetectSystem/CoffeeDetectSystem.exe`。

## 技术栈

*   **模型训练**：YOLOv8 (Ultralytics)
*   **模型推理**：ONNX Runtime (CPU/GPU)
*   **图形界面**：PyQt6 + PyQt-Fluent-Widgets
*   **图像处理**：OpenCV (cv2)

## 目录结构

```
caffe_detect/
├── app/
│   ├── main.py              # 主程序入口
│   ├── best.onnx            # 训练好的模型文件
│   ├── classes.txt          # 类别标签文件
│   ├── requirements.txt     # 依赖列表
│   ├── build.spec           # PyInstaller 打包配置
│   └── README_BUILD.md      # 打包说明文档
├── dataset/                 # 数据集 (开发用)
└── README.md                # 项目说明文档
```

---
**注意**：本系统依赖于预训练的 `best.onnx` 模型文件，请确保该文件位于 `app` 目录下且与代码版本匹配。
