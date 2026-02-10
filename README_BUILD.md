# YOLOv26 咖啡豆SACC烘培程度检测系统

## 1. 环境准备

请确保已安装 Python 3.8+。

安装依赖：
```bash
pip install -r requirements.txt
pip install pyinstaller
```

## 2. 运行程序

在开发模式下运行：
```bash
python main.py
```

## 3. 打包 EXE

使用 PyInstaller 进行打包：

```bash
pyinstaller build.spec
```

打包完成后，可执行文件位于 `dist/CoffeeDetectSystem/CoffeeDetectSystem.exe`。

## 注意事项

- 打包配置已在 `build.spec` 中定义，会自动包含 `best.onnx` 和 `classes.txt`。
- 如果遇到 `numpy` 版本问题，请确保安装的是兼容 `onnxruntime` 的版本（通常 `<2.0`）。
