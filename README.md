# YOLOV8 + TensorRT 模型构建以及部署
基于 YOLOv8 的高性能目标检测模型，通过 TensorRT 实现模型转换、加速与高效部署，支持 GPU 环境下的实时推理与优化。

### 文件结构
- train_export：使用python进行yolov8的模型训练和onnx、tensorRT模型导出，请使用该文件夹下的train_export.py中的get_tensorRT函数将onnx文件转换为适配gpu的TensorRT模型。
- tensorrt_pred_cpp：使用tensorRT模型进行推理，代码请参考main.cpp，使用cmake进行构建。

### 环境配置
- python：
  - 主要用于训练和导出，需要pytorch、ultralytics和tensorrt，安装cuda版pytorch时会自动安装cuda，因此不用特意下载cuda。
  - 若tensorRT导出模型失败，可以尝试降低tensorRT版本。
  - 请使用pip install安装。
- c++：
  - 主要用于推理，需要opencv、tensorrt、cuda、cudnn，请选择适配gpu的cuda和tensorRT版本，尽量最新，安装好后请在tensorrt_pred_cpp/CMakeLists.txt中修改相应路径。
  - **经修改，目前代码仅支持TensorRT>10的版本。**

### 模型训练和导出
1. 下载yolov8预训练模型
2. 设置数据集格式，dataset文件夹下存在images和labels文件夹，这两个文件夹下应分为训练集、验证集和测试集。训练集用于训练模型，验证集用于在训练中检验模型性能，测试集用于训练后测试模型性能。请在train_export/armor.yaml中修改相应内容。
  ```
  path: ./dataset
  train: images/train
  val: images/val
  test: images/test
```
3. 在train_export.py文件中修改相应代码后运行：`python train_export.py`

### 模型推理
windows:
1. 配置环境和CMakeLists.txt
2. 修改main.cpp文件中模型或其他文件路径地址。
3. 编译
  ```
  mkdir build
  cd build
  cmake ..
  cmake --build . --config Release
  ```
4. 将模型和其他文件(如图片或视频)复制到编译出来的文件夹中，文件夹应包含如下文件：
  ```
  yolov8.exe
  model.trt
  0001.jpg
  0002.mp4
  ```
5. 运行exe文件。

### 附录
TensorRT推理代码参考: https://github.com/triple-Mu/YOLOv8-TensorRT/tree/main

yolov8官方文档: https://docs.ultralytics.com/zh/models/yolov8/

cuda、cudnn安装参考: https://blog.csdn.net/m0_62907835/article/details/145441697