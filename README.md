# CamProcessor

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

基于类激活映射（CAM）和灰度共生矩阵（GLCM）的图像处理工具，可生成目标掩码、可视化叠加和透明背景图像。

## 功能特性

- 🖼️ 单张图像处理管道
- 🔍 结合深度特征和纹理特征
- 🎯 生成三种输出结果：
  - 二进制掩码（0/1矩阵）
  - 可视化叠加图像
  - 透明背景图像
- 💻 纯CPU实现，无需GPU

## 安装

### 依赖环境
- Python 3.7+
- OpenCV 4.5+
- PyTorch 1.8+
- fast-glcm

### 快速安装
```bash
pip install pdm
pdm install --no-editable --verbose
pdm run python test.py
```

## 快速开始

### 基础使用

```
from cam_processor import CamProcessor

# 初始化处理器
processor = CamProcessor(model_path="./weights/camnet.pth")

# 处理图像
mask, overlay, transparent = processor.process_image("./input.jpg")

# 保存结果
cv2.imwrite("mask.png", mask * 255)          # 二值掩码
cv2.imwrite("overlay.png", overlay)          # 可视化叠加
cv2.imwrite("transparent.png", transparent)  # 透明背景
```

### 参数说明

#### `process_image()` 返回值：

| 参数        | 类型            | 说明                  |
| :---------- | :-------------- | :-------------------- |
| mask        | ndarray (H,W)   | 0/1二进制掩码矩阵     |
| overlay     | ndarray (H,W,3) | BGR格式可视化叠加图像 |
| transparent | ndarray (H,W,4) | BGRA格式透明背景图像  |

## 高级配置

### 自定义处理尺寸

```
processor = CamProcessor(model_path="your_model.pth")
processor.image_size = (1024, 1024)  # 修改处理分辨率
```

### 批处理示例

```
import glob

processor = CamProcessor("./model.pth")

for img_path in glob.glob("./images/*.jpg"):
    mask, overlay, transparent = processor.process_image(img_path)
    # 保存逻辑...
```

## 常见问题

### Q1: 如何处理输出图像的色差问题？

- 确保使用OpenCV默认的BGR格式处理
- 保存透明图像时不要进行通道转换
- 推荐使用PNG格式保存结果

### Q2: 处理速度较慢怎么办？

- 减小`image_size`参数值
- 使用更快的GLCM实现（如修改fast_glcm参数）
- 对输入图像进行预缩放

### Q3: 如何自定义特征融合权重？

修改`_fuse_features`方法：

```
def _fuse_features(self, cam, glcm):
    # 示例：调整权重比例
    return np.where(0.6*glcm + 0.4*cam > 0.55, 1, 0)
```

## Docker容器运行
构建镜像
```bash
sudo docker build -t dlserver .
```

首次运行DLserver容器实例
```bash
# 由于server.py中代码指定了服务器监听端口为8000
DOCKER_LPORT=8000
# 这里可以自己设置主机的监听端口
HOST_LPORT=4396
# 加上-d可以以后台模式运行docker
sudo docker run -p $HOST_LPORT:$DOCKER_LPORT -v ./output:/app --name instance_dlserver dlserver

```

在首次运行后再次运行实例
```bash
sudo docker start instance_dlserver
# 挂载卷
sudo docker start  instance_dlserver
```


删除实例
```bash
sudo docker rm instance_dlserver
```

删除镜像
```bash
sudo docker rmi dlserver
```


## 本地测试
DLserver容器对外暴露端口映射到Host的PORT1上。


## TODO
1. 目前server的监听端口和server对client的callback端口都是在代码中写死的，应该以CLI Args的方式给出。
2. 

## 贡献指南

欢迎提交Pull Request！建议遵循以下规范：

1. 使用Google风格代码注释
2. 添加对应的单元测试
3. 更新相关文档

## 许可证

[MIT License](https://license/)