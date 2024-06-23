# 模型训练与剪枝项目

## 项目概述
本项目旨在通过对卷积神经网络模型进行训练和剪枝，评估其在性能和效率方面的表现。具体步骤包括初次训练、模型剪枝以及剪枝后的再训练。

## 文件结构
- `main.py`：主程序文件，包含训练和剪枝的主要流程。
- `model.py`：定义了模型的结构。
- `prune.py`：实现了模型剪枝的功能。
- `utils.py`：包含训练和测试的辅助函数。
- `data.py`：定义了数据加载和预处理的功能。
- `README.md`：项目说明文档。

## 环境依赖
- Python 3.11
- PyTorch 1.8.1
- CUDA 10.1

## 数据集
使用 COCO 2017 数据集，包括 `train2017` 和 `val2017` 数据集。

## 数据预处理
数据预处理包括以下步骤：
- 调整尺寸为 64x64
- 随机水平翻转
- 随机旋转 10 度
- 归一化（均值和标准差均为 0.5）

## 模型结构
模型为一个简单的卷积神经网络，包含以下层：
1. 卷积层1：输入通道数为3，输出通道数为32，卷积核大小为3x3
2. 池化层：最大池化，池化窗口大小为2x2
3. 卷积层2：输入通道数为32，输出通道数为64，卷积核大小为3x3
4. 池化层：最大池化，池化窗口大小为2x2
5. 卷积层3：输入通道数为64，输出通道数为128，卷积核大小为3x3
6. 池化层：最大池化，池化窗口大小为2x2
7. 全连接层1：输入节点数为128*8*8，输出节点数为512
8. Dropout层：丢弃率为0.25
9. 全连接层2：输入节点数为512，输出节点数为10

## 使用方法
### 安装依赖
1. 克隆项目到本地：
    ```bash
    git clone https://github.com/ma-jch/DL
    cd project
    ```
2. 安装所需的 Python 包：
    ```bash
    pip install -r requirements.txt
    ```

### 下载数据集
1. 下载 COCO 2017 数据集，并解压到 `./coco_dataset` 目录下。
    - [COCO 2017 Train images](http://images.cocodataset.org/zips/train2017.zip)
    - [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
    - [COCO 2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

### 运行主程序
1. 运行主程序进行模型训练和剪枝：
    ```bash
    python main.py
    ```

## 项目说明
- `main.py`：该文件包含了模型的训练和剪枝流程。在运行该文件时，模型将进行初次训练，之后对模型进行剪枝，并对剪枝后的模型进行再训练以评估其性能。
- `model.py`：定义了一个简单的卷积神经网络模型。
- `prune.py`：实现了模型剪枝的功能，通过L1剪枝方法对模型进行剪枝。
- `utils.py`：包含训练和测试的辅助函数，用于模型训练和评估。
- `data.py`：定义了数据加载和预处理的功能，使用 COCO 2017 数据集进行训练和测试。

## 注意事项
- 请确保使用的硬件支持 CUDA 以加速训练过程。
- 如果内存不足，可以适当调整 `batch_size` 和 `num_workers` 参数。

## 联系方式
如果有任何问题或建议，请联系 [XXXXXXXXXXX]。

