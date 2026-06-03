[English](README.md) | [中文](README.zh-CN.md)

# BTC-Net V2：面向高光谱图像压缩的空间优先级分层重建
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11371342)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=Python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=PyTorch&logoColor=white)](https://pytorch.org/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=MWan-deeplearner.BTC-Net-V2&left_text=Total%20Views&color=00629B)

## 作者
* **周喜川**<sup>1</sup>（Xichuan Zhou）
* **万明扬**<sup>1,4</sup>（Mingyang Wan，主要贡献者与维护者）
* **于杭**<sup>1</sup>（Hang Yu）、**陈涛**<sup>1</sup>（Tao Chen）、**何如龙**<sup>2</sup>（Rulong He）、**沈象飞**<sup>2</sup>（Xiangfei Shen）、**陈黎辉**<sup>1</sup>（Lihui Chen）、**刘海军**<sup>1,3</sup>（Haijun Liu）

<sup>1</sup> 重庆大学，中国  
<sup>2</sup> 海军工程大学，中国  
<sup>3</sup> 通讯作者：[haijun_liu@cqu.edu.cn]  
<sup>4</sup> 主要联系：[13477304346@163.com]

## 更新
* **[2026/02/05]** 发布 BTC-Net V2 基本信息。
* **[2026/06/02]** 发布模型代码、预训练检查点（8-bit 和 32-bit）以及评估脚本。
* **[2026/06/03]** 发布训练脚本及完整复现流程，包含率失真优化、混合精度训练和进度跟踪功能。

## 摘要
*近年来，面向星载应用的高光谱图像（HSI）有损压缩受到了广泛关注。星载设备有限的存储容量和受限的传输带宽，使得在传输数据的压缩率与本地重建图像的质量之间取得平衡变得极具挑战。为解决这一难题，我们提出了 BTC-Net V2——BTC-Net 的进阶版本——大幅提升了前代方法的性能。具体而言，在编码器阶段，我们采用大核卷积替代前代版本中使用的三层卷积网络，从而实现了更轻量化的编码器设计，以契合星载 HSI 压缩的资源约束。此外，在解码器阶段，我们将前代版本中"先特征增强后上采样"的范式替换为以空间维度上采样为优先的空间优先级分层架构。这一设计确保了后续的特征增强骨干网络在更大的空间尺度上运行，直接缓解了 BTC-Net 中空间相关性的损失，最终实现了更高的重建质量。大量的实验结果表明，我们提出的 BTC-Net V2 在 PSNR 指标上比 BTC-Net 提升了 3.35 dB——达到 40.78 dB，而 BTC-Net 为 37.43 dB——并且以更低的比特率运行（0.054 bpppb，而 BTC-Net 为 0.060 bpppb）。我们的代码可在 https://github.com/MWan-deeplearner/BTC-Net-V2 获取。*

## 模型架构

BTC-Net V2 遵循编码器-解码器范式，采用可学习的量化和熵编码。

### 编码器
单个**量化卷积层**（`QuantConv2d`）替代了 BTC-Net V1 中的三层网络：
- 大核卷积：11×4，步长为 4，一次性实现激进的空间下采样
- 通过均匀量化与直通估计器（STE），支持可配置的量化位宽（8-bit / 32-bit）
- 带可学习斜率的 PReLU 激活函数

### 解码器：空间优先级分层重建（SPHR）
解码器遵循**空间优先、光谱后置**的重建顺序：

1. **SpatialRestorer**（空间恢复器）：基于 PixelShuffle 的上采样，将压缩后的潜在表示恢复到原始空间分辨率（H×W）
2. **DetailRestorer**（细节恢复器）：具有多尺度跳跃连接的 U-Net 风格骨干网络，用于精炼空间细节：
   - 多尺度处理（1× → 1/2× → 1/4× → 1/2× → 1×）
   - 每个尺度使用 **DRAM**（细节精炼注意力模块）模块
3. **SpectralRestorer**（光谱恢复器）：最终的 1×1 卷积，将特征投影回原始光谱维度，并通过残差跳跃连接与 SpatialRestorer 的输出相加

### DRAM：细节精炼注意力模块
每个 DRAM 模块由两个子模块按顺序组成：
- **SESAM**（空间增强自注意力模块）：通道级自注意力机制，建模长距离空间依赖关系，带可学习温度参数 α
- **SEFFM**（光谱增强前馈模块）：门控前馈网络，将光谱特征扩展 γ 倍（默认为 4），增强光谱表示能力

## 项目结构
```
.
├── README.md
├── README.zh-CN.md                 # 中文文档
├── test.py                         # 带熵编码的评估脚本
├── train.py                        # 带率失真优化的训练脚本
├── model/
│   ├── __init__.py
│   ├── BTCNetV2.py                 # 主模型（编码器 + SPHR 解码器）
│   ├── SPHRDecoder.py              # 空间优先级分层重建解码器
│   ├── DRAM.py                     # DRAM 模块（SESAM + SEFFM）
│   └── QuantizedConv2d.py          # 带直通估计的量化卷积
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # HSI 数据集加载器（.mat 文件）
│   ├── huffman_coder.py            # Huffman 熵编码管线
│   ├── metrics.py                  # PSNR、SAM、RMSE 评估指标
│   └── bar.py                      # 训练进度展示工具
├── checkpoint/
│   ├── BTCNetV2_8bit.pth           # 8-bit 量化预训练权重（~48 MB）
│   └── BTCNetV2_32bit.pth          # 32-bit 浮点预训练权重（~48 MB）
└── data/
    └── AVIRIS/
        ├── train/                  # 训练数据（.mat 文件）
        └── test/                   # 示例测试数据（.mat 文件）
```

## 安装

### 环境要求
- Python 3.12
- PyTorch 2.1+
- einops
- scipy
- numpy
- Pillow

请在**终端**中运行以下命令来创建并激活环境：
```bash
# 创建新的 conda 环境并激活
conda create -n btcnetv2 python=3.12
conda activate btcnetv2
# 安装依赖
pip install torch>=2.1 einops scipy numpy Pillow
```

## 数据集
### 1. 基准数据集介绍
在本工作中，我们在多个广泛使用的高光谱**基准数据集**上评估了我们的方法，包括：

| 数据集 | 传感器 | 描述 | 链接 |
| :--- | :--- | :--- | :---: |
| **AVIRIS** | 机载 | 经典高光谱飞行数据 | [Link](https://aviris.jpl.nasa.gov/data/get_aviris_data.html) |
| **WHU-Hi** | 无人机载 | 武汉大学高分辨率数据集 | [Link](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) |
| **Hyperion** | EO-1 卫星 | 星载高光谱任务数据 | [Link](https://earthexplorer.usgs.gov/) |

### 2. 数据准备与组织
请下载数据集并将其组织到 `data/` 目录中。每个数据集应有自己的子目录，其中包含 `train` 和 `test` 文件夹，用于存放相应的样本。

#### 目录结构：
```
.
└── your_project_root/
    ├── data/
    │   ├── AVIRIS/
    │   │   ├── train/         # 将 AVIRIS 训练样本放入此处
    │   │   └── test/          # 将 AVIRIS 测试样本放入此处
    │   ├── WHU-Hi/
    │   │   ├── train/         # 将 WHU-Hi 训练样本放入此处
    │   │   └── test/          # 将 WHU-Hi 测试样本放入此处
    │   └── Hyperion/
    │       ├── train/         # 将 Hyperion 训练样本放入此处
    │       └── test/          # 将 Hyperion 测试样本放入此处
    ├── train.py               # 训练脚本
    ├── test.py                # 测试脚本
    └── ...
```

数据文件应为 `.mat` 格式。每个 `.mat` 文件应包含一个单独的变量，存储高光谱数据立方体。

## 预训练检查点

在 `checkpoint/` 目录中提供了两种预训练模型变体：

| 检查点 | 量化方式 | 大小 | 描述 |
| :--- | :--- | :--- | :--- |
| `BTCNetV2_8bit.pth` | 8-bit | ~48 MB | 量化权重与激活；更低的比特率 |
| `BTCNetV2_32bit.pth` | 32-bit | ~48 MB | 全精度权重与激活；更高质量 |

两个检查点均使用默认配置：172 个输入通道，27 个压缩通道，缩放因子 4，32 个基础特征，γ=4。

## 评估

使用 `test.py` 在 HSI 数据上评估预训练模型。脚本报告四项指标：**bpp**（每像素每波段比特数）、**PSNR**、**SAM**（光谱角度映射器）和 **RMSE**。

### 快速开始
```bash
python test.py
```

### 配置参数
`test.py` 中的关键超参数：

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `ORIGINAL_CHANNELS` | 172 | 输入光谱波段数 |
| `COMPRESSED_CHANNELS` | 27 | 压缩后的潜在通道数 |
| `QUANT_BIT` | 8 | 量化位宽（8 或 32） |
| `SCALE` | 4 | 空间下采样因子 |
| `NUM_FEATURES` | 32 | 解码器中的基础特征通道数 |
| `GAMMA` | 4 | SEFFM 中的光谱扩展因子 |
| `CHECKPOINT` | `checkpoint/BTCNetV2_8bit.pth` | 预训练权重路径 |
| `DATA_DIR` | `data/AVIRIS/test` | 测试数据目录 |
| `DEVICE` | `cuda:0` | 推理设备 |

要切换到 32-bit 模式，将 `QUANT_BIT` 改为 `32`，`CHECKPOINT` 改为 `checkpoint/BTCNetV2_32bit.pth`。

### 输出
脚本按以下格式输出每张图像的评估结果：
```
bpp, psnr, sam, rmse:
<BPP 值>
<PSNR 值（dB）>
<SAM 值>
<RMSE 值>
```

### 评估指标
- **bpp**（Bits Per Pixel Per Band）：压缩率——越低越好
- **PSNR**（Peak Signal-to-Noise Ratio）：重建保真度——越高越好
- **SAM**（Spectral Angle Mapper）：光谱失真度——越低越好
- **RMSE**（Root Mean Square Error）：逐波段误差——越低越好

## 训练

使用 `train.py` 从头训练 BTC-Net V2。该脚本实现了基于 L1 损失的率失真优化，支持混合精度训练，并按周期记录训练指标。

### 快速开始
```bash
python train.py
```

### 配置参数
`train.py` 中的关键超参数：

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `ORIGINAL_CHANNELS` | 172 | 输入光谱波段数 |
| `COMPRESSED_CHANNELS` | 27 | 压缩后的潜在通道数 |
| `QUANT_BIT` | 6 | 训练时的量化位宽 |
| `SCALE` | 4 | 空间下采样因子 |
| `NUM_FEATURES` | 32 | 解码器中的基础特征通道数 |
| `GAMMA` | 4 | SEFFM 中的光谱扩展因子 |
| `BATCH_SIZE` | 12 | 训练批大小 |
| `MAX_EPOCHS` | 5000 | 最大训练轮数 |
| `CROP_HEIGHT` | 128 | 训练时的随机裁剪高度 |
| `WIDTH` | 4 | 滑动窗口的宽度 |
| `MARGINAL` | 60 | 重叠块的数量 |
| `DEVICE` | `cuda:0` | 训练设备 |
| `TRAIN_DATA_DIR` | `data/AVIRIS/train` | 训练数据目录 |
| `TEST_DATA_DIR` | `data/AVIRIS/test` | 验证数据目录 |
| `MODEL_SAVE_PATH` | `checkpoint/BTCNetV2_8bit.pth` | 检查点保存路径 |

### 训练流程
- **损失函数**：L1（平均绝对误差），用于率失真优化
- **优化器**：Adam，学习率 1e-5
- **调度器**：ExponentialLR，每轮衰减因子 gamma=0.999
- **验证**：每轮定期评估（可通过 `VALID_PERIOD` 配置）
- **检查点保存**：根据峰值 PSNR 自动保存最优模型
- **进度展示**：以紧凑表格格式实时显示训练指标（loss、SAM、RMSE、PSNR）

## 许可证
本工作采用 MIT 许可证。

## 致谢
本工作基于 [Python](https://www.python.org/downloads/release/python-3120/)、[PyTorch](https://pytorch.org/)、[AVIRIS](https://aviris.jpl.nasa.gov/)、[WHU-Hi](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) 和 [Hyperion](https://earthexplorer.usgs.gov/) 实现。感谢他们的出色工作！

## 引用
```bibtex
@article{zhou2025btcnetv2,
  title     = {BTC-Net V2: Spatial-Priority Hierarchical Reconstruction for Hyperspectral Image Compression},
  author    = {Zhou, Xichuan and Wan, Mingyang and Yu, Hang and Chen, Tao and He, Rulong and Shen, Xiangfei and Chen, Lihui and Liu, Haijun},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2025},
  doi       = {10.1109/TGRS.2025.11371342},
  publisher = {IEEE}
}
```
