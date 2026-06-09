[English](README.md) | [中文](README.zh-CN.md)

# BTC-Net V2: Spatial-Priority Hierarchical Reconstruction for Hyperspectral Image Compression
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11371342)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/the-bad-one/BTC-Net-V2)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=Python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=PyTorch&logoColor=white)](https://pytorch.org/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=MWan-deeplearner.BTC-Net-V2&left_text=Total%20Views&color=00629B)

## Authors
* **Xichuan Zhou**<sup>1</sup>
* **Mingyang Wan**<sup>1,4</sup> (Main Contributor & Maintainer)
* **Hang Yu**<sup>1</sup>, **Tao Chen**<sup>1</sup>, **Rulong He**<sup>2</sup>, **Xiangfei Shen**<sup>2</sup>, **Lihui Chen**<sup>1</sup>, **Haijun Liu**<sup>1,3</sup>

<sup>1</sup> Chongqing University, China  
<sup>2</sup> Naval University of Engineering, China  
<sup>3</sup> Corresponding Author: [haijun_liu@cqu.edu.cn]  
<sup>4</sup> Primary Contact: [mingyang_wan@163.com]

## Updates
* **[2026/02/05]** Release basic information about BTC-Net V2.
* **[2026/06/02]** Release model code, pretrained checkpoints (8-bit & 32-bit), and evaluation script.
* **[2026/06/03]** Release training script with full reproduction pipeline, including rate-distortion optimization, mixed-precision support, and progress tracking.

## Abstract
*In recent years, lossy compression of hyperspectral images (HSIs) for spaceborne applications has garnered significant attention. The limited storage capacity and constrained transmission bandwidth of spaceborne equipment make it challenging to simultaneously balance the compression rate of transmitted data and the quality of locally reconstructed images. To address this challenge, we propose BTC-Net V2—an advanced iteration of BTC-Net—that substantially enhances the performance of its predecessor. Specifically, in the encoder stage, we adopt a large-kernel convolution to replace the three-layer convolutional network used in the previous version, enabling a more lightweight encoder design that aligns with the resource constraints of spaceborne HSI compression. Moreover, in the decoder stage, we replace the "feature enhancement followed by upsampling" paradigm of the previous version with a spatial-priority hierarchical architecture that prioritizes spatial dimension upsampling. This design ensures the subsequent feature enhancement backbone operates on a larger spatial scale—directly mitigating the loss of spatial correlations in BTC-Net and ultimately yielding higher reconstruction quality. Extensive experimental results demonstrate that our proposed BTC-Net V2 outperforms BTC-Net by 3.35 dB in PSNR—achieving 40.78 dB versus BTC-Net's 37.43 dB—while operating at an even lower bit rate (0.054 bpppb compared to BTC-Net's 0.060 bpppb). Our code is available at https://github.com/MWan-deeplearner/BTC-Net-V2.*

## Model Architecture

BTC-Net V2 follows an encoder-decoder paradigm with learned quantization and entropy coding.

### Encoder
A single **Quantized Convolution layer** (`QuantConv2d`) replaces the three-layer network from BTC-Net V1:
- Large kernel: 11×4 with stride 4, achieving aggressive spatial downsampling in one shot
- Supports configurable quantization bit-width (8-bit / 32-bit) via uniform quantization with straight-through estimation (STE)
- PReLU activation with learnable slope

### Decoder: Spatial-Priority Hierarchical Reconstruction (SPHR)
The decoder follows a **spatial-first, spectral-last** reconstruction order:

1. **SpatialRestorer**: PixelShuffle-based upsampling that recovers the original spatial resolution (H×W) from the compressed latent
2. **DetailRestorer**: A U-Net-style backbone with skip connections at multiple scales that refines spatial details:
   - Multi-scale processing (1× → 1/2× → 1/4× → 1/2× → 1×)
   - Each scale uses **DRAM** (Detail Refinement Attention Module) blocks
3. **SpectralRestorer**: Final 1×1 convolution that projects features back to the original spectral dimension, with a residual skip connection from the SpatialRestorer output

### DRAM: Detail Refinement Attention Module
Each DRAM block consists of two sub-modules applied in sequence:
- **SESAM** (Spatial Enhancement Self-Attention Module): Channel-wise self-attention that models long-range spatial dependencies, with a learnable temperature parameter α
- **SEFFM** (Spectral Enhancement Feed-Forward Module): Gated feed-forward network that expands spectral features by a factor γ (default 4), enhancing spectral representation capacity

## Project Structure
```
.
├── README.md
├── test.py                         # Evaluation script with entropy coding
├── train.py                        # Training script with rate-distortion optimization
├── model/
│   ├── __init__.py
│   ├── BTCNetV2.py                 # Main model (encoder + SPHR decoder)
│   ├── SPHRDecoder.py              # Spatial-Priority Hierarchical Reconstruction decoder
│   ├── DRAM.py                     # DRAM block (SESAM + SEFFM)
│   └── QuantizedConv2d.py          # Quantized convolution with STE
├── utils/
│   ├── __init__.py
│   ├── dataset.py                  # HSI dataset loader (.mat files)
│   ├── huffman_coder.py            # Huffman entropy coding pipeline
│   ├── metrics.py                  # PSNR, SAM, RMSE evaluation metrics
│   └── bar.py                      # Training progress display utilities
├── checkpoint/
│   ├── BTCNetV2_8bit.pth           # 8-bit quantized pretrained weights (~48 MB)
│   └── BTCNetV2_32bit.pth          # 32-bit floating-point pretrained weights (~48 MB)
└── data/
    └── AVIRIS/
        ├── train/                  # Training data (.mat files)
        └── test/                   # Sample test data (.mat files)
```

## Installation

### Requirements
- Python 3.12
- PyTorch 2.1+
- einops
- scipy
- numpy
- Pillow

To create and activate the environment, please run the following commands in your **terminal**:
```bash
# Create a new conda environment and activate it
conda create -n btcnetv2 python=3.12
conda activate btcnetv2
# Install dependencies
pip install torch>=2.1 einops scipy numpy Pillow
```

## Datasets
### 1. Benchmark Descriptions
In this work, we evaluated our method on several widely-used hyperspectral **benchmarks**, including:

| Dataset | Sensor | Description | Link |
| :--- | :--- | :--- | :---: |
| **AVIRIS** | Airborne | Classic hyperspectral flight data | [Link](https://aviris.jpl.nasa.gov/data/get_aviris_data.html) |
| **WHU-Hi** | UAV-borne | High-resolution datasets from Wuhan University | [Link](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) |
| **Hyperion** | EO-1 Satellite | Spaceborne hyperspectral mission data | [Link](https://earthexplorer.usgs.gov/) |

### 2. Data Preparation & Organization
Please download the datasets and organize them into the `data/` directory. Each dataset should have its own sub-directory, containing `train` and `test` folders for the respective samples.

#### Directory Structure:
```
.
└── your_project_root/
    ├── data/
    │   ├── AVIRIS/
    │   │   ├── train/         # Place AVIRIS training samples here
    │   │   └── test/          # Place AVIRIS testing samples here
    │   ├── WHU-Hi/
    │   │   ├── train/         # Place WHU-Hi training samples here
    │   │   └── test/          # Place WHU-Hi testing samples here
    │   └── Hyperion/
    │       ├── train/         # Place Hyperion training samples here
    │       └── test/          # Place Hyperion testing samples here
    ├── train.py               # Train script
    ├── test.py                # Test script
    └── ...
```

Data files should be in `.mat` format. Each `.mat` file is expected to contain a single variable holding the hyperspectral data cube.

## Pretrained Checkpoints

Two pretrained model variants are provided in the `checkpoint/` directory:

| Checkpoint | Quantization | Size | Description |
| :--- | :--- | :--- | :--- |
| `BTCNetV2_8bit.pth` | 8-bit | ~48 MB | Quantized weights & activations; lower bitrate |
| `BTCNetV2_32bit.pth` | 32-bit | ~48 MB | Full-precision weights & activations; higher quality |

Both checkpoints use the default configuration: 172 input channels, 27 compressed channels, scale factor 4, 32 base features, γ=4.

## Evaluation

Use `test.py` to evaluate a pretrained model on HSI data. The script reports four metrics: **bpp** (bits per pixel per band), **PSNR**, **SAM** (Spectral Angle Mapper), and **RMSE**.

### Quick Start
```bash
python test.py
```

### Configuration
The key hyperparameters in `test.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `ORIGINAL_CHANNELS` | 172 | Input spectral bands |
| `COMPRESSED_CHANNELS` | 27 | Compressed latent channels |
| `QUANT_BIT` | 8 | Quantization bit-width (8 or 32) |
| `SCALE` | 4 | Spatial downsampling factor |
| `NUM_FEATURES` | 32 | Base feature channels in decoder |
| `GAMMA` | 4 | SEFFM spectral expansion factor |
| `CHECKPOINT` | `checkpoint/BTCNetV2_8bit.pth` | Path to pretrained weights |
| `DATA_DIR` | `data/AVIRIS/test` | Test data directory |
| `DEVICE` | `cuda:0` | Inference device |

To switch to 32-bit mode, change `QUANT_BIT` to `32` and `CHECKPOINT` to `checkpoint/BTCNetV2_32bit.pth`.

### Output
The script prints per-image results in the format:
```
bpp, psnr, sam, rmse:
<BPP value>
<PSNR value (dB)>
<SAM value>
<RMSE value>
```

### Metrics
- **bpp** (bits per pixel per band): Compression rate — lower is better
- **PSNR** (Peak Signal-to-Noise Ratio): Reconstruction fidelity — higher is better
- **SAM** (Spectral Angle Mapper): Spectral distortion — lower is better
- **RMSE** (Root Mean Square Error): Per-band error — lower is better

## Training

Use `train.py` to train BTC-Net V2 from scratch. The script implements rate-distortion optimization with L1 loss, supports mixed-precision training, and logs per-epoch metrics.

### Quick Start
```bash
python train.py
```

### Configuration
The key hyperparameters in `train.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `ORIGINAL_CHANNELS` | 172 | Input spectral bands |
| `COMPRESSED_CHANNELS` | 27 | Compressed latent channels |
| `QUANT_BIT` | 6 | Quantization bit-width for training |
| `SCALE` | 4 | Spatial downsampling factor |
| `NUM_FEATURES` | 32 | Base feature channels in decoder |
| `GAMMA` | 4 | SEFFM spectral expansion factor |
| `BATCH_SIZE` | 12 | Training batch size |
| `MAX_EPOCHS` | 5000 | Maximum training epochs |
| `CROP_HEIGHT` | 128 | Random crop height for training patches |
| `WIDTH` | 4 | Patch width for sliding window |
| `MARGINAL` | 60 | Number of overlapping patches |
| `DEVICE` | `cuda:0` | Training device |
| `TRAIN_DATA_DIR` | `data/AVIRIS/train` | Training data directory |
| `TEST_DATA_DIR` | `data/AVIRIS/test` | Validation data directory |
| `MODEL_SAVE_PATH` | `checkpoint/BTCNetV2_8bit.pth` | Checkpoint save path |

### Training Pipeline
- **Loss function**: L1 (mean absolute error) for rate-distortion optimization
- **Optimizer**: Adam with learning rate 1e-5
- **Scheduler**: ExponentialLR with gamma=0.999 per epoch
- **Validation**: Periodic evaluation every epoch (configurable via `VALID_PERIOD`)
- **Checkpointing**: Automatically saves best model based on peak PSNR
- **Progress display**: Real-time training metrics (loss, SAM, RMSE, PSNR) in a compact table format

## License
This work is licensed under the MIT License.

## Acknowledgements
This work is implemented based on [Python](https://www.python.org/downloads/release/python-3120/), [PyTorch](https://pytorch.org/), [AVIRIS](https://aviris.jpl.nasa.gov/), [WHU-Hi](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) and [Hyperion](https://earthexplorer.usgs.gov/). Thanks for their awesome work!

## Citation
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
