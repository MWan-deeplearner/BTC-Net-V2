# BTC-Net V2: Spatial-Priority Hierarchical Reconstruction for Hyperspectral Image Compression
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11371342)
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
<sup>4</sup> Primary Contact: [13477304346@163.com]

## ⌛ Updates
**[2026/02/05] Release basic information about BTC-Net V2.**

## 📝 Abstract
*In recent years, lossy compression of hyperspectral images (HSIs) for spaceborne applications has garnered significant attention. The limited storage capacity and constrained transmission bandwidth of spaceborne equipment make it challenging to simultaneously balance the compression rate of transmitted data and the quality of locally reconstructed images. To address this challenge, we propose BTC-Net V2—an advanced iteration of BTC-Net—that substantially enhances the performance of its predecessor. Specifically, in the encoder stage, we adopt a large-kernel convolution to replace the three-layer convolutional network used in the previous version, enabling a more lightweight encoder design that aligns with the resource constraints of spaceborne HSI compression. Moreover, in the decoder stage, we replace the “feature enhancement followed by upsampling” paradigm of the previous version with a spatial-priority hierarchical architecture that prioritizes spatial dimension upsampling. This design ensures the subsequent feature enhancement backbone operates on a larger spatial scale—directly mitigating the loss of spatial correlations in BTC-Net and ultimately yielding higher reconstruction quality. Extensive experimental results demonstrate that our proposed BTC-Net V2 outperforms BTC-Net by 3.35 dB in PSNR—achieving 40.78 dB versus BTC-Net’s 37.43 dB—while operating at an even lower bit rate (0.054 bpppb compared to BTC-Net’s 0.060 bpppb). Our code is available at https://github.com/MWan-deeplearner/BTC-Net-V2.*

## ⚙ Installation
To create and activate the environment, please run the following commands in your **terminal**:
```bash
# Create a new conda environment and activate it
conda create -n btcnetv2 python=3.12
conda activate btcnetv2
# Install dependencies
pip install -r requirements.txt
```

## 📂 Datasets
### 1. Benchmark Descriptions
In this work, we evaluated our method on several widely-used hyperspectral **benchmarks**, including:

| Dataset | Sensor | Description | Link |
| :--- | :--- | :--- | :---: |
| **AVIRIS** | Airborne | Classic hyperspectral flight data | [🔗](https://aviris.jpl.nasa.gov/data/get_aviris_data.html) |
| **WHU-Hi** | UAV-borne | High-resolution datasets from Wuhan University | [🔗](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) |
| **Hyperion** | EO-1 Satellite | Spaceborne hyperspectral mission data | [🔗](https://earthexplorer.usgs.gov/) |

### 2. Data Preparation & Organization
Please download the datasets and organize them into the `data/` directory. Each dataset should have its own sub-directory, containing `train` and `test` folders for the respective samples.

#### Directory Structure:
```bash
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

## 🔥 Training
**Coming soon!**

## 🍭 Evaluation
**Coming soon!**

## 📓 License
This work is licensed under MIT license.

## 🥰 Acknowledgement
This work is implemented based on [Python](https://www.python.org/downloads/release/python-3120/), [PyTorch](https://pytorch.org/), [AVIRIS](https://aviris.jpl.nasa.gov/), [WHU-Hi](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) and [Hyperion](https://earthexplorer.usgs.gov/). Thanks for their awesome work!

## 📖 Citation
**Coming soon!**