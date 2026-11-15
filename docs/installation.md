# 本地安装与验证

检测日期：2026-11-15（Europe/Stockholm）。仅使用本机，不使用 Docker、远程计算或 sudo。

## 硬件实测

Ubuntu 22.04.5，内核 6.8.0-138-generic，i7-12700H，内存约 16GB，RTX 3060 Laptop 6144MiB。
NVIDIA 驱动 595.99.02；nvidia-smi 的 CUDA 13.2 是驱动支持上限，不是已安装的 PyTorch runtime。
安装 PyTorch CUDA 12.6 wheel，不更改 NVIDIA 驱动或系统 CUDA。开始时桌面占约 1229MiB 显存、磁盘剩余约 193GB。
已有 vlaBaseline 是 Python 3.12.15，仅检查，未修改。

## 版本依据

选择已有 LIBERO processor 接口的稳定 LeRobot v0.4.4，而非不断变化的 main。
源码 commit：8fff0fde7c79f23a93d845d1a50e985de01f8b8a。
此版本声明 hf-libero >=0.1.3,<0.2、Transformers >=4.57.1,<5。
hf-libero 0.1.3 是 Hugging Face 维护的 LIBERO 移植，要求 robosuite==1.4.0、robomimic==0.2.0、MuJoCo>=3。
本项目选定值见 constraints.txt，完整解析版本安装后写入 artifacts/pip-freeze.txt。
PyTorch 2.7.1 与 torchvision 0.22.1 使用 cu126，torchcodec 0.5.0 与 torch 2.7 配对。
首次尝试 NumPy 1.26.4 被 resolver 拒绝：必需的 rerun-sdk >=0.24 要求 NumPy>=2。
随后改用上游 requirements-ubuntu.txt 同款 NumPy 2.2.6，真实模拟验证另行记录；OpenCV 固定 4.11。

官方来源（本次已访问并核对源码）：

- https://huggingface.co/docs/lerobot/v0.4.4/en/libero
- https://github.com/huggingface/lerobot/tree/v0.4.4
- https://github.com/Lifelong-Robot-Learning/LIBERO
- https://github.com/huggingface/LIBERO
- https://pypi.org/pypi/hf-libero/0.1.3/json
- https://huggingface.co/HuggingFaceVLA/smolvla_libero
- https://pytorch.org/get-started/previous-versions/
- https://github.com/pytorch/torchcodec#installing-torchcodec

## 从新终端安装

```bash
cd /home/zy/vla_study
conda env create -f environment.yml
# 首次获取上游源码；已有 vendor/lerobot 时不用重复 clone
mkdir -p vendor
git clone --branch v0.4.4 https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot checkout 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
bash scripts/install_dependencies.sh
source scripts/activate.sh
python scripts/check_environment.py --simulator
```

LIBERO_CONFIG_PATH 指向项目 .runtime/libero，不修改 ~/.libero。HF_HOME 指向项目缓存。
模型和视觉底座、模拟资产均需要公开网络下载。下载失败必须保留报错，不能视为环境通过。
EGL 是首选；只有看到实际 EGL 故障后再研究 OSMesa，不预先更改系统渲染配置。

模型 revision 固定为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。
模拟资产 dataset revision 固定为 `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，约 493MB。
通过脚本下载到项目 HF 缓存，再将专用环境中的本地 assets 链接至该不可变快照。

## 环境隔离的实测修正

Conda Python 默认读取 ~/.local/lib/python3.10/site-packages，发现旧 rosbags 缺少 lz4。
本项目设置 PYTHONNOUSERSITE=1，重新解析安装后 `pip check` 无错误。没有修改旧 ROS 包。
GPU 真实矩阵运算通过：torch 2.7.1+cu126、CUDA runtime 12.6。

辅助 VLM/processor revision 固定为 `4cde9d208e19519c36917aa170fffc7e381aa0ad`。
Baseline 启动时通过 scripts/download_vlm.py 下载该固定快照，将项目私有缓存的 main alias 固定到它，再使用 HF_HUB_OFFLINE=1 评估，防止底座 tokenizer/processor 漂移。runner 同样检查该 alias 并离线加载。
