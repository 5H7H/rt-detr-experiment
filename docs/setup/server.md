# Linux / NVIDIA 运行说明

下面所有 Python 命令在 `rtdetr_pytorch/` 内执行。首先创建自己的环境，安装与 CUDA 驱动匹配的 PyTorch/torchvision；历史版本见[实验设置](../experiments.md)。

```bash
cd rtdetr_pytorch
python -m pip install -r requirements-experiment.txt
```

## 数据

将 VisDrone 原始 split 文件夹放在 `configs/dataset/VisDrone/`，或使用符号链接；每个 split 内包含 images/ 与 annotations/。转换：

```bash
python tools/convert_visdrone_to_coco.py
```

图像与转换后 JSON 路径在 `configs/dataset/visdrone_detection.yml` 配置。可以改为你自己的路径；不上传数据集。源码 `src/data/` 和配置 `configs/dataset/*.yml` 必须保留。

## 公共初始化

```bash
python tools/prepare_initialization.py --output-dir weights/seed42
```

第一次会下载官方 backbone 权重。生成 baseline.pth 和 sgf.pth，两者公共权重完全一致；不会覆盖已有输出目录。如果持有原实验的初始 baseline 权重，可追加 `--baseline-checkpoint /path/to/initial_model_seed42.pth` 复用；不要传训练完成的权重来冒充初始权重。

## 正式训练

先检查两个配置的 output_dir；新的运行必须使用新的目录。原训练入口本身不会阻止复用输出目录。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/train.py   -c configs/experiments/r50_baseline_72ep.yml   -t weights/seed42/baseline.pth --amp --seed 42

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/train.py   -c configs/experiments/sgf_r50_72ep.yml   -t weights/seed42/sgf.pth --amp --seed 42
```

两条命令分别运行，不建议在显存不足时同时运行。终端关闭后仍需运行时使用 tmux 或单独后台 worker；上面是前台命令。

## 验证与测试

```bash
python -m unittest discover -s tests -v
python tools/train.py -c configs/experiments/sgf_r50_72ep.yml   -r /path/to/checkpoint.pth --test-only
```

仅加载自己信任的 PyTorch checkpoint。验证前也应把 output_dir 指向新目录，避免与旧输出混用。旧 AMD/WSL 教程见 [WINDOWS_WSL_ROCM_CN.md](WINDOWS_WSL_ROCM_CN.md)，它不是当前 NVIDIA 实验的操作前提。
