# RT-DETR 小目标检测研究入口

这个工作区已经从官方 RT-DETR 仓库中精简为 PyTorch 检测训练基线。先看本文件，再看配置和模型代码。

## 你现在只需要理解的主线

1. 选一个模型配置，例如 `rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml`。
2. 配置文件会组合数据、模型、优化器和运行参数。
3. `rtdetr_pytorch/tools/train.py` 读取配置并启动训练。
4. 训练过程进入 `src/solver`，模型主体来自 `src/zoo/rtdetr`。
5. 数据集和增强来自 `src/data`。

## 目录作用

| 路径 | 作用 | 你需要关注吗 |
| --- | --- | --- |
| `rtdetr_pytorch/configs` | 训练配置。改数据路径、类别数、输入尺度、epoch、batch size 都从这里开始。 | 必看 |
| `rtdetr_pytorch/tools/train.py` | 训练和验证入口。一般不用改。 | 会用 |
| `rtdetr_pytorch/tools/infer.py` | 单张/少量图片推理入口，用来快速看检测效果。 | 会用 |
| `rtdetr_pytorch/src/zoo/rtdetr` | RT-DETR 模型核心，包括 encoder、decoder、loss、matcher、postprocess。 | 重点 |
| `rtdetr_pytorch/src/nn/backbone` | Backbone，包括 PResNet、DLA、RegNet。 | 可能改 |
| `rtdetr_pytorch/src/data` | COCO 数据集读取、数据增强、dataloader。 | 重点 |
| `rtdetr_pytorch/src/solver` | 训练循环、评估循环、checkpoint 保存。 | 后期再看 |
| `rtdetr_pytorch/src/core` | YAML 配置注册和对象构建机制。 | 暂时少看 |
| `rtdetr_pytorch/src/optim` | optimizer、EMA、AMP。 | 暂时少看 |
| `rtdetr_pytorch/src/misc` | 日志、分布式训练、可视化工具。 | 暂时少看 |

## RT-DETR 核心文件速查

| 文件 | 作用 |
| --- | --- |
| `src/zoo/rtdetr/rtdetr.py` | 把 backbone、encoder、decoder 组装成完整检测模型。 |
| `src/zoo/rtdetr/hybrid_encoder.py` | 多尺度特征融合。小目标研究通常会优先看这里。 |
| `src/zoo/rtdetr/rtdetr_decoder.py` | Transformer decoder、query、预测头。 |
| `src/zoo/rtdetr/rtdetr_criterion.py` | 损失函数。小目标加权或改 loss 时看这里。 |
| `src/zoo/rtdetr/matcher.py` | 匈牙利匹配。改匹配代价时看这里。 |
| `src/zoo/rtdetr/rtdetr_postprocessor.py` | 推理后处理，把输出转成最终框和类别。 |
| `src/zoo/rtdetr/box_ops.py` | bbox 格式转换、IoU、GIoU 等工具。 |

## 小目标检测优先改哪里

建议按这个顺序推进，别一上来全改：

1. 数据和输入尺度：`configs/dataset/coco_detection.yml`、`configs/rtdetr/include/dataloader.yml`
2. 多尺度特征融合：`src/zoo/rtdetr/hybrid_encoder.py`
3. Query 和 decoder：`src/zoo/rtdetr/rtdetr_decoder.py`
4. Loss 和匹配策略：`src/zoo/rtdetr/rtdetr_criterion.py`、`src/zoo/rtdetr/matcher.py`
5. Backbone 输出层级：`src/nn/backbone/presnet.py` 和 `configs/rtdetr/include/rtdetr_r50vd.yml`

## 已经移除的外围内容

为了让项目更适合论文实验，已经移除这些暂时无关内容：

- `hubconf.py`：Torch Hub 加载入口。
- `tools/export_onnx.py`：ONNX 导出脚本。
- `src/data/cifar10`：分类数据集示例。
- `src/nn/arch/classification.py`：分类任务封装。
- `src/nn/backbone/test_resnet.py`：CIFAR 风格 ResNet 测试骨架。
- `src/nn/backbone/utils.py`：未被当前 backbone 使用的中间层提取工具。
- `src/nn/criterion/utils.py`：未被当前检测损失使用的旧目标格式转换工具。
- `onnx`、`onnxruntime` 依赖：训练检测模型暂时不需要。

如果后续需要部署或导出 ONNX，可以从官方仓库恢复这些文件。

## 常用命令

以下命令要在已经装好匹配版 PyTorch/torchvision 的 Linux、WSL 或容器中执行。
AMD 显卡的 Windows 迁移不要直接在 PowerShell 中安装通用 PyPI 版 torch；请先按
[`WINDOWS_WSL_ROCM_CN.md`](WINDOWS_WSL_ROCM_CN.md) 建立 WSL2 + ROCm 容器环境。

安装项目依赖：

```bash
cd "rtdetr_pytorch"
python3 -m pip install -r requirements.txt
```

训练轻量基线：

```bash
python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml
```

用预训练权重微调：

```bash
python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t path/to/checkpoint.pth
```

只做验证：

```bash
python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint.pth --test-only
```

## 使用 VisDrone

VisDrone 原始 TXT 标注需要先转成当前训练管线使用的 COCO JSON。进入
`rtdetr_pytorch` 后执行：

```bash
python3 tools/convert_visdrone_to_coco.py
python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml
```

转换器会把 VisDrone 类别 1–10 映射为模型标签 0–9，并忽略类别 0、11
以及 score 为 0 的区域。

当前 train、val、test-dev 的图片与标注已经一一配对。转换器默认检查完整性；
如果任意图片缺少同名 TXT，会停止并报告缺失数量，避免误用不完整数据训练。

### WSL + ROCm 7.2.3（RX 7800 XT）

完整的 Windows 新机安装、项目和数据集搬运、恢复训练及故障排查步骤见
[`WINDOWS_WSL_ROCM_CN.md`](WINDOWS_WSL_ROCM_CN.md)。下面只保留当前机器已验证的
容器命令速查。

在 WSL 中启动容器并直接打开它的 Bash 主终端：

```bash
docker start -ai rocm723_visdrone
```

也可以运行 `./start_visdrone_container.sh`；容器停止时它会执行上述启动命令，容器已经
运行时则自动连接主终端。只想离开终端并保持容器运行时，按 `Ctrl+P`，再按
`Ctrl+Q`；输入 `exit` 会停止容器。

进入容器后直接训练：

```bash
python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml
```

使用官方稳定镜像中的配套版本，不要在镜像内用 nightly 覆盖它们：

- PyTorch `2.9.1+rocm7.2.3`
- torchvision `0.24.0+rocm7.2.3`

WSL 容器需要同时挂载 `libdxcore`、宿主机 HSA runtime 和 `librocdxg`。
下面是本机已经验证能识别 RX 7800 XT 的完整启动命令：

```bash
docker run -dit --name rocm723_visdrone \
  --device=/dev/dxg \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32g \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro \
  -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1:ro \
  -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so:ro \
  -v /home/lan-ubuntu/rocm-work:/workspace \
  -v /home/lan-ubuntu/models:/models \
  -w "/workspace/my_lab/bachelor paper/rtdetr_pytorch" \
  rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash

docker exec rocm723_visdrone python3 -m pip install -r requirements.txt
docker exec -it rocm723_visdrone \
  python3 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml
```

验证 GPU 和 torchvision NMS：

```bash
docker exec rocm723_visdrone python3 -c \
  "import torch, torchvision; from torchvision.ops import nms; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(nms(torch.tensor([[0.,0.,10.,10.]], device='cuda'), torch.tensor([0.9], device='cuda'), 0.5))"
```

`amd-smi`/`rocm-smi` 在 WSL 中可能报告找不到 Linux `amdgpu` 模块；这不等于
PyTorch 未使用 GPU。以 `torch.cuda.is_available()`、设备名称和 `/dev/dxg` 为准。

## 建议的阅读顺序

1. `configs/rtdetr/rtdetr_r18vd_6x_coco.yml`
2. `configs/rtdetr/include/rtdetr_r50vd.yml`
3. `src/zoo/rtdetr/rtdetr.py`
4. `src/zoo/rtdetr/hybrid_encoder.py`
5. `src/zoo/rtdetr/rtdetr_decoder.py`
6. `src/data/transforms.py`
