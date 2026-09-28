# RT-DETR / SGF-DETR：VisDrone 小目标检测实验

本仓库基于 [RT-DETR 官方实现](https://github.com/lyuwenyu/RT-DETR)，研究将 stride-4 的 S2 特征通过空间门控残差融合注入 R50 的 stride-8 F3。它是个人研究实验仓库，不是 RT-DETR 官方仓库。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 理解 SGF 结构 | [结构与代码位置](docs/architecture.md) |
| 安装、准备数据、训练 | [运行说明](docs/setup/server.md) |
| 核对实验参数与评估口径 | [实验记录](docs/experiments.md) |
| 查看结果 | [结果表与完整曲线数据](results/baseline_vs_sgf/README.md) |
| 读取日志、λ、各类别 AP | [分析脚本说明](docs/analysis.md) |

## 已完成实验

VisDrone train 6471 张 / val 548 张；R50、72 epochs、seed 42、batch 1、梯度累积 4、AMP、EMA。下表为最后一轮 EMA 验证结果，单位为 %。

| 模型 | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| R50 baseline | 25.401 | 43.692 | 24.723 | 15.757 | 36.076 | 45.774 |
| SGF-DETR | 25.504 | 44.069 | 24.815 | 15.904 | 35.989 | 47.149 |

这是单随机种子的结果：小目标 AP 增加约 0.146 个百分点，尚不能证明稳定提升。两组最佳总 AP 和最佳小目标 AP 均在第 72 轮。尚未完成重复种子实验、门控消融和 FLOPs/FPS 测量。

## 目录

```text
docs/                  结构、运行、实验与分析说明
results/               轻量指标与逐轮日志（不含权重）
rtdetr_pytorch/
  src/                 完整模型、数据加载和训练代码
  configs/experiments/ 基线与 SGF 两个实验入口
  configs/dataset/     数据路径及类别配置（不是数据集本体）
  tools/               训练、初始化和分析脚本
  tests/               SGF 的输出等价性与梯度验证
```

网络关键文件是 `rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py`。同一代码通过 `sgf_s2_channels: null` 使用基线，通过 `256` 启用 R50 的 SGF 分支。

数据集、模型权重、环境、完整训练输出不上传 GitHub。训练前按说明设置数据路径，并为每次运行选择新的输出目录。

原始官方文档保留在 [docs/upstream](docs/upstream)，旧 Windows/AMD 环境教程保留在 [docs/setup](docs/setup)。这些旧文档中的模型成绩和命令不代表本次实验。

## 来源与许可

保留上游 [Apache-2.0 LICENSE](LICENSE)。RT-DETR 来源与论文引用见 [上游 README](docs/upstream/README.md)。SGF 为本仓库的实验性扩展；GFF 是门控融合设计的参考，而非本实现的同名复现。
