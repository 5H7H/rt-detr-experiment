# 实验设置与来源

| 项目 | 两组共同设置 |
|---|---|
| 数据 | VisDrone train 6471，val 548，10 类 |
| 训练 | 72 epochs；seed 42；物理 batch 1；累积 4，末组 3 |
| 优化器 | AdamW；普通参数 lr 1e-4；backbone lr 1e-5；weight decay 1e-4（部分 bias/norm 为 0） |
| 调度 | MultiStepLR milestone 1000，因此本次 72 轮无学习率衰减 |
| 稳定性 | AMP、EMA decay 0.9999 / warmups 2000；梯度裁剪 0.1 |
| 尺寸 | 训练 480–800 多尺度，640 在采样表重复 3 次；验证 640×640 |
| 验证 | batch 1；EMA；COCOeval bbox；maxDets=[1,10,100] |
| 初始化 | 官方 R50 backbone 预训练，检测器其余参数初始化；两组公共参数完全相同 |

SGF 未从训练完成的基线权重继续训练，而是从公共初始权重开始。完整初始权重不在 GitHub；重建方法见运行说明。浮点平台、库版本、随机数消费顺序仍可影响结果，seed 相同不等于每次运算完全一致。

服务器已验证环境：Python 3.12.3、torch 2.9.1+cu128、torchvision 0.24.1+cu128、transformers 4.57.6、pycocotools 2.0.11、PyYAML 6.0.3、scipy 1.18.1。历史训练使用 2 个 CPU 线程及约 4 GiB PyTorch 显存分配上限；便携命令不默认限制显存。

数据转换保留类别 1–10 并映射为 0–9，跳过原始类别 0/11 和 score=0 标注。这套 COCO 格式评估不应不加说明地视作 VisDrone 官方挑战评估。比较外部论文时还需对齐忽略区域处理、maxDets、数据划分和预训练来源。

原始运行标识：`r50_baseline_accum4_v2`、`sgf_r50_72ep_v1`。结果目录中的 JSONL 是服务器逐轮指标的原始副本，epoch 从 0 编号；展示时加 1。源代码来自实际 SGF 实验副本，训练器与两组原实验一致；新增便携配置只改变 include/output 路径。

目前结果只有一个种子，不能给出显著性结论。不要混合不同轮次的最好指标当作同一个 checkpoint 的结果。
