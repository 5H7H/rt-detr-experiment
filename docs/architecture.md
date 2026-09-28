# SGF 结构

Backbone 输出 S2/S3/S4/S5，stride 分别为 4/8/16/32。保持基线 AIFI 与三尺度 CCFF；完成 top-down 得到 F3 后，执行：

```text
X2 = SiLU(BN(Conv1x1(PixelUnshuffle(S2, factor=2))))
G  = sigmoid(Conv1x1(concat(X2, F3)))
F3_new = F3 + lambda * G * X2
```

R50 的 S2 有 256 个通道；S2D 后为 1024，经投影恢复为 256。G 是单通道空间门控，在通道间广播。lambda 是零初始化的可学习标量，不限制正负。

F3_new 同时作为最细输出和 bottom-up PAN 的输入；decoder 仍只接收 stride 8/16/32，不增加 stride-4 检测层。

代码：[hybrid_encoder.py](../rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py)，查看 `SpatialGatedFusion`、`HybridEncoder.__init__`、`HybridEncoder.forward`。

lambda=0 且公共权重一致时，基线与 SGF 初始推理输出一致。首个反向传播时 lambda 可以获得梯度，而投影/门控的梯度为零；lambda 非零后分支权重开始学习，这是公式的预期行为。

新增 263170 参数；总参数从 42747522 增至 43010692。参数增量不能替代 FLOPs/FPS 测量。
