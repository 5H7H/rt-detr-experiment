# 查看实验结果

在 `rtdetr_pytorch/` 运行；前两项 log 可以指向本仓库的轻量日志，也可以指向服务器 artifacts/log.txt。

```bash
python tools/analyze_results.py   --baseline-log ../results/baseline_vs_sgf/r50_baseline_accum4_v2.jsonl   --sgf-log ../results/baseline_vs_sgf/sgf_r50_72ep_v1.jsonl   --csv /tmp/sgf_comparison.csv
```

脚本显示最后一轮、各自最佳 AP/APs 及其轮次；不会把不同轮次拼接成同一结果。CSV 参数可省略，已有 CSV 不覆盖。

读取 λ 和各类别 AP/APs（需要 PyTorch；只加载自己信任的文件）：

```bash
python tools/analyze_results.py   --checkpoint /path/to/sgf/artifacts/checkpoint.pth   --baseline-eval /path/to/baseline/artifacts/eval/latest.pth   --sgf-eval /path/to/sgf/artifacts/eval/latest.pth   --annotations configs/dataset/visdrone_annotations/val.json
```

门控统计需要模型和图像，不能仅从权重直接读出：

```bash
python tools/inspect_sgf_gate.py   --checkpoint /path/to/sgf/artifacts/checkpoint.pth --limit 16
```

默认 CPU、EMA、验证集前 16 张图像，显示门控均值、每图分位数的均值、饱和比例和注入项/F3 的 L2 范数比；不会更新模型或保存权重。`--limit 548` 可覆盖全部验证图像。少量样本统计不能证明门控关注了小目标。
