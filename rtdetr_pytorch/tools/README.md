# 脚本

- `train.py`：训练/验证入口。
- `prepare_initialization.py`：构造公共初始权重相同的基线/SGF。
- `analyze_results.py`：逐轮指标、λ 和各类别 AP，可导出 CSV。
- `inspect_sgf_gate.py`：对验证图像统计门控及注入强度。
- `convert_visdrone_to_coco.py`：转换数据标注。
- `infer.py`：图像推理。

完整命令见[运行说明](../../docs/setup/server.md)与[分析说明](../../docs/analysis.md)。
