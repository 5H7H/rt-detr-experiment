# 本次仓库整理的验证

2026-09-28，在隔离目录和实际实验 Python 环境中完成：

- 所有 Python 文件语法解析通过。
- SGF 两项单元测试通过：S2D 可逆、lambda=0 时输出等价、lambda 非零后分支梯度传播；三尺度 encoder 输出等价。
- 新的基线/SGF 配置展开后与原实验参数相同，比较时仅排除 output_dir、include 路径和显式 null 的默认项。
- 使用原实验初始 baseline 权重运行 prepare_initialization.py，公共参数逐张量完全一致。
- analyze_results.py 读取两组完整 72 轮日志、EMA lambda 及真实 COCOeval 类别指标成功。
- inspect_sgf_gate.py 在 CPU 上对 2 张真实验证图像运行成功。历史 16 张图像诊断统计另列于结果说明，不能将两个样本量混为同一次统计。
- 数据加载源码与 YAML 配置不再被 gitignore 排除。

本次整理没有重新训练模型，也没有更改原实验权重或结果。测试未覆盖新的 GPU、其他库版本、多卡训练及部署导出。
