# 训练信息（最新最优模型：strength_v2_fast）

> 本目录归档 `model/best.pt` 对应 run 的完整训练信息。
> run 目录：`runs/baseline_v2_strength_fast`（2026-08-24，GPU PCI 82:00.0）

## 一页摘要

| 项 | 值 |
|---|---|
| 训练数据 | `data/parquet_lossless/`（同 v2 语料 140k 句，pseudo_strength_v2 规则） |
| 配置 | `train_v1_50m_strength_v2_fast.yaml`（fast 配方） |
| 训练时长 | ~2.4h（fp16，V100，UUID 钉卡 82:00.0） |
| 收敛 | max_steps 15000 跑满，**best@15000（val_composite 0.99804）** |
| **test 集结果** | viseme acc **0.99992** / macro F1 **0.99990** |
| | strength MAE **0.180**（zh 0.142 / en 0.303） |
| 参数量 | 49,859,639（18 类 viseme + strength 0-100） |

## fast 配方三项优化（相对原 200k 步配方）

1. **短调度**：max_steps 200k->15k、warmup 0.05->0.03、patience 5->3
2. **val_composite 选型**：`val_viseme_macro_f1 − α×val_strength_mae/100`（α=1.0），
   解决 macro F1 在噪声平台无区分度的问题--best 跟住了 strength 持续收敛
3. **编码缓存**（`articulm/data/cache.py`）：启动 26min->6min

**收益**：strength MAE 0.356 -> **0.180（-49%）**，viseme macro F1 0.99981 -> 0.99990，
训练时长 2.2h -> 1h（不含一次性编码）。

## val_composite 验证曲线（每 1000 步）

| step | macro F1 | strength MAE | composite |
|---|---|---|---|
| 1000 | 0.99959 | 0.475 | 0.99484 |
| 3000 | 0.99975 | 0.374 | 0.99601 |
| 6000 | 0.99974 | 0.307 | 0.99667 |
| 10000 | 0.99982 | 0.233 | 0.99749 |
| 12000 | 0.99986 | 0.190 | 0.99795 |
| 15000 | 0.99985 | **0.181** | **0.99804** |

## 负结果备忘（30k 加长版）

max_steps 提到 30000 反而更差（MAE 0.298，早停@7000/best@4000）：cosine 按 30k
规划拉长衰减，lr 在 4000-7000 区间仍偏高导致震荡。**结论：本任务「更多步数」
必须配「足够快的衰减」，15k 是甜点。**详见 training_report_v2_strength.md §6.2.1。

## 文件清单

| 文件 | 内容 |
|---|---|
| `training_report_v2_strength.md` | 完整训练报告：v2 规则对照实验 + 4 次硬件事故 + fast 配方 + 30k 负结果 + 三方对照 |
| `train_config.yaml` / `model_config.yaml` / `data_config.yaml` | 本 run 实际加载的配置（trainer 落盘副本，与 config/ 目录中同名文件一致） |
| `training_summary.json` | 最终摘要（global_step 15000 / best@15000 / stopped_by max_steps） |
| `events.jsonl` | 全部训练事件（train_step / validation / checkpoint，含 val_composite 全曲线） |
| `gpu_usage.csv` | GPU 监控采样（30s 间隔，本 run 用卡 82:00.0 / 序列号 ...7200） |
| `feature_vocab.json` | 本 run 词表（与 `../features/` 相同；热启动微调必须配对使用） |
| `metrics.json` | test 评估总指标（acc / macro F1 / 分类表 / 混淆矩阵 / 分语言 strength MAE） |
| `per_class.csv` / `confusion_matrix.csv` | 18 类逐类 P/R/F1 与混淆矩阵 |
| `strength_report.csv` | strength MAE 分切片（总体 / 分语言 / 分 viseme / 分句长 / 分边界） |
| `failure_cases.jsonl` | 全部误分类样本（test 集 26 条） |
| `data_report_v2.json` / `split_report_v2.json` | 数据统计与切分/泄漏检查报告 |

## 复现

```bash
# 数据（本仓库已含 parquet；如需从 JSONL 重建）
bash scripts/to_parquet_lossless.sh <corpus.jsonl> data/parquet_lossless

# 训练
python -m articulm.train --config config/train_v1_50m_strength_v2_fast.yaml \
    --run-dir runs/<new> --device cuda
# 注：该配置的 data_config 指向 JSONL 版数据；改指 config/data_v1_strength_v2_parquet.yaml 即为 parquet 输入

# 评估（还原 test JSONL 后）
python scripts/verify_parquet_lossless.py data/v2/test.jsonl data/parquet_lossless \
    --write-jsonl /tmp/test.jsonl
python -m articulm.evaluate --checkpoint runs/<new>/checkpoints/best.pt \
    --data /tmp/test.jsonl --label-set synthetic --out-dir <out>
```

## 硬件备忘（本机）

本 run 在唯一健康卡上完成（PCI 82:00.0 / 序列号 0328205007200 / UUID
`GPU-a4dd272a-9b88-362a-ec49-0c4be9dbc12b`）。该机另有 3 张卡硬件损坏
（Xid 79 掉总线 / ECC×2），**CUDA_VISIBLE_DEVICES 数字索引与 PCI 地址在该机
均会漂移，必须用 UUID 钉卡**。详见 training_report_v2_strength.md §4.0.x。
