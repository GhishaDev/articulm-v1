# ArticuLM-V1

音素 -> 口型（viseme + strength）的 50M 参数模型，用于虚拟主播口型驱动管线。
本仓库包含：模型代码、基于**无损 Parquet** 的全量训练数据、训练/评估/推理代码、脚本与模型权重。

## 目录

| 路径 | 内容 |
|---|---|
| `articulm/` | 模型、训练、评估、推理与调用方服务全部代码（`service.py` / `inference.py` / `data/parquet.py`） |
| `config/` | 全部配置；训练数据入口 `config/data_v1_strength_v2_parquet.yaml` |
| `scripts/` | 无损 parquet 转换与验证（`to_parquet_lossless.sh` / `verify_parquet_lossless.py`）、GPU 监控/自愈等 |
| `tests/` | 测试套件（含无损 parquet 加载一致性测试） |
| `model/` | 训练好的权重 `best.pt` / `last.pt`（571MB，Git LFS） |
| `features/feature_vocab.json` | 18 个特征域词表 |
| `data/parquet_lossless/` | **全量训练数据**（140k 句 / 6,678,301 token，无损，往返验证 0 差异） |
| `training_info/` | 最新模型（best.pt）的完整训练信息：报告、配置、事件日志、GPU 记录、评估产物 |
| `docs/` | 架构 / 训练计划 / 伪强度方法等文档 |

## 训练数据（无损 Parquet）

| 文件 | 内容 | 大小 |
|---|---|---|
| `data/parquet_lossless/train.tokens.parquet` + `train.samples.parquet` | 126,000 句 / 6,009,118 token | 158 MB + 9 MB |
| `data/parquet_lossless/validation.*.parquet` | 7,000 句 | 9.5 MB |
| `data/parquet_lossless/test.*.parquet` | 7,000 句 | 9.4 MB |

- 由 `scripts/to_parquet_lossless.sh` 从 JSONL 生成（clickhouse-local，无需 pyarrow）
- **无损**：保留全部字段（含 teacher/timing 元数据），JSON null 用 Nullable 列保真
- `scripts/verify_parquet_lossless.py` 已验证三切分 140k 句与原 JSONL **逐句语义 100% 一致**，且可 `--write-jsonl` 还原 JSONL

## 训练

```bash
# Parquet 数据加载（train.py 按 .tokens.parquet 后缀自动分发，语义与 JSONL 一致）
python -m articulm.train --config config/train_v1_50m_strength_v2_fast.yaml \
    --run-dir runs/<new> --device cuda
# 该配置的 data_config 需指向 config/data_v1_strength_v2_parquet.yaml，
# 或直接复制修改任意训练配置的 data_config 字段。
```

评估 / 门禁 / 校验：

```bash
python -m articulm.evaluate --checkpoint runs/<new>/checkpoints/best.pt \
    --data <jsonl 或先还原 parquet> --label-set synthetic --out-dir <out>
python -m articulm.data.validate --config config/data_v1_strength_v2_parquet.yaml
```

## 迁移到其它 GPU 机器

训练方法详见 `docs/15_gpu_training_setup.md`（torch wheel 选择、精度表、
按显存调 batch 的完整表格）。大多数情况下**零改动**即可跑：

| 项 | 默认行为 | 何时需要改 |
|---|---|---|
| `precision: auto` | V100/T4 自动 fp16+GradScaler，A100/L40S/H100 自动 bf16 | 不用改；显式写 bf16 在 V100 上会报错拒绝 |
| `--device cuda` | 自动选卡 | 不用改；多卡机器可用 `CUDA_VISIBLE_DEVICES=N` 指定 |
| `max_phoneme_tokens_per_batch: 32000` | 按 16GB V100 标定（总占用 ~6GB，见 docs/15 §5） | **显存 <16GB 时减半并同步加倍 `gradient_accumulation_steps` 保持等效 batch**；更大显存可调大加速 |
| `num_workers: 8` | 数据加载并行 | 按 CPU 核数调整（0 = 主进程加载，会饿 GPU） |

两点说明：
1. `training_info/` 与 `training_report_v2_strength.md` 里的 **UUID 钉卡规则是本仓库
   训练机（3 张卡硬件损坏）特有的**；健康机器直接 `--device cuda` 或数字索引即可。
2. OOM 处方：`max_phoneme_tokens_per_batch` 减半 + `gradient_accumulation_steps` 加倍，
   其余不动。

## 推理

```python
from articulm.inference import ModelPredictor
from articulm.service import predict_api_response

predictor = ModelPredictor.load("model/best.pt")

request = {
    "text": "你好",
    "visemes": [
        {"word": "你", "wordIndex": 0, "charIndex": 0, "ipa": "n"},
        {"word": "你", "wordIndex": 0, "charIndex": 0, "ipa": "i"},
        {"word": "好", "wordIndex": 1, "charIndex": 0, "ipa": "x"},
        {"word": "好", "wordIndex": 1, "charIndex": 0, "ipa": "a"},
    ],
}
out = predict_api_response(request, predictor)
# out["visemes"] -> [{ipa, shapeV2(18类名称), strength, word, wordIndex, charIndex}, ...]
```

## 模型

- 参数量 49,859,639；18 类 viseme（`articulm/visemes.py` 为 id->名称映射）+ strength（0-100）
- test 集：viseme accuracy 0.99992 / macro F1 0.99990 / strength MAE 0.180（zh 0.142 / en 0.303）
- 训练配方：fast 调度（15k 步、warmup 0.03、val_composite 选型）

## 依赖

```bash
pip install -r requirements.txt
# clickhouse-local（parquet 读取与转换）：
sudo apt install clickhouse-local
```

## 上传注意

- `model/*.pt`（571MB）、`data/parquet_lossless/*.parquet`（最大 158MB）超过 GitHub 100MB 单文件上限，
  已配 `.gitattributes` 走 **Git LFS**。
- 若不用 LFS：可删除 `model/` 与 `data/parquet_lossless/`，只上传代码（<5MB）。
