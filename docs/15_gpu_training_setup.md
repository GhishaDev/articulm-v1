# GPU Training Setup — ArticuLM-V1

从本地开发环境迁到 GPU 机器时需要改什么、不需要改什么。

**核心结论：YAML 里没有任何字段是"为了跑在 GPU 上"而必须改的。** 设备自动检测，
`precision: auto` 自动按卡决定。必须做的事在 YAML 之外；YAML 里该改的是性能调优，
其中 `num_workers` 一项影响最大。

配套文档：`docs/11_training_operations.md`（运维总览）、
`docs/13_training_launcher_guide.md`（启动脚本）。

---

## 1. 一页速查

| 项目 | 需要改吗 | 动作 |
|---|---|---|
| torch wheel | **必须** | 装 CUDA 版，见 §2 |
| `precision` | 不用 | `auto` 已按卡正确决策，见 §3 |
| 设备指定 | 不用 | 自动检测 CUDA > MPS > CPU |
| `pin_memory` | 不用 | 检测到 CUDA 自动开启 |
| **`num_workers`** | **强烈建议** | `0` → `4~16`，见 §4 |
| `max_phoneme_tokens_per_batch` | 建议 | 按显存放大，见 §5 |
| `gradient_accumulation_steps` | 建议 | batch 放大后相应降低 |
| `max_steps` / eval / checkpoint 频率 | 建议 | 按语料规模重算，见 §6 |
| 学习率 | 视情况 | 有效 batch 大幅变化时需重调，见 §5 |
| 多卡 | **不支持** | 无 DDP，见 §8 |

---

## 2. 必须做：装对的 torch wheel

唯一的硬性前提。仓库当前环境是 macOS CPU/MPS 版，到 GPU 机器上必须换。

```bash
# 先看清楚宿主机
nvidia-smi

# CUDA 12.1 (A100 / H100 / L40S)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (V100 / T4)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 必须为 `True`。

> **最常见的坑**：装成 CPU-only wheel。训练照样能跑完，但全程在 CPU 上 fp32 —— 白烧
> 几十小时。启动脚本 stage 0 会把这件事挑明：
>
> ```bash
> scripts/run_training.sh --dry-run
> ```
>
> 盯这一行：
>
> ```text
> precision:      bf16  (auto: NVIDIA A100-SXM4-40GB supports bf16)
> ```
>
> 若显示 `fp32 (cpu device runs fp32)`，立刻停 —— CUDA 没被这个解释器看到。

---

## 3. 精度：不需要改

`precision: auto` 已按卡做出正确选择。各卡实测结果：

| GPU | 计算能力 | `auto` 选择 | GradScaler | 显式请求 `bf16` |
|---|---|---|---|---|
| Tesla V100 | sm_70 | **fp16** | 开 | **报错拒绝** |
| Tesla T4 | sm_75 | **fp16** | 开 | **报错拒绝** |
| A100 | sm_80 | **bf16** | 关 | 允许 |
| L40S | sm_89 | **bf16** | 关 | 允许 |
| H100 | sm_90 | **bf16** | 关 | 允许 |

设计要点：

- **V100 类卡请求 bf16 会直接报错，不会静默降级。** 这是 `CLAUDE.md` 的硬约束
  （V100 使用 FP16，不得假设 BF16），代码里有单测钉住。
- bf16 有 fp32 的动态范围，因此不需要 loss scaling；fp16 需要 GradScaler，自动开启。
- CPU / MPS 一律 fp32。在这些设备上请求 fp16/bf16 会降级为 fp32 并打印原因，不报错。

只在两种情况下需要显式指定：

```yaml
training:
  precision: fp32    # 排查数值问题，牺牲速度换确定性
  precision: fp16    # A100 上想与 V100 结果对齐
```

---

## 4. 最关键的一项改动：`num_workers`

```yaml
training:
  batching:
    num_workers: 8        # 默认是 0
```

**默认 `0` 意味着主进程串行完成数据加载与 collation。** 在 CPU 上无所谓，在 GPU 上
这会直接饿死 GPU —— padding、attention/loss mask 构建、slice 元数据收集全部在主
进程，GPU 空等 CPU。

建议取 CPU 物理核数的一半到全部，一般 **4–16**。

实测连通性（YAML → DataLoader）：

```text
num_workers=0  ->  loader.num_workers=0  persistent_workers=False  pin_memory=False
num_workers=4  ->  loader.num_workers=4  persistent_workers=True   pin_memory=True
```

- `persistent_workers` 在 `num_workers > 0` 时自动开启，避免每个 epoch 重建 worker。
- `pin_memory` 在检测到 CUDA 时自动开启，它是 trainer 里 `non_blocking=True`
  主机到设备拷贝真正异步的前提。两者都不需要在 YAML 里配。

> **如果你自己写驱动脚本**：`num_workers > 0` 时 DataLoader 用 spawn 起 worker，
> worker 会重新导入入口 `__main__`。因此入口必须是**可导入的文件或模块**，且实际
> 逻辑要放在 `if __name__ == "__main__":` 之下。用 `python - <<EOF` 这类 stdin
> 脚本会挂住并报 `FileNotFoundError: .../<stdin>`。
> `python -m articulm.train` 与 `scripts/run_training.sh` 都不受影响。

---

## 5. 按显存调 batch

### 固定开销

与 batch 无关的部分：**权重 + 梯度 + 2 个 AdamW 动量 = 0.74 GB**（fp32 优化器状态，
49,852,365 参数）。

### 激活显存

`max_phoneme_tokens_per_batch` 指的是 **padded** token 数（`batch_size × max_len`），
因为那才是决定激活显存的量。用仓库内的估算器算出：

| tokens/batch | fp32 激活 | fp16/bf16 激活 | bf16 总计（含固定开销） |
|---:|---:|---:|---:|
| 2,000 | 0.67 G | 0.33 G | 1.08 G |
| 4,000 | 1.34 G | 0.67 G | 1.41 G |
| **6,000**（当前默认） | 2.00 G | 1.00 G | **1.74 G** |
| 8,000 | 2.67 G | 1.34 G | 2.08 G |
| 12,000 | 4.01 G | 2.00 G | 2.75 G |
| 16,000 | 5.34 G | 2.67 G | 3.41 G |
| 24,000 | 8.01 G | 4.01 G | 4.75 G |
| 32,000 | 10.68 G | 5.34 G | 6.08 G |

当前默认 6000 在 bf16 下只占约 1.7 GB —— **对任何现代卡都过于保守**，那是为本地
CPU 冒烟测试留的余量。

### 按卡建议起点

| 卡 | 建议 tokens/batch | 估算总计 | ×3 实际估计 | 余量 |
|---|---:|---:|---:|---|
| V100 16G / T4 16G | 16,000 | 3.41 G | ~10.2 G | 够 |
| V100 32G / A100 40G | 32,000 | 6.08 G | ~18.3 G | 够 |
| A100 80G / H100 80G | 65,536 | 11.68 G | ~35.0 G | 够 |

```yaml
training:
  batching:
    strategy: dynamic_phoneme_tokens
    max_phoneme_tokens_per_batch: 16000   # V100 16G / T4 16G
    # max_phoneme_tokens_per_batch: 32000 # V100 32G / A100 40G
    # max_phoneme_tokens_per_batch: 65536 # A100 80G / H100 80G
    gradient_accumulation_steps: 1        # batch 放大后可降下来
    num_workers: 8
```

这些是**起点，不是上限**。第一次跑起来后看 `nvidia-smi` 的实际占用再往上推；
`events.jsonl` 里的 `tokens_per_s` 是判断是否已经吃满 GPU 的指标。

### 两个必须注意的地方

**估算只算了主要项。** 残差流、注意力投影、FFN 中间层被计入；cuDNN workspace、
显存碎片、autograd 图的次要项没有。实际占用常是估算的 **2–3 倍**。
**OOM 时相信 OOM，不要相信估算。**

**放大 batch 后学习率通常要跟着调。** 当前 `learning_rate: 3.0e-4` 是配
6000 tokens × accum 2 调的。有效 batch 放大 4 倍以上时，考虑按 √k 或线性缩放，
并同步调整 `scheduler.warmup_ratio`。这需要实验确认，本文不给定值。

### OOM 时怎么退

```yaml
training:
  batching:
    max_phoneme_tokens_per_batch: 8000    # 减半
    gradient_accumulation_steps: 4        # 相应放大，保持有效 batch 不变
```

有效 batch ≈ `max_phoneme_tokens_per_batch × gradient_accumulation_steps`，
两者反向调整即可在不改变优化行为的前提下降低峰值显存。

---

## 6. 长训练的运维项

```yaml
training:
  max_steps: 200000            # 按语料规模与目标 epoch 数重算

  evaluation:
    every_steps: 1000          # 大语料上可放宽到 2000–5000

  checkpoint:
    every_steps: 1000
    keep_last_n: 3
    save_best_by: val_viseme_macro_f1

  early_stopping:
    enabled: true
    patience_evaluations: 5
```

**磁盘要先算一遍**：单个 checkpoint **实测 571 MB**（权重 + 梯度 + AdamW 动量）。
`keep_last_n: 3` 加上 `best.pt` / `last.pt` 约 **2.8 GB**，门禁阶段还会另写一份。
轮转只删 `step_*.pt`，`best.pt` 与 `last.pt` 永不删除。

---

## 7. 现成配置：`config/train_v1_50m_gpu.yaml`

已落成文件，可直接使用。与 `config/train_v1_50m.yaml` 的差异经 diff 确认**只有
下面 5 项性能设置**，10×640 基线、loss 定义、label source 权重全部未变：

| 字段 | 基线 | GPU 版 |
|---|---:|---:|
| `max_phoneme_tokens_per_batch` | 6000 | **32000** |
| `gradient_accumulation_steps` | 2 | **1** |
| `num_workers` | 0 | **8** |
| `evaluation.every_steps` | 1000 | **2000** |
| `checkpoint.every_steps` | 1000 | **2000** |

实测该配置在 dev 语料上构批正常（`num_workers=8`，padding 紧密贴住 32,000 上限，
全部句子覆盖，无 mask 泄漏）：

```text
batch 0:  395 句  max_len= 81  padded=31,995  real=26,414
batch 1:   92 句  max_len=117  padded=10,764  real= 8,394
batch 2:  842 句  max_len= 38  padded=31,996  real=20,472
batch 3:  561 句  max_len= 57  padded=31,977  real=26,251
```

文件内容：

```yaml
# config/train_v1_50m_gpu.yaml
experiment:
  name: articulm_v1_50m_baseline_gpu
  seed: 42
  output_dir: runs/articulm_v1_50m_baseline_gpu

model_config: config/model_v1_50m.yaml
data_config: config/data_v1.yaml

training:
  stage: synthetic_pretraining
  precision: auto                         # 不用改，auto 已正确

  max_steps: 200000

  optimizer:
    type: adamw
    learning_rate: 3.0e-4                 # 有效 batch 大幅变化时需重调
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1.0e-8

  scheduler:
    type: cosine
    warmup_ratio: 0.05

  batching:
    strategy: dynamic_phoneme_tokens
    max_phoneme_tokens_per_batch: 32000   # ← 按显存放大
    gradient_accumulation_steps: 1        # ← 相应降低
    num_workers: 8                        # ← 最关键的一项

  gradient_clip_norm: 1.0

  loss:
    viseme:
      type: cross_entropy
      weight: 1.0
      label_smoothing: 0.05
    strength:
      type: smooth_l1
      weight: 0.3
      normalize_target_to_0_1: true
      beta: 0.1
      source_weights:
        pseudo_strength_v1: 1.0
        human: 1.0

  evaluation:
    every_steps: 2000

  checkpoint:
    every_steps: 2000
    save_last: true
    save_best_by: val_viseme_macro_f1
    keep_last_n: 3

  early_stopping:
    enabled: true
    patience_evaluations: 5

  logging:
    every_steps: 50
```

启动：

```bash
scripts/run_training.sh --config config/train_v1_50m_gpu.yaml --dry-run     # 先核对
scripts/run_training.sh --config config/train_v1_50m_gpu.yaml --background
```

---

## 8. 到 GPU 上仍需确认 / 不支持的部分

### 混合精度的真实数值行为未实测

精度**决策逻辑**有单测覆盖（§3 表格即测试结果），但 `GradScaler.unscale_/step`
的 fp16 路径与 bf16 的实际 kernel 行为**没有在真实 CUDA 上跑过** —— 开发机无 CUDA。

到 GPU 上第一件事应该是让门禁在真实混合精度下过一遍：

```bash
scripts/run_training.sh --stage gate
```

fp16 下若出现 NaN，门禁的 `saw_non_finite_loss` 检查会直接拦住 —— 这正是它存在的
意义。门禁通过即说明前向、反向、loss scaling、checkpoint 在该精度下都成立。

### 多卡不支持

没有 DDP、没有 `DistributedSampler`。单卡可用；多卡需要在 trainer 里新增工作。
当前架构下 `torch.nn.DataParallel` 也未验证，不建议直接套用。

### 主机内存是先于显存撞上的墙

与 GPU 无关，但会先遇到：语料是**一次性解析进主机内存**的，峰值约
**1 KB / phoneme token**。

| 句数 | tokens | 峰值主机内存 |
|---:|---:|---:|
| 10,000 | 0.4 M | 0.4 GB |
| 100,000 | 4.3 M | 4.2 GB |
| 300,000 | 12.9 M | 12.6 GB |
| 1,000,000 | 43.1 M | **42.1 GB** |

超过约 30 万句需要先做流式 Dataset，这与显存大小无关。

---

## 9. 迁移检查清单

- [ ] `nvidia-smi` 能看到卡
- [ ] `torch.cuda.is_available()` 为 `True`，`torch.version.cuda` 非 `None`
- [ ] `scripts/run_training.sh --dry-run` 的 `precision:` 行显示 fp16 或 bf16
      （不是 fp32）
- [ ] `num_workers` 已从 `0` 调到 `4~16`
- [ ] `max_phoneme_tokens_per_batch` 已按显存放大
- [ ] 有效 batch 变化较大时已重新考虑 `learning_rate` 与 `warmup_ratio`
- [ ] 磁盘余量 ≥ `keep_last_n × 571 MB + 2 × 571 MB`，并为门禁另留一份
- [ ] 主机内存足够容纳语料（见 §8 表格）
- [ ] **门禁已在目标 GPU 的真实精度下通过一次**
- [ ] `--background` 启动后已确认 `train.pid` 与 `events.jsonl` 正常写入
