# 16. 下一阶段计划：Human Gold 微调 + 真实数据评估

> 状态：计划（待数据侧确认后执行）
> 前置：基线训练已完成并归档（`archive/baseline_v1_20260821_synthetic/`，
> best@8000：test macro F1 0.99984 / strength MAE 0.340）
> 编写日期：2026-08-22

## 0. 背景与出发点

合成数据上基线已触及标签噪声上限（viseme acc 0.99988，40/333,295 误分类全部为
停顿/句尾类边界混淆）。**合成测试集上已无提升空间，真正的未知量是真实语料上的表现**。
下一阶段回答两个问题：

1. 当前基线在真实数据上差多少？（真实数据评估，先于任何微调）
2. Human Gold 微调能把差距缩小到多少？（微调 + 对比评估）

阶段依赖关系：

```text
Phase B（真实数据评估，无需训练）──┐
                                    ├─> Phase C（Human Gold 微调）──> Phase D（最终对比与归档）
Phase A（Human Gold 数据制备）──────┘
```

Phase B 不依赖 Phase A，可并行启动：只要有一批真实标注（或人工抽查）样本即可测基线。

---

## Phase A：Human Gold 数据制备（数据侧，前置）

### A1. 现状盘点

- 数据工厂（articulm_dataset_factory）当前**没有** Human Gold 标注数据：
  `selected/*.scores.jsonl` 只是文本筛选分数，teacher 路径全部产出程序化标签
- 模型侧契约已就绪：`strength_source ∈ {human, human_gold}` 会被
  `HUMAN_GOLD_STRENGTH_SOURCES` 识别并参与 source_weights 加权；
  CLAUDE.md 硬规则：**程序化 strength 永远不得标为 Human Gold**

### A2. 标注规模建议（分级，避免一次性投入）

| 批次 | 规模 | 用途 |
|---|---|---|
| HG-pilot | 500-1,000 句（中英各半） | 标注规范校准 + 标注者一致性（IAA）验证 + Phase B 评估 |
| HG-v1 | 5,000-10,000 句 | 微调训练集 |
| HG-test | 1,000-2,000 句（**冻结，永不参与训练/调参**） | 最终 Human Gold 测试集（docs/03 要求固定 held-out） |

### A3. 标注规范要点（数据侧执行清单）

- 输出契约与现有 `articulm_v1_sample_v1` 完全一致（18 类 viseme + strength 0-100）
- 每条样本记录 `strength_source: "human"`、`viseme_source: "human"` 及标注者 ID
- IAA 验收：抽样 10% 双人标注，viseme 一致率 >= 0.90，strength MAE(标注者间) <= 5.0，
  不达标先修规范再扩量
- 切分沿用工厂现有 split 工具（句级 + 近重复去重，防 HG 训练/测试泄漏）
- **验收口径：数据经 `articulm.data.validate` exit 0，且 `strength_source` 无伪标**

### A4. 模型侧配套小改动（本仓库，约半天）

| 项 | 说明 | 现状 |
|---|---|---|
| `--init-from` 参数 | 仅加载权重（fresh optimizer/step/stage），区别于断点续训语义的 `--resume` | **缺失，需新增**（`resume` 会连 optimizer/scheduler/step 一起恢复） |
| 微调配置 YAML | `config/train_v1_50m_human_gold.yaml`（见 Phase C） | 需新增 |
| HG 评估配置 | `data_hg_test.yaml` 指向 HG-test | 需新增 |

---

## Phase B：真实数据评估（模型侧，可立即启动）

目的：在任何微调之前量化"合成 -> 真实"的 domain gap，为微调收益建立参照。

### B1. 评估对象

| 模型 | checkpoint |
|---|---|
| 基线（本次） | `archive/baseline_v1_20260821_synthetic/model/best.pt`（step 8000） |

### B2. 评估集（按可得性降级）

1. **首选**：HG-pilot（Phase A 前 500-1,000 句）
2. **过渡方案**（HG 数据就绪前）：从工厂 teacher 原始请求中抽取真实文本 +
   人工抽查 100-200 句做错误分析（不做定量结论，只做错误类型学）

### B3. 评估协议

```bash
CUDA_VISIBLE_DEVICES=3 python -m articulm.evaluate \
    --checkpoint archive/baseline_v1_20260821_synthetic/model/best.pt \
    --data data/hg_pilot.jsonl --label-set human \
    --out-dir runs/hg_eval_baseline/reports/hg_pilot
```

- `--label-set human` 与 synthetic 结果**分开报告**（evaluate 已强制单一 label_set）
- 重点切片：停顿/句尾类（10/14/17，合成集上仅有的错误来源）、静音 token、
  中英分开、句长桶
- 产出：`hg_pilot_baseline_eval.md`（gap 量化 + 错误类型学 + 对 Phase C 的建议）

### B4. 判读标准

| HG-pilot 结果 | 解读 | 对 Phase C 的动作 |
|---|---|---|
| macro F1 >= 0.95 且 MAE <= 5 | 合成训练泛化良好 | 小规模微调即可（或直接冻结） |
| 0.80 <= F1 < 0.95 | 存在 domain gap，可微调修复 | 按计划微调，优先看错误集中在哪些类 |
| F1 < 0.80 或 MAE > 15 | 深层分布差异 | 先做错误归因（特征缺失？标注规范不一致？），可能需要回数据侧 |

---

## Phase C：Human Gold 微调

### C1. 训练配置（`config/train_v1_50m_human_gold.yaml` 要点）

```yaml
experiment:
  name: articulm_v1_50m_human_gold_ft
  seed: 42

training:
  stage: human_gold_finetuning        # 区分事件日志与报告归属
  precision: auto                      # V100 -> fp16（沿用）
  max_steps: 20000                     # 微调步数上限（数据量 5-10k 句，防过拟合）
  optimizer:
    type: adamw
    learning_rate: 2.0e-5              # backbone（docs/04: 1e-5 ~ 5e-5）
    head_learning_rate: 2.0e-4         # heads（docs/04: 1e-4 ~ 3e-4，差分已支持）
    weight_decay: 0.01
  scheduler:
    type: cosine
    warmup_ratio: 0.10                 # 微调数据少，warmup 占比略高
  loss:
    viseme: {type: cross_entropy, weight: 1.0, label_smoothing: 0.05}
    strength: {type: smooth_l1, weight: 1.0, beta: 0.1,   # CLAUDE.md: HG 阶段 1.0x
               normalize_target_to_0_1: true,
               source_weights: {human: 1.0, human_gold: 1.0,
                                pseudo_strength_v1: 0.0}}  # 程序化来源权重降为 0（见 C3）
  batching: {strategy: dynamic_phoneme_tokens,
             max_phoneme_tokens_per_batch: 16000,          # HG 数据少，batch 调小
             gradient_accumulation_steps: 1, num_workers: 8}
  evaluation: {every_steps: 500}         # 数据少，评估更密
  checkpoint: {every_steps: 500, save_last: true,
               save_best_by: val_viseme_macro_f1, keep_last_n: 3}
  early_stopping: {enabled: true, patience_evaluations: 8}  # 微调曲线噪声大，patience 放宽
```

### C2. 启动方式

```bash
CUDA_VISIBLE_DEVICES=2 python -m articulm.train \
    --config config/train_v1_50m_human_gold.yaml \
    --init-from archive/baseline_v1_20260821_synthetic/model/best.pt \
    --vocab archive/baseline_v1_20260821_synthetic/features/feature_vocab.json \
    --run-dir runs/human_gold_ft_v1 --device cuda
```

三个关键点：

1. `--init-from`：**仅权重**（本阶段新增）；backbone + heads 全部从基线初始化
2. `--vocab`：**冻结词表**必须用基线训练时的 `feature_vocab.json`，
   否则 embedding 索引错位（静默错误，最危险）
3. 新 run-dir `runs/human_gold_ft_v1`，绝不写入 `runs/baseline_gpu` 或归档目录

### C3. 关键设计决策与理由

| 决策 | 理由 |
|---|---|
| strength loss 1.0（synthetic 是 0.3） | CLAUDE.md 规定的 HG 配方；HG strength 是真 GT |
| `pseudo_strength_v1` source_weight 0.0 | 微调集里若混入程序化标签会稀释 GT 信号；0 权重即"只在 HG 句子上学 strength"。若 HG 句数 < 5k 导致过拟合，可回调至 0.1-0.3 并记录 |
| 差分 LR（backbone 2e-5 / heads 2e-4） | docs/04 建议；backbone 已良好，主要让 head 适配真实分布 |
| max_steps 20k + 密集早停 | 5-10k 句微调极易过拟合；patience=8 次评估容忍噪声 |
| label_smoothing 保持 0.05 | 与基线一致，避免引入第二个变量 |

### C4. 门禁（CLAUDE.md：长训练前必须通过）

1. shape tests：`pytest -q`（全量）
2. tiny-overfit 门禁：用 HG 格式的 64 句玩具数据重跑 gate（验证新 loss 权重 + 
   source_weights 路径 + `--init-from` 通路）
3. 冒烟：`--limit 200 --max_steps 200` 在 GPU 2 上跑通，检查 events.jsonl 无异常

### C5. 消融与对照（可选，GPU 2/3 都空闲时）

| 实验 | 目的 |
|---|---|
| 只调 heads（backbone 冻结） | 判断 gap 在表征还是在决策边界 |
| `localconv` 开关（已有 config flag） | 验证局部卷积对停顿/边界类的作用 |

对照实验各自独立 run-dir + 独立归档。

---

## Phase D：最终验证与归档

### D1. 评估矩阵（全部用 GPU 3）

| 评估 | 数据 | label-set | 模型 |
|---|---|---|---|
| D1 合成测试集（回归） | data/test.jsonl | synthetic | 基线 / 微调后 |
| D2 HG-test | data/hg_test.jsonl | human | 基线 / 微调后 |
| D3 HG-test strength 专项 | 同上 | human | 两者 |

验收线（微调后，HG-test）：

- viseme macro F1 >= 0.92（HG 标注本身有噪声，0.95+ 不现实）
- strength MAE <= 8.0（0-100 量纲；人工标注一致性通常 MAE 5 左右，模型逼近标注者间一致性即为上限）
- 合成测试集回归跌幅 <= 0.5 个百分点（防灾难遗忘）

### D2. 报告与归档

- 报告：`reports/human_gold_ft_report.md`，结构沿用本次 training_report.md
  （摘要/环境/数据/门禁/超参/进度/最终验证），外加**基线对照表**（D1-D3 矩阵）
- 归档：`archive/human_gold_ft_v1_<日期>/`，结构沿用 `baseline_v1_20260821_synthetic/`
  （README + config + data + features + training + evaluation + gate + model + code + environment）

---

## 时间线估算（按 Phase A 标注到位为界）

| 阶段 | 工时 | 备注 |
|---|---|---|
| Phase A 模型侧配套（`--init-from` + 配置 + 测试） | 0.5-1 天 | 不依赖标注数据，**可先做** |
| Phase B（HG-pilot 到位后） | 0.5 天 | 评估 + 错误分析报告 |
| Phase A 数据侧（标注 5-10k 句） | 取决于标注团队产能 | 建议 pilot 500 句先行 |
| Phase C 门禁 + 微调 | 1 天（含 20k 步约 2-3h） | |
| Phase D 评估 + 归档 | 0.5 天 | |

**建议立即执行**：Phase A 模型侧配套（`--init-from`）+ Phase B 过渡方案（错误类型学抽查），
两者都不阻塞在标注数据上。

---

## Phase A'：小规模 HG 数据的应对阶梯（补充）

预训练 126k 句 vs HG 标注量级不匹配是预期内的（标准 pretrain->finetune 范式，
微调通常只需预训练 1-5% 的数据）。按实际到手量级分级应对：

| HG 到手量级 | 策略 |
|---|---|
| 10k+ 句 | 原方案全量微调（backbone 2e-5 + heads 2e-4） |
| 3k-10k 句 | 全量微调 + 防过拟合三件套：max_steps 降 10k、混入 10-20% 合成 replay、`pseudo_strength_v1` source_weight 回调 0.1-0.3（程序化先验从"被替代"变回"被校正"） |
| 1k-3k 句 | heads-only 微调（backbone 冻结，有效参数 ~35 万）+ 标定基线对照：`strength' = a_k x pseudo + b_k`（per-class 仿射，零训练）；微调必须打败标定才算数 |
| <1k 句 | 放弃微调，只做 per-class 标定 + Phase B 评估；剩余预算全部给 HG-test |

原则：**评估集优先于训练集**--HG-test（冻结）是唯一能量化真实差距的资产，
宁可训练集缩水。预算 3k 句时切分 500(pilot) / 1.5k(train) / 1k(test)。

两个不花 HG 预算的先行实验：

1. **标定可行性检验**：teacher 原始真实文本跑基线推理 + 人工抽查 100-200 句，
   区分"分布差异"与"规则性偏移"（后者标定可修，无需微调）
2. **合成模拟微调**：取 2k 句合成数据当"伪 HG"走完微调流程，测该量级数据
   对已收敛模型的拉偏幅度，为真实 HG 的过拟合风险定标

## 风险与对策

| 风险 | 对策 |
|---|---|
| HG 标注与合成标签系统性不一致（同一文本两种"真相"） | Phase B 先量化 gap；若不一致主要来自标注规范，先修规范再微调，否则微调学的是噪声 |
| 微调过拟合（HG 数据少） | max_steps 20k + 密集早停 + 合成集回归监控（跌幅 > 0.5pp 即报警） |
| 灾难遗忘合成分布 | 可选混入 10-20% 合成样本（source_weights 控制 strength，viseme 无来源区分需在文档记录） |
| 词表漂移（HG 数据含新音素） | `--vocab` 冻结基线词表，新音素落 [UNK]；若 HG-pilot 中 [UNK] 率 > 1%，回数据侧扩词表并重训 |
| GPU 再出硬件故障（现仅 2/3 可用） | 训练钉 GPU 2、评估钉 GPU 3；checkpoint keep_last_n=3 保证可恢复；坏一张卡即暂停等用户决策 |
| 把程序化标签误标为 Human Gold | 数据校验阶段强制检查 `strength_source` 白名单；标注工具输出即源头，不做事后改写 |
