# Training Data Contract — ArticuLM-V1

**用途**：上游数据侧（`articulm_data_pipeline`）与模型训练工程之间的数据接口契约。
数据交付前请按 [§7 交付前自查](#7-交付前自查) 逐项确认。

**约束**：本文描述的是训练代码**实际执行**的校验规则，不是建议。所有规则由
`articulm/data/schema.py` 强制，违规样本在训练启动前就会被拒绝，不会自动修补。

| 版本 | `schema_version` | 校验入口 |
|---|---|---|
| v1 | `articulm_v1_sample_v1` | `python -m articulm.data.validate --config config/data_v1.yaml` |

相关文档：`docs/02_input_output_schema.md`（schema 设计）、
`docs/03_training_data_spec.md`（数据规格）、
`docs/10_data_pipeline_integration.md`（管线集成）、
`docs/12_training_sample_examples.md`（样本示例）。

---

## 1. 文件格式

**JSONL** —— 一行一个完整 JSON 对象，一行一句。

```text
data/
  train.jsonl          一行一句
  validation.jsonl
  test.jsonl
```

- 编码 UTF-8，不加 BOM
- **不是** JSON 数组：文件不以 `[` 开头，行间无逗号
- 允许空行（被跳过）
- 单行长度无硬限制，但单句 phoneme 数不得超过 `max_seq_len`（默认 256）

如果只交付一个未切分的整体语料，切分由训练侧执行（见 [§8](#8-切分与去重由训练侧执行)）。

---

## 2. 句级结构

```json
{
  "schema_version": "articulm_v1_sample_v1",
  "sample_id": "zh_broadcast_000123",
  "text": "预计2026年销售额增长12.5%。",
  "normalized_text": "预计二零二六年销售额增长百分之十二点五。",
  "normalization_version": "tn_v1",
  "tokens": [ /* 每个 phoneme 一个对象 */ ],
  "teacher_metadata": { "raw_value": 100 }
}
```

| 字段 | 必需 | 类型 | 说明 |
|---|---|---|---|
| `tokens` | **是** | array | phoneme 序列。缺失/非数组/为空 → 拒收 |
| `schema_version` | 强烈建议 | string | 存在时必须完全等于 `articulm_v1_sample_v1` |
| `sample_id` | 强烈建议 | string | 全局唯一。报错定位、failure case 追溯都依赖它 |
| `text` | 建议 | string | 原文，参与去重签名 |
| `normalized_text` | 建议 | string | TN 后文本。**存在时优先作为去重身份** |
| `normalization_version` | 建议 | string | TN 规则版本，便于溯源 |
| `teacher_metadata` | 否 | object | teacher 侧信息的**唯一合法容器** |

### 训练单位是整句

一行必须是一个完整句子的 phoneme 序列。**不能**把单个 phoneme 拆成独立行：

```json
// 错误：每个 phoneme 一行，模型拿不到上下文
{"phoneme": "n", "viseme_id": 14}
{"phoneme": "i", "viseme_id": 3}
```

上下文正是这个模型存在的意义 —— 同一个 phoneme 在不同前后音、不同短语位置下
应当输出不同的 viseme 概率与 strength。拆开就没有可学的东西了。

### `normalized_text` 与训练/推理一致性

训练与推理必须使用**同一套** Text Normalization 规则。不能训练时把 `2026`
读作"二零二六"、推理时变成"二千零二十六"。建议始终写入 `normalized_text` 与
`normalization_version`，训练侧会用它作为去重身份。

---

## 3. Token 结构

每个 token 有 7 个字段块，展开成 18 个编码列：

```json
{
  "phoneme": "n",
  "language": "zh",
  "surface_tone": 2,
  "stress": 0,
  "syllable_role": "onset",
  "articulatory": {
    "type": "consonant",
    "height": null,
    "backness": null,
    "rounded": null,
    "place": "alveolar",
    "manner": "nasal",
    "voiced": true,
    "aspirated": false
  },
  "boundary": {
    "word_start": true,
    "word_end": false,
    "phrase_start": true,
    "phrase_end": false,
    "boundary_type": "none"
  },
  "labels": {
    "viseme_id": 14,
    "strength": 64.7,
    "viseme_source": "website_rule",
    "strength_source": "pseudo_strength_v1"
  }
}
```

### 3.1 语言学特征

| 字段 | 必需 | 类型 | 取值 | 违规后果 |
|---|---|---|---|---|
| `phoneme` | **是** | string | 非空、非纯空白 | 拒收 |
| `language` | **是** | string | `zh` / `en` | 拒收 |
| `surface_tone` | 是 | int | 中文 `1..5`；英文 `0` | 拒收 |
| `stress` | 是 | int | 中文 `0`；英文 `0/1/2` | 拒收 |
| `syllable_role` | 建议 | string | `onset`/`nucleus`/`coda`/`other`/`silence` | 缺省 `other` |

`phoneme` 是**开放集**：训练侧从 train split 构建词表，未见音素在推理时映射到
`[UNK]`。请保持音素记号在整个语料中**一致**（同一个音不要既写 `zh` 又写 `zh_`），
否则词表会碎片化。`validate` 会报出 unknown phoneme 率。

`stress` 是**音节级**特征，请广播到该音节的所有 phoneme，而不是只标在元音上。

### 3.2 `articulatory`（8 个子字段）

| 子字段 | 类型 | 示例取值 |
|---|---|---|
| `type` | string / null | `consonant` `vowel` `silence` |
| `height` | string / null | `high` `mid` `low` `close` `open` |
| `backness` | string / null | `front` `central` `back` |
| `rounded` | bool / null | |
| `place` | string / null | `bilabial` `labiodental` `alveolar` `retroflex` `velar` `glottal` … |
| `manner` | string / null | `plosive` `nasal` `fricative` `affricate` `approximant` `lateral` … |
| `voiced` | bool / null | |
| `aspirated` | bool / null | |

**`null` 是合法且有意义的**，不是"缺数据"，而是"该属性对这个音素不适用" ——
元音没有 `place`，辅音没有 `height`。它映射到专用 `[NA]` 类别，与 `[UNK]`
（真未见值）严格区分。

#### 表示"不适用"只能用 `null`（或省略）

这是一个**不会报错、但会静默劣化模型**的陷阱。实测行为：

| 写法 | 归一化结果 | 后果 |
|---|---|---|
| `null` | `[NA]` | 正确 |
| 省略该子字段 | `[NA]` | 正确 |
| `""` / `"  "` | `[NA]` | 等价，可接受 |
| `"none"` / `"NONE"` | `none` | **错误：多出一个独立类别** |
| `"null"` | `null` | **错误：多出一个独立类别** |
| `"N/A"` | `n/a` | **错误：多出一个独立类别** |
| `"-"` | `-` | **错误：多出一个独立类别** |

后四种都能通过校验，但会在词表里制造一个与 `[NA]` 并列的伪类别，把本该合并的
"不适用"样本劈成两半。**混用更糟** —— 一半 `null` 一半 `"none"` 会让同一语义
落到两个 embedding 上。

注意 `boundary.boundary_type` 的合法取值本身就包含字符串 `"none"`，那里的
`"none"` 是有意义的取值，不是"不适用"。两者不要混淆。

整个 `articulatory` 块可以省略（全部按 `[NA]` 处理），但**不能出现未列出的子字段**
—— 多余键会被拒收，这是为了让拼写错误（例如 `mannner`）立刻暴露而不是静默丢弃。

字符串取值大小写不敏感（内部统一转小写），但请保持一致。取值不必局限于上表 ——
未列出的值会进入词表，`validate` 报告里可以看到实际分布。

### 3.3 `boundary`（5 个子字段）

| 子字段 | 类型 | 取值 |
|---|---|---|
| `word_start` | bool | 缺省 `false` |
| `word_end` | bool | 缺省 `false` |
| `phrase_start` | bool | 缺省 `false` |
| `phrase_end` | bool | 缺省 `false` |
| `boundary_type` | string | `none` / `minor` / `major`，缺省 `none` |

同样不允许未列出的子字段。建议在长播报句里显式标出短语边界 —— 评估协议
（`docs/06`）把 phrase-final 音素列为重点难例切片。

### 3.4 `labels`（监督目标）

| 字段 | 必需 | 类型 | 范围 |
|---|---|---|---|
| `viseme_id` | **是** | int | `0..17` 闭区间 |
| `strength` | **是** | float | `0..100` 闭区间，不允许 NaN/Inf |
| `viseme_source` | 强烈建议 | string | 例 `website_rule` `teacher_rule` `human` |
| `strength_source` | **强烈建议** | string | 例 `pseudo_strength_v1` `human` |

**`strength_source` 直接决定 loss 权重**，请务必如实填写：

```yaml
# config/train_v1_50m.yaml
loss:
  strength:
    weight: 0.3
    source_weights:
      pseudo_strength_v1: 1.0
      human: 1.0
```

程序先验与人工标注混在一个文件里是允许且被支持的 —— 训练侧按 `strength_source`
逐 token 加权。但**标错来源会静默污染训练**：把程序值标成 `human` 会让它获得
Human Gold 的权重。这是本契约里最需要数据侧配合的一点。

`strength` 的**每个 token 都必须有**。phoneme 数与 label 数不一致的样本会被拒收，
不会自动错位补齐。

---

## 4. 禁止出现在 token 特征层级的字段

```text
viseme_id    strength    shapeV2    Talk    raw_value
```

这五个字段出现在 token 顶层（或 `features` 子块里）→ **立即拒收**。

**错误：**

```json
{ "phoneme": "a", "viseme_id": 2, "raw_value": 100 }
```

**正确** —— target 只能在 `labels` 里，teacher 侧信息只能在 `teacher_metadata` 里：

```json
{
  "teacher_metadata": { "raw_value": 100, "shapeV2": "...", "duration": 0.08 },
  "tokens": [
    { "phoneme": "a",
      "labels": { "viseme_id": 2, "strength": 84.0,
                  "strength_source": "pseudo_strength_v1" } }
  ]
}
```

`duration` / `timing` 允许保留在文件中（V1 不作为 encoder 输入），parser 不读取。
其余五个字段则是 label leakage，必须拒收 —— `docs/09` 把"teacher labels 进入
encoder 输入"列为整个 run 的作废条件。

### `raw_value` 不得被改名为 Human Gold

```json
// 错误：网页 rule 值被标成人工标注
{ "labels": { "strength": 100, "strength_source": "human", "raw_value": 100 } }
```

`strength_source` 为 `human` 且同时携带 `raw_value` → 拒收。
`raw_value ≈ 100` 不是真实 Strength GT。

---

## 5. 完整示例

### 5.1 中文（`你好。`）

4 个 phoneme。实际文件中必须压成一行。

```json
{
  "schema_version": "articulm_v1_sample_v1",
  "sample_id": "zh_nihao_001",
  "text": "你好。",
  "tokens": [
    {
      "phoneme": "n", "language": "zh", "surface_tone": 2, "stress": 0,
      "syllable_role": "onset",
      "articulatory": { "type": "consonant", "height": null, "backness": null,
                        "rounded": null, "place": "alveolar", "manner": "nasal",
                        "voiced": true, "aspirated": false },
      "boundary": { "word_start": true, "word_end": false,
                    "phrase_start": true, "phrase_end": false,
                    "boundary_type": "none" },
      "labels": { "viseme_id": 14, "strength": 64.7,
                  "viseme_source": "website_rule",
                  "strength_source": "pseudo_strength_v1" }
    },
    {
      "phoneme": "i", "language": "zh", "surface_tone": 2, "stress": 0,
      "syllable_role": "nucleus",
      "articulatory": { "type": "vowel", "height": "high", "backness": "front",
                        "rounded": false, "place": null, "manner": null,
                        "voiced": true, "aspirated": false },
      "boundary": { "word_start": false, "word_end": true,
                    "phrase_start": false, "phrase_end": false,
                    "boundary_type": "none" },
      "labels": { "viseme_id": 3, "strength": 76.0,
                  "viseme_source": "website_rule",
                  "strength_source": "pseudo_strength_v1" }
    },
    {
      "phoneme": "x", "language": "zh", "surface_tone": 3, "stress": 0,
      "syllable_role": "onset",
      "articulatory": { "type": "consonant", "height": null, "backness": null,
                        "rounded": null, "place": "velar", "manner": "fricative",
                        "voiced": false, "aspirated": false },
      "boundary": { "word_start": true, "word_end": false,
                    "phrase_start": false, "phrase_end": false,
                    "boundary_type": "none" },
      "labels": { "viseme_id": 15, "strength": 58.0,
                  "viseme_source": "website_rule",
                  "strength_source": "pseudo_strength_v1" }
    },
    {
      "phoneme": "a", "language": "zh", "surface_tone": 3, "stress": 0,
      "syllable_role": "nucleus",
      "articulatory": { "type": "vowel", "height": "low", "backness": "central",
                        "rounded": false, "place": null, "manner": null,
                        "voiced": true, "aspirated": false },
      "boundary": { "word_start": false, "word_end": true,
                    "phrase_start": false, "phrase_end": true,
                    "boundary_type": "major" },
      "labels": { "viseme_id": 2, "strength": 84.0,
                  "viseme_source": "website_rule",
                  "strength_source": "pseudo_strength_v1" }
    }
  ]
}
```

> 上例的 `strength` 数值（64.7 / 76.0 / 58.0 / 84.0）来自 `docs/12` 的示意值，
> `strength_source` 标注为 `pseudo_strength_v1` —— **不是人工标注**。正式标签必须
> 由数据管线实际产出。

### 5.2 英文 token

```json
{
  "phoneme": "w",
  "language": "en",
  "surface_tone": 0,
  "stress": 1,
  "syllable_role": "onset",
  "articulatory": { "type": "consonant", "height": null, "backness": null,
                    "rounded": null, "place": "bilabial", "manner": "glide",
                    "voiced": true, "aspirated": false },
  "boundary": { "word_start": true, "word_end": false,
                "phrase_start": true, "phrase_end": false,
                "boundary_type": "none" },
  "labels": { "viseme_id": 10, "strength": 67.1,
              "viseme_source": "teacher_rule",
              "strength_source": "pseudo_strength_v1" }
}
```

英文恒 `surface_tone = 0`，`stress` 才承载 prosody。

### 5.3 中英混合（同句内切换）

`新的AI模型将在GPU服务器上运行。` 在 `的` → `AI` 处的实际切换：

```text
phoneme=e    language=zh  surface_tone=5  stress=0  role=nucleus    ← 的（轻声）
phoneme=ey   language=en  surface_tone=0  stress=1  role=nucleus    ← AI
```

同一句内允许任意次 `language` 切换。每个 token 独立遵守各自语言的
tone/stress 约定。

### 5.4 推理输入（无标签）

同一套 schema，整体省略 `labels`：

```json
{"schema_version":"articulm_v1_sample_v1","sample_id":"x","text":"你好。","tokens":[{"phoneme":"n","language":"zh","surface_tone":2,"stress":0,"syllable_role":"onset","articulatory":{"type":"consonant","place":"alveolar","manner":"nasal","voiced":true,"aspirated":false},"boundary":{"word_start":true,"phrase_start":true,"boundary_type":"none"}}]}
```

现成样例：`examples/sample.jsonl`。

---

## 6. 拒收规则全表

以下 20 条均已实测确认。报错信息定位到**文件:行号[sample_id].字段路径**，例如：

```text
data/train.jsonl:2[bad_one].tokens[0]: Chinese surface_tone must be one of [1,2,3,4,5], got 9
data/train.jsonl:2: invalid JSON (Expecting property name enclosed in double quotes: ...)
```

| # | 触发条件 | 报错信息（节选） |
|---|---|---|
| 1 | `tokens` 缺失 | `missing 'tokens'; the training unit is a full phoneme sequence` |
| 2 | `tokens` 为空数组 | `'tokens' must not be empty` |
| 3 | `tokens` 非数组 | `'tokens' must be a list` |
| 4 | `schema_version` 不匹配 | `schema_version 'v0' != expected 'articulm_v1_sample_v1'` |
| 5 | 句长超 `max_seq_len` | `sequence length N exceeds data.max_seq_len 256` |
| 6 | `phoneme` 空或纯空白 | `'phoneme' must be a non-empty string` |
| 7 | `language` 缺失 | `'language' is required` |
| 8 | `language` 不在支持列表 | `language='ja' is not in data.language.supported ['zh','en']` |
| 9 | 中文 `surface_tone ∉ 1..5` | `Chinese surface_tone must be one of [1,2,3,4,5], got 0` |
| 10 | 中文 `stress != 0` | `Chinese stress must be 0, got 1` |
| 11 | 英文 `surface_tone != 0` | `English surface_tone must be 0, got 3` |
| 12 | 英文 `stress ∉ 0/1/2` | `English stress must be one of [0,1,2], got 3` |
| 13 | `viseme_id ∉ 0..17` | `labels.viseme_id must be in [0,17], got 18` |
| 14 | `strength ∉ 0..100` | `labels.strength must be in [0.0,100.0], got 101.0` |
| 15 | `strength` 为 NaN/Inf | `NaN/Inf is not allowed, got nan` |
| 16 | `labels` 缺失 / 缺 `viseme_id` / 缺 `strength` | `missing 'labels'` / `missing 'viseme_id'` |
| 17 | token 层级出现 target/teacher 字段 | `target/teacher fields ['viseme_id'] must not appear as encoder features` |
| 18 | `strength_source=human` 且带 `raw_value` | `a programmatic raw value must not be relabelled as Human Gold` |
| 19 | `articulatory`/`boundary` 有未知子字段 | `unknown fields ['mannner']` |
| 20 | 行不是合法 JSON | `<file>:<line>: invalid JSON (...)` |

任何一条命中 → 整个训练任务在启动前终止。**不做自动修补、不跳过坏行。**

---

## 7. 交付前自查

在交付前，数据侧可以直接运行训练侧的校验器，无需等模型侧反馈。

### 7.1 运行校验

```bash
# 指向你的文件
python -m articulm.data.validate --config config/data_v1.yaml \
  --json-out reports/data_report.json

# 只看某个 split，或只查前 N 句快速迭代
python -m articulm.data.validate --config config/data_v1.yaml --split train --limit 1000
```

退出码 `0` = 全部通过。

### 7.2 报告里需要确认的项

```text
Sentences:                100,000          与预期句数一致
Phoneme Tokens:           3,240,816
Seq Length p50/p95/max:   32 / 71 / 214    max 必须 <= 256
Unknown Phoneme Rate:     0.0300%          越低越好；异常高说明记号不一致

Viseme Classes:
   0: ...  (x.xx%)                          18 类都应有样本
  ...
  MISSING CLASSES: [7, 13]                  出现这行说明覆盖不足

Strength:
  mean / std / p05 / p50 / p95 / max        分布是否合理，是否堆在 100

Languages:                zh: 98.7%, en: 1.3%
Surface tones:            1..5 都应出现
Strength label sources:   {'pseudo_strength_v1': 3240816}
Human Gold strength:      0 tokens          与实际标注量一致
```

### 7.3 检查清单

- [ ] JSONL，一行一句，UTF-8 无 BOM
- [ ] `schema_version` = `articulm_v1_sample_v1`
- [ ] `sample_id` 全局唯一
- [ ] 每句 phoneme 数 = label 数，且 ≤ 256
- [ ] 中文 `surface_tone ∈ 1..5` 且 `stress = 0`
- [ ] 英文 `surface_tone = 0` 且 `stress ∈ 0/1/2`
- [ ] `viseme_id ∈ 0..17`，`strength ∈ 0..100`，无 NaN/Inf
- [ ] `articulatory` 的不适用属性统一用 `null`（或省略），**绝不用 `"none"`/`"null"`/`"N/A"`/`"-"`**
- [ ] `articulatory` / `boundary` 无拼写错误的多余子字段
- [ ] token 顶层无 `viseme_id`/`strength`/`shapeV2`/`Talk`/`raw_value`
- [ ] `strength_source` 如实反映来源；程序先验**不得**标为 `human`
- [ ] 音素记号在全语料内一致（unknown rate 可接受）
- [ ] 18 个 viseme 类都有样本（`MISSING CLASSES` 为空）
- [ ] `normalized_text` + `normalization_version` 已写入（推荐）
- [ ] `python -m articulm.data.validate` 退出码为 0

---

## 8. 切分与去重由训练侧执行

**建议数据侧交付未切分的整体语料**，train/validation/test 切分由训练侧完成：

```bash
python -m articulm.data.split --config config/data_v1.yaml \
  --input data/corpus.jsonl --report-out reports/split_report.json
```

原因：切分必须保证重复句与近重复句**不跨 split**，否则 held-out 指标虚高，
按 `docs/09` 整个 run 作废。训练侧的切分器会：

1. 按 `normalized_text`（无则 `text`）与 phoneme 序列双签名合并精确重复
2. 用 phoneme n-gram 的 bottom-k sketch 找近重复，再算精确 Jaccard
3. 重复组**整组**分配到同一 split，事后独立复核无泄漏
4. 按行**逐字节复制**原始 JSONL，切分不改变任何数据语义

数据侧需要知道的两点：

- **`normalized_text` 会显著提升去重质量** —— 它是首选去重身份。
- **语料重复度过高会导致比例达不到 90/5/5**。重复组不会为凑比例被拆开；
  如果某个 split 为空，切分命令直接失败（退出码 1）而不是静默产出无验证集的配置。
  报告里的 `Redundant sentences` 就是语料的重复质量指标。

如果数据侧已自行切分，请确保跨 split 无重复/近重复，并在交付说明中写明切分方式。

---

## 9. 数据集 manifest（推荐随语料交付）

```json
{
  "dataset_version": "v1.0",
  "num_sentences": 100000,
  "num_phoneme_tokens": 3240816,
  "languages": ["zh", "en"],
  "source_batches": ["batch_001", "batch_002"],
  "viseme_label_source": "teacher_rule",
  "strength_label_source": "pseudo_strength_v1",
  "normalization_version": "tn_v1",
  "schema_version": "articulm_v1_sample_v1",
  "split": "unsplit"
}
```

训练侧会把实际统计写进 run 目录，与此 manifest 对照即可发现交付偏差。

---

## 10. 变更策略

任何字段增删、取值集合变化、或 label 语义调整，都必须：

1. 递增 `schema_version`（例如 `articulm_v1_sample_v2`）
2. 同步更新本文与 `articulm/data/schema.py`
3. 说明旧 checkpoint 的兼容性影响 —— 词表变化会使旧 checkpoint 不可直接复用

**不要在保持 `schema_version` 不变的情况下改变数据语义。** 训练侧的校验只能
检出结构违规，检不出"同一个字段悄悄换了含义"，而后者会直接污染模型且难以追溯。
