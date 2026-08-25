# Training Sample Examples

本文件提供 ArticuLM-V1 的训练样本示例，便于开发 Dataset、Collator、Schema Validator、Loss 和单元测试。

> 重要说明
>
> - 示例遵守当前模型定义：每个 phoneme 输出一个 `viseme_id` 与一个 `strength`。
> - `viseme_id` 为 0～17。
> - `strength` 为 0～100。
> - `shapeV2 / Talk / raw_value / timing` 如果存在，只属于 teacher metadata，不进入 Encoder。
> - 下面部分 Strength 数值用于展示数据结构与训练流程；除明确标注为 Human Gold 外，不应解释为真实人工 Ground Truth。
> - 真实训练文件应以数据生成 Pipeline 的实际输出为准。

---

## 1. 最小中文样本：`你好。`

这个例子适合用于：

- Dataset 单元测试
- Feature Vocabulary 测试
- Forward shape 测试
- Tiny Overfit
- Viseme / Strength Loss 测试

普通话表层声调：

```text
你：3 → 2
好：3
```

因此 phoneme 级 tone 广播后：

```text
n  → 2
i  → 2
x  → 3
a  → 3
```

示例 JSONL：

```json
{
  "schema_version": "articulm_v1_sample_v1",
  "sample_id": "zh_nihao_001",
  "text": "你好。",
  "tokens": [
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
    },
    {
      "phoneme": "i",
      "language": "zh",
      "surface_tone": 2,
      "stress": 0,
      "syllable_role": "nucleus",
      "articulatory": {
        "type": "vowel",
        "height": "high",
        "backness": "front",
        "rounded": false,
        "place": null,
        "manner": null,
        "voiced": true,
        "aspirated": false
      },
      "boundary": {
        "word_start": false,
        "word_end": true,
        "phrase_start": false,
        "phrase_end": false,
        "boundary_type": "none"
      },
      "labels": {
        "viseme_id": 3,
        "strength": 76.0,
        "viseme_source": "website_rule",
        "strength_source": "pseudo_strength_v1"
      }
    },
    {
      "phoneme": "x",
      "language": "zh",
      "surface_tone": 3,
      "stress": 0,
      "syllable_role": "onset",
      "articulatory": {
        "type": "consonant",
        "height": null,
        "backness": null,
        "rounded": null,
        "place": "velar",
        "manner": "fricative",
        "voiced": false,
        "aspirated": false
      },
      "boundary": {
        "word_start": true,
        "word_end": false,
        "phrase_start": false,
        "phrase_end": false,
        "boundary_type": "none"
      },
      "labels": {
        "viseme_id": 15,
        "strength": 58.0,
        "viseme_source": "website_rule",
        "strength_source": "pseudo_strength_v1"
      }
    },
    {
      "phoneme": "a",
      "language": "zh",
      "surface_tone": 3,
      "stress": 0,
      "syllable_role": "nucleus",
      "articulatory": {
        "type": "vowel",
        "height": "low",
        "backness": "central",
        "rounded": false,
        "place": null,
        "manner": null,
        "voiced": true,
        "aspirated": false
      },
      "boundary": {
        "word_start": false,
        "word_end": true,
        "phrase_start": false,
        "phrase_end": true,
        "boundary_type": "major"
      },
      "labels": {
        "viseme_id": 2,
        "strength": 84.0,
        "viseme_source": "website_rule",
        "strength_source": "pseudo_strength_v1"
      }
    }
  ]
}
```

模型真正使用的输入字段：

```text
phoneme
language
surface_tone
stress
syllable_role
articulatory
boundary
```

模型监督目标：

```text
viseme_id
strength
```

---

## 2. 中文播报样本：数字 / 日期 / 百分比

示例文本：

```text
预计2026年全年销售额将增长12.5%。
```

这类样本用于覆盖虚拟主播常见的：

- 年份
- 连续数字
- 小数
- 百分比
- 财经播报
- 长 phoneme sequence
- 数字与中文词语边界

训练数据仍然必须先经过统一 Text Normalization / G2P Frontend。

概念流程：

```text
预计2026年全年销售额将增长12.5%。
        ↓
Text Normalization
        ↓
预计二零二六年全年销售额将增长百分之十二点五。
        ↓
G2P / Surface Tone
        ↓
phoneme-level sequence
        ↓
ArticuLM training tokens
```

训练文件中建议同时保留：

```json
{
  "text": "预计2026年全年销售额将增长12.5%。",
  "normalized_text": "预计二零二六年全年销售额将增长百分之十二点五。",
  "normalization_version": "tn_v1"
}
```

> 训练和推理必须使用同一套 Text Normalization 规则。不要训练时把 `2026` 读作“二零二六”，推理时却变成“二千零二十六”。

部分 token 示例：

| phoneme | tone | role | boundary | label |
|---|---:|---|---|---|
| ... | ... | ... | ... | ... |
| n | 2 | onset/coda | word context | viseme + strength |
| i | 2 | nucleus | word context | viseme + strength |
| ... | ... | ... | ... | ... |

这里不在文档中手工伪造完整 phoneme/Viseme 标签；实际值必须由统一 Frontend 与训练样本生成 Pipeline 产生。

---

## 3. 长播报句与 Phrase Boundary

示例文本：

```text
根据最新公布的数据，今年上半年国内市场整体保持稳定增长，其中新能源、人工智能和高端制造等领域表现较为突出。
```

建议在训练样本中显式标记短语边界：

```text
根据最新公布的数据，
↑ phrase_end = true

今年上半年国内市场整体保持稳定增长，
↑ phrase_end = true

其中新能源、人工智能和高端制造等领域表现较为突出。
↑ sentence_end = true
```

phoneme token 的 boundary 示例：

```json
{
  "phoneme": "x",
  "boundary": {
    "word_start": false,
    "word_end": true,
    "phrase_start": false,
    "phrase_end": true,
    "boundary_type": "minor"
  }
}
```

这类样本用于评估模型是否学会：

```text
同一个 phoneme
+
不同 phrase position
+
不同前后 phoneme
        ↓
不同 Viseme probability / Strength
```

---

## 4. 多音字与上下文消歧样本

示例 A：

```text
这家银行今天正常营业。
```

示例 B：

```text
他们继续向前行走。
```

两个句子都包含：

```text
行
```

但 pronunciation 不同。

训练数据必须保存 **G2P 后的实际 phoneme sequence**，模型本身不负责从汉字判断多音字读音。

正确职责：

```text
Text
  ↓
G2P / Frontend
  ↓
正确 pronunciation
  ↓
ArticuLM
```

因此此类数据的价值主要是：

- phoneme context coverage
- pronunciation frontend integration test
- 相同字符在不同发音情况下的实际主播覆盖

而不是让 ArticuLM 做汉字多音字分类。

---

## 5. 中英混合样本

示例：

```text
新的AI模型将在GPU服务器上运行。
```

经过 frontend 后，中文 phoneme 使用：

```text
language = zh
surface_tone = 1..5
stress = 0
```

英文 token 使用：

```text
language = en
surface_tone = 0
stress = 0/1/2
```

同一句内允许 language 切换。

示意：

| token | language | surface_tone | stress |
|---|---|---:|---:|
| 新 | zh | 1 | 0 |
| 的 | zh | 5 | 0 |
| AI 的英文 phoneme | en | 0 | 1/0 |
| 模 | zh | 2 | 0 |
| 型 | zh | 2/3（以 frontend 为准） | 0 |

具体 tone / stress 必须以实际 Frontend 输出为准，不在训练代码中猜测。

---

## 6. 英文样本

示例：

```text
We can move.
```

概念 phoneme：

```text
/w i k ə n m u v/
```

英文 prosody 约定：

```text
surface_tone = 0
stress = 0 / 1 / 2
```

Stress 是 syllable-level feature，建议广播到该 syllable 的所有 phoneme。

结构示例：

```json
{
  "sample_id": "en_we_can_move_001",
  "text": "We can move.",
  "tokens": [
    {
      "phoneme": "m",
      "language": "en",
      "surface_tone": 0,
      "stress": 1,
      "syllable_role": "onset",
      "articulatory": {
        "type": "consonant",
        "place": "bilabial",
        "manner": "nasal",
        "voiced": true,
        "aspirated": false
      },
      "boundary": {
        "word_start": true,
        "word_end": false,
        "phrase_start": false,
        "phrase_end": false,
        "boundary_type": "none"
      },
      "labels": {
        "viseme_id": 8,
        "strength": 82.0,
        "viseme_source": "example",
        "strength_source": "pseudo_strength_v1"
      }
    }
  ]
}
```

上面的 label 仅用于展示 Schema；正式标签以实际训练数据生成器输出为准。

---

## 7. Synthetic 与 Human Gold 的同一样本

程序预训练阶段：

```json
{
  "phoneme": "a",
  "labels": {
    "viseme_id": 2,
    "strength": 86.0,
    "viseme_source": "website_rule",
    "strength_source": "pseudo_strength_v1"
  }
}
```

Human Gold 修正后可能变为：

```json
{
  "phoneme": "a",
  "labels": {
    "viseme_id": 2,
    "strength": 74.0,
    "viseme_source": "human",
    "strength_source": "human"
  }
}
```

因此训练代码必须允许按 `strength_source` 使用不同 Loss 权重。

例如：

```text
Pseudo Strength:
weight = 0.3

Human Gold Strength:
weight = 1.0
```

不要覆盖原始 label source 信息。

---

## 8. Batch Padding 示例

原始 sequence：

```text
Sample A: 4 phonemes
Sample B: 7 phonemes
```

Padding 后：

```text
A: p1 p2 p3 p4 PAD PAD PAD
B: p1 p2 p3 p4 p5  p6  p7
```

`attention_mask`：

```text
A: 1 1 1 1 0 0 0
B: 1 1 1 1 1 1 1
```

`loss_mask`：

```text
A: 1 1 1 1 0 0 0
B: 1 1 1 1 1 1 1
```

任何 PAD token 都不能参与：

- attention
- Viseme CE
- Strength Huber
- Accuracy / F1 / MAE

---

## 9. 建议的 Tiny Overfit 数据

为了验证模型实现，建议构造一个固定小数据集：

```text
64 sentences
```

覆盖：

- 18 个 Viseme 均至少出现若干次
- tone 1～5
- onset / nucleus / coda
- phrase_start / phrase_end
- 短句和中等长度句
- 数字句
- 至少几个多音字上下文
- 少量中英混合

Tiny Overfit 的目标不是验证泛化，而是验证：

```text
Dataset
→ Collator
→ Model
→ Loss
→ Backward
→ Optimizer
```

是否正确。

如果 50M 模型不能在这个小集合上强烈过拟合，应优先排查实现问题。

---

## 10. 推荐用于单元测试的固定样本

建议仓库增加：

```text
tests/fixtures/sample_zh.jsonl
tests/fixtures/sample_en.jsonl
tests/fixtures/sample_mixed.jsonl
```

至少覆盖：

### `sample_zh.jsonl`

```text
你好。
请稍等一下。
预计2026年销售额增长12.5%。
```

### `sample_en.jsonl`

```text
We can move.
```

### `sample_mixed.jsonl`

```text
新的AI模型将在GPU服务器上运行。
```

这些 fixture 应用于：

- schema validation
- vocab build
- dataset load
- collator
- forward shape
- loss masking
- inference serialization

---

## 11. 训练前数据检查示例

训练脚本启动前输出类似：

```text
Dataset Version:          v1.0
Sentences:                100,000
Phoneme Tokens:           3,240,816
Mean Seq Length:          32.4
P95 Seq Length:           71
Max Seq Length:           214
Unknown Phoneme Rate:     0.03%

Viseme Classes:
0: ...
1: ...
...
17: ...

Strength:
mean: ...
std: ...
p05: ...
p50: ...
p95: ...

Languages:
zh: 98.7%
en: 1.3%
```

这里的数值仅是报告格式示例，真实训练必须由 Dataset Validator 计算。

---

## 12. 不合格样本示例

### Label leakage

错误：

```json
{
  "features": {
    "phoneme": "a",
    "viseme_id": 2
  }
}
```

原因：

```text
viseme_id 是 target，不能作为 Encoder input。
```

### 把 raw value 当真实 GT

错误：

```json
{
  "strength": 100,
  "strength_source": "human"
}
```

但实际 100 来自网页 rule value。

应该：

```json
{
  "teacher_metadata": {
    "raw_value": 100
  },
  "labels": {
    "strength": 84.0,
    "strength_source": "pseudo_strength_v1"
  }
}
```

### Token / Label 数量不一致

错误：

```text
phonemes: 12
viseme labels: 11
```

这种样本必须在进入训练前 fail fast 或 reject，不能自动错位补齐。
