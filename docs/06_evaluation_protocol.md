# Evaluation Protocol

Use three levels:

## Level A — Token metrics

- Viseme Accuracy
- Viseme Macro-F1
- Strength MAE / RMSE

## Level B — Sequence/context metrics

Break down by:

- phoneme context
- tone context
- word/phrase boundary
- sentence position
- sequence length

## Level C — Virtual-anchor perceptual evaluation

Render fixed scripts and evaluate:

- lip shape
- coarticulation transition
- motion Strength
- overall naturalness

Maintain fixed sets:

```text
synthetic_validation
synthetic_test
human_gold_validation
human_gold_test
perceptual_eval_script
```

Important difficult slices:

- third-tone sequences
- 一 / 不 sandhi
- polyphonic characters
- numbers / dates / amounts
- named entities
- long broadcast phrases
- phrase-final phonemes
- mixed Chinese/English if enabled

A model must not be accepted solely because synthetic accuracy increased.
