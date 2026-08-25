# Training Data Specification

Training unit:

```text
one sentence / utterance = one phoneme sequence sample
```

Do not split phonemes into independent rows without context.

Required per-token fields:

```text
phoneme
language
surface_tone
stress
syllable_role
articulatory
boundary
viseme_id
strength
```

Label sources:

- rule/teacher Viseme: suitable for large-scale pretraining
- pseudo Strength: programmatic prior, not real GT
- Human Gold: final correction, validation and product evaluation

Recommended synthetic split:

```text
train 90%
validation 5%
test 5%
```

Split at sentence level after deduplication. Prevent near-duplicate leakage across splits.

Human Gold:

- fixed held-out test set
- never tune pseudo rules on the final test set
- report separately from synthetic test

Suggested milestones:

```text
10k–50k sentences: engineering validation
100k+ sentences: first synthetic-pretraining baseline
300k–1M+: scaling stage
```

Before each run report:

- sentence count
- phoneme-token count
- mean/p95 sequence length
- per-viseme class distribution
- Strength histogram
- per-viseme Strength histogram
- language ratio
- tone/stress distribution
- boundary distribution
- unknown phoneme rate

Validation:

- token/label counts match
- viseme in 0..17
- strength in 0..100
- no NaN/Inf
- known categorical values or [UNK]
- padding alignment correct
