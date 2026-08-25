# Input / Output Schema

## Sequence-level JSONL sample

```json
{
  "sample_id": "sample_001",
  "text": "你好，请稍等。",
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
        "strength": 62.0,
        "viseme_source": "rule",
        "strength_source": "pseudo_strength_v1"
      }
    }
  ]
}
```

Chinese:

```text
surface_tone ∈ {1,2,3,4,5}
stress = 0
language = zh
```

English:

```text
surface_tone = 0
stress ∈ {0,1,2}
language = en
```

Syllable role:

```text
onset / nucleus / coda / other / silence
```

Boundary:

```text
word_start
word_end
phrase_start
phrase_end
boundary_type ∈ {none, minor, major}
```

Labels:

```text
viseme_id: int 0..17
strength: float 0..100
```

Teacher metadata may retain:

```text
shapeV2
Talk
raw_value
timing
duration
```

but these are not encoder inputs.

Output example:

```json
{
  "text": "你好",
  "outputs": [
    {"phoneme": "n", "viseme_id": 14, "strength": 62.0},
    {"phoneme": "i", "viseme_id": 3, "strength": 70.0},
    {"phoneme": "x", "viseme_id": 15, "strength": 57.0},
    {"phoneme": "a", "viseme_id": 2, "strength": 90.0}
  ]
}
```
