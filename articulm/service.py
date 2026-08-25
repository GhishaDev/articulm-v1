"""Caller-facing inference service: tdoge viseme API response -> model output.

The tdoge viseme API already performs the g2p + alignment; this module only
needs the minimal per-phoneme grouping from its response (``ipa``,
``wordIndex``/``charIndex``, ``word``), reconstructs the exact encoder features
(language, surface tone, syllable role, articulatory, boundaries) and returns
the model's viseme + strength predictions in the same shape. Timing fields are
optional and echoed back only when the caller supplies them.

```python
from articulm.service import predict_api_response
from articulm.inference import ModelPredictor

predictor = ModelPredictor.load("archive/strength_v2_fast_20260824/model/best.pt")

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
# out["visemes"] == [{"ipa", "shapeV2", "strength", "word", "wordIndex",
#                     "charIndex"}, ...]
```

The predicted ``shapeV2`` is the 18-class name (e.g. ``"304_Out"``); ``strength``
is the 0-100 mouth-magnitude. Only baseline decoding — no smoothing or rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data.schema import (
    ArticulatoryFeatures,
    BoundaryFeatures,
    PhonemeToken,
    Sample,
)
from .inference import ModelPredictor
from .linguistic.articulation import articulation, assign_syllable_roles
from .linguistic.surface_tone import build_surface_tones, is_zh

_MAJOR_PUNC = set("。！？.!?")
_ALL_PUNC = set("，；：、,;:。！？.!?")


def _is_silence_char(word: str) -> bool:
    return all(ch in _ALL_PUNC or ch.isspace() for ch in (word or ""))


def _language(word: str) -> str:
    if not word:
        return "zh"
    if all(is_zh(ch) for ch in word if not ch.isspace() and ch not in _ALL_PUNC):
        return "zh"
    return "en"


@dataclass
class _Row:
    ipa: str
    char_index: int
    word_index: int
    word: str


def _extract_phonemes(response: dict[str, Any]) -> list[_Row]:
    visemes = response.get("visemes") or []
    rows = []
    for v in visemes:
        ipa = v.get("ipa")
        if ipa is None or ipa in ("", "-"):
            ipa = "_"  # punctuation / silence token
        rows.append(
            _Row(
                ipa=ipa,
                char_index=int(v.get("charIndex", 0)),
                word_index=int(v.get("wordIndex", 0)),
                word=v.get("word") or "",
            )
        )
    if not rows:
        raise ValueError("response contains no visemes")
    return rows


def _build_sample(rows: list[_Row], text: str) -> Sample:
    """Reconstruct the encoder's feature-complete token stream."""
    tones = build_surface_tones(text) if any(is_zh(c) for c in text) else {}
    char_index_of: dict[int, int] = {}  # char position in text -> sequential index

    enriched: list[dict[str, Any]] = []
    for pos, r in enumerate(rows):
        # Group key for syllable-role assignment = (char, occurrence) so a
        # repeated char (e.g. the two 你) is not merged into one syllable.
        key = (r.char_index, sum(1 for x in rows[:pos] if x.char_index == r.char_index))
        lang = _language(r.word)
        is_silence = _is_silence_char(r.word)
        tone = 5 if (lang == "zh" and is_silence) else 0
        if lang == "zh" and not is_silence:
            tone = tones.get(r.char_index, 5)
        enriched.append(
            {
                "phoneme": r.ipa,
                "language": lang,
                "surface_tone": tone,
                "stress": 0,
                "_syll_key": key,
                "_word_index": r.word_index,
                "_char_index": r.char_index,
                "_word": r.word,
            }
        )

    assign_syllable_roles(enriched, group_by="_syll_key")

    # word / phrase boundaries from the API's own grouping and punctuation.
    word_ids = [row["_word_index"] for row in enriched]
    punct_positions = [i for i, row in enumerate(enriched) if _is_silence_char(row["_word"])]
    phrase_end = set(punct_positions)  # the silence token itself is phrase-final
    phrase_start: set[int] = set()
    for p in punct_positions:
        if p + 1 < len(enriched):
            phrase_start.add(p + 1)
    if enriched:
        phrase_end.add(len(enriched) - 1)  # sentence final

    tokens = []
    for i, row in enumerate(enriched):
        a = articulation(row["phoneme"])
        feat = ArticulatoryFeatures(
            type=a["type"], height=a["height"], backness=a["backness"],
            rounded=a["rounded"], place=a["place"], manner=a["manner"],
            voiced=a["voiced"], aspirated=a["aspirated"],
        )
        b = BoundaryFeatures(
            word_start="true" if (i == 0 or word_ids[i] != word_ids[i - 1]) else "false",
            word_end="true" if (i == len(enriched) - 1 or word_ids[i] != word_ids[i + 1]) else "false",
            phrase_start="true" if i in phrase_start else "false",
            phrase_end="true" if i in phrase_end else "false",
            boundary_type=(
                "major"
                if row["_word"] and all(c in _MAJOR_PUNC for c in row["_word"])
                else ("minor" if i in punct_positions else "none")
            ),
        )
        tokens.append(
            PhonemeToken(
                phoneme=row["phoneme"],
                language=row["language"],
                surface_tone=row["surface_tone"],
                stress=row["stress"],
                syllable_role=row["syllable_role"],
                articulatory=feat,
                boundary=b,
                labels=None,
            )
        )

    return Sample(sample_id="api", text=text, tokens=tuple(tokens))


def predict_api_response(
    response: dict[str, Any],
    predictor: ModelPredictor,
    *,
    text: str | None = None,
) -> dict[str, Any]:
    """Predict viseme + strength for a tdoge viseme API response.

    Returns ``{"text", "visemes": [...]}`` mirroring the input's flat
    ``visemes`` list, with each entry carrying the model's ``shapeV2``
    (18-class name) and ``strength`` alongside the original timing/indices.
    """
    source_text = text or response.get("spokenText") or response.get("text") or ""
    rows = _extract_phonemes(response)
    sample = _build_sample(rows, source_text)

    result = predictor.predict_samples([sample])[0]
    outputs = result["outputs"]

    visemes = []
    for i, row in enumerate(rows):
        out = outputs[i]
        src = (response.get("visemes") or [])[i] if i < len(response.get("visemes") or []) else {}
        entry: dict[str, Any] = {
            "ipa": row.ipa if row.ipa != "_" else "-",
            "shapeV2": out["viseme"],
            "strength": out["strength"],
        }
        # Echo back the caller's grouping fields (word/wordIndex/charIndex), and
        # timing only when the caller supplied it — the request may be minimal.
        if "word" in src:
            entry["word"] = src["word"]
        if "wordIndex" in src:
            entry["wordIndex"] = src["wordIndex"]
        if "charIndex" in src:
            entry["charIndex"] = src["charIndex"]
        if "startPercent" in src:
            entry["startPercent"] = src["startPercent"]
        if "endPercent" in src:
            entry["endPercent"] = src["endPercent"]
        visemes.append(entry)

    return {"text": source_text, "visemes": visemes}
