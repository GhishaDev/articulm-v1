"""Illustrative phoneme corpus builder for fixtures and tiny-overfit data.

THIS IS DEVELOPMENT TOOLING, NOT PRODUCTION LOGIC.

Everything here — the phoneme inventory, the articulatory table, the
phoneme->viseme map and the strength formula — is *illustrative*. It exists
only so the repository has deterministic, schema-valid JSONL to exercise the
Dataset / Collator / model / loss / metric code paths.

Hard rules honoured by this file:

* No value produced here is Human Gold. Every emitted label carries
  ``viseme_source="fixture_rule"`` and ``strength_source="pseudo_strength_v1"``.
* Real training data must come from ``articulm_data_pipeline``; nothing in
  ``articulm/`` imports this module.
* The one exception to "illustrative" is :func:`doc_canonical_zh_nihao`,
  which reproduces the ``你好。`` sample from ``docs/12_training_sample_examples.md``
  verbatim so tests can pin the documented schema.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from articulm import SCHEMA_VERSION

# --------------------------------------------------------------------------
# Phoneme inventory (illustrative)
# --------------------------------------------------------------------------

# Mandarin initials, longest-first so greedy matching works.
ZH_INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)
# Mandarin rime segments, longest-first.
ZH_RIME_SEGMENTS = ("ng", "er", "a", "o", "e", "i", "u", "v", "n", "r")

VOWELS = frozenset(
    {"a", "o", "e", "i", "u", "v", "er", "iy", "uw", "ah", "ey", "ay"}
)

# phoneme -> articulatory description. `None` maps to `[NA]` downstream.
ARTICULATORY: dict[str, dict[str, Any]] = {
    # -- Mandarin consonants ------------------------------------------------
    "b": {"place": "bilabial", "manner": "plosive", "voiced": False, "aspirated": False},
    "p": {"place": "bilabial", "manner": "plosive", "voiced": False, "aspirated": True},
    "m": {"place": "bilabial", "manner": "nasal", "voiced": True, "aspirated": False},
    "f": {"place": "labiodental", "manner": "fricative", "voiced": False, "aspirated": False},
    "d": {"place": "alveolar", "manner": "plosive", "voiced": False, "aspirated": False},
    "t": {"place": "alveolar", "manner": "plosive", "voiced": False, "aspirated": True},
    "n": {"place": "alveolar", "manner": "nasal", "voiced": True, "aspirated": False},
    "l": {"place": "alveolar", "manner": "lateral", "voiced": True, "aspirated": False},
    "g": {"place": "velar", "manner": "plosive", "voiced": False, "aspirated": False},
    "k": {"place": "velar", "manner": "plosive", "voiced": False, "aspirated": True},
    "h": {"place": "velar", "manner": "fricative", "voiced": False, "aspirated": False},
    "j": {"place": "alveolo-palatal", "manner": "affricate", "voiced": False, "aspirated": False},
    "q": {"place": "alveolo-palatal", "manner": "affricate", "voiced": False, "aspirated": True},
    "x": {"place": "alveolo-palatal", "manner": "fricative", "voiced": False, "aspirated": False},
    "zh": {"place": "retroflex", "manner": "affricate", "voiced": False, "aspirated": False},
    "ch": {"place": "retroflex", "manner": "affricate", "voiced": False, "aspirated": True},
    "sh": {"place": "retroflex", "manner": "fricative", "voiced": False, "aspirated": False},
    "r": {"place": "retroflex", "manner": "approximant", "voiced": True, "aspirated": False},
    "z": {"place": "alveolar", "manner": "affricate", "voiced": False, "aspirated": False},
    "c": {"place": "alveolar", "manner": "affricate", "voiced": False, "aspirated": True},
    "s": {"place": "alveolar", "manner": "fricative", "voiced": False, "aspirated": False},
    "y": {"place": "palatal", "manner": "glide", "voiced": True, "aspirated": False},
    "w": {"place": "bilabial", "manner": "glide", "voiced": True, "aspirated": False},
    "ng": {"place": "velar", "manner": "nasal", "voiced": True, "aspirated": False},
    # -- English consonants -------------------------------------------------
    "jh": {"place": "postalveolar", "manner": "affricate", "voiced": True, "aspirated": False},
    "v": {"place": "labiodental", "manner": "fricative", "voiced": True, "aspirated": False},
    # -- Vowels -------------------------------------------------------------
    "a": {"height": "low", "backness": "central", "rounded": False},
    "o": {"height": "mid", "backness": "back", "rounded": True},
    "e": {"height": "mid", "backness": "central", "rounded": False},
    "i": {"height": "high", "backness": "front", "rounded": False},
    "u": {"height": "high", "backness": "back", "rounded": True},
    "er": {"height": "mid", "backness": "central", "rounded": False},
    "iy": {"height": "high", "backness": "front", "rounded": False},
    "uw": {"height": "high", "backness": "back", "rounded": True},
    "ah": {"height": "mid", "backness": "central", "rounded": False},
    "ey": {"height": "mid", "backness": "front", "rounded": False},
    "ay": {"height": "low", "backness": "central", "rounded": False},
}
# `v` is ü in Mandarin and /v/ in English; disambiguated by language below.
ZH_V_VOWEL = {"height": "high", "backness": "front", "rounded": True}

# Illustrative phoneme -> viseme map. Anchored on the examples in
# docs/12 (i->3, a->2, n->14, m->8) and extended so all 18 classes appear.
# ids 16/17 stand in for the teacher's sentence-final `smile_closed` / pause
# classes in this illustrative map (real semantics live in the data pipeline).
VISEME_MAP: dict[str, int] = {
    "e": 0,
    "er": 1,
    "a": 2, "ah": 2, "ay": 2,
    "i": 3, "iy": 3,
    "o": 4,
    "u": 5, "uw": 5,
    "ey": 7,
    "m": 8, "b": 8, "p": 8,
    "f": 9,
    "w": 10,
    "y": 11,
    "g": 12, "k": 12,
    "jh": 13,
    "n": 14, "ng": 14, "l": 14, "d": 14, "t": 14,
    "x": 15, "h": 15, "j": 15, "q": 15,
    "sh": 15, "zh": 15, "ch": 15, "s": 15, "c": 15,
    "z": 17, "r": 16,
}
V_VISEME_ZH = 6  # ü
V_VISEME_EN = 9  # /v/, groups with the labiodental fricative f


class CorpusError(ValueError):
    pass


# --------------------------------------------------------------------------
# Syllable -> phoneme segmentation
# --------------------------------------------------------------------------


def split_mandarin_syllable(syllable: str) -> tuple[str | None, list[str]]:
    """Split a pinyin syllable into (initial, rime segments)."""
    rest = syllable.lower().replace("ü", "v")
    initial: str | None = None
    for candidate in ZH_INITIALS:
        if rest.startswith(candidate) and len(rest) > len(candidate):
            initial = candidate
            rest = rest[len(candidate) :]
            break

    segments: list[str] = []
    while rest:
        for candidate in ZH_RIME_SEGMENTS:
            if rest.startswith(candidate):
                segments.append(candidate)
                rest = rest[len(candidate) :]
                break
        else:
            raise CorpusError(f"cannot segment rime remainder {rest!r} of {syllable!r}")

    if not segments:
        raise CorpusError(f"syllable {syllable!r} produced no vowel segments")
    return initial, segments


def _articulatory(phoneme: str, language: str) -> dict[str, Any]:
    if phoneme == "v":
        base = dict(ZH_V_VOWEL) if language == "zh" else dict(ARTICULATORY["v"])
    else:
        base = dict(ARTICULATORY[phoneme])
    is_vowel = phoneme in VOWELS or (phoneme == "v" and language == "zh")
    out: dict[str, Any] = {
        "type": "vowel" if is_vowel else "consonant",
        "height": base.get("height"),
        "backness": base.get("backness"),
        "rounded": base.get("rounded"),
        "place": base.get("place"),
        "manner": base.get("manner"),
        "voiced": base.get("voiced", True),
        "aspirated": base.get("aspirated", False),
    }
    return out


def _viseme_id(phoneme: str, language: str) -> int:
    if phoneme == "v":
        return V_VISEME_ZH if language == "zh" else V_VISEME_EN
    try:
        return VISEME_MAP[phoneme]
    except KeyError as exc:
        raise CorpusError(f"no illustrative viseme for phoneme {phoneme!r}") from exc


_ROLE_BASE = {"onset": 58.0, "nucleus": 76.0, "coda": 50.0, "other": 62.0}


def _pseudo_strength(
    phoneme: str,
    role: str,
    surface_tone: int,
    stress: int,
    phrase_end: bool,
    viseme_id: int,
) -> float:
    """Deterministic illustrative strength prior in [0,100].

    Not Human Gold. Not a validated perceptual model. Its only job is to give
    the Strength Head a learnable, context-dependent target in fixtures.
    """
    value = _ROLE_BASE.get(role, 62.0)
    value += {1: 4.0, 2: 2.0, 3: -3.0, 4: 6.0, 5: -8.0}.get(surface_tone, 0.0)
    value += {0: 0.0, 1: 7.0, 2: 3.0}.get(stress, 0.0)
    if phrase_end:
        value -= 9.0
    # Stable per-phoneme jitter (crc32 is deterministic across processes,
    # unlike hash()).
    seed = zlib.crc32(f"{phoneme}|{role}|{viseme_id}".encode())
    value += (seed % 900) / 100.0 - 4.5
    return round(min(100.0, max(0.0, value)), 1)


# --------------------------------------------------------------------------
# Word / sentence assembly
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    """One orthographic word plus its pronunciation units.

    For ``language="zh"`` units are ``(pinyin_syllable, surface_tone)``.
    For ``language="en"`` units are ``(phoneme, stress)``.
    """

    text: str
    language: str
    units: tuple[tuple[str, int], ...]


def zh(text: str, syllables: tuple[tuple[str, int], ...]) -> Word:
    return Word(text=text, language="zh", units=syllables)


def en(text: str, phonemes: tuple[tuple[str, int], ...]) -> Word:
    return Word(text=text, language="en", units=phonemes)


def zh_word_tokens(word: Word) -> list[dict[str, Any]]:
    """Expand one Chinese word into partially-filled token dicts."""
    syllables = word.units
    tokens: list[dict[str, Any]] = []
    for syllable, tone in syllables:
        initial, segments = split_mandarin_syllable(syllable)
        phonemes: list[tuple[str, str]] = []
        if initial:
            phonemes.append((initial, "onset"))
        for index, segment in enumerate(segments):
            phonemes.append((segment, "nucleus" if index == 0 else "coda"))
        for phoneme, role in phonemes:
            tokens.append(
                {
                    "phoneme": phoneme,
                    "language": "zh",
                    "surface_tone": tone,
                    "stress": 0,
                    "syllable_role": role,
                }
            )
    return tokens


def en_word_tokens(word: Word) -> list[dict[str, Any]]:
    """Expand one English word into partially-filled token dicts."""
    phonemes = word.units
    tokens: list[dict[str, Any]] = []
    for index, (phoneme, stress) in enumerate(phonemes):
        is_vowel = phoneme in VOWELS
        if is_vowel:
            role = "nucleus"
        else:
            next_is_vowel = (
                index + 1 < len(phonemes) and phonemes[index + 1][0] in VOWELS
            )
            role = "onset" if next_is_vowel else "coda"
        tokens.append(
            {
                "phoneme": phoneme,
                "language": "en",
                "surface_tone": 0,
                "stress": stress,
                "syllable_role": role,
            }
        )
    return tokens


def build_sample(
    sample_id: str,
    text: str,
    phrases: Iterable[Iterable[Word]],
    *,
    normalized_text: str | None = None,
) -> dict[str, Any]:
    """Assemble a schema-valid sample from phrases of words.

    ``phrases`` is a list of phrases; each phrase is a list of words. Word and
    phrase boundaries are derived from that nesting, so boundary features stay
    consistent with the text structure instead of being hand-typed.
    """
    phrase_list = [list(p) for p in phrases]
    if not phrase_list or not any(phrase_list):
        raise CorpusError(f"{sample_id}: no words provided")

    tokens: list[dict[str, Any]] = []
    for phrase_index, phrase in enumerate(phrase_list):
        is_last_phrase = phrase_index == len(phrase_list) - 1
        for word_index, word in enumerate(phrase):
            if word.language == "en":
                word_tokens = en_word_tokens(word)
            elif word.language == "zh":
                word_tokens = zh_word_tokens(word)
            else:
                raise CorpusError(f"{sample_id}: unsupported language {word.language!r}")
            if not word_tokens:
                raise CorpusError(f"{sample_id}: word {word.text!r} produced no tokens")

            first_word = word_index == 0
            last_word = word_index == len(phrase) - 1
            for token_index, token in enumerate(word_tokens):
                at_word_start = token_index == 0
                at_word_end = token_index == len(word_tokens) - 1
                phrase_start = first_word and at_word_start
                phrase_end = last_word and at_word_end
                boundary_type = (
                    ("major" if is_last_phrase else "minor") if phrase_end else "none"
                )
                token["boundary"] = {
                    "word_start": at_word_start,
                    "word_end": at_word_end,
                    "phrase_start": phrase_start,
                    "phrase_end": phrase_end,
                    "boundary_type": boundary_type,
                }
                tokens.append(token)

    for token in tokens:
        language = token["language"]
        phoneme = token["phoneme"]
        token["articulatory"] = _articulatory(phoneme, language)
        viseme_id = _viseme_id(phoneme, language)
        token["labels"] = {
            "viseme_id": viseme_id,
            "strength": _pseudo_strength(
                phoneme,
                token["syllable_role"],
                token["surface_tone"],
                token["stress"],
                token["boundary"]["phrase_end"],
                viseme_id,
            ),
            "viseme_source": "fixture_rule",
            "strength_source": "pseudo_strength_v1",
        }

    sample: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "text": text,
        "tokens": tokens,
    }
    if normalized_text is not None:
        sample["normalized_text"] = normalized_text
        sample["normalization_version"] = "fixture_tn_v1"
    return sample


# --------------------------------------------------------------------------
# Canonical documented sample
# --------------------------------------------------------------------------


def doc_canonical_zh_nihao() -> dict[str, Any]:
    """The ``你好。`` sample transcribed verbatim from docs/12.

    Label values here are the documentation's illustrative numbers. They are
    pinned only so schema tests assert against the documented shape.
    """
    return {
        "schema_version": SCHEMA_VERSION,
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
                    "height": None,
                    "backness": None,
                    "rounded": None,
                    "place": "alveolar",
                    "manner": "nasal",
                    "voiced": True,
                    "aspirated": False,
                },
                "boundary": {
                    "word_start": True,
                    "word_end": False,
                    "phrase_start": True,
                    "phrase_end": False,
                    "boundary_type": "none",
                },
                "labels": {
                    "viseme_id": 14,
                    "strength": 64.7,
                    "viseme_source": "website_rule",
                    "strength_source": "pseudo_strength_v1",
                },
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
                    "rounded": False,
                    "place": None,
                    "manner": None,
                    "voiced": True,
                    "aspirated": False,
                },
                "boundary": {
                    "word_start": False,
                    "word_end": True,
                    "phrase_start": False,
                    "phrase_end": False,
                    "boundary_type": "none",
                },
                "labels": {
                    "viseme_id": 3,
                    "strength": 76.0,
                    "viseme_source": "website_rule",
                    "strength_source": "pseudo_strength_v1",
                },
            },
            {
                "phoneme": "x",
                "language": "zh",
                "surface_tone": 3,
                "stress": 0,
                "syllable_role": "onset",
                "articulatory": {
                    "type": "consonant",
                    "height": None,
                    "backness": None,
                    "rounded": None,
                    "place": "velar",
                    "manner": "fricative",
                    "voiced": False,
                    "aspirated": False,
                },
                "boundary": {
                    "word_start": True,
                    "word_end": False,
                    "phrase_start": False,
                    "phrase_end": False,
                    "boundary_type": "none",
                },
                "labels": {
                    "viseme_id": 15,
                    "strength": 58.0,
                    "viseme_source": "website_rule",
                    "strength_source": "pseudo_strength_v1",
                },
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
                    "rounded": False,
                    "place": None,
                    "manner": None,
                    "voiced": True,
                    "aspirated": False,
                },
                "boundary": {
                    "word_start": False,
                    "word_end": True,
                    "phrase_start": False,
                    "phrase_end": True,
                    "boundary_type": "major",
                },
                "labels": {
                    "viseme_id": 2,
                    "strength": 84.0,
                    "viseme_source": "website_rule",
                    "strength_source": "pseudo_strength_v1",
                },
            },
        ],
    }
