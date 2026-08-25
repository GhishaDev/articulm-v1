"""Articulatory feature tables and pseudo-strength rules.

Extracted from the original ``generate_articulm_training_samples.py`` so the
corpus pipeline and the downstream ArticuLM sample generator share one phoneme
inventory and one articulation / syllable-role / pseudo-strength implementation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

ALIASES = {
    "ph": "pʰ", "th": "tʰ", "kh": "kʰ", "tsh": "tsʰ", "tɕh": "tɕʰ", "tʂh": "tʂʰ",
    # the tdoge teacher API emits plain "r" for the retroflex approximant
    "r": "ɹ",
    # and U+0261 (IPA "g") for the voiced velar stop, as opposed to ASCII "g"
    "ɡ": "g",
    # the Chinese voice transcribes English /k/ as ``c`` (palatal) / ``q``
    # (uvular); both are the same back stop for viseme purposes
    "c": "k", "q": "k",
    # and the "ch" affricate as the U+02A7 ligature instead of "t"+"ʃ"
    "ʧ": "tʃ",
}

# Non-phoneme symbols the teacher API emits: sentence-final rest (``∅``), pauses
# at punctuation (``-``/``_``), the length mark (``ː``), the liaison marker
# (``‿``) and standalone stress marks (``ˈ``/``ˌ``). These are real mouth states
# (or prosodic annotations) for an avatar model, so they are typed ``silence``
# rather than ``unknown`` — ``unknown`` must stay meaningful as "phoneme we
# failed to model".
SILENCE = {"∅", "-", "_", "ː", "‿", "ˈ", "ˌ"}

VOWELS: dict[str, tuple[str, str, bool]] = {
    "i": ("high", "front", False),
    "y": ("high", "front", True),
    "u": ("high", "back", True),
    "ɿ": ("high", "central", False),
    "ʅ": ("high", "central", False),
    "e": ("mid", "front", False),
    "ɛ": ("mid", "front", False),
    "ə": ("mid", "central", False),
    "ɤ": ("mid", "back", False),
    "o": ("mid", "back", True),
    "ɔ": ("mid", "back", True),
    "ɚ": ("mid", "central", False),
    "a": ("low", "central", False),
    "ɑ": ("low", "back", False),
    "ɐ": ("low", "central", False),
    # English vowels
    "ɪ": ("high", "front", False),
    "ʊ": ("high", "back", True),
    "æ": ("low", "front", False),
    "ʌ": ("mid", "central", False),
    "ɜ": ("mid", "central", False),
    # Diphthongs emitted as single tokens by the teacher API. Height/backness
    # describe the *onset* target, which is what drives the visible mouth shape.
    "eɪ": ("mid", "front", False),
    "aɪ": ("low", "central", False),
    "aʊ": ("low", "central", False),
    "oʊ": ("mid", "back", True),
    "ɔɪ": ("mid", "back", True),
    # open back rounded vowel (British "lot/cloth") the Chinese voice emits
    "ɒ": ("low", "back", True),
}

CONS: dict[str, tuple[str, str, bool, bool]] = {
    "p": ("bilabial", "stop", False, False),
    "pʰ": ("bilabial", "stop", False, True),
    "b": ("bilabial", "stop", True, False),
    "m": ("bilabial", "nasal", True, False),
    "f": ("labiodental", "fricative", False, False),
    "v": ("labiodental", "fricative", True, False),
    "t": ("alveolar", "stop", False, False),
    "tʰ": ("alveolar", "stop", False, True),
    "d": ("alveolar", "stop", True, False),
    "n": ("alveolar", "nasal", True, False),
    "l": ("alveolar", "lateral", True, False),
    "s": ("alveolar", "fricative", False, False),
    "z": ("alveolar", "fricative", True, False),
    "ts": ("alveolar", "affricate", False, False),
    "tsʰ": ("alveolar", "affricate", False, True),
    "tʂ": ("retroflex", "affricate", False, False),
    "tʂʰ": ("retroflex", "affricate", False, True),
    "ʂ": ("retroflex", "fricative", False, False),
    "ʐ": ("retroflex", "fricative", True, False),
    "ɻ": ("retroflex", "approximant", True, False),
    "tɕ": ("alveolopalatal", "affricate", False, False),
    "tɕʰ": ("alveolopalatal", "affricate", False, True),
    "ɕ": ("alveolopalatal", "fricative", False, False),
    "k": ("velar", "stop", False, False),
    "kʰ": ("velar", "stop", False, True),
    "g": ("velar", "stop", True, False),
    "ŋ": ("velar", "nasal", True, False),
    "x": ("velar", "fricative", False, False),
    "h": ("glottal", "fricative", False, False),
    "j": ("palatal", "approximant", True, False),
    "w": ("labiovelar", "approximant", True, False),
    "ɥ": ("labiopalatal", "approximant", True, False),
    # English consonants
    "θ": ("dental", "fricative", False, False),
    "ð": ("dental", "fricative", True, False),
    "ʃ": ("postalveolar", "fricative", False, False),
    "ʒ": ("postalveolar", "fricative", True, False),
    "tʃ": ("postalveolar", "affricate", False, False),
    "dʒ": ("postalveolar", "affricate", True, False),
    "ɹ": ("alveolar", "approximant", True, False),
}

CODAS = {"n", "ŋ", "ɹ"}

_TONE_STRESS_RE = re.compile(r"([012])$")


def norm_phoneme(raw: str | None) -> str:
    p = (raw or "").strip()
    stripped = p.replace("ˈ", "").replace("ˌ", "")
    if re.fullmatch(r"[A-Za-z]+[012]", stripped):
        stripped = stripped[:-1]
    result = ALIASES.get(stripped, stripped)
    # preserve a standalone stress mark (otherwise it collapses to "" and reads
    # as "unknown" rather than the prosodic marker it is)
    return result if result else p


def articulation(raw: str | None) -> dict[str, Any]:
    p = norm_phoneme(raw)
    if p in SILENCE:
        return {
            "type": "silence", "height": None, "backness": None, "rounded": None,
            "place": None, "manner": None, "voiced": False, "aspirated": False,
        }
    if p in VOWELS:
        h, b, r = VOWELS[p]
        return {
            "type": "vowel", "height": h, "backness": b, "rounded": r,
            "place": None, "manner": None, "voiced": True, "aspirated": False,
        }
    if p in CONS:
        place, manner, voiced, aspirated = CONS[p]
        return {
            "type": "consonant", "height": None, "backness": None, "rounded": None,
            "place": place, "manner": manner, "voiced": voiced, "aspirated": aspirated,
        }
    return {
        "type": "unknown", "height": None, "backness": None, "rounded": None,
        "place": None, "manner": None, "voiced": None, "aspirated": None,
    }


def assign_syllable_roles(rows: list[dict], group_by: str = "text_position") -> None:
    """Assign ``syllable_role`` (onset/nucleus/coda/other) to phoneme rows.

    Rows are grouped into syllables by ``group_by`` (``text_position`` for
    Chinese, ``syllable_id`` for English).
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get(group_by)
        if key is not None:
            groups[key].append(r)

    for group in groups.values():
        vowels = [i for i, r in enumerate(group) if norm_phoneme(r["phoneme"]) in VOWELS]
        for i, r in enumerate(group):
            r["phoneme_index_in_syllable"] = i
            r["phoneme_count_in_syllable"] = len(group)
        if all(norm_phoneme(r["phoneme"]) in SILENCE for r in group):
            for r in group:
                r["syllable_role"] = "silence"
            continue
        if not vowels:
            for i, r in enumerate(group):
                r["syllable_role"] = "onset" if i == 0 else "other"
            continue
        first_v, last_v = vowels[0], vowels[-1]
        for i, r in enumerate(group):
            p = norm_phoneme(r["phoneme"])
            if i < first_v:
                r["syllable_role"] = "onset"
            elif first_v <= i <= last_v:
                r["syllable_role"] = "nucleus"
            elif p in CONS:
                # any consonant after the final vowel is a coda (covers English
                # codas beyond Chinese n/ŋ, e.g. stop/bath/fish)
                r["syllable_role"] = "coda"
            else:
                r["syllable_role"] = "other"


STRENGTH_RULES = ("pseudo_strength_v1", "pseudo_strength_v2")


def pseudo_strength(
    token: dict,
    prev_token: dict | None = None,
    next_token: dict | None = None,
    *,
    rule: str = "pseudo_strength_v1",
) -> float:
    """Programmatic pseudo-strength prior (not human ground truth).

    ``rule`` selects the rule version (see STRENGTH_RULES); the default keeps
    v1 semantics byte-for-byte:

    * v1 - the stress multiplier applies to every token carrying stress 0/1/2.
      Chinese tokens hold ``stress=0`` by contract, so they are all scaled by
      the unstressed factor 0.92 (an implicit "Chinese coefficient").
    * v2 - stress is an English-only feature; the zh ``stress=0`` is a contract
      filler, not "unstressed", so Chinese tokens get no stress multiplier.
      English values are identical to v1.
    """
    if rule not in STRENGTH_RULES:
        raise ValueError(f"unknown strength rule: {rule!r} (expected one of {STRENGTH_RULES})")
    f = token["articulatory"]
    typ, role = f["type"], token.get("syllable_role")

    # A pause / rest position has no articulatory effort by definition.
    if typ == "silence":
        return 0.0

    if typ == "vowel":
        s = {"low": 88.0, "mid": 78.0, "high": 68.0}.get(f["height"], 72.0)
        if f["rounded"]:
            s += 4.0
    elif typ == "consonant":
        s = {
            "stop": 74.0, "affricate": 70.0, "fricative": 60.0,
            "nasal": 62.0, "lateral": 56.0, "approximant": 50.0,
        }.get(f["manner"], 58.0)
        if f["place"] == "bilabial":
            s += 14.0
        elif f["place"] == "labiodental":
            s += 6.0
    else:
        s = 55.0

    if role == "nucleus":
        s += 5.0
    elif role == "coda":
        s -= 4.0

    tone_factor = {1: 1.00, 2: 1.02, 3: 0.96, 4: 1.04, 5: 0.86}
    if token.get("surface_tone") in tone_factor:
        s *= tone_factor[token["surface_tone"]]

    stress = token.get("stress")
    # v2: stress modulates English tokens only (zh stress is a contract filler).
    stress_applies = rule == "pseudo_strength_v1" or token.get("language") == "en"
    if stress_applies:
        if stress == 1:
            s *= 1.06
        elif stress == 2:
            s *= 1.03
        elif stress == 0:
            s *= 0.92

    if token.get("word_start"):
        s += 1.5
    if token.get("word_end"):
        s -= 1.0
    if token.get("phrase_end"):
        s *= 0.92

    if prev_token and typ == "vowel" and prev_token["articulatory"].get("place") == "bilabial":
        s += 2.0
    if (prev_token and next_token and typ == "consonant"
            and prev_token["articulatory"]["type"] == "vowel"
            and next_token["articulatory"]["type"] == "vowel"):
        s += 2.0
    if f.get("manner") == "approximant":
        s -= 3.0

    return round(max(0.0, min(100.0, s)), 1)
