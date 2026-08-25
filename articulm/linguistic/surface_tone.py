"""Surface tone (变调) computation.

This is the canonical implementation shared by the corpus pipeline and the
ArticuLM training-sample generator. Rules implemented:

* third-tone sandhi: runs of tone-3 syllables become all-2 except the last
  (``3+3 -> 2+3``, ``3+3+3 -> 2+2+3``).
* 一 sandhi: ``一`` before a 4th tone -> 2; before 1/2/3 -> 4; after ``第``
  or at end -> 1.
* 不 sandhi: ``不`` before a 4th tone -> 2; otherwise -> 4.

.. note::
    This is an **engineering approximation**. A real TTS frontend's surface
    tone should eventually take priority over these rules.
"""

from __future__ import annotations

import re

from pypinyin import Style, pinyin

MAJOR_PUNC = set("。！？.!?")
MINOR_PUNC = set("，；：、,;:")
ALL_PUNC = MAJOR_PUNC | MINOR_PUNC

_TONE_RE = re.compile(r"([1-5])$")


def is_zh(ch: str) -> bool:
    return len(ch) == 1 and "一" <= ch <= "鿿"


def language_of(s: str) -> str:
    """Classify a token as ``zh`` / ``en`` / ``other``."""
    if s and all(is_zh(c) for c in s):
        return "zh"
    if re.fullmatch(r"[A-Za-z]+", s or ""):
        return "en"
    return "other"


def extract_tone(py: str | None) -> int | None:
    """Extract the trailing tone digit (1-5) from tone-numbered pinyin."""
    if not py:
        return None
    m = _TONE_RE.search(py)
    return int(m.group(1)) if m else None


def _pinyin_tones(text: str) -> list[list[str]]:
    return pinyin(
        text,
        style=Style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        errors=lambda x: list(x),
    )


def tone_info(text: str) -> list[dict]:
    """Return per-character tone info with lexical (base) and surface tones.

    ``一`` and ``不`` are pinned to their lexical tones (1 and 4) here because
    pypinyin applies dictionary-driven 一/不 sandhi inconsistently; our own rules
    then recompute the surface tone deterministically.
    """
    pys = _pinyin_tones(text)
    info: list[dict] = []
    for i, (ch, p) in enumerate(zip(text, pys)):
        t = extract_tone(p[0]) if is_zh(ch) else None
        if ch == "一":
            t = 1
        elif ch == "不":
            t = 4
        info.append({"i": i, "ch": ch, "base": t, "surface": t})
    return info


def _next_zh(info: list[dict], i: int) -> int | None:
    for j in range(i + 1, len(info)):
        ch = info[j]["ch"]
        if ch in ALL_PUNC or ch.isspace():
            return None
        if is_zh(ch):
            return j
    return None


def _prev_zh(info: list[dict], i: int) -> int | None:
    for j in range(i - 1, -1, -1):
        ch = info[j]["ch"]
        if ch in ALL_PUNC or ch.isspace():
            return None
        if is_zh(ch):
            return j
    return None


def build_surface_tones(
    text: str,
    *,
    sandhi_third_tone: bool = True,
    sandhi_yi: bool = True,
    sandhi_bu: bool = True,
) -> dict[int, int]:
    """Compute surface tones, keyed by character position in ``text``."""
    info = tone_info(text)

    if sandhi_yi:
        for i, x in enumerate(info):
            if x["ch"] != "一":
                continue
            p = _prev_zh(info, i)
            if p is not None and info[p]["ch"] == "第":
                x["surface"] = 1
                continue
            n = _next_zh(info, i)
            if n is None:
                x["surface"] = 1
            elif info[n]["base"] == 4:
                x["surface"] = 2
            elif info[n]["base"] in (1, 2, 3):
                x["surface"] = 4
            else:
                x["surface"] = 1

    if sandhi_bu:
        for i, x in enumerate(info):
            if x["ch"] != "不":
                continue
            n = _next_zh(info, i)
            x["surface"] = 2 if n is not None and info[n]["base"] == 4 else 4

    if sandhi_third_tone:
        i = 0
        while i < len(info):
            if not is_zh(info[i]["ch"]) or info[i]["surface"] != 3:
                i += 1
                continue
            run = [i]
            j = i + 1
            while j < len(info):
                ch = info[j]["ch"]
                if ch in ALL_PUNC or ch.isspace() or not is_zh(ch) or info[j]["surface"] != 3:
                    break
                run.append(j)
                j += 1
            for k in run[:-1]:
                info[k]["surface"] = 2
            i = max(j, i + 1)

    return {x["i"]: x["surface"] for x in info if is_zh(x["ch"])}


def surface_tone_sequence(
    text: str,
    *,
    sandhi_third_tone: bool = True,
    sandhi_yi: bool = True,
    sandhi_bu: bool = True,
) -> list[int]:
    """Surface tones of the sentence's han syllables, in reading order."""
    tones = build_surface_tones(
        text,
        sandhi_third_tone=sandhi_third_tone,
        sandhi_yi=sandhi_yi,
        sandhi_bu=sandhi_bu,
    )
    return [tones[i] for i in sorted(tones)]


def lexical_tone_sequence(text: str) -> list[int]:
    """Lexical tones of the sentence's han syllables, in reading order."""
    return [x["base"] for x in tone_info(text) if is_zh(x["ch"]) and x["base"] is not None]
