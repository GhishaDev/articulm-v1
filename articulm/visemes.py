"""Canonical viseme id -> name mapping for ArticuLM-V1.

The 18-class space is defined by ``tdoge_shape_map_v2`` (the dataset factory's
``teacher/viseme_shape_map.json``); the class order is frozen by the training
contract, so the names are a stable table in the model repo. Ids 16 and 17 were
added by the 16->18 expansion (``smile_closed`` sentence-final, ``~`` pause).
"""

from __future__ import annotations

VISEME_NAMES: tuple[str, ...] = (
    "301_cAt",
    "302_High",
    "303_Idea",
    "304_Out",
    "305_Dream",
    "307_aBAck",
    "308_Left",
    "309_Fill",
    "310_Weep",
    "312_Hot",
    "313_Samia",
    "314_RRed",
    "315_SHeep",
    "316_THin",
    "318_KAKA",
    "closed",
    "smile_closed",
    "~",
)


def viseme_name(viseme_id: int) -> str:
    """Return the shapeV2 name for a viseme id, or ``"<unknown>"`` if out of range."""
    if 0 <= viseme_id < len(VISEME_NAMES):
        return VISEME_NAMES[viseme_id]
    return f"<unknown:{viseme_id}>"
