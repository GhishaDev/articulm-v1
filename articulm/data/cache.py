"""Disk cache for eagerly encoded datasets.

Encoding 6M phoneme tokens in pure Python takes ~20 minutes; on a machine with
unreliable GPUs (see reports/training_report_v2_strength.md §4.0) every restart
pays that cost again before reaching the first training step. This module
persists the encoded tensors next to the source JSONL so only the first run
encodes, and any later run with an unchanged corpus/vocabulary loads them.

The cache key covers everything the encoding depends on: source file identity
(path, size, mtime_ns, sample count), the full vocabulary fingerprint, the
strength source weights and the strength scale. A key mismatch is a cache miss,
never a stale read.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .dataset import EncodedSample

CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheKeyInput:
    source_path: Path
    num_samples: int
    vocab_fingerprint: str
    source_weights: tuple[tuple[str, float], ...]
    strength_scale: float


def vocabulary_fingerprint(vocab: Any) -> str:
    """Stable digest of a FeatureVocabulary's full mapping tables."""
    payload = json.dumps(vocab.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _source_file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def compute_cache_key(item: CacheKeyInput) -> str:
    parts = [
        f"format={CACHE_FORMAT_VERSION}",
        f"path={item.source_path}",
        f"file={_source_file_fingerprint(item.source_path)}",
        f"samples={item.num_samples}",
        f"vocab={item.vocab_fingerprint}",
        f"weights={sorted(item.source_weights)}",
        f"scale={item.strength_scale}",
    ]
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def default_cache_path(source_path: str | Path) -> Path:
    """``data/train.jsonl`` -> ``data/train.enc-cache.pt``."""
    path = Path(source_path)
    return path.with_name(path.stem + ".enc-cache.pt")


def load_encoded_cache(
    cache_path: str | Path, expected_key: str
) -> tuple[EncodedSample, ...] | None:
    """Return the cached encodings when the key matches, else ``None``."""
    path = Path(cache_path)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - corrupt caches simply fall back
        return None
    if not isinstance(payload, dict) or payload.get("key") != expected_key:
        return None
    encoded = payload.get("encoded")
    if not isinstance(encoded, tuple):
        return None
    return encoded


def save_encoded_cache(
    cache_path: str | Path, key: str, encoded: tuple[EncodedSample, ...]
) -> bool:
    """Atomically write the cache; ``False`` when the directory is unwritable."""
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        torch.save({"format": CACHE_FORMAT_VERSION, "key": key, "encoded": encoded}, temporary)
        temporary.replace(path)
    except OSError:
        return False
    return True
