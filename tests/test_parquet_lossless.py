"""Lossless-parquet data loader: parity with the JSONL loader."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from articulm.config import DataConfig
from articulm.data.parquet import load_samples_parquet
from articulm.data.schema import load_samples

pytestmark = pytest.mark.skipif(
    shutil.which("clickhouse-local") is None,
    reason="clickhouse-local not available",
)


@pytest.fixture(scope="session")
def lossless_parquet(fixture_paths: dict[str, Path], tmp_path_factory) -> Path:
    """Convert the zh fixture JSONL into lossless parquet tables."""
    out_dir = tmp_path_factory.mktemp("lossless")
    source = next(iter(fixture_paths.values()))
    script = Path(__file__).resolve().parent.parent / "scripts" / "to_parquet_lossless.sh"
    subprocess.run(
        ["bash", str(script), str(source), str(out_dir)],
        check=True, capture_output=True,
    )
    return out_dir / (source.stem + ".tokens.parquet")


def test_lossless_parquet_matches_jsonl_loader(
    data_config: DataConfig, fixture_paths, lossless_parquet: Path
):
    """The parquet loader must produce the identical Samples as the JSONL
    loader - same sentence order, same tokens, same normalisation."""
    source = next(iter(fixture_paths.values()))
    from_jsonl = load_samples(source, data_config)
    from_parquet = load_samples_parquet(lossless_parquet, data_config)

    assert len(from_jsonl) == len(from_parquet)
    for a, b in zip(from_jsonl, from_parquet, strict=True):
        assert a.sample_id == b.sample_id
        assert a.text == b.text
        assert len(a.tokens) == len(b.tokens)
        for ta, tb in zip(a.tokens, b.tokens, strict=True):
            assert ta == tb  # dataclass equality covers every feature field


def test_lossless_parquet_limit_caps_sentences(
    data_config: DataConfig, lossless_parquet: Path
):
    all_samples = load_samples_parquet(lossless_parquet, data_config)
    limited = load_samples_parquet(lossless_parquet, data_config, limit=2)
    assert 1 <= len(limited) <= 2
    assert [s.sample_id for s in limited] == [s.sample_id for s in all_samples[: len(limited)]]


def test_lossless_parquet_missing_samples_table(
    data_config: DataConfig, lossless_parquet: Path, tmp_path: Path
):
    """Tokens parquet without its samples table fails fast."""
    orphan = tmp_path / "orphan.tokens.parquet"
    shutil.copy(lossless_parquet, orphan)
    with pytest.raises(Exception, match="samples table not found"):
        load_samples_parquet(orphan, data_config)


def test_lossless_parquet_rejects_wrong_suffix(
    data_config: DataConfig, lossless_parquet: Path, tmp_path: Path
):
    """A parquet path that is not '<split>.tokens.parquet' fails fast."""
    wrong = tmp_path / "not_tokens.parquet"
    shutil.copy(lossless_parquet, wrong)
    with pytest.raises(Exception, match="tokens.parquet"):
        load_samples_parquet(wrong, data_config)
