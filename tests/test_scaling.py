"""Device portability and memory-footprint guards.

These pin the two properties that decide whether the project can move to a GPU
box and to a large corpus:

* every module works after ``.to(device)`` — including the lazily rebuilt
  RoPE cache, which used to be constructed on the CPU unconditionally;
* the dataset does not retain the parsed ``Sample`` objects, which cost ~14x
  what the encoded tensors do and are never read again after encoding.
"""

from __future__ import annotations

import pytest
import torch

from articulm.data.collator import build_dataloader
from articulm.data.dataset import PhonemeSequenceDataset
from articulm.data.vocab import FEATURE_KEYS
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.model.transformer import RotaryPositionEmbedding

ACCELERATORS = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="no MPS device"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device"
        ),
    ),
]


# -- RoPE cache device safety ---------------------------------------------


@pytest.mark.parametrize("device", ACCELERATORS)
def test_rope_cache_rebuild_stays_on_device(device):
    """A sequence longer than the initial cache must not fall back to CPU."""
    rope = RotaryPositionEmbedding(head_dim=32, max_seq_len=8).to(device)
    assert rope.inverse_frequency.device.type == device

    longer = torch.randn(1, 2, 40, 32, device=device)
    rotated = rope(longer)
    assert rotated.shape == longer.shape
    assert rotated.device.type == device
    assert rope.cos_cache.device.type == device


@pytest.mark.parametrize("device", ACCELERATORS)
def test_rope_is_norm_preserving_on_device(device):
    rope = RotaryPositionEmbedding(head_dim=32, max_seq_len=64).to(device)
    tensor = torch.randn(2, 2, 20, 32, device=device)
    rotated = rope(tensor)
    assert torch.allclose(
        rotated.norm(dim=-1), tensor.norm(dim=-1), atol=1e-4
    )


def test_rope_rebuild_matches_a_freshly_sized_cache():
    """Growing the cache must produce the same values as building it upfront."""
    grown = RotaryPositionEmbedding(head_dim=16, max_seq_len=4)
    upfront = RotaryPositionEmbedding(head_dim=16, max_seq_len=32)
    tensor = torch.randn(1, 1, 32, 16)
    assert torch.allclose(grown(tensor), upfront(tensor), atol=1e-6)


# -- full model on device -------------------------------------------------


@pytest.mark.parametrize("device", ACCELERATORS)
def test_model_forward_and_backward_on_device(device, tiny_model_config, vocab, dataset):
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab).to(device)
    loader = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=3, shuffle=False
    )
    batch = next(iter(loader)).to(device)

    output = model(batch.feature_ids, batch.attention_mask)
    assert output.viseme_logits.device.type == device
    assert output.strength_norm.device.type == device

    loss = output.viseme_logits[batch.loss_mask].float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("device", ACCELERATORS)
def test_batch_to_device_moves_every_tensor(device, dataset):
    loader = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=2, shuffle=False
    )
    batch = next(iter(loader)).to(device)
    for name in (
        "feature_ids",
        "attention_mask",
        "loss_mask",
        "lengths",
        "viseme_targets",
        "strength_targets",
        "strength_weight",
        "human_gold_strength",
    ):
        tensor = getattr(batch, name)
        assert tensor is not None, name
        assert tensor.device.type == device, name


@pytest.mark.parametrize("device", ACCELERATORS)
def test_padding_isolation_holds_on_device(device, tiny_model_config, vocab, dataset):
    """The zero-at-PAD guarantee must not be a CPU-only artefact."""
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab).to(device)
    model.eval()
    loader = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=len(dataset), shuffle=False
    )
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    padded = ~batch.attention_mask
    assert bool(padded.any())
    assert float(output.hidden_states[padded].abs().max()) == 0.0


# -- dataset memory retention ---------------------------------------------


def test_dataset_drops_parsed_samples_by_default(all_samples, vocab):
    ds = PhonemeSequenceDataset(all_samples, vocab)
    assert ds.samples == ()
    assert len(ds.encoded) == len(all_samples)
    assert len(ds) == len(all_samples)


def test_dataset_can_retain_parsed_samples_on_request(all_samples, vocab):
    ds = PhonemeSequenceDataset(all_samples, vocab, retain_parsed_samples=True)
    assert len(ds.samples) == len(all_samples)


def test_subset_works_without_retained_samples(all_samples, vocab):
    ds = PhonemeSequenceDataset(all_samples, vocab)
    subset = ds.subset(2)
    assert len(subset) == 2
    assert subset.encoded[0].sample_id == ds.encoded[0].sample_id
    assert subset.vocab is ds.vocab


def test_parsed_samples_become_collectable_after_encoding(fixture_paths, data_config, vocab):
    """The point of dropping ``.samples``: the parsed objects must be freeable.

    The dataset only ever held *references*, so this is not about peak memory
    during construction — it is about the parsed corpus staying pinned alive
    for the whole run when only the tensors are still needed.
    """
    import gc
    import weakref

    from articulm.data.schema import load_samples

    samples = load_samples(fixture_paths["zh"], data_config)
    watchers = [weakref.ref(sample) for sample in samples]

    dataset = PhonemeSequenceDataset(samples, vocab)
    del samples
    gc.collect()

    assert all(watcher() is None for watcher in watchers), (
        "parsed Sample objects are still reachable; something retained them"
    )
    # The encoded tensors survive and are all training needs.
    assert len(dataset.encoded) == len(watchers)
    assert dataset.encoded[0].feature_ids.numel() > 0


def test_retaining_parsed_samples_keeps_them_alive(fixture_paths, data_config, vocab):
    """The opt-in flag must actually pin them, so callers that need the
    originals still get them."""
    import gc
    import weakref

    from articulm.data.schema import load_samples

    samples = load_samples(fixture_paths["zh"], data_config)
    watcher = weakref.ref(samples[0])

    dataset = PhonemeSequenceDataset(samples, vocab, retain_parsed_samples=True)
    del samples
    gc.collect()

    assert watcher() is not None
    assert len(dataset.samples) == len(dataset.encoded)


def test_dataloader_pin_memory_is_opt_in(dataset):
    """pin_memory must default off so CPU-only runs do not pay for it."""
    default = build_dataloader(dataset, strategy="fixed_samples", batch_size=2)
    assert default.pin_memory is False
    pinned = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=2, pin_memory=True
    )
    assert pinned.pin_memory is True


def test_dynamic_batching_also_honours_pin_memory(dataset):
    loader = build_dataloader(
        dataset,
        strategy="dynamic_phoneme_tokens",
        max_phoneme_tokens_per_batch=300,
        pin_memory=True,
    )
    assert loader.pin_memory is True


def test_feature_ids_use_a_compact_layout(dataset):
    """One int64 column per field; no per-token Python objects in the tensor."""
    item = dataset.encoded[0]
    assert item.feature_ids.dtype == torch.long
    assert item.feature_ids.shape == (item.length, len(FEATURE_KEYS))
    assert item.feature_ids.is_contiguous()
