"""Batching, padding and masking.

Padding contract (docs/12 section 8):

```text
attention_mask : 1 for real phoneme tokens, 0 for PAD
loss_mask      : 1 for real, supervised tokens, 0 otherwise
```

PAD tokens never contribute to attention, Viseme CE, Strength Huber or any
metric. Viseme targets are padded with ``IGNORE_INDEX`` so a mask bug turns
into a loud error rather than a silent class-0 bias.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import EncodedSample, PhonemeSequenceDataset
from .vocab import FEATURE_KEYS, PAD_ID

IGNORE_INDEX = -100

# Sequence-length buckets used for slice metrics.
LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("len_1_8", 1, 8),
    ("len_9_16", 9, 16),
    ("len_17_32", 17, 32),
    ("len_33_64", 33, 64),
    ("len_65_128", 65, 128),
    ("len_129_plus", 129, 1 << 30),
)

SLICE_FIELDS = (
    "language",
    "surface_tone",
    "stress",
    "syllable_role",
    "phrase_position",
    "length_bucket",
)


def length_bucket(length: int) -> str:
    for name, low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return name
    return LENGTH_BUCKETS[-1][0]


@dataclass
class Batch:
    """A padded batch of phoneme sequences."""

    # [B, T, F] long — per-field categorical ids, PAD_ID where padded.
    feature_ids: torch.Tensor
    # [B, T] bool — True for real tokens.
    attention_mask: torch.Tensor
    # [B, T] bool — True for tokens that contribute to loss and metrics.
    loss_mask: torch.Tensor
    # [B] long — true sequence lengths.
    lengths: torch.Tensor
    # [B, T] long — viseme class, IGNORE_INDEX where padded. None if unlabelled.
    viseme_targets: torch.Tensor | None = None
    # [B, T] float32 in [0,1] — strength/100, 0 where padded.
    strength_targets: torch.Tensor | None = None
    # [B, T] float32 — per-token strength loss multiplier, 0 where padded.
    strength_weight: torch.Tensor | None = None
    # [B, T] bool — True where the strength label is Human Gold.
    human_gold_strength: torch.Tensor | None = None

    sample_ids: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    # Flat, in masked row-major order: one entry per real token in the batch.
    phonemes: tuple[str, ...] = ()
    token_sample_index: tuple[int, ...] = ()
    token_position: tuple[int, ...] = ()
    slices: dict[str, tuple[str, ...]] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.feature_ids.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.feature_ids.shape[1])

    @property
    def num_real_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def num_supervised_tokens(self) -> int:
        return int(self.loss_mask.sum().item())

    @property
    def has_labels(self) -> bool:
        return self.viseme_targets is not None and self.strength_targets is not None

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Batch:
        """Move tensor fields to ``device``; string metadata stays on CPU."""

        def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device, non_blocking=non_blocking)

        return Batch(
            feature_ids=self.feature_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            loss_mask=self.loss_mask.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            viseme_targets=move(self.viseme_targets),
            strength_targets=move(self.strength_targets),
            strength_weight=move(self.strength_weight),
            human_gold_strength=move(self.human_gold_strength),
            sample_ids=self.sample_ids,
            texts=self.texts,
            phonemes=self.phonemes,
            token_sample_index=self.token_sample_index,
            token_position=self.token_position,
            slices=self.slices,
        )


class PhonemeCollator:
    """Collate :class:`EncodedSample` items into a padded :class:`Batch`."""

    def __init__(
        self,
        *,
        max_seq_len: int | None = None,
        collect_slices: bool = True,
        collect_phonemes: bool = True,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.collect_slices = collect_slices
        self.collect_phonemes = collect_phonemes

    def __call__(self, items: Sequence[EncodedSample]) -> Batch:
        if not items:
            raise ValueError("cannot collate an empty batch")

        lengths = [item.length for item in items]
        max_len = max(lengths)
        if self.max_seq_len is not None and max_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {max_len} exceeds max_seq_len {self.max_seq_len}; "
                "truncation would silently drop supervision"
            )

        batch_size = len(items)
        num_fields = len(FEATURE_KEYS)

        feature_ids = torch.full(
            (batch_size, max_len, num_fields), PAD_ID, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        labelled = all(item.viseme_ids is not None for item in items)
        partially_labelled = any(item.viseme_ids is not None for item in items)
        if partially_labelled and not labelled:
            raise ValueError(
                "cannot mix labelled and unlabelled samples in one batch; "
                "keep training and inference datasets separate"
            )

        viseme_targets: torch.Tensor | None = None
        strength_targets: torch.Tensor | None = None
        strength_weight: torch.Tensor | None = None
        human_gold: torch.Tensor | None = None
        if labelled:
            viseme_targets = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
            strength_targets = torch.zeros((batch_size, max_len), dtype=torch.float32)
            strength_weight = torch.zeros((batch_size, max_len), dtype=torch.float32)
            human_gold = torch.zeros((batch_size, max_len), dtype=torch.bool)

        phonemes: list[str] = []
        token_sample_index: list[int] = []
        token_position: list[int] = []
        slice_values: dict[str, list[str]] = {name: [] for name in SLICE_FIELDS}

        for row, item in enumerate(items):
            length = item.length
            feature_ids[row, :length] = item.feature_ids
            attention_mask[row, :length] = True

            if labelled:
                assert viseme_targets is not None and strength_targets is not None
                assert strength_weight is not None and human_gold is not None
                assert item.viseme_ids is not None and item.strength is not None
                assert item.strength_weight is not None and item.human_gold_strength is not None
                viseme_targets[row, :length] = item.viseme_ids
                strength_targets[row, :length] = item.strength
                strength_weight[row, :length] = item.strength_weight
                human_gold[row, :length] = item.human_gold_strength

            if self.collect_phonemes:
                phonemes.extend(item.phonemes)
                token_sample_index.extend([row] * length)
                token_position.extend(range(length))

            if self.collect_slices:
                bucket = length_bucket(length)
                slice_values["language"].extend(item.languages)
                slice_values["surface_tone"].extend(
                    f"tone_{tone}" for tone in item.surface_tones
                )
                slice_values["stress"].extend(f"stress_{s}" for s in item.stresses)
                slice_values["syllable_role"].extend(item.syllable_roles)
                slice_values["phrase_position"].extend(item.phrase_positions)
                slice_values["length_bucket"].extend([bucket] * length)

        # loss_mask starts identical to attention_mask; a future curriculum can
        # narrow it further, never widen it.
        loss_mask = attention_mask.clone()

        return Batch(
            feature_ids=feature_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            lengths=torch.tensor(lengths, dtype=torch.long),
            viseme_targets=viseme_targets,
            strength_targets=strength_targets,
            strength_weight=strength_weight,
            human_gold_strength=human_gold,
            sample_ids=tuple(item.sample_id for item in items),
            texts=tuple(item.text for item in items),
            phonemes=tuple(phonemes),
            token_sample_index=tuple(token_sample_index),
            token_position=tuple(token_position),
            slices=(
                {name: tuple(values) for name, values in slice_values.items()}
                if self.collect_slices
                else None
            ),
        )


class DynamicTokenBatchSampler(Sampler[list[int]]):
    """Group sentences of similar length under a padded-token budget.

    The budget counts *padded* tokens (``batch_size * max_len_in_batch``)
    because that is what determines activation memory, not the raw token sum.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens_per_batch: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        longest = max(lengths) if lengths else 0
        if longest > max_tokens_per_batch:
            raise ValueError(
                f"max_tokens_per_batch ({max_tokens_per_batch}) cannot hold the longest "
                f"sequence ({longest})"
            )
        self.lengths = list(lengths)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        order = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i))
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index in order:
            candidate_max = max(current_max, self.lengths[index])
            if current and candidate_max * (len(current) + 1) > self.max_tokens_per_batch:
                batches.append(current)
                current = [index]
                current_max = self.lengths[index]
            else:
                current.append(index)
                current_max = candidate_max
        if current:
            batches.append(current)

        if self.drop_last and len(batches) > 1:
            batches = batches[:-1]

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())


def build_dataloader(
    dataset: PhonemeSequenceDataset,
    *,
    strategy: str,
    batch_size: int = 8,
    max_phoneme_tokens_per_batch: int = 6000,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    max_seq_len: int | None = None,
    collect_slices: bool = True,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader using fixed-size or dynamic token batching.

    ``pin_memory`` should be on for CUDA training: it is what makes the
    ``non_blocking=True`` host-to-device copies in the trainer actually
    asynchronous. It is pointless (and wasteful) on CPU-only runs.
    """
    collator = PhonemeCollator(max_seq_len=max_seq_len, collect_slices=collect_slices)
    common = {
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if strategy == "fixed_samples":
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            drop_last=drop_last,
            **common,
        )

    if strategy == "dynamic_phoneme_tokens":
        sampler = DynamicTokenBatchSampler(
            dataset.lengths,
            max_phoneme_tokens_per_batch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)

    raise ValueError(f"unknown batching strategy {strategy!r}")
