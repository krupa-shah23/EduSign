"""
data/dataloader.py
-------------------
Builds train/val/test DataLoaders on top of data/dataset.py's
SignLanguageDataset.

Responsible for:
    - Reading the metadata CSV and splitting records by the "split" column
    - Wrapping each split in a SignLanguageDataset (train gets augmentation,
      val/test do not — augmenting val/test would corrupt your evaluation
      numbers, so this is enforced here, not left to caller discretion)
    - A collate_fn (even though SignLanguageDataset already returns
      fixed-length padded tensors, so the default collate would technically
      work — an explicit collate_fn is kept here as the single place to
      extend later if e.g. you start returning variable-length sequences
      instead of pre-padding in __getitem__)
    - Constructing the actual torch.utils.data.DataLoader objects with the
      right shuffle/batch_size/num_workers per split

What this file does NOT do:
    - Quality filtering — assumed already done by scripts/build_dataset.py
      before the CSV reaches this file.
    - Augmentation policy itself — that lives wherever the augment callable
      is defined (e.g. a transforms module); this file only decides WHICH
      splits receive it.

This file is fully testable without WLASL: see the smoke test at the bottom,
which builds a synthetic metadata CSV + fake .npy files and exercises all
three loaders end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.landmark_config import feature_dim
from configs.train_config import BATCH_SIZE, MAX_SEQ_LEN, NUM_WORKERS
from data.dataset import SignLanguageDataset

VALID_SPLITS = ("train", "val", "test")


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Stacks a list of (landmarks, mask, label) tuples into batched tensors.

    Since SignLanguageDataset already pads every item to a fixed
    max_seq_len in __getitem__, torch.stack works directly — no ragged
    sequences reach this function. Kept as an explicit named function
    (not the default collate) so this is the one place to change if the
    padding strategy ever moves from per-item to per-batch (e.g. padding
    only to the longest sequence IN the batch, which is more efficient
    but is a deliberate design decision to make explicitly, not drift into).

    Args:
        batch: list of (landmarks (L,D), mask (L,), label ()) tuples
    Returns:
        landmarks: (B, L, D) float32
        mask:      (B, L) bool
        labels:    (B,) long
    """
    landmarks, masks, labels = zip(*batch)
    return (
        torch.stack(landmarks, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(labels, dim=0),
    )


def _load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required_cols = {"video_id", "label_id", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"metadata CSV is missing required columns: {missing}. "
            f"Expected at least: {required_cols}"
        )
    unknown_splits = set(df["split"].unique()) - set(VALID_SPLITS)
    if unknown_splits:
        raise ValueError(
            f"metadata CSV contains unexpected split values: {unknown_splits}. "
            f"Expected only {VALID_SPLITS}. If you have an 'unknown' split "
            f"bucket, resolve it (e.g. merge into train) before this point — "
            f"see scripts/build_dataset.py's unknown-split handling."
        )
    return df


def build_dataloaders(
    metadata_path: str | Path,
    landmarks_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    num_workers: int = NUM_WORKERS,
    augment_fn=None,
) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from a metadata CSV + landmarks directory.

    Args:
        metadata_path : path to clip_metadata.csv (or a filtered version of it)
        landmarks_dir : directory containing "{video_id}.npy" files
        batch_size    : applies to all three splits (override per-split by
                        calling SignLanguageDataset/DataLoader directly if
                        you ever need asymmetric batch sizes)
        max_seq_len   : passed straight to SignLanguageDataset
        num_workers   : DataLoader worker processes; 0 is safer for debugging
                        (single process, easier tracebacks), >0 for real training
        augment_fn    : callable applied ONLY to the train split. None means
                        no augmentation anywhere — explicit opt-in, not a
                        silent default, since silently training without
                        augmentation is an easy mistake to not notice.

    Returns:
        dict with keys "train", "val", "test", each a torch.utils.data.DataLoader.
        A split with zero matching rows still returns a valid (empty) DataLoader
        rather than raising — but a warning is printed, since an empty split
        is almost always a bug (e.g. wrong split-column values upstream).
    """
    df = _load_metadata(metadata_path)
    D = feature_dim()

    loaders: dict[str, DataLoader] = {}
    for split in VALID_SPLITS:
        subset = df[df["split"] == split]
        records = subset.to_dict("records")

        if len(records) == 0:
            print(
                f"[dataloader] WARNING: split={split!r} has 0 records. "
                f"This is almost always a bug upstream (check the 'split' "
                f"column in {metadata_path}) — continuing with an empty loader."
            )

        is_train = split == "train"
        dataset = SignLanguageDataset(
            records=records,
            landmarks_dir=landmarks_dir,
            max_seq_len=max_seq_len,
            feature_dim_=D,
            augment=augment_fn if is_train else None,   # never augment val/test
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,          # only shuffle train
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return loaders


# ============================================================================
# Smoke test — synthetic metadata + fake .npy files, no WLASL required
# ============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    import numpy as np

    print("=" * 60)
    print("dataloader.py smoke test (synthetic data, no WLASL required)")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp(prefix="dl_smoketest_"))
    landmarks_dir = tmp_dir / "landmarks"
    landmarks_dir.mkdir()
    metadata_path = tmp_dir / "clip_metadata.csv"

    D = feature_dim()
    rng = np.random.default_rng(0)

    rows = []
    split_plan = (["train"] * 12) + (["val"] * 4) + (["test"] * 4)
    for i, split in enumerate(split_plan):
        video_id = f"clip_{i:03d}"
        T = int(rng.integers(20, 200))   # variable lengths, some > MAX_SEQ_LEN
        arr = rng.normal(size=(T, D)).astype(np.float32)
        np.save(landmarks_dir / f"{video_id}.npy", arr)
        rows.append({"video_id": video_id, "label_id": i % 5, "split": split})

    pd.DataFrame(rows).to_csv(metadata_path, index=False)

    try:
        loaders = build_dataloaders(
            metadata_path=metadata_path,
            landmarks_dir=landmarks_dir,
            batch_size=4,
            num_workers=0,   # 0 for the smoke test — easier to debug
        )

        print(f"\nBuilt loaders: {list(loaders.keys())}")
        assert set(loaders.keys()) == set(VALID_SPLITS)

        expected_counts = {"train": 12, "val": 4, "test": 4}
        for split, loader in loaders.items():
            n = len(loader.dataset)
            print(f"\n[{split}] dataset size: {n} (expected {expected_counts[split]})")
            assert n == expected_counts[split]

            batch_landmarks, batch_mask, batch_labels = next(iter(loader))
            print(
                f"  batch landmarks: {tuple(batch_landmarks.shape)}, "
                f"mask: {tuple(batch_mask.shape)}, labels: {tuple(batch_labels.shape)}"
            )
            assert batch_landmarks.shape[1:] == (MAX_SEQ_LEN, D)
            assert batch_mask.shape[1] == MAX_SEQ_LEN
            assert batch_mask.dtype == torch.bool
            assert batch_labels.dtype == torch.long

        # --- shuffle behavior: train should shuffle, val/test should not ---
        print(f"\n[Check] train.shuffle=True, val/test.shuffle=False (via sampler type)")
        from torch.utils.data import RandomSampler, SequentialSampler
        assert isinstance(loaders["train"].sampler, RandomSampler)
        assert isinstance(loaders["val"].sampler, SequentialSampler)
        assert isinstance(loaders["test"].sampler, SequentialSampler)
        print("  Correct: train uses RandomSampler, val/test use SequentialSampler.")

        # --- augmentation only applied to train ---
        print(f"\n[Check] augment_fn only applied to train split")
        calls = {"train": 0, "val": 0, "test": 0}

        def counting_augment(landmarks, real_frames):
            return landmarks, real_frames

        # Rebuild with an augment_fn and manually confirm only train's dataset has it set
        loaders2 = build_dataloaders(
            metadata_path=metadata_path,
            landmarks_dir=landmarks_dir,
            batch_size=4,
            num_workers=0,
            augment_fn=counting_augment,
        )
        assert loaders2["train"].dataset.augment is counting_augment
        assert loaders2["val"].dataset.augment is None
        assert loaders2["test"].dataset.augment is None
        print("  Correct: augment_fn set on train only, val/test have augment=None.")

        # --- malformed metadata raises clearly ---
        print(f"\n[Edge case] Missing required column should raise ValueError")
        bad_csv = tmp_dir / "bad_metadata.csv"
        pd.DataFrame([{"video_id": "x", "split": "train"}]).to_csv(bad_csv, index=False)
        try:
            build_dataloaders(bad_csv, landmarks_dir)
            raise AssertionError("Expected ValueError but none was raised")
        except ValueError as e:
            print(f"  Correctly raised: {e}")

        # --- unexpected split value raises clearly ---
        print(f"\n[Edge case] Unknown split value should raise ValueError")
        bad_split_csv = tmp_dir / "bad_split.csv"
        pd.DataFrame(
            [{"video_id": "clip_000", "label_id": 0, "split": "unknown"}]
        ).to_csv(bad_split_csv, index=False)
        try:
            build_dataloaders(bad_split_csv, landmarks_dir)
            raise AssertionError("Expected ValueError but none was raised")
        except ValueError as e:
            print(f"  Correctly raised: {e}")

        # --- empty split warns but doesn't crash ---
        print(f"\n[Edge case] Empty split should warn, not crash")
        rows_no_test = [r for r in rows if r["split"] != "test"]
        no_test_csv = tmp_dir / "no_test.csv"
        pd.DataFrame(rows_no_test).to_csv(no_test_csv, index=False)
        loaders3 = build_dataloaders(no_test_csv, landmarks_dir, batch_size=4, num_workers=0)
        assert len(loaders3["test"].dataset) == 0
        print("  Correct: empty test split produced a valid empty DataLoader.")

        print("\n" + "=" * 60)
        print("All dataloader.py smoke tests passed.")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp_dir)