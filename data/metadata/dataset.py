"""
data/dataset.py
----------------
The PyTorch Dataset class for WLASL landmark sequences.

Responsible for exactly four things (deliberately not more):
    1. Loading a clip's variable-length .npy landmark array from disk
    2. Loading its integer label (gloss id)
    3. Padding/truncating to a fixed length + building the attention mask
    4. Returning (landmarks, mask, label) as tensors, ready to collate

What this file does NOT do (by design, lives elsewhere):
    - Quality filtering (missing-detection thresholds, normalization-success
      checks) — that's scripts/build_dataset.py's job, upstream of this class.
      By the time a record reaches SignLanguageDataset, it's assumed clean.
    - Batching/shuffling/DataLoader construction — that's data/dataloader.py.
    - Augmentation policy decisions — augmentation transforms can be passed
      in, but this class doesn't decide what augmentation should exist.

Padding convention (must match models/transformer.py and configs/train_config.py):
    - Sequences are padded/truncated to MAX_SEQ_LEN frames.
    - mask[t] = True  -> real frame
    - mask[t] = False -> padding
    - Padding is zeros, appended at the END of the sequence (not pre-padded).
    - Truncation, if a clip is longer than MAX_SEQ_LEN, takes the FIRST
      MAX_SEQ_LEN frames (documented explicitly below — change this in one
      place if you'd rather center-crop or sample).

This file is fully testable without the real WLASL dataset: the
`if __name__ == "__main__"` block below builds a tiny fake dataset on disk
(random .npy files + a synthetic metadata CSV) and exercises every code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.landmark_config import feature_dim
from configs.train_config import MAX_SEQ_LEN


class SignLanguageDataset(Dataset):
    """
    Dataset over extracted landmark .npy files + a metadata CSV.

    Expected metadata CSV columns (subset of what scripts/build_dataset.py
    and scripts/extract_landmarks.py already produce):
        video_id   : str  — filename stem, used to locate "{video_id}.npy"
        label_id   : int  — integer gloss class id (-1 reserved for "unknown",
                             callers should already have filtered these out
                             upstream; this class does not silently drop them)
        split      : str  — "train" / "val" / "test" (used by dataloader.py,
                             not filtered here — this class doesn't know about
                             splits, it just loads whatever records it's given)

    Args:
        records       : list[dict] — pre-filtered metadata rows (e.g. from
                        pandas.DataFrame.to_dict("records")). This class does
                        NOT read the CSV itself — keeping I/O and filtering
                        decisions in the caller (build_dataset.py / dataloader.py)
                        means this class is trivially testable with fake records.
        landmarks_dir : Path — directory containing "{video_id}.npy" files
        max_seq_len   : int — fixed sequence length after pad/truncate
        feature_dim_  : int — expected last-dim size of each loaded array;
                        mismatches raise loudly rather than silently reshaping
        augment       : optional callable (landmarks: np.ndarray, real_frames: int)
                        -> (landmarks, real_frames). Applied AFTER loading but
                        BEFORE padding. Pass None for no augmentation (val/test).
    """

    def __init__(
        self,
        records: list[dict],
        landmarks_dir: str | Path,
        max_seq_len: int = MAX_SEQ_LEN,
        feature_dim_: int | None = None,
        augment=None,
    ):
        self.records = records
        self.landmarks_dir = Path(landmarks_dir)
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim_ if feature_dim_ is not None else feature_dim()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        video_id = rec["video_id"]
        label_id = int(rec["label_id"])

        npy_path = self.landmarks_dir / f"{video_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Landmark file missing for video_id={video_id!r}: {npy_path}. "
                f"This record should have been filtered out upstream by "
                f"scripts/build_dataset.py — check the metadata CSV."
            )

        landmarks = np.load(npy_path).astype(np.float32)   # (T, D), variable T

        if landmarks.ndim != 2 or landmarks.shape[1] != self.feature_dim:
            raise ValueError(
                f"{video_id}: loaded landmark shape {landmarks.shape} doesn't "
                f"match expected feature_dim={self.feature_dim}. "
                f"This usually means the .npy was extracted under a different "
                f"landmark_config than the one currently active — check config_hash."
            )

        real_frames = landmarks.shape[0]

        if self.augment is not None:
            landmarks, real_frames = self.augment(landmarks, real_frames)

        padded, mask = self._pad_or_truncate(landmarks, real_frames)

        return (
            torch.from_numpy(padded),                       # (max_seq_len, D) float32
            torch.from_numpy(mask),                          # (max_seq_len,)   bool
            torch.tensor(label_id, dtype=torch.long),         # scalar
        )

    def _pad_or_truncate(
        self, landmarks: np.ndarray, real_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad with zeros at the end, or truncate to the first max_seq_len frames.

        Truncation policy: take the FIRST max_seq_len frames, not a random
        or centered window. This is a deliberate, simple default — flag it
        here so it's a one-line change later if you decide center-cropping
        preserves more sign content for long clips.

        Returns:
            padded : (max_seq_len, D) float32
            mask   : (max_seq_len,) bool, True = real frame
        """
        T, D = landmarks.shape
        L = self.max_seq_len

        padded = np.zeros((L, D), dtype=np.float32)
        mask = np.zeros(L, dtype=bool)

        if T >= L:
            padded[:] = landmarks[:L]
            mask[:] = True
            effective = L
        else:
            padded[:T] = landmarks
            mask[:T] = True
            effective = T

        # real_frames may differ from T if augmentation changed it (e.g. temporal
        # jitter shifting content) — but mask is built from actual array length T,
        # since that's what's physically present in `landmarks` right now.
        del effective  # not currently used downstream; kept for future logging
        return padded, mask


# ============================================================================
# Smoke test — builds a tiny FAKE dataset on disk, no real data required
# ============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 60)
    print("SignLanguageDataset smoke test (synthetic data, no WLASL required)")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp(prefix="sld_smoketest_"))
    landmarks_dir = tmp_dir / "landmarks"
    landmarks_dir.mkdir()

    D = feature_dim()
    print(f"feature_dim() = {D}")

    rng = np.random.default_rng(0)

    # Build fake clips with deliberately varied lengths, including:
    #  - shorter than MAX_SEQ_LEN (needs padding)
    #  - longer than MAX_SEQ_LEN (needs truncation)
    #  - exactly MAX_SEQ_LEN (edge case)
    fake_lengths = [40, 150, 200, 1, 149, 151]
    records = []
    for i, T in enumerate(fake_lengths):
        video_id = f"fake_{i:03d}"
        arr = rng.normal(size=(T, D)).astype(np.float32)
        np.save(landmarks_dir / f"{video_id}.npy", arr)
        records.append({"video_id": video_id, "label_id": i % 5, "split": "train"})

    try:
        ds = SignLanguageDataset(records, landmarks_dir, max_seq_len=MAX_SEQ_LEN)
        print(f"\nDataset length: {len(ds)} (expected {len(fake_lengths)})")
        assert len(ds) == len(fake_lengths)

        for i, T in enumerate(fake_lengths):
            landmarks, mask, label = ds[i]
            expected_real = min(T, MAX_SEQ_LEN)
            print(
                f"  [{i}] real_frames={T:>4} -> "
                f"landmarks={tuple(landmarks.shape)}, "
                f"mask.sum()={int(mask.sum())} (expected {expected_real}), "
                f"label={label.item()}"
            )
            assert landmarks.shape == (MAX_SEQ_LEN, D)
            assert mask.shape == (MAX_SEQ_LEN,)
            assert mask.dtype == torch.bool
            assert int(mask.sum()) == expected_real
            # Padding region must be exactly zero
            if T < MAX_SEQ_LEN:
                assert torch.all(landmarks[T:] == 0.0), "Padding region is not zero!"
            # Real region must be untouched (no augmentation in this test)
            assert mask[:expected_real].all()
            if T >= MAX_SEQ_LEN:
                assert mask.all(), "Truncated/exact clip should have a full mask"

        # --- Missing file should raise FileNotFoundError, not silently skip ---
        print("\n[Edge case] Missing .npy file should raise FileNotFoundError")
        bad_records = [{"video_id": "does_not_exist", "label_id": 0, "split": "train"}]
        ds_bad = SignLanguageDataset(bad_records, landmarks_dir, max_seq_len=MAX_SEQ_LEN)
        try:
            _ = ds_bad[0]
            raise AssertionError("Expected FileNotFoundError but none was raised")
        except FileNotFoundError as e:
            print(f"  Correctly raised: {e}")

        # --- Wrong feature_dim should raise ValueError ---
        print("\n[Edge case] Mismatched feature_dim should raise ValueError")
        wrong_dim_id = "wrong_dim_clip"
        np.save(landmarks_dir / f"{wrong_dim_id}.npy",
                rng.normal(size=(50, D + 5)).astype(np.float32))
        ds_wrong = SignLanguageDataset(
            [{"video_id": wrong_dim_id, "label_id": 0, "split": "train"}],
            landmarks_dir, max_seq_len=MAX_SEQ_LEN,
        )
        try:
            _ = ds_wrong[0]
            raise AssertionError("Expected ValueError but none was raised")
        except ValueError as e:
            print(f"  Correctly raised: {e}")

        # --- Augmentation hook gets called and respected ---
        print("\n[Edge case] Augmentation hook is applied before padding")
        def fake_augment(landmarks, real_frames):
            # Zero everything out so we can verify the hook actually ran
            return np.zeros_like(landmarks), real_frames

        ds_aug = SignLanguageDataset(
            records, landmarks_dir, max_seq_len=MAX_SEQ_LEN, augment=fake_augment
        )
        landmarks_aug, mask_aug, _ = ds_aug[0]
        assert torch.all(landmarks_aug[mask_aug] == 0.0)
        print("  Augmentation hook correctly applied before padding.")

        print("\n" + "=" * 60)
        print("All dataset.py smoke tests passed.")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp_dir)