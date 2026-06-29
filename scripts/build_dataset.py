"""
scripts/build_dataset.py
-------------------------
Reads the clip_metadata.csv (produced by extract_landmarks.py) and builds
PyTorch Dataset objects ready for training.

Also runs a quality-filter pass to flag/drop clips with too many missing
landmarks or normalization failures before they pollute training.

Usage:
    python scripts/build_dataset.py \
        --metadata data/metadata/clip_metadata.csv \
        --landmarks data/landmarks \
        --output data/metadata/dataset_splits.json

This does NOT trigger a re-extraction — it only reads from what
extract_landmarks.py already produced.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.landmark_config import MAX_SEQ_LEN, feature_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality filters — tune these thresholds as you learn your data
# ---------------------------------------------------------------------------

# Drop a clip if more than this % of frames had no pose (= no normalization anchor)
MAX_MISSING_POSE_PCT = 40.0

# Drop a clip if normalization failed for the ENTIRE clip
REQUIRE_NORMALIZATION_SUCCESS = True

# Drop a clip if it has label_id == -1 (video not in annotation JSON)
REQUIRE_KNOWN_LABEL = True


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class WLASLDataset(Dataset):
    """
    PyTorch Dataset over extracted WLASL landmark .npy files.

    Returns:
        landmarks : FloatTensor of shape (MAX_SEQ_LEN, feature_dim())
        mask      : BoolTensor of shape (MAX_SEQ_LEN,) — True = real frame
        label     : LongTensor scalar
    """

    def __init__(self, records: list[dict], landmarks_dir: Path, augment: bool = False):
        """
        records       : list of dicts from the filtered metadata CSV
        landmarks_dir : path to the directory containing .npy files
        augment       : whether to apply training augmentations
        """
        self.records = records
        self.landmarks_dir = landmarks_dir
        self.augment = augment
        self.D = feature_dim()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        npy_path = self.landmarks_dir / f"{rec['video_id']}.npy"

        # Load — should always exist since we filtered on npy_path
        landmarks = np.load(npy_path).astype(np.float32)   # (MAX_SEQ_LEN, D)

        # Build mask: real_frames worth of 1s, rest 0s
        real_frames = int(rec["real_frames"])
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask[:min(real_frames, MAX_SEQ_LEN)] = 1.0

        if self.augment:
            landmarks, mask = self._augment(landmarks, mask, real_frames)

        return (
            torch.from_numpy(landmarks),   # (MAX_SEQ_LEN, D) float
            torch.from_numpy(mask),        # (MAX_SEQ_LEN,)   float → cast to bool in model
            torch.tensor(rec["label_id"], dtype=torch.long),
        )

    def _augment(self, landmarks, mask, real_frames):
        """
        Light augmentations that preserve sign meaning:
          - Temporal jitter: random start offset within padded region
          - Gaussian noise on landmark coords (small σ)
          - Random horizontal flip (mirrors left↔right hand — valid in ASL
            for most lexical signs, though NOT valid for directional verbs;
            accept the noise for now, revisit if accuracy suffers)
        """
        T = real_frames

        # Temporal jitter: shift clip start by ±10% of clip length
        jitter = int(0.1 * T)
        if jitter > 0:
            shift = np.random.randint(-jitter, jitter + 1)
            if shift > 0:
                landmarks[shift:T + shift] = landmarks[:T]
                landmarks[:shift] = 0
            elif shift < 0:
                s = -shift
                landmarks[:T - s] = landmarks[s:T]
                landmarks[T - s:T] = 0

        # Gaussian noise
        noise = np.random.normal(0, 0.005, landmarks.shape).astype(np.float32)
        landmarks += noise

        # Horizontal flip: negate x coords (every even-indexed dim in our layout
        # corresponds to an x coordinate — left_hand x, right_hand x, pose x, face x)
        if np.random.rand() < 0.5:
            landmarks[:, 0::2] = -landmarks[:, 0::2]   # flip all x coords

        return landmarks, mask


# ---------------------------------------------------------------------------
# Build split datasets from metadata CSV
# ---------------------------------------------------------------------------

def load_and_filter(metadata_path: Path, landmarks_dir: Path) -> pd.DataFrame:
    """Load metadata CSV and apply quality filters."""
    df = pd.read_csv(metadata_path)
    original_len = len(df)

    # Must have a valid .npy file
    df = df[df["npy_path"].notna() & (df["npy_path"] != "")]
    df = df[df["npy_path"].apply(lambda p: Path(p).exists())]

    # Must have a known label
    if REQUIRE_KNOWN_LABEL:
        df = df[df["label_id"] != -1]

    # Must have succeeded normalization
    if REQUIRE_NORMALIZATION_SUCCESS:
        df = df[df["normalization_success"] == True]

    # Pose missing rate threshold
    df = df[df["missing_pose_pct"] <= MAX_MISSING_POSE_PCT]

    log.info(f"Quality filter: {original_len} clips → {len(df)} kept")
    log.info(f"  Dropped: {original_len - len(df)}")

    return df


def build_splits(df: pd.DataFrame, landmarks_dir: Path) -> dict:
    """
    Split into train/val/test using the 'split' column from WLASL JSON.
    Returns dict: {"train": WLASLDataset, "val": WLASLDataset, "test": WLASLDataset}
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        records = subset.to_dict("records")
        augment = (split_name == "train")
        splits[split_name] = WLASLDataset(records, landmarks_dir, augment=augment)
        log.info(f"  {split_name}: {len(records)} clips")

    # Handle any clips with split='unknown' — add to train
    unknown = df[~df["split"].isin(["train", "val", "test"])]
    if len(unknown) > 0:
        log.warning(f"  {len(unknown)} clips have unknown split — adding to train")
        combined_records = splits["train"].records + unknown.to_dict("records")
        splits["train"] = WLASLDataset(combined_records, landmarks_dir, augment=True)
        log.info(f"  train (after unknown merge): {len(splits['train'])} clips")

    return splits


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print useful statistics about the filtered dataset."""
    log.info("\n── Dataset statistics ──────────────────────────────────")
    log.info(f"  Total clips         : {len(df)}")
    log.info(f"  Unique glosses      : {df['label_id'].nunique()}")
    log.info(f"  Avg pose missing    : {df['missing_pose_pct'].mean():.1f}%")
    log.info(f"  Avg lhand missing   : {df['missing_left_hand_pct'].mean():.1f}%")
    log.info(f"  Avg rhand missing   : {df['missing_right_hand_pct'].mean():.1f}%")
    log.info(f"  Avg real frames     : {df['real_frames'].mean():.0f}")
    log.info(f"  Feature dim         : {feature_dim()}")

    if "split" in df.columns:
        log.info("  Split counts:")
        for s, cnt in df["split"].value_counts().items():
            log.info(f"    {s:<8}: {cnt}")
    log.info("─" * 54)


# ---------------------------------------------------------------------------
# Main (also usable as a quick smoke-test)
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    metadata_path  = Path(args.metadata)
    landmarks_dir  = Path(args.landmarks)
    output_path    = Path(args.output)

    df = load_and_filter(metadata_path, landmarks_dir)
    print_dataset_stats(df)

    splits = build_splits(df, landmarks_dir)

    # Quick DataLoader smoke test
    log.info("\nSmoke-testing DataLoader on train split (2 batches)...")
    train_loader = DataLoader(
        splits["train"],
        batch_size=16,
        shuffle=True,
        num_workers=0,   # keep 0 for the smoke test
        pin_memory=torch.cuda.is_available(),
    )

    for batch_i, (lm, mask, labels) in enumerate(train_loader):
        log.info(f"  Batch {batch_i}: landmarks={lm.shape}, mask={mask.shape}, labels={labels.shape}")
        assert lm.shape == (min(16, len(splits["train"])), MAX_SEQ_LEN, feature_dim()), \
            f"Unexpected landmarks shape: {lm.shape}"
        assert mask.shape[1] == MAX_SEQ_LEN
        assert labels.dtype == torch.long
        if batch_i >= 1:
            break

    log.info("DataLoader smoke test passed.")

    # Save split summary for reference
    summary = {
        split_name: {
            "n_clips": len(ds),
            "n_classes": df["label_id"].nunique(),
            "feature_dim": feature_dim(),
            "max_seq_len": MAX_SEQ_LEN,
        }
        for split_name, ds in splits.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Split summary saved to {output_path}")

    return splits   # so train.py can import and call build_splits() directly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyTorch datasets from extracted landmarks")
    parser.add_argument("--metadata",   default="data/metadata/clip_metadata.csv")
    parser.add_argument("--landmarks",  default="data/landmarks")
    parser.add_argument("--output",     default="data/metadata/dataset_splits.json")
    args = parser.parse_args()
    main(args)