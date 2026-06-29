"""
configs/landmark_config.py
--------------------------
Single source of truth for the per-frame feature representation.

Scope: what one frame looks like — which landmarks, how many dims,
       coordinate conventions, and data-quality ceilings.

Not in scope: sequence length for training, batch size, learning rate.
Those live in configs/train_config.py.

Downstream modules must call feature_dim() — never hardcode the integer.
Every extracted dataset should record config_hash() alongside its metadata
so experiments are reproducible even if this file changes between runs.
"""

import hashlib
import json

# ---------------------------------------------------------------------------
# Which landmark groups to include
# ---------------------------------------------------------------------------
USE_LEFT_HAND  = True   # 21 landmarks × 2 coords = 42 dims
USE_RIGHT_HAND = True   # 21 landmarks × 2 coords = 42 dims
USE_POSE       = True   # 33 landmarks × 3 coords = 99 dims
USE_FACE       = True   # 39-landmark subset × 2 coords = 78 dims

# ---------------------------------------------------------------------------
# Coordinate conventions
# Z is kept for pose (meaningful depth from MediaPipe's model).
# Dropped for hands and face — less reliable from a single webcam.
# ---------------------------------------------------------------------------
POSE_COORDS = 3   # (x, y, z)
HAND_COORDS = 2   # (x, y)
FACE_COORDS = 2   # (x, y)

N_HAND_LANDMARKS = 21
N_POSE_LANDMARKS = 33

# ---------------------------------------------------------------------------
# Face landmark subset — 39 indices
# Captures: eyebrow raises, eye aperture, mouth shape (open/closed, corners).
# All three carry grammatical weight in ASL (non-manual markers).
# Not using all 468 face mesh points — too noisy, too many dims.
#
# All indices validated at import time against MediaPipe's 468-point mesh
# (valid range 0–467). An out-of-range index raises ValueError before
# any video is processed.
# ---------------------------------------------------------------------------
FACE_INDICES = [
    # Left eyebrow (5)
    46, 53, 52, 51, 56,
    # Right eyebrow (5)
    276, 283, 282, 295, 285,
    # Left eye (6)
    33, 160, 158, 133, 153, 144,
    # Right eye (6)
    362, 385, 387, 263, 373, 380,
    # Mouth outer (8)
    61, 291, 39, 269, 0, 17, 405, 181,
    # Mouth inner (8)
    78, 308, 191, 415, 80, 88, 310, 318,
    # Nose tip anchor (1)
    4,
]
# 39 indices × 2 coords = 78 face dims

# ---------------------------------------------------------------------------
# Extraction ceilings (data-quality, not training parameters)
#
# ENABLE_TRUNCATION = False: save the full variable-length sequence so you
# can analyze the true length distribution from the metadata CSV before
# committing to a truncation threshold.  Flip to True once you've chosen one.
#
# MAX_FRAMES is a hard safety ceiling that applies regardless of
# ENABLE_TRUNCATION — it exists only to abort clearly-broken files
# (e.g. a corrupt video reporting 50,000 frames) without hanging the
# extractor for hours.  Set it well above any realistic signing clip.
# ---------------------------------------------------------------------------
ENABLE_TRUNCATION = False
# Maximum number of frames the extractor will process from a
# single clip.
#
# Protects against corrupted videos that report absurd frame
# counts (e.g. tens of thousands of frames).
#
# This is NOT a training parameter.
# Padding/truncation for training belongs in train_config.py.
MAX_FRAMES = 1000

# ---------------------------------------------------------------------------
# Validation — runs once at import, before any video is processed
# ---------------------------------------------------------------------------
_MAX_MEDIAPIPE_FACE_INDEX = 467  # MediaPipe face mesh: 468 landmarks (0–467)

def _validate() -> None:
    bad   = [i for i in FACE_INDICES if i > _MAX_MEDIAPIPE_FACE_INDEX]
    dupes = [i for i in set(FACE_INDICES) if FACE_INDICES.count(i) > 1]
    if bad:
        raise ValueError(
            f"FACE_INDICES contains indices out of MediaPipe range "
            f"(max {_MAX_MEDIAPIPE_FACE_INDEX}): {bad}"
        )
    if dupes:
        raise ValueError(f"FACE_INDICES contains duplicate indices: {dupes}")

_validate()   # fails loudly at import time — never silently at clip #8,000

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def feature_dim() -> int:
    """Total feature dimension per frame under the current config."""
    dim = 0
    if USE_LEFT_HAND:  dim += N_HAND_LANDMARKS * HAND_COORDS
    if USE_RIGHT_HAND: dim += N_HAND_LANDMARKS * HAND_COORDS
    if USE_POSE:       dim += N_POSE_LANDMARKS * POSE_COORDS
    if USE_FACE:       dim += len(FACE_INDICES) * FACE_COORDS
    return dim


def feature_breakdown() -> dict:
    """
    Per-group breakdown of landmark counts, coord dims, and total dims.
    Useful for logging and debugging — prefer this over hardcoding integers.

    Example output (all groups enabled):
        {
            "left_hand":  {"n_landmarks": 21, "coords": 2, "dims": 42},
            "right_hand": {"n_landmarks": 21, "coords": 2, "dims": 42},
            "pose":       {"n_landmarks": 33, "coords": 3, "dims": 99},
            "face":       {"n_landmarks": 39, "coords": 2, "dims": 78},
            "total_dims": 261,
        }
    """
    groups = {}

    if USE_LEFT_HAND:
        n = N_HAND_LANDMARKS
        groups["left_hand"] = {"n_landmarks": n, "coords": HAND_COORDS, "dims": n * HAND_COORDS}
    else:
        groups["left_hand"] = {"n_landmarks": 0, "coords": HAND_COORDS, "dims": 0}

    if USE_RIGHT_HAND:
        n = N_HAND_LANDMARKS
        groups["right_hand"] = {"n_landmarks": n, "coords": HAND_COORDS, "dims": n * HAND_COORDS}
    else:
        groups["right_hand"] = {"n_landmarks": 0, "coords": HAND_COORDS, "dims": 0}

    if USE_POSE:
        n = N_POSE_LANDMARKS
        groups["pose"] = {"n_landmarks": n, "coords": POSE_COORDS, "dims": n * POSE_COORDS}
    else:
        groups["pose"] = {"n_landmarks": 0, "coords": POSE_COORDS, "dims": 0}

    if USE_FACE:
        n = len(FACE_INDICES)
        groups["face"] = {"n_landmarks": n, "coords": FACE_COORDS, "dims": n * FACE_COORDS}
    else:
        groups["face"] = {"n_landmarks": 0, "coords": FACE_COORDS, "dims": 0}

    groups["total_dims"] = feature_dim()
    return groups


def config_hash() -> str:
    """
    Short deterministic hash of the current landmark configuration.

    Derived from: enabled flags, coordinate conventions, and FACE_INDICES.
    Changes automatically when any of those change — no manual version bump.

    Store this in your metadata CSV alongside every extracted clip so you
    can detect mismatches between a checkpoint's training data and a new
    extraction run.

    Example: "a3f7c2d1"
    """
    state = {
        "USE_LEFT_HAND":  USE_LEFT_HAND,
        "USE_RIGHT_HAND": USE_RIGHT_HAND,
        "USE_POSE":       USE_POSE,
        "USE_FACE":       USE_FACE,
        "POSE_COORDS":    POSE_COORDS,
        "HAND_COORDS":    HAND_COORDS,
        "FACE_COORDS":    FACE_COORDS,
        "N_HAND_LANDMARKS": N_HAND_LANDMARKS,
        "N_POSE_LANDMARKS": N_POSE_LANDMARKS,
        "FACE_INDICES": FACE_INDICES,  
    }
    blob = json.dumps(state, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# CLI: python configs/landmark_config.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bd = feature_breakdown()
    print("Feature vector breakdown:")
    for group, info in bd.items():
        if group == "total_dims":
            print(f"  {'total':<12}: {info} dims")
        else:
            enabled = info['dims'] > 0
            print(
                f"  {group:<12}: {info['n_landmarks']} landmarks "
                f"× {info['coords']} coords = {info['dims']} dims"
                + ("" if enabled else "  [DISABLED]")
            )
    print(f"\n  config_hash : {config_hash()}")
    print(f"  MAX_FRAMES  : {MAX_FRAMES}  (hard safety ceiling)")
    print(f"  truncation  : {'enabled' if ENABLE_TRUNCATION else 'disabled — saving full-length sequences'}")