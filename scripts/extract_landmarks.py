"""
scripts/extract_landmarks.py
-----------------------------
Reads each video clip, runs MediaPipe Holistic frame-by-frame,
normalizes landmarks (shoulder-centre translation + shoulder-width scaling),
and saves one variable-length .npy tensor per clip.

Saves shape (T, feature_dim()) where T is the actual frame count — no padding.
Padding to a fixed length happens later in the DataLoader collate function.

Also writes per-clip metadata (missing detection rates, frame counts,
config hash, timing) so clips can be quality-filtered before training.

Usage:
    python scripts/extract_landmarks.py \
        --videos   data/raw_videos \
        --json     data/WLASL_v0.3.json \
        --output   data/landmarks \
        --metadata data/metadata/clip_metadata.csv

Resume support: clips whose .npy already exists are skipped, so you can
safely interrupt and restart without reprocessing completed clips.

Architecture:
    read_frames()         → yields decoded BGR frames from one video file
    extract_sequence()    → MediaPipe + normalization over a frame list
    process_clip()        → orchestrates the above, builds metadata dict
    main()                → iterates clips, writes CSV, manages I/O
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Project root on path so `configs` is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.landmark_config import (
    USE_LEFT_HAND, USE_RIGHT_HAND, USE_POSE, USE_FACE,
    FACE_INDICES, FACE_COORDS, POSE_COORDS, HAND_COORDS,
    N_HAND_LANDMARKS, N_POSE_LANDMARKS,
    MAX_FRAMES, ENABLE_TRUNCATION,
    feature_dim, feature_breakdown, config_hash,
)

# ---------------------------------------------------------------------------
# Logging — stdout + file, so you can tail the file on a long run
# ---------------------------------------------------------------------------
Path("data/metadata").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/metadata/extraction.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache feature_dim() — it's a function call with four conditionals,
# and calling it per-frame across 21,000 clips × ~30 fps adds up.
# ---------------------------------------------------------------------------
_FEATURE_DIM: int = feature_dim()
_CONFIG_HASH: str = config_hash()
_PIPELINE_VERSION = "v1"
_NORMALIZATION = "shoulder_midpoint_width"

# ---------------------------------------------------------------------------
# MediaPipe setup — module-level, not per-clip
# ---------------------------------------------------------------------------
_mp_holistic = mp.solutions.holistic


# ============================================================================
# Normalization helpers
# ============================================================================

def get_shoulder_anchor(
    pose_landmarks,
) -> tuple[tuple[float, float] | None, float | None]:
    """
    Compute the shoulder-midpoint and shoulder-width from MediaPipe pose
    landmarks.

    Uses landmarks 11 (left shoulder) and 12 (right shoulder).

    Returns:
        (midpoint_xy, shoulder_width)  — both None if pose is absent or
        shoulder width is degenerate (< 1e-6, i.e. shoulders overlapping
        in the projected 2D frame).

    Why shoulder-based normalization:
        Translating by the shoulder midpoint makes the representation
        position-invariant (signer can stand anywhere in frame).
        Dividing by shoulder width makes it scale-invariant (signer can
        be closer or further from the camera).  Both are irrelevant to
        sign meaning and waste model capacity if left in.

    Why (None, None) on missing pose rather than falling back to frame
    centre / fixed scale:
        A fallback value would silently inject a different normalization
        convention for that frame, corrupting the representation in a way
        that's hard to detect.  Returning None lets the caller mark the
        frame as "no anchor" and forward-fill from a neighbouring frame
        that did have a good pose detection.
    """
    if pose_landmarks is None:
        return None, None

    lm = pose_landmarks.landmark
    ls = lm[11]   # left shoulder
    rs = lm[12]   # right shoulder

    mid_x = (ls.x + rs.x) / 2.0
    mid_y = (ls.y + rs.y) / 2.0
    width = abs(rs.x - ls.x)

    if width < 1e-6:
        # Degenerate: shoulders projected on top of each other.
        # Treat the same as missing pose — don't divide by near-zero.
        return None, None

    return (mid_x, mid_y), width


def _norm_xy(x: float, y: float, mid: tuple, width: float) -> tuple[float, float]:
    """Translate by shoulder midpoint, scale by shoulder width (x, y only)."""
    return (x - mid[0]) / width, (y - mid[1]) / width


def _norm_xyz(
    x: float, y: float, z: float, mid: tuple, width: float
) -> tuple[float, float, float]:
    """
    Same as _norm_xy but carries z, scaled by the same width for consistency.

    Note: z / width puts depth on the same scale as x and y.  An alternative
    is to drop z entirely (pose z from a monocular webcam is an estimate).
    We keep it here because MediaPipe's pose z is moderately reliable for
    upper-body depth ordering, but flag the decision explicitly so it can
    be ablated later without confusion.
    """
    return (x - mid[0]) / width, (y - mid[1]) / width, z / width


# ============================================================================
# Frame-level extraction
# ============================================================================

def _extract_frame(
    results,
    mid: tuple | None,
    width: float | None,
    out: np.ndarray,
) -> dict[str, bool]:
    """
    Write one frame's normalized landmarks into `out` (shape: (_FEATURE_DIM,)).

    Groups are written in a fixed order:
        left_hand → right_hand → pose → face
    A missing group (no detection, or no anchor) leaves its slice as zeros.
    The tensor layout is deterministic regardless of detection results.

    Args:
        results : MediaPipe Holistic results for this frame
        mid     : shoulder midpoint (x, y), or None if pose missing
        width   : shoulder width, or None if pose missing
        out     : pre-allocated float32 array of shape (_FEATURE_DIM,)
                  Written in-place — no allocation inside this function.

    Returns:
        flags : dict[str, bool] — True if the group was detected & written
    """
    out[:] = 0.0   # reset — caller reuses this buffer across frames
    can_norm = mid is not None

    flags = {"left_hand": False, "right_hand": False, "pose": False, "face": False}
    cursor = 0

    # ---- Left hand ------------------------------------------------
    if USE_LEFT_HAND:
        n = N_HAND_LANDMARKS * HAND_COORDS
        if results.left_hand_landmarks and can_norm:
            flags["left_hand"] = True
            for lm in results.left_hand_landmarks.landmark:
                out[cursor], out[cursor + 1] = _norm_xy(lm.x, lm.y, mid, width)
                cursor += 2
        else:
            cursor += n

    # ---- Right hand -----------------------------------------------
    if USE_RIGHT_HAND:
        n = N_HAND_LANDMARKS * HAND_COORDS
        if results.right_hand_landmarks and can_norm:
            flags["right_hand"] = True
            for lm in results.right_hand_landmarks.landmark:
                out[cursor], out[cursor + 1] = _norm_xy(lm.x, lm.y, mid, width)
                cursor += 2
        else:
            cursor += n

    # ---- Pose -----------------------------------------------------
    if USE_POSE:
        n = N_POSE_LANDMARKS * POSE_COORDS
        if results.pose_landmarks and can_norm:
            flags["pose"] = True
            for lm in results.pose_landmarks.landmark:
                out[cursor], out[cursor + 1], out[cursor + 2] = _norm_xyz(
                    lm.x, lm.y, lm.z, mid, width
                )
                cursor += 3
        else:
            cursor += n

    # ---- Face -----------------------------------------------------
    if USE_FACE:
        n = len(FACE_INDICES) * FACE_COORDS   # FACE_COORDS, not the literal 2
        if results.face_landmarks and can_norm:
            flags["face"] = True
            lms = results.face_landmarks.landmark
            for idx in FACE_INDICES:
                lm = lms[idx]
                out[cursor], out[cursor + 1] = _norm_xy(lm.x, lm.y, mid, width)
                cursor += 2
        else:
            cursor += n

    return flags


# ============================================================================
# Missing-frame fill
# ============================================================================

def _fill_missing(sequence: np.ndarray, no_anchor: np.ndarray) -> np.ndarray:
    """
    Forward-fill then backward-fill frames where pose was absent (no
    normalization anchor), so the model sees plausible landmark values
    rather than hard zeros at those positions.

    Args:
        sequence  : (T, D) float32, modified in-place
        no_anchor : (T,) bool — True where pose was missing

    Returns:
        sequence (same array, modified in-place)

    If ALL frames are missing (entire clip has no pose detection), the
    array stays zero-filled and the clip should be flagged in metadata.
    This function does not drop the clip — the caller decides.
    """
    T = len(sequence)
    if not no_anchor.any():
        return sequence   # fast path: nothing to fill

    if no_anchor.all():
        return sequence   # nothing to fill from — leave as zeros

    # Forward fill: propagate last good frame forward
    last_good: np.ndarray | None = None
    for t in range(T):
        if not no_anchor[t]:
            last_good = sequence[t]   # no copy — we reference in-place
        elif last_good is not None:
            sequence[t] = last_good

    # Backward fill: fix any leading missing frames
    # (only runs if the clip starts with missing pose)
    if no_anchor[0]:
        # Find first good frame
        first_good_t = int(np.argmin(no_anchor))   # first False
        first_good = sequence[first_good_t]
        for t in range(first_good_t):
            sequence[t] = first_good

    return sequence


# ============================================================================
# Video reading
# ============================================================================

def read_frames(video_path: Path) -> tuple[list[np.ndarray], float, int]:
    """
    Decode all frames from a video file.

    Respects MAX_FRAMES and ENABLE_TRUNCATION from landmark_config:
      - If ENABLE_TRUNCATION is True, stops reading after MAX_FRAMES frames.
      - If ENABLE_TRUNCATION is False, MAX_FRAMES acts only as a hard safety
        ceiling to abort clearly-broken files (e.g. corrupt videos reporting
        50,000 frames).

    Args:
        video_path : path to the .mp4 clip

    Returns:
        frames              : list of BGR uint8 ndarrays
        fps                 : reported frame rate (defaults to 25.0 if unreadable)
        total_frames_in_file: CAP_PROP_FRAME_COUNT (may be inaccurate for some
                              containers — treat as approximate)

    Raises:
        IOError  : if the file cannot be opened
        RuntimeError : if no frames could be decoded
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    try:
        while True:
            if len(frames) >= MAX_FRAMES:
                # Hard ceiling — abort regardless of ENABLE_TRUNCATION
                log.warning(
                    f"  {video_path.stem}: hit MAX_FRAMES={MAX_FRAMES} ceiling "
                    f"(total_in_file={total_frames_in_file}) — truncating"
                )
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if ENABLE_TRUNCATION and len(frames) >= MAX_FRAMES:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    return frames, fps, total_frames_in_file


# ============================================================================
# Sequence extraction (MediaPipe + normalization)
# ============================================================================

def extract_sequence(
    frames: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """
    Run MediaPipe Holistic on each frame and build the landmark sequence.

    Args:
        frames : list of BGR uint8 ndarrays from read_frames()

    Returns:
        sequence     : float32 ndarray of shape (T, feature_dim()),
                       variable length — NOT padded
        missing      : dict of per-group missing-frame counts
                       keys: "pose", "left_hand", "right_hand", "face"
        no_anchor    : bool ndarray of shape (T,) — True where pose was
                       absent (landmark values forward-filled from neighbours)
    """
    T = len(frames)
    D = _FEATURE_DIM

    # Pre-allocate the full sequence array — written in-place per frame
    sequence   = np.zeros((T, D), dtype=np.float32)
    no_anchor  = np.zeros(T, dtype=bool)
    frame_buf  = np.zeros(D, dtype=np.float32)   # reused per frame, no alloc in loop

    missing = {"pose": 0, "left_hand": 0, "right_hand": 0, "face": 0}

    with _mp_holistic.Holistic(
        static_image_mode=False,      # tracking mode: faster, maintains state
        model_complexity=1,           # 0=lite, 1=full, 2=heavy — 1 is the balance
        smooth_landmarks=True,        # temporal smoothing across frames
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for t, bgr_frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)

            mid, width = get_shoulder_anchor(results.pose_landmarks)
            if mid is None:
                no_anchor[t] = True

            flags = _extract_frame(results, mid, width, frame_buf)
            sequence[t] = frame_buf   # copy the filled buffer into the sequence row

            if not flags["pose"]:       missing["pose"] += 1
            if not flags["left_hand"]:  missing["left_hand"] += 1
            if not flags["right_hand"]: missing["right_hand"] += 1
            if not flags["face"]:       missing["face"] += 1

    _fill_missing(sequence, no_anchor)

    return sequence, missing, no_anchor


# ============================================================================
# Top-level per-clip processor
# ============================================================================

def process_clip(video_path: Path) -> tuple[np.ndarray | None, dict]:
    """
    Extract, normalize, and return the landmark sequence for one video clip.

    Saves variable-length output: shape (T, feature_dim()), where T is the
    actual number of decoded frames.  No padding is applied here — padding
    to a uniform length is the DataLoader's responsibility.

    Args:
        video_path : path to the .mp4 clip

    Returns:
        sequence : float32 ndarray of shape (T, feature_dim()), or None on failure
        meta     : dict of per-clip statistics for the metadata CSV

    Raises:
        Nothing — all exceptions are caught and returned in meta["error"].
        Callers can always check `sequence is None` for failure.
    """
    t_start = time.perf_counter()

    # --- read ---
    try:
        frames, fps, total_frames_in_file = read_frames(video_path)
    except Exception as exc:
        return None, {
            "error": str(exc),
            "fps": 0.0,
            "frame_count": 0,
            "processed_frame_count": 0,
            "total_frames_in_file": 0,
            "saved_frames": 0,
            "missing_pose_frames": 0,
            "missing_left_hand_frames": 0,
            "missing_right_hand_frames": 0,
            "missing_face_frames": 0,
            "missing_pose_pct": 0.0,
            "missing_left_hand_pct": 0.0,
            "missing_right_hand_pct": 0.0,
            "missing_face_pct": 0.0,
            "no_anchor_frames": 0,
            "normalization_success": False,
            "extraction_time_s": round(time.perf_counter() - t_start, 3),
            "config_hash": _CONFIG_HASH,
            "pipeline_version": _PIPELINE_VERSION,
            "normalization": _NORMALIZATION,
        }

    # --- extract ---
    try:
        sequence, missing, no_anchor = extract_sequence(frames)
    except Exception as exc:
        return None, {
            "error": str(exc),
            "fps": round(fps, 2),
            "frame_count": len(frames),
            "processed_frame_count": 0,
            "total_frames_in_file": total_frames_in_file,
            "saved_frames": 0,
            "missing_pose_frames": 0,
            "missing_left_hand_frames": 0,
            "missing_right_hand_frames": 0,
            "missing_face_frames": 0,
            "missing_pose_pct": 0.0,
            "missing_left_hand_pct": 0.0,
            "missing_right_hand_pct": 0.0,
            "missing_face_pct": 0.0,
            "no_anchor_frames": 0,
            "normalization_success": False,
            "extraction_time_s": round(time.perf_counter() - t_start, 3),
            "config_hash": _CONFIG_HASH,
            "pipeline_version": _PIPELINE_VERSION,
            "normalization": _NORMALIZATION,
        }

    T = len(sequence)
    norm_success = not no_anchor.all()

    meta = {
        "error": "",
        "fps": round(fps, 2),
        "frame_count": len(frames),           # frames read from disk
        "processed_frame_count": T,           # frames MediaPipe processed
        "total_frames_in_file": total_frames_in_file,
        "saved_frames": T,                    # shape[0] of the saved .npy
        "missing_pose_frames":       missing["pose"],
        "missing_left_hand_frames":  missing["left_hand"],
        "missing_right_hand_frames": missing["right_hand"],
        "missing_face_frames":       missing["face"],
        "missing_pose_pct":       round(missing["pose"]       / T * 100, 1),
        "missing_left_hand_pct":  round(missing["left_hand"]  / T * 100, 1),
        "missing_right_hand_pct": round(missing["right_hand"] / T * 100, 1),
        "missing_face_pct":       round(missing["face"]       / T * 100, 1),
        "no_anchor_frames": int(no_anchor.sum()),
        "normalization_success": norm_success,
        "extraction_time_s": round(time.perf_counter() - t_start, 3),
        "config_hash": _CONFIG_HASH,
        "pipeline_version": _PIPELINE_VERSION,
        "normalization": _NORMALIZATION,
    }

    return sequence, meta


# ============================================================================
# Label map
# ============================================================================

def build_label_map(wlasl_json_path: Path) -> tuple[dict, dict]:
    """
    Build video_id → annotation info and gloss → label_id maps from WLASL JSON.

    Label IDs are assigned in JSON appearance order (not alphabetical) to
    preserve compatibility with WLASL's official train/val/test splits.

    Returns:
        video_id_to_info : {video_id: {"gloss": str, "label_id": int, "split": str}}
        label_map        : {gloss: label_id}
    """
    with open(wlasl_json_path, "r") as f:
        data = json.load(f)

    label_map: dict[str, int] = {}
    video_id_to_info: dict[str, dict] = {}

    for entry in data:
        gloss = entry["gloss"]
        if gloss not in label_map:
            label_map[gloss] = len(label_map)
        label_id = label_map[gloss]

        for inst in entry.get("instances", []):
            video_id_to_info[inst["video_id"]] = {
                "gloss":    gloss,
                "label_id": label_id,
                "split":    inst.get("split", "unknown"),
            }

    return video_id_to_info, label_map


# ============================================================================
# Main
# ============================================================================

def main(args: argparse.Namespace) -> None:
    Path("data/metadata").mkdir(parents=True, exist_ok=True)

    # Log feature config at the start of every run — makes logs self-contained
    log.info(f"config_hash: {_CONFIG_HASH}")
    log.info(f"ENABLE_TRUNCATION: {ENABLE_TRUNCATION}, MAX_FRAMES: {MAX_FRAMES}")
    log.info("Feature vector breakdown:")
    for group, info in feature_breakdown().items():
        if group == "total_dims":
            log.info(f"  {'total':<12}: {info} dims")
        else:
            log.info(
                f"  {group:<12}: {info['n_landmarks']} landmarks "
                f"× {info['coords']} coords = {info['dims']} dims"
            )

    log.info(f"Loading WLASL JSON from {args.json}")
    video_id_to_info, label_map = build_label_map(Path(args.json))
    log.info(
        f"  {len(label_map)} unique glosses, "
        f"{len(video_id_to_info)} total instances"
    )

    # Save label map (gloss → integer id) for reference
    label_map_path = Path(args.metadata).parent / "label_map.csv"
    with open(label_map_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gloss", "label_id"])
        writer.writeheader()
        for gloss, lid in sorted(label_map.items(), key=lambda x: x[1]):
            writer.writerow({"gloss": gloss, "label_id": lid})
    log.info(f"Label map → {label_map_path}")

    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir    = Path(args.videos)
    video_files  = sorted(video_dir.glob("*.mp4"))
    total        = len(video_files)
    log.info(f"Found {total} .mp4 files in {video_dir}")

    # Metadata CSV — kept open throughout so rows flush incrementally
    # (safe to inspect mid-run and survive a crash without losing progress)
    metadata_path = Path(args.metadata)
    meta_fields = [
        "video_id", "gloss", "label_id", "split",
        "fps", "frame_count", "processed_frame_count", "total_frames_in_file",
        "saved_frames",
        "missing_pose_frames",      "missing_pose_pct",
        "missing_left_hand_frames", "missing_left_hand_pct",
        "missing_right_hand_frames","missing_right_hand_pct",
        "missing_face_frames",      "missing_face_pct",
        "no_anchor_frames", "normalization_success",
        "extraction_time_s", "pipeline_version", "normalization", "config_hash",
        "npy_path", "error",
    ]
    meta_file   = open(metadata_path, "w", newline="")
    meta_writer = csv.DictWriter(meta_file, fieldnames=meta_fields)
    meta_writer.writeheader()

    succeeded = 0
    failed    = 0
    skipped   = 0

    try:
        for i, video_path in enumerate(video_files, 1):
            video_id = video_path.stem
            npy_path = output_dir / f"{video_id}.npy"

            info     = video_id_to_info.get(video_id, {})
            gloss    = info.get("gloss",    "UNKNOWN")
            label_id = info.get("label_id", -1)
            split    = info.get("split",    "unknown")

            # Resume support: skip clips already on disk
            if npy_path.exists():
                skipped += 1
                log.debug(f"[{i}/{total}] {video_id}: already extracted, skipping")
                continue

            log.info(f"[{i}/{total}] {video_id} ({gloss}) ...")

            # process_clip catches all exceptions internally —
            # a bad clip never kills the run
            sequence, meta = process_clip(video_path)

            row = {
                "video_id": video_id,
                "gloss":    gloss,
                "label_id": label_id,
                "split":    split,
                "npy_path": str(npy_path) if sequence is not None else "",
                **meta,
            }

            if sequence is not None:
                # Variable-length save — shape (T, feature_dim()), NO padding
                np.save(npy_path, sequence)
                succeeded += 1
                log.info(
                    f"  → saved {sequence.shape}  "
                    f"pose_miss={meta['missing_pose_pct']}%  "
                    f"lhand_miss={meta['missing_left_hand_pct']}%  "
                    f"rhand_miss={meta['missing_right_hand_pct']}%  "
                    f"t={meta['extraction_time_s']}s"
                )
            else:
                failed += 1
                log.warning(f"  → FAILED: {meta.get('error', 'unknown')}")

            meta_writer.writerow(row)
            meta_file.flush()   # ensures CSV is readable if the run is interrupted

    finally:
        # Guaranteed to close even on KeyboardInterrupt or unexpected exception
        meta_file.close()

    log.info("=" * 60)
    log.info("Extraction complete.")
    log.info(f"  Total    : {total}")
    log.info(f"  Succeeded: {succeeded}")
    log.info(f"  Skipped  : {skipped} (already existed on disk)")
    log.info(f"  Failed   : {failed}")
    log.info(f"  Metadata : {metadata_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from WLASL clips"
    )
    parser.add_argument(
        "--videos",
        default="data/raw_videos",
        help="Directory of downloaded .mp4 clips",
    )
    parser.add_argument(
        "--json",
        default="data/WLASL_v0.3.json",
        help="WLASL annotation JSON",
    )
    parser.add_argument(
        "--output",
        default="data/landmarks",
        help="Directory to save .npy files (variable-length, one per clip)",
    )
    parser.add_argument(
        "--metadata",
        default="data/metadata/clip_metadata.csv",
        help="Output metadata CSV (written incrementally, safe to inspect mid-run)",
    )
    args = parser.parse_args()
    main(args)