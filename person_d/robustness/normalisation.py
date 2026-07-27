"""
normalisation.py

Face/head-size-based normalisation for the (T, 543, 3) landmark tensors
exported by Person C.

For each frame:
    1. head_center = mean (x, y) of the face landmarks present that frame
    2. head_size    = bounding-box diagonal (x, y) of those face landmarks
    3. normalized   = (landmarks - head_center) / head_size
       (z is scaled by head_size but not re-centered, since MediaPipe's z
       is already roughly zero-centered on the hips/torso)

If a frame's face landmarks are entirely missing, or the computed head size
is degenerate (zero / near-zero), normalisation is undefined for that frame
and the whole frame is written out as NaN rather than risking a division
blow-up or silently-wrong scaling. This module does not attempt to guess a
substitute (e.g. holding a previous head size) — that is a hold-last-value
concern, not a normalisation concern, and callers should run
hold_last_value.py first if they want that behaviour.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

_DEFAULT_MIN_HEAD_SIZE = 1e-4


def compute_head_center(face_landmarks: np.ndarray) -> np.ndarray:
    """
    face_landmarks: (T, F, 3) array (just the face segment).
    Returns (T, 2) array of per-frame (x, y) centers, NaN where the whole
    face is missing that frame.
    """
    # nanmean over an all-NaN slice raises a warning and returns NaN, which
    # is exactly the behaviour we want here.
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(face_landmarks[:, :, :2], axis=1)


def compute_head_size(
    face_landmarks: np.ndarray, min_head_size: float = _DEFAULT_MIN_HEAD_SIZE
) -> np.ndarray:
    """
    face_landmarks: (T, F, 3) array (just the face segment).
    Returns (T,) array of per-frame bounding-box diagonals over (x, y).
    Frames with no valid face points, or a diagonal below `min_head_size`,
    are returned as NaN (degenerate / undefined head size).
    """
    import warnings

    xy = face_landmarks[:, :, :2]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mins = np.nanmin(xy, axis=1)  # (T, 2), NaN if all-NaN along axis 1
        maxs = np.nanmax(xy, axis=1)  # (T, 2)
    span = maxs - mins  # (T, 2)
    diag = np.sqrt(np.sum(span**2, axis=1))  # (T,)

    invalid = np.isnan(diag) | (diag < min_head_size)
    diag = diag.copy()
    diag[invalid] = np.nan
    return diag


def normalize_landmarks(
    landmarks: np.ndarray,
    metadata: Dict[str, Any],
    *,
    min_head_size: float = _DEFAULT_MIN_HEAD_SIZE,
) -> np.ndarray:
    """
    Normalise all landmarks (face, pose, both hands) by per-frame head
    center and head size, computed from the face segment.

    Parameters
    ----------
    landmarks:
        (T, N, 3) array, as produced by Person C's export (optionally
        already passed through hold_last_value). Not mutated.
    metadata:
        The metadata dict for this recording (from io.load_metadata),
        used to locate the face segment via metadata["segments"]["face"].
    min_head_size:
        Head sizes below this are treated as degenerate; the entire frame
        is output as NaN in that case.

    Returns
    -------
    (T, N, 3) array, same shape/dtype as input.
    """
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        raise ValueError(f"Expected shape (T, N, 3), got {landmarks.shape}")

    face_seg = metadata["segments"]["face"]
    face_landmarks = landmarks[:, face_seg["start"] : face_seg["end"] + 1, :]

    head_center = compute_head_center(face_landmarks)  # (T, 2)
    head_size = compute_head_size(face_landmarks, min_head_size=min_head_size)  # (T,)

    T, N, C = landmarks.shape
    out = np.full_like(landmarks, np.nan)

    valid_frames = ~np.isnan(head_size)
    if not valid_frames.any():
        return out

    hs = head_size[valid_frames].reshape(-1, 1, 1)  # (T_valid, 1, 1)
    center_xy = head_center[valid_frames].reshape(-1, 1, 2)  # (T_valid, 1, 2)

    frame_data = landmarks[valid_frames]  # (T_valid, N, 3)
    normed = frame_data.copy()
    normed[:, :, :2] = (frame_data[:, :, :2] - center_xy) / hs
    normed[:, :, 2] = frame_data[:, :, 2] / hs[:, :, 0]

    out[valid_frames] = normed
    return out
