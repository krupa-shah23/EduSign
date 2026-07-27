"""
hold_last_value.py

Configurable hold-last-value interpolation for the (T, 543, 3) landmark
tensors exported by Person C.

A landmark is considered "missing" at frame t if any of its (x, y, z)
coordinates is NaN (Person C's export always writes whole-group NaN, so in
practice all three coordinates are NaN together, but this module checks
generically rather than assuming that invariant).

Fill rule: for each landmark independently, walk forward through time. When
a missing frame is found, fill it with the most recent valid value for that
landmark, as long as no more than `max_hold_frames` consecutive frames have
been held since that valid value. Once the hold budget is exceeded, the
landmark is left as NaN until a new valid value appears. Landmarks that have
never had a valid value yet are left as NaN (nothing to hold from).
Landmarks that are not missing are left completely unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def hold_last_value(
    landmarks: np.ndarray,
    max_hold_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Fill missing landmarks by holding the last valid value.

    Parameters
    ----------
    landmarks:
        (T, N, 3) array, as produced by Person C's export. Not mutated.
    max_hold_frames:
        Maximum number of consecutive frames a value may be held for.
        `None` (default) means hold indefinitely until a new valid value
        appears. `0` means never hold (equivalent to returning a copy).

    Returns
    -------
    (T, N, 3) array, same shape/dtype as input, with eligible NaNs filled.
    """
    if landmarks.ndim != 3 or landmarks.shape[2] != 3:
        raise ValueError(f"Expected shape (T, N, 3), got {landmarks.shape}")
    if max_hold_frames is not None and max_hold_frames < 0:
        raise ValueError("max_hold_frames must be >= 0 or None")

    T, N, C = landmarks.shape
    out = landmarks.copy()

    # missing[t, n] is True if landmark n is missing (any NaN coord) at frame t
    missing = np.isnan(landmarks).any(axis=2)

    last_valid = np.full((N, C), np.nan, dtype=landmarks.dtype)
    has_valid = np.zeros(N, dtype=bool)
    hold_count = np.zeros(N, dtype=np.int64)

    for t in range(T):
        miss_t = missing[t]
        valid_t = ~miss_t

        # Update the "last seen" cache with this frame's valid landmarks,
        # and reset their hold counters.
        if valid_t.any():
            last_valid[valid_t] = landmarks[t, valid_t]
            has_valid[valid_t] = True
            hold_count[valid_t] = 0

        # For missing landmarks that have a valid value to hold from,
        # fill in as long as we're within the hold budget.
        fillable = miss_t & has_valid
        if fillable.any():
            hold_count[fillable] += 1
            if max_hold_frames is None:
                within_budget = fillable
            else:
                within_budget = fillable & (hold_count <= max_hold_frames)
            if within_budget.any():
                out[t, within_budget] = last_valid[within_budget]

    return out
