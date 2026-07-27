"""
io.py

Loading helpers for the landmark recordings exported by Person C
(see person_c_stub/landmark_schema.md for the format contract).

No writing/export logic here — this module only reads recordings Person C
already produced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, NamedTuple

import numpy as np


class Recording(NamedTuple):
    landmarks: np.ndarray    # (T, 543, 3) float32
    timestamps: np.ndarray   # (T,) float64
    metadata: Dict[str, Any]


def load_landmarks(directory: str, name: str) -> np.ndarray:
    """Load `<name>_landmarks.npy` as a (T, 543, 3) float32 array."""
    path = os.path.join(directory, f"{name}_landmarks.npy")
    return np.load(path)


def load_timestamps(directory: str, name: str) -> np.ndarray:
    """Load `<name>_timestamps.npy` as a (T,) float64 array."""
    path = os.path.join(directory, f"{name}_timestamps.npy")
    return np.load(path)


def load_metadata(directory: str, name: str) -> Dict[str, Any]:
    """Load `<name>_metadata.json` as a dict."""
    path = os.path.join(directory, f"{name}_metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def load_recording(directory: str, name: str) -> Recording:
    """Load landmarks + timestamps + metadata for one recording."""
    landmarks = load_landmarks(directory, name)
    timestamps = load_timestamps(directory, name)
    metadata = load_metadata(directory, name)

    if landmarks.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"'{name}': landmarks has {landmarks.shape[0]} frames but "
            f"timestamps has {timestamps.shape[0]}"
        )
    if landmarks.shape[0] != metadata.get("number_of_frames"):
        raise ValueError(
            f"'{name}': landmarks has {landmarks.shape[0]} frames but "
            f"metadata says {metadata.get('number_of_frames')}"
        )

    return Recording(landmarks=landmarks, timestamps=timestamps, metadata=metadata)


def get_segment(landmarks: np.ndarray, metadata: Dict[str, Any], segment: str) -> np.ndarray:
    """Slice out one landmark group (face/pose/left_hand/right_hand) using metadata."""
    seg = metadata["segments"][segment]
    return landmarks[:, seg["start"] : seg["end"] + 1, :]
