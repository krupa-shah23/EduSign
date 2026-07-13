"""
generate_sample_data.py

Generates the five sample recordings Person D needs to start work
immediately.

IMPORTANT — why this script exists:
This sandbox has no webcam and no sample video files available, so
capture.py / export_landmarks.py cannot be run against a real camera here.
Instead, this script synthesizes landmark tensors that are byte-for-byte
IN THE SAME FORMAT that export_landmarks.save_recording() would produce
(same shape, dtype, segment layout, NaN convention, metadata fields) so
Person D can build and test against the real interface contract today.

Each synthetic recording is built with save_recording() from
export_landmarks.py, so there is no separate/parallel format definition —
if the schema ever changes, both real and synthetic recordings pick up the
change automatically.

Once a real webcam or sample video is available, replace calls to this
script with `python export_landmarks.py <name> --video <path>` or
`--device 0`; nothing else in Person D's code needs to change, since the
output format is identical.

Scenarios generated (as required):
    1. sample01_normal_signing            - both hands present, steady motion
    2. sample02_fast_signing              - both hands present, higher-frequency motion
    3. sample03_left_hand_occlusion       - left hand drops out for a stretch of frames
    4. sample04_right_hand_occlusion      - right hand drops out for a stretch of frames
    5. sample05_low_light_degradation     - intermittent, noisy dropout across all groups
"""

from __future__ import annotations

import numpy as np

from export_landmarks import (
    NUM_LANDMARKS_TOTAL,
    NUM_FACE,
    NUM_POSE,
    NUM_LEFT_HAND,
    NUM_RIGHT_HAND,
    SEGMENTS,
    save_recording,
)

RNG = np.random.default_rng(seed=42)
FPS = 30.0


def _base_pose(num_frames: int) -> np.ndarray:
    """
    Build a plausible-looking (num_frames, 543, 3) tensor of normalised
    coordinates with gentle per-landmark motion, before any dropout is
    applied. This is purely for producing realistic-looking sample data —
    it is not derived from any real signing.
    """
    # Each landmark gets a fixed base position plus small smooth motion.
    base_xy = RNG.uniform(0.2, 0.8, size=(NUM_LANDMARKS_TOTAL, 2))
    base_z = RNG.uniform(-0.1, 0.1, size=(NUM_LANDMARKS_TOTAL, 1))
    base = np.concatenate([base_xy, base_z], axis=1)  # (543, 3)

    t = np.linspace(0, 2 * np.pi, num_frames)
    motion = 0.03 * np.sin(t)[:, None, None]  # (T, 1, 1)
    noise = RNG.normal(0, 0.002, size=(num_frames, NUM_LANDMARKS_TOTAL, 3))

    frames = base[None, :, :] + motion + noise
    return frames.astype(np.float32)


def _apply_dropout(
    landmarks: np.ndarray, segment: str, frame_start: int, frame_end: int
) -> None:
    """In-place: set the given segment to NaN for frames [frame_start, frame_end)."""
    seg = SEGMENTS[segment]
    landmarks[frame_start:frame_end, seg["start"] : seg["end"] + 1, :] = np.nan


def make_normal_signing(num_frames: int = 150) -> np.ndarray:
    return _base_pose(num_frames)


def make_fast_signing(num_frames: int = 150) -> np.ndarray:
    landmarks = _base_pose(num_frames)
    # Faster signing -> higher-frequency, larger-amplitude hand motion.
    t = np.linspace(0, 8 * np.pi, num_frames)
    fast_motion = 0.08 * np.sin(t)[:, None, None]
    for hand in ("left_hand", "right_hand"):
        seg = SEGMENTS[hand]
        landmarks[:, seg["start"] : seg["end"] + 1, :2] += fast_motion
    return landmarks


def make_left_hand_occlusion(num_frames: int = 150) -> np.ndarray:
    landmarks = _base_pose(num_frames)
    # Left hand missing for a contiguous stretch (~frames 60-100).
    _apply_dropout(landmarks, "left_hand", 60, 100)
    return landmarks


def make_right_hand_occlusion(num_frames: int = 150) -> np.ndarray:
    landmarks = _base_pose(num_frames)
    # Right hand missing for a contiguous stretch (~frames 40-90).
    _apply_dropout(landmarks, "right_hand", 40, 90)
    return landmarks


def make_low_light_degradation(num_frames: int = 150) -> np.ndarray:
    landmarks = _base_pose(num_frames)
    # Simulate flaky tracking: random short dropouts across all segments,
    # plus extra noise to represent poor tracking confidence even when
    # landmarks are present.
    landmarks += RNG.normal(0, 0.01, size=landmarks.shape).astype(np.float32)

    for segment in ("face", "pose", "left_hand", "right_hand"):
        num_dropouts = RNG.integers(4, 8)
        for _ in range(num_dropouts):
            start = int(RNG.integers(0, num_frames - 5))
            length = int(RNG.integers(1, 6))
            _apply_dropout(landmarks, segment, start, min(start + length, num_frames))

    return landmarks


def main(output_dir: str = "sample_landmarks") -> None:
    recordings = {
        "sample01_normal_signing": make_normal_signing(),
        "sample02_fast_signing": make_fast_signing(),
        "sample03_left_hand_occlusion": make_left_hand_occlusion(),
        "sample04_right_hand_occlusion": make_right_hand_occlusion(),
        "sample05_low_light_degradation": make_low_light_degradation(),
    }

    for name, landmarks in recordings.items():
        num_frames = landmarks.shape[0]
        timestamps = np.arange(num_frames, dtype=np.float64) / FPS
        save_recording(
            name,
            landmarks,
            timestamps,
            FPS,
            output_dir,
            image_width=1280,
            image_height=720,
            source_kind="synthetic_placeholder",
            source_path=None,
        )
        print(f"Wrote {name} ({num_frames} frames) to {output_dir}/")


if __name__ == "__main__":
    main()
