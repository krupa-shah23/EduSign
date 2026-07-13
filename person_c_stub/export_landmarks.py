"""
export_landmarks.py

Converts a stream of MediaPipe Holistic results (from capture.py) into the
fixed-shape landmark tensor format described in landmark_schema.md, and
writes it to disk as:

    <name>_landmarks.npy    (num_frames, 543, 3) float32
    <name>_timestamps.npy   (num_frames,)        float64
    <name>_metadata.json

No transformer / training / recognition code lives here. This file's only
job is capture-results -> fixed-shape-array -> disk.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np

from capture import iter_holistic_results, get_source_fps

# --- Fixed schema constants (see landmark_schema.md) -----------------------

NUM_FACE = 468
NUM_POSE = 33
NUM_LEFT_HAND = 21
NUM_RIGHT_HAND = 21
NUM_LANDMARKS_TOTAL = NUM_FACE + NUM_POSE + NUM_LEFT_HAND + NUM_RIGHT_HAND  # 543

SEGMENTS = {
    "face": {"start": 0, "end": NUM_FACE - 1, "count": NUM_FACE},
    "pose": {"start": NUM_FACE, "end": NUM_FACE + NUM_POSE - 1, "count": NUM_POSE},
    "left_hand": {
        "start": NUM_FACE + NUM_POSE,
        "end": NUM_FACE + NUM_POSE + NUM_LEFT_HAND - 1,
        "count": NUM_LEFT_HAND,
    },
    "right_hand": {
        "start": NUM_FACE + NUM_POSE + NUM_LEFT_HAND,
        "end": NUM_FACE + NUM_POSE + NUM_LEFT_HAND + NUM_RIGHT_HAND - 1,
        "count": NUM_RIGHT_HAND,
    },
}

COORDINATE_SYSTEM = (
    "normalized image coordinates (MediaPipe Holistic raw output), "
    "origin top-left, x-right, y-down, z relative depth"
)
MISSING_VALUE_REPRESENTATION = "NaN"


def _landmark_group_to_array(landmark_list, expected_count: int) -> np.ndarray:
    """
    Convert one MediaPipe landmark group (or None) into a (expected_count, 3)
    float32 array. Missing groups become all-NaN, per the schema.
    """
    if landmark_list is None:
        return np.full((expected_count, 3), np.nan, dtype=np.float32)

    points = landmark_list.landmark
    if len(points) != expected_count:
        # Defensive: MediaPipe should always return a fixed count per group,
        # but guard against unexpected input rather than silently misaligning.
        raise ValueError(
            f"Expected {expected_count} landmarks, got {len(points)}"
        )

    arr = np.empty((expected_count, 3), dtype=np.float32)
    for i, point in enumerate(points):
        arr[i, 0] = point.x
        arr[i, 1] = point.y
        arr[i, 2] = point.z
    return arr


def results_to_frame_array(results) -> np.ndarray:
    """
    Convert a single MediaPipe Holistic result into a (543, 3) float32 array
    following the fixed segment ordering: face, pose, left_hand, right_hand.
    """
    face = _landmark_group_to_array(results.face_landmarks, NUM_FACE)
    pose = _landmark_group_to_array(results.pose_landmarks, NUM_POSE)
    left_hand = _landmark_group_to_array(results.left_hand_landmarks, NUM_LEFT_HAND)
    right_hand = _landmark_group_to_array(results.right_hand_landmarks, NUM_RIGHT_HAND)
    return np.concatenate([face, pose, left_hand, right_hand], axis=0)


def build_landmark_order() -> List[str]:
    """Flat, human-readable label per row of axis 1, in schema order."""
    order = []
    order += [f"face_{i}" for i in range(NUM_FACE)]
    order += [f"pose_{i}" for i in range(NUM_POSE)]
    order += [f"left_hand_{i}" for i in range(NUM_LEFT_HAND)]
    order += [f"right_hand_{i}" for i in range(NUM_RIGHT_HAND)]
    return order


def record_from_source(
    source,
    *,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Run capture + holistic extraction on `source` and return
    (landmarks, timestamps, fps, image_width, image_height) as in-memory arrays, 
    without writing anything to disk. Useful for testing or custom pipelines.
    """
    frame_arrays: List[np.ndarray] = []
    timestamps: List[float] = []
    image_width: int = 0
    image_height: int = 0

    for frame_index, timestamp, results, frame_width, frame_height in iter_holistic_results(
        source, max_frames=max_frames
    ):
        frame_arrays.append(results_to_frame_array(results))
        timestamps.append(timestamp)
        image_width = frame_width
        image_height = frame_height

    if not frame_arrays:
        landmarks = np.empty((0, NUM_LANDMARKS_TOTAL, 3), dtype=np.float32)
    else:
        landmarks = np.stack(frame_arrays, axis=0)

    fps = get_source_fps(source)
    return landmarks, np.array(timestamps, dtype=np.float64), fps, image_width, image_height


def save_recording(
    name: str,
    landmarks: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    output_dir: str,
    *,
    image_width: int = 0,
    image_height: int = 0,
    source_kind: str = "video_file",
    source_path: Optional[str] = None,
) -> None:
    """
    Write <name>_landmarks.npy, <name>_timestamps.npy and
    <name>_metadata.json into output_dir, per landmark_schema.md.
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (NUM_LANDMARKS_TOTAL, 3):
        raise ValueError(
            f"landmarks must have shape (T, {NUM_LANDMARKS_TOTAL}, 3), "
            f"got {landmarks.shape}"
        )
    if landmarks.shape[0] != timestamps.shape[0]:
        raise ValueError("landmarks and timestamps must have matching frame counts")

    os.makedirs(output_dir, exist_ok=True)

    landmarks = landmarks.astype(np.float32, copy=False)
    timestamps = timestamps.astype(np.float64, copy=False)

    np.save(os.path.join(output_dir, f"{name}_landmarks.npy"), landmarks)
    np.save(os.path.join(output_dir, f"{name}_timestamps.npy"), timestamps)

    metadata = {
        "schema_version": 1,
        "fps": float(fps),
        "number_of_frames": int(landmarks.shape[0]),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "landmark_order": build_landmark_order(),
        "coordinate_system": COORDINATE_SYSTEM,
        "missing_value_representation": MISSING_VALUE_REPRESENTATION,
        "segments": SEGMENTS,
        "source": source_kind,
        "source_path": source_path,
    }
    with open(os.path.join(output_dir, f"{name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def export_from_video(
    name: str,
    video_path: str,
    output_dir: str,
    *,
    max_frames: Optional[int] = None,
) -> None:
    """Convenience wrapper: capture a video file and save the recording."""
    landmarks, timestamps, fps, image_width, image_height = record_from_source(
        video_path, max_frames=max_frames
    )
    save_recording(
        name,
        landmarks,
        timestamps,
        fps,
        output_dir,
        image_width=image_width,
        image_height=image_height,
        source_kind="video_file",
        source_path=video_path,
    )


def export_from_webcam(
    name: str,
    output_dir: str,
    *,
    max_frames: int,
    device_index: int = 0,
) -> None:
    """
    Convenience wrapper: capture `max_frames` frames from a webcam and save
    the recording. `max_frames` is required here (unlike video files) since
    a live webcam has no natural end point.
    """
    landmarks, timestamps, fps, image_width, image_height = record_from_source(
        device_index, max_frames=max_frames
    )
    save_recording(
        name,
        landmarks,
        timestamps,
        fps,
        output_dir,
        image_width=image_width,
        image_height=image_height,
        source_kind="webcam",
        source_path=None,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export MediaPipe Holistic landmarks from a webcam or video file."
    )
    parser.add_argument("name", help="Base name for the output files")
    parser.add_argument(
        "--video", help="Path to a video file (omit to use the webcam)"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Webcam device index (default 0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to capture (required for webcam, optional for video)",
    )
    parser.add_argument(
        "--output-dir", default="sample_landmarks", help="Output directory"
    )
    args = parser.parse_args()

    if args.video:
        export_from_video(
            args.name, args.video, args.output_dir, max_frames=args.max_frames
        )
    else:
        if args.max_frames is None:
            parser.error("--max-frames is required when capturing from a webcam")
        export_from_webcam(
            args.name,
            args.output_dir,
            max_frames=args.max_frames,
            device_index=args.device,
        )

    print(f"Saved recording '{args.name}' to {args.output_dir}/")
