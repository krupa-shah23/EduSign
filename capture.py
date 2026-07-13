"""
capture.py

Reads frames from either a webcam or a supplied video file and runs
MediaPipe Holistic on each frame.

This module has exactly one job: turn a video source into a stream of
MediaPipe Holistic results. It knows nothing about how those results get
saved to disk (see export_landmarks.py) and contains no model / training
code of any kind. It is an interface stub, not a research artifact.

Typical usage
-------------
    from capture import iter_holistic_results

    for frame_index, timestamp, results in iter_holistic_results(source=0):
        ...  # results is a MediaPipe Holistic result object

    for frame_index, timestamp, results in iter_holistic_results(
        source="path/to/video.mp4"
    ):
        ...
"""

from __future__ import annotations

import time
from typing import Generator, Optional, Tuple, Union

import cv2

# NOTE: mediapipe's legacy `solutions` API (which provides the combined
# Holistic model) is imported lazily inside iter_holistic_results() rather
# than at module load time. Some mediapipe installs (e.g. minimal / newer
# "Tasks"-only builds) don't ship `mediapipe.solutions`; deferring the
# import means this module — and export_landmarks.py, which imports it —
# can still be imported and used (e.g. for saving/loading pre-recorded
# landmark data) on such installs. The import only fails, loudly, at the
# point actual capture is attempted.


def iter_holistic_results(
    source: Union[int, str] = 0,
    *,
    max_frames: Optional[int] = None,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Generator[Tuple[int, float, "mp.solutions.holistic.Holistic", int, int], None, None]:
    """
    Yield (frame_index, timestamp_seconds, holistic_results, frame_width, frame_height) 
    for each frame of the given source.

    Parameters
    ----------
    source:
        `0` (or any int) for a webcam device index, or a path/string to a
        video file.
    max_frames:
        Stop after this many frames. `None` means "until the source ends"
        (a video file) or "until interrupted" (a live webcam).
    model_complexity, min_detection_confidence, min_tracking_confidence:
        Passed straight through to `mediapipe.solutions.holistic.Holistic`.

    Yields
    ------
    frame_index : int
        0-based frame counter, in capture order.
    timestamp_seconds : float
        Seconds since the first frame was read. For a video file this is
        derived from the file's own timestamps (`cv2.CAP_PROP_POS_MSEC`)
        when available, falling back to a frame-count / fps estimate.
    holistic_results :
        The raw object returned by `Holistic.process()`. Has
        `.face_landmarks`, `.pose_landmarks`, `.left_hand_landmarks`,
        `.right_hand_landmarks` attributes (each `None` if not detected).
    frame_width : int
        Width of the original captured frame in pixels.
    frame_height : int
        Height of the original captured frame in pixels.
    """
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except AttributeError as e:
        raise ImportError(
            "This mediapipe install does not expose `mediapipe.solutions` "
            "(the legacy Holistic API). Install a mediapipe build that "
            "includes solutions, e.g. `pip install mediapipe` on a "
            "supported platform (Linux x86_64 / macOS / Windows)."
        ) from e

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_wall_time = time.time()

    try:
        with mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as holistic:
            frame_index = 0
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                success, frame_bgr = cap.read()
                if not success:
                    break

                # MediaPipe expects RGB input.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)

                # Prefer the video file's own timestamp when reading a
                # file; fall back to wall-clock time for a live webcam.
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if isinstance(source, str) and pos_msec and pos_msec > 0:
                    timestamp = pos_msec / 1000.0
                elif isinstance(source, str):
                    timestamp = frame_index / fps
                else:
                    timestamp = time.time() - start_wall_time

                yield frame_index, timestamp, results, frame_width, frame_height
                frame_index += 1
    finally:
        cap.release()


def get_source_fps(source: Union[int, str]) -> float:
    """Return the fps reported by the capture device/file (best effort)."""
    cap = cv2.VideoCapture(source)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 30.0
    finally:
        cap.release()
