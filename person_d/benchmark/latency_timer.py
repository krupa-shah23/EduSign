"""
latency_timer.py

Measures per-stage latency (file loading, hold-last-value processing,
normalisation, and total) for the Month 1 pipeline, using
time.perf_counter().

No model / confidence-estimation / evaluation code here — purely timing
instrumentation around the other three Person D modules.
"""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hold_last_value import hold_last_value
from normalisation import normalize_landmarks

# Load local io.py explicitly (rather than `from io import ...`) since its
# filename collides with Python's standard-library `io` module, which is
# already present in sys.modules by the time this file is imported.
_io_spec = importlib.util.spec_from_file_location(
    "person_d_stub_io", os.path.join(os.path.dirname(__file__), "io.py")
)
_io_module = importlib.util.module_from_spec(_io_spec)
_io_spec.loader.exec_module(_io_module)
load_recording = _io_module.load_recording


class Stopwatch:
    """
    Minimal reusable timing context manager built on time.perf_counter().

        with Stopwatch() as sw:
            do_work()
        print(sw.elapsed)
    """

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self.elapsed: Optional[float] = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start


def time_pipeline(
    directory: str,
    name: str,
    *,
    max_hold_frames: Optional[int] = None,
    min_head_size: float = 1e-4,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run load -> hold_last_value -> normalize on one recording, timing each
    stage with time.perf_counter().

    Returns
    -------
    timings: dict with keys 'load_time', 'hold_last_value_time',
             'normalisation_time', 'total_time' (all in seconds).
    normalized_landmarks: (T, N, 3) array, the final pipeline output.
    """
    total_start = time.perf_counter()

    with Stopwatch() as sw_load:
        recording = load_recording(directory, name)
    load_time = sw_load.elapsed

    with Stopwatch() as sw_hold:
        filled = hold_last_value(recording.landmarks, max_hold_frames=max_hold_frames)
    hold_time = sw_hold.elapsed

    with Stopwatch() as sw_norm:
        normalized = normalize_landmarks(
            filled, recording.metadata, min_head_size=min_head_size
        )
    norm_time = sw_norm.elapsed

    total_time = time.perf_counter() - total_start

    timings = {
        "load_time": load_time,
        "hold_last_value_time": hold_time,
        "normalisation_time": norm_time,
        "total_time": total_time,
    }
    return timings, normalized
