import importlib.util
import os

import numpy as np
import pytest

from latency_timer import Stopwatch, time_pipeline

# `io.py` collides with the stdlib `io` module name, so load it explicitly.
_io_spec = importlib.util.spec_from_file_location(
    "person_d_stub_io",
    os.path.join(os.path.dirname(__file__), "..", "utils", "data_io.py"),
)
_io_module = importlib.util.module_from_spec(_io_spec)
_io_spec.loader.exec_module(_io_module)
load_recording = _io_module.load_recording

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "person_c", "sample_landmarks")


def test_stopwatch_measures_nonnegative_elapsed():
    with Stopwatch() as sw:
        sum(range(10000))
    assert sw.elapsed is not None
    assert sw.elapsed >= 0.0


@pytest.mark.parametrize(
    "name",
    [
        "sample01_normal_signing",
        "sample03_left_hand_occlusion",
        "sample05_low_light_degradation",
    ],
)
def test_time_pipeline_returns_expected_keys(name):
    timings, normalized = time_pipeline(SAMPLE_DIR, name)

    for key in (
        "load_time",
        "hold_last_value_time",
        "normalisation_time",
        "total_time",
    ):
        assert key in timings
        assert isinstance(timings[key], float)
        assert timings[key] >= 0.0


def test_time_pipeline_total_at_least_sum_of_stages():
    timings, _ = time_pipeline(SAMPLE_DIR, "sample01_normal_signing")
    stage_sum = (
        timings["load_time"]
        + timings["hold_last_value_time"]
        + timings["normalisation_time"]
    )
    # total_time wraps the same three stages, so it should be >= their sum
    # (equal modulo negligible overhead of the outer perf_counter calls).
    assert timings["total_time"] >= stage_sum - 1e-6


def test_time_pipeline_output_shape_matches_source_recording():
    rec = load_recording(SAMPLE_DIR, "sample01_normal_signing")
    _, normalized = time_pipeline(SAMPLE_DIR, "sample01_normal_signing")
    assert normalized.shape == rec.landmarks.shape


def test_time_pipeline_respects_max_hold_frames():
    timings_a, out_a = time_pipeline(
        SAMPLE_DIR, "sample03_left_hand_occlusion", max_hold_frames=0
    )
    timings_b, out_b = time_pipeline(
        SAMPLE_DIR, "sample03_left_hand_occlusion", max_hold_frames=None
    )
    # Unlimited hold should leave no more NaNs than zero-hold.
    assert np.isnan(out_b).sum() <= np.isnan(out_a).sum()
