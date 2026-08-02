import importlib.util
import os

import numpy as np
import pytest

from hold_last_value import hold_last_value

# `io.py` collides with the stdlib `io` module name, so load it explicitly.
_io_spec = importlib.util.spec_from_file_location(
    "person_d_stub_io",
    os.path.join(os.path.dirname(__file__), "..", "utils", "data_io.py"),
)
_io_module = importlib.util.module_from_spec(_io_spec)
_io_spec.loader.exec_module(_io_module)
load_recording = _io_module.load_recording

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "person_c", "sample_landmarks")

def test_valid_landmarks_unchanged():
    landmarks = np.random.default_rng(0).uniform(0, 1, size=(10, 543, 3)).astype(
        np.float32
    )
    out = hold_last_value(landmarks)
    np.testing.assert_array_equal(out, landmarks)


def test_fills_from_last_valid():
    landmarks = np.zeros((5, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 2.0, 3.0]
    landmarks[1] = np.nan
    landmarks[2] = np.nan
    landmarks[3] = [9.0, 9.0, 9.0]
    landmarks[4] = np.nan

    out = hold_last_value(landmarks)

    np.testing.assert_array_equal(out[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out[1, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out[2, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out[3, 0], [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(out[4, 0], [9.0, 9.0, 9.0])


def test_leading_nan_stays_nan():
    landmarks = np.zeros((3, 1, 3), dtype=np.float32)
    landmarks[0] = np.nan
    landmarks[1] = np.nan
    landmarks[2] = [5.0, 5.0, 5.0]

    out = hold_last_value(landmarks)

    assert np.isnan(out[0, 0]).all()
    assert np.isnan(out[1, 0]).all()
    np.testing.assert_array_equal(out[2, 0], [5.0, 5.0, 5.0])


def test_max_hold_frames_budget_respected():
    landmarks = np.zeros((6, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    landmarks[1:6] = np.nan  # 5 consecutive missing frames

    out = hold_last_value(landmarks, max_hold_frames=2)

    # Held for frames 1 and 2 (within budget of 2), then NaN from frame 3 on.
    np.testing.assert_array_equal(out[1, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[2, 0], [1.0, 1.0, 1.0])
    assert np.isnan(out[3, 0]).all()
    assert np.isnan(out[4, 0]).all()
    assert np.isnan(out[5, 0]).all()


def test_max_hold_frames_zero_never_holds():
    landmarks = np.zeros((3, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    landmarks[1] = np.nan
    landmarks[2] = np.nan

    out = hold_last_value(landmarks, max_hold_frames=0)

    np.testing.assert_array_equal(out[0, 0], [1.0, 1.0, 1.0])
    assert np.isnan(out[1, 0]).all()
    assert np.isnan(out[2, 0]).all()


def test_hold_resets_after_new_valid_value():
    landmarks = np.zeros((5, 1, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]
    landmarks[1] = np.nan
    landmarks[2] = [2.0, 2.0, 2.0]
    landmarks[3] = np.nan
    landmarks[4] = np.nan

    out = hold_last_value(landmarks, max_hold_frames=1)

    np.testing.assert_array_equal(out[1, 0], [1.0, 1.0, 1.0])  # held once, ok
    np.testing.assert_array_equal(out[2, 0], [2.0, 2.0, 2.0])  # new valid value
    np.testing.assert_array_equal(out[3, 0], [2.0, 2.0, 2.0])  # held once again, ok
    assert np.isnan(out[4, 0]).all()  # exceeds budget of 1


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        hold_last_value(np.zeros((5, 543, 2)))


def test_negative_max_hold_frames_raises():
    with pytest.raises(ValueError):
        hold_last_value(np.zeros((5, 543, 3)), max_hold_frames=-1)


@pytest.mark.parametrize(
    "name",
    [
        "sample01_normal_signing",
        "sample03_left_hand_occlusion",
        "sample04_right_hand_occlusion",
        "sample05_low_light_degradation",
    ],
)
def test_on_sample_recordings_shape_and_dtype_preserved(name):
    rec = load_recording(SAMPLE_DIR, name)
    out = hold_last_value(rec.landmarks, max_hold_frames=15)

    assert out.shape == rec.landmarks.shape
    assert out.dtype == rec.landmarks.dtype


def test_on_sample_recording_reduces_nan_count():
    rec = load_recording(SAMPLE_DIR, "sample03_left_hand_occlusion")
    out = hold_last_value(rec.landmarks)  # unlimited hold

    nan_before = np.isnan(rec.landmarks).sum()
    nan_after = np.isnan(out).sum()

    assert nan_after < nan_before


def test_on_sample_recording_never_changes_valid_values():
    rec = load_recording(SAMPLE_DIR, "sample01_normal_signing")
    out = hold_last_value(rec.landmarks, max_hold_frames=5)

    valid_mask = ~np.isnan(rec.landmarks)
    np.testing.assert_array_equal(out[valid_mask], rec.landmarks[valid_mask])
