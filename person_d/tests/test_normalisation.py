import importlib.util
import os

import numpy as np
import pytest

from normalisation import compute_head_center, compute_head_size, normalize_landmarks

# `io.py` collides with the stdlib `io` module name, so load it explicitly.
_io_spec = importlib.util.spec_from_file_location(
    "person_d_stub_io",
    os.path.join(os.path.dirname(__file__), "..", "utils", "data_io.py"),
)
_io_module = importlib.util.module_from_spec(_io_spec)
_io_spec.loader.exec_module(_io_module)
load_recording = _io_module.load_recording

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "person_c", "sample_landmarks")

FAKE_META = {
    "segments": {
        "face": {"start": 0, "end": 3, "count": 4},
        "pose": {"start": 4, "end": 5, "count": 2},
        "left_hand": {"start": 6, "end": 6, "count": 1},
        "right_hand": {"start": 7, "end": 7, "count": 1},
    }
}


def _toy_landmarks():
    # 1 frame, 8 landmarks total (4 face, 2 pose, 1 left hand, 1 right hand)
    landmarks = np.zeros((1, 8, 3), dtype=np.float32)
    # Face forms a unit square from (0,0) to (1,1) -> center (0.5,0.5), diag sqrt(2)
    landmarks[0, 0] = [0.0, 0.0, 0.0]
    landmarks[0, 1] = [1.0, 0.0, 0.0]
    landmarks[0, 2] = [1.0, 1.0, 0.0]
    landmarks[0, 3] = [0.0, 1.0, 0.0]
    # pose + hands somewhere else
    landmarks[0, 4] = [0.5, 0.5, 0.1]
    landmarks[0, 5] = [2.0, 2.0, 0.2]
    landmarks[0, 6] = [0.5, 1.5, 0.0]
    landmarks[0, 7] = [1.5, 0.5, 0.0]
    return landmarks


def test_compute_head_center_and_size():
    landmarks = _toy_landmarks()
    face = landmarks[:, 0:4, :]

    center = compute_head_center(face)
    size = compute_head_size(face)

    np.testing.assert_allclose(center[0], [0.5, 0.5])
    np.testing.assert_allclose(size[0], np.sqrt(2), rtol=1e-5)


def test_normalize_landmarks_shape_preserved():
    landmarks = _toy_landmarks()
    out = normalize_landmarks(landmarks, FAKE_META)
    assert out.shape == landmarks.shape


def test_normalize_landmarks_correct_values():
    landmarks = _toy_landmarks()
    out = normalize_landmarks(landmarks, FAKE_META)

    diag = np.sqrt(2)
    # pose landmark 4: (0.5,0.5) -> centered (0,0) -> scaled (0,0)
    np.testing.assert_allclose(out[0, 4, :2], [0.0, 0.0], atol=1e-5)
    # pose landmark 5: (2,2) -> centered (1.5,1.5) -> scaled by diag
    np.testing.assert_allclose(out[0, 5, :2], [1.5 / diag, 1.5 / diag], atol=1e-5)
    # z scaled by head size too
    np.testing.assert_allclose(out[0, 5, 2], 0.2 / diag, atol=1e-5)


def test_all_nan_face_produces_nan_frame():
    landmarks = _toy_landmarks()
    landmarks[0, 0:4, :] = np.nan  # wipe out the whole face group

    out = normalize_landmarks(landmarks, FAKE_META)
    assert np.isnan(out[0]).all()


def test_degenerate_zero_head_size_produces_nan_frame():
    landmarks = _toy_landmarks()
    # Collapse all face points to the same location -> head size 0.
    landmarks[0, 0:4, :2] = [0.3, 0.3]

    out = normalize_landmarks(landmarks, FAKE_META, min_head_size=1e-4)
    assert np.isnan(out[0]).all()


def test_partial_face_nan_still_computes_center_and_size():
    landmarks = _toy_landmarks()
    landmarks[0, 3] = np.nan  # drop one of four face points, three remain

    out = normalize_landmarks(landmarks, FAKE_META)
    assert not np.isnan(out[0, 4]).any()  # pose landmark still normalisable


def test_valid_landmarks_never_nan_when_head_size_valid():
    landmarks = _toy_landmarks()
    out = normalize_landmarks(landmarks, FAKE_META)
    # None of the originally-valid landmarks should become NaN.
    original_valid = ~np.isnan(landmarks).any(axis=2)
    assert not np.isnan(out[original_valid]).any()


@pytest.mark.parametrize(
    "name",
    [
        "sample01_normal_signing",
        "sample02_fast_signing",
        "sample05_low_light_degradation",
    ],
)
def test_on_sample_recordings_shape_preserved(name):
    rec = load_recording(SAMPLE_DIR, name)
    out = normalize_landmarks(rec.landmarks, rec.metadata)
    assert out.shape == rec.landmarks.shape


def test_on_sample_recording_no_inf_values():
    rec = load_recording(SAMPLE_DIR, "sample01_normal_signing")
    out = normalize_landmarks(rec.landmarks, rec.metadata)
    finite_or_nan = np.isnan(out) | np.isfinite(out)
    assert finite_or_nan.all()
