# Setup and Sample Regeneration Guide

## Prerequisites

The interface stub requires `mediapipe`, `opencv-python`, and `numpy`:

```bash
pip install mediapipe opencv-python numpy
```

## Regenerate Sample Data (with new metadata fields)

The sample data has been updated to include:
- `schema_version: 1`
- `image_width` and `image_height`

To regenerate the synthetic samples with these new fields:

```bash
python generate_sample_data.py
```

This will overwrite the existing samples in `sample_landmarks/` with new versions that include the additional metadata fields.

## Capture Real Webcam Sample

Once dependencies are installed, capture a real webcam recording:

```bash
python capture_real_webcam.py
```

**What this does:**
- Captures ~30 seconds (900 frames) from your default webcam
- Processes each frame through MediaPipe Holistic
- Saves the results as `real_webcam_signing_landmarks.npy`, `real_webcam_signing_timestamps.npy`, and `real_webcam_signing_metadata.json`

**Requirements:**
- A connected webcam
- Good lighting for MediaPipe to track reliably
- Position yourself so your face and hands are visible

**For shorter/longer captures**, edit the `max_frames` parameter in `capture_real_webcam.py`:
- 150 frames ≈ 5 seconds at 30 FPS
- 450 frames ≈ 15 seconds
- 900 frames ≈ 30 seconds (default)
- 1500 frames ≈ 50 seconds

## Alternative: Capture from Video File

To process an existing video file instead of using a webcam:

```bash
python export_landmarks.py <name> --video path/to/video.mp4 --output-dir sample_landmarks
```

Example:
```bash
python export_landmarks.py real_video_signing --video ~/signing_clips/sample.mp4 --output-dir sample_landmarks
```

## Verify the New Metadata

Check that the new fields are present in the metadata:

```python
import json

with open("sample_landmarks/real_webcam_signing_metadata.json") as f:
    meta = json.load(f)

print(f"Schema version: {meta['schema_version']}")
print(f"Resolution: {meta['image_width']}x{meta['image_height']}")
print(f"FPS: {meta['fps']}")
print(f"Number of frames: {meta['number_of_frames']}")
```

---

See `landmark_schema.md` for the full metadata specification and `README.md` for usage examples.
