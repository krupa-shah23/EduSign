# Landmark Interface Schema

This document is the **interface contract** between Person C's capture stub and
Person D's robustness module (hold-last-value, normalisation, latency timing).
Person D should be able to consume any `.npy`/`.json` pair produced by this
stub without reading any of Person C's code.

---

## 1. Tensor shape

Each recording is saved as a single NumPy array:

```
shape = (num_frames, 543, 3)
dtype = float32
```

- **Axis 0 (`num_frames`)** — one entry per video frame, in capture order.
- **Axis 1 (`543`)** — one entry per landmark point (see ordering below).
  543 = 468 (face) + 33 (pose) + 21 (left hand) + 21 (right hand). This is
  the standard total point count for MediaPipe Holistic and is fixed for
  every file this stub produces.
- **Axis 2 (`3`)** — the `(x, y, z)` coordinate of that landmark.
  MediaPipe's `visibility`/`presence` scores are **not** included, so every
  file has exactly this 3-channel shape regardless of landmark type.

There is no per-recording variation in shape other than `num_frames`. Person
D can safely assume `array.shape[1:] == (543, 3)` for every file.

## 2. Landmark ordering (axis 1)

The 543 points are concatenated in this fixed order:

| Segment      | Index range (inclusive) | Count | Source                         |
|--------------|--------------------------|-------|---------------------------------|
| Face         | 0 – 467                  | 468   | `results.face_landmarks`        |
| Pose         | 468 – 500                | 33    | `results.pose_landmarks`        |
| Left hand    | 501 – 521                | 21    | `results.left_hand_landmarks`   |
| Right hand   | 522 – 542                | 21    | `results.right_hand_landmarks`  |

Within each segment, landmark order matches MediaPipe Holistic's own
canonical index order (e.g. pose index 0 = nose, hand index 0 = wrist, etc.).
This is exactly the same ordering MediaPipe returns internally — Person C's
code does not re-order or re-index anything.

The same ordering is repeated as a flat list in each recording's metadata
JSON under `landmark_order`, so Person D's code can validate/slice
programmatically instead of hardcoding the table above.

## 3. Coordinate system

- Coordinates are **exactly as produced by MediaPipe Holistic** — normalised
  image coordinates in `[0, 1]`, origin at the top-left of the frame,
  `x` → right, `y` → down. `z` is MediaPipe's roughly-depth value, in the
  same relative scale MediaPipe uses (smaller = closer to camera, roughly
  hip-depth-scaled for pose; not true metric depth).
- No cropping, resizing, mirroring, or normalisation has been applied.
  (Face/head-size normalisation is explicitly Person D's job — see
  `README.md`.)

## 4. Missing-value convention

If a landmark group is not detected in a given frame (e.g. a hand leaves the
frame, occlusion, low light), **every coordinate for that group in that
frame is `NaN`** — not zero, not the last known value.

- Rationale: zero is a valid normalised coordinate (top-left corner) and
  would be silently misinterpreted as "hand at top-left." `NaN` is
  unambiguous and lets Person D's hold-last-value logic detect gaps with a
  simple `np.isnan(...)` check rather than guessing.
- Missing data is always **whole-group**: MediaPipe Holistic returns hand
  landmarks as a group of 21 points or nothing, so partial-hand NaNs never
  occur. Face and pose behave the same way (all-468 or all-NaN,
  all-33 or all-NaN).
- `missing_value_representation` in the metadata JSON always reads `"NaN"`
  as a machine-checkable confirmation of this convention.

## 5. Data types

| Field                | Type              |
|-----------------------|-------------------|
| Landmark tensor        | `float32`         |
| Frame index             | `int64` (0-based, stored in metadata + implicit via array row) |
| Timestamp                | `float64` seconds, stored in a companion `*_timestamps.npy` (see below) |

## 6. Files produced per recording

For a recording named `<name>`, three files are written into
`sample_landmarks/` (or wherever `export_landmarks.py` is pointed):

```
<name>_landmarks.npy     # shape (num_frames, 543, 3), float32
<name>_timestamps.npy    # shape (num_frames,), float64, seconds since recording start
<name>_metadata.json     # see structure below
```

Frame index is not stored separately — row `i` of both `.npy` files
corresponds to frame `i`, and `frame_index == i`.

### `<name>_metadata.json` structure

```json
{
  "schema_version": 1,
  "fps": 30.0,
  "number_of_frames": 150,
  "image_width": 1280,
  "image_height": 720,
  "landmark_order": ["face_0", "face_1", ..., "right_hand_20"],
  "coordinate_system": "normalized image coordinates (MediaPipe Holistic raw output), origin top-left, x-right, y-down, z relative depth",
  "missing_value_representation": "NaN",
  "segments": {
    "face":       {"start": 0,   "end": 467, "count": 468},
    "pose":       {"start": 468, "end": 500, "count": 33},
    "left_hand":  {"start": 501, "end": 521, "count": 21},
    "right_hand": {"start": 522, "end": 542, "count": 21}
  },
  "source": "webcam | video_file | synthetic_placeholder",
  "source_path": "path/to/video.mp4 or null for webcam/synthetic"
}
```

### Metadata field reference

| Field                          | Type    | Purpose                                                                 |
|--------------------------------|---------|-------------------------------------------------------------------------|
| `schema_version`               | int     | Version of this metadata schema (currently 1). Use for compatibility.   |
| `fps`                          | float   | Frames per second of the capture.                                       |
| `number_of_frames`             | int     | Total number of frames (T). Matches the axis-0 size of landmarks.       |
| `image_width`                  | int     | Width of the original captured frame in pixels.                         |
| `image_height`                 | int     | Height of the original captured frame in pixels.                        |
| `landmark_order`               | list    | Flat list of landmark names in tensor order (axis 1).                   |
| `coordinate_system`            | string  | Human-readable description of the coordinate convention.                |
| `missing_value_representation` | string  | "NaN" — indicates how missing landmarks are encoded.                    |
| `segments`                     | dict    | Start/end indices and counts for each landmark group (face/pose/hands).  |
| `source`                       | string  | How the recording was created: "webcam", "video_file", or "synthetic".  |
| `source_path`                  | string  | Path to the source video file (null for webcam or synthetic data).      |

## 7. How Person D should load a recording

```python
import json
import numpy as np

name = "sample01_normal_signing"

landmarks = np.load(f"sample_landmarks/{name}_landmarks.npy")   # (T, 543, 3)
timestamps = np.load(f"sample_landmarks/{name}_timestamps.npy")  # (T,)
with open(f"sample_landmarks/{name}_metadata.json") as f:
    meta = json.load(f)

assert landmarks.shape == (meta["number_of_frames"], 543, 3)

# Slice out a segment using the metadata, no hardcoding required:
seg = meta["segments"]["left_hand"]
left_hand = landmarks[:, seg["start"]:seg["end"] + 1, :]   # (T, 21, 3)

# Detect missing frames for hold-last-value logic:
missing_mask = np.isnan(left_hand).any(axis=(1, 2))  # (T,) bool
```

This is the entire contract. Nothing about model architecture, training, or
recognition is implied or required to consume these files.
