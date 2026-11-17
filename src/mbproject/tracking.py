"""Step 1 of the tracking study: turn the RAVDESS facial-tracking CSVs into features.

The RAVDESS authors ran OpenFace 2.1.0 on every video and released one CSV per
video (FacialTracking_Actors_01-24.zip). Each CSV has one row per video frame with:
  - 17 action-unit (AU) intensities, e.g. AU12_r = lip corner puller (smile), 0 to 5
  - 68 facial landmark points (x_0..x_67, y_0..y_67) in pixels
  - 34 face-shape parameters (p_0..p_33)
  - a tracking confidence value

We summarize each video (about 110 frames) into a few fixed-length vectors
("blocks") so that every recording can be compared with every other one.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json

# The 17 AU intensity columns in the OpenFace CSVs.
AU_COLUMNS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r",
              "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
X_COLUMNS = [f"x_{i}" for i in range(68)]
Y_COLUMNS = [f"y_{i}" for i in range(68)]
SHAPE_COLUMNS = [f"p_{i}" for i in range(34)]

# In the 68-point landmark layout, points 36 and 45 are the outer eye corners.
RIGHT_EYE, LEFT_EYE = 36, 45

# What each feature block means (also written to data/tracking/extraction.json).
BLOCKS = {
    "au_level": "average of each AU over the video (how strong the expressions are)",
    "au_motion": "standard deviation of each AU (how much each AU changes)",
    "au_velocity": "average frame-to-frame change of each AU (how fast the face moves)",
    "lmk_shape": "average of the normalized landmarks (face geometry)",
    "lmk_motion": "standard deviation of the normalized landmarks (how the face shape moves)",
    "pdm_shape": "average of OpenFace's 34 shape parameters (compact face geometry)",
}


def find_tracking_source(config):
    """Return the tracking ZIP (or its unzipped folder) inside the RAVDESS folder."""
    name = config["tracking"]["archive"]
    zip_path = Path(config["dataset_dir"]) / f"{name}.zip"
    folder_path = Path(config["dataset_dir"]) / name
    if zip_path.exists():
        return zip_path
    if folder_path.is_dir():
        return folder_path
    raise FileNotFoundError(f"Put {zip_path.name} in {config['dataset_dir']} "
                            "(download: https://zenodo.org/records/3255102)")


def list_csv_files(source):
    """Map each CSV file name (e.g. '01-01-03-02-01-01-12.csv') to its location."""
    source = Path(source)
    if source.suffix == ".zip":
        names = zipfile.ZipFile(source).namelist()
    else:
        names = [str(p.relative_to(source)) for p in source.rglob("*.csv")]
    files = {}
    for name in names:
        base = Path(name).name
        # Skip folders and hidden macOS files like "__MACOSX/._name.csv".
        if base.endswith(".csv") and "__MACOSX" not in name and not base.startswith("._"):
            files[base] = name
    return files


def read_csv(source, files, recording_id):
    """Read the tracking CSV for one recording.

    Our recording_id drops the first field of the RAVDESS file name, so the
    tracking file for recording '01-03-02-01-01-12' is '01-01-03-02-01-01-12.csv'
    (the leading '01' means audio+video).
    """
    file_name = f"01-{recording_id}.csv"
    if file_name not in files:
        raise FileNotFoundError(f"Tracking file {file_name} is missing")
    source = Path(source)
    if source.suffix == ".zip":
        raw = zipfile.ZipFile(source).read(files[file_name])
    else:
        raw = (source / files[file_name]).read_bytes()
    table = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    return table


def normalize_landmarks(x, y):
    """Make landmarks comparable between frames and people.

    For every frame:
      1. move the midpoint between the outer eye corners to (0, 0)   -> removes head position
      2. rotate so the line between the eyes is horizontal            -> removes head tilt
      3. divide by the distance between the eyes                      -> removes camera distance
    Returns an array of shape (frames, 136): x and y of all 68 points.
    """
    points = np.stack([x, y], axis=-1).astype(float)          # (frames, 68, 2)
    right_eye = points[:, RIGHT_EYE]
    left_eye = points[:, LEFT_EYE]

    center = (right_eye + left_eye) / 2
    eye_vector = left_eye - right_eye
    eye_distance = np.linalg.norm(eye_vector, axis=1)
    angle = np.arctan2(eye_vector[:, 1], eye_vector[:, 0])

    # Rotation matrix for -angle, one per frame.
    cos, sin = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cos, -sin], [sin, cos]]).transpose(2, 0, 1)   # (frames, 2, 2)

    centered = points - center[:, None, :]
    rotated = np.einsum("fij,fpj->fpi", rotation, centered)
    scaled = rotated / eye_distance[:, None, None]
    return scaled.reshape(len(points), -1)


def summarize_recording(table, min_confidence, min_frames):
    """Turn one CSV (one row per frame) into the six feature blocks."""
    # Keep only frames where OpenFace was confident it found the face.
    good = table[table.confidence >= min_confidence].sort_values("frame")
    if len(good) < min_frames:
        raise ValueError(f"Only {len(good)} confident frames (need {min_frames})")

    au = good[AU_COLUMNS].to_numpy(float)                      # (frames, 17)

    # Velocity: change between neighbouring frames. Only use pairs of frames that
    # really are next to each other (a dropped frame would make a fake jump).
    neighbours = np.diff(good.frame.to_numpy()) == 1
    au_change = np.abs(np.diff(au, axis=0))[neighbours]

    landmarks = normalize_landmarks(good[X_COLUMNS].to_numpy(), good[Y_COLUMNS].to_numpy())
    shape = good[SHAPE_COLUMNS].to_numpy(float)

    blocks = {
        "au_level": au.mean(axis=0),
        "au_motion": au.std(axis=0),
        "au_velocity": au_change.mean(axis=0),
        "lmk_shape": landmarks.mean(axis=0),
        "lmk_motion": landmarks.std(axis=0),
        "pdm_shape": shape.mean(axis=0),
    }
    quality = {"frames": len(table), "kept_frames": len(good)}
    return blocks, quality


def extract_tracking(config):
    """Compute features for every recording in data/manifest.csv and save them.

    Output: data/tracking/features.npz (rows in manifest order), quality.csv,
    and extraction.json (settings and frame counts).
    """
    settings = config["tracking"]
    data_dir = Path(config["data_dir"])
    manifest = pd.read_csv(data_dir / "manifest.csv", dtype={"recording_id": str})
    source = find_tracking_source(config)
    files = list_csv_files(source)
    print(f"Reading {source.name}: {len(files)} CSV files", flush=True)

    features = {name: [] for name in BLOCKS}
    quality = []
    for count, row in enumerate(manifest.itertuples(), 1):
        table = read_csv(source, files, row.recording_id)
        blocks, q = summarize_recording(table, settings["min_confidence"], settings["min_frames"])
        for name in BLOCKS:
            features[name].append(blocks[name])
        quality.append({"recording_id": row.recording_id, **q})
        if count % 240 == 0:
            print(f"  {count}/{len(manifest)} recordings", flush=True)

    features = {name: np.array(rows) for name, rows in features.items()}
    for name, array in features.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Feature block {name} has NaN or infinite values")

    out = data_dir / "tracking"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", recording_ids=manifest.recording_id.to_numpy(str), **features)
    quality = pd.DataFrame(quality)
    quality.to_csv(out / "quality.csv", index=False)
    write_json(out / "extraction.json", {
        "source": source.name,
        "recordings": len(manifest),
        "min_confidence": settings["min_confidence"],
        "frames_total": int(quality.frames.sum()),
        "frames_kept": int(quality.kept_frames.sum()),
        "fewest_frames_kept": int(quality.kept_frames.min()),
        "blocks": {name: {"size": int(features[name].shape[1]), "meaning": BLOCKS[name]} for name in BLOCKS},
    })
    print(f"Saved features for {len(manifest)} recordings "
          f"({quality.kept_frames.sum()} of {quality.frames.sum()} frames kept)", flush=True)


def load_tracking(config):
    """Load the saved feature blocks (run extract_tracking first)."""
    path = Path(config["data_dir"]) / "tracking" / "features.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}
