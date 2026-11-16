"""Tests for the facial-tracking study (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from mbproject.tracking import AU_COLUMNS, SHAPE_COLUMNS, X_COLUMNS, Y_COLUMNS, normalize_landmarks, summarize_recording
from mbproject.tracking_eval import Embedding


def fake_tracking_csv(frames=40, confidence=0.95, seed=0):
    """A made-up OpenFace table with the columns our code reads."""
    rng = np.random.default_rng(seed)
    table = {"frame": np.arange(1, frames + 1), "confidence": np.full(frames, confidence)}
    for column in AU_COLUMNS:
        table[column] = rng.uniform(0, 3, frames)
    face = rng.normal(300, 30, (68, 2))
    for i in range(68):
        table[X_COLUMNS[i]] = face[i, 0] + rng.normal(0, 1, frames)
        table[Y_COLUMNS[i]] = face[i, 1] + rng.normal(0, 1, frames)
    for column in SHAPE_COLUMNS:
        table[column] = rng.normal(0, 1, frames)
    return pd.DataFrame(table)


def test_landmarks_ignore_head_position_tilt_and_distance():
    # Moving, rotating and scaling the whole face must not change the normalized landmarks.
    rng = np.random.default_rng(1)
    points = rng.normal(0, 50, (3, 68, 2))
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = 2.5 * points @ rotation.T + np.array([120.0, -40.0])
    original = normalize_landmarks(points[..., 0], points[..., 1])
    transformed = normalize_landmarks(moved[..., 0], moved[..., 1])
    assert np.allclose(original, transformed)


def test_summary_has_expected_sizes_and_drops_low_confidence_frames():
    table = fake_tracking_csv()
    table.loc[:4, "confidence"] = 0.1             # 5 bad frames
    blocks, quality = summarize_recording(table, min_confidence=0.8, min_frames=10)
    assert quality == {"frames": 40, "kept_frames": 35}
    sizes = {name: values.shape for name, values in blocks.items()}
    assert sizes == {"au_level": (17,), "au_motion": (17,), "au_velocity": (17,),
                     "lmk_shape": (136,), "lmk_motion": (136,), "pdm_shape": (34,)}


def test_too_few_good_frames_raises_an_error():
    with pytest.raises(ValueError):
        summarize_recording(fake_tracking_csv(confidence=0.5), min_confidence=0.8, min_frames=10)


def test_embedding_only_learns_from_training_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 10))
    actors = np.repeat(np.arange(6), 20)
    model = Embedding(lda_dims=4, shrinkage=0.1).fit(X[:60], actors[:60])   # first 3 actors only
    before = model.transform(X[60:])
    # Transforming other data must not change the fitted model.
    model.transform(X[60:] * 100)
    assert np.allclose(model.transform(X[60:]), before)
    # LDA gives at most (people - 1) = 2 dimensions for 3 training actors.
    assert model.transform(X).shape == (120, 2)
