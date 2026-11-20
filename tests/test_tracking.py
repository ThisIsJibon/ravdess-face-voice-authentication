"""Tests for the facial-tracking study (no real data needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from mbproject.tracking import (AU_COLUMNS, SHAPE_COLUMNS, X_COLUMNS, Y_COLUMNS, extraction_settings,
                                features_are_current, normalize_landmarks, summarize_recording)
from mbproject.tracking_eval import Embedding, check_main_run, check_trial_index, save_plot_scores
from mbproject.tracking_report import eer_method_note, plot_example_scores


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


def test_no_consecutive_good_frames_raises_a_clear_error():
    # Every other frame is low confidence: enough good frames, but none are neighbours,
    # so AU velocity would be the mean of nothing (NaN).
    table = fake_tracking_csv()
    table.loc[1::2, "confidence"] = 0.1
    with pytest.raises(ValueError, match="next to each other"):
        summarize_recording(table, min_confidence=0.8, min_frames=10)


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


def test_misordered_trial_index_stops_the_run(tmp_path):
    # If the saved face/voice rows were in a different order, fusion would add up
    # scores of different recordings. The check must catch a swap of two rows.
    probes = pd.DataFrame({"recording_id": ["a", "b", "c"], "actor": [1, 2, 3]})
    actors = np.array([1, 2, 3])
    genuine = probes.actor.to_numpy()[:, None] == actors[None, :]
    path = tmp_path / "development_trial_index.npz"
    np.savez(path, probe_recording_ids=np.array(["a", "b", "c"]), probe_actors=np.array([1, 2, 3]),
             claim_actors=actors, genuine=genuine)
    check_trial_index(path, probes, actors, genuine)          # same order: no error
    np.savez(path, probe_recording_ids=np.array(["b", "a", "c"]), probe_actors=np.array([2, 1, 3]),
             claim_actors=actors, genuine=genuine[[1, 0, 2]])
    with pytest.raises(ValueError):
        check_trial_index(path, probes, actors, genuine)



def test_saved_example_scores_produce_all_plots(tmp_path):
    # Save example scores the way run_tracking does, then draw the report plots from them.
    genuine = np.eye(24, dtype=bool).repeat(20, axis=0)          # 480 probes x 24 claimed actors
    rng = np.random.default_rng(0)
    # Genuine scores a bit higher than impostor scores, like a working system.
    scores = {name: np.clip(rng.normal(0.5, 0.1, genuine.shape) + 0.3 * genuine, 0, 1)
              for name in ["face", "voice", "au", "au+voice", "tracking"]}
    save_plot_scores(tmp_path / "plot_scores.npz", genuine, scores)
    figures = tmp_path / "figures"
    figures.mkdir()
    plot_example_scores(tmp_path / "plot_scores.npz", figures, seed=73, reference_k=16, best_label="PDM shape")
    expected = ["score_distributions.png", "roc_det.png"] + [
        f"hw3_original/{kind}_({system}).png" for system in ["au", "au+voice"]
        for kind in ["score_distribution_plot", "det_curve", "roc_curve"]]
    for name in expected:
        assert (figures / name).is_file(), f"missing {name}"


def test_plot_scores_with_the_wrong_shape_are_rejected(tmp_path):
    # A score matrix that does not line up with the labels must stop the run.
    genuine = np.eye(3, dtype=bool)
    with pytest.raises(ValueError):
        save_plot_scores(tmp_path / "plot_scores.npz", genuine, {"au": np.zeros((2, 3))})


def fake_tracking_config(tmp_path):
    """A config with a fake tracking archive and manifest, enough to describe an extraction."""
    (tmp_path / "RAVDESS").mkdir()
    (tmp_path / "RAVDESS" / "Tracking.zip").write_bytes(b"fake archive")
    (tmp_path / "data" / "tracking").mkdir(parents=True)
    (tmp_path / "data" / "manifest.csv").write_text("recording_id\n01\n")
    (tmp_path / "data" / "tracking" / "features.npz").write_bytes(b"")
    return {"dataset_dir": str(tmp_path / "RAVDESS"), "data_dir": str(tmp_path / "data"),
            "tracking": {"archive": "Tracking", "min_confidence": 0.8, "min_frames": 10}}


def test_saved_features_are_out_of_date_when_settings_change(tmp_path):
    config = fake_tracking_config(tmp_path)
    saved = tmp_path / "data" / "tracking" / "extraction.json"
    saved.write_text(json.dumps({"settings": extraction_settings(config)}))
    assert features_are_current(config)
    config["tracking"]["min_confidence"] = 0.95          # a changed setting must force re-extraction
    assert not features_are_current(config)
    config["tracking"]["min_confidence"] = 0.8
    (tmp_path / "data" / "manifest.csv").write_text("recording_id\n02\n")   # so must a changed manifest
    assert not features_are_current(config)


def test_main_run_with_different_settings_stops_the_run(tmp_path):
    config = {"enrollment_seeds": [73], "enrollment_sizes": [1, 4, 16], "degradation_enrollment_sizes": [4],
              "audio_snr_db": [None, 0], "split_seed": 73}
    with pytest.raises(FileNotFoundError):                # main run not finished yet
        check_main_run(tmp_path, config)
    (tmp_path / "run_complete.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "config_resolved.json").write_text(json.dumps(config))
    check_main_run(tmp_path, config)                      # same settings: fine
    changed = {**config, "enrollment_sizes": [1, 4, 8]}  # e.g. main scores at k = 16, tracking at k = 8
    with pytest.raises(ValueError, match="enrollment_sizes"):
        check_main_run(tmp_path, changed)


def test_changed_zip_contents_of_the_same_size_are_noticed(tmp_path):
    config = fake_tracking_config(tmp_path)
    (tmp_path / "data" / "tracking" / "extraction.json").write_text(json.dumps({"settings": extraction_settings(config)}))
    assert features_are_current(config)
    (tmp_path / "RAVDESS" / "Tracking.zip").write_bytes(b"FAKE ARCHIVE")      # same size, new contents
    assert not features_are_current(config)


def test_changed_csv_in_an_extracted_folder_is_noticed(tmp_path):
    config = fake_tracking_config(tmp_path)
    (tmp_path / "RAVDESS" / "Tracking.zip").unlink()                         # use a folder instead of a ZIP
    folder = tmp_path / "RAVDESS" / "Tracking" / "Actor_01"
    folder.mkdir(parents=True)
    (folder / "01-01-01-01-01-01-01.csv").write_text("frame,confidence\n1,0.9\n")
    (tmp_path / "data" / "tracking" / "extraction.json").write_text(json.dumps({"settings": extraction_settings(config)}))
    assert features_are_current(config)
    (folder / "01-01-01-01-01-01-01.csv").write_text("frame,confidence\n1,0.5\n")
    assert not features_are_current(config)


def test_eer_note_reports_the_real_gap_between_methods():
    # HW3 says 10%, the exact ROC EER says 2%: the note must say 8.00 points, not "they agree".
    summary = pd.DataFrame({"representation": ["lda"], "k": [4], "hw3_eer_mean": [0.10], "eer_mean": [0.02]})
    fusion = pd.DataFrame({"k": [4], "audio_snr_db": ["clean"], "hw3_eer_mean": [0.03], "eer_mean": [0.03]})
    note = eer_method_note(summary, fusion, 4)
    assert "8.00 percentage points" in note
    assert "agree" not in note
