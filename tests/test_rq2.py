"""Tests for the RQ2 weight sweep (no real data needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from mbproject.common import unit
from mbproject.dataset import make_splits, parse_name
from mbproject.experiment import fuse, run
from mbproject.features import feature_signature
from mbproject.rq2_weights import (
    SYSTEM_LABELS,
    apply_minmax,
    dprime,
    fit_minmax,
    run_rq2,
    select_weight,
    settings,
)


def test_minmax_maps_the_dev_range_to_0_1_and_clips_test_scores():
    bounds = fit_minmax(np.array([0.2, 0.6, 1.0]))
    assert bounds == (0.2, 1.0)
    np.testing.assert_allclose(apply_minmax(np.array([0.0, 0.2, 0.6, 1.0, 1.2]), bounds), [0, 0, 0.5, 1, 1])


def test_minmax_rejects_constant_dev_scores():
    with pytest.raises(ValueError):
        fit_minmax(np.full(5, 0.7))


def separable_dev_scores(seed=0):
    """Face separates perfectly; voice is weaker. Fusion at 0.5 still has zero errors."""
    rng = np.random.default_rng(seed)
    genuine = np.r_[np.ones(100, bool), np.zeros(1000, bool)]
    face = np.where(genuine, rng.normal(0.9, 0.01, genuine.size), rng.normal(0.3, 0.01, genuine.size))
    voice = np.where(genuine, rng.normal(0.65, 0.03, genuine.size), rng.normal(0.35, 0.03, genuine.size))
    return face, voice, genuine


def test_zero_error_ties_are_broken_by_dprime_not_by_falling_back_to_half():
    face, voice, genuine = separable_dev_scores()
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Every weight has zero dev errors, which is what made the original RQ2 tuning always pick 0.5.
    assert all(np.max(fuse(face, voice, w)[~genuine]) < np.min(fuse(face, voice, w)[genuine]) for w in weights)
    best = max(weights, key=lambda w: dprime(fuse(face, voice, w), genuine))
    assert best != 0.5
    assert select_weight(face, voice, genuine, weights, 0.001) == best


def test_errors_still_come_before_dprime():
    # A weight with fewer dev errors must win even if another weight has a larger d-prime.
    genuine = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    # Face: wide gap but one genuine score (0.30) below one impostor score (0.35), so EER > 0.
    face = np.r_[np.ones(19), 0.30, np.zeros(19), 0.35]
    # Voice: no errors, but the two groups are spread out, so d-prime is smaller.
    voice = np.r_[np.linspace(0.5, 0.9, 20), np.linspace(0.1, 0.49, 20)]
    assert dprime(face, genuine) > dprime(voice, genuine)
    assert select_weight(face, voice, genuine, [0.0, 1.0], 0.05) == 0.0


def test_floating_point_noise_does_not_break_the_closest_to_half_tie():
    # Identical face and voice scores make every weight tie on EER, FRR and d-prime.
    # 0.3 and 0.7 are equally close to 0.5, but abs(0.7 - 0.5) < abs(0.3 - 0.5) in floats.
    genuine = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    scores = np.r_[np.linspace(0.4, 0.9, 50), np.linspace(0.1, 0.6, 50)]
    assert select_weight(scores, scores, genuine, [0.7, 0.3], 0.05) == 0.3


@pytest.mark.parametrize("bad", [
    {"face_weights": [0.5, 0.1]}, {"face_weights": [0.0, 1.0]}, {"face_weights": [0.0, 1.2]},
    {"far_target": 0}, {"enrollment_sizes": []}, {"enrollment_sizes": [0]}, {"enrollment_sizes": [-1]},
    {"enrollment_sizes": [1.5]}, {"enrollment_sizes": [1, 1]}, {"enrollment_sizes": [True]},
    {"face_weight": [0.5]},  # typo for face_weights
])
def test_settings_reject_unusable_values(bad):
    with pytest.raises(ValueError):
        settings({"rq2": bad})


def test_settings_default_when_the_block_is_missing():
    assert settings({}) == {"enrollment_sizes": [1, 2, 4], "far_target": 0.001,
                            "face_weights": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}


def fake_study(tmp_path, face_noise=3.0, voice_noise=4.0):
    """A made-up study: 24 actors x 60 recordings with clustered fake embeddings.

    The noise levels give roughly 10-35% EER, so mistakes in scoring or labels
    would show up as clearly different numbers.
    """
    rows = []
    for actor in range(1, 25):
        for emotion in range(1, 9):
            for intensity in ([1] if emotion == 1 else [1, 2]):
                for statement in (1, 2):
                    for rep in (1, 2):
                        rows.append(parse_name(f"03-01-{emotion:02}-{intensity:02}-{statement:02}-{rep:02}-{actor:02}.wav"))
    frame = make_splits(pd.DataFrame(rows), 73)
    config = {"data_dir": str(tmp_path / "data"), "model_dir": str(tmp_path / "models"),
              "results_dir": str(tmp_path / "results"), "split_seed": 73, "enrollment_seeds": [73, 101],
              "audio_snr_db": [None], "face_blur_sigma": [0], "frame_fractions": [0.5],
              "rq2": {"enrollment_sizes": [1, 2, 4], "face_weights": [0.0, 0.25, 0.5, 0.75, 1.0], "far_target": 0.01}}
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "manifest.json").write_text("{}")
    features = tmp_path / "data" / "features"
    features.mkdir(parents=True)
    frame.to_csv(tmp_path / "data" / "manifest.csv", index=False)
    signature = feature_signature(config)
    rng = np.random.default_rng(5)
    face_centers, voice_centers = rng.normal(size=(25, 128)), rng.normal(size=(25, 192))
    for row in frame.itertuples():
        np.savez(features / f"{row.recording_id}.npz",
                 face_sigma0=unit(face_centers[row.actor] + rng.normal(0, face_noise, 128)),
                 audio_clean=unit(voice_centers[row.actor] + rng.normal(0, voice_noise, 192)))
        (features / f"{row.recording_id}.json").write_text(json.dumps({"signature": signature}))
    return config


def test_full_run_writes_the_table_sweep_and_figure(tmp_path):
    config = fake_study(tmp_path)
    table = run_rq2(config)
    out = tmp_path / "results" / "rq2_weights"
    for name in ["rq2_table.csv", "weight_sweep.csv", "systems.csv", "selected_weights.csv",
                 "rq2_weight_sweep.png", "rq2_complete.json"]:
        assert (out / name).exists(), name
    assert len(table) == 3 * len(SYSTEM_LABELS)
    assert list(table.system[:len(SYSTEM_LABELS)]) == list(SYSTEM_LABELS)
    sweep = pd.read_csv(out / "weight_sweep.csv")
    assert len(sweep) == 2 * 3 * 5
    selected = pd.read_csv(out / "selected_weights.csv")
    assert selected.selected_face_weight.between(0, 1, inclusive="neither").all()
    assert sweep.groupby(["seed", "k"]).selected.sum().eq(1).all()
    # The fake data is hard enough to give real errors, but far better than chance (50%).
    voice = table.loc[table.system == "voice"].set_index("k").hw3_eer_pct
    assert (voice > 1).all() and (voice < 45).all()
    systems = pd.read_csv(out / "systems.csv")
    # Min-max is a straight-line rescale: the exact EER of voice-only/face-only equals the sweep ends.
    for system, weight in [("voice", 0.0), ("face", 1.0)]:
        a = systems.loc[systems.system == system].set_index(["seed", "k"]).eer.sort_index()
        b = sweep.loc[sweep.face_weight == weight].set_index(["seed", "k"]).eer.sort_index()
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    assert json.loads((out / "rq2_complete.json").read_text())["status"] == "complete"


def test_unnormalized_rows_reproduce_the_main_pipeline(tmp_path):
    # Same templates, probes and claims as experiment.run, so threshold-free metrics must match exactly.
    config = fake_study(tmp_path)
    config.update(enrollment_sizes=[1, 2, 4], degradation_enrollment_sizes=[1], far_targets=[0.01, 0.001, 0.0001],
                  face_weight_candidates=[0.5], bootstrap_replicates=10)
    run(config)
    run_rq2(config)
    main = pd.read_csv(tmp_path / "results" / "metrics.csv")
    main = main.loc[(main.family == "clean_enrollment") & (main.k_face == main.k_voice)]
    main = main.assign(k=main.k_face, system=main.system.replace({"fusion_fixed": "fusion_raw_equal"}))
    rq2 = pd.read_csv(tmp_path / "results" / "rq2_weights" / "systems.csv")
    merged = rq2.merge(main, on=["seed", "k", "system"], suffixes=("_rq2", "_main"))
    assert len(merged) == 2 * 3 * 3  # seeds x k x (voice, face, unnormalized 50/50)
    for metric in ["eer", "hw3_eer", "dprime"]:
        np.testing.assert_allclose(merged[f"{metric}_rq2"], merged[f"{metric}_main"], rtol=0, atol=1e-12)


def test_test_probes_never_influence_normalization_or_weight_choice(tmp_path):
    config = fake_study(tmp_path)
    run_rq2(config)
    before = pd.read_csv(tmp_path / "results" / "rq2_weights" / "selected_weights.csv")
    frame = pd.read_csv(tmp_path / "data" / "manifest.csv", dtype={"recording_id": str})
    rng = np.random.default_rng(9)
    for recording in frame.loc[frame.split == "test", "recording_id"]:
        np.savez(tmp_path / "data" / "features" / f"{recording}.npz",
                 face_sigma0=unit(rng.normal(size=128)), audio_clean=unit(rng.normal(size=192)))
    config["results_dir"] = str(tmp_path / "results_scrambled")
    run_rq2(config)
    after = pd.read_csv(tmp_path / "results_scrambled" / "rq2_weights" / "selected_weights.csv")
    pd.testing.assert_frame_equal(before, after)


def test_run_stops_on_stale_features(tmp_path):
    config = fake_study(tmp_path)
    config["split_seed"] = 74  # changes the feature signature, so the cache is stale
    with pytest.raises(ValueError, match="Stale"):
        run_rq2(config)


def test_run_stops_on_missing_features(tmp_path):
    config = fake_study(tmp_path)
    next((tmp_path / "data" / "features").glob("*.npz")).unlink()
    with pytest.raises(ValueError, match="Missing"):
        run_rq2(config)
