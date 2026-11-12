from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mbproject.common import ROOT, read_config, stable_seed, unit
from mbproject.dataset import enrollment_order, make_splits, parse_name
from mbproject.experiment import cosine_scores, fuse
from mbproject.features import add_noise


def synthetic_manifest():
    rows = []
    for actor in range(1,25):
        for emotion in range(1,9):
            for intensity in ([1] if emotion == 1 else [1,2]):
                for statement in (1,2):
                    for rep in (1,2):
                        rows.append(parse_name(f"03-01-{emotion:02}-{intensity:02}-{statement:02}-{rep:02}-{actor:02}.wav"))
    return pd.DataFrame(rows)


def test_splits_prevent_repetition_and_modality_leakage():
    frame = make_splits(synthetic_manifest(),73)
    assert len(frame) == 1440
    assert frame.groupby(["actor","split"]).size().eq(20).all()
    assert frame.groupby("stimulus_group").split.nunique().eq(1).all()
    pd.testing.assert_frame_equal(frame,make_splits(synthetic_manifest(),73))
    assert not frame.split.equals(make_splits(synthetic_manifest(),101).split)


def test_enrollment_orders_are_nested_reproducible_and_exclude_queries():
    frame = make_splits(synthetic_manifest(),73)
    orders = enrollment_order(frame,73)
    assert orders == enrollment_order(frame,73)
    assert orders != enrollment_order(frame,101)
    queries = set(frame.loc[frame.split != "enroll","recording_id"])
    for ids in orders.values():
        assert len(ids) == len(set(ids)) == 20
        assert set(ids[:1]) <= set(ids[:4]) <= set(ids[:16])
        assert not set(ids)&queries


@pytest.mark.parametrize("snr",[20,10,0,-5])
def test_controlled_noise_has_requested_measured_snr(snr):
    signal = np.sin(np.linspace(0,100,16000)).astype(np.float32)
    degraded,measured = add_noise(signal,snr,73)
    assert measured == pytest.approx(snr,abs=1e-5)
    np.testing.assert_array_equal(degraded,add_noise(signal,snr,73)[0])
    np.testing.assert_allclose(signal,np.sin(np.linspace(0,100,16000)).astype(np.float32))


def test_clean_audio_is_unchanged_and_silence_cannot_be_given_snr():
    signal = np.array([.1,.2,.3],dtype=np.float32)
    np.testing.assert_array_equal(add_noise(signal,None,73)[0],signal)
    with pytest.raises(ValueError):
        add_noise(np.zeros(100),10,73)


def test_cosine_matching_and_fusion_have_expected_direction_and_scale():
    vectors = np.array([[1,0],[0,1],[-1,0]],dtype=np.float32)
    scores = cosine_scores(vectors,vectors)
    np.testing.assert_allclose(np.diag(scores),1)
    assert scores[0,1] == .5
    assert scores[0,2] == 0
    assert fuse(.8,.4,.25) == pytest.approx(.5)
    with pytest.raises(ValueError):
        fuse(.8,.4,1.1)


def test_zero_embedding_fails_instead_of_becoming_valid_template():
    with pytest.raises(ValueError):
        unit(np.zeros(128))


def test_filename_pairing_uses_actor_and_every_stimulus_field():
    audio = parse_name("03-01-05-02-01-02-24.wav")
    video = parse_name("01-01-05-02-01-02-24.mp4")
    assert audio["recording_id"] == video["recording_id"]
    assert audio["actor"] == 24
    assert audio["stimulus_group"] == "01-05-02-01"
    with pytest.raises(ValueError):
        parse_name("03-02-05-02-01-02-24.wav")


@pytest.mark.skipif(not (ROOT/"data/media/Actor_17/01-01-02-02-01-02-17.mp4").exists(),
                    reason="Optional real-data regression requires prepared RAVDESS and models")
def test_single_actor_clip_survives_small_false_detection_on_hair():
    from mbproject.features import FaceEncoder
    encoder = FaceEncoder(read_config())
    crops,metadata = encoder.crops(ROOT/"data/media/Actor_17/01-01-02-02-01-02-17.mp4",[.25,.5,.75])
    assert len(crops) == 3
    assert all(c.shape == (112,112,3) for c in crops)
    assert max(metadata["detected_face_counts"]) == 2
    assert metadata["selection_rule"] == "largest_detected_box_area"
