"""Strict speech pairing and stimulus-group-disjoint recording splits."""
from __future__ import annotations

import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import stable_seed, write_json


def parse_name(name):
    values = Path(name).stem.split("-")
    if len(values) != 7 or not all(v.isdigit() for v in values):
        raise ValueError(f"Invalid RAVDESS filename: {name}")
    modality, channel, emotion, intensity, statement, repetition, actor = map(int, values)
    if channel != 1 or not 1 <= actor <= 24 or modality not in (1, 2, 3):
        raise ValueError(f"Not an eligible speech recording: {name}")
    return dict(recording_id="-".join(values[1:]), actor=actor, emotion=emotion,
                intensity=intensity, statement=statement, repetition=repetition,
                stimulus_group="-".join(values[1:5]), modality=modality)


def make_splits(frame, seed):
    """Keep repetitions AND modalities of a stimulus together across all actors.

    Every actor: 20 enrollment, 20 development, 20 test recordings. Neutral
    stimulus groups are distributed over two partitions; other emotion groups
    use shuffled round-robin assignment to maintain broad emotion coverage.
    """
    groups = frame[["stimulus_group", "emotion"]].drop_duplicates().sort_values("stimulus_group")
    if len(groups) != 30:
        raise ValueError(f"Expected 30 speech stimulus groups, got {len(groups)}")
    rng = np.random.default_rng(seed)
    assignment = {}
    counts = [0, 0, 0]
    for emotion in sorted(groups.emotion.unique()):
        emotion_groups = groups.loc[groups.emotion == emotion, "stimulus_group"].to_numpy().copy()
        rng.shuffle(emotion_groups)
        for group in emotion_groups:
            candidates = np.flatnonzero(np.array(counts) == min(counts))
            part = int(rng.choice(candidates))
            assignment[group] = ("enroll", "dev", "test")[part]
            counts[part] += 1
    result = frame.copy()
    result["split"] = result.stimulus_group.map(assignment)
    assert counts == [10, 10, 10]
    assert result.groupby("stimulus_group").split.nunique().max() == 1
    assert result.groupby(["actor", "split"]).size().eq(20).all()
    return result


def _extract_member(z, member, target):
    """Write only a caller-chosen basename: no archive-controlled output paths."""
    info = z.getinfo(member)
    if target.exists() and target.stat().st_size == info.file_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with z.open(member) as source, temp.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    temp.replace(target)


def prepare(config):
    raw, out = Path(config["dataset_dir"]), Path(config["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    archive_audit = []
    rows = []
    with zipfile.ZipFile(raw / "Audio_Speech_Actors_01-24.zip") as za:
        audio_members = {}
        for name in za.namelist():
            if name.lower().endswith(".wav") and not name.startswith("__MACOSX"):
                meta = parse_name(name)
                if meta["modality"] == 3:
                    if meta["recording_id"] in audio_members:
                        raise ValueError("Duplicate audio recording")
                    audio_members[meta["recording_id"]] = name
        for actor in range(1, 25):
            archive = raw / f"Video_Speech_Actor_{actor:02}.zip"
            print(f"Preparing speech actor {actor:02}/24", flush=True)
            with zipfile.ZipFile(archive) as zv:
                entries = [n for n in zv.namelist() if n.endswith(".mp4") and Path(n).name.startswith("01-")]
                if len(entries) != 60:
                    raise ValueError(f"Expected 60 AV videos in {archive}, found {len(entries)}")
                archive_audit.append(dict(archive=archive.name, bytes=archive.stat().st_size,
                                          av_recordings=len(entries)))
                for member in sorted(entries):
                    meta = parse_name(member)
                    if meta["actor"] != actor:
                        raise ValueError("Actor/archive mismatch")
                    am = audio_members.pop(meta["recording_id"])
                    wav = out / "media" / f"Actor_{actor:02}" / Path(am).name
                    video = out / "media" / f"Actor_{actor:02}" / Path(member).name
                    _extract_member(za, am, wav)
                    _extract_member(zv, member, video)
                    meta.pop("modality")
                    rows.append(dict(**meta, audio_path=str(wav.relative_to(out)),
                                     video_path=str(video.relative_to(out)),
                                     audio_crc32=za.getinfo(am).CRC, video_crc32=zv.getinfo(member).CRC))
        if audio_members:
            raise ValueError("Unpaired audio recordings")
    frame = pd.DataFrame(rows).sort_values(["actor", "recording_id"]).reset_index(drop=True)
    if frame.recording_id.duplicated().any() or len(frame) != 1440:
        raise ValueError("Expected 1440 unique paired speech recordings")
    frame = make_splits(frame, config["split_seed"])
    frame.to_csv(out / "manifest.csv", index=False)
    write_json(out / "dataset_audit.json", dict(recordings=len(frame), actors=24,
        samples_per_actor=60, split_counts=Counter(frame.split), split_seed=config["split_seed"],
        pairing="Same actor and exact channel/emotion/intensity/statement/repetition identifier",
        excluded="Song and duplicate video-only encodings; no independent samples fabricated",
        split_unit="Stimulus group (both repetitions), shared across actors and modalities",
        archives=archive_audit))
    print(f"Prepared {len(frame)} paired speech recordings", flush=True)
    return frame


def enrollment_order(frame, seed):
    orders = {}
    for actor, group in frame.loc[frame.split == "enroll"].groupby("actor"):
        ids = sorted(group.recording_id)
        rng = np.random.default_rng(stable_seed(f"enroll:{seed}:{actor}"))
        orders[int(actor)] = list(np.array(ids)[rng.permutation(len(ids))])
    return orders
