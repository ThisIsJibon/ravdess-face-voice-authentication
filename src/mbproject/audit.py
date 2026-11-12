"""Machine-readable implementation and experimental-integrity checks."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256, write_json
from .experiment import condition_id
from .features import feature_signature
from .metrics import fit_policies, rates_at_threshold


def audit(config):
    data,out = Path(config["data_dir"]),Path(config["results_dir"])
    checks = {}
    frame = pd.read_csv(data/"manifest.csv",dtype={"recording_id":str})
    checks["1440_unique_paired_recordings"] = len(frame) == 1440 and not frame.recording_id.duplicated().any()
    checks["24_actors_60_each"] = frame.actor.nunique() == 24 and frame.groupby("actor").size().eq(60).all()
    checks["20_recordings_each_split_per_actor"] = frame.groupby(["actor","split"]).size().eq(20).all()
    checks["stimulus_group_disjoint"] = frame.groupby("stimulus_group").split.nunique().eq(1).all()
    checks["all_media_exist"] = all((data/p).is_file() for key in ("audio_path","video_path") for p in frame[key])
    crc_ok = True
    if checks["all_media_exist"]:
        for row in frame.itertuples():
            for path,expected in [(row.audio_path,row.audio_crc32),(row.video_path,row.video_crc32)]:
                actual = 0
                with (data/path).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024),b""):
                        actual = zlib.crc32(block,actual)
                crc_ok &= actual == expected
    checks["all_media_match_archive_crcs"] = checks["all_media_exist"] and crc_ok
    checks["complete_feature_cache"] = all((data/"features"/f"{r}.npz").exists() for r in frame.recording_id)
    signature = feature_signature(config)
    metas = [json.loads((data/"features"/f"{r}.json").read_text()) for r in frame.recording_id]
    checks["feature_cache_signatures_current"] = all(meta["signature"] == signature for meta in metas)
    checks["measured_snr_matches_design"] = all(abs(meta["measured_snr_db"][f"audio_snr{s:g}"]-s) < 1e-4
                                              for meta in metas for s in config["audio_snr_db"] if s is not None)
    sharpness = [np.median([meta["laplacian_variance"][f"face_sigma{s:g}"] for meta in metas]) for s in config["face_blur_sigma"]]
    checks["median_sharpness_decreases_with_blur"] = all(a>b for a,b in zip(sharpness[:-1],sharpness[1:]))
    embeddings_ok = True
    for recording in frame.recording_id:
        with np.load(data/"features"/f"{recording}.npz",allow_pickle=False) as z:
            for name in z.files:
                vector = z[name]
                embeddings_ok &= vector.shape == ((128,) if name.startswith("face") else (192,))
                embeddings_ok &= np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1) < 1e-5
    checks["all_embeddings_finite_unit_length_and_expected_dimension"] = embeddings_ok
    if (data/"cache_migrations.json").exists():
        migration = json.loads((data/"cache_migrations.json").read_text())
        checks["migrated_embeddings_unchanged"] = all(sha256(data/"features"/f"{r['recording_id']}.npz") == r["unchanged_embedding_sha256"] for r in migration["records"])
    model_manifest = json.loads((Path(config["model_dir"])/"manifest.json").read_text())
    checks["model_files_match_pinned_manifest"] = all(sha256(Path(config["model_dir"])/name) == meta["sha256"] for name,meta in model_manifest.items())
    provenance = json.loads((Path(__file__).resolve().parents[2]/"docs/hw3_provenance.json").read_text())
    checks["original_hw3_unmodified"] = sha256(Path(__file__).parent/"vendor/hw3_evaluator.py") == provenance["sha256"]
    checks["evaluation_completed"] = (out/"run_complete.json").exists()
    if checks["evaluation_completed"]:
        status = json.loads((out/"run_complete.json").read_text())
        metrics = pd.read_csv(out/"metrics.csv")
        checks["all_evaluation_rows_present"] = len(metrics) == status["metric_rows"]
        n = len(config["enrollment_sizes"])
        expected_conditions = 3*n-2 + len(config["degradation_enrollment_sizes"])*(len(config["audio_snr_db"])*len(config["face_blur_sigma"])-1)
        checks["full_predefined_condition_grid"] = len(metrics) == expected_conditions*len(config["enrollment_seeds"])*4
        checks["all_seeds_evaluated"] = set(metrics.seed) == set(config["enrollment_seeds"])
        checks["correct_genuine_impostor_counts"] = metrics.n_genuine.eq(480).all() and metrics.n_impostor.eq(11040).all()
        checks["finite_primary_metrics"] = np.isfinite(metrics[["eer","hw3_eer","dprime","auc"]].to_numpy()).all()
        checks["bounded_error_rates"] = metrics.eer.between(0,1).all() and metrics.auc.between(0,1).all()
        checks["policies_frozen_across_conditions"] = all(
            group[[c for c in metrics.columns if c.startswith("threshold_dev_")]].nunique().eq(1).all()
            for _,group in metrics.groupby(["seed","system"]))
        policies = json.loads((out/"policies.json").read_text())
        with np.load(out/"development_trial_index.npz",allow_pickle=False) as z:
            dev_labels = z["genuine"].ravel()
        policies_ok = True
        for seed in config["enrollment_seeds"]:
            with np.load(out/"development_scores"/f"{seed}.npz",allow_pickle=False) as z:
                for system in z.files:
                    fitted = fit_policies(z[system].ravel(),dev_labels,config["far_targets"])
                    policies_ok &= fitted == policies[str(seed)]["policies"][system]
        checks["thresholds_reproduce_from_development_only"] = policies_ok
        with np.load(out/"trial_index.npz",allow_pickle=False) as z:
            genuine,probe_ids = z["genuine"],set(z["probe_recording_ids"])
        orders_ok = True
        for seed in config["enrollment_seeds"]:
            orders = json.loads((out/"enrollment"/f"seed_{seed}.json").read_text())
            for actor,ids in orders.items():
                valid = set(frame.loc[(frame.actor == int(actor))&(frame.split == "enroll"),"recording_id"])
                orders_ok &= set(ids) == valid and not (set(ids)&probe_ids) and len(ids) == len(set(ids))
        checks["nested_enrollment_ids_valid_and_disjoint"] = orders_ok
        first_seed = config["enrollment_seeds"][0]
        k = max(config["enrollment_sizes"])
        condition = condition_id(k,k,None,0)
        with np.load(out/"trial_scores"/str(first_seed)/f"{condition}.npz",allow_pickle=False) as z:
            checks["fixed_fusion_matches_saved_scores"] = np.allclose(z["fusion_fixed"],.5*z["face"]+.5*z["voice"],atol=1e-12)
            row = metrics.loc[(metrics.seed == first_seed)&(metrics.condition == condition)&(metrics.system == "fusion_fixed")].iloc[0]
            rates = rates_at_threshold(z["fusion_fixed"].ravel(),genuine.ravel(),row["threshold_dev_far_0.01"])
            checks["operating_metrics_recompute_from_trials"] = abs(rates["frr"]-row["frr_at_dev_far_0.01"]) < 1e-12 and abs(rates["far"]-row["far_at_dev_far_0.01"]) < 1e-12
    checks = {key:bool(value) for key,value in checks.items()}
    result = dict(passed=all(checks.values()),checks=checks,
                  scope="Implementation checks only; report, literature review, ethics prose, and presentation remain outside this task")
    write_json(out/"audit.json",result)
    return result
