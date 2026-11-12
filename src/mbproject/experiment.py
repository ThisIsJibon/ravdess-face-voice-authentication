from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import audio_key, face_key, sha256, unit, write_json
from .dataset import enrollment_order
from .features import feature_signature
from .metrics import eer, evaluate, fit_policies, rates_at_threshold, threshold_at_far

SYSTEMS = ("voice", "face", "fusion_fixed", "fusion_tuned")


def cosine_scores(probes, templates):
    # Theoretical affine scaling is fixed in advance and keeps HW3's [0,1] scale.
    return np.clip((unit(probes) @ unit(templates).T + 1)/2,0,1).astype(np.float64)


def fuse(face, voice, face_weight):
    if not 0 <= face_weight <= 1:
        raise ValueError("Fusion weight must lie in [0,1]")
    return face_weight*face + (1-face_weight)*voice


def condition_id(kf,kv,snr,sigma):
    return f"kf{kf}_kv{kv}_{audio_key(snr)}_{face_key(sigma)}"


def load_features(config, frame):
    signature = feature_signature(config)
    features = {}
    for row in frame.itertuples():
        path = Path(config["data_dir"])/"features"/f"{row.recording_id}.npz"
        if not path.exists():
            raise ValueError(f"Missing {path}; run extract first")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["signature"] != signature:
            raise ValueError(f"Stale features for {row.recording_id}")
        with np.load(path,allow_pickle=False) as z:
            features[row.recording_id] = {key:z[key] for key in z.files}
    return features


def templates(features, orders, key, k):
    return np.stack([unit(np.mean([features[r][key] for r in orders[a][:k]],axis=0)) for a in sorted(orders)])


def tune_weight(face,voice,genuine,candidates):
    choices = []
    for weight in candidates:
        scores = fuse(face,voice,weight)
        threshold = threshold_at_far(scores.ravel(),genuine.ravel(),.01)
        frr = rates_at_threshold(scores.ravel(),genuine.ravel(),threshold)["frr"]
        choices.append((eer(scores,genuine),frr,abs(weight-.5),weight))
    return min(choices)[-1]


def run(config):
    started = time.perf_counter()
    out, data = Path(config["results_dir"]), Path(config["data_dir"])
    out.mkdir(parents=True,exist_ok=True)
    (out/"run_complete.json").unlink(missing_ok=True)
    write_json(out/"config_resolved.json",config)
    frame = pd.read_csv(data/"manifest.csv", dtype={"recording_id":str})
    features = load_features(config,frame)
    actors = np.array(sorted(frame.actor.unique()))
    reference = max(config["enrollment_sizes"])
    if reference > frame.loc[frame.split == "enroll"].groupby("actor").size().min():
        raise ValueError("Enrollment size exceeds available clean recordings")
    dev = frame.loc[frame.split == "dev"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    labels_dev = dev.actor.to_numpy()[:,None] == actors[None,:]
    labels_test = test.actor.to_numpy()[:,None] == actors[None,:]
    sizes = config["enrollment_sizes"]
    clean_pairs = sorted(set([(k,k) for k in sizes]+[(k,reference) for k in sizes]+[(reference,k) for k in sizes]))
    conditions = [(kf,kv,None,0,"clean_enrollment") for kf,kv in clean_pairs]
    for k,snr,sigma in itertools.product(config["degradation_enrollment_sizes"],config["audio_snr_db"],config["face_blur_sigma"]):
        if snr is not None or sigma != 0:
            conditions.append((k,k,snr,sigma,"degradation"))
    records, per_user, policies_all, selections = [], [], {}, []
    np.savez_compressed(out/"trial_index.npz",probe_recording_ids=test.recording_id.to_numpy(dtype=str),
        probe_actors=test.actor.to_numpy(),claim_actors=actors,genuine=labels_test)
    np.savez_compressed(out/"development_trial_index.npz",probe_recording_ids=dev.recording_id.to_numpy(dtype=str),
        probe_actors=dev.actor.to_numpy(),claim_actors=actors,genuine=labels_dev)
    for seed in config["enrollment_seeds"]:
        print(f"Evaluating enrollment seed {seed} ({len(conditions)} conditions)",flush=True)
        orders = enrollment_order(frame,seed)
        write_json(out/"enrollment"/f"seed_{seed}.json",orders)
        cache = {}
        def score(part,key,k):
            token = (part,key,k)
            if token not in cache:
                subset = dev if part == "dev" else test
                probes = np.stack([features[r][key] for r in subset.recording_id])
                template_key = "face_sigma0" if key.startswith("face") else "audio_clean"
                cache[token] = cosine_scores(probes,templates(features,orders,template_key,k))
            return cache[token]
        df, dv = score("dev","face_sigma0",reference), score("dev","audio_clean",reference)
        weight = tune_weight(df,dv,labels_dev,config["face_weight_candidates"])
        dev_scores = dict(voice=dv,face=df,fusion_fixed=fuse(df,dv,.5),fusion_tuned=fuse(df,dv,weight))
        (out/"development_scores").mkdir(exist_ok=True)
        np.savez_compressed(out/"development_scores"/f"{seed}.npz",**dev_scores)
        policies = {s:fit_policies(v.ravel(),labels_dev.ravel(),config["far_targets"]) for s,v in dev_scores.items()}
        # Strongest fusion is selected ONLY from clean development data.
        chosen = min(("fusion_fixed","fusion_tuned"),key=lambda s:(eer(dev_scores[s],labels_dev),
                     rates_at_threshold(dev_scores[s].ravel(),labels_dev.ravel(),policies[s]["far_0.01"])["frr"],
                     s != "fusion_fixed"))
        selections.append(dict(seed=seed,face_weight=weight,selected_fusion=chosen,
                               dev_fixed_eer=eer(dev_scores["fusion_fixed"],labels_dev),
                               dev_tuned_eer=eer(dev_scores["fusion_tuned"],labels_dev)))
        policies_all[str(seed)] = dict(face_weight=weight,policies=policies,selected_fusion=chosen,
                                       reference_enrollment=reference)
        for kf,kv,snr,sigma,family in conditions:
            name = condition_id(kf,kv,snr,sigma)
            f,v = score("test",face_key(sigma),kf),score("test",audio_key(snr),kv)
            values = dict(voice=v,face=f,fusion_fixed=fuse(f,v,.5),fusion_tuned=fuse(f,v,weight))
            score_dir = out/"trial_scores"/str(seed)
            score_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(score_dir/f"{name}.npz",**values)
            base = dict(seed=seed,condition=name,family=family,k_face=kf,k_voice=kv,
                        audio_snr_db="clean" if snr is None else str(snr),face_sigma=sigma,
                        selected_fusion=chosen,face_weight=weight)
            for system,matrix in values.items():
                records.append(dict(**base,system=system,**evaluate(matrix.ravel(),labels_test.ravel(),
                                            policies[system],config["far_targets"])))
                # Probe-user clusters retain the common fixed claim roster.
                threshold = policies[system]["far_0.01"]
                for actor in actors:
                    mask = test.actor.to_numpy() == actor
                    rates = rates_at_threshold(matrix[mask].ravel(),labels_test[mask].ravel(),threshold)
                    per_user.append(dict(**base,system=system,actor=int(actor),**rates))
        # Checkpoint compact tables each seed; partial results are explicitly incomplete.
        pd.DataFrame(records).to_csv(out/"metrics.csv",index=False)
        pd.DataFrame(per_user).to_csv(out/"per_user.csv",index=False)
        write_json(out/"policies.json",policies_all)
    result = pd.DataFrame(records)
    pd.DataFrame(selections).to_csv(out/"fusion_selection.csv",index=False)
    group = ["condition","family","k_face","k_voice","audio_snr_db","face_sigma","system"]
    quantities = ["eer","hw3_eer","dprime","auc","far_at_dev_far_0.01","frr_at_dev_far_0.01",
                  "frr_at_dev_far_0.001","frr_at_dev_far_0.0001"]
    summary = result.groupby(group,dropna=False)[quantities].agg(["mean","std","min","max"]).reset_index()
    summary.columns = ["_".join(filter(None,c)) if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/"summary.csv",index=False)
    paired_bootstrap(pd.DataFrame(per_user),config,out)
    # Export portable enrollment templates for a functioning claim-verification demo.
    first_seed = config["enrollment_seeds"][0]
    orders = enrollment_order(frame,first_seed)
    np.savez_compressed(out/"authenticator_templates.npz",actors=actors,
        face=templates(features,orders,"face_sigma0",reference),
        voice=templates(features,orders,"audio_clean",reference))
    write_json(out/"run_complete.json",dict(status="complete",conditions_per_seed=len(conditions),
        enrollment_seeds=len(config["enrollment_seeds"]),systems=len(SYSTEMS),metric_rows=len(records),
        genuine_trials_per_condition=int(labels_test.sum()),impostor_trials_per_condition=int((~labels_test).sum()),
        seconds=time.perf_counter()-started,manifest_sha256=sha256(data/"manifest.csv"),
        model_manifest_sha256=sha256(Path(config["model_dir"])/"manifest.json"),
        feature_signature=feature_signature(config),
        implementation_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                               ["experiment.py","metrics.py","dataset.py"]},
        uncertainty="Seed spread measures enrollment resampling only. Bootstrap intervals resample probe identities conditional on the fixed enrolled claim roster."))
    print(f"Completed {len(records)} system/condition/seed evaluations",flush=True)


def paired_bootstrap(users,config,out):
    rng = np.random.default_rng(config["split_seed"])
    rows = []
    grouped = users.groupby(["condition","system","actor"],sort=True).frr.mean()
    for condition in sorted(users.condition.unique()):
        table = grouped.loc[condition].unstack("system")
        for a,b in [("fusion_fixed","face"),("fusion_fixed","voice"),("fusion_tuned","fusion_fixed")]:
            delta = (table[a]-table[b]).to_numpy()
            boot = delta[rng.integers(0,len(delta),size=(config["bootstrap_replicates"],len(delta)))].mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            rows.append(dict(condition=condition,system_a=a,system_b=b,
                mean_frr_difference=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi),
                target="FRR at clean-development FAR=0.01 threshold; average over enrollment seeds",
                resampling_unit="probe actor; fixed enrolled claim roster",replicates=len(boot)))
    pd.DataFrame(rows).to_csv(out/"paired_actor_bootstrap.csv",index=False)
