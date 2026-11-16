"""Step 2 of the tracking study: test the tracking features as a biometric.

To make the comparison with face and voice fair, we reuse everything from the
main experiment instead of making our own versions:
  - the same enroll / dev / test split   (data/manifest.csv)
  - the same enrollment recordings       (enrollment_order, same 10 seeds)
  - the same test trials                 (results/full/trial_index.npz)
  - the same template + cosine score rule, the same HW3 evaluator,
    and thresholds fixed on dev data at k = 16
  - the saved face and voice scores      (results/full/trial_scores) for fusion

Rule for learning: anything that is "learned" (z-score statistics, LDA) uses the
ENROLL split only, settings are picked on the DEV split only, and the TEST split
is only used to report the final numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .common import ROOT, audio_key, unit, write_json
from .dataset import enrollment_order
from .metrics import eer, evaluate, fit_policies
from .tracking import extract_tracking, load_tracking

# Columns we average over the 10 seeds in the summary tables.
METRICS = ["eer", "hw3_eer", "dprime", "auc", "far_at_dev_far_0.01", "frr_at_dev_far_0.01",
           "frr_at_dev_far_0.001", "frr_at_dev_far_0.0001"]


def output_dir(config):
    path = Path(config["tracking"]["output_dir"])
    return path if path.is_absolute() else ROOT / path


# ----------------------------------------------------------------------------
# Scoring (same rules as experiment.py)
# ----------------------------------------------------------------------------
def cosine_scores(probes, templates):
    """Cosine similarity mapped from [-1, 1] to [0, 1]. Same formula as experiment.py."""
    return np.clip((unit(probes) @ unit(templates).T + 1) / 2, 0, 1).astype(np.float64)


def make_templates(embeddings, row_of, orders, k):
    """One template per actor = normalized average of their first k enrollment recordings."""
    return np.stack([unit(embeddings[[row_of[r] for r in orders[actor][:k]]].mean(axis=0))
                     for actor in sorted(orders)])


def score_matrix(embeddings, row_of, probe_ids, orders, k):
    """Score every probe recording against every actor's template -> (probes, 24)."""
    probes = embeddings[[row_of[r] for r in probe_ids]]
    return cosine_scores(probes, make_templates(embeddings, row_of, orders, k))


# ----------------------------------------------------------------------------
# Features -> embeddings
# ----------------------------------------------------------------------------
class Embedding:
    """z-score the features, then optionally apply LDA.

    z-score: subtract the mean and divide by the standard deviation, so every
    feature has the same scale. LDA (Linear Discriminant Analysis) then finds the
    directions where different people are far apart and the same person's
    recordings are close together. This helps ignore what the actor is saying
    or which emotion they show.
    """

    def __init__(self, lda_dims=0, shrinkage=0.0):
        self.lda_dims = lda_dims
        self.shrinkage = shrinkage

    def fit(self, X, actors):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0               # avoid dividing by zero
        self.lda = None
        if self.lda_dims:
            # LDA can give at most (number of people - 1) dimensions.
            dims = min(self.lda_dims, len(np.unique(actors)) - 1, X.shape[1])
            self.lda = LinearDiscriminantAnalysis(solver="eigen", n_components=dims,
                                                  shrinkage=self.shrinkage or None)
            self.lda.fit(self.zscore(X), actors)
        return self

    def zscore(self, X):
        return (X - self.mean) / self.std

    def transform(self, X):
        Z = self.zscore(X)
        return self.lda.transform(Z) if self.lda is not None else Z

    def name(self):
        return "raw" if not self.lda_dims else f"lda_d{self.lda_dims}_s{self.shrinkage:g}"


def check_trial_index(path, probes, actors, genuine):
    """Stop if a saved trial index does not match our probes, claims and labels.

    Fusion averages our scores with the saved face/voice scores cell by cell, so
    row i must be the same recording and column j the same claimed actor in both.
    """
    with np.load(path, allow_pickle=False) as saved:
        if list(saved["probe_recording_ids"]) != list(probes.recording_id):
            raise ValueError(f"{path.name}: probe recordings differ from the main study")
        if not np.array_equal(saved["probe_actors"], probes.actor.to_numpy()):
            raise ValueError(f"{path.name}: probe actors differ from the main study")
        if not np.array_equal(saved["claim_actors"], actors):
            raise ValueError(f"{path.name}: claimed actors differ from the main study")
        if not np.array_equal(np.ravel(saved["genuine"]), genuine.ravel()):
            raise ValueError(f"{path.name}: genuine/impostor labels differ from the main study")


def average_over_seeds(rows, group_by):
    """Mean / std / min / max of each metric over the 10 seeds."""
    table = pd.DataFrame(rows).groupby(group_by)[METRICS].agg(["mean", "std", "min", "max"]).reset_index()
    table.columns = ["_".join(c for c in col if c) for col in table.columns]
    return table


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------
def run_tracking(config):
    start = time.time()
    settings = config["tracking"]
    data_dir, main_results = Path(config["data_dir"]), Path(config["results_dir"])
    out = output_dir(config)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load the split and the tracking features -------------------------------
    manifest = pd.read_csv(data_dir / "manifest.csv", dtype={"recording_id": str})
    if not (data_dir / "tracking" / "features.npz").exists():
        extract_tracking(config)
    blocks = load_tracking(config)
    assert list(blocks["recording_ids"]) == list(manifest.recording_id), "features not in manifest order"
    row_of = {rid: i for i, rid in enumerate(manifest.recording_id)}   # recording_id -> row number

    enroll = manifest[manifest.split == "enroll"]
    dev = manifest[manifest.split == "dev"].reset_index(drop=True)
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    actors = np.array(sorted(manifest.actor.unique()))
    # genuine[i, j] is True when probe i really belongs to claimed actor j.
    genuine_dev = dev.actor.to_numpy()[:, None] == actors[None, :]
    genuine_test = test.actor.to_numpy()[:, None] == actors[None, :]

    # --- Check we use exactly the same trials and enrollments as face/voice --------
    check_trial_index(main_results / "trial_index.npz", test, actors, genuine_test)
    check_trial_index(main_results / "development_trial_index.npz", dev, actors, genuine_dev)
    seeds, sizes = config["enrollment_seeds"], config["enrollment_sizes"]
    orders = {seed: enrollment_order(manifest, seed) for seed in seeds}
    for seed in seeds:
        saved = main_results / "enrollment" / f"seed_{seed}.json"
        if saved.exists():
            saved_order = {int(a): ids for a, ids in json.loads(saved.read_text()).items()}
            assert saved_order == orders[seed], f"enrollment order for seed {seed} differs"
    reference_k = max(sizes)                        # thresholds are fixed at k = 16, like the main study
    pick_k = settings["selection_enrollment_size"]  # LDA settings are picked on dev at k = 4

    # --- Experiment A: which feature sets identify a person? -----------------------
    # A feature set is a list of blocks glued together, e.g. au_all = level + motion + velocity.
    feature_sets = {name: np.hstack([blocks[b] for b in parts])
                    for name, parts in settings["feature_sets"].items()}
    enroll_rows = [row_of[r] for r in enroll.recording_id]
    enroll_actors = enroll.actor.to_numpy()

    def dev_eer(embeddings):
        """Average dev EER over the 10 seeds at k = pick_k (used to choose LDA settings)."""
        return np.mean([eer(score_matrix(embeddings, row_of, dev.recording_id, orders[s], pick_k), genuine_dev)
                        for s in seeds])

    embeddings, chosen, grid = {}, {}, []
    for name, X in feature_sets.items():
        # Raw version: z-score only.
        raw = Embedding().fit(X[enroll_rows], enroll_actors)
        embeddings[(name, "raw")] = raw.transform(X)
        raw_dev = dev_eer(embeddings[(name, "raw")])
        grid.append({"feature_set": name, "representation": "raw", "dev_eer": raw_dev})

        # LDA version: try each setting, keep the one with the lowest DEV error.
        best = None
        for dims in settings["lda_dimensions"]:
            for shrink in settings["lda_shrinkage"]:
                model = Embedding(dims, shrink).fit(X[enroll_rows], enroll_actors)
                E = model.transform(X)
                score = dev_eer(E)
                grid.append({"feature_set": name, "representation": model.name(), "dev_eer": score})
                if best is None or score < best[0]:
                    best = (score, model, E)
        embeddings[(name, "lda")] = best[2]
        chosen[name] = {"feature_set": name, "size": X.shape[1], "lda": best[1].name(),
                        "lda_dims": best[1].lda_dims, "shrinkage": best[1].shrinkage, "dev_eer": best[0]}
        print(f"{name:16s} {X.shape[1]:4d} features | dev EER raw {raw_dev:.3f}, "
              f"LDA {best[0]:.3f} ({best[1].name()})", flush=True)
    pd.DataFrame(grid).to_csv(out / "dev_selection_grid.csv", index=False)
    pd.DataFrame(chosen.values()).to_csv(out / "dev_selected.csv", index=False)

    # --- Experiments A + C on TEST: every feature set, every enrollment size -------
    rows = []
    for (name, rep), E in embeddings.items():
        for seed in seeds:
            # Thresholds come from DEV scores at k = 16, then stay fixed (same as main study).
            dev_scores = score_matrix(E, row_of, dev.recording_id, orders[seed], reference_k)
            thresholds = fit_policies(dev_scores.ravel(), genuine_dev.ravel(), config["far_targets"])
            for k in sizes:
                test_scores = score_matrix(E, row_of, test.recording_id, orders[seed], k)
                result = evaluate(test_scores.ravel(), genuine_test.ravel(), thresholds, config["far_targets"])
                rows.append({"feature_set": name, "representation": rep, "seed": seed, "k": k, **result})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    average_over_seeds(rows, ["feature_set", "representation", "k"]).to_csv(out / "summary.csv", index=False)
    print(f"Experiments A and C done: {len(rows)} evaluations", flush=True)

    # Two tracking systems go into fusion: AU only, and the best set chosen on DEV.
    best_set = min(chosen, key=lambda n: chosen[n]["dev_eer"])
    tracking_systems = {"au": embeddings[("au_all", "lda")], "tracking": embeddings[(best_set, "lda")]}

    # --- Experiment B: fusion with the saved face and voice scores ----------------
    # Fusion rule = the team's fixed rule: plain average of the scores.
    combos = [("face", "voice"), ("au", "voice"), ("au", "face"), ("au", "face", "voice"),
              ("tracking", "voice"), ("tracking", "face", "voice")]
    noise_levels = [None] + [snr for snr in config["audio_snr_db"] if snr is not None]
    conditions = [(k, None) for k in sizes] + \
                 [(k, snr) for k in config["degradation_enrollment_sizes"] for snr in noise_levels if snr is not None]
    fusion_rows = []
    for seed in seeds:
        saved_dev = np.load(main_results / "development_scores" / f"{seed}.npz")
        dev_scores = {"face": saved_dev["face"], "voice": saved_dev["voice"]}
        for system in ("face", "voice"):
            if dev_scores[system].shape != genuine_dev.shape:
                raise ValueError(f"saved dev {system} scores for seed {seed} have the wrong shape")
        for system, E in tracking_systems.items():
            dev_scores[system] = score_matrix(E, row_of, dev.recording_id, orders[seed], reference_k)
        for combo in combos:
            dev_scores["+".join(combo)] = np.mean([dev_scores[s] for s in combo], axis=0)
        thresholds = {name: fit_policies(s.ravel(), genuine_dev.ravel(), config["far_targets"])
                      for name, s in dev_scores.items()}

        for k, snr in conditions:
            # Face and voice test scores saved by the main study (noise is on the probe audio only).
            saved = np.load(main_results / "trial_scores" / str(seed) / f"kf{k}_kv{k}_{audio_key(snr)}_face_sigma0.npz")
            scores = {"face": saved["face"], "voice": saved["voice"]}
            if scores["face"].shape != genuine_test.shape or scores["voice"].shape != genuine_test.shape:
                raise ValueError(f"saved test scores for seed {seed}, k = {k} have the wrong shape")
            for system, E in tracking_systems.items():
                scores[system] = score_matrix(E, row_of, test.recording_id, orders[seed], k)
            for combo in combos:
                scores["+".join(combo)] = np.mean([scores[s] for s in combo], axis=0)
            for name, s in scores.items():
                result = evaluate(s.ravel(), genuine_test.ravel(), thresholds[name], config["far_targets"])
                fusion_rows.append({"systems": name, "audio_snr_db": "clean" if snr is None else str(snr),
                                    "seed": seed, "k": k, **result})
    pd.DataFrame(fusion_rows).to_csv(out / "fusion_metrics.csv", index=False)
    average_over_seeds(fusion_rows, ["systems", "audio_snr_db", "k"]).to_csv(out / "fusion_summary.csv", index=False)
    print(f"Experiment B done: {len(fusion_rows)} evaluations", flush=True)

    # --- Sanity checks (controls) -------------------------------------------------
    controls = []

    # 1) Shuffle who each test probe belongs to. A correct pipeline must drop to ~50%.
    rng = np.random.default_rng(config["split_seed"])
    for system, E in tracking_systems.items():
        s = score_matrix(E, row_of, test.recording_id, orders[seeds[0]], pick_k)
        shuffled_actor = rng.permutation(test.actor.to_numpy())
        controls.append({"control": "shuffled_labels", "system": system,
                         "eer": eer(s, shuffled_actor[:, None] == actors[None, :])})

    # 2) Actors excluded from LDA fitting: learn z-score + LDA on 12 actors, test on
    #    the other 12. Shows how much of the LDA gain is learning these particular actors.
    #    Note: the LDA settings (pick) were chosen earlier on dev data from all 24 actors.
    for system, feature_set in [("au", "au_all"), ("tracking", best_set)]:
        X, pick = feature_sets[feature_set], chosen[feature_set]
        for split in range(settings["unseen_actor_splits"]):
            train_actors = np.random.default_rng(1000 + split).choice(actors, size=12, replace=False)
            test_actors = np.setdiff1d(actors, train_actors)
            train = enroll[enroll.actor.isin(train_actors)]
            probes = test[test.actor.isin(test_actors)]
            genuine = probes.actor.to_numpy()[:, None] == test_actors[None, :]
            held_out_orders = {a: orders[seeds[0]][a] for a in test_actors}
            for rep, model in [("raw", Embedding()), ("lda", Embedding(pick["lda_dims"], pick["shrinkage"]))]:
                model.fit(X[[row_of[r] for r in train.recording_id]], train.actor.to_numpy())
                s = score_matrix(model.transform(X), row_of, probes.recording_id, held_out_orders, pick_k)
                controls.append({"control": f"unseen_actors_{rep}", "system": system, "split": split,
                                 "eer": eer(s, genuine)})

    # 3) Same sentence+emotion vs different: compare pairs of test recordings.
    same_actor = test.actor.to_numpy()[:, None] == test.actor.to_numpy()[None, :]
    same_content = test.stimulus_group.to_numpy()[:, None] == test.stimulus_group.to_numpy()[None, :]
    each_pair_once = np.triu(np.ones_like(same_actor), k=1).astype(bool)
    for system, E in tracking_systems.items():
        T = E[[row_of[r] for r in test.recording_id]]
        similarity = cosine_scores(T, T)
        for label, mask in [("same_content_pairs", same_content), ("different_content_pairs", ~same_content)]:
            pairs = each_pair_once & mask
            controls.append({"control": label, "system": system, "eer": eer(similarity[pairs], same_actor[pairs])})
    pd.DataFrame(controls).to_csv(out / "controls.csv", index=False)

    write_json(out / "run_complete.json", {
        "status": "complete", "seconds": round(time.time() - start, 1),
        "best_feature_set": best_set, "best_lda": chosen[best_set]["lda"],
        "chosen_on": f"development EER at k = {pick_k}",
        "genuine_trials": int(genuine_test.sum()), "impostor_trials": int((~genuine_test).sum())})
    print(f"Best feature set on dev data: {best_set} ({chosen[best_set]['lda']})", flush=True)

    from .tracking_report import write_report
    write_report(config)
    print(f"Done. Results are in {out}", flush=True)
