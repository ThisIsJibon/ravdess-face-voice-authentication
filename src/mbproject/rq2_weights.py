"""RQ2 extension: face/voice fusion weights at small clean enrollment sizes.

RQ1 showed that errors are highest when people enroll only a few recordings
(k = 1, 2, 4). This study asks whether combining face and voice with the right
weight helps at those sizes.

For each enrollment seed and each small k:
  1. Build face and voice templates from the same enrollment recordings as RQ1.
  2. Score every development and test probe against all 24 enrolled actors.
  3. Min-max normalize each modality using development scores only, so a 50%
     face weight really gives both modalities an equal say.
  4. Fuse with face weights 0.0, 0.1, ..., 1.0 (0 = voice only, 1 = face only)
     and measure every weight on the test probes.
  5. Pick one "tuned" weight per seed and k from development data only.

Thresholds and weights are fitted separately for every k here, because the
question is the best fusion achievable at that enrollment size. (The main
pipeline instead freezes them at k = 16 to isolate the enrollment effect.)
The test set never chooses anything. Clean data only; no degradations.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import write_json
from .dataset import enrollment_order
from .experiment import cosine_scores, fuse, load_features, templates
from .features import feature_signature
from .metrics import FastEvaluator, eer, rates_at_threshold, threshold_at_far

DEFAULTS = {"enrollment_sizes": [1, 2, 4], "face_weights": [round(0.1 * i, 1) for i in range(11)], "far_target": 0.001}
# Systems in the summary table, in display order.
SYSTEM_LABELS = {
    "voice": "Voice only",
    "face": "Face only",
    "fusion_raw_equal": "Fusion 50/50, not normalized (same rule as RQ1)",
    "fusion_norm_equal": "Fusion 50/50, normalized",
    "fusion_norm_tuned": "Fusion, normalized, dev-tuned weight",
}


def settings(config):
    """RQ2 settings from the config's optional `rq2` block, checked for sense."""
    given = config.get("rq2") or {}
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown rq2 settings {sorted(unknown)}; expected {sorted(DEFAULTS)}")
    s = {**DEFAULTS, **given}
    sizes = s["enrollment_sizes"]
    if (not sizes or not all(isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("rq2.enrollment_sizes must be a non-empty list of unique positive whole numbers")
    weights = [float(w) for w in s["face_weights"]]
    if weights != sorted(set(weights)) or not all(0 <= w <= 1 for w in weights):
        raise ValueError("rq2.face_weights must be unique, increasing and between 0 and 1")
    if not any(0 < w < 1 for w in weights):
        raise ValueError("rq2.face_weights needs at least one weight strictly between 0 and 1 to tune")
    if not 0 < float(s["far_target"]) < 1:
        raise ValueError("rq2.far_target must be between 0 and 1")
    return {"enrollment_sizes": list(sizes), "face_weights": weights,
                "far_target": float(s["far_target"])}


# ----------------------------------------------------------------------------
# Normalization, measurement and weight selection
# ----------------------------------------------------------------------------
def fit_minmax(dev_scores):
    """Lowest and highest development score of one modality."""
    low, high = float(np.min(dev_scores)), float(np.max(dev_scores))
    if not high > low:
        raise ValueError("Cannot min-max normalize development scores that are all equal")
    return low, high


def apply_minmax(scores, bounds):
    """Map the development range to [0, 1]; test scores outside it are clipped to 0 or 1."""
    low, high = bounds
    return np.clip((np.asarray(scores, dtype=np.float64) - low) / (high - low), 0, 1)


def dprime(scores, genuine):
    """HW3 d-prime: how far apart genuine and impostor scores are. Still varies when EER is 0."""
    scores, genuine = np.ravel(scores), np.ravel(genuine).astype(bool)
    return float(FastEvaluator(500, scores[genuine], scores[~genuine], "rq2").get_dprime())


def measure(test_scores, test_genuine, threshold, far_target):
    """Test metrics for one system: HW3 EER, interpolated EER, d-prime, and FRR/FAR at a dev threshold."""
    scores, genuine = np.ravel(test_scores), np.ravel(test_genuine).astype(bool)
    hw3 = FastEvaluator(500, scores[genuine], scores[~genuine], "rq2")
    fpr, fnr, _ = hw3.compute_rates()
    hw3_eer, _ = hw3.get_EER(fpr, fnr)
    rates = rates_at_threshold(scores, genuine, threshold)
    return {"hw3_eer": float(hw3_eer), "eer": eer(scores, genuine), "dprime": float(hw3.get_dprime()),
            f"frr_at_dev_far_{far_target:g}": rates["frr"], f"far_at_dev_far_{far_target:g}": rates["far"]}


def select_weight(dev_face, dev_voice, dev_genuine, candidates, far_target):
    """Choose a face weight from development data only.

    Order of preference: lowest exact (interpolated ROC) EER, then lowest FRR
    at the FAR target, then highest d-prime, then closest to 0.5. The d-prime
    step matters when several weights tie at zero errors, as happened in the
    original RQ2 tuning where face alone was perfect and the rule fell back to
    0.5. Values are rounded so floating-point noise cannot break a real tie.
    """
    genuine = np.ravel(dev_genuine).astype(bool)
    choices = []
    for weight in candidates:
        scores = np.ravel(fuse(dev_face, dev_voice, weight))
        threshold = threshold_at_far(scores, genuine, far_target)
        frr = rates_at_threshold(scores, genuine, threshold)["frr"]
        key = (eer(scores, genuine), frr, -dprime(scores, genuine), abs(weight - 0.5))
        choices.append((*(round(value, 9) for value in key), weight))
    return min(choices)[-1]


def evaluate_weight(face, voice, weight, labels, far_target):
    """Fuse with one face weight, fit the threshold on dev scores, measure on test scores.

    `face` and `voice` hold "dev" and "test" score matrices; `labels` is (dev_genuine, test_genuine).
    """
    dev_genuine, test_genuine = labels
    dev_scores = np.ravel(fuse(face["dev"], voice["dev"], weight))
    threshold = threshold_at_far(dev_scores, np.ravel(dev_genuine), far_target)
    return measure(fuse(face["test"], voice["test"], weight), test_genuine, threshold, far_target)


# ----------------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------------
def output_dir(config):
    return Path(config["results_dir"]) / "rq2_weights"


def run_rq2(config):
    start = time.perf_counter()
    s = settings(config)
    out = output_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rq2_complete.json").unlink(missing_ok=True)
    frame = pd.read_csv(Path(config["data_dir"]) / "manifest.csv", dtype={"recording_id": str})
    features = load_features(config, frame)  # fails on missing or stale embeddings
    pool = frame.loc[frame.split == "enroll"].groupby("actor").size().min()
    if max(s["enrollment_sizes"]) > pool:
        raise ValueError(f"rq2.enrollment_sizes go above the {pool} enrollment recordings per actor")
    actors = np.array(sorted(frame.actor.unique()))
    dev = frame.loc[frame.split == "dev"].reset_index(drop=True)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    dev_genuine = dev.actor.to_numpy()[:, None] == actors[None, :]
    test_genuine = test.actor.to_numpy()[:, None] == actors[None, :]
    probes = {(part, key): np.stack([features[r][key] for r in subset.recording_id])
              for part, subset in [("dev", dev), ("test", test)] for key in ("face_sigma0", "audio_clean")}
    target, candidates = s["far_target"], [w for w in s["face_weights"] if 0 < w < 1]
    sweep, systems, selections = [], [], []
    for seed in config["enrollment_seeds"]:
        print(f"RQ2 weight sweep: enrollment seed {seed}", flush=True)
        orders = enrollment_order(frame, seed)  # same recordings, same order as RQ1
        for k in s["enrollment_sizes"]:
            raw = {}
            for modality, key in [("face", "face_sigma0"), ("voice", "audio_clean")]:
                template = templates(features, orders, key, k)
                raw[modality] = {part: cosine_scores(probes[(part, key)], template) for part in ("dev", "test")}
            bounds = {m: fit_minmax(raw[m]["dev"]) for m in raw}
            norm = {m: {p: apply_minmax(raw[m][p], bounds[m]) for p in ("dev", "test")} for m in raw}
            weight = select_weight(norm["face"]["dev"], norm["voice"]["dev"], dev_genuine, candidates, target)
            selections.append({"seed": seed, "k": k, "selected_face_weight": weight,
                                   "face_dev_min": bounds["face"][0], "face_dev_max": bounds["face"][1],
                                   "voice_dev_min": bounds["voice"][0], "voice_dev_max": bounds["voice"][1]})
            labels = (dev_genuine, test_genuine)
            for w in s["face_weights"]:
                sweep.append(dict(seed=seed, k=k, face_weight=w, selected=w == weight,
                                  **evaluate_weight(norm["face"], norm["voice"], w, labels, target)))
            for system, (parts, w) in {"voice": (raw, 0.0), "face": (raw, 1.0), "fusion_raw_equal": (raw, 0.5),
                                       "fusion_norm_equal": (norm, 0.5), "fusion_norm_tuned": (norm, weight)}.items():
                systems.append(dict(seed=seed, k=k, system=system, face_weight=w,
                                    **evaluate_weight(parts["face"], parts["voice"], w, labels, target)))
    sweep, systems, selections = pd.DataFrame(sweep), pd.DataFrame(systems), pd.DataFrame(selections)
    sweep.to_csv(out / "weight_sweep.csv", index=False)
    systems.to_csv(out / "systems.csv", index=False)
    selections.to_csv(out / "selected_weights.csv", index=False)
    table = summary_table(systems, selections, target)
    table.to_csv(out / "rq2_table.csv", index=False)
    plot_sweep(sweep, selections, target, out / "rq2_weight_sweep.png", len(config["enrollment_seeds"]))
    write_json(out / "rq2_complete.json", dict(
        status="complete", enrollment_seeds=config["enrollment_seeds"], **s,
        genuine_trials=int(test_genuine.sum()), impostor_trials=int((~test_genuine).sum()),
        feature_signature=feature_signature(config), seconds=time.perf_counter() - start,
        note="Weights and thresholds fitted per k on clean development data only; test never selects."))
    print(table.to_string(index=False), flush=True)
    print(f"Saved RQ2 weight sweep to {out}", flush=True)
    return table


def summary_table(systems, selections, far_target):
    """Mean (and spread) over seeds for each k and system: the table for the slides."""
    frr = f"frr_at_dev_far_{far_target:g}"
    table = systems.groupby(["k", "system"], sort=False).agg(
        hw3_eer_pct=("hw3_eer", "mean"), hw3_eer_pct_std=("hw3_eer", "std"),
        dprime=("dprime", "mean"), dprime_std=("dprime", "std"),
        frr_pct=(frr, "mean"), frr_pct_std=(frr, "std")).reset_index()
    for column in ["hw3_eer_pct", "hw3_eer_pct_std", "frr_pct", "frr_pct_std"]:
        table[column] *= 100
    table = table.rename(columns={"frr_pct": f"frr_pct_at_dev_far_{far_target:g}",
                                  "frr_pct_std": f"frr_pct_at_dev_far_{far_target:g}_std"})
    picked = selections.groupby("k").selected_face_weight.agg(
        lambda w: ", ".join(f"{v:g} x{n}" for v, n in w.value_counts().sort_index().items()))
    table["face_weight"] = [picked[k] if system == "fusion_norm_tuned"
                            else {"voice": "0", "face": "1"}.get(system, "0.5")
                            for k, system in zip(table.k, table.system)]
    table.insert(2, "label", table.system.map(SYSTEM_LABELS))
    return table.round(4)


def plot_sweep(sweep, selections, far_target, path, n_seeds):
    """EER, d-prime and FRR against face weight; one line per enrollment size."""
    frr = f"frr_at_dev_far_{far_target:g}"
    panels = [("hw3_eer", 100, "HW3 EER (%)"), ("dprime", 1, "d-prime (HW3)"),
              (frr, 100, f"FRR (%) at dev FAR = {far_target * 100:g}%")]
    colors = ["#2563a6", "#c06014", "#18856b", "#8057ae", "#bf3b43"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for ax, (metric, scale, label) in zip(axes, panels):
        for color, (k, group) in zip(colors * 4, sweep.groupby("k")):
            stats = group.groupby("face_weight")[metric].agg(["mean", "min", "max"]) * scale
            ax.plot(stats.index, stats["mean"], "o-", color=color, markersize=3, label=f"k = {k}")
            ax.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.12)
            common = selections.loc[selections.k == k].selected_face_weight.mode().min()
            ax.plot([common], [stats.loc[common, "mean"]], "*", color=color, markersize=13)
        ax.set(xlabel="Face weight (0 = voice only, 1 = face only)", ylabel=label, xlim=(-0.03, 1.03))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"RQ2: fusion weight at small clean enrollment sizes (normalized scores; "
                 f"line = mean of {n_seeds} seeds, shading = min to max, star = most common dev-tuned weight)",
                 fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
