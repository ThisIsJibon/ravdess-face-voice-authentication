"""Step 3 of the tracking study: make the tables (REPORT.md) and figures."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")          # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve

from .tracking_eval import output_dir
from .vendor.hw3_evaluator import Evaluator

# Readable names for the feature sets (number = how many features).
NAMES = {
    "au_level": "AU level (17)", "au_motion": "AU motion (17)", "au_velocity": "AU velocity (17)",
    "au_level_motion": "AU level + motion (34)", "au_all": "AU all (51)",
    "lmk_shape": "Landmark shape (136)", "lmk_motion": "Landmark motion (136)", "pdm_shape": "PDM shape (34)",
    "movement_all": "All movement (170)", "shape_all": "All shape (170)", "tracking_all": "All tracking (357)",
}


def percent(value):
    return f"{100 * value:.2f}%"


def plot_example_scores(scores_path, figures, seed, reference_k, best_label):
    """Figures 4 and 5 plus the HW3 plots, drawn from the saved example scores.

    The example is one seed, k = 16, clean audio: the same seed, k and style as the
    main study's rq2_score_distributions / rq2_roc_det plots. best_label names the
    feature set picked on dev data for the "tracking" system (e.g. "All tracking").
    """
    with np.load(scores_path) as saved:
        plot_scores = {name: saved[name].ravel() for name in saved.files}
    genuine = plot_scores.pop("genuine").astype(bool)
    plot_names = {"face": "Face (SFace)", "voice": "Voice (ECAPA)", "au": "AU + LDA",
                  "au+voice": "AU + voice", "tracking": f"{best_label} + LDA"}
    plot_colors = {"face": "#222222", "voice": "#c0392b", "au": "#2f6db5",
                   "au+voice": "#8e44ad", "tracking": "#27ae60"}

    # Figure 4: genuine vs impostor histograms. Less overlap = better system.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, system in zip(axes.ravel(), ["voice", "au", "au+voice", "tracking"]):
        for label, mask, color in [("Genuine", genuine, "#18856b"), ("Impostor", ~genuine, "#bf3b43")]:
            ax.hist(plot_scores[system][mask], bins=np.linspace(0, 1, 51), density=True,
                    histtype="step", linewidth=1.7, label=label, color=color)
        ax.set(title=plot_names[system], xlabel="Scaled cosine similarity", ylabel="Density", xlim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle(f"Genuine / impostor distributions: clean, seed {seed}, k = {reference_k}")
    fig.savefig(figures / "score_distributions.png", dpi=200)
    plt.close(fig)

    # Figure 5: ROC (top-left is better) and DET (bottom-left is better).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for system in ["face", "voice", "au", "au+voice"]:
        fpr, tpr, _ = roc_curve(genuine, plot_scores[system], drop_intermediate=False)
        axes[0].plot(fpr, tpr, label=plot_names[system], color=plot_colors[system])
        # DET uses normal-deviate axes so small error rates are easier to see.
        axes[1].plot(norm.ppf(np.clip(fpr, 1e-5, 1 - 1e-5)), norm.ppf(np.clip(1 - tpr, 1e-5, 1 - 1e-5)),
                     label=plot_names[system], color=plot_colors[system])
    axes[0].set(xlabel="False acceptance rate", ylabel="Genuine acceptance rate", title="ROC: clean probes")
    ticks = np.array([.0001, .001, .01, .1, .5, .9])
    axes[1].set_xticks(norm.ppf(ticks), ["0.01", "0.1", "1", "10", "50", "90"])
    axes[1].set_yticks(norm.ppf(ticks), ["0.01", "0.1", "1", "10", "50", "90"])
    axes[1].set(xlabel="False acceptance rate (%)", ylabel="False rejection rate (%)", title="DET")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Seed {seed}; {reference_k} enrollment recordings per actor")
    fig.savefig(figures / "roc_det.png", dpi=200)
    plt.close(fig)

    # HW3 script's own plots (score distribution, DET, ROC) for AU and AU + voice,
    # called exactly like the main study does for face and voice.
    hw3_folder = figures / "hw3_original"
    hw3_folder.mkdir(exist_ok=True)
    previous = os.getcwd()
    try:
        os.chdir(hw3_folder)                     # the HW3 script saves into the current folder
        for system in ["au", "au+voice"]:
            evaluator = Evaluator(500, plot_scores[system][genuine], plot_scores[system][~genuine], system)
            fpr, fnr, tpr = evaluator.compute_rates()
            evaluator.plot_score_distribution()
            evaluator.plot_det_curve(fpr, fnr)
            evaluator.plot_roc_curve(fpr, tpr)
            plt.close("all")
    finally:
        os.chdir(previous)


def write_report(config):
    out = output_dir(config)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    k = config["tracking"]["selection_enrollment_size"]
    info = json.loads((out / "run_complete.json").read_text())
    summary = pd.read_csv(out / "summary.csv")
    fusion = pd.read_csv(out / "fusion_summary.csv")
    controls = pd.read_csv(out / "controls.csv")
    chosen = pd.read_csv(out / "dev_selected.csv").set_index("feature_set")
    # The "tracking" system is whichever feature set won on dev data, so name it from the run.
    best_label = NAMES[info["best_feature_set"]].split(" (")[0]          # e.g. "All tracking"

    def feature_eer(name, rep, size, column="hw3_eer_mean"):
        row = summary[(summary.feature_set == name) & (summary.representation == rep) & (summary.k == size)]
        return float(row[column].iloc[0])

    def system_eer(systems, size, noise="clean", column="hw3_eer_mean"):
        row = fusion[(fusion.systems == systems) & (fusion.k == size) & (fusion.audio_snr_db.astype(str) == noise)]
        return float(row[column].iloc[0])

    # ---- Figure 1: error of every feature set, raw vs LDA ----
    names = [n for n in NAMES if n in chosen.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(names))
    ax.barh([p + 0.2 for p in positions], [100 * feature_eer(n, "raw", k) for n in names],
            height=0.4, color="#b0b7c3", label="raw (z-score only)")
    ax.barh([p - 0.2 for p in positions], [100 * feature_eer(n, "lda", k) for n in names],
            height=0.4, color="#2f6db5", label="with LDA")
    ax.axvline(100 * system_eer("voice", k), color="#c0392b", ls="--", label="voice")
    ax.set_yticks(list(positions), [NAMES[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"Test EER (%), k = {k}")
    ax.set_title("Which facial-tracking features identify a person?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "feature_sets_eer.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: error vs number of enrollment recordings ----
    sizes = sorted(summary.k.unique())
    lines = [("face", "Face (SFace)", "#222222", "-"), ("voice", "Voice (ECAPA)", "#c0392b", "-"),
             ("au", "AU + LDA", "#2f6db5", "-"), ("au+voice", "AU + voice", "#2f6db5", "--"),
             ("tracking", f"{best_label} + LDA", "#27ae60", "-"), ("tracking+voice", f"{best_label} + voice", "#27ae60", "--")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for systems, label, color, style in lines:
        ax.plot(sizes, [100 * system_eer(systems, s) for s in sizes], style, marker="o", color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("Enrollment recordings per person (k)")
    ax.set_ylabel("Test EER (%)")
    ax.set_title("Error vs enrollment size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "eer_vs_enrollment.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: noisy voice with and without tracking ----
    noise = ["clean", "20", "10", "0"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for systems, label, color in [("voice", "Voice", "#c0392b"), ("au+voice", "AU + voice", "#2f6db5"),
                                  ("tracking+voice", f"{best_label} + voice", "#27ae60")]:
        ax.plot(range(4), [100 * system_eer(systems, k, n) for n in noise], marker="o", color=color, label=label)
    ax.set_xticks(range(4), ["clean", "20 dB", "10 dB", "0 dB"])
    ax.set_xlabel("Noise added to the probe audio")
    ax.set_ylabel(f"Test EER (%), k = {k}")
    ax.set_title("Does facial tracking help voice under noise?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "noise_voice_fusion.png", dpi=200)
    plt.close(fig)

    # ---- Figures 4 and 5 + HW3 plots, from the example scores saved by run_tracking ----
    plot_example_scores(out / "plot_scores.npz", figures, config["enrollment_seeds"][0],
                        max(config["enrollment_sizes"]), best_label)
    seed, reference_k = config["enrollment_seeds"][0], max(config["enrollment_sizes"])

    # ---- REPORT.md ----
    lines_md = [
        "# Facial-tracking study results",
        "",
        f"Made by `python -m mbproject tracking`. Test split, mean of the {len(config['enrollment_seeds'])} "
        "enrollment seeds, HW3 EER unless noted. Same 480 genuine / 11,040 impostor trials as the face/voice study.",
        "",
        "HW3 EER checks 500 fixed thresholds. 'Exact EER' interpolates the ROC curve between the real scores. "
        "The two agree closely here; for very small EERs, read both.",
        "",
        f"## A. Feature sets (k = {k})",
        "",
        "| Feature set | Raw EER | LDA EER | LDA exact EER | LDA d-prime | LDA setting (picked on dev) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        lines_md.append(f"| {NAMES[n]} | {percent(feature_eer(n, 'raw', k))} | {percent(feature_eer(n, 'lda', k))} | "
                        f"{percent(feature_eer(n, 'lda', k, 'eer_mean'))} | {feature_eer(n, 'lda', k, 'dprime_mean'):.2f} | {chosen.loc[n, 'lda']} |")
    lines_md += ["", f"For comparison at k = {k}: face {percent(system_eer('face', k))}, voice {percent(system_eer('voice', k))}.",
                 "", f"Best feature set on dev data: **{NAMES[info['best_feature_set']]}** ({info['best_lda']}).",
                 "", "![feature sets](figures/feature_sets_eer.png)", "",
                 "## B. Fusion (average of scores)", "",
                 "| Systems | Clean | 20 dB | 10 dB | 0 dB | Exact EER (clean) | Test FAR / FRR at dev FAR 1% (clean) | Same at 0 dB |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for systems in ["face", "voice", "au", "tracking", "face+voice", "au+voice", "tracking+voice",
                    "au+face", "au+face+voice", "tracking+face+voice"]:
        # The threshold is set on dev data for FAR = 1%; show the FAR it really gives on test next to the FRR,
        # so a lower FRR is not just a looser threshold.
        operating = [f"{percent(system_eer(systems, k, n, 'far_at_dev_far_0.01_mean'))} / "
                     f"{percent(system_eer(systems, k, n, 'frr_at_dev_far_0.01_mean'))}" for n in ["clean", "0"]]
        lines_md.append(f"| {systems} | " + " | ".join(percent(system_eer(systems, k, n)) for n in noise) +
                        f" | {percent(system_eer(systems, k, column='eer_mean'))} | {operating[0]} | {operating[1]} |")
    lines_md += ["", "Noise is added to the probe audio only. Face and tracking come from the clean video.", "",
                 "![noise](figures/noise_voice_fusion.png)", "",
                 "## C. Enrollment size", "",
                 "| System | " + " | ".join(f"k={s}" for s in sizes) + " |",
                 "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for systems, label, *_ in lines:
        lines_md.append(f"| {label} | " + " | ".join(percent(system_eer(systems, s)) for s in sizes) + " |")
    lines_md += ["", "![enrollment](figures/eer_vs_enrollment.png)", "",
                 f"## D. Score distributions and ROC/DET (seed {seed}, k = {reference_k}, clean)", "",
                 "Same seed and k as the main study's plots. The HW3 script's own score distribution, "
                 "DET and ROC plots for AU and AU + voice are in `figures/hw3_original/`.", "",
                 "![score distributions](figures/score_distributions.png)", "",
                 "![ROC and DET](figures/roc_det.png)", "", "## Controls", "",
                 "| Control | System | EER | std over splits |", "| --- | --- | --- | --- |"]
    for row in controls.groupby(["control", "system"]).eer.agg(["mean", "std"]).reset_index().itertuples():
        spread = "" if pd.isna(row.std) else percent(row.std)
        lines_md.append(f"| {row.control} | {row.system} | {percent(row.mean)} | {spread} |")
    lines_md += ["",
                 "Controls use the interpolated ROC EER and the first enrollment seed (not the 10-seed HW3 EER above).",
                 "- shuffled_labels: probe identities scrambled; scores should fall to chance (about 50%).",
                 "- unseen_actors_raw / _lda: actors excluded from LDA fitting. z-score and LDA learned on 12 actors, tested on the other 12 "
                 "(20 random splits). Exploratory only: the LDA settings and best feature set were chosen on dev data "
                 "from all 24 actors, and this control also changes the training population, the claimed-identity roster "
                 "and the maximum LDA dimensions, so the gap is not only 'LDA learning these actors'.",
                 "- same/different_stimulus_pairs: pairs of test recordings with the same or a different stimulus "
                 "(statement + emotion + intensity together). It does not separate sentence from emotion; "
                 "same-stimulus genuine pairs are the two repetitions of that stimulus.",
                 "- Seed spread measures enrollment resampling only, not new users or sessions."]
    (out / "REPORT.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'REPORT.md'} and 5 figures (+ HW3 plots)", flush=True)
