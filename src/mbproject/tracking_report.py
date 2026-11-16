"""Step 3 of the tracking study: make the tables (REPORT.md) and figures."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")          # save figures to files, no window
import matplotlib.pyplot as plt
import pandas as pd

from .tracking_eval import output_dir

# Readable names for the feature sets (number = how many features).
NAMES = {
    "au_level": "AU level (17)", "au_motion": "AU motion (17)", "au_velocity": "AU velocity (17)",
    "au_level_motion": "AU level + motion (34)", "au_all": "AU all (51)",
    "lmk_shape": "Landmark shape (136)", "lmk_motion": "Landmark motion (136)", "pdm_shape": "PDM shape (34)",
    "movement_all": "All movement (170)", "shape_all": "All shape (170)", "tracking_all": "All tracking (357)",
}


def percent(value):
    return f"{100 * value:.2f}%"


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
             ("tracking", "All tracking + LDA", "#27ae60", "-"), ("tracking+voice", "All tracking + voice", "#27ae60", "--")]
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
                                  ("tracking+voice", "All tracking + voice", "#27ae60")]:
        ax.plot(range(4), [100 * system_eer(systems, k, n) for n in noise], marker="o", color=color, label=label)
    ax.set_xticks(range(4), ["clean", "20 dB", "10 dB", "0 dB"])
    ax.set_xlabel("Noise added to the probe audio")
    ax.set_ylabel(f"Test EER (%), k = {k}")
    ax.set_title("Does facial tracking help voice under noise?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "noise_voice_fusion.png", dpi=200)
    plt.close(fig)

    # ---- REPORT.md ----
    lines_md = [
        "# Facial-tracking study results",
        "",
        "Made by `python -m mbproject tracking`. Test split, mean of the 10 enrollment seeds, HW3 EER "
        "unless noted. Same 480 genuine / 11,040 impostor trials as the face/voice study.",
        "",
        f"## A. Feature sets (k = {k})",
        "",
        "| Feature set | Raw EER | LDA EER | LDA d-prime | LDA setting (picked on dev) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n in names:
        lines_md.append(f"| {NAMES[n]} | {percent(feature_eer(n, 'raw', k))} | {percent(feature_eer(n, 'lda', k))} | "
                        f"{feature_eer(n, 'lda', k, 'dprime_mean'):.2f} | {chosen.loc[n, 'lda']} |")
    lines_md += ["", f"For comparison at k = {k}: face {percent(system_eer('face', k))}, voice {percent(system_eer('voice', k))}.",
                 "", f"Best feature set on dev data: **{NAMES[info['best_feature_set']]}** ({info['best_lda']}).",
                 "", "![feature sets](figures/feature_sets_eer.png)", "",
                 "## B. Fusion (average of scores)", "",
                 "| Systems | Clean | 20 dB | 10 dB | 0 dB | FRR at dev FAR 1% (clean) |",
                 "| --- | --- | --- | --- | --- | --- |"]
    for systems in ["face", "voice", "au", "tracking", "face+voice", "au+voice", "tracking+voice",
                    "au+face", "au+face+voice", "tracking+face+voice"]:
        lines_md.append(f"| {systems} | " + " | ".join(percent(system_eer(systems, k, n)) for n in noise) +
                        f" | {percent(system_eer(systems, k, column='frr_at_dev_far_0.01_mean'))} |")
    lines_md += ["", "Noise is added to the probe audio only. Face and tracking come from the clean video.", "",
                 "![noise](figures/noise_voice_fusion.png)", "",
                 "## C. Enrollment size", "",
                 "| System | " + " | ".join(f"k={s}" for s in sizes) + " |",
                 "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for systems, label, *_ in lines:
        lines_md.append(f"| {label} | " + " | ".join(percent(system_eer(systems, s)) for s in sizes) + " |")
    lines_md += ["", "![enrollment](figures/eer_vs_enrollment.png)", "", "## Controls", "",
                 "| Control | System | EER | std over splits |", "| --- | --- | --- | --- |"]
    for row in controls.groupby(["control", "system"]).eer.agg(["mean", "std"]).reset_index().itertuples():
        spread = "" if pd.isna(row.std) else percent(row.std)
        lines_md.append(f"| {row.control} | {row.system} | {percent(row.mean)} | {spread} |")
    lines_md += ["",
                 "- shuffled_labels: probe identities scrambled, so about 50% is expected.",
                 "- unseen_actors_raw / _lda: z-score and LDA learned on 12 actors, tested on the other 12 (20 random splits).",
                 "- same/different_content_pairs: pairs of test recordings with the same or a different sentence + emotion."]
    (out / "REPORT.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'REPORT.md'} and 3 figures", flush=True)
