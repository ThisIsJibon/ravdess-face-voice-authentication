"""Scientific plots generated exclusively from saved experimental outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve

from .experiment import SYSTEMS, condition_id
from .vendor.hw3_evaluator import Evaluator

LABELS = dict(voice="Voice",face="Face",fusion_fixed="Fusion: equal weight",fusion_tuned="Fusion: dev weight")
COLORS = dict(voice="#2563a6",face="#c06014",fusion_fixed="#18856b",fusion_tuned="#8057ae")


def generate(config):
    out = Path(config["results_dir"])
    if not (out/"run_complete.json").exists():
        raise ValueError("Generate plots only after a complete evaluation")
    folder = out/"figures"
    folder.mkdir(exist_ok=True)
    metrics = pd.read_csv(out/"metrics.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
        "grid.alpha":.2,"figure.dpi":120,"savefig.dpi":180})
    def save(fig,name):
        fig.savefig(folder/f"{name}.png",bbox_inches="tight")
        plt.close(fig)
    reference = max(config["enrollment_sizes"])
    clean = metrics.loc[metrics.family == "clean_enrollment"]
    diag = clean.loc[clean.k_face == clean.k_voice]
    for metric,label,name in [("eer","Equal error rate (%)","rq1_enrollment_eer"),
                              ("dprime","d-prime (HW3)","rq2_enrollment_dprime"),
                              ("frr_at_dev_far_0.01","Test FRR (%) at fixed dev FAR=1% threshold","rq1_enrollment_operating_frr")]:
        fig,ax = plt.subplots(figsize=(8.2,4.6),layout="constrained")
        scale = 1 if metric == "dprime" else 100
        for system in SYSTEMS:
            group = diag.loc[diag.system == system].groupby("k_face")[metric]
            mean,lo,hi = group.mean()*scale,group.min()*scale,group.max()*scale
            ax.plot(mean.index,mean,"o-",color=COLORS[system],label=LABELS[system])
            ax.fill_between(mean.index,lo,hi,color=COLORS[system],alpha=.12)
        ax.set(xlabel="Enrollment recordings per actor per modality",ylabel=label,
               title="Enrollment scarcity: same probes, frozen models and decision policies")
        ax.set_xticks(config["enrollment_sizes"])
        ax.legend(fontsize=8)
        fig.supxlabel(f"Lines: mean of {len(config['enrollment_seeds'])} enrollment seeds; shading: seed minimum to maximum",fontsize=8)
        save(fig,name)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout="constrained",sharey=True)
    for ax,vary in zip(axes,["face","voice"]):
        changing,held = f"k_{vary}","k_voice" if vary == "face" else "k_face"
        for system in SYSTEMS:
            group = clean.loc[(clean[held] == reference)&(clean.system == system)].groupby(changing).eer.mean()*100
            ax.plot(group.index,group,"o-",label=LABELS[system],color=COLORS[system])
        ax.set(xlabel=f"{vary.title()} enrollment recordings",ylabel="EER (%)",
               title=f"Vary {vary}; other modality fixed at {reference}")
        ax.set_xticks(config["enrollment_sizes"])
    axes[-1].legend(fontsize=8)
    save(fig,"rq1_independent_modality_enrollment")
    seed = config["enrollment_seeds"][0]
    clean_name = condition_id(reference,reference,None,0)
    with np.load(out/"trial_index.npz",allow_pickle=False) as z:
        genuine = z["genuine"].ravel()
    with np.load(out/"trial_scores"/str(seed)/f"{clean_name}.npz",allow_pickle=False) as z:
        scores = {s:z[s].ravel() for s in SYSTEMS}
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for system in SYSTEMS:
        fpr,tpr,_ = roc_curve(genuine,scores[system],drop_intermediate=False)
        axes[0].plot(fpr,tpr,label=LABELS[system],color=COLORS[system])
        axes[1].plot(norm.ppf(np.clip(fpr,1e-5,1-1e-5)),norm.ppf(np.clip(1-tpr,1e-5,1-1e-5)),
                     label=LABELS[system],color=COLORS[system])
    axes[0].set(xlabel="False acceptance rate",ylabel="Genuine acceptance rate",title="ROC: clean probes")
    ticks = np.array([.0001,.001,.01,.1,.5,.9])
    axes[1].set_xticks(norm.ppf(ticks),["0.01","0.1","1","10","50","90"])
    axes[1].set_yticks(norm.ppf(ticks),["0.01","0.1","1","10","50","90"])
    axes[1].set(xlabel="False acceptance rate (%)",ylabel="False rejection rate (%)",title="DET: normal-deviate axes")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Representative seed {seed}; {reference} enrollment recordings per modality")
    save(fig,"rq2_roc_det")
    fig,axes = plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,system in zip(axes.ravel(),SYSTEMS):
        for label,mask,color in [("Genuine",genuine,"#18856b"),("Impostor",~genuine,"#bf3b43")]:
            ax.hist(scores[system][mask],bins=np.linspace(0,1,51),density=True,
                    histtype="step",linewidth=1.7,label=label,color=color)
        ax.set(title=LABELS[system],xlabel="Scaled cosine similarity",ylabel="Density",xlim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle(f"Genuine / impostor distributions: clean, seed {seed}, k={reference}")
    save(fig,"rq2_score_distributions")
    for metric,label,suffix in [("eer","EER (%)","eer"),
                                ("frr_at_dev_far_0.01","FRR (%) at frozen dev FAR=1% threshold","frr")]:
        fig,axes = plt.subplots(1,4,figsize=(14,4.1),layout="constrained",sharey=True)
        selected = metrics.loc[(metrics.k_face == reference)&(metrics.k_voice == reference)]
        grouped = selected.groupby(["system","face_sigma","audio_snr_db"])[metric].mean()*100
        grids = []
        snr_keys = ["clean" if v is None else str(v) for v in config["audio_snr_db"]]
        for system in SYSTEMS:
            grid = grouped.loc[system].unstack("audio_snr_db").reindex(index=config["face_blur_sigma"],columns=snr_keys)
            grids.append(grid.to_numpy())
        vmax = max(.01,max(np.nanmax(g) for g in grids))
        for ax,system,grid in zip(axes,SYSTEMS,grids):
            im = ax.imshow(grid,aspect="auto",vmin=0,vmax=vmax,cmap="magma_r")
            for (i,j),value in np.ndenumerate(grid):
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if value > vmax*.55 else "black")
            ax.set_xticks(range(len(snr_keys)),snr_keys)
            ax.set_yticks(range(len(config["face_blur_sigma"])),config["face_blur_sigma"])
            ax.set(xlabel="Audio SNR (dB)",title=LABELS[system])
            ax.grid(False)
        axes[0].set_ylabel("Gaussian blur sigma (aligned 112px face)")
        fig.colorbar(im,ax=axes,label=label,shrink=.8)
        fig.suptitle(f"Noise x blur: k={reference}; mean over enrollment seeds")
        save(fig,f"rq3_degradation_{suffix}_heatmaps")
    users = pd.read_csv(out/"per_user.csv")
    extreme = condition_id(reference,reference,config["audio_snr_db"][-1],config["face_blur_sigma"][-1])
    fig,axes = plt.subplots(2,1,figsize=(11,6.4),layout="constrained",sharex=True)
    for ax,name,title in zip(axes,[clean_name,extreme],["Clean","Strongest configured degradation"]):
        for system in SYSTEMS:
            group = users.loc[(users.condition == name)&(users.system == system)].groupby("actor").frr.mean()*100
            ax.plot(group.index,group,"o-",color=COLORS[system],label=LABELS[system],markersize=3)
        ax.set(ylabel="FRR (%)",title=title)
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_xticks(range(1,25))
    axes[1].set_xlabel("RAVDESS actor (descriptive user variation; not a fairness conclusion)")
    save(fig,"per_actor_operating_frr")
    quality_plot(config,save)
    rq3_curves(config,metrics,out,save)
    # Execute original HW3 plot methods, retaining the submitted code byte-for-byte.
    legacy = folder/"hw3_original"
    legacy.mkdir(exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(legacy)
        for system in SYSTEMS:
            evaluator = Evaluator(500,scores[system][genuine],scores[system][~genuine],system)
            fpr,fnr,tpr = evaluator.compute_rates()
            evaluator.plot_score_distribution()
            evaluator.plot_det_curve(fpr,fnr)
            evaluator.plot_roc_curve(fpr,tpr)
    finally:
        os.chdir(previous)
    print(f"Saved experiment figures and original HW3 figures to {folder}",flush=True)


def quality_plot(config,save):
    records = [json.loads(p.read_text()) for p in (Path(config["data_dir"])/"features").glob("*.json")]
    fig,axes = plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    for snr in config["audio_snr_db"]:
        if snr is not None:
            values = [r["measured_snr_db"][f"audio_snr{snr:g}"] for r in records]
            axes[0].scatter([snr],[np.mean(values)],color="#2563a6")
    axes[0].plot([0,20],[0,20],"--",color="gray")
    axes[0].set(xlabel="Requested audio SNR (dB)",ylabel="Measured audio SNR (dB)",title="Noise manipulation check")
    for sigma in config["face_blur_sigma"]:
        values = [r["laplacian_variance"][f"face_sigma{sigma:g}"] for r in records]
        lo,mid,hi = np.quantile(values,[.1,.5,.9])
        axes[1].errorbar([sigma],[mid],yerr=[[mid-lo],[hi-mid]],fmt="o",capsize=5,color="#c06014")
    axes[1].set(xlabel="Gaussian blur sigma (pixels)",ylabel="Laplacian variance",yscale="log",
                title="Blur check: median and 10th-90th percentiles")
    save(fig,"degradation_manipulation_checks")


def rq3_curves(config,metrics,out,save):
    """RQ3 severity curves for every degradation enrollment size, plus a summary table.

    Columns: audio noise only (face sharp), face blur only (audio clean), and both
    together at matched severity steps. Rows: EER and FRR at the frozen clean
    development FAR=1% threshold. Lines are means over enrollment seeds; bands
    show +/- one standard deviation across seeds.
    """
    snrs = ["clean" if v is None else str(v) for v in config["audio_snr_db"]]
    sigmas = list(config["face_blur_sigma"])
    data = metrics.assign(audio_snr_db=metrics.audio_snr_db.astype(str))
    snr_tick = lambda s: s if s == "clean" else f"{s} dB"
    sweeps = [("Audio noise only (face sharp)",[(s,sigmas[0]) for s in snrs],[snr_tick(s) for s in snrs]),
              ("Face blur only (audio clean)",[(snrs[0],g) for g in sigmas],[f"sigma {g:g}" for g in sigmas]),
              ("Both together",list(zip(snrs,sigmas)),[f"{snr_tick(s)}\nsigma {g:g}" for s,g in zip(snrs,sigmas)])]
    panels = [("eer","EER (%)"),("frr_at_dev_far_0.01","FRR (%) at frozen\nclean dev FAR=1% threshold")]
    for k in config["degradation_enrollment_sizes"]:
        subset = data.loc[(data.k_face == k)&(data.k_voice == k)]
        fig,axes = plt.subplots(2,3,figsize=(14,7.4),layout="constrained",sharey="row")
        for col,(title,points,ticks) in enumerate(sweeps):
            x = np.arange(len(points))
            for row,(metric,ylabel) in enumerate(panels):
                ax = axes[row,col]
                for system in SYSTEMS:
                    values = [subset.loc[(subset.system == system)&(subset.audio_snr_db == snr)
                                         &(subset.face_sigma == sigma),metric]*100 for snr,sigma in points]
                    if any(v.empty for v in values):
                        raise ValueError(f"Missing RQ3 rows for k={k}, system={system}")
                    mean = np.array([v.mean() for v in values])
                    sd = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in values])
                    ax.plot(x,mean,"o-" if system != "fusion_tuned" else "s--",color=COLORS[system],
                            label=LABELS[system],markersize=4,linewidth=1.6)
                    ax.fill_between(x,np.clip(mean-sd,0,None),mean+sd,color=COLORS[system],alpha=.12,linewidth=0)
                ax.set_xticks(x,ticks)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(title)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"RQ3 degradation severity, k={k} clean enrollment recordings per modality; mean +/- SD over enrollment seeds")
        save(fig,f"rq3_severity_curves_k{k}")
    table = (data.loc[(data.k_face == data.k_voice)&data.k_face.isin(config["degradation_enrollment_sizes"])]
             .groupby(["k_face","audio_snr_db","face_sigma","system"])
             [["eer","frr_at_dev_far_0.01","far_at_dev_far_0.01"]].agg(["mean","std"]))
    table.columns = ["_".join(c) for c in table.columns]
    table = table.reset_index().rename(columns={"k_face":"k"})
    table["audio_snr_db"] = pd.Categorical(table.audio_snr_db,snrs,ordered=True)
    table.sort_values(["k","audio_snr_db","face_sigma","system"]).to_csv(out/"rq3_summary.csv",index=False)
