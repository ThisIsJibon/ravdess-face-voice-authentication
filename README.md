# Mobile Biometrics Final Project

Python implementation of RAVDESS face + voice **identity verification** for the three approved research questions. This repository contains the experiments, a working claim-verification demo and the final report (`Mobile_Biometrics_Report.docx`). The presentation is not included.

## Run the whole project step by step

Use **Python 3.11**. Run commands from the repository root, which contains `pyproject.toml`. All paths below are relative to that folder. The submission uses **30 enrollment seeds**; commands without `--config` use the older ten-seed study instead.

### 1. Get the source

```bash
git clone https://github.com/ThisIsJibon/ravdess-face-voice-authentication.git
cd ravdess-face-voice-authentication
```

If you already have the repository, open a terminal in its root and continue below. You can also download and extract the source ZIP from GitHub. A source ZIP does not support Git commands.

### 2. Put the dataset archives in place

Download the speech archives from [official RAVDESS](https://doi.org/10.5281/zenodo.1188976). Download facial tracking from the [separate tracking release](https://doi.org/10.5281/zenodo.3255102) for the AU study. Keep the files zipped:

```text
ravdess-face-voice-authentication/
  pyproject.toml
  RAVDESS/
    Audio_Speech_Actors_01-24.zip
    Video_Speech_Actor_01.zip
    Video_Speech_Actor_02.zip
    ...
    Video_Speech_Actor_24.zip
    FacialTracking_Actors_01-24.zip
```

The core experiment needs the audio archive and **all 24 video speech archives**. The tracking command also needs the tracking archive. Song archives are unused. Do not extract the archives manually; `prepare` extracts the paired speech clips. Allow space for the original archives, two extracted media/cache directories, models, the environment and generated results. The stronger-blur run has a separate data directory and can extract another copy of the media.

### 3. Install and activate an environment

Choose the instructions for your OS. Installation and the first model download need internet access. The experiments process biometric data locally.

**Windows PowerShell, CPU:**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Cpu
.\.venv\Scripts\Activate.ps1
python --version
```

For a compatible NVIDIA GPU and driver, omit `-Cpu` to install the script's CUDA 12.8 build. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. This policy change lasts for that process. You can also skip activation and replace every `python` below with `& .\.venv\Scripts\python.exe`.

**macOS, CPU:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1
python -m pip install -e . pytest==8.3.5
python -m pip check
```

**Linux, CPU:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . pytest==8.3.5
python -m pip check
```

Do not copy another computer's `.venv` or install the Windows CUDA `requirements-lock.txt` on macOS/Linux. Reopen the environment with the activation command whenever you open a new terminal. On macOS/Linux, `.venv/bin/python` also works without activation.

### 4. Check the installation

After activation, these commands work in PowerShell and macOS/Linux shells:

```bash
python --version
python -m pip check
python -m mbproject --help
python -m pytest -q
```

Confirm Python reports **3.11.x**. The tests check the implementation; they do not run the complete dataset experiment.

### 5. Run the 30-seed core study

```bash
python -u -m mbproject --config configs/study_submission_30_seeds.yaml all
```

`all` runs `prepare`, `extract`, `evaluate`, `plots` and `audit`, in that order. Extraction downloads the pinned models when needed. The run covers RQ1, the original RQ2 fusion comparison and the original RQ3 noise/blur grid. It reuses valid existing media and feature caches. It does **not** run the separate `rq2` or `tracking` commands.

To run the core stages separately, use this sequence **instead of** `all`:

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml prepare
python -m mbproject --config configs/study_submission_30_seeds.yaml models
python -u -m mbproject --config configs/study_submission_30_seeds.yaml extract
python -m mbproject --config configs/study_submission_30_seeds.yaml evaluate
python -m mbproject --config configs/study_submission_30_seeds.yaml plots
python -m mbproject --config configs/study_submission_30_seeds.yaml audit
```

Keep `--config` **before** the stage name. `models` pre-downloads the encoders; extraction also downloads them automatically if missing. A first run includes neural inference and can take much longer than re-evaluating cached features. Duration depends on hardware and disk speed.

### 6. Run the RQ2 normalization and weight study

After the core study:

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml rq2
```

This compares raw 50/50, normalized 50/50 and development-tuned normalized fusion at k = 1, 2 and 4. The sweep includes endpoint weights 0 and 1 as references; tuning selects among mixed weights 0.1 through 0.9. It writes `results/30_seeds/rq2_weights/`, including `rq2_table.csv`, `selected_weights.csv`, `systems.csv`, `rq2_weight_sweep.png` and `rq2_complete.json`. Test data does not select weights or thresholds.

### 7. Run the AU and landmark study

After the core study and with the tracking ZIP in `RAVDESS/`:

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml tracking
```

This uses supplied OpenFace tracking as another facial representation, with the same enrollment seeds, partitions and trials. Start with `results/tracking_30_seeds/REPORT.md`, its CSV tables and figures. See [the tracking study](docs/TRACKING_STUDY.md) for feature definitions and controls.

### 8. Run the stronger-blur RQ3 study

```bash
python -u -m mbproject --config configs/study_strong_blur_30_seeds.yaml all
```

This adds sigma 6 and 8, using `data/strong_blur/` for media/features and `results/strong_blur_30_seeds/` for results. It keeps the same 30 enrollment seeds and original fixed fusion policy. It does not use the clean-only RQ2 extension's selected weight. A separate cache is required because blur levels affect the feature signature.

### 9. Confirm completion and read the outputs

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml audit
python -m mbproject --config configs/study_strong_blur_30_seeds.yaml audit
```

Both audits should report `"passed": true`. Check these locations:

| Run | Files to read first |
|---|---|
| Core: `results/30_seeds/` | `run_complete.json`, `audit.json`, `summary.csv`, `metrics.csv`, `figures/` |
| RQ2: `results/30_seeds/rq2_weights/` | `rq2_complete.json`, `rq2_table.csv`, `selected_weights.csv`, `rq2_weight_sweep.png` |
| AU: `results/tracking_30_seeds/` | `run_complete.json`, `REPORT.md`, `summary.csv`, `fusion_summary.csv`, `controls.csv`, `figures/` |
| Strong blur: `results/strong_blur_30_seeds/` | `run_complete.json`, `audit.json`, `rq3_summary.csv`, `figures/` |

Completion markers must belong to the current configuration; incomplete extraction is not a completed experiment. Use `hw3_eer` for the primary EER in the report. CSV error rates are fractions unless the column name explicitly indicates percent; multiply fractions by 100 for percentages. `eer` is supplementary interpolated ROC EER. Thirty seeds vary enrollment selection, not the number of independent test recordings.

### 10. Try a genuine and impostor claim

After the core run, print a held-out actor 1 recording ID:

```bash
python -c "import pandas as pd; f=pd.read_csv('data/manifest.csv'); print(f.loc[(f.split == 'test') & (f.actor == 1), ['recording_id', 'actor']].head().to_string(index=False))"
```

Replace `RECORDING_ID` below with one printed value:

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml verify --recording RECORDING_ID --claim 1
python -m mbproject --config configs/study_submission_30_seeds.yaml verify --recording RECORDING_ID --claim 2
python -m mbproject --config configs/study_submission_30_seeds.yaml verify --recording RECORDING_ID --claim 1 --snr 0 --sigma 4
```

The first claim uses the true actor; the second claims a different actor. The output contains scores, the saved development threshold and an ACCEPT/REJECT decision. Genuine claims can still be rejected and impostor claims can be accepted. This is a research demo without replay or liveness protection.

### Troubleshooting and reruns

- **`No module named mbproject`:** activate this repository's environment and run `python -m pip install -e .`.
- **`No module named pip`:** the environment was created without pip. Run `python -m ensurepip --upgrade`, then repeat the package installation steps for your OS.
- **Missing archives:** check the exact filenames in step 2 and keep all archives inside the root `RAVDESS/` folder.
- **No feature cache:** run `prepare` and `extract` with the same configuration before `evaluate` or `rq2`; run the core `all` before `tracking`.
- **Stale cache/signature mismatch:** use the original code/configuration for the old cache, or a separate `data_dir` and `results_dir` for changed extraction settings. Preserve existing results before deliberately replacing them.
- **Interrupted extraction:** rerun the same `all` or `extract` command to reuse valid cached recordings. Rerun later stages after extraction completes; do not treat a partial run as final results.
- **Only need new tables or plots:** use `evaluate` or `plots` with the matching `--config`; these reuse complete cached features/results respectively. Evaluation and plotting replace generated outputs in that run's results directory.
- **GPU unavailable:** use the CPU installation. The supplied configurations use `device: auto`; CPU extraction can be slower.
- **Sharing results:** `.gitignore` excludes archives, environments, data, models, results, `output/` and presentations. A Git push does not upload ignored local deliverables. An already tracked report remains tracked even if an ignore rule names it.

The sections below describe the protocol and legacy ten-seed comparisons. Use the explicit 30-seed commands above to reproduce the submission.

## What is implemented

| Research question | Experiments |
|---|---|
| RQ1: Enrollment data scarcity | 1, 2, 4, 8, 12, and 16 recordings per actor. Vary both modalities together; additionally vary one while fixing the other at 16. |
| RQ2: Multimodal fusion | Voice, face, equal-weight score fusion, and development-tuned weighted score fusion on the same genuine/impostor trials. |
| RQ3: Data degradation | Probe audio: clean, 20, 10, 0 dB additive white Gaussian noise. Probe face: Gaussian blur sigma 0, 1, 2, 4 pixels on aligned 112 x 112 faces. Full 4 x 4 grid at enrollment sizes 1, 4, 16. |

There are **61 conditions x 10 enrollment seeds x 4 systems = 2,440 evaluations**. Clean enrollment conditions serve both RQ1 and RQ2. The three questions progress from sample availability to fusion comparisons to robustness of the same systems. Each seed selects a preferred fusion using **development data only**; `fusion_selection.csv` records that selection. No improvement is assumed, and both fusion methods are retained even when a unimodal system performs better.

### Extended RQ1 enrollment evaluation

An additional 30-seed evaluation is configured in `configs/study_30_seeds.yaml`. It includes the original ten enrollment seeds and 20 additional unique seeds, using the same dataset partitions, models, enrollment sizes, conditions, and evaluation systems.

The revised submission report uses **30 enrollment seeds throughout RQ1, RQ2, RQ3 and the AU extension**. Use **HW3 EER** as the primary metric in the slides as well: clean voice at k = 1 to 16 is **7.31% to 2.21%**. The **7.32%** starting value is the supplementary interpolated ROC EER for the same 30 seeds. The earlier **7.69% to 2.22%** figures describe the original ten-seed HW3 run.

`configs/study_submission_30_seeds.yaml` reproduces the submission's core, RQ2 weight sweep and tracking study. `configs/study_strong_blur_30_seeds.yaml` reproduces its full stronger-blur grid using the separate `data/strong_blur` feature cache. Verified submission outputs are `results/30_seeds/`, `results/30_seeds/rq2_weights/`, `results/tracking_30_seeds/` and `results/strong_blur_30_seeds/`. Each comparison uses the same 30 seeds; the AU controls explicitly use the first seed and their own stated actor splits.

```bash
python -m mbproject --config configs/study_submission_30_seeds.yaml evaluate
python -m mbproject --config configs/study_submission_30_seeds.yaml rq2
python -m mbproject --config configs/study_submission_30_seeds.yaml tracking
python -m mbproject --config configs/study_strong_blur_30_seeds.yaml prepare
python -m mbproject --config configs/study_strong_blur_30_seeds.yaml extract
python -m mbproject --config configs/study_strong_blur_30_seeds.yaml evaluate
python -m mbproject --config configs/study_strong_blur_30_seeds.yaml plots
```

Use the revised report tables for the other slide results too. For example, the 30-seed k = 4 noisy-voice comparison is **13.76% voice versus 2.01% AU + voice**. At k = 16 with 0 dB audio and sigma 4 blur, mean HW3 EER is **11.43% voice, 1.53% face and 0.76% fusion**. These are enrollment-seed means, not additional independent test recordings.

The extended run contains **61 conditions x 30 enrollment seeds x 4 systems = 7,320 evaluations**.

Run the extended evaluation and regenerate its figures with:

```bash
python -m mbproject --config configs/study_30_seeds.yaml evaluate
python -m mbproject --config configs/study_30_seeds.yaml plots
```

Its outputs are stored separately under `results/30_seeds/`. The original `configs/study.yaml` and `results/full/` remain unchanged.

The additional seeds examine variation in enrollment selection. They do not increase the number of independent test recordings: each condition still contains 480 genuine and 11,040 impostor comparisons. The original ten seeds are included in the 30-seed run, so comparisons between the two runs are overlapping rather than independent replications.

### Extended RQ3 blur evaluation

The original face blur conditions have 0% EER at sigma 0, 1 and 2, with errors emerging at sigma 4. An additional run in `configs/study_strong_blur.yaml` adds two heavier levels: **face blur sigma 0, 1, 2, 4, 6, 8**. Everything else (seeds, splits, models, SNR levels, enrollment sizes) is identical to `configs/study.yaml`. In the verification run, all 2,440 evaluations shared with the original run reproduced its metrics exactly.

The run contains **85 conditions x 10 enrollment seeds x 4 systems = 3,400 evaluations**. It needs its own feature extraction because the blur levels are part of the feature-cache signature:

```powershell
& .\.venv\Scripts\python.exe -u -m mbproject --config configs/study_strong_blur.yaml all
```

Outputs are stored separately under `data_strong_blur/` and `results/strong_blur/`; `results/full/` is not touched. On a GTX 1650 laptop GPU the teammate's full run took about 30 minutes. The independent CPU verification reused the original raw media with a separate feature cache; its resolved configuration is saved in `results/strong_blur/config_resolved.json`.

Verified results at **k = 16 with clean audio**, averaged over the ten enrollment seeds:

| Blur sigma | Face HW3 EER (%) | Face interpolated EER (%) | Equal-weight fusion HW3 EER (%) | Equal-weight fusion interpolated EER (%) |
|---|---:|---:|---:|---:|
| 4 | 1.56 | 1.56 | 0.15 | 0.16 |
| 6 | 9.92 | 9.89 | 0.90 | 0.90 |
| 8 | 28.18 | 28.23 | 1.79 | 1.77 |

At the frozen clean-development FAR=1% thresholds, face FRR is 82.98% at sigma 6 and 100% at sigma 8. At sigma 8, fusion FRR is 11.15%, versus 3.23% for clean voice alone. Their actual test FARs differ: 0.014% for fusion and 1.252% for voice. These are rejection rates for genuine **attempts**, not percentages of distinct users. Fusion improves EER over either input at sigma 6 and 8, but the frozen operating policies have different error trade-offs. The figures above come from the independent rerun; keep each numerical claim tied to its saved run and EER definition.

### RQ3 severity curves

`mbproject plots` also writes `figures/rq3_severity_curves_k{1,4,16}.png`, so the k=1 and k=4 degradation results are visible alongside the k=16 heatmaps. Each figure shows interpolated EER and FRR at the frozen clean-development FAR=1% threshold, with means and +/- one standard deviation over enrollment seeds. The panels vary audio noise with sharp faces, face blur with clean audio, and **every configured blur level with audio held at the lowest configured SNR** (0 dB in the supplied configurations). Thus the joint panel includes sigma 6 and 8 without treating audio SNR and blur sigma as equivalent scales. Percentage bands are clipped to 0-100 for display. `rq3_summary.csv` contains mean and standard deviation of interpolated EER, HW3 EER, d-prime, and operating FAR/FRR; error-rate columns are fractions.

### Running on Google Colab

Colab exports `MPLBACKEND=module://matplotlib_inline...`, which the project's separate Python 3.11 environment cannot load. The package now falls back to the file-only `Agg` backend in that case, so `python -m mbproject all` runs unchanged on Colab.

### RQ2 weight sweep at small enrollment sizes

RQ1 shows the highest errors with few enrollment recordings. This extension asks whether a better-weighted face + voice fusion helps at k = 1, 2 and 4 (clean data only). For each enrollment seed and k it:

- builds templates from the same enrollment recordings as RQ1 and scores every development and test probe;
- min-max normalizes face and voice scores using development scores only, so both modalities span the same 0-1 range before they are weighted;
- fuses with face weights 0.0, 0.1, ..., 1.0 (0 = voice only, 1 = face only) and measures each weight on the test probes: HW3 EER, d-prime, and FRR at a development-fitted FAR = 0.1% threshold;
- picks one tuned weight from development data only: lowest exact EER, then lowest FRR at FAR = 0.1%, then highest d-prime, then closest to 0.5. The d-prime step only decides when several weights tie on errors, for example at zero development errors. The report's original tuning chose 0.5 in all ten seeds; with face alone at zero development errors, such a tie is the likely reason.

It reuses the cached embeddings (run `extract` first) and does not change the main pipeline or its results. Unlike the main pipeline, thresholds and weights are fitted separately for each k, because the question is the best fusion achievable at that size. For the same reason, the FRR column is not comparable with the main pipeline's FRR, whose thresholds are frozen at k = 16.

```bash
python -m mbproject rq2
python -m mbproject --config configs/study_30_seeds.yaml rq2
```

Outputs go to `<results_dir>/rq2_weights/`: `rq2_table.csv` (seed means and standard deviations for voice, face, the original unnormalized 50/50 fusion, normalized 50/50 fusion and the tuned fusion at each k), `weight_sweep.csv`, `systems.csv`, `selected_weights.csv` and the figure `rq2_weight_sweep.png`. Settings are in the `rq2` block of the config. The voice, face and unnormalized 50/50 fusion rows reproduce the main pipeline's clean EER and d-prime at the same k exactly, because they use identical scores and neither metric depends on the threshold. The normalized sweep's 0 and 1 weights can differ slightly from the voice-only and face-only rows (see `docs/EXPERIMENT_PROTOCOL.md`).

## Data and protocol

- 24 actors, 60 speech recordings each: **1,440 naturally paired audio/video recordings**. Identity and every stimulus field must match across modalities. Song is excluded to keep all actors equally represented. Duplicate video-only encodings are excluded.
- Each actor: **20 enrollment-pool, 20 development, 20 test recordings**. The two repetitions of each stimulus group stay together. A stimulus group is channel/emotion/intensity/statement, shared across actors. Group assignment is fixed using seed 73.
- Each test condition: **480 genuine and 11,040 impostor trials**. Every probe claims each of the 24 enrolled identities; the own-actor claim is genuine, other claims are impostors. Face and voice use the same probe and claimed actor on every fusion trial.
- All sampled frames of a recording are pooled into **one recording embedding**. Frames are never counted as independent enrollment samples or independent tests.
- Frozen pretrained identity encoders: SpeechBrain ECAPA-TDNN (192 dimensions) for voice and OpenCV SFace (128 dimensions) with YuNet detection/alignment for face. No RAVDESS fine-tuning or test-dependent feature selection. The main pipeline does not use expression action units as identity features; the separate facial-tracking study below tests them as an alternative face representation.
- A template is the L2-normalized average of the selected clean recording embeddings. Prefixes of a saved random order make enrollment subsets nested. The same ordering is used for both modalities when their enrollment counts match.
- Matching is cosine similarity, mapped by `(cosine + 1) / 2` into [0, 1] using a fixed theoretical transformation. A higher score means stronger evidence for a match. Fixed fusion uses 50% face and 50% voice. Tuned fusion selects a face weight from 0.1 through 0.9, using reference-enrollment development EER, then development FRR at FAR 1%, with ties preferring weights near 0.5.
- Every system's decision thresholds and fusion weight are fitted at **16 clean enrollment recordings**, separately for each enrollment seed, then frozen for all enrollment sizes and degradation levels. This keeps the enrollment comparison from also changing calibration.

See `docs/EXPERIMENT_PROTOCOL.md` for operational definitions and limits, and `docs/IMPLEMENTATION_CHECKLIST.md` for assignment traceability.

## Authentication demo

Find a held-out recording identifier, then verify its claimed actor:

```powershell
& .\.venv\Scripts\python.exe -c "import pandas as pd; f=pd.read_csv('data/manifest.csv'); print(f.loc[(f.split == 'test') & (f.actor == 1), ['recording_id','actor']].head().to_string(index=False))"

# Replace RECORDING_ID with a value printed above.
& .\.venv\Scripts\python.exe -m mbproject verify --recording RECORDING_ID --claim 1
& .\.venv\Scripts\python.exe -m mbproject verify --recording RECORDING_ID --claim 2

# A degraded held-out probe; thresholds remain fixed.
& .\.venv\Scripts\python.exe -m mbproject verify --recording RECORDING_ID --claim 1 --snr 0 --sigma 4

# Or use paired external audio and video with the enrolled RAVDESS roster.
& .\.venv\Scripts\python.exe -m mbproject verify --audio path\to\probe.wav --video path\to\probe.mp4 --claim 1
```

The output is an ACCEPT/REJECT decision plus face/voice/fused scores, development-selected threshold, and fusion weight. `scripts/demo.ps1 -TrueActor 1 -ClaimActor 1` selects a valid test recording automatically; change `-ClaimActor` to 2 for an impostor claim. Cached demo probes are restricted to the test partition. This is a research prototype, with no liveness or replay-attack defense.

The raw-input and cached-probe paths were checked on the same held-out recording. GPU calculations with single-item versus four-item batches can differ in the final digits (observed voice-score difference about 0.000011). This agreement check uses a numeric tolerance of 0.0001. Authentication accepts scores at or above the saved threshold, and both paths give the same decision. Details are saved in `results/full/demo_verification.json`.

## Outputs

All outputs are generated from actual recordings; no synthetic performance results are substituted.

| Location | Contents |
|---|---|
| `data/manifest.csv`, `dataset_audit.json` | Pairing, labels, media CRCs, stimulus groups, fixed splits, counts |
| `data/features/*.npz`, `*.json` | Cached embeddings, frame indices, measured SNR, sharpness, model/code signature |
| `models/manifest.json` | Pinned download URLs, model byte counts, SHA-256 hashes |
| `results/full/metrics.csv` | Full metrics for every seed, condition, and system |
| `results/full/summary.csv` | Mean/std/min/max over enrollment seeds |
| `results/full/trial_index.npz`, `trial_scores/` | Auditable genuine/impostor trial identities and every matching score |
| `results/full/enrollment/`, `policies.json` | Exact enrollment recording orders and frozen thresholds/weights |
| `results/full/development_trial_index.npz`, `development_scores/` | Reference development trials used to reproduce calibration policies |
| `results/full/per_user.csv` | Probe-user FAR/FRR at fixed development FAR 1% policies |
| `results/full/paired_actor_bootstrap.csv` | Paired FRR differences with probe-actor bootstrap intervals |
| `results/full/fusion_selection.csv` | Development-only selection of fusion for each seed |
| `results/full/figures/` | Enrollment plots, ROC, DET, score distributions, degradation heatmaps, user variation, manipulation checks |
| `results/full/figures/hw3_original/` | Figures produced by the original HW3 methods |
| `results/full/audit.json`, `run_complete.json` | Automated integrity checks and completion metadata |

`eer` is the interpolated empirical ROC EER; `hw3_eer` is the submitted HW3 500-threshold estimate. `dprime` comes directly from the original HW3 evaluator. Values are fractions unless a plot explicitly labels percent. Columns containing `at_dev_` use development thresholds. Columns prefixed `oracle_test_` describe the test ROC only and are **not operationally calibrated decisions**.

The test FAR resolution is 1/11,040, approximately 0.0000906. A result at a target FAR of 0.0001 is supported by only about one impostor error; it is not evidence that a deployed system achieves that risk. Trials share actors, templates, and recordings. Seeds do not multiply the number of independent test recordings. See the protocol before drawing significance or deployment claims.

## Reproducibility and sources

`configs/study.yaml` controls the study. Source and model provenance are recorded, original HW3 code is preserved byte-for-byte, noise is seeded per recording, and extracted features are cached. A changed extraction implementation/configuration fails on stale caches rather than silently mixing incompatible features. Use a separate `data_dir` for an intentionally different extraction configuration.

Primary implementation sources:

- RAVDESS: https://zenodo.org/records/1188976 and https://doi.org/10.1371/journal.pone.0196391
- OpenCV face pipeline: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
- OpenCV Zoo model repository: https://github.com/opencv/opencv_zoo
- SpeechBrain ECAPA model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

The supplied instructor feedback approves RAVDESS. Use is educational and local; respect the dataset's CC BY-NC-SA 4.0 terms and model terms. The dataset describes acted speech under controlled conditions. Results do not establish mobile-device field performance, population fairness, liveness, cross-session security, or robustness to other attacks.

## Facial-tracking study (action units and landmarks)

An additional study tests the RAVDESS facial-tracking release (OpenFace action units and landmarks) as an alternative feature representation of the face, alone and fused with voice and SFace, using the same splits, trials and metrics. After `all`, put `FacialTracking_Actors_01-24.zip` in `RAVDESS/` and run `python -m mbproject tracking`. Method and results: [docs/TRACKING_STUDY.md](docs/TRACKING_STUDY.md).

For cloning, dataset placement, installation on another machine, and team branches, see [TEAM_SETUP.md](TEAM_SETUP.md).

