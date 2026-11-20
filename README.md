# Mobile Biometrics Final Project

Python implementation of RAVDESS face + voice **identity verification** for the three approved research questions. This repository contains the experiments and a working claim-verification demo. It does not contain the final report or presentation.

## Run on this computer

Open PowerShell in `C:\Users\jibon\AcademicWorld\Coding\MBFinalProject`.

```powershell
# Run the full pipeline. Existing valid media/features are reused.
& .\.venv\Scripts\python.exe -u -m mbproject all

# Run implementation tests and the completed-study integrity audit.
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m mbproject audit

# Re-evaluate cached embeddings or regenerate figures without inference.
& .\.venv\Scripts\python.exe -m mbproject evaluate
& .\.venv\Scripts\python.exe -m mbproject plots
```

On another machine, install **Python 3.11**, copy the original `RAVDESS` speech archives into this folder, then run `scripts/setup.ps1`. Use `scripts/setup.ps1 -Cpu` on a computer without a suitable NVIDIA GPU. The setup creates an isolated `.venv`; it does not replace global Python packages. The full original song dataset can remain in `RAVDESS` but is not used. Model downloads are automatic and pinned to specific upstream revisions. Internet is needed only for setup and first model download. The experiments run locally.

## What is implemented

| Research question | Experiments |
|---|---|
| RQ1: Enrollment data scarcity | 1, 2, 4, 8, 12, and 16 recordings per actor. Vary both modalities together; additionally vary one while fixing the other at 16. |
| RQ2: Multimodal fusion | Voice, face, equal-weight score fusion, and development-tuned weighted score fusion on the same genuine/impostor trials. |
| RQ3: Data degradation | Probe audio: clean, 20, 10, 0 dB additive white Gaussian noise. Probe face: Gaussian blur sigma 0, 1, 2, 4 pixels on aligned 112 x 112 faces. Full 4 x 4 grid at enrollment sizes 1, 4, 16. |

There are **61 conditions x 10 enrollment seeds x 4 systems = 2,440 evaluations**. Clean enrollment conditions serve both RQ1 and RQ2. The three questions progress from sample availability to fusion comparisons to robustness of the same systems. Each seed selects a preferred fusion using **development data only**; `fusion_selection.csv` records that selection. No improvement is assumed, and both fusion methods are retained even when a unimodal system performs better.

### Extended RQ1 enrollment evaluation

An additional 30-seed evaluation is configured in `configs/study_30_seeds.yaml`. It includes the original ten enrollment seeds and 20 additional unique seeds, using the same dataset partitions, models, enrollment sizes, conditions, and evaluation systems.

The extended run contains **61 conditions x 30 enrollment seeds x 4 systems = 7,320 evaluations**.

Run the extended evaluation and regenerate its figures with:

```bash
python -m mbproject --config configs/study_30_seeds.yaml evaluate
python -m mbproject --config configs/study_30_seeds.yaml plots
```

Its outputs are stored separately under `results/30_seeds/`. The original `configs/study.yaml` and `results/full/` remain unchanged.

The additional seeds examine variation in enrollment selection. They do not increase the number of independent test recordings: each condition still contains 480 genuine and 11,040 impostor comparisons. The original ten seeds are included in the 30-seed run, so comparisons between the two runs are overlapping rather than independent replications.

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

