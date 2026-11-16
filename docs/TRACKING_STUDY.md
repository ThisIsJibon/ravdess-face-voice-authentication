# Facial-tracking study: action units and landmarks for authentication

**Question.** Can the facial *movement* and *geometry* measurements in the RAVDESS facial-tracking release (OpenFace action units and landmarks) authenticate a person, alone or fused with our face (SFace) and voice (ECAPA) systems? This follows Dr. Neal's suggestion to look at faces while people speak.

**Short answer.** Yes, but mostly because of face *shape*, not movement:
- AU-only features with LDA reach 3.77% EER at k = 4. That is close to voice (3.34%) and far behind face appearance (0%).
- Adding landmark and shape features gives 0.03% EER, but shape features are a form of face geometry, not behavior.
- Fusing AU with voice cuts voice error by about 4x (3.34% to 0.85% at k = 4). Under 0 dB noise it cuts error from 13.86% to 2.07%, because the face video is unaffected by audio noise.
- LDA learns from the same 24 actors (20 enrollment recordings each), while each template uses only k recordings. So the 3.77% answers a question about these 24 known actors. When LDA is fitted without 12 of the actors and tested on them, AU + LDA gives 17.5% EER and all-tracking + LDA gives 4.5%.
- On movement alone: AU frame-to-frame change (velocity) gives 12.40%, all movement features 3.37%, and all shape features 0.43% (all with LDA). Shape works better on these recordings; we did not measure how much of the 0.03% combined result comes from shape.

Unless noted, numbers are test-split means over 10 enrollment seeds, HW3 EER, on the same trials as the main study (the controls in section 6 use a different setup, explained there). Full tables: `results/tracking/REPORT.md` (generated).

---

## 1. Dataset

| Item | Value |
| --- | --- |
| Source | `FacialTracking_Actors_01-24.zip`, https://zenodo.org/records/3255102 (417 MB, CC BY-NC-SA 4.0) |
| Tool that produced it | OpenFace 2.1.0, run by the RAVDESS authors on the original videos |
| Files | 2,452 CSVs: 1,440 speech + 1,012 song. We use the 1,440 audio-visual speech files (`01-01-*.csv`) |
| One CSV | One row per video frame (about 110 frames per clip), 712 columns |
| Columns we use | `confidence`, 17 AU intensities `AU01_r … AU45_r` (0 to 5), 68 2D landmarks `x_0..67`, `y_0..67`, 34 shape parameters `p_0..p_33` |

**Join.** The tracking file `01-01-03-02-01-01-12.csv` describes the same clip as our recording `01-03-02-01-01-12`; the first field (`01` = audio-video) is dropped. All 1,440 manifest recordings matched exactly, and the code stops if any is missing.

**Frame quality.** Frames with tracking confidence below 0.8 are dropped. That removed only 6 of 159,222 frames. Every recording kept at least 88 frames.

## 2. From frames to one feature vector per recording (`src/mbproject/tracking.py`)

Each recording is summarized into six blocks. Each block uses only that recording's own frames, so no split or label information can leak in.

| Block | How it is computed | Size | Measures |
| --- | --- | --- | --- |
| AU level | mean of each AU over frames | 17 | typical expression strength |
| AU motion | standard deviation of each AU | 17 | how much each AU varies |
| AU velocity | mean of absolute change between consecutive frames | 17 | how fast the face moves |
| Landmark shape | mean of normalized landmarks | 136 | face geometry |
| Landmark motion | standard deviation of normalized landmarks | 136 | how the face shape moves |
| PDM shape | mean of OpenFace shape parameters `p_0..p_33` | 34 | compact face geometry |

**Landmark normalization**, per frame, with a fixed rule and no fitted reference: centre on the midpoint of the outer eye corners (points 36 and 45), rotate so the eye line is horizontal, and divide by the distance between the eyes. This removes head position, head tilt and distance from camera.

The blocks are combined into 11 feature sets, defined in `configs/study.yaml` under `tracking.feature_sets`. Examples: `au_all` = AU level + motion + velocity (51 dimensions); `movement_all` = AU motion + AU velocity + landmark motion; `shape_all` = landmark shape + PDM shape; `tracking_all` = everything (357 dimensions).

## 3. Same protocol as the face/voice study (`src/mbproject/tracking_eval.py`)

Nothing about the split or trials is recomputed independently. The code reads the main study's own files and stops if anything differs.

| Element | Source |
| --- | --- |
| 20 enroll / 20 dev / 20 test recordings per actor, seed 73 | `data/manifest.csv` |
| Enrollment orders for the 10 seeds | `enrollment_order()`, checked against `results/full/enrollment/seed_*.json` |
| Test trials: 480 genuine + 11,040 impostor, same probe order | checked against `results/full/trial_index.npz` |
| Template | L2-normalized mean of the first k enrollment recordings (same rule) |
| Score | cosine similarity mapped by (cos + 1) / 2 (same rule) |
| Metrics | unchanged HW3 evaluator (EER, d-prime) plus interpolated EER and FAR/FRR |
| Thresholds | fitted on dev scores at k = 16, then frozen for every k (same rule) |

## 4. Learning step: z-score and LDA

Raw features mix two things: who the person is, and what they are saying or feeling. Linear Discriminant Analysis (LDA) learns a projection that spreads different people apart and pulls each person's recordings together, down-weighting directions that mostly change with sentence or emotion.

1. **Z-score** each feature with the mean and standard deviation of the **enrollment** split.
2. **LDA** is fitted on the **enrollment** split only (480 recordings, 24 actors).
3. **Settings** (3 dimension counts × 4 shrinkage levels) are chosen per feature set by the lowest **development** EER at k = 4, averaged over the 10 seeds. The full grid is saved in `dev_selection_grid.csv`.
4. The **test** split is used once, to report results.

"Raw" results skip step 2 (z-score only). Both are reported.

## 5. Experiments

**A: which features identify a person** (k = 4, test EER)

| Feature set | Raw | LDA |
| --- | --- | --- |
| AU level (17) | 21.36% | 9.94% |
| AU motion (17) | 26.38% | 19.26% |
| AU velocity (17) | 20.23% | 12.40% |
| AU all (51) | 19.76% | **3.77%** |
| Landmark motion (136) | 33.12% | 7.42% |
| All movement (170) | 27.98% | 3.37% |
| PDM shape (34) | 2.93% | 0.33% |
| Landmark shape (136) | 10.55% | 0.82% |
| All shape (170) | 7.94% | 0.43% |
| All tracking (357) | 9.92% | **0.03%** |
| *Reference: voice / face* | *3.34% / 0.00%* | |

Shape features identify people far better than movement features. Without any learning, PDM shape alone (2.93%) already beats every movement set. So most of the identity in the tracking data is face geometry, which is a cruder version of what SFace already captures from appearance.

**C: enrollment size** (test EER)

| System | k=1 | k=4 | k=16 |
| --- | --- | --- | --- |
| Voice | 7.69% | 3.34% | 2.22% |
| AU + LDA | 7.58% | 3.77% | 2.52% |
| AU + LDA fused with voice | 3.01% | 0.85% | 0.39% |
| All tracking + LDA | 0.14% | 0.03% | 0.03% |
| Face appearance | 0.00% | 0.00% | 0.00% |

**B: fusion** (k = 4, test EER; probe audio noise only)

| System | Clean | 20 dB | 10 dB | 0 dB |
| --- | --- | --- | --- | --- |
| Voice | 3.34% | 4.43% | 6.43% | 13.86% |
| AU + voice (average of scores) | 0.85% | 1.04% | 1.29% | 2.07% |
| All tracking + voice (average of scores) | 0.01% | 0.00% | 0.00% | 0.09% |

Fusion uses the team's fixed rule: the plain average of the systems' scores, with thresholds fitted on dev data at k = 16. In practical terms, at the dev threshold for 1% false accepts, voice alone wrongly rejects 6.77% of genuine attempts and AU + voice rejects 0.88%. More fusion combinations (AU + face, AU + face + voice and others) are in `REPORT.md`.

## 6. Controls

The controls use the interpolated ROC EER (not the HW3 EER) and the first enrollment seed only. The excluded-actor control repeats over 20 random 12/12 actor splits and reports the mean.

| Control | Result | Meaning |
| --- | --- | --- |
| Shuffled probe identities | 50.0% (AU), 51.3% (all tracking) | Scores fell to chance after scrambling identities. This is a sanity check; it cannot rule out every kind of leakage |
| Evaluation on actors excluded from LDA fitting: z-score and LDA fitted on 12 actors, tested on the other 12 (20 splits, k = 4) | AU: 21.5% raw → 17.5% LDA. All tracking: 12.6% → 4.5% | Part of the LDA gain comes from learning these 24 actors. Not fully actor-independent: the LDA settings and the best feature set were picked earlier on dev data from all 24 actors |
| Test pairs with the same vs different sentence/emotion (AU + LDA) | 2.9% vs 10.4% | What is said still affects AU features even after LDA |

## 7. Limitations

- **One session and 24 actors.** Shape features look excellent partly because each actor was filmed once, with the same camera and pose.
- **LDA learns from the enrolled users.** SFace and ECAPA were trained on other people. The excluded-actor control shows the size of this gap, but its settings were chosen on all 24 actors, so it is not a fully independent new-user test.
- **No blur or degradation test for tracking.** The tracking CSVs were computed from clean video. Testing blur would need OpenFace re-run on blurred video.
- **Shape is not behavior.** Only the movement sets (AU motion, AU velocity, landmark motion) measure behavior; the strong results come from geometry.

## 8. How to reproduce

After the main pipeline (`python -m mbproject all`) has finished:

1. Put `FacialTracking_Actors_01-24.zip` (or its extracted folder) in `RAVDESS/`.
2. Run `.venv/bin/python -m mbproject tracking` (on Windows: `.\.venv\Scripts\python.exe -m mbproject tracking`). It takes about 1 to 2 minutes. Use `--refresh` to re-extract features.
3. Outputs go to `data/tracking/` (features and frame quality) and `results/tracking/`:
   - `REPORT.md` (all tables)
   - `summary.csv`, `metrics.csv` (Experiments A and C)
   - `fusion_summary.csv`, `fusion_metrics.csv` (Experiment B)
   - `controls.csv`, `dev_selection_grid.csv`, `dev_selected.csv`
   - `figures/`

Tests: `python -m pytest -q tests/test_tracking.py`.
