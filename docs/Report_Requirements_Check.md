# Report requirements check

Reviewed November 22, 2026 against **Project Description v2.pdf** (updated November 15) and **CAP6101_Report_Rubric.pdf** supplied by the user. This check applies to the revised root file `Mobile_Biometrics_Report.docx` and supersedes the earlier report checklist.

The rendered report has **five content pages and one separate reference page**. All six page images were visually inspected. **173 numerical checks passed**, recomputing table entries and principal narrative results from saved experiment outputs. This is a requirements review, not an instructor-assigned grade or a Turnitin assessment.

| Requirement | Evidence in the revised report |
|---|---|
| Three explicit experimental RQs | Page 1 lists RQ1–RQ3, enrollment sizes, fusion comparisons, degradation levels and measurable outcomes. |
| RQ1 finding motivates RQ2 | Pages 1 and 3 explain improving voice EER versus zero clean face EER. RQ2 compares fusion within matched k; enrollment changes are not counted as fusion effects. |
| RQ2 finding motivates RQ3 | Pages 1 and 4 explain the lack of clean EER improvement over face. RQ3 tests degraded probes with the original fusion policy. |
| Original and extended experiments distinguished | Pages 1, 3 and 4 separate original k16-tuned 50/50 fusion from normalized per-k 90/10 tuning. The latter is clean-only and is not claimed to be used in RQ3. |
| Eight annotated research articles | Table 1 on pages 1–2 includes eight articles, full citations, direct DOI links, objectives, methods and relevant findings. |
| Conditions, changed variables and controls for every RQ | Page 3 specifies nested enrollment, independent modality sweeps with the other at 16, matched-k scoring comparisons and noise × blur grids with clean enrollment. |
| Dataset and identity alignment | Page 2 cites RAVDESS, gives 24 actors, 60 clips each, 20/20/20 partitions, natural modality alignment, grouped repetitions and shared identities across partitions. |
| Preprocessing and features | Page 2 describes audio trimming, sampling, log-mel inputs, frozen ECAPA embeddings, video frame sampling, YuNet alignment, frozen SFace and recording-level pooling. |
| Matching and trial construction | Page 3 describes normalized mean templates, cosine mapping and 480 genuine/11,040 impostor trials per condition. |
| Multimodal score fusion | Page 3 gives the weighted sum, candidate weights, development-only selection and thresholds. Table 3 separates normalization from weight tuning. |
| Action units and landmarks | Page 4 describes OpenFace, 51 AU features, 357 tracking features, enrollment-pool z-scoring/LDA and AU + voice averaging. It discloses that representation training uses all 20 pool clips while k varies the template. |
| Quantified degradation | Page 4 defines Gaussian noise, measured SNR, post-alignment Gaussian blur in pixels, Laplacian variance and sigma 6/8 extension. |
| HW3 script and metrics | Page 3 names the unmodified HW3 evaluator and tested accelerated subclass, defines EER/d-prime/FRR/FAR and distinguishes interpolated EER. |
| Implementation and seeds | Pages 2–3 give Python 3.11.18, core library versions, Apple Silicon resources, split seed and all 30 enrollment seeds. RQ1, RQ2, RQ3 and AU use the same 30-seed protocol; controls use their explicitly stated first seed/actor splits. |
| RQ1 numbered results and direct answer | Table 2 gives all six clean k values for voice, face and fusion in HW3 EER (%) and d-prime. Analysis quantifies diminishing gains and avoids a universal reliable k. |
| RQ2 numbered results and direct answer | Tables 2–3 compare matched k. Page 4 reports tied/worse clean EER versus face, the 90 tuning choices (30 seeds × 3 k values) and AU fusion without claiming superiority over SFace. |
| RQ3 numbered results and direct answer | Table 4 and page 5 report original/strong blur, HW3 EER, d-prime and test FRR/FAR at all three development targets (1%, 0.1%, 0.01%). |
| Actual FAR versus target | Table 4 labels actual test FAR. Page 5 explains differing achieved FARs, EER versus frozen-threshold usability and severe joint-degradation failure. |
| Distributions and ROC | Figure 1 shows labeled genuine/impostor score distributions and modality/fusion ROC curves for representative seed 73 and a specified condition. |
| Uncertainty and scope | Page 5 covers the actor bootstrap, enrollment-dependent degraded EER, AU content/shuffle/transfer controls, correlated trials, small roster, field/latency/spoofing limits and future calibration experiments. |
| Ethics with credible citations | Exactly three paragraphs on page 5 cite biometric security/error research, NIST and RAVDESS. Risks include template exposure, exclusion, surveillance and unmeasured demographic fairness. |
| Implemented versus proposed safeguards | Page 5 separates local processing and source-control exclusions from proposed encryption, access control/logging, protected templates, deletion, replay detection and fallback access. |
| Page limit and layout | Five content pages including visuals; references start separately on page 6; page numbers 1–6. |
| Required typography | Arial 11 body/table text, single spacing, one-inch margins, justified prose, centered bold 14-point title, left-aligned bold 14-point headings and 9-point captions. |
| Visual numbering and prose references | Tables 1–4 and Figure 1 are numbered, captioned and referenced. Table 1 continues on page 2 with a header. |
| References and URLs | Fourteen complete numbered references have DOI/direct source URLs. Table 1 citations match their full reference entries. |
| Proofreading and visual QA | All six final page images were inspected for clipping, overlap, split rows, legibility and numbering. No tracked changes remain. |
| Similarity score and grade | Cannot be established locally; no Turnitin percentage or guaranteed instructor grade is claimed. |

Local verification artifacts, excluded from Git:

- `results/verification/submission_review_nov22/submission_verification.json`: current report hash, page count and structural checks.
- `results/verification/submission_review_nov22/submission_numeric_checks.csv`: all 173 numerical checks.
- `results/verification/submission_review_nov22/report_render/`: current internal QA PDF and page PNGs.
- `results/verification/submission_review_nov22/before_Mobile_Biometrics_Report.docx`: backup of the user's latest report before this review.
- `results/verification/submission_review_nov22/build_reviewed_report.py` and `verify_reviewed_report.py`: revision and verification scripts.

Scientific claims were checked against `results/30_seeds`, `results/30_seeds/rq2_weights`, `results/strong_blur_30_seeds` and `results/tracking_30_seeds`. Rendering used the packaged document renderer with an isolated temporary runtime and LibreOffice 26.8.3.1; native Word automation was unavailable. QA images/PDF are internal artifacts; the deliverable is the root Word report.

The 30-seed revision replaces all numerical result tables and related discussion using fresh extension reruns. It explicitly distinguishes HW3 EER (7.31% to 2.21% for clean voice) from interpolated EER (7.32% to 2.21%). All 69 tests and both 25-check run audits passed. The main 2,440 and strong-blur 3,400 legacy rows exactly match their corresponding rows in the larger runs; the 270 RQ2 baseline rows reproduce the main study. Evidence is in `results/verification/report_30_seeds/run_consistency.json`.

The November 22 review checked the latest user-edited report and deck. It corrected a clean voice result labeled as 0 dB, AU control EERs described as accuracy, the explanation of how the 90/10 weight was selected, and statements that overstated fusion's benefit. The report retains the independent-modality enrollment comparison and AU operating-point results. Repeated analysis was shortened to keep five content pages. The test suite and both 25-check audits passed again; the current logs are in `results/verification/submission_review_nov22/`.

The reviewed presentation is `output/Group 2 Presentation Submission.pptx`. It has 16 slides, with a new methodology diagram on slide 4, after the dataset slide. Four clicks reveal the split, feature extraction, template matching, and fusion/evaluation stages. The diagram uses editable shapes and native fade animations. The RQ1 and RQ3 replacement charts are editable and contain embedded workbooks populated from the verified 30-seed results. All 16 slide previews were inspected, and package, slide geometry, and chart/workbook checks passed. An Office-format import retained four click-triggered fades. Native PowerPoint Slide Show playback could not be tested here. The original root presentation is retained, and both source files have private backups in the review folder.
