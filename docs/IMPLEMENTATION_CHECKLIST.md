# Assignment implementation traceability

Source: supplied `Project_Description.pdf` (last updated October 14, 2026) and instructor feedback provided in the chat. This is an implementation checklist, not a claim of a guaranteed grade. No separate point-by-point grading rubric was supplied.

| Assignment / feedback item | Implementation evidence |
|---|---|
| Investigate at least three connected RQs | `configs/study.yaml`; clean scarcity experiments -> fusion comparisons on the same trials -> noise/blur robustness of the same systems |
| Multimodal, including face | Naturally aligned RAVDESS SFace face + ECAPA voice recording embeddings |
| Implement a form of fusion | Equal-weight and development-tuned weighted score-level fusion |
| Use HW3 Python performance script | Unmodified `src/mbproject/vendor/hw3_evaluator.py`; provenance hash; inherited d-prime and EER invoked for all conditions; original plots invoked |
| EER, d-prime, score-distribution plots | `metrics.csv`, `summary.csv`, `figures/`, and `figures/hw3_original/` |
| ROC / DET and error tradeoffs | Empirical ROC/normal-deviate DET; frozen development FAR policies at 0.01/0.001/0.0001; actual test FAR and FRR; oracle test ROC quantities separately named |
| Enrollment varies in each modality, individually and combined | Equal-count diagonal plus face-only and voice-only count sweeps with the other fixed at 16 |
| Isolate enrollment count | Fixed split/probes/models/features/calibration/thresholds; nested random subsets over ten seeds |
| Avoid assumed fusion improvement | Retain all systems and full comparisons; development-only choice; no result-dependent filtering |
| Specify and quantify degradation | Clean/20/10/0 dB AWGN; sigma 0/1/2/4 on aligned 112px crops; measured SNR and Laplacian variance recorded |
| Explain identity alignment | Actor + all stimulus fields matched; no artificial cross-dataset identities |
| Explicit samples and splits | 24 x 60 speech recordings; 20/20/20 enrollment/dev/test per actor; group-disjoint repetitions |
| Genuine and impostor score generation | Every test probe against every claim identity; 480 genuine / 11,040 impostor per condition |
| Reproducibility | Pinned packages/model revisions, local environment, saved manifest/model hashes, seeds, enrollment IDs, policies, and all trial scores |
| Evidence integrity | Tests, cache signatures, strict failure on acquisition errors, `audit.json` |
| User variability | Per-actor operating errors; descriptive plot and paired probe-actor bootstrap differences |

The supplied instructor comment says RAVDESS is fine to use. The implementation treats that as approval for its paired audio/video modalities and preserves natural identities.

Excluded at the user's request: final technical report, eight-article annotated bibliography, final ethics statement with citations, formatted references, and presentation. The implementation records the methods/results needed for those later tasks. The report also requires explaining the actual RQ progression after inspecting results; these narrative judgments are not replaced by this code checklist.
