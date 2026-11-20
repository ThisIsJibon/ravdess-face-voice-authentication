# Report requirements check

Checked against the report instructions in the supplied `Project_Description.pdf` and the instructor comments in this chat. The finished report has **five main pages and one separate reference page**. All six rendered pages were inspected, and every table value was rechecked against the saved results in `results/full` (face and voice) and `results/tracking` (action units).

| Requirement | Result and evidence |
|---|---|
| Three connected research questions | Pass. RQ1, RQ2 and RQ3 are listed on page 1. Their progression and results are explained on pages 1 and 3-5. |
| Questions avoid assuming improvement | Pass. Fusion is compared with both modalities; negative and tied results are reported. |
| Explain how findings inform subsequent questions | Pass. Zero clean SFace EER in RQ1 frames RQ2; because fusion gave no clean gain in RQ2, RQ3 tests it under degraded probes. |
| At least eight relevant research articles | Pass. Table 1 contains eight peer-reviewed articles on pages 1-2. |
| Full citation and direct source link for each article | Pass. Table 1 gives each article's authors, venue, year and DOI link. The full citation with title and pages is in the numbered Vancouver-style reference list. |
| Literature objective, methods and key results | Pass. Table 1 supplies all three fields for each article. Entries are brief paraphrases. |
| Experimental conditions and controlled variables | Pass. Page 2 explains the count sweeps, factorial degradation grid, nested enrollment subsets and fixed components. |
| Face and another biometric modality | Pass. Face and voice are naturally paired from the same RAVDESS recording identifiers. |
| Implemented fusion method | Pass. Page 2 names score-level fusion and describes the equal-weight and development-tuned rules. |
| Dataset citation, users and samples | Pass. Page 2 reports RAVDESS, 24 actors, 60 recordings each and 1,440 paired recordings. |
| Enrollment, development and test partitions | Pass. Page 2 gives 20 recordings per actor per partition, stimulus grouping and the absence of RAVDESS model training. |
| Identity alignment | Pass. Matching actor and stimulus identifiers are described; artificial cross-dataset identities were not used. |
| Prevent recording and frame leakage | Pass. Repetitions stay in the same partition; three frames produce one recording embedding. |
| Preprocessing and features | Pass. Page 2 specifies audio trimming, 16 kHz sampling, ECAPA-TDNN embeddings, sampled video frames, YuNet/SFace processing and normalization. |
| Genuine and impostor score construction | Pass. Page 2 explains all 24 claims per probe and 480 genuine/11,040 impostor trials per condition. |
| Facial action-unit extension | Pass. Page 3 describes the OpenFace 2.1.0 features, enrollment-only z-score and LDA, and AU + voice score fusion. Tables 2-3 report AU results, and page 5 reports the controls as exploratory. |
| Specific, quantified degradations | Pass. Clean/20/10/0 dB noise and blur sigma 0/1/2/4 are specified. Measured SNR and Laplacian variance validate the manipulations. |
| Enrollment varied independently and together | Pass. Page 2 states both designs; the RQ2 results discuss the asymmetric endpoints. |
| Python, libraries, hardware and seeds | Pass. Page 3 gives software versions, the computer used and all split/enrollment seeds. |
| HW3 evaluator use | Pass. Page 3 explains that the original HW3 file is unmodified and is called through a subclass that only speeds up threshold counting; the report distinguishes HW3 EER from interpolated ROC EER. |
| Results organized by RQ | Pass. RQ1 is on page 3, RQ2 on pages 3-4, and RQ3 on pages 4-5. |
| Required metrics and score distributions | Pass. Tables 2-3 report HW3 EER and d-prime; Figure 1 shows genuine/impostor distributions. Figure 2 adds ROC/DET. |
| FRR under FAR constraints | Pass. Table 3 reports actual test FRR/FAR at development FAR targets 0.01, 0.001 and 0.0001. Page 5 separately reports descriptive test-ROC FRRs at the same FAR ceilings. |
| Actual FAR distinguished from target FAR | Pass. The report explicitly notes that development targets do not guarantee test FAR. |
| Claims supported by measured results | Pass. Every numeric entry in the enrollment and degradation tables was recomputed from the saved experiment metrics, including the AU rows. Noise EER narrative values were also checked. |
| Uncertainty and scope | Pass. Page 5 qualifies correlated trials, seed reuse, the actor bootstrap, the 0.0001 FAR constraint, the small roster and the lack of phone-field/spoofing evaluation. |
| Ethics statement no longer than three paragraphs | Pass. Exactly three paragraphs on page 5. |
| At least three credible ethics citations | Pass. Five distinct sources support the ethics statement, including biometric security research, NIST and RAVDESS provenance/licensing. |
| Risks, misuse and fairness | Pass. The ethics section connects template exposure, replay, false errors, surveillance misuse and fairness limitations to this system. |
| Implemented versus proposed safeguards | Pass. Local processing and data handling are separated from proposed encryption, template protection, logging, retention controls, attack detection and fallback access. |
| Consistent references and persistent URLs | Pass. All 14 numbered references include a DOI or direct source URL. References begin on page 6. |
| Five-page limit excluding references | Pass. Main content, tables and figures occupy pages 1-5. |
| 11-point Calibri or Arial | Pass. Body and table text use 11-point Arial. |
| Single spacing, one-inch margins, justified body | Pass. Verified in document properties and paragraph formatting. |
| Title and section formatting | Pass. The title is centered, black, bold and 14 point. Section headings are left-aligned, black, bold and 14 point. |
| Numbered tables and figures | Pass. Tables 1-3 and Figures 1-2 are numbered and referenced in the text. Table 1 continues onto page 2 with its header row repeated. |
| Caption sizes and page numbers | Pass. Captions are 9 point; all six pages are numbered. |
| No unnecessary introduction or background | Pass. The report begins with the required research questions. |
| No em dashes | Pass. Neither em dashes nor en dashes occur in the report. |
| Proofreading and visual inspection | Pass. All six final rendered pages were reviewed for clipping, table wrapping, chart labels, borders, spacing and legibility. |
| Turnitin similarity threshold | Not locally measurable. No similarity percentage is claimed. The university's checker must verify the stated 20% threshold. |

This checklist records compliance with the supplied instructions. It does not assign a grade or replace instructor assessment.
