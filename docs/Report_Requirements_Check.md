# Report requirements check

Checked against the report instructions in the supplied `Project_Description.pdf` and the instructor comments in this chat. The finished report has **five main pages and one separate reference page**. All six rendered pages were inspected. All 31 automated content, formatting, citation and numeric checks passed.

| Requirement | Result and evidence |
|---|---|
| Three connected research questions | Pass. RQ1, RQ2 and RQ3 are listed on page 1. Their progression and results are explained on pages 1 and 3-5. |
| Questions avoid assuming improvement | Pass. Fusion is compared with both modalities; negative and tied results are reported. |
| Explain how findings inform subsequent questions | Pass. The clean face ceiling frames RQ2; the development-selected fusion is examined under degradation in RQ3. |
| At least eight relevant research articles | Pass. Table 1 contains eight peer-reviewed articles on pages 1-2. |
| Full citation and direct source link for each article | Pass. All eight have authors, full article title, publication details, year and DOI URL. Article titles, authors, years, pages and DOI records were checked against registered metadata. Conventional journal abbreviations follow the numbered Vancouver reference style. |
| Literature objective, methods and key results | Pass. Table 1 supplies all three fields for each article. Entries are brief paraphrases. |
| Experimental conditions and controlled variables | Pass. Page 2 explains the count sweeps, factorial degradation grid, nested enrollment subsets and fixed components. |
| Face and another biometric modality | Pass. Face and voice are naturally paired from the same RAVDESS recording identifiers. |
| Implemented fusion method | Pass. Page 3 describes equal-weight and development-tuned score fusion. |
| Dataset citation, users and samples | Pass. Page 2 reports RAVDESS, 24 actors, 60 recordings each and 1,440 paired recordings. |
| Enrollment, development and test partitions | Pass. Page 2 gives 20 recordings per actor per partition, stimulus grouping and the absence of RAVDESS model training. |
| Identity alignment | Pass. Matching actor and stimulus identifiers are described; artificial cross-dataset identities were not used. |
| Prevent recording and frame leakage | Pass. Repetitions stay in the same partition; three frames produce one recording embedding. |
| Preprocessing and features | Pass. Pages 2-3 specify audio trimming, 16 kHz sampling, ECAPA-TDNN embeddings, sampled video frames, YuNet/SFace processing and normalization. |
| Genuine and impostor score construction | Pass. Page 3 explains all 24 claims per probe and 480 genuine/11,040 impostor trials per condition. |
| Specific, quantified degradations | Pass. Clean/20/10/0 dB noise and blur sigma 0/1/2/4 are specified. Measured SNR and Laplacian variance validate the manipulations. |
| Enrollment varied independently and together | Pass. Page 2 states both designs; page 4 discusses the asymmetric endpoints. |
| Python, libraries, hardware and seeds | Pass. Page 3 gives software versions, GPU/CPU and all split/enrollment seeds. |
| HW3 evaluator use | Pass. Page 3 explains the unchanged HW3 EER and d-prime definitions; the report distinguishes these from interpolated ROC EER. |
| Results organized by RQ | Pass. RQ1 is on page 3, RQ2 on page 4, and RQ3 on pages 4-5. |
| Required metrics and score distributions | Pass. Tables 2-3 report HW3 EER and d-prime; Figure 1 shows genuine/impostor distributions. Figure 2 adds ROC/DET. |
| FRR under FAR constraints | Pass. Table 3 reports actual test FRR/FAR at development FAR targets 0.01, 0.001 and 0.0001. Page 5 separately reports descriptive test-ROC FRRs at the same FAR ceilings. |
| Actual FAR distinguished from target FAR | Pass. The report explicitly notes that development targets do not guarantee test FAR. |
| Claims supported by measured results | Pass. Every numeric entry in the enrollment and degradation tables was recomputed from the saved experiment metrics. Noise EER narrative values were also checked. |
| Uncertainty and scope | Pass. Page 5 qualifies correlated trials, seed reuse, the actor bootstrap, the 0.0001 FAR constraint, the small roster and the lack of phone-field/spoofing evaluation. |
| Ethics statement no longer than three paragraphs | Pass. Exactly three paragraphs on page 5. |
| At least three credible ethics citations | Pass. Five distinct sources support the ethics statement, including biometric security research, NIST and RAVDESS provenance/licensing. |
| Risks, misuse and fairness | Pass. The ethics section connects template exposure, replay, false errors, surveillance misuse and fairness limitations to this system. |
| Implemented versus proposed safeguards | Pass. Local processing and data handling are separated from proposed encryption, template protection, logging, retention controls, attack detection and fallback access. |
| Consistent references and persistent URLs | Pass. All 12 numbered references include a DOI or direct source URL. References begin on page 6. |
| Five-page limit excluding references | Pass. Main content, tables and figures occupy pages 1-5. |
| 11-point Calibri or Arial | Pass. Body and table text use 11-point Arial. |
| Single spacing, one-inch margins, justified body | Pass. Verified in document properties and paragraph formatting. |
| Title and section formatting | Pass. The title is centered, black, bold and 14 point. Section headings are left-aligned, black, bold and 14 point. |
| Numbered tables and figures | Pass. Tables 1-3 and Figures 1-2 are numbered and referenced in the text. The continuation of Table 1 is labeled. |
| Caption sizes and page numbers | Pass. Captions are 9 point; all six pages are numbered. |
| No unnecessary introduction or background | Pass. The report begins with the required research questions. |
| No em dashes | Pass. Neither em dashes nor en dashes occur in the report. |
| Proofreading and visual inspection | Pass. All six final rendered pages were reviewed for clipping, table wrapping, chart labels, borders, spacing and legibility. |
| Turnitin similarity threshold | Not locally measurable. No similarity percentage is claimed. The university's checker must verify the stated 20% threshold. |

This checklist records compliance with the supplied instructions. It does not assign a grade or replace instructor assessment.
