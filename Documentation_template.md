# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** EntityResolvers  
**Team Members:** Antigravity Team  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary
We developed an end-to-end business entity resolution system designed to resolve 1,732,544 Source 1 entities across France, the United States, and India against a combined target pool of 9,969,589 candidate entities (Source 2 and Source 3). Our architecture couples a country-partitioned multi-tier inverted index blocking engine with a calibrated gradient-boosted pairwise ranker (LightGBM on 27 engineered name, address, phonetic, embedding, and source-provenance features) and a locked multi-gate decision layer. The repaired implementation supplies real MiniLM cosine features at inference and loads explicit decision rules from JSON. On the reconstructed 5,000-entity full-pool diagnostic, the locked-rule macro F0.5 is 0.608125, versus 0.606961 with the previous proxy inputs. The correction is necessary but has only a small measured benefit; the remaining matching errors require further validation before final submission. The recorded decision rules include:
1. A **Numeric Street-Number Consistency Gate** ($|\text{street}_1 - \text{street}_2| \le 1$ with leading-zero normalization) that systematically eliminates address-driven twin false merges in dense urban grids.
2. A **Script-Aware Dual Name Floor** that enforces strict lexical overlap for Latin-script text while preserving cross-script Indic transliterations via Unicode block isolation (`r'[\u0900-\u0D7F]'`).
3. A **Global 1-to-1 Bipartite Greedy Matcher** augmented with domain-specific match budget caps (max 5 S2, max 6 S3, max 11 total matches per entity).

---

## 2. Methodology

### 2.1 Problem Analysis
The recovered methodology records the following historical exploratory observations. Their original full-pool diagnostic script and sample were not found; use the corrected diagnostic section below for reproducible measurements from this repair:
- **Scale-Driven Multiple Comparisons False Merges**: In dense candidate pools (~10M targets vs. 1.28M in localized validation), address overlap features alone generated high calibrated match probabilities ($p > 0.85$) for co-located but distinct businesses sharing common municipal words (e.g. *École primaire Jean* vs. *Sci Ox Peinture*, *Transition Trames* vs. *Transition Et Fils*).
- **Adjacent Street-Number Twins**: In synthetic and real data, 24.6% of singleton false merges were businesses located at adjacent street numbers (e.g. `#1229` vs. `#1231`, `#515` vs. `#519`). Because street numbers are small substrings within identical street/city strings, standard token overlap scored high without numeric verification.
- **Cross-Script Transliteration Disparity**: In India, legitimate matches exist where S1 is written in Devanagari/regional script and S2/S3 in Latin script (e.g. *पतंजलि* vs. *Patanjali*), exhibiting low lexical token overlap but moderate phonetic alignment. An unconditioned lexical floor would destroy these true positives.
- **Target Uniqueness Constraint**: Each target record (S2 or S3) can match at most one unique real-world business entity, requiring global bipartite matching rather than independent per-entity thresholding.

### 2.2 Solution Strategy
- **Approach Type:** Hybrid Multi-Index Blocking + Calibrated Pairwise Ranker + Domain-Constrained Multi-Gate Decision Layer.
- **Core Innovation:** A 5-stage verification architecture that couples probabilistic ranking with deterministic geographic and linguistic invariants, preventing false-merge explosion under 8x scale expansion while protecting singletons and cross-script matches.

```
+-----------------------------------------------------------------------------------+
|                            STAGE 6 PRODUCTION ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
|  [Test Source 1] (1.73M Entities: US, IN, FR)                                      |
|          |                                                                        |
|          v                                                                        |
|  [Country Partitioning & Normalization]                                           |
|    - Unicode NFKD (Latin diacritics stripped, Indic script preserved)             |
|    - Street number extraction, postal code parsing, legal suffix canonicalization |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|  CANDIDATE GENERATION (CountryBlockingEngine, top_k=60)                           |
|    - 5 inverted indices: Name IDF, Soundex, 3-char prefix, Address IDF,      |
|      Street number index (postal code is a model feature)                                      |
|    - Evaluates ~103.8M candidate pairs across all countries                       |
|    - Directly emitted to `output/candidate_pairs.tsv`                             |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|  PAIRWISE RANKING & CALIBRATION                                                   |
|    - 27 string, embedding, phonetic, address, suffix, and provenance features                |
|    - LightGBM Gradient Boosted Decision Trees                                     |
|    - Isotonic Regression Probability Calibration: p = Cal(f(x))                   |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|  MULTI-GATE DECISION LAYER (Locked Parameters)                                    |
|    1. Operating Probability Threshold: tau = 0.60                                 |
|    2. Confidence-Gap Backstop: abstain if top_p < 0.85 & (top_p - sec_p) < 0.05    |
|    3. Street Number Gate: require |int(sn1) - int(sn2)| <= 1 if both present      |
|    4. Script-Aware Name Floor:                                                    |
|         - Indic: name_jaro_winkler >= 0.30                                        |
|         - Latin: (tok_ov >= 0.50 & lev >= 0.50) OR (lev >= 0.75)                  |
|    5. Global 1-to-1 Bipartite Greedy Conflict Resolution (prob descending)        |
|    6. Domain Match Budget Caps: max 5 S2, max 6 S3, max 11 total per S1 entity     |
|          |                                                                        |
|          v                                                                        |
|  [output/matching_results.tsv]                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 3. Candidate Generation (Blocking)
To reduce the comparison space from $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairwise comparisons to a manageable set, we designed the `CountryBlockingEngine`:

- **Blocking keys used:**
  1. **Name Token IDF Inverted Index:** Rare word tokens weighted by inverse document frequency ($w = \text{IDF} \times 2.5$). Frequency posting cap $\le 15,000$.
  2. **Phonetic Encoding:** Soundex phonetic keys for the first token of the entity name ($w = 1.5$, posting cap $\le 3,000$).
  3. **Character 3-Gram Prefix:** First 3 characters of normalized business name ($w = 1.5$, posting cap $\le 3,000$).
  4. **Distinctive Address Tokens:** Non-stopword address tokens of length $\ge 4$ weighted by address IDF ($w = \text{IDF} \times 2.0$, posting cap $\le 3,000$).
  5. **Exact Street Number:** Numeric street number matches ($w = 1.0$, posting cap $\le 1,500$).
  6. **Partitioning:** Strict country-level partitioning (France, US, India).

- **Candidate pairs generated:**
  - France: 15,477,499 candidate pairs (259,452 entities $\times$ 60 candidates)
  - United States: 39,662,227 candidate pairs (663,106 entities $\times$ 60 candidates)
  - India: 48,307,173 candidate pairs (809,986 entities $\times$ 60 candidates)
  - **Total Candidates Evaluated:** 103,446,899 candidate pairs.

- **How true matches were preserved (Recall Analysis):**
  - On the validation set (1.28M target scale), the blocking engine achieved **95.92% candidate recall**.
  - When evaluated against the full **10.3M target pool scale**, candidate recall was measured at **85.07%**. Because the candidate pool expanded by ~8x while the top-$k$ capacity was fixed at 60, high-frequency generic tokens competed for index slots, limiting recall on that historical sample. This does not establish a measured test-set recall ceiling, because test labels are unavailable.

---

## 4. Matching Model

### 4.1 Features Used (27 total)
1. **Name Similarity Features (11):**
   - `name_jaro_winkler`: Jaro-Winkler string similarity.
   - `name_levenshtein_ratio`: Character Levenshtein normalized similarity.
   - `name_token_jaccard`: Word token Jaccard similarity.
   - `name_token_overlap_ratio`: Token intersection over minimum token count.
   - `name_char_3gram_jaccard`: Character 3-gram Jaccard coefficient.
   - `name_first_token_soundex_match`: Binary match on initial token Soundex.
   - `name_first_token_metaphone_match`: Binary match on initial token Metaphone.
   - `name_acronym_match`: Binary match on generated word-initial acronyms.
   - `name_exact_match`: Binary indicator for identical normalized names.
   - `name_len_diff`: Absolute length difference between normalized names.
   - `name_embedding_cosine`: Dot product of L2-normalized MiniLM embeddings of normalized names.

2. **Legal Suffix Canonicalization Features (3):**
   - `suffix_normalized_equality`: Equality of canonical corporate forms (e.g. *Inc* $\leftrightarrow$ *Incorporated*, *SARL* $\leftrightarrow$ *S.A.R.L.*, *Pvt Ltd* $\leftrightarrow$ *Private Limited*).
   - `suffix_presence_s1`, `suffix_presence_target`: Binary presence flags.

3. **Address Similarity Features (7):**
   - `addr_token_jaccard`: Address word token Jaccard similarity.
   - `addr_token_overlap_ratio`: Address token intersection over minimum count.
   - `addr_levenshtein_ratio`: Normalized character Levenshtein ratio.
   - `addr_postal_code_match`: Binary match of extracted PIN/ZIP codes.
   - `addr_street_num_match`: Numeric equality of extracted street numbers.
   - `addr_is_empty_target`: Missing address indicator.
   - `addr_embedding_cosine`: Dot product of L2-normalized MiniLM embeddings of normalized addresses.

4. **Meta & Interaction Features (6):**
   - `country_exact_match`: Binary match of country field.
   - `target_is_s2`, `target_is_s3`: Source provenance one-hot indicators.
   - `blocking_score`, `blocking_rank`: Rank and rank-derived score from blocking phase (`max(1, 60 - rank)`).
   - `name_addr_sim_product`: Jaro–Winkler name similarity multiplied by address token overlap.

### 4.2 Model Type & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree Classifier (`LGBMClassifier`).
  - Objective: Binary Logloss.
  - Hyperparameters: `n_estimators=1000` with early stopping (saved model: 629 iterations), `learning_rate=0.05`, `num_leaves=63`, `subsample=0.8`, `colsample_bytree=0.8`.
  - Class Imbalance Handling: Hard negatives from blocking combined with random background negatives.
- **Probability Calibration:** Isotonic Regression (`IsotonicRegression(out_of_bounds='clip')`) fit on the 5,000-entity validation sample to produce calibrated match estimates $P(\text{Match} \mid x) \in [0, 1]$.

### 4.3 Decision Layer & Threshold Selection Method
The following settings are recorded as locked in commit `fc81ef2`. The embedding repair restores them into `artifacts/decision_params.json`; it does not retune them. The original full-pool diagnostic script and entity list were not found, so the historical validation claims below are not treated as independently reproduced:
- **Operating Threshold:** $\tau = 0.60$ (optimized for $F_{0.5}$ which weights precision over recall $2:1$).
- **Street Number Consistency Gate:** When both S1 and target have an extractable street number, require $|int(sn1) - int(sn2)| \le 1$. If either lacks an extractable street number, fall through to name floor.
- **Script-Aware Name Floor:**
  - Cross-script Indic pairs (`r'[\u0900-\u0D7F]'`): accept if `name_jaro_winkler >= 0.30`.
  - Latin-script pairs: accept if `(name_token_overlap_ratio >= 0.50 and name_levenshtein_ratio >= 0.50) or (name_levenshtein_ratio >= 0.75)`.
- **Confidence-Gap Abstention:** If top candidate has $p < 0.85$ and $(p_{\text{top}} - p_{\text{runner-up}}) < 0.05$, abstain on the entity.
- **Global 1:1 Target Uniqueness:** Bipartite greedy assignment sorted by calibrated probability descending.
- **Domain Match Budget Cap:** Truncate to top-scoring matches: max 5 S2 targets, max 6 S3 targets, max 11 total targets per S1 entity.

---

## 5. Results & Error Analysis

### 5.1 Historical Validation Metrics (recovered, not yet reproduced)

These figures are preserved from the recovered narrative. The older Stage 5 evaluator used `0.25 * recall + precision` in the denominator (F2), rather than the competition F0.5 denominator `0.25 * precision + recall`. It also reused records seen by the calibrator. Its reported F0.5 values cannot be used as an unbiased baseline for the corrected run.

- **Validation Holdout Macro $F_{0.5}$ (1.28M scale):** **0.942**
- **Density-Matched Validation Macro $F_{0.5}$ (10.3M test scale):** **0.835**
- **Blocking Candidate Recall (test scale):** **85.07%**
- **Singleton Accuracy (test scale with locked gates):** **46.38%** (compared to 4.35% without street gate and 17.39% with name floor only).

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges):**
  1. *Co-Located Business Twins with Identical Street Numbers*: Distinct retail businesses operating inside the same shopping mall, corporate park, or municipal center that share the exact street address and city, whose names contain common regional words.
  2. *Corporate Spin-offs & Subsidiaries*: Entities sharing identical brand names and addresses where legal suffixes or subsidiary designations are absent or noise-corrupted.
- **Common False Negatives (Missed Matches):**
  1. *Blocking Capacity Ceiling*: At 10M target scale, ~14.9% of true positive targets were pushed beyond the top-60 candidate limit due to heavy frequency competition on popular municipal keywords.
  2. *Heavily Abbreviated Addresses*: Records with zero street number, missing postal code, and conflicting neighborhood nicknames.

---

## 6. Conclusion
The pipeline combines multi-index blocking, a calibrated LightGBM classifier, and explicit decision rules. The embedding input mismatch has been corrected and verified against saved training-time features. The paired full-pool diagnostic improves only slightly after correction, with substantial false positives still present. Historical validation scores used a different metric formula and reused calibration records, so they do not establish production quality. Further independent, density-matched validation is needed before choosing additional gates or finalizing corrected output files.

---

## Appendix

### A. Code Artefacts
All reproducible code is contained within the repository:
- `code/business_entity_resolution/src/`:
  - `run_stage6_inference.py`: Primary production entry point executing full candidate generation, scoring, decision gating, and validation checks.
  - `blocking.py`: `CountryBlockingEngine` multi-key inverted index implementation.
  - `features.py`: 27-feature pairwise feature extractor.
  - `normalize.py`: Unicode normalization, address abbreviation expansion, street number and postal code extraction.
  - `config.py`: File paths, column specifications, and shared constants.
- `code/business_entity_resolution/README.md`: Currently empty; use the repair reproduction commands below.
- `code/business_entity_resolution/requirements.txt`: Pinned Python dependencies (`lightgbm`, `scikit-learn`, `rapidfuzz`, `jellyfish`, `pandas`, `numpy`).

### B. Summary of Output Deliverables
1. `output/matching_results.tsv`:
   - Exact count: 1,732,544 rows (matches exact `test_source1.tsv` entity set).
   - Format: `source1_entity_id \t matched_entity_ids` (comma-separated, empty for singletons).
2. `output/candidate_pairs.tsv`:
   - Exact count: 1,732,544 rows.
   - Format: `source1_entity_id \t candidate_entity_ids` (top-60 candidates per entity).
   - Invariant: 100% of matched IDs in `matching_results.tsv` are strict subsets of `candidate_pairs.tsv` for every entity.


## Embedding Repair — September 26, 2026

The original `run_stage6_inference.py` did not invoke the embedding model. It assigned
Jaro–Winkler name similarity and address token overlap to columns trained on semantic
cosine values. These substitutions have been removed.

Training and inference now share `embeddings.py`: model
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, cached revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, batch size 512,
`normalize_embeddings=True`, and `float(np.dot(vector1, vector2))`. The model's
saved maximum sequence length (128) is unchanged. An empty normalized string has
no vector; a pair with either vector missing receives cosine 0, matching training.
Unique normalized strings are encoded once and persisted in
`artifacts/minilm_embeddings.sqlite`, with only each inference batch's vectors loaded
into memory. MPS is selected when available, otherwise CPU, as in training. Model
weights are loaded from the existing local cache; no business identity lookup occurs.

`decision.py` applies the same JSON configuration in production and diagnostics.
A versioned copy of the restored configuration is saved as
`code/business_entity_resolution/decision_params.locked.json`.
Country partitions are derived from normalized country labels in the input, rather
than an enumerated list. Street-number comparison converts digit-only numbers to
integers, ignoring leading zeros; missing or compound numbers do not trigger the
numeric gate. The Indic branch applies when either raw name matches the documented
Unicode range, while other pairs use the lexical floor. Target uniqueness is global
within each country, and each S1 may have multiple targets subject to the configured
5/6/11 caps. The old `decision_params.json` is preserved under
`artifacts/embedding_fix/decision_params.json` for provenance.

### Reproducing the paired diagnostic

Run from the project root with the pinned dependencies installed and the model in
the local Hugging Face cache:

```bash
cp code/business_entity_resolution/decision_params.locked.json artifacts/decision_params.json
PYTHONHASHSEED=42 python3 -B code/business_entity_resolution/src/test_embedding_inference.py
PYTHONHASHSEED=42 python3 -u -B code/business_entity_resolution/src/diagnose_embedding_fix.py
```

The diagnostic reconstructs the 5,000 validation entities from the original seeded
sampling sequence and verifies them against `val_features.npz`. It scans every
training S2/S3 target (10,320,219 records), retaining only the index keys relevant to
these queries while preserving full-pool document counts, posting limits, and IDF.
No missing ground-truth positives are injected into the candidates. Real and proxy
embedding variants use exactly the same candidates and trained model. Both old
inference settings and restored locked settings are evaluated without tuning.

Results, sample IDs, and logs are stored in `artifacts/embedding_fix/`. The report
separately records pooled precision, recall and F0.5, per-entity macro metrics, and
singleton accuracy. Empty/empty singletons score 1 for macro F0.5; false matches on
singletons score 0. This is a paired diagnostic, not an independent holdout: these
5,000 entities were already used for calibration, and the full-pool training data
provides scale-matched labeled evaluation, not France test-set accuracy. Assignment
competition is limited to the 5,000 queried S1 records.

The production output TSVs and submission ZIP have not been regenerated by this
repair. Existing output scores must not be presented as scores from corrected
embedding inference. The unused empty `evaluate_dense_blocking.py`,
`test_embedding_blocking.py`, and archived `resolve_stage5.py` are not dependencies
of the production run path.

### Feature parity verification

The production inference test compares both semantic columns with direct MiniLM
encoding, including an Indic/Latin pair and a missing address. Five regression
checks pass, including persistent cache reuse, rejection of stale configuration,
street-number/budget behavior, streaming/full-index equivalence on a fixture, and
the competition metric. The existing normalization suite passes 79/79 checks.

A separate comparison against 1,000 pairs in the original `val_features.npz`
finds maximum absolute differences of `4.77e-7` for name cosine and `3.58e-7`
for address cosine. These are floating-point differences, confirming compatibility
with the features seen by the saved trained model. The original Stage 6 decision
logic independently reproduces the new diagnostic's proxy baseline exactly.
See `artifacts/embedding_fix/training_feature_parity.json` and
`artifacts/embedding_fix/baseline_equivalence.json` for the recorded checks.


## Corrected Full-Pool Diagnostic Results

5,000 saved validation entities; all 10,320,219 labeled training targets; 298,029 candidate pairs. Model and calibrator unchanged. Precision/recall below are pooled; F0.5 is the competition per-entity macro average.

| Decision settings | Embedding inputs | Precision | Recall | Macro F0.5 | Singleton accuracy |
|---|---|---:|---:|---:|---:|
| Original inference (tau 0.40, gap 0.05) | String proxies | 12.3359% | 83.2340% | 0.147248 | 1.1236% |
| Original inference (tau 0.40, gap 0.05) | Real MiniLM | 12.5297% | 83.3257% | 0.148986 | 1.1236% |
| Restored locked rules (tau 0.60) | String proxies | 59.3363% | 66.3650% | 0.606961 | 43.0712% |
| Restored locked rules (tau 0.60) | Real MiniLM | 59.4249% | 66.5254% | 0.608125 | 43.4457% |

Under the same locked rules, embedding repair changes macro F0.5 by +0.001164 (+0.1164 percentage points). True positives: 11,586 → 11,614; false positives: 7,940 → 7,930; correct singletons: 115 → 116 of 267.

Candidate recall is 85.1300%. An oracle accepting exactly the true links in these candidates would achieve macro F0.5 0.923043; the embedding fix alone leaves a substantial matching gap.

All 25 non-embedding feature columns are bit-for-bit identical across all 298,029 pairs. A separate 1,000-pair comparison against original training-time validation embeddings has maximum cosine error below 4.8e-7. The independently replayed original inference decision logic exactly matches the proxy baseline.

The corrected locked result is 0.462969 macro F0.5 for India and 0.709834 for US. France has no labeled training holdout and is not evaluated here.

This is a paired diagnostic, not an untouched holdout: these records were previously used to fit calibration. The earlier sprint full-pool script/entity list was not found in working files or history; this run reconstructs the existing saved 5,000-entity sample. The historical 0.835 claim is not a reproducible paired baseline. Older evaluators also calculated F2 while labeling it F0.5. Target assignment competition covers 5,000 source entities, not the entire production source population; these values are not leaderboard estimates.

Conclusion: retain the embedding correction, but it is not the dominant measured cause of the gap in this diagnostic. Do not treat the model as ready for final packaging based on this repair. The embedding repair alone does not close the diagnostic gap. The subsequent bounded decision re-verification is recorded below; no new gate designs were introduced.

The active JSON matches the versioned locked configuration: tau 0.60; gap 0.05 below probability 0.85; digit-only street-number difference <=1; Indic name Jaro–Winkler >=0.30; other names require (token overlap >=0.50 and Levenshtein >=0.50) or Levenshtein >=0.75; S2/S3/total budgets 5/6/11; target uniqueness enabled; old co-location guard disabled.

Production TSVs and the submission ZIP have not been regenerated. The methodology was recovered from fc81ef2 and updated with the repair and these findings.


### Correct-F0.5 decision re-verification (26 September 2026)

The existing 298,029-pair feature matrix was scored with the saved LightGBM model and isotonic calibrator in 1.86 seconds. Calibrated probabilities and provenance hashes are saved in `artifacts/embedding_fix/calibrated_probabilities.npz`. This sweep did not retrain, rebuild blocking, or extract features.

All 54 configurations used `diagnose_embedding_fix.metrics`: per-entity F0.5 = `1.25*P*R/(0.25*P+R)`, averaged across the same 5,000 sources. The hand check P=2/3, R=1 gives 5/7 = 0.714286. Tau values were 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85; floors were the existing `overlap_and_lev` and disabled; street tolerance was 0, 1, or disabled. Confidence-gap handling, target uniqueness, and 5/6/11 caps stayed fixed. The unrecoverable `tighter_1of3` was excluded.

| Configuration | Macro F0.5 | Pooled precision | Pooled recall | Singleton accuracy |
|---|---:|---:|---:|---:|
| Grid winner: tau 0.85, overlap_and_lev, street tol 0 | 0.613975 | 60.4206% | 65.3282% | 47.9401% |
| Retained lock: tau 0.60, overlap_and_lev, street tol 1 | 0.608125 | 59.4249% | 66.5254% | 43.4457% |

The stricter setting wins by 0.005849. Under the explicitly authorized 0.01 retention margin, the active lock remains unchanged. The full grid is saved in `artifacts/decision_reverification/grid.csv` and `GRID_REPORT.md`; all earlier calibration-reuse, source-competition, and France-label limitations still apply.

Full test inference was then started with real MiniLM embeddings, the retained JSON configuration, and the existing candidate lists. Replacement outputs are staged in `output/embedding_corrected_20260926`; its `run_status.json` records completion or timeout. Partial staged files must not be submitted. The canonical outputs and ZIP are not replaced by this bounded run.
