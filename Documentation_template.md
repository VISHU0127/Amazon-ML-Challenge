# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** EntityResolvers  
**Team Members:** Antigravity Team  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary
We developed an end-to-end, scale-resilient business entity resolution system designed to resolve 1,732,544 Source 1 entities across France, the United States, and India against a combined target pool of 9,969,589 candidate entities (Source 2 and Source 3). Our architecture couples a country-partitioned multi-tier inverted index blocking engine with a calibrated gradient-boosted pairwise ranker (LightGBM on 26 engineered string-distance, phonetic, geospatial, and source-provenance features) and a locked multi-gate decision layer. Key innovations include:
1. A **Numeric Street-Number Consistency Gate** ($|\text{street}_1 - \text{street}_2| \le 1$ with leading-zero normalization) that systematically eliminates address-driven twin false merges in dense urban grids.
2. A **Script-Aware Dual Name Floor** that enforces strict lexical overlap for Latin-script text while preserving cross-script Indic transliterations via Unicode block isolation (`r'[\u0900-\u0D7F]'`).
3. A **Global 1-to-1 Bipartite Greedy Matcher** augmented with domain-specific match budget caps (max 5 S2, max 6 S3, max 11 total matches per entity).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis and validation audits across the 50,000 training entities and 10M test targets, we identified several critical failure modes and data properties:
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
|    - 6-tier inverted indices: Name IDF, Soundex, 3-char prefix, Address IDF,      |
|      Street number index, Postal code prefix                                      |
|    - Evaluates ~103.8M candidate pairs across all countries                       |
|    - Directly emitted to `output/candidate_pairs.tsv`                             |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|  PAIRWISE RANKING & CALIBRATION                                                   |
|    - 26 string, phonetic, address, suffix, and provenance features                |
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
  - When evaluated against the full **10.3M target pool scale**, candidate recall was measured at **85.07%**. Because the candidate pool expanded by ~8x while the top-$k$ capacity was fixed at 60, high-frequency generic tokens competed for index slots, establishing an inherent 85.07% upper bound on test-time recall.

---

## 4. Matching Model

### 4.1 Features Used (26 total)
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
   - `name_embedding_cosine`: High-fidelity dense semantic proxy.

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
   - `addr_embedding_cosine`: Address semantic overlap proxy.

4. **Meta & Interaction Features (5):**
   - `country_exact_match`: Binary match of country field.
   - `target_is_s2`, `target_is_s3`: Source provenance one-hot indicators.
   - `blocking_score`, `blocking_rank`: Rank and multi-key score from blocking phase.

### 4.2 Model Type & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree Classifier (`LGBMClassifier`).
  - Objective: Binary Logloss.
  - Hyperparameters: `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`.
  - Class Imbalance Handling: Hard negatives from blocking combined with random background negatives.
- **Probability Calibration:** Isotonic Regression (`IsotonicRegression(out_of_bounds='clip')`) fit on out-of-fold validation predictions to produce true posterior match probabilities $P(\text{Match} \mid x) \in [0, 1]$.

### 4.3 Decision Layer & Threshold Selection Method
All decision parameters were locked based on extensive density-matched validation at the full 10.3M target scale:
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

### 5.1 Validation Metrics
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
By pairing an efficient multi-index blocking engine with a calibrated LightGBM ranker and a rigorous geographic/script-aware decision layer, our solution achieves high macro $F_{0.5}$ precision while maintaining robustness across 1.73M entities and 103M candidate evaluations. The project demonstrated that at multi-million target scale, pure ML probability thresholds must be grounded by structural invariants — specifically numeric street-number consistency and script-aware name similarity floors — to resist false-merge density inflation.

---

## Appendix

### A. Code Artefacts
All reproducible code is contained within the repository:
- `code/business_entity_resolution/src/`:
  - `run_stage6_inference.py`: Primary production entry point executing full candidate generation, scoring, decision gating, and validation checks.
  - `blocking.py`: `CountryBlockingEngine` multi-key inverted index implementation.
  - `features.py`: 26-feature pairwise feature extractor.
  - `normalize.py`: Unicode normalization, address abbreviation expansion, street number and postal code extraction.
  - `config.py`: File paths, column specifications, and shared constants.
- `code/business_entity_resolution/README.md`: Step-by-step reproduction instructions and execution guide.
- `code/business_entity_resolution/requirements.txt`: Pinned Python dependencies (`lightgbm`, `scikit-learn`, `rapidfuzz`, `jellyfish`, `pandas`, `numpy`).

### B. Summary of Output Deliverables
1. `output/matching_results.tsv`:
   - Exact count: 1,732,544 rows (matches exact `test_source1.tsv` entity set).
   - Format: `source1_entity_id \t matched_entity_ids` (comma-separated, empty for singletons).
2. `output/candidate_pairs.tsv`:
   - Exact count: 1,732,544 rows.
   - Format: `source1_entity_id \t candidate_entity_ids` (top-60 candidates per entity).
   - Invariant: 100% of matched IDs in `matching_results.tsv` are strict subsets of `candidate_pairs.tsv` for every entity.
