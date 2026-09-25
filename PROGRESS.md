# PROGRESS LOG — Business Entity Resolution Pipeline

---

## Stage 0 — Setup & Data Audit (2026-09-25)

### What was built
- Project structure: `code/business_entity_resolution/src/`, `output/`, `artifacts/`
- `config.py`: centralised, parameterized path configuration (no hardcoded paths)
- `requirements.txt`: pinned dependencies (pandas 2.2.3, scikit-learn 1.5.2, rapidfuzz 3.10.1, lightgbm 4.5.0, jellyfish 1.1.0, tqdm 4.67.1)
- `data_audit.py`: comprehensive data audit script
- Validation split saved to `artifacts/val_split.json`

### Data audit summary

| File | Rows |
|------|------|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

- Null rates: S1 = 0%; S2/S3 business_address ~3% empty
- Train countries: US 60%, India 40%
- Test countries: India 47%, US 38%, France 15% (unseen!)
- Singletons: 5.58%; Avg matches per S1: 3.46; Max: 11
- Validation split: 331,023 val / 1,875,798 train (15%/85%), grouped by S1 entity

---

## Stage 1 — Normalization (2026-09-25)

### What was built
- `normalize.py` — core normalization module with:
  - `normalize_name()`: lowercase, NFKD (Latin only), URL strip, legal suffix removal, &→and, punctuation strip
  - `normalize_address()`: lowercase, multi-word abbrev (Hindi states before NFKD), single-word abbrev, NFKD, punctuation strip
  - `normalize_country()`: lowercase, strip (open vocabulary)
  - `extract_legal_suffixes()`, `extract_postal_code()`, `extract_street_number()`
- `mine_patterns.py` — mined suffix/abbreviation tables from 18K+ training pairs
- `test_normalize.py` — **52/52 unit tests passing**

### Legal suffix table
Top: limited(6318), private(5644), llc(5222), inc(3354), ltd(2484), pvt(1704), services(907), partners(674), group(669), associates(623), corp(576), llp(561). Hindi: प्राइवेट, लिमिटेड. French: sarl, sas, sa, eurl, sci.

### Example before/after

| Raw Name | Normalized |
|----------|-----------|
| `Clairvoyant Récord Private Ltd` | `clairvoyant record` |
| `सुप्रीम आईटी प्राइवेट लिमिटेड` | `सुप्रीम आईटी` |
| `www.cardiology.com` | `cardiology` |
| `Unique & Sons Private Limited` | `unique and sons` |

| Raw Address | Normalized |
|-------------|-----------|
| `88 Olive Circle, Lebanon, TN` | `88 olive cir lebanon tn` |
| `Pune, महाराष्ट्र` | `pune mh` |
| `##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD` | `19821 wheelwright dr montgomery village md` |
---
---

## Stage 2 — Blocking / Candidate Generation (2026-09-25)

### What was built
- `blocking.py` — multi-strategy candidate generation engine implementing 3 independent strategies:
  1. **Name Token Inverted Index with IDF Weighting**: indexes normalized name tokens (len >= 2) with hyphen/slash expansion, weights postings by inverse document frequency to prioritize discriminative tokens over high-frequency generic words.
  2. **Phonetic & Prefix Keys**: indexes Soundex/Metaphone of first name token and 3-character prefixes to catch typos and minor spelling variations.
  3. **Distinctive Address Tokens & Street Numbers**: indexes address tokens (len >= 4, excluding common street stopwords) and numeric street numbers to capture transliterations (e.g. English vs Devanagari script) and acronyms where names differ drastically but physical locations match.
- **Open-Vocabulary Country Partitioning**: partitions indexing and candidate retrieval by normalized country string without hardcoding to `{US, India}`. Peak memory is kept under 1.5GB on 8GB machines.
- **Cross-Country Fallback Mechanism**: if country-partitioned candidate retrieval yields 0 candidates (or if an unknown/empty country is encountered), automatically falls back to an unconstrained multi-strategy search across the global candidate pool so novel formatting variations can never drop candidates to zero.
- **Rank Fusion & Capping**: fuses scores across all 3 strategies and caps at top-60 candidates per S1 entity.
- `run_stage2_blocking.py` — execution script evaluating recall and reduction ratio against Ground Truth and exporting candidate TSVs.

### Ground Truth Country Consistency Verification
- Inspected all 7,638,365 ground-truth matched pairs across the entire training dataset:
  - Raw country mismatches: **0 (0.0000%)**
  - Normalized country mismatches: **0 (0.0000%)**
  - Result: 100% of true matches strictly share identical country labels across S1, S2, and S3.

### Fusion Rank Drop Root-Cause Analysis
- Investigated the rank distribution of true matches retrieved by unconstrained multi-strategy that fell outside top-35:
  1. **Missing Target Address Scale Discrepancy**: As identified in Stage 0, ~3.4% of S2/S3 records have completely empty addresses. In such cases, Strategy 3 (address tokens + street number) contributes 0 points. For common or multi-word business names, distractor records in the target pool that match even a single common address token or locality can accumulate additive score across both strategies and outrank a perfect name match whose target address is null.
  2. **Compound / Hyphenated Tokens**: Names with hyphens (e.g. `Uptown-Yoga!`, `Red-Pizza`) were previously parsed as single compound tokens. Added hyphen/slash token expansion (`_extract_all_name_tokens`) so both composite and individual sub-words are indexed and matched.

### Recall by Candidate Cap (Validation Split, 25,000 S1 Entities, 1.28M Targets)
- **Total true matches in eval set**: 86,464
- **Cap top-35**: 94.68% recall (81,861 matches) | Reduction ratio: 99.999661%
- **Cap top-50**: 95.44% recall | Reduction ratio: 99.999515%
- **Cap top-60**: **95.92% recall** (82,936 matches) | Reduction ratio: **99.999419%**
- **Cap top-75**: **96.69% recall** | Reduction ratio: 99.999273%
- **Cap top-100**: **96.89% recall** | Reduction ratio: 99.999031%
- **Adopted Configuration**: Default candidate cap set to **60** (with top-75 option), achieving 95.92% recall at 99.9994% reduction ratio.
- **Throughput**: ~550 S1 entities/second
- **Output Artifact**: `artifacts/candidate_pairs_eval.tsv` (validated against official TSV schema: PASS)

---

---

## Stage 3 — Feature Engineering (2026-09-25)

### What was built
- `features.py` — pair-wise feature extraction engine computing 25 discriminative features per (S1, Candidate Target) pair:
  - **Name Similarity Features (10)**:
    1. `name_jaro_winkler`: Jaro-Winkler similarity on normalized names
    2. `name_levenshtein_ratio`: normalized Levenshtein similarity
    3. `name_token_jaccard`: token intersection / union
    4. `name_token_overlap_ratio`: token intersection / min(tokens1, tokens2)
    5. `name_char_3gram_jaccard`: character 3-gram Jaccard similarity
    6. `name_first_token_soundex_match`: binary phonetic equality on first word
    7. `name_first_token_metaphone_match`: binary metaphone equality on first word
    8. `name_acronym_match`: binary flag if acronym of S1 matches Target or vice-versa
    9. `name_exact_match`: raw or normalized exact match flag
    10. `name_len_diff`: absolute character length difference
  - **Legal Suffix Features (3)** (from Stage 1 follow-up):
    11. `suffix_normalized_equality`: binary flag (1.0 if canonical legal suffixes match, 0.0 if mismatch)
    12. `suffix_presence_s1`: binary flag if S1 had a recognized legal suffix
    13. `suffix_presence_target`: binary flag if candidate had a recognized legal suffix
  - **Address Similarity Features (6)**:
    14. `addr_token_jaccard`: word token Jaccard similarity of normalized addresses
    15. `addr_token_overlap_ratio`: word token overlap ratio
    16. `addr_levenshtein_ratio`: normalized Levenshtein similarity
    17. `addr_postal_code_match`: 1.0 if postal codes match, 0.0 if both exist and mismatch, 0.5 if either missing
    18. `addr_street_num_match`: 1.0 if street numbers match, 0.0 if mismatch, 0.5 if either missing
    19. `addr_is_empty_target`: binary flag indicating missing target address (handles the ~3.4% null rate)
  - **Meta & Interaction Features (6)**:
    20. `country_exact_match`: binary flag (1.0 if normalized countries match, 0.0 otherwise) — open vocabulary, not one-hot
    21. `target_is_s2`: binary source indicator for Source 2
    22. `target_is_s3`: binary source indicator for Source 3
    23. `blocking_score`: composite rank fusion score from blocking engine
    24. `blocking_rank`: candidate rank in retrieved candidate list (1 to 60)
    25. `name_addr_sim_product`: non-linear interaction term (`name_jaro_winkler * addr_token_overlap_ratio`)
- `build_feature_matrices.py` — pipeline script generating balanced training and validation matrices using true matches and hard negatives from the blocking stage.

### Feature Matrix Summary
- **Training Matrix (`artifacts/train_features.npz`)**:
  - Shape: `(261,434, 25)`
  - Positives (true matches): **51,763**
  - Hard Negatives (top non-matches from blocking): **209,671**
  - Size on disk: 7.11 MB
- **Validation Matrix (`artifacts/val_features.npz`)**:
  - Shape: `(88,091, 25)`
  - Positives: **17,458**
  - Hard Negatives: **70,633**
  - Size on disk: 2.40 MB
- **Metadata**: Saved to `artifacts/feature_summary.json`

### What's next
- Stage 4: Model Training — Train LightGBM binary classifier on cached feature matrix split by S1 entity, calibrate output probabilities (Platt/isotonic), evaluate validation AUC/PR and feature importances, and save model artifacts.
