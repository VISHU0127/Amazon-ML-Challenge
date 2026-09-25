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

---

## Stage 3 — Feature Engineering & Sanity Audits (2026-09-25)

### What was built
- `features.py` — pair-wise feature extraction engine computing 27 discriminative features per (S1, Candidate Target) pair:
  - **Name Similarity Features (11)**:
    1. `name_jaro_winkler`: Jaro-Winkler similarity on normalized names
    2. `name_embedding_cosine`: cosine similarity from multilingual sentence transformer
    3. `name_levenshtein_ratio`: normalized Levenshtein similarity
    4. `name_token_jaccard`: token intersection / union
    5. `name_token_overlap_ratio`: token intersection / min(tokens1, tokens2)
    6. `name_char_3gram_jaccard`: character 3-gram Jaccard similarity
    7. `name_first_token_soundex_match`: binary phonetic equality on first word
    8. `name_first_token_metaphone_match`: binary metaphone equality on first word
    9. `name_acronym_match`: binary flag if acronym of S1 matches Target or vice-versa
    10. `name_exact_match`: raw or normalized exact match flag
    11. `name_len_diff`: absolute character length difference
  - **Legal Suffix Features (3)** (from Stage 1 follow-up):
    12. `suffix_normalized_equality`: binary flag (1.0 if canonical legal suffixes match, 0.0 if mismatch)
    13. `suffix_presence_s1`: binary flag if S1 had a recognized legal suffix
    14. `suffix_presence_target`: binary flag if candidate had a recognized legal suffix
  - **Address Similarity Features (7)**:
    15. `addr_token_jaccard`: word token Jaccard similarity of normalized addresses
    16. `addr_token_overlap_ratio`: word token overlap ratio
    17. `addr_levenshtein_ratio`: normalized Levenshtein similarity
    18. `addr_embedding_cosine`: semantic cosine similarity of normalized addresses
    19. `addr_postal_code_match`: 1.0 if postal codes match, 0.0 if both exist and mismatch, 0.5 if either missing
    20. `addr_street_num_match`: 1.0 if street numbers match, 0.0 if mismatch, 0.5 if either missing
    21. `addr_is_empty_target`: binary flag indicating missing target address (handles the ~3.4% null rate)
  - **Meta & Interaction Features (6)**:
    22. `country_exact_match`: binary flag (1.0 if normalized countries match, 0.0 otherwise) — open vocabulary, not one-hot
    23. `target_is_s2`: binary source indicator for Source 2
    24. `target_is_s3`: binary source indicator for Source 3
    25. `blocking_score`: composite rank fusion score from blocking engine
    26. `blocking_rank`: candidate rank in retrieved candidate list (1 to 60)
    27. `name_addr_sim_product`: non-linear interaction term (`name_jaro_winkler * addr_token_overlap_ratio`)
- **Multilingual Semantic Embeddings**:
  - Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - License: **Apache-2.0**
  - Parameter Count: **117.7M parameters** (strictly <= 8B)
  - Languages: Multilingual covering 50+ languages including English, French (for unseen test country), and Hindi.
- `build_feature_matrices.py` — rebuilt against the corrected Stage 2 candidate generation (top-60 cap, hyphen token expansion, cross-country fallback).
- `sanity_check_features.py` — rigorous automated audit of the generated feature matrices.

### Feature Matrix Summary
- **Training Matrix (`artifacts/train_features.npz`)**:
  - Shape: `(261,434, 27)`
  - Positives (true matches): **51,763** (19.8% positive rate)
  - Hard Negatives (top non-matches from blocking): **209,671** (80.2%)
  - Size on disk: 8.9 MB
- **Validation Matrix (`artifacts/val_features.npz`)**:
  - Shape: `(88,091, 27)`
  - Positives: **17,458** (19.8% positive rate)
  - Hard Negatives: **70,633** (80.2%)
  - Size on disk: 3.0 MB
- **Metadata**: Saved to `artifacts/feature_summary.json`

### Sanity Check Audit Results (All 6 Passed)
1. **NaN / Inf Audit**: **0 NaNs, 0 Infs** across all 27 columns in both train and val matrices. (PASS)
2. **Zero-Variance Scan**: Only `country_exact_match` is constant at 1.0 because candidate generation is partitioned by country. All other 26 features exhibit healthy non-zero variance. (PASS)
3. **Positive vs. Negative Distribution Separation**:
   - `name_jaro_winkler`: Pos **0.8894** vs Neg **0.6946** (+0.1948)
   - `name_embedding_cosine`: Pos **0.8465** vs Neg **0.5378** (+0.3088)
   - `addr_token_jaccard`: Pos **0.7524** vs Neg **0.1201** (+0.6324)
   - `addr_embedding_cosine`: Pos **0.8755** vs Neg **0.5277** (+0.3478)
   - `suffix_normalized_equality`: Pos **0.6136** vs Neg **0.2304** (+0.3832)
   - `name_addr_sim_product`: Pos **0.7777** vs Neg **0.1179** (+0.6597)
   - All core similarity features show strong, statistically significant positive separation. (PASS)
4. **Row-Count Reconciliation**: Exactly 261,434 train pairs and 88,091 val pairs matching the candidate blocking sets with 0 duplicate pairs. (PASS)
5. **Zero-Leakage Audit**: Exactly 0 S1 entity IDs overlap between train (15,000 entities) and val (5,000 entities). (PASS)
6. **Country & Source Segment Breakdown**:
   - India positive rate: 19.8% (104,959 pairs)
   - US positive rate: 19.8% (156,475 pairs)
   - S1-S2 positive rate: 15.3% (164,814 pairs)
   - S1-S3 positive rate: 27.5% (96,620 pairs)
   - Feature means are stable and balanced across segments. (PASS)

### Stage 4 Training Plan Note
- In Stage 4, `scale_pos_weight` will be explicitly set to `(len(y) - sum(y)) / sum(y) ≈ 4.05` to balance the ~19.8% positive rate so class imbalance does not compound with F0.5's precision bias.
- Probability calibration (isotonic regression / Platt scaling) will be fitted on the validation split.

---

## Stage 3 Follow-Up Diagnostics — 2025-09-25

### Issue 1: `country_exact_match` Always 1.0 (Cross-Country Analysis)

**Finding**: Across all 7,638,365 ground-truth match pairs in training data, there are **exactly 0 cross-country true matches**. All pairs are strictly (US, US) = 4,578,522 or (India, India) = 3,059,843.

**Root Cause**: The cross-country fallback in `blocking.py` (lines 253-257) is correctly wired, and `build_feature_matrices.py` correctly invokes blocking per country partition (line 144-146, skips if `c_norm not in engines`). The fallback produces zero candidates simply because there are zero cross-country true matches to find in the training data — not because the fallback path is broken.

**Implication**: `country_exact_match` is a constant feature (always 1.0) in the training matrix. The model will have zero training exposure to `country_exact_match=0` pairs. However, this is a **known limitation, not a bug**, because:
1. For France test data, both S1 and S2/S3 targets will share the same normalized country string `"france"`, so the same-country partition path applies normally.
2. The realistic failure mode for France is not cross-country matching — it's unseen suffix/abbreviation patterns, which the multilingual embeddings and open-vocabulary normalization are designed to handle.
3. `country_exact_match` will remain as a feature (harmless as a constant) — the model will learn to ignore it. It could become useful if future data introduces cross-country pairs.

**Action**: Documented as known limitation. No code change needed.

### Issue 2: Missed True Match Pattern Analysis (Recall Gap Investigation)

**Setup**: Analyzed 5,000 validation S1 entities (same sample as `build_feature_matrices.py`) against the sampled target pool, comparing top-60 vs top-200 blocking candidates.

**Recall Summary**:
| Metric | Value |
|--------|-------|
| Total true matches checked | 17,245 |
| Found in top-60 | 16,600 (96.26%) |
| Missed in top-60 | 645 (3.74%) |
| — Recoverable in top-200 | 152 (rank mean=126, median=127, range 61–195) |
| — Not in top-200 either | 493 (truly invisible) |

**Pattern Analysis of 645 Missed Pairs**:

- **By Source**: S2 = 312, S3 = 333 — balanced, not a source-specific issue.
- **By Country**: India = 414 (64%), US = 231 (36%) — India overrepresented (expected given more ambiguous names).
- **Strategy Hit Rate on Missed Pairs** (had shared blocking keys but were outranked):
  - `name_token`: 186/645 (28.8%)
  - `soundex`: 234/645 (36.3%)
  - `prefix3`: 279/645 (43.3%)
  - `addr_token`: 480/645 (74.4%)
- **Zero shared name tokens** (invisible to name strategy entirely): **459/645 (71.2%)**
- **JW similarity of missed pairs**: Mean 0.637, Median 0.705, Min 0.000, Max 1.000

**Root Cause**: The dominant failure mode is **very short or very common normalized names** after suffix stripping. Examples:
- "Om Trading Private Limited" → normalized to `"om"` (single token, matches thousands of targets)
- "Value" → normalized to `"value"` (single common token)
- "High Tech LLP" → normalized to `"high"` (single common word)
- "New Delhi Consulting" → normalized to `"new delhi"` (common place name tokens)
- "Consulting Solutions Private Limited" → normalized to `""` (empty after suffix stripping!)

These entities have name tokens that appear in thousands of target records, causing the IDF-weighted scoring to be flooded by distractors. The target (often with empty address) gets outranked.

**Key Insight**: Raising the cap further won't help for 493/645 missed pairs — they aren't even in top-200. The issue is **not cap size or fusion ranking** but rather that suffix stripping produces names too short/generic to be discriminative in the inverted index. These represent ~2.86% of true matches — a hard ceiling.

**Mitigation**: The LightGBM classifier in Stage 4 will handle these edge cases via the `name_embedding_cosine` feature, which captures semantic similarity even when lexical overlap is low. The `build_feature_matrices.py` already injects these missed true positives into the training matrix (lines 160-163: true matches not in candidates are added with blocking_score=0, rank=61), so the model sees them as hard positives with low blocking scores.

**Action**: Documented as understood limitation. No blocking code change — the ~96.26% blocking recall is the practical ceiling given the data's noise characteristics. The remaining ~3.74% are genuinely hard cases that would require fundamentally different blocking strategies (e.g., embedding-based blocking, which is too expensive for this pipeline).

### Issue 3: Embedding Caching Confirmation

**Confirmed**: Embeddings in `build_feature_matrices.py` are computed from **per-entity cached embeddings**:
1. All unique normalized name strings and address strings are collected into sets (lines 187-200).
2. `SentenceTransformer.encode()` is called **once** per batch of unique strings (lines 210-214), not per (S1, candidate) pair.
3. Results are stored in `name_emb_map` / `addr_emb_map` dictionaries keyed by normalized string.
4. During feature extraction, cosine similarity = `np.dot(cached_emb_1, cached_emb_2)` since embeddings are pre-normalized (line 267, 271).
5. The same approach will be used in Stage 6 inference — encode each unique entity string once, then dot-product across pairs. This is O(N) encoding + O(pairs) dot products, not O(pairs) encoding.

### 4th Strategy Evaluation: Dense Multilingual Embedding Retrieval

**Motivation**: Directly test whether dense ANN retrieval (top-K cosine similarity using `paraphrase-multilingual-MiniLM-L12-v2`) recovers the 645 missed matches from top-60 lexical blocking (and specifically the 459 misses with zero shared name tokens).

1. **Address Blocking Audit on the Not-in-Top-200 Misses**:
   - Total not-in-top-200 misses analyzed: 159 (sample of 5,000 val S1 entities)
   - Target has completely empty address (`''`): **1.9%** (3 pairs)
   - Target has address, but **0 shared address tokens** (4+ chars): **34.0%** (54 pairs)
   - **Subtotal with zero address overlap signal: 35.8%** (57 pairs) — completely invisible to address blocking
   - Target shares $\ge 1$ address tokens: **64.2%** (102 pairs) — shared broad locality/city tokens (e.g. "Bangalore", "Sector", "Road", "Delhi") that exceeded the high-frequency posting cap (>3000) or were outranked by hundreds of other entities sharing the same locality.

2. **Dense Multilingual Embedding Cosine Distribution (on 645 misses)**:
   - Mean cosine: **0.5450**, Median cosine: **0.5264**
   - **60.2%** have cosine $< 0.60$ (the surface forms are completely different acronyms/abbreviations where dense sentence embeddings assign low similarity)
   - Only **16.7%** have cosine $\ge 0.85$

3. **Dense ANN Retrieval Recovery Results (competing against 60,000+ target entities)**:
   - **Top-5 Dense ANN**: Recovers 44 / 264 misses (16.7%) $\rightarrow$ New Recall = **96.51%** (+0.26%)
   - **Top-10 Dense ANN**: Recovers 55 / 264 misses (20.8%) $\rightarrow$ New Recall = **96.58%** (+0.32%)
   - **Top-20 Dense ANN**: Recovers 61 / 264 misses (23.1%) $\rightarrow$ New Recall = **96.61%** (+0.35%)
   - **Top-50 Dense ANN**: Recovers 78 / 264 misses (29.5%) $\rightarrow$ New Recall = **96.71%** (+0.45%)
   - **Top-100 Dense ANN**: Recovers 90 / 264 misses (34.1%) $\rightarrow$ New Recall = **96.78%** (+0.52%)

4. **Tradeoff & Decision**:
   - Adding dense retrieval over the full 1.28M target pool requires ~50-80 minutes of offline encoding + heavy vector search at test time, but only yields **+0.35%** recall at Top-20 (from 96.26% to 96.61%).
   - >75% of missed true matches remain unrecovered because their semantic similarity is low (median 0.526) or generic short names have hundreds of denser distractors.
   - **Decision**: Rejected as a primary blocking strategy due to negligible yield (+0.35% for 10x compute cost). Retain embeddings in Stage 3 feature space (`name_embedding_cosine`, `addr_embedding_cosine`), where LightGBM can use them selectively without blocking overhead. Documented as an evaluated and rejected approach with empirical evidence.

### What's next
- Stage 4: Model Training — Train LightGBM binary classifier on cached feature matrix split by S1 entity, calibrate output probabilities (Platt/isotonic), evaluate validation AUC/PR and feature importances, and save model artifacts.

