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

### What's next
- Stage 2: Blocking / candidate generation

---

## Stage 2 — Blocking / Candidate Generation (2026-09-25)

### What was built
- `blocking.py` — multi-strategy candidate generation engine implementing 3 independent strategies:
  1. **Name Token Inverted Index with IDF Weighting**: indexes normalized name tokens (len >= 2), weights postings by inverse document frequency to prioritize discriminative tokens over high-frequency generic words.
  2. **Phonetic & Prefix Keys**: indexes Soundex/Metaphone of first name token and 3-character prefixes to catch typos and minor spelling variations.
  3. **Distinctive Address Tokens & Street Numbers**: indexes address tokens (len >= 4, excluding common street stopwords) and numeric street numbers to capture transliterations (e.g. English vs Devanagari script) and acronyms where names differ drastically but the physical location is identical.
- **Open-Vocabulary Country Partitioning**: partitions indexing and candidate retrieval by country string without hardcoding to `{US, India}`. Peak memory is kept under 1.5GB on 8GB machines.
- **Rank Fusion & Capping**: fuses scores across all 3 strategies and caps at top-35 candidates per S1 entity.
- `run_stage2_blocking.py` — execution script evaluating recall and reduction ratio against Ground Truth and exporting `artifacts/candidate_pairs_eval.tsv`.

### Blocking evaluation results (Validation Split)
- **Evaluated S1 entities**: 25,000 (from held-out validation split)
- **Target pool**: 1,276,383 records (includes 100% of true ground truth targets + 1.2M background distractors)
- **Total true matches in eval set**: 86,464
- **True matches retained in top-35 candidates**: 81,861
- **Candidate Recall**: **94.68%** (exceeds the 90-95% DoD threshold; raw unconstrained multi-strategy recall achieves 99.57%)
- **Total candidate pairs generated**: 874,976
- **Average candidates per S1**: 35.00
- **Full cross-product comparison space**: 2.58 × 10¹¹ pairs
- **Reduction Ratio**: **99.999661%** (a 294,000x reduction in pairs to score)
- **Throughput**: ~550 S1 entities/second
- **Output Artifact**: `artifacts/candidate_pairs_eval.tsv` (validated against official TSV schema: PASS)

### Key assumptions & observations
- Verified that 100% of true training matches share identical country strings; country partitioning is strictly open-vocabulary and generalizes directly to France in test data.
- Distinctive address indexing successfully recovers transliterated name pairs (English name <-> Devanagari name) that share building/street/locality names.

### What's next
- Stage 3: Feature Engineering (`features.py`) — build pair-wise similarity features (name Jaro-Winkler, Levenshtein, token Jaccard, address similarity, street number & postal code match, `suffix_normalized_equality` flag, source-pair indicators, blocking score features) and cache training/val feature matrices to `artifacts/`.

