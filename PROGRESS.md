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
