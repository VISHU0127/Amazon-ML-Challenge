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

| File | Rows | Columns |
|------|------|---------|
| train_source1 | 2,206,821 | entity_id, business_name, business_address, country |
| train_source2 | 5,034,616 | same |
| train_source3 | 5,285,603 | same |
| train_ground_truth | 2,206,821 | source1_entity_id, matched_entity_ids |
| test_source1 | 1,732,544 | entity_id, business_name, business_address, country |
| test_source2 | 4,887,273 | same |
| test_source3 | 5,082,316 | same |

**Null / empty rates:**
- Source 1 (train & test): **0% empty** on all columns
- Source 2: business_address empty 3.36% (train), 2.65% (test)
- Source 3: business_address empty 3.33% (train), 2.68% (test)
- business_name and country: **never empty** in any source

**Country distribution:**
- Train: US 60%, India 40% (both sources)
- Test: India ~47%, US ~38%, **France ~15%** (unseen in training!)

**Ground truth stats:**
- Singletons (no match): 123,247 (5.58%)
- With ≥1 match: 2,083,574 (94.42%)
- Avg matches per S1 entity: 3.46 (3.67 among non-singletons)
- Max matches: 11
- Match distribution peak: 3 matches (24%), then 4 (22%), 2 (17%)
- S2 match IDs: 3,693,619 | S3 match IDs: 3,944,746
- All GT S1 IDs verified present in train_source1 (zero orphans)

**Validation split:**
- 331,023 val S1 entities (15.0%) / 1,875,798 train (85.0%)
- Grouped by S1 entity — zero leakage
- Singleton rates balanced: val 5.59%, train 5.58%

### Assumptions
- Country is treated as an open-vocabulary string (no one-hot encoding)
- The ~3% empty business_address rate in S2/S3 means address features must gracefully handle missing data
- France entities (~15% of test) have no training signal — pipeline must generalise from name/address similarity patterns learned on US/India

### What's next
- Stage 1: Build `normalize.py` with lowercase, Unicode NFKD, diacritics stripping, legal-suffix canonicalization, address abbreviation tables, postal code & street number extraction

### Validation script
- `utils/validate_submission.py` confirmed runnable (Python 3.8+, stdlib only)
