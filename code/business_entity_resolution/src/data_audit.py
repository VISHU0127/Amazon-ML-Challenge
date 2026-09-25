#!/usr/bin/env python3
"""
Stage 0 — Data Audit & Validation Split

Loads all train/test source files + ground truth, prints comprehensive stats,
and creates a grouped validation split by source1_entity_id.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Add src to path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    ARTIFACTS_DIR, VAL_SPLIT_PATH, RANDOM_SEED, VAL_FRACTION,
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEP = "\t"

# ═══════════════════════════════════════════════════════════════════════
# 1. Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STAGE 0 — DATA AUDIT")
print("=" * 70)

t0 = time.time()

datasets = {
    "train_source1": TRAIN_SOURCE1,
    "train_source2": TRAIN_SOURCE2,
    "train_source3": TRAIN_SOURCE3,
    "train_ground_truth": TRAIN_GROUND_TRUTH,
    "test_source1": TEST_SOURCE1,
    "test_source2": TEST_SOURCE2,
    "test_source3": TEST_SOURCE3,
}

dfs = {}
for name, path in datasets.items():
    print(f"\nLoading {name} from {path} ...")
    dfs[name] = pd.read_csv(path, sep=SEP, dtype=str, keep_default_na=False)
    print(f"  → {dfs[name].shape[0]:,} rows × {dfs[name].shape[1]} cols")

print(f"\nAll files loaded in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# 2. Schema & null rates per source file
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SCHEMA & NULL / EMPTY RATES")
print("=" * 70)

source_files = ["train_source1", "train_source2", "train_source3",
                "test_source1", "test_source2", "test_source3"]

for name in source_files:
    df = dfs[name]
    print(f"\n── {name} ──")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Rows:    {len(df):,}")
    for col in df.columns:
        n_empty = (df[col].str.strip() == "").sum()
        pct = 100 * n_empty / len(df)
        print(f"   {col:25s}  empty/null: {n_empty:>10,}  ({pct:5.2f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 3. Country value counts per source
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COUNTRY VALUE COUNTS")
print("=" * 70)

for name in source_files:
    df = dfs[name]
    print(f"\n── {name} ──")
    vc = df["country"].value_counts()
    for country, count in vc.items():
        print(f"   {country:20s}  {count:>10,}  ({100*count/len(df):5.2f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 4. Ground-truth analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUND-TRUTH ANALYSIS")
print("=" * 70)

gt = dfs["train_ground_truth"].copy()
print(f"Columns: {list(gt.columns)}")
print(f"Total S1 entities in ground truth: {len(gt):,}")

# Parse matched_entity_ids
gt["match_list"] = gt["matched_entity_ids"].apply(
    lambda x: x.split(",") if x.strip() else []
)
gt["n_matches"] = gt["match_list"].apply(len)

n_singletons = (gt["n_matches"] == 0).sum()
n_with_matches = (gt["n_matches"] > 0).sum()
total_match_ids = gt["n_matches"].sum()

print(f"\nSingletons (no matches):           {n_singletons:>10,}  ({100*n_singletons/len(gt):5.2f}%)")
print(f"With at least 1 match:             {n_with_matches:>10,}  ({100*n_with_matches/len(gt):5.2f}%)")
print(f"Total match IDs across all rows:   {total_match_ids:>10,}")
print(f"Avg matches per S1 entity (all):   {gt['n_matches'].mean():.4f}")
print(f"Avg matches per S1 entity (>0):    {gt.loc[gt['n_matches']>0, 'n_matches'].mean():.4f}")
print(f"Max matches for single S1 entity:  {gt['n_matches'].max()}")

# Distribution of match counts
print("\nMatch-count distribution:")
for n in sorted(gt["n_matches"].unique()):
    cnt = (gt["n_matches"] == n).sum()
    if cnt > 0:
        print(f"   {n:4d} matches: {cnt:>10,}  ({100*cnt/len(gt):5.2f}%)")
    if n >= 15:
        remaining = (gt["n_matches"] > n).sum()
        if remaining > 0:
            print(f"   >  {n} matches: {remaining:>10,}  ({100*remaining/len(gt):5.2f}%)")
        break

# Check S2 vs S3 in matches
all_match_ids = [mid for mids in gt["match_list"] for mid in mids]
s2_count = sum(1 for m in all_match_ids if m.startswith("S2-"))
s3_count = sum(1 for m in all_match_ids if m.startswith("S3-"))
print(f"\nMatch IDs from S2: {s2_count:,}")
print(f"Match IDs from S3: {s3_count:,}")

# Verify all S1 entities in ground truth exist in train_source1
s1_ids_gt = set(gt["source1_entity_id"])
s1_ids_src = set(dfs["train_source1"]["entity_id"])
print(f"\nS1 IDs in ground truth:    {len(s1_ids_gt):,}")
print(f"S1 IDs in train_source1:   {len(s1_ids_src):,}")
print(f"GT S1 IDs in source1:      {len(s1_ids_gt & s1_ids_src):,}")
print(f"GT S1 IDs NOT in source1:  {len(s1_ids_gt - s1_ids_src):,}")

# ═══════════════════════════════════════════════════════════════════════
# 5. Validation split by source1_entity_id (grouped)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CREATING VALIDATION SPLIT")
print("=" * 70)

rng = np.random.RandomState(RANDOM_SEED)

all_s1_ids = gt["source1_entity_id"].unique()
n_val = int(len(all_s1_ids) * VAL_FRACTION)

shuffled = rng.permutation(all_s1_ids)
val_ids = set(shuffled[:n_val])
train_ids = set(shuffled[n_val:])

print(f"Total S1 entities:    {len(all_s1_ids):,}")
print(f"Validation entities:  {len(val_ids):,}  ({100*len(val_ids)/len(all_s1_ids):.1f}%)")
print(f"Train entities:       {len(train_ids):,}  ({100*len(train_ids)/len(all_s1_ids):.1f}%)")

# Verify no overlap
assert len(val_ids & train_ids) == 0, "Leakage: overlap between train and val S1 IDs!"

# Check singleton/match proportions in each split
gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
gt_train = gt[gt["source1_entity_id"].isin(train_ids)]
val_singleton_rate = (gt_val["n_matches"] == 0).mean()
train_singleton_rate = (gt_train["n_matches"] == 0).mean()
print(f"Val singleton rate:   {val_singleton_rate:.4f}")
print(f"Train singleton rate: {train_singleton_rate:.4f}")

# Save split
split_data = {
    "val_s1_ids": sorted(val_ids),
    "train_s1_ids": sorted(train_ids),
    "val_size": len(val_ids),
    "train_size": len(train_ids),
    "seed": RANDOM_SEED,
    "val_fraction": VAL_FRACTION,
}

with open(VAL_SPLIT_PATH, "w") as f:
    json.dump(split_data, f, indent=2)
print(f"\nSplit saved to {VAL_SPLIT_PATH}")

# ═══════════════════════════════════════════════════════════════════════
# 6. Test set overview
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST SET OVERVIEW")
print("=" * 70)

test_s1 = dfs["test_source1"]
test_s2 = dfs["test_source2"]
test_s3 = dfs["test_source3"]

print(f"Test S1 entities: {len(test_s1):,}")
print(f"Test S2 entities: {len(test_s2):,}")
print(f"Test S3 entities: {len(test_s3):,}")

print("\nTest country distribution:")
for name in ["test_source1", "test_source2", "test_source3"]:
    print(f"\n  {name}:")
    vc = dfs[name]["country"].value_counts()
    for country, count in vc.items():
        print(f"    {country:20s}  {count:>10,}  ({100*count/len(dfs[name]):5.2f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"AUDIT COMPLETE in {time.time() - t0:.1f}s")
print("=" * 70)
