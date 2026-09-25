#!/usr/bin/env python3
"""
Stage 2 — Run Blocking Evaluation & Generate Candidates

Runs multi-strategy blocking on the validation split (and training data),
measures Candidate Recall and Reduction Ratio against Ground Truth,
saves candidate pairs, and verifies schema validity.
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import evaluate_blocking_recall, generate_candidates_by_country, save_candidate_pairs
from config import (
    ARTIFACTS_DIR,
    COL_ENTITY_ID,
    COL_S1_ENTITY_ID,
    OUTPUT_DIR,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    VAL_SPLIT_PATH,
)


def run_stage2_evaluation(sample_size: int = 25000, top_k: int = 60):
    print("=" * 70)
    print("STAGE 2 — BLOCKING / CANDIDATE GENERATION EVALUATION")
    print("=" * 70)
    t0 = time.time()

    # 1. Load validation split
    if os.path.exists(VAL_SPLIT_PATH):
        with open(VAL_SPLIT_PATH, "r") as f:
            split_info = json.load(f)
        val_s1_ids = set(split_info.get("val_s1_ids", []))
        print(f"Loaded validation split: {len(val_s1_ids):,} S1 IDs")
    else:
        val_s1_ids = set()

    # 2. Load S1 data
    print("Loading Source 1...")
    s1_all = pd.read_csv(TRAIN_SOURCE1, sep="	", dtype=str, keep_default_na=False)

    if val_s1_ids:
        s1_eval = s1_all[s1_all[COL_ENTITY_ID].isin(val_s1_ids)].copy()
        if len(s1_eval) > sample_size:
            s1_eval = s1_eval.sample(n=sample_size, random_state=42)
        eval_label = f"Validation Split ({len(s1_eval):,} entities)"
    else:
        s1_eval = s1_all.sample(n=sample_size, random_state=42)
        eval_label = f"Training Random Sample ({len(s1_eval):,} entities)"

    eval_s1_ids = set(s1_eval[COL_ENTITY_ID])
    print(f"Evaluation set: {eval_label}")

    # 3. Load Ground Truth for evaluation set
    print("Loading Ground Truth...")
    gt_all = pd.read_csv(TRAIN_GROUND_TRUTH, sep="	", dtype=str, keep_default_na=False)
    gt_eval = gt_all[gt_all[COL_S1_ENTITY_ID].isin(eval_s1_ids)].copy()

    needed_true_targets = set()
    for mids in gt_eval["matched_entity_ids"]:
        for m in mids.split(","):
            if m.strip():
                needed_true_targets.add(m.strip())
    total_true_matches = sum(len(m.split(",")) for m in gt_eval["matched_entity_ids"] if m.strip())
    print(f"Total true match pairs in eval set: {total_true_matches:,}")
    print(f"Distinct target entities in ground truth: {len(needed_true_targets):,}")

    # 4. Load Target Sources (S2 + S3)
    print("Loading Target Sources (S2 + S3)...")
    s2_full = pd.read_csv(TRAIN_SOURCE2, sep="	", dtype=str, keep_default_na=False)
    s3_full = pd.read_csv(TRAIN_SOURCE3, sep="	", dtype=str, keep_default_na=False)
    total_target_population = len(s2_full) + len(s3_full)
    print(f"Total target population: {total_target_population:,} (S2: {len(s2_full):,}, S3: {len(s3_full):,})")

    sample_background_s2 = s2_full.sample(n=min(600000, len(s2_full)), random_state=42)
    sample_background_s3 = s3_full.sample(n=min(600000, len(s3_full)), random_state=42)
    missing_true_s2 = s2_full[s2_full[COL_ENTITY_ID].isin(needed_true_targets)]
    missing_true_s3 = s3_full[s3_full[COL_ENTITY_ID].isin(needed_true_targets)]

    target_eval = pd.concat([
        sample_background_s2, sample_background_s3,
        missing_true_s2, missing_true_s3
    ], ignore_index=True).drop_duplicates(subset=[COL_ENTITY_ID])
    print(f"Evaluated target pool: {len(target_eval):,} target entities (includes 100% of true targets + 1.2M background distractors)")

    del s2_full, s3_full, sample_background_s2, sample_background_s3, missing_true_s2, missing_true_s3
    import gc
    gc.collect()

    # 5. Run Blocking
    candidates_map = generate_candidates_by_country(
        s1_df=s1_eval,
        target_df=target_eval,
        top_k=top_k,
        verbose=True,
    )

    # 6. Evaluate Blocking Recall & Reduction Ratio
    metrics = evaluate_blocking_recall(
        candidates_map=candidates_map,
        gt_df=gt_eval,
        total_target_entities=total_target_population,
    )

    print("=" * 70)
    print("BLOCKING EVALUATION RESULTS")
    print("=" * 70)
    print(f"Evaluated S1 Entities:     {len(candidates_map):,}")
    print(f"Candidate top-K cap:       {top_k}")
    print(f"Total Candidates:          {metrics['total_candidates']:,}")
    print(f"Avg Candidates / S1:       {metrics['avg_candidates_per_s1']:.2f}")
    print(f"Total True Matches:        {metrics['total_true_matches']:,}")
    print(f"Found True Matches:        {metrics['found_true_matches']:,}")
    print(f"Candidate Recall:          {metrics['candidate_recall'] * 100:.2f}%")
    print(f"Full Cross Product:        {metrics['full_cross_product']:.2e}")
    print(f"Reduction Ratio:           {metrics['reduction_ratio'] * 100:.6f}%")
    print(f"Total Time:                {time.time() - t0:.1f}s")
    print("=" * 70)

    # 7. Save candidate pairs to artifacts
    cand_path = os.path.join(ARTIFACTS_DIR, "candidate_pairs_eval.tsv")
    save_candidate_pairs(candidates_map, cand_path)

    # 8. Verify candidate file format
    print("Verifying TSV schema...")
    with open(cand_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("	")
        first_row = f.readline().strip().split("	")
    print(f"Header: {header}")
    assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid header: {header}"
    print(f"First row S1: {first_row[0]}, Candidates count: {len(first_row[1].split(',')) if len(first_row)>1 and first_row[1] else 0}")
    print("TSV schema verification: PASS")

    metrics_path = os.path.join(ARTIFACTS_DIR, "blocking_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


if __name__ == "__main__":
    sample_size = 25000
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
    run_stage2_evaluation(sample_size=sample_size, top_k=60)
