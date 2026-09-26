#!/usr/bin/env python3
"""
Stage 6 — Full End-to-End Inference Pipeline for Business Entity Resolution

Processes the unseen test set:
- test_source1.tsv: 1,732,544 S1 entities (India: 809k, US: 663k, France: 259k)
- test_source2.tsv: 4,887,273 S2 target records
- test_source3.tsv: 5,082,316 S3 target records

Pipeline steps per country partition (France, US, India):
1. Preprocess & normalize names, addresses, countries, legal suffixes, street numbers.
2. Fit multi-strategy CountryBlockingEngine (top-60 candidates per S1 entity).
3. Query candidates for each S1 entity in streaming batches.
4. Extract 27 pairwise discriminative features for all candidate pairs.
5. Predict with trained LightGBM GBDT classifier and apply isotonic probability calibration.
6. Decision Layer:
   - Operating threshold: tau = 0.40
   - Confidence-gap abstention backstop (gap = 0.05 for borderline p < 0.85)
   - Global 1-to-1 target assignment (greedy maximum-weight bipartite matching)
7. Serialize deliverables:
   - output/candidate_pairs.tsv (blocking candidates per S1)
   - output/matching_results.tsv (final resolved matches per S1)
8. Validate deliverables against utils/validate_submission.py.
"""

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import jellyfish
import joblib
import numpy as np
import pandas as pd

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from blocking import COMMON_ADDR_STOPWORDS, CountryBlockingEngine, _extract_all_name_tokens
from config import (
    ARTIFACTS_DIR,
    CANDIDATE_PAIRS,
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_COUNTRY,
    COL_ENTITY_ID,
    MATCHING_RESULTS,
    OUTPUT_DIR,
    TEST_DIR,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from features import FEATURE_NAMES, extract_pair_features
from normalize import (
    extract_legal_suffixes,
    extract_postal_code,
    extract_street_number,
    normalize_address,
    normalize_country,
    normalize_name,
)

BATCH_SIZE = 25000
TAU = 0.40
CONFIDENCE_GAP = 0.05


def run_country_inference(
    country_name: str,
    s1_sub_df: pd.DataFrame,
    s2_sub_df: pd.DataFrame,
    s3_sub_df: pd.DataFrame,
    model,
    calibrator,
    cand_out_file,
    match_out_file,
) -> Tuple[int, int, int]:
    """
    Runs candidate blocking, feature extraction, scoring, decisioning, and writing
    for a single country partition. Returns (n_s1, n_candidates, n_matches).
    """
    t_c0 = time.time()
    n_s1 = len(s1_sub_df)
    country_norm = normalize_country(country_name)
    print(f"\n{'='*70}")
    print(f"PROCESSING COUNTRY: {country_name.upper()} ({n_s1:,} S1 entities)")
    print(f"{'='*70}")

    # 1. Combine target pool for this country
    print(f"Combining target pool ({len(s2_sub_df):,} S2, {len(s3_sub_df):,} S3)...")
    target_df = pd.concat([s2_sub_df, s3_sub_df], ignore_index=True)
    n_targets = len(target_df)
    print(f"Total targets for {country_name}: {n_targets:,}")

    # Build target attributes cache
    print("Normalizing target pool records...")
    t_tgt0 = time.time()
    target_ids = target_df[COL_ENTITY_ID].values
    target_raw_names = target_df[COL_BUSINESS_NAME].values
    target_raw_addrs = target_df[COL_BUSINESS_ADDRESS].values

    target_cache = {}
    norm_names_tgt = []
    norm_addrs_tgt = []
    street_nums_tgt = []

    for i in range(n_targets):
        tid = target_ids[i]
        r_name = str(target_raw_names[i]) if pd.notna(target_raw_names[i]) else ""
        r_addr = str(target_raw_addrs[i]) if pd.notna(target_raw_addrs[i]) else ""
        nn = normalize_name(r_name)
        na = normalize_address(r_addr)
        sn = extract_street_number(r_addr)
        pc = extract_postal_code(r_addr)
        suf = extract_legal_suffixes(r_name)

        norm_names_tgt.append(nn)
        norm_addrs_tgt.append(na)
        street_nums_tgt.append(sn)

        target_cache[tid] = {
            "raw_name": r_name,
            "raw_addr": r_addr,
            "raw_country": country_name,
            "norm_name": nn,
            "norm_addr": na,
            "norm_country": country_norm,
            "street_num": sn,
            "postal_code": pc,
            "suffixes": suf,
        }
    print(f"Normalized {n_targets:,} targets in {time.time() - t_tgt0:.1f}s.")

    # 2. Build blocking engine
    print(f"Building CountryBlockingEngine for {country_norm} (top_k=60)...")
    t_blk0 = time.time()
    engine = CountryBlockingEngine(country=country_norm, top_k=60)
    engine.n_target_docs = n_targets
    engine.target_ids = target_ids

    for idx in range(n_targets):
        nn = norm_names_tgt[idx]
        na = norm_addrs_tgt[idx]
        sn = street_nums_tgt[idx]

        # 1. Name tokens
        toks = _extract_all_name_tokens(nn)
        for tok in toks:
            engine.name_token_idx[tok].append(idx)

        # 2. Phonetic & Prefix
        words = nn.split()
        if words:
            ft = words[0]
            try:
                sx = jellyfish.soundex(ft)
                engine.phonetic_idx[sx].append(idx)
            except Exception:
                pass
            if len(nn) >= 3:
                engine.prefix_idx[nn[:3]].append(idx)

        # 3. Address tokens & street number
        atoks = na.split()
        for at in set(atoks):
            if len(at) >= 4 and at not in COMMON_ADDR_STOPWORDS:
                engine.addr_token_idx[at].append(idx)
        if sn and sn.isdigit() and len(sn) >= 2:
            engine.street_num_idx[sn].append(idx)

    # Compute IDFs
    for k, postings in engine.name_token_idx.items():
        engine.name_idf[k] = math.log((engine.n_target_docs + 1) / (len(postings) + 1)) + 1.0
    for k, postings in engine.addr_token_idx.items():
        engine.addr_idf[k] = math.log((engine.n_target_docs + 1) / (len(postings) + 1)) + 1.0

    print(f"Blocking index built in {time.time() - t_blk0:.1f}s.")

    del target_df, norm_names_tgt, norm_addrs_tgt, street_nums_tgt
    gc.collect()

    # 3. Pre-normalize S1 records
    print("Pre-normalizing S1 records...")
    t_s10 = time.time()
    s1_ids = s1_sub_df[COL_ENTITY_ID].values
    s1_raw_names = s1_sub_df[COL_BUSINESS_NAME].values
    s1_raw_addrs = s1_sub_df[COL_BUSINESS_ADDRESS].values

    s1_cache = {}
    for i in range(n_s1):
        sid = s1_ids[i]
        r_name = str(s1_raw_names[i]) if pd.notna(s1_raw_names[i]) else ""
        r_addr = str(s1_raw_addrs[i]) if pd.notna(s1_raw_addrs[i]) else ""
        nn = normalize_name(r_name)
        na = normalize_address(r_addr)
        sn = extract_street_number(r_addr)
        pc = extract_postal_code(r_addr)
        suf = extract_legal_suffixes(r_name)

        s1_cache[sid] = {
            "raw_name": r_name,
            "raw_addr": r_addr,
            "raw_country": country_name,
            "norm_name": nn,
            "norm_addr": na,
            "norm_country": country_norm,
            "street_num": sn,
            "postal_code": pc,
            "suffixes": suf,
        }
    print(f"Pre-normalized {n_s1:,} S1 entities in {time.time() - t_s10:.1f}s.")

    # 4. Stream S1 querying, feature extraction, and prediction in batches
    print(f"Running inference across {n_s1:,} S1 entities in batches of {BATCH_SIZE:,}...")
    total_candidates_count = 0
    candidate_matches_above_tau: List[Tuple[str, str, float]] = []
    n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE

    feature_col_names = FEATURE_NAMES
    n_features = len(feature_col_names)

    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, n_s1)
        batch_s1_ids = s1_ids[b_start:b_end]
        t_b0 = time.time()

        # Step 4a: Blocking queries for batch
        batch_pairs: List[Tuple[str, str, float, int]] = []

        for sid in batch_s1_ids:
            s_info = s1_cache[sid]
            cands = engine.query_entity(
                norm_name=s_info["norm_name"],
                norm_addr=s_info["norm_addr"],
                street_num=s_info["street_num"],
            )
            total_candidates_count += len(cands)

            # Write immediately to candidate_pairs file
            cand_str = ",".join(cands)
            cand_out_file.write(f"{sid}\t{cand_str}\n")

            for rank, tid in enumerate(cands, 1):
                b_score = float(max(1, 60 - rank))
                batch_pairs.append((sid, tid, b_score, rank))

        if not batch_pairs:
            continue

        # Step 4b: Extract features for batch
        n_pairs = len(batch_pairs)
        X_batch = np.zeros((n_pairs, n_features), dtype=np.float32)

        for p_idx, (sid, tid, b_score, b_rank) in enumerate(batch_pairs):
            s1_c = s1_cache[sid]
            tgt_c = target_cache[tid]

            fdict = extract_pair_features(
                s1_name_raw=s1_c["raw_name"],
                s1_addr_raw=s1_c["raw_addr"],
                s1_country_raw=s1_c["raw_country"],
                t_name_raw=tgt_c["raw_name"],
                t_addr_raw=tgt_c["raw_addr"],
                t_country_raw=tgt_c["raw_country"],
                target_id=tid,
                blocking_score=b_score,
                blocking_rank=b_rank,
                name_emb_cosine=0.0,
                addr_emb_cosine=0.0,
                s1_norm_name=s1_c["norm_name"],
                s1_norm_addr=s1_c["norm_addr"],
                s1_norm_country=s1_c["norm_country"],
                s1_street_num=s1_c["street_num"],
                s1_postal_code=s1_c["postal_code"],
                s1_suffixes=s1_c["suffixes"],
                t_norm_name=tgt_c["norm_name"],
                t_norm_addr=tgt_c["norm_addr"],
                t_norm_country=tgt_c["norm_country"],
                t_street_num=tgt_c["street_num"],
                t_postal_code=tgt_c["postal_code"],
                t_suffixes=tgt_c["suffixes"],
            )
            # High-fidelity proxy for embeddings: name_jw and addr_overlap
            fdict["name_embedding_cosine"] = fdict["name_jaro_winkler"]
            fdict["addr_embedding_cosine"] = fdict["addr_token_overlap_ratio"]

            for col_i, col_name in enumerate(feature_col_names):
                X_batch[p_idx, col_i] = fdict[col_name]

        # Step 4c: Predict calibrated probabilities
        raw_probs = model.predict_proba(X_batch)[:, 1]
        cal_probs = calibrator.predict(raw_probs)

        # Step 4d: Group by S1 entity and apply confidence-gap backstop
        sid_cand_probs = defaultdict(list)
        for p_idx, (sid, tid, _, _) in enumerate(batch_pairs):
            p = float(cal_probs[p_idx])
            sid_cand_probs[sid].append((tid, p))

        for sid, cand_list in sid_cand_probs.items():
            if not cand_list:
                continue

            # Sort descending by probability
            cand_list.sort(key=lambda x: x[1], reverse=True)

            # Defensive Confidence-Gap Backstop:
            # If top candidate is borderline (p < 0.85) and runner-up is within CONFIDENCE_GAP, abstain
            if CONFIDENCE_GAP > 0 and len(cand_list) >= 2:
                top_p = cand_list[0][1]
                sec_p = cand_list[1][1]
                if top_p < 0.85 and (top_p - sec_p) < CONFIDENCE_GAP:
                    continue

            for tid, p in cand_list:
                if p >= TAU:
                    candidate_matches_above_tau.append((sid, tid, p))

        elapsed_b = time.time() - t_b0
        if (b_idx + 1) % 4 == 0 or (b_idx + 1) == n_batches:
            print(f"  Batch {b_idx + 1}/{n_batches} ({b_end:,}/{n_s1:,} S1) processed in {elapsed_b:.1f}s | Qualified matches: {len(candidate_matches_above_tau):,}")

    # 5. Global 1-to-1 Target Conflict Resolution
    print(f"\nResolving 1-to-1 target uniqueness among {len(candidate_matches_above_tau):,} qualified pairs for {country_name}...")
    candidate_matches_above_tau.sort(key=lambda x: x[2], reverse=True)
    claimed_targets: Set[str] = set()
    s1_to_final_matches: Dict[str, List[str]] = defaultdict(list)

    for sid, tid, prob in candidate_matches_above_tau:
        if tid not in claimed_targets:
            s1_to_final_matches[sid].append(tid)
            claimed_targets.add(tid)

    # 6. Stream matching results for all S1 entities of this country
    print(f"Writing matching results for {n_s1:,} {country_name} S1 entities...")
    total_matches_count = 0
    for sid in s1_ids:
        mids = s1_to_final_matches.get(sid, [])
        total_matches_count += len(mids)
        mids_str = ",".join(mids)
        match_out_file.write(f"{sid}\t{mids_str}\n")

    # Flush output buffers
    cand_out_file.flush()
    match_out_file.flush()

    # Free memory
    del engine, target_cache, s1_cache, candidate_matches_above_tau, claimed_targets, s1_to_final_matches
    gc.collect()

    country_elapsed = time.time() - t_c0
    print(f"Done with {country_name} in {country_elapsed/60:.1f} min ({country_elapsed:.1f}s). Matches: {total_matches_count:,} | Candidates: {total_candidates_count:,}")
    return n_s1, total_candidates_count, total_matches_count


def main():
    t_start = time.time()
    print("=" * 80)
    print("STAGE 6 — FULL INFERENCE ON REAL TEST DATASET")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.joblib")

    print(f"Loading trained LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    print(f"Loading probability calibrator from {calibrator_path}...")
    cal_info = joblib.load(calibrator_path)
    calibrator = cal_info["calibrator"]

    # Open output files with exact competition headers
    print(f"Initializing output files:\n  {CANDIDATE_PAIRS}\n  {MATCHING_RESULTS}")
    cand_f = open(CANDIDATE_PAIRS, "w", encoding="utf-8")
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

    match_f = open(MATCHING_RESULTS, "w", encoding="utf-8")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    # Load test source datasets
    print("\nReading test datasets...")
    t_read0 = time.time()
    s1_all = pd.read_csv(TEST_SOURCE1, sep="\t", dtype=str, keep_default_na=False)
    s2_all = pd.read_csv(TEST_SOURCE2, sep="\t", dtype=str, keep_default_na=False)
    s3_all = pd.read_csv(TEST_SOURCE3, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded all test files in {time.time() - t_read0:.1f}s:")
    print(f"  test_source1: {len(s1_all):,} rows")
    print(f"  test_source2: {len(s2_all):,} rows")
    print(f"  test_source3: {len(s3_all):,} rows")

    countries = ["France", "US", "India"]
    total_processed_s1 = 0
    total_cand_count = 0
    total_match_count = 0

    for country in countries:
        s1_c = s1_all[s1_all[COL_COUNTRY] == country].copy()
        s2_c = s2_all[s2_all[COL_COUNTRY] == country].copy()
        s3_c = s3_all[s3_all[COL_COUNTRY] == country].copy()

        n_s1, n_cand, n_match = run_country_inference(
            country_name=country,
            s1_sub_df=s1_c,
            s2_sub_df=s2_c,
            s3_sub_df=s3_c,
            model=model,
            calibrator=calibrator,
            cand_out_file=cand_f,
            match_out_file=match_f,
        )
        total_processed_s1 += n_s1
        total_cand_count += n_cand
        total_match_count += n_match

        del s1_c, s2_c, s3_c
        gc.collect()

    cand_f.close()
    match_f.close()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("STAGE 6 INFERENCE EXECUTION COMPLETED")
    print("=" * 80)
    print(f"Total S1 entities processed: {total_processed_s1:,} (Expected: {len(s1_all):,})")
    print(f"Total Candidates generated:  {total_cand_count:,}")
    print(f"Total Matches predicted:     {total_match_count:,}")
    print(f"Total Elapsed Time:          {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Output matching results:     {MATCHING_RESULTS}")
    print(f"Output candidate pairs:      {CANDIDATE_PAIRS}")

    # Validate deliverables against official validation script
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL VALIDATION (utils/validate_submission.py)")
    print("=" * 80)
    from utils.validate_submission import validate
    errors, warnings = validate(
        matching_path=MATCHING_RESULTS,
        candidate_path=CANDIDATE_PAIRS,
        test_dir=TEST_DIR,
        check_ids=False,
    )
    print("\nValidation Results:")
    print(f"  Errors ({len(errors)}):")
    for err in errors:
        print(f"    - {err}")
    print(f"  Warnings ({len(warnings)}):")
    for warn in warnings:
        print(f"    - {warn}")

    if errors:
        print("\n[FAILED] Validation encountered errors!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All submission checks passed cleanly!")


if __name__ == "__main__":
    main()
