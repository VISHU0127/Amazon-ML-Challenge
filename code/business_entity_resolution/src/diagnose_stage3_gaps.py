#!/usr/bin/env python3
"""
Optimized diagnostic for Issue 2: Missed True Match Pattern Analysis.
Uses a sampled val set (5000 S1 entities) and builds blocking engines
over a sampled target pool (same as build_feature_matrices.py).
"""

import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from blocking import CountryBlockingEngine, _extract_all_name_tokens
from config import (
    ARTIFACTS_DIR,
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_COUNTRY,
    COL_ENTITY_ID,
    COL_MATCHED_IDS,
    COL_S1_ENTITY_ID,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    VAL_SPLIT_PATH,
    RANDOM_SEED,
)
from normalize import (
    extract_street_number,
    normalize_address,
    normalize_country,
    normalize_name,
)
import jellyfish
from rapidfuzz import distance as rf_dist


def main():
    t0 = time.time()
    rng = np.random.RandomState(RANDOM_SEED)

    print("=" * 70)
    print("ISSUE 2: Missed True Match Pattern Analysis (Optimized)")
    print("=" * 70)

    # Load split
    with open(VAL_SPLIT_PATH, "r") as f:
        split_info = json.load(f)
    val_s1_pool = split_info["val_s1_ids"]

    # Sample same 5000 val entities used in build_feature_matrices.py
    val_s1_sample = set(rng.choice(val_s1_pool, size=min(5000, len(val_s1_pool)), replace=False))

    # Load data
    print("Loading data...")
    s1_df = pd.read_csv(TRAIN_SOURCE1, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(TRAIN_SOURCE2, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(TRAIN_SOURCE3, sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str, keep_default_na=False)

    # S1 dict
    s1_dict = {
        r[COL_ENTITY_ID]: r
        for r in s1_df[s1_df[COL_ENTITY_ID].isin(val_s1_sample)].to_dict("records")
    }

    # GT map for val sample
    gt_map: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()
    for _, row in gt_df[gt_df[COL_S1_ENTITY_ID].isin(val_s1_sample)].iterrows():
        s1_id = row[COL_S1_ENTITY_ID]
        mids_str = row[COL_MATCHED_IDS].strip()
        mids = {m.strip() for m in mids_str.split(",") if m.strip()} if mids_str else set()
        gt_map[s1_id] = mids
        needed_targets.update(mids)

    val_with_matches = {s1_id: mids for s1_id, mids in gt_map.items() if mids}
    print(f"Val S1 with matches: {len(val_with_matches):,}, total true targets needed: {len(needed_targets):,}")

    # Build target pool (same sampling as build_feature_matrices.py to reproduce)
    sample_s2 = s2_df.sample(n=min(500000, len(s2_df)), random_state=RANDOM_SEED)
    sample_s3 = s3_df.sample(n=min(500000, len(s3_df)), random_state=RANDOM_SEED)
    true_s2 = s2_df[s2_df[COL_ENTITY_ID].isin(needed_targets)]
    true_s3 = s3_df[s3_df[COL_ENTITY_ID].isin(needed_targets)]
    target_df = pd.concat([sample_s2, sample_s3, true_s2, true_s3], ignore_index=True).drop_duplicates(subset=[COL_ENTITY_ID])

    target_dict = {
        r[COL_ENTITY_ID]: r
        for r in target_df.to_dict("records")
    }
    print(f"Target pool: {len(target_dict):,}")

    del s2_df, s3_df, sample_s2, sample_s3, true_s2, true_s3
    gc.collect()

    # Build blocking engines by country
    target_df["norm_country"] = target_df[COL_COUNTRY].apply(normalize_country)

    print("Building blocking engines (top-60)...")
    engines_60: Dict[str, CountryBlockingEngine] = {}
    for c in target_df["norm_country"].unique():
        tgt_c = target_df[target_df["norm_country"] == c]
        engine = CountryBlockingEngine(country=c, top_k=60)
        engine.fit(tgt_c, verbose=True)
        engines_60[c] = engine

    print("Building blocking engines (top-200 for unconstrained analysis)...")
    engines_200: Dict[str, CountryBlockingEngine] = {}
    for c in target_df["norm_country"].unique():
        tgt_c = target_df[target_df["norm_country"] == c]
        engine = CountryBlockingEngine(country=c, top_k=200)
        engine.fit(tgt_c, verbose=True)
        engines_200[c] = engine

    # Analyze missed matches
    print(f"\nAnalyzing {len(val_with_matches):,} val S1 entities for missed true matches...")
    missed_pairs = []
    total_true = 0
    found_in_60 = 0
    found_in_200 = 0

    for s1_id in tqdm(val_with_matches, desc="Analyzing"):
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        c_norm = normalize_country(s1_rec[COL_COUNTRY])
        true_mids = val_with_matches[s1_id]

        nn = normalize_name(s1_rec[COL_BUSINESS_NAME])
        na = normalize_address(s1_rec[COL_BUSINESS_ADDRESS])
        sn = extract_street_number(s1_rec[COL_BUSINESS_ADDRESS])

        # Top-60
        cands_60 = set(engines_60[c_norm].query_entity(nn, na, sn)) if c_norm in engines_60 else set()

        # Top-200
        if c_norm in engines_200:
            cands_200_list = engines_200[c_norm].query_entity(nn, na, sn)
            cands_200_set = set(cands_200_list)
            cands_200_rank = {tid: rank for rank, tid in enumerate(cands_200_list, 1)}
        else:
            cands_200_set = set()
            cands_200_rank = {}

        for mid in true_mids:
            total_true += 1
            if mid in cands_60:
                found_in_60 += 1
                continue

            # Missed in top-60
            in_200 = mid in cands_200_set
            if in_200:
                found_in_200 += 1
            rank_200 = cands_200_rank.get(mid, -1)

            if mid not in target_dict:
                continue

            t_rec = target_dict[mid]
            t_nn = normalize_name(t_rec[COL_BUSINESS_NAME])
            t_na = normalize_address(t_rec[COL_BUSINESS_ADDRESS])
            t_cn = normalize_country(t_rec[COL_COUNTRY])

            jw = rf_dist.JaroWinkler.similarity(nn, t_nn)

            # Token overlap
            expanded_s1 = _extract_all_name_tokens(nn)
            expanded_t = _extract_all_name_tokens(t_nn)
            shared_name_toks = expanded_s1 & expanded_t

            name_toks_s1 = set(nn.split())
            name_toks_t = set(t_nn.split())
            tok_jaccard = len(name_toks_s1 & name_toks_t) / len(name_toks_s1 | name_toks_t) if (name_toks_s1 | name_toks_t) else 0

            # Check which strategies found shared keys
            strategy_hits = []

            if shared_name_toks:
                strategy_hits.append("name_token")

            ft1 = nn.split()[0] if nn.split() else ""
            ft2 = t_nn.split()[0] if t_nn.split() else ""
            if ft1 and ft2:
                try:
                    if jellyfish.soundex(ft1) == jellyfish.soundex(ft2):
                        strategy_hits.append("soundex")
                except:
                    pass
            if len(nn) >= 3 and len(t_nn) >= 3 and nn[:3] == t_nn[:3]:
                strategy_hits.append("prefix3")

            atoks_s1 = set(na.split())
            atoks_t = set(t_na.split())
            shared_addr = {t for t in atoks_s1 & atoks_t if len(t) >= 4}
            if shared_addr:
                strategy_hits.append("addr_token")

            missed_pairs.append({
                "s1_id": s1_id,
                "target_id": mid,
                "source": "S2" if mid.startswith("S2") else "S3",
                "country": c_norm,
                "s1_name": s1_rec[COL_BUSINESS_NAME],
                "t_name": t_rec[COL_BUSINESS_NAME],
                "s1_name_norm": nn,
                "t_name_norm": t_nn,
                "jaro_winkler": jw,
                "token_jaccard": tok_jaccard,
                "shared_name_tokens": len(shared_name_toks),
                "strategies_hit": strategy_hits,
                "n_strategies_hit": len(strategy_hits),
                "rank_in_top200": rank_200,
                "found_in_200": in_200,
                "s1_addr": s1_rec[COL_BUSINESS_ADDRESS],
                "t_addr": t_rec[COL_BUSINESS_ADDRESS],
                "country_match": c_norm == t_cn,
            })

    recall_60 = found_in_60 / total_true if total_true > 0 else 0
    missed_total = total_true - found_in_60

    print(f"\n{'=' * 70}")
    print(f"RECALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total true matches: {total_true:,}")
    print(f"Found in top-60:    {found_in_60:,} ({recall_60:.4%})")
    print(f"Missed in top-60:   {missed_total:,}")
    print(f"  Recoverable in top-200: {found_in_200:,}")
    print(f"  Not in top-200 either:  {missed_total - found_in_200:,}")

    if missed_pairs:
        print(f"\n{'=' * 70}")
        print(f"PATTERN ANALYSIS ({len(missed_pairs)} missed pairs)")
        print(f"{'=' * 70}")

        source_counter = Counter(m["source"] for m in missed_pairs)
        print(f"\nBy Source: {dict(source_counter)}")

        country_counter = Counter(m["country"] for m in missed_pairs)
        print(f"By Country: {dict(country_counter)}")

        strat_counter = Counter(m["n_strategies_hit"] for m in missed_pairs)
        print(f"\nBy # strategies with shared blocking keys:")
        for k in sorted(strat_counter.keys()):
            print(f"  {k} strategies: {strat_counter[k]}")

        all_strats = ["name_token", "soundex", "prefix3", "addr_token"]
        print(f"\nStrategy hit rate on MISSED pairs:")
        for s in all_strats:
            hit = sum(1 for m in missed_pairs if s in m["strategies_hit"])
            total = len(missed_pairs)
            print(f"  {s}: {hit}/{total} ({hit/total:.1%}) — these had shared keys but were OUTRANKED")

        zero_name = sum(1 for m in missed_pairs if m["shared_name_tokens"] == 0)
        print(f"\n  Zero shared name tokens (invisible to name strategy): {zero_name}")

        # JW distribution
        jw_vals = [m["jaro_winkler"] for m in missed_pairs]
        print(f"\nJaro-Winkler of missed pairs:")
        print(f"  Mean: {np.mean(jw_vals):.4f}, Median: {np.median(jw_vals):.4f}")
        print(f"  Min: {np.min(jw_vals):.4f}, Max: {np.max(jw_vals):.4f}")

        # In-200 vs not-in-200 split
        in200 = [m for m in missed_pairs if m["found_in_200"]]
        not200 = [m for m in missed_pairs if not m["found_in_200"]]
        print(f"\nRecoverable in top-200 ({len(in200)}):")
        if in200:
            ranks = [m["rank_in_top200"] for m in in200]
            print(f"  Rank distribution: mean={np.mean(ranks):.1f}, median={np.median(ranks):.0f}, "
                  f"min={np.min(ranks)}, max={np.max(ranks)}")
            print(f"  => These are outranked by distractors, not invisible to blocking")

        print(f"\nNot in top-200 ({len(not200)}) — truly invisible to all strategies:")
        if not200:
            jw_not200 = [m["jaro_winkler"] for m in not200]
            print(f"  JW: mean={np.mean(jw_not200):.4f}, median={np.median(jw_not200):.4f}")
            strat_not200 = Counter(m["n_strategies_hit"] for m in not200)
            print(f"  By strategies: {dict(strat_not200)}")

        # Print top 20 examples
        print(f"\n{'=' * 70}")
        print(f"TOP 20 MISSED PAIR EXAMPLES (sorted by JW desc)")
        print(f"{'=' * 70}")
        missed_sorted = sorted(missed_pairs, key=lambda x: x["jaro_winkler"], reverse=True)
        for i, m in enumerate(missed_sorted[:20]):
            r200 = f"rank {m['rank_in_top200']}" if m['found_in_200'] else "NOT IN TOP-200"
            print(f"\n  [{i+1}] [{m['source']}/{m['country']}] JW={m['jaro_winkler']:.3f} "
                  f"TokJacc={m['token_jaccard']:.3f} Strats={m['strategies_hit']} "
                  f"200={r200}")
            print(f"      S1:  '{m['s1_name']}' | Addr: '{m['s1_addr']}'")
            print(f"      Tgt: '{m['t_name']}' | Addr: '{m['t_addr']}'")

        # Print 10 examples NOT in top-200 (hardest cases)
        if not200:
            print(f"\n{'=' * 70}")
            print(f"TOP 10 HARDEST MISSES (not in top-200)")
            print(f"{'=' * 70}")
            not200_sorted = sorted(not200, key=lambda x: x["jaro_winkler"], reverse=True)
            for i, m in enumerate(not200_sorted[:10]):
                print(f"\n  [{i+1}] [{m['source']}/{m['country']}] JW={m['jaro_winkler']:.3f} "
                      f"TokJacc={m['token_jaccard']:.3f} Strats={m['strategies_hit']}")
                print(f"      S1:  '{m['s1_name']}' | Norm: '{m['s1_name_norm']}'")
                print(f"      Tgt: '{m['t_name']}' | Norm: '{m['t_name_norm']}'")
                print(f"      S1 Addr: '{m['s1_addr']}'")
                print(f"      Tgt Addr: '{m['t_addr']}'")

    # Save results
    results = {
        "total_true": total_true,
        "found_in_60": found_in_60,
        "recall_60": round(recall_60, 6),
        "missed_total": missed_total,
        "recovered_in_200": found_in_200,
        "not_in_200": missed_total - found_in_200,
        "by_source": dict(Counter(m["source"] for m in missed_pairs)),
        "by_country": dict(Counter(m["country"] for m in missed_pairs)),
        "mean_jw_missed": round(float(np.mean([m["jaro_winkler"] for m in missed_pairs])), 4) if missed_pairs else None,
    }
    out_path = os.path.join(ARTIFACTS_DIR, "stage3_issue2_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
