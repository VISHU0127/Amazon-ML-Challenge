#!/usr/bin/env python3
"""
Stage 3 — Build and Cache Training & Validation Feature Matrices
Includes Multilingual Sentence Transformer Embeddings:
- Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- License: Apache-2.0
- Parameters: 117.7M (<= 8B)
- Features: 27 total (name & address lexical, suffix equality, semantic embeddings, blocking rank/scores)
"""

import gc
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from blocking import CountryBlockingEngine
from config import (
    ARTIFACTS_DIR,
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_COUNTRY,
    COL_ENTITY_ID,
    COL_MATCHED_IDS,
    COL_S1_ENTITY_ID,
    RANDOM_SEED,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    VAL_SPLIT_PATH,
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

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_LICENSE = "Apache-2.0"
EMBEDDING_MODEL_PARAMS = 117653760


def build_and_cache_datasets(
    n_train_s1: int = 15000,
    n_val_s1: int = 5000,
    neg_to_pos_ratio: int = 4,
    random_seed: int = 42,
):
    print("=" * 70)
    print("STAGE 3 — BUILD & CACHE FEATURE MATRICES (WITH MULTILINGUAL EMBEDDINGS)")
    print("=" * 70)
    t0 = time.time()
    rng = np.random.RandomState(random_seed)

    # 1. Load validation split
    with open(VAL_SPLIT_PATH, "r") as f:
        split_info = json.load(f)
    train_s1_pool = split_info["train_s1_ids"]
    val_s1_pool = split_info["val_s1_ids"]
    print(f"Loaded split: {len(train_s1_pool):,} train S1, {len(val_s1_pool):,} val S1")

    train_s1_sample = set(rng.choice(train_s1_pool, size=min(n_train_s1, len(train_s1_pool)), replace=False))
    val_s1_sample = set(rng.choice(val_s1_pool, size=min(n_val_s1, len(val_s1_pool)), replace=False))
    all_sampled_s1 = train_s1_sample | val_s1_sample
    print(f"Sampled {len(train_s1_sample):,} train S1 entities and {len(val_s1_sample):,} val S1 entities.")

    # 2. Load Ground Truth
    print("Loading ground truth...")
    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="	", dtype=str, keep_default_na=False)
    gt_map: Dict[str, Set[str]] = {}
    needed_true_targets: Set[str] = set()

    for _, row in gt[gt[COL_S1_ENTITY_ID].isin(all_sampled_s1)].iterrows():
        s1_id = row[COL_S1_ENTITY_ID]
        mids = {m.strip() for m in row[COL_MATCHED_IDS].split(",") if m.strip()}
        gt_map[s1_id] = mids
        needed_true_targets.update(mids)
    print(f"Target IDs in ground truth for sampled S1: {len(needed_true_targets):,}")

    # 3. Load S1 data
    print("Loading Source 1...")
    s1_df = pd.read_csv(TRAIN_SOURCE1, sep="	", dtype=str, keep_default_na=False)
    s1_dict = {
        r[COL_ENTITY_ID]: (r[COL_BUSINESS_NAME], r[COL_BUSINESS_ADDRESS], r[COL_COUNTRY])
        for r in s1_df[s1_df[COL_ENTITY_ID].isin(all_sampled_s1)].to_dict("records")
    }

    # 4. Load Target Sources
    print("Loading Target Sources (S2 + S3)...")
    s2_full = pd.read_csv(TRAIN_SOURCE2, sep="	", dtype=str, keep_default_na=False)
    s3_full = pd.read_csv(TRAIN_SOURCE3, sep="	", dtype=str, keep_default_na=False)

    sample_s2 = s2_full.sample(n=min(500000, len(s2_full)), random_state=random_seed)
    sample_s3 = s3_full.sample(n=min(500000, len(s3_full)), random_state=random_seed)
    true_s2 = s2_full[s2_full[COL_ENTITY_ID].isin(needed_true_targets)]
    true_s3 = s3_full[s3_full[COL_ENTITY_ID].isin(needed_true_targets)]

    target_df = pd.concat([sample_s2, sample_s3, true_s2, true_s3], ignore_index=True).drop_duplicates(subset=[COL_ENTITY_ID])
    target_dict = {
        r[COL_ENTITY_ID]: (r[COL_BUSINESS_NAME], r[COL_BUSINESS_ADDRESS], r[COL_COUNTRY])
        for r in target_df.to_dict("records")
    }
    print(f"Target pool constructed: {len(target_dict):,} targets")

    del s2_full, s3_full, sample_s2, sample_s3, true_s2, true_s3
    gc.collect()

    # 5. Build corrected blocking indices by country (top-60 cap, token expansion)
    print("Building corrected blocking indices (top_k=60)...")
    target_df["norm_country"] = target_df[COL_COUNTRY].apply(normalize_country)
    engines: Dict[str, CountryBlockingEngine] = {}

    for c in target_df["norm_country"].unique():
        tgt_c = target_df[target_df["norm_country"] == c]
        engine = CountryBlockingEngine(country=c, top_k=60)
        engine.fit(tgt_c, verbose=False)
        engines[c] = engine

    # 6. Generate candidate pairs with labels
    def create_pairs_for_s1_set(s1_id_set: Set[str], desc: str) -> Tuple[List[Tuple[str, str, float, int]], List[int]]:
        pairs: List[Tuple[str, str, float, int]] = []
        labels: List[int] = []
        print(f"Generating candidate pairs & labels for {desc} ({len(s1_id_set):,} entities)...")

        for s1_id in s1_id_set:
            if s1_id not in s1_dict:
                continue
            name, addr, country = s1_dict[s1_id]
            c_norm = normalize_country(country)
            true_mids = gt_map.get(s1_id, set())

            if c_norm not in engines:
                continue
            engine = engines[c_norm]

            nn = normalize_name(name)
            na = normalize_address(addr)
            sn = extract_street_number(addr)

            cands = engine.query_entity(norm_name=nn, norm_addr=na, street_num=sn)
            cands_set = set(cands)

            for rank, tid in enumerate(cands, 1):
                if tid in true_mids:
                    pairs.append((s1_id, tid, float(max(1, 60 - rank)), rank))
                    labels.append(1)

            for tid in true_mids:
                if tid not in cands_set and tid in target_dict:
                    pairs.append((s1_id, tid, 0.0, 61))
                    labels.append(1)

            hard_negs = [tid for tid in cands if tid not in true_mids]
            n_pos = len(true_mids)
            n_negs_to_take = max(3, min(len(hard_negs), n_pos * neg_to_pos_ratio))

            for rank, tid in enumerate(hard_negs[:n_negs_to_take], 1):
                pairs.append((s1_id, tid, float(max(1, 60 - rank)), rank))
                labels.append(0)

        return pairs, labels

    train_pairs, train_labels = create_pairs_for_s1_set(train_s1_sample, "Training Split")
    val_pairs, val_labels = create_pairs_for_s1_set(val_s1_sample, "Validation Split")

    print(f"Train pairs: {len(train_pairs):,} (Pos: {sum(train_labels):,}, Neg: {len(train_labels)-sum(train_labels):,})")
    print(f"Val pairs:   {len(val_pairs):,} (Pos: {sum(val_labels):,}, Neg: {len(val_labels)-sum(val_labels):,})")

    # 7. Pre-compute Multilingual Sentence Embeddings on Unique Strings
    all_pairs = train_pairs + val_pairs
    unique_s1_ids = {p[0] for p in all_pairs}
    unique_tgt_ids = {p[1] for p in all_pairs}

    # Collect unique names and addresses
    unique_names = set()
    unique_addrs = set()
    for sid in unique_s1_ids:
        n, a, _ = s1_dict[sid]
        nn = normalize_name(n)
        na = normalize_address(a)
        if nn: unique_names.add(nn)
        if na: unique_addrs.add(na)
    for tid in unique_tgt_ids:
        n, a, _ = target_dict[tid]
        nn = normalize_name(n)
        na = normalize_address(a)
        if nn: unique_names.add(nn)
        if na: unique_addrs.add(na)

    print(f"Computing Multilingual Embeddings for {len(unique_names):,} unique names and {len(unique_addrs):,} unique addresses...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    t_emb0 = time.time()
    name_list = list(unique_names)
    addr_list = list(unique_addrs)

    name_embs = emb_model.encode(name_list, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
    name_emb_map = {name_list[i]: name_embs[i] for i in range(len(name_list))}

    addr_embs = emb_model.encode(addr_list, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
    addr_emb_map = {addr_list[i]: addr_embs[i] for i in range(len(addr_list))}
    print(f"Embeddings generated in {time.time() - t_emb0:.1f}s on {device}.")

    # Free embedding model from memory
    del emb_model, name_embs, addr_embs
    gc.collect()

    # 8. Feature Matrix Extraction function with embeddings
    def build_matrix(pairs, labels, desc):
        print(f"Extracting 27 features for {desc} ({len(pairs):,} pairs)...")
        n_pairs = len(pairs)
        n_features = len(FEATURE_NAMES)
        X = np.zeros((n_pairs, n_features), dtype=np.float32)

        # Pre-cache S1 attributes
        s1_cache = {}
        for sid in {p[0] for p in pairs}:
            name, addr, country = s1_dict[sid]
            nn = normalize_name(name)
            na = normalize_address(addr)
            s1_cache[sid] = {
                "raw_name": name, "raw_addr": addr, "raw_country": country,
                "norm_name": nn, "norm_addr": na, "norm_country": normalize_country(country),
                "street_num": extract_street_number(addr),
                "postal_code": extract_postal_code(addr),
                "suffixes": extract_legal_suffixes(name),
                "name_emb": name_emb_map.get(nn),
                "addr_emb": addr_emb_map.get(na),
            }

        # Pre-cache Target attributes
        tgt_cache = {}
        for tid in {p[1] for p in pairs}:
            name, addr, country = target_dict[tid]
            nn = normalize_name(name)
            na = normalize_address(addr)
            tgt_cache[tid] = {
                "raw_name": name, "raw_addr": addr, "raw_country": country,
                "norm_name": nn, "norm_addr": na, "norm_country": normalize_country(country),
                "street_num": extract_street_number(addr),
                "postal_code": extract_postal_code(addr),
                "suffixes": extract_legal_suffixes(name),
                "name_emb": name_emb_map.get(nn),
                "addr_emb": addr_emb_map.get(na),
            }

        for i, (sid, tid, b_score, b_rank) in enumerate(pairs):
            s1_c = s1_cache[sid]
            tgt_c = tgt_cache[tid]

            # Cosine similarities
            name_cos = 0.0
            if s1_c["name_emb"] is not None and tgt_c["name_emb"] is not None:
                name_cos = float(np.dot(s1_c["name_emb"], tgt_c["name_emb"]))

            addr_cos = 0.0
            if s1_c["addr_emb"] is not None and tgt_c["addr_emb"] is not None:
                addr_cos = float(np.dot(s1_c["addr_emb"], tgt_c["addr_emb"]))

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
                name_emb_cosine=name_cos,
                addr_emb_cosine=addr_cos,
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

            for j, fn in enumerate(FEATURE_NAMES):
                val = fdict[fn]
                # Ensure never NaN or Inf
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                X[i, j] = val

        y = np.array(labels, dtype=np.int32)
        return X, y

    X_train, y_train = build_matrix(train_pairs, train_labels, "Training Split")
    X_val, y_val = build_matrix(val_pairs, val_labels, "Validation Split")

    # 9. Save matrices
    train_path = os.path.join(ARTIFACTS_DIR, "train_features.npz")
    val_path = os.path.join(ARTIFACTS_DIR, "val_features.npz")

    train_s1_arr = np.array([p[0] for p in train_pairs])
    train_tgt_arr = np.array([p[1] for p in train_pairs])
    val_s1_arr = np.array([p[0] for p in val_pairs])
    val_tgt_arr = np.array([p[1] for p in val_pairs])

    np.savez_compressed(
        train_path,
        X=X_train,
        y=y_train,
        feature_names=np.array(FEATURE_NAMES),
        s1_ids=train_s1_arr,
        target_ids=train_tgt_arr,
    )
    print(f"Saved training features to {train_path} ({os.path.getsize(train_path) / 1024**2:.1f} MB)")

    np.savez_compressed(
        val_path,
        X=X_val,
        y=y_val,
        feature_names=np.array(FEATURE_NAMES),
        s1_ids=val_s1_arr,
        target_ids=val_tgt_arr,
    )
    print(f"Saved validation features to {val_path} ({os.path.getsize(val_path) / 1024**2:.1f} MB)")

    # 10. Save metadata
    summary = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "num_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "train_positives": int(sum(y_train)),
        "train_negatives": int(len(y_train) - sum(y_train)),
        "val_positives": int(sum(y_val)),
        "val_negatives": int(len(y_val) - sum(y_val)),
        "embedding_model": {
            "name": EMBEDDING_MODEL_NAME,
            "license": EMBEDDING_MODEL_LICENSE,
            "parameters": EMBEDDING_MODEL_PARAMS,
        },
        "generation_time_seconds": round(time.time() - t0, 1),
    }

    summary_path = os.path.join(ARTIFACTS_DIR, "feature_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved feature summary to {summary_path}")

    print("=" * 70)
    print("STAGE 3 COMPLETE")
    print("=" * 70)
    print(f"Features ({len(FEATURE_NAMES)} total): {FEATURE_NAMES}")
    print(f"X_train shape: {X_train.shape} (Positives: {summary['train_positives']:,}, Negatives: {summary['train_negatives']:,})")
    print(f"X_val shape:   {X_val.shape} (Positives: {summary['val_positives']:,}, Negatives: {summary['val_negatives']:,})")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_MODEL_PARAMS/1e6:.1f}M params, License: {EMBEDDING_MODEL_LICENSE})")
    print(f"Total execution time: {summary['generation_time_seconds']}s")
    print("=" * 70)


if __name__ == "__main__":
    build_and_cache_datasets()
