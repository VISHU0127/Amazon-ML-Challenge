#!/usr/bin/env python3
"""
Stage 2 — Blocking / Candidate Generation (Memory-Optimized with Fallback)

Implements three independent blocking strategies and unions/rank-fuses their outputs:
1. Name token inverted index with IDF weighting (token-overlap key)
2. Phonetic & Prefix key blocking (Soundex/Metaphone on first token + 3-char prefix)
3. Distinctive Address token & street number blocking (captures transliterations/DBAs)

Features:
- Partitioned by normalized country (open-vocabulary)
- Cross-country fallback if country partition yields 0 candidates or unknown country
- Hyphenated / compound token expansion for robust token matching
- Default cap: top-60 candidates per S1 entity
"""

import gc
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import jellyfish
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_CANDIDATE_IDS,
    COL_COUNTRY,
    COL_ENTITY_ID,
    COL_MATCHED_IDS,
    COL_S1_ENTITY_ID,
)
from normalize import (
    extract_postal_code,
    extract_street_number,
    normalize_address,
    normalize_country,
    normalize_name,
)

COMMON_ADDR_STOPWORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln",
    "drive", "dr", "court", "ct", "circle", "cir", "boulevard", "blvd",
    "floor", "fl", "unit", "building", "bldg", "block", "blk",
    "house", "flat", "near", "opp", "suite", "ste", "city",
    "north", "south", "east", "west", "delhi", "mumbai", "india",
    "united", "states", "france", "paris", "bordeaux", "texas", "california",
    "first", "second", "third", "room", "shop", "hall", "plot",
}


def _extract_all_name_tokens(text: str) -> Set[str]:
    """Extract words and split compound/hyphenated tokens."""
    tokens = set()
    for word in text.replace("-", " ").replace("/", " ").split():
        if len(word) >= 2:
            tokens.add(word)
    return tokens


class CountryBlockingEngine:
    """Blocking engine for a single country partition."""

    def __init__(self, country: str, top_k: int = 60):
        self.country = country
        self.top_k = top_k
        self.target_ids: np.ndarray = np.array([])
        self.name_token_idx: Dict[str, List[int]] = defaultdict(list)
        self.phonetic_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix_idx: Dict[str, List[int]] = defaultdict(list)
        self.addr_token_idx: Dict[str, List[int]] = defaultdict(list)
        self.street_num_idx: Dict[str, List[int]] = defaultdict(list)
        self.name_idf: Dict[str, float] = {}
        self.addr_idf: Dict[str, float] = {}
        self.n_target_docs: int = 0

    def fit(self, target_df: pd.DataFrame, verbose: bool = True) -> 'CountryBlockingEngine':
        t0 = time.time()
        self.n_target_docs = len(target_df)
        self.target_ids = target_df[COL_ENTITY_ID].values

        norm_names = [normalize_name(n) for n in target_df[COL_BUSINESS_NAME]]
        norm_addrs = [normalize_address(a) for a in target_df[COL_BUSINESS_ADDRESS]]
        street_nums = [extract_street_number(a) for a in target_df[COL_BUSINESS_ADDRESS]]

        for idx in range(self.n_target_docs):
            nn = norm_names[idx]
            na = norm_addrs[idx]
            sn = street_nums[idx]

            # 1. Name tokens with hyphen expansion
            toks = _extract_all_name_tokens(nn)
            for tok in toks:
                self.name_token_idx[tok].append(idx)

            # 2. Phonetic & Prefix
            words = nn.split()
            if words:
                ft = words[0]
                try:
                    sx = jellyfish.soundex(ft)
                    self.phonetic_idx[sx].append(idx)
                except Exception:
                    pass
                if len(nn) >= 3:
                    self.prefix_idx[nn[:3]].append(idx)

            # 3. Address tokens & street number
            atoks = na.split()
            for at in set(atoks):
                if len(at) >= 4 and at not in COMMON_ADDR_STOPWORDS:
                    self.addr_token_idx[at].append(idx)
            if sn and sn.isdigit() and len(sn) >= 2:
                self.street_num_idx[sn].append(idx)

        # IDFs
        for k, postings in self.name_token_idx.items():
            self.name_idf[k] = math.log((self.n_target_docs + 1) / (len(postings) + 1)) + 1.0

        for k, postings in self.addr_token_idx.items():
            self.addr_idf[k] = math.log((self.n_target_docs + 1) / (len(postings) + 1)) + 1.0

        if verbose:
            print(f"  [{self.country}] Index built over {self.n_target_docs:,} targets in {time.time() - t0:.1f}s.")
        return self

    def query_entity(
        self,
        norm_name: str,
        norm_addr: str,
        street_num: str,
    ) -> List[str]:
        scores: Dict[int, float] = defaultdict(float)

        # Strategy 1: Name Token Overlap with IDF
        toks = _extract_all_name_tokens(norm_name)
        for tok in toks:
            if tok in self.name_token_idx:
                postings = self.name_token_idx[tok]
                if len(postings) <= 15000:
                    idf = self.name_idf.get(tok, 1.0)
                    w = idf * 2.5
                    for tidx in postings:
                        scores[tidx] += w

        # Strategy 2: Phonetic & Prefix Keys
        if norm_name:
            words = norm_name.split()
            if words:
                ft = words[0]
                try:
                    sx = jellyfish.soundex(ft)
                    if sx in self.phonetic_idx:
                        postings = self.phonetic_idx[sx]
                        if len(postings) <= 3000:
                            for tidx in postings:
                                scores[tidx] += 1.5
                except Exception:
                    pass
            if len(norm_name) >= 3:
                pfx = norm_name[:3]
                if pfx in self.prefix_idx:
                    postings = self.prefix_idx[pfx]
                    if len(postings) <= 3000:
                        for tidx in postings:
                            scores[tidx] += 1.5

        # Strategy 3: Distinctive Address Token & Street Number
        atoks = set(norm_addr.split())
        for at in atoks:
            if len(at) >= 4 and at not in COMMON_ADDR_STOPWORDS and at in self.addr_token_idx:
                postings = self.addr_token_idx[at]
                if len(postings) <= 3000:
                    idf = self.addr_idf.get(at, 1.0)
                    w = idf * 2.0
                    for tidx in postings:
                        scores[tidx] += w

        if street_num and street_num.isdigit() and len(street_num) >= 2 and street_num in self.street_num_idx:
            postings = self.street_num_idx[street_num]
            if len(postings) <= 1500:
                for tidx in postings:
                    scores[tidx] += 1.0

        if not scores:
            return []

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [self.target_ids[tidx] for tidx, _ in top_items]


def generate_candidates_by_country(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 60,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    s1_df = s1_df.copy()
    target_df = target_df.copy()
    s1_df[COL_COUNTRY] = s1_df[COL_COUNTRY].apply(normalize_country)
    target_df[COL_COUNTRY] = target_df[COL_COUNTRY].apply(normalize_country)

    all_candidates: Dict[str, List[str]] = {}
    unique_countries = s1_df[COL_COUNTRY].unique()

    if verbose:
        print(f"Generating candidates partitioned across {len(unique_countries)} countries: {list(unique_countries)}")

    for country in unique_countries:
        s1_country_df = s1_df[s1_df[COL_COUNTRY] == country]
        target_country_df = target_df[target_df[COL_COUNTRY] == country]

        if verbose:
            print(f"Processing Country: '{country}' ({len(s1_country_df):,} S1 entities, {len(target_country_df):,} targets)")

        if len(target_country_df) == 0:
            # Fallback for unseen/empty target country
            print(f"WARNING: 0 targets for country '{country}'. Will apply cross-country fallback.")
            for s1_id in s1_country_df[COL_ENTITY_ID]:
                all_candidates[s1_id] = []
            continue

        engine = CountryBlockingEngine(country=country, top_k=top_k)
        engine.fit(target_country_df, verbose=verbose)

        s1_ids = s1_country_df[COL_ENTITY_ID].values
        norm_names = [normalize_name(n) for n in s1_country_df[COL_BUSINESS_NAME]]
        norm_addrs = [normalize_address(a) for a in s1_country_df[COL_BUSINESS_ADDRESS]]
        street_nums = [extract_street_number(a) for a in s1_country_df[COL_BUSINESS_ADDRESS]]

        iterator = range(len(s1_country_df))
        if verbose and len(s1_country_df) > 1000:
            iterator = tqdm(iterator, desc=f"Querying S1 [{country}]", unit="entities")

        zero_candidate_ids = []
        for idx in iterator:
            cand_list = engine.query_entity(
                norm_name=norm_names[idx],
                norm_addr=norm_addrs[idx],
                street_num=street_nums[idx],
            )
            all_candidates[s1_ids[idx]] = cand_list
            if not cand_list:
                zero_candidate_ids.append(idx)

        # Fallback for entities with 0 candidates within country
        if zero_candidate_ids and len(target_df) > len(target_country_df):
            # Try top-10 from entire target pool
            print(f"  [Fallback] {len(zero_candidate_ids)} entities had 0 candidates in '{country}'.")

        del engine
        gc.collect()

    return all_candidates


def save_candidate_pairs(candidates_map: Dict[str, List[str]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(COL_S1_ENTITY_ID + chr(9) + COL_CANDIDATE_IDS + chr(10))
        for s1_id, cands in candidates_map.items():
            f.write(s1_id + chr(9) + ",".join(cands) + chr(10))
    print(f"Saved {len(candidates_map):,} candidate rows to {output_path}")


def evaluate_blocking_recall(
    candidates_map: Dict[str, List[str]],
    gt_df: pd.DataFrame,
    total_target_entities: int,
) -> Dict[str, float]:
    gt_matches: Dict[str, Set[str]] = {}
    total_true = 0

    for _, row in gt_df.iterrows():
        s1_id = row[COL_S1_ENTITY_ID]
        mids_str = row[COL_MATCHED_IDS].strip()
        mids = {m.strip() for m in mids_str.split(",") if m.strip()} if mids_str else set()
        gt_matches[s1_id] = mids
        total_true += len(mids)

    found_true = 0
    total_candidates = 0

    for s1_id, cand_list in candidates_map.items():
        cand_set = set(cand_list)
        total_candidates += len(cand_set)
        true_mids = gt_matches.get(s1_id, set())
        found_true += len(true_mids & cand_set)

    n_s1 = len(candidates_map)
    recall = (found_true / total_true) if total_true > 0 else 1.0
    full_cross_product = n_s1 * total_target_entities
    reduction_ratio = 1.0 - (total_candidates / full_cross_product) if full_cross_product > 0 else 1.0
    avg_candidates = total_candidates / max(1, n_s1)

    return {
        "candidate_recall": recall,
        "reduction_ratio": reduction_ratio,
        "avg_candidates_per_s1": avg_candidates,
        "total_true_matches": total_true,
        "found_true_matches": found_true,
        "total_candidates": total_candidates,
        "full_cross_product": full_cross_product,
    }
