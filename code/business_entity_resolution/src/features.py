#!/usr/bin/env python3
"""
Stage 3 — Feature Engineering for Business Entity Resolution

Extracts discriminative pair-wise features for (Source 1, Candidate Target) pairs:
- Name similarity features (Jaro-Winkler, Levenshtein, token Jaccard, char 3-gram, phonetic, acronym, embedding cosine)
- Suffix canonicalization features (suffix_normalized_equality, suffix presence)
- Address similarity features (token Jaccard, overlap ratio, Levenshtein, postal match, street num match, null indicator, embedding cosine)
- Meta features (country equality, target source indicator S2/S3, blocking score & rank, interaction features)

Open-vocabulary throughout; zero hardcoded country assumptions.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import numpy as np
import pandas as pd
from rapidfuzz import distance as rf_dist
import jellyfish

from normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_legal_suffixes,
    extract_postal_code,
    extract_street_number,
)

FEATURE_NAMES = [
    # ── Name features ──
    "name_jaro_winkler",
    "name_embedding_cosine",
    "name_levenshtein_ratio",
    "name_token_jaccard",
    "name_token_overlap_ratio",
    "name_char_3gram_jaccard",
    "name_first_token_soundex_match",
    "name_first_token_metaphone_match",
    "name_acronym_match",
    "name_exact_match",
    "name_len_diff",
    # ── Suffix features ──
    "suffix_normalized_equality",
    "suffix_presence_s1",
    "suffix_presence_target",
    # ── Address features ──
    "addr_token_jaccard",
    "addr_token_overlap_ratio",
    "addr_levenshtein_ratio",
    "addr_embedding_cosine",
    "addr_postal_code_match",
    "addr_street_num_match",
    "addr_is_empty_target",
    # ── Meta & Interaction features ──
    "country_exact_match",
    "target_is_s2",
    "target_is_s3",
    "blocking_score",
    "blocking_rank",
    "name_addr_sim_product",
]


def _char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Return set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _extract_acronym(s: str) -> str:
    """Return acronym string from initial letters of words."""
    words = s.split()
    return "".join(w[0] for w in words if w)


def extract_pair_features(
    s1_name_raw: str,
    s1_addr_raw: str,
    s1_country_raw: str,
    t_name_raw: str,
    t_addr_raw: str,
    t_country_raw: str,
    target_id: str,
    blocking_score: float = 0.0,
    blocking_rank: int = 1,
    name_emb_cosine: float = 0.0,
    addr_emb_cosine: float = 0.0,
    s1_norm_name: Optional[str] = None,
    s1_norm_addr: Optional[str] = None,
    s1_norm_country: Optional[str] = None,
    s1_street_num: Optional[str] = None,
    s1_postal_code: Optional[str] = None,
    s1_suffixes: Optional[List[str]] = None,
    t_norm_name: Optional[str] = None,
    t_norm_addr: Optional[str] = None,
    t_norm_country: Optional[str] = None,
    t_street_num: Optional[str] = None,
    t_postal_code: Optional[str] = None,
    t_suffixes: Optional[List[str]] = None,
) -> Dict[str, float]:
    nn1 = s1_norm_name if s1_norm_name is not None else normalize_name(s1_name_raw)
    nn2 = t_norm_name if t_norm_name is not None else normalize_name(t_name_raw)
    na1 = s1_norm_addr if s1_norm_addr is not None else normalize_address(s1_addr_raw)
    na2 = t_norm_addr if t_norm_addr is not None else normalize_address(t_addr_raw)
    nc1 = s1_norm_country if s1_norm_country is not None else normalize_country(s1_country_raw)
    nc2 = t_norm_country if t_norm_country is not None else normalize_country(t_country_raw)

    sn1 = s1_street_num if s1_street_num is not None else extract_street_number(s1_addr_raw)
    sn2 = t_street_num if t_street_num is not None else extract_street_number(t_addr_raw)
    pc1 = s1_postal_code if s1_postal_code is not None else extract_postal_code(s1_addr_raw)
    pc2 = t_postal_code if t_postal_code is not None else extract_postal_code(t_addr_raw)

    sfx1 = s1_suffixes if s1_suffixes is not None else extract_legal_suffixes(s1_name_raw)
    sfx2 = t_suffixes if t_suffixes is not None else extract_legal_suffixes(t_name_raw)

    # ── Name features ──
    name_jw = rf_dist.JaroWinkler.similarity(nn1, nn2)
    name_lev = rf_dist.Levenshtein.normalized_similarity(nn1, nn2)

    toks1 = set(nn1.split())
    toks2 = set(nn2.split())
    intersection_name = toks1 & toks2
    union_name = toks1 | toks2
    name_jaccard = (len(intersection_name) / len(union_name)) if union_name else 1.0
    min_tokens_name = min(len(toks1), len(toks2))
    name_overlap = (len(intersection_name) / min_tokens_name) if min_tokens_name > 0 else 1.0

    ng1 = _char_ngrams(nn1, 3)
    ng2 = _char_ngrams(nn2, 3)
    name_3gram_jaccard = (len(ng1 & ng2) / len(ng1 | ng2)) if (ng1 | ng2) else 1.0

    ft1 = nn1.split()[0] if nn1.split() else ""
    ft2 = nn2.split()[0] if nn2.split() else ""
    soundex_match = 0.0
    metaphone_match = 0.0
    if ft1 and ft2:
        try:
            soundex_match = 1.0 if jellyfish.soundex(ft1) == jellyfish.soundex(ft2) else 0.0
            metaphone_match = 1.0 if jellyfish.metaphone(ft1) == jellyfish.metaphone(ft2) else 0.0
        except Exception:
            pass

    ac1 = _extract_acronym(nn1)
    ac2 = _extract_acronym(nn2)
    acronym_match = 0.0
    if (len(ac1) >= 2 and ac1 == nn2) or (len(ac2) >= 2 and ac2 == nn1) or (len(ac1) >= 2 and ac1 == ac2):
        acronym_match = 1.0

    name_exact = 1.0 if (nn1 == nn2 or s1_name_raw.strip().lower() == t_name_raw.strip().lower()) else 0.0
    name_len_diff = float(abs(len(nn1) - len(nn2)))

    # Suffix features
    suffix_equality = 1.0 if sfx1 == sfx2 else 0.0
    suffix_pres_s1 = 1.0 if sfx1 else 0.0
    suffix_pres_t = 1.0 if sfx2 else 0.0

    # Address features
    addr_is_empty_target = 1.0 if not t_addr_raw.strip() else 0.0
    if na1 and na2:
        atoks1 = set(na1.split())
        atoks2 = set(na2.split())
        addr_inter = atoks1 & atoks2
        addr_union = atoks1 | atoks2
        addr_jaccard = (len(addr_inter) / len(addr_union)) if addr_union else 1.0
        min_addr_toks = min(len(atoks1), len(atoks2))
        addr_overlap = (len(addr_inter) / min_addr_toks) if min_addr_toks > 0 else 1.0
        addr_lev = rf_dist.Levenshtein.normalized_similarity(na1, na2)
    else:
        addr_jaccard = 0.0
        addr_overlap = 0.0
        addr_lev = 0.0

    if pc1 and pc2:
        postal_match = 1.0 if pc1 == pc2 else 0.0
    else:
        postal_match = 0.5

    if sn1 and sn2:
        street_match = 1.0 if sn1 == sn2 else 0.0
    else:
        street_match = 0.5

    country_match = 1.0 if nc1 == nc2 else 0.0
    target_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    target_is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    name_addr_prod = name_jw * addr_overlap

    return {
        "name_jaro_winkler": float(name_jw),
        "name_embedding_cosine": float(name_emb_cosine),
        "name_levenshtein_ratio": float(name_lev),
        "name_token_jaccard": float(name_jaccard),
        "name_token_overlap_ratio": float(name_overlap),
        "name_char_3gram_jaccard": float(name_3gram_jaccard),
        "name_first_token_soundex_match": soundex_match,
        "name_first_token_metaphone_match": metaphone_match,
        "name_acronym_match": acronym_match,
        "name_exact_match": name_exact,
        "name_len_diff": name_len_diff,
        "suffix_normalized_equality": suffix_equality,
        "suffix_presence_s1": suffix_pres_s1,
        "suffix_presence_target": suffix_pres_t,
        "addr_token_jaccard": float(addr_jaccard),
        "addr_token_overlap_ratio": float(addr_overlap),
        "addr_levenshtein_ratio": float(addr_lev),
        "addr_embedding_cosine": float(addr_emb_cosine),
        "addr_postal_code_match": postal_match,
        "addr_street_num_match": street_match,
        "addr_is_empty_target": addr_is_empty_target,
        "country_exact_match": country_match,
        "target_is_s2": target_is_s2,
        "target_is_s3": target_is_s3,
        "blocking_score": float(blocking_score),
        "blocking_rank": float(blocking_rank),
        "name_addr_sim_product": float(name_addr_prod),
    }
