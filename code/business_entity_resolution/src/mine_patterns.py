#!/usr/bin/env python3
"""
Mine noisy training pairs to discover legal-suffix and address-abbreviation
patterns. Outputs samples for manual inspection and frequency tables for
building canonicalization maps.
"""

import os
import sys
import re
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    ARTIFACTS_DIR,
)

SEP = "\t"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Loading data...")
s1 = pd.read_csv(TRAIN_SOURCE1, sep=SEP, dtype=str, keep_default_na=False)
s2 = pd.read_csv(TRAIN_SOURCE2, sep=SEP, dtype=str, keep_default_na=False)
s3 = pd.read_csv(TRAIN_SOURCE3, sep=SEP, dtype=str, keep_default_na=False)
gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep=SEP, dtype=str, keep_default_na=False)
print("Loaded.")

# Build lookup dicts
s1_names = dict(zip(s1["entity_id"], s1["business_name"]))
s1_addrs = dict(zip(s1["entity_id"], s1["business_address"]))

all_src = pd.concat([s2, s3], ignore_index=True)
src_names = dict(zip(all_src["entity_id"], all_src["business_name"]))
src_addrs = dict(zip(all_src["entity_id"], all_src["business_address"]))

# ── Sample matched pairs ───────────────────────────────────────────────
# Take a random sample of GT rows with matches
gt_with_matches = gt[gt["matched_entity_ids"].str.strip() != ""].sample(
    n=min(5000, len(gt)), random_state=42
)

name_pairs = []
addr_pairs = []

for _, row in gt_with_matches.iterrows():
    s1_id = row["source1_entity_id"]
    s1_name = s1_names.get(s1_id, "")
    s1_addr = s1_addrs.get(s1_id, "")
    for mid in row["matched_entity_ids"].split(","):
        mid = mid.strip()
        if mid:
            src_name = src_names.get(mid, "")
            src_addr = src_addrs.get(mid, "")
            if s1_name and src_name:
                name_pairs.append((s1_name, src_name))
            if s1_addr and src_addr:
                addr_pairs.append((s1_addr, src_addr))

print(f"Collected {len(name_pairs)} name pairs, {len(addr_pairs)} address pairs")

# ── Mine legal suffixes ────────────────────────────────────────────────
# Look at tokens that appear at the end of business names
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|'
    r'llp|plc|pvt|private|public|group|holdings|enterprises|enterprise|'
    r'associates|association|assoc|foundation|partners|partnership|'
    r'services|service|solutions|solution|industries|industrial|'
    r'technologies|technology|tech|consulting|consultants|consultant|'
    r'international|intl|global|worldwide|national|'
    r'sarl|sas|sa|eurl|sasu|sci|snc|gmbh|ag|'  # French/German
    r'प्राइवेट|लिमिटेड|प्रा\.?|लि\.?|एलएलपी|'  # Hindi
    r'dba|tr(?:ading)?|mfg|manufacturing)\b\.?',
    re.IGNORECASE
)

# Count how often each suffix form appears
suffix_counter = Counter()
for n1, n2 in name_pairs:
    for name in [n1, n2]:
        for m in LEGAL_SUFFIX_PATTERN.finditer(name.lower()):
            suffix_counter[m.group().strip('.')] += 1

print("\n=== TOP LEGAL SUFFIX FORMS ===")
for suffix, cnt in suffix_counter.most_common(60):
    print(f"  {suffix:25s} {cnt:>6,}")

# ── Mine suffix variation pairs ────────────────────────────────────────
# Find cases where suffixes differ between matched name pairs
print("\n=== SUFFIX VARIATION EXAMPLES (first 40) ===")
shown = 0
for n1, n2 in name_pairs[:2000]:
    s1_suffixes = set(m.group().lower().strip('.') for m in LEGAL_SUFFIX_PATTERN.finditer(n1))
    s2_suffixes = set(m.group().lower().strip('.') for m in LEGAL_SUFFIX_PATTERN.finditer(n2))
    if s1_suffixes != s2_suffixes and (s1_suffixes or s2_suffixes):
        print(f"  [{n1}]  →  [{n2}]")
        print(f"    suffixes: {s1_suffixes} vs {s2_suffixes}")
        shown += 1
        if shown >= 40:
            break

# ── Mine address abbreviations ─────────────────────────────────────────
ADDR_TOKEN_PATTERN = re.compile(r'\b\w+\.?\b')

addr_token_counter = Counter()
for a1, a2 in addr_pairs:
    for addr in [a1, a2]:
        for tok in ADDR_TOKEN_PATTERN.findall(addr.lower()):
            addr_token_counter[tok] += 1

print("\n=== TOP ADDRESS TOKENS ===")
for tok, cnt in addr_token_counter.most_common(100):
    print(f"  {tok:25s} {cnt:>8,}")

# ── Find address abbreviation variations ───────────────────────────────
print("\n=== ADDRESS VARIATION EXAMPLES (first 50) ===")
shown = 0
for a1, a2 in addr_pairs[:3000]:
    if a1.lower().strip() != a2.lower().strip() and a1 and a2:
        print(f"  [{a1}]  →  [{a2}]")
        shown += 1
        if shown >= 50:
            break

# ── Sample some raw name pairs for inspection ──────────────────────────
print("\n=== RAW NAME PAIR SAMPLES (first 30) ===")
for i, (n1, n2) in enumerate(name_pairs[:30]):
    print(f"  {i+1:2d}. [{n1}]  ↔  [{n2}]")

print("\nDone. Use these patterns to build normalize.py tables.")
