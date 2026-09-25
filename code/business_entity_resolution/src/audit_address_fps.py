#!/usr/bin/env python3
"""
Diagnostic: Audit validation false positives for the 'high address overlap + low name similarity' failure mode.
Specifically investigates whether commercial complexes / co-located businesses in Indian/US addresses cause false merges.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import ARTIFACTS_DIR

def audit_address_overlap_fps():
    val_path = os.path.join(ARTIFACTS_DIR, "val_features.npz")
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.joblib")

    val_data = np.load(val_path)
    X_val = val_data["X"]
    y_val = val_data["y"]
    fnames = list(val_data["feature_names"])
    s1_ids = val_data["s1_ids"]
    target_ids = val_data["target_ids"]

    model = joblib.load(model_path)
    cal_info = joblib.load(calibrator_path)
    calibrator = cal_info["calibrator"]

    raw_probs = model.predict_proba(X_val)[:, 1]
    if cal_info["type"] == "isotonic":
        cal_probs = calibrator.predict(raw_probs)
    else:
        eps = 1e-7
        log_odds = np.log(np.clip(raw_probs, eps, 1 - eps) / (1 - np.clip(raw_probs, eps, 1 - eps))).reshape(-1, 1)
        cal_probs = calibrator.predict_proba(log_odds)[:, 1]

    addr_overlap_idx = fnames.index("addr_token_overlap_ratio")
    name_jw_idx = fnames.index("name_jaro_winkler")
    name_emb_idx = fnames.index("name_embedding_cosine")
    name_tok_jac_idx = fnames.index("name_token_jaccard")
    prod_sim_idx = fnames.index("name_addr_sim_product")

    # Evaluate at multiple candidate decision thresholds tau in [0.5, 0.7, 0.8, 0.9]
    print("=" * 70)
    print("FALSE POSITIVE AUDIT: High Address Overlap + Low Name Similarity")
    print("=" * 70)

    for tau in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        preds = (cal_probs >= tau).astype(int)
        fps = (preds == 1) & (y_val == 0)
        tps = (preds == 1) & (y_val == 1)
        fns = (preds == 0) & (y_val == 1)

        n_fp = fps.sum()
        n_tp = tps.sum()
        n_fn = fns.sum()
        prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * rec + prec) if (0.25 * rec + prec) > 0 else 0

        # Sub-cluster of FPs: addr_overlap >= 0.7 BUT name_jw < 0.75 or name_emb < 0.65
        fp_addr_overlap = X_val[fps, addr_overlap_idx]
        fp_name_jw = X_val[fps, name_jw_idx]
        fp_name_emb = X_val[fps, name_emb_idx]

        co_located_mask = (fp_addr_overlap >= 0.7) & ((fp_name_jw < 0.75) | (fp_name_emb < 0.65))
        n_colocated_fp = co_located_mask.sum()
        pct_of_fps = (n_colocated_fp / n_fp * 100) if n_fp > 0 else 0

        print(f"\nThreshold tau = {tau:.2f}:")
        print(f"  TP: {n_tp:,} | FP: {n_fp:,} | FN: {n_fn:,} | Prec: {prec:.4f} | Rec: {rec:.4f} | F0.5: {f05:.4f}")
        print(f"  Co-located FPs (Addr Overlap >= 0.7 & Low Name Sim): {n_colocated_fp:,} / {n_fp:,} ({pct_of_fps:.1f}% of all FPs)")

    # Print top examples of co-located false positives at tau = 0.7
    preds_70 = (cal_probs >= 0.7).astype(int)
    fps_70 = (preds_70 == 1) & (y_val == 0)
    co_located_70 = fps_70 & (X_val[:, addr_overlap_idx] >= 0.7) & ((X_val[:, name_jw_idx] < 0.75) | (X_val[:, name_emb_idx] < 0.65))
    indices = np.where(co_located_70)[0]

    print("\n" + "=" * 70)
    print(f"Top 10 Co-Located False Positive Examples at tau = 0.70 (Total: {len(indices)})")
    print("=" * 70)
    for idx in indices[:10]:
        print(f"S1: {s1_ids[idx]} <-> Target: {target_ids[idx]} | Prob: {cal_probs[idx]:.4f}")
        print(f"  Addr Overlap: {X_val[idx, addr_overlap_idx]:.3f} | Name JW: {X_val[idx, name_jw_idx]:.3f} | Name Emb: {X_val[idx, name_emb_idx]:.3f} | Prod: {X_val[idx, prod_sim_idx]:.3f}")

if __name__ == "__main__":
    audit_address_overlap_fps()
