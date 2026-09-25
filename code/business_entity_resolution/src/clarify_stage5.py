#!/usr/bin/env python3
"""
Clarification & In-Depth Audit Script for Stage 5 Questions:
1. Full tau-vs-F0.5 curve with precision, recall, and macro F0.5 separately on Tuning and Holdout.
2. Confidence-gap abstention rule: exact count of candidate pairs suppressed, and precision/recall impact.
3. Global 1:1 assignment confirmation: verify only candidates with p >= tau are considered, and zero-match singletons are preserved.
4. Co-location guardrail: suppression count and correctness (FPs blocked vs TPs blocked) on the INDEPENDENT holdout split.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import ARTIFACTS_DIR, RANDOM_SEED, TRAIN_GROUND_TRUTH
from evaluate_stage5 import compute_entity_f05

def main():
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

    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str, keep_default_na=False)
    gt_map = {}
    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        mids_str = row["matched_entity_ids"].strip()
        gt_map[sid] = {m.strip() for m in mids_str.split(",") if m.strip()} if mids_str else set()

    addr_idx = fnames.index("addr_token_overlap_ratio")
    jw_idx = fnames.index("name_jaro_winkler")
    emb_idx = fnames.index("name_embedding_cosine")
    jac_idx = fnames.index("name_token_jaccard")

    unique_s1 = sorted(list(set(s1_ids)))
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_s1 = rng.permutation(unique_s1)
    split_idx = len(unique_s1) // 2
    tuning_s1 = set(shuffled_s1[:split_idx])
    holdout_s1 = set(shuffled_s1[split_idx:])

    s1_to_candidates = defaultdict(list)
    for i in range(len(X_val)):
        s1_to_candidates[s1_ids[i]].append({
            "target_id": target_ids[i],
            "prob": float(cal_probs[i]),
            "addr_overlap": float(X_val[i, addr_idx]),
            "name_jw": float(X_val[i, jw_idx]),
            "name_emb": float(X_val[i, emb_idx]),
            "name_jaccard": float(X_val[i, jac_idx]),
            "label": int(y_val[i]),
        })

    def run_eval(eval_s1_set, tau, apply_guard=True, gap=0.0, apply_g11=True):
        initial_matches = []
        suppressed_by_gap = 0
        suppressed_by_guard = 0
        guard_tp_blocked = 0
        guard_fp_blocked = 0

        for sid in eval_s1_set:
            cands = s1_to_candidates.get(sid, [])
            if not cands:
                continue

            sorted_cands = sorted(cands, key=lambda x: x["prob"], reverse=True)

            if gap > 0 and len(sorted_cands) >= 2:
                top_p = sorted_cands[0]["prob"]
                sec_p = sorted_cands[1]["prob"]
                if top_p < 0.85 and (top_p - sec_p) < gap:
                    suppressed_by_gap += 1
                    continue

            for c in sorted_cands:
                p = c["prob"]
                if p < tau:
                    continue

                if apply_guard:
                    if p < 0.85 and c["addr_overlap"] >= 0.70 and c["name_jw"] < 0.40 and c["name_emb"] < 0.40 and c["name_jaccard"] == 0:
                        suppressed_by_guard += 1
                        if c["label"] == 1:
                            guard_tp_blocked += 1
                        else:
                            guard_fp_blocked += 1
                        continue

                initial_matches.append((sid, c["target_id"], p, c["label"]))

        final_preds = {sid: set() for sid in eval_s1_set}
        if apply_g11:
            initial_matches.sort(key=lambda x: x[2], reverse=True)
            claimed = set()
            for sid, tid, p, lbl in initial_matches:
                if tid not in claimed:
                    final_preds[sid].add(tid)
                    claimed.add(tid)
        else:
            for sid, tid, p, lbl in initial_matches:
                final_preds[sid].add(tid)

        # Compute pair-level Precision & Recall
        total_pred_pairs = 0
        correct_pred_pairs = 0
        total_true_pairs = 0

        # Compute entity-level metrics
        f05_scores = []
        singleton_correct = 0
        singleton_total = 0

        for sid in eval_s1_set:
            true_mids = gt_map.get(sid, set())
            pred_mids = final_preds.get(sid, set())

            total_true_pairs += len(true_mids)
            total_pred_pairs += len(pred_mids)
            correct_pred_pairs += len(true_mids & pred_mids)

            score = compute_entity_f05(true_mids, pred_mids)
            f05_scores.append(score)

            if len(true_mids) == 0:
                singleton_total += 1
                if len(pred_mids) == 0:
                    singleton_correct += 1

        macro_f05 = float(np.mean(f05_scores))
        prec = correct_pred_pairs / total_pred_pairs if total_pred_pairs > 0 else 1.0
        rec = correct_pred_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
        singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

        return {
            "macro_f05": macro_f05,
            "precision": prec,
            "recall": rec,
            "singleton_acc": singleton_acc,
            "pred_count": total_pred_pairs,
            "true_count": total_true_pairs,
            "correct_count": correct_pred_pairs,
            "suppressed_by_gap": suppressed_by_gap,
            "suppressed_by_guard": suppressed_by_guard,
            "guard_fp_blocked": guard_fp_blocked,
            "guard_tp_blocked": guard_tp_blocked,
        }

    # ══════════════════════════════════════════════════════════════════════
    # QUESTION 1: Full tau-vs-F0.5 curve with Precision & Recall
    # ══════════════════════════════════════════════════════════════════════
    taus = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("=" * 85)
    print("QUESTION 1: FULL TAU-VS-F0.5 CURVE WITH PRECISION & RECALL")
    print("=" * 85)
    print(f"{'Tau':>5} | {'Tuning Prec':>11} {'Tuning Rec':>10} {'Tuning F0.5':>11} | {'Holdout Prec':>12} {'Holdout Rec':>11} {'Holdout F0.5':>12}")
    print("-" * 85)

    curve_records = []
    for tau in taus:
        t_res = run_eval(tuning_s1, tau=tau, apply_guard=True, gap=0.0, apply_g11=True)
        h_res = run_eval(holdout_s1, tau=tau, apply_guard=True, gap=0.0, apply_g11=True)
        curve_records.append({"tau": tau, "tuning": t_res, "holdout": h_res})
        print(f"{tau:5.2f} | {t_res['precision']:11.4f} {t_res['recall']:10.4f} {t_res['macro_f05']:11.4f} | {h_res['precision']:12.4f} {h_res['recall']:11.4f} {h_res['macro_f05']:12.4f}")

    # Inspect calibrated probabilities distribution to explain tau < 0.5 behavior
    print("\nCalibration Distribution Insight:")
    cal_quantiles = np.percentile(cal_probs[cal_probs > 0.1], [1, 5, 10, 25, 50, 75, 90, 99])
    print(f"  Calibrated Probabilities (>0.1) Quantiles: {dict(zip(['1%', '5%', '10%', '25%', '50%', '75%', '90%', '99%'], np.round(cal_quantiles, 4)))}")
    # Count of pairs in [0.40, 0.50]
    p_40_50_mask = (cal_probs >= 0.40) & (cal_probs < 0.50)
    print(f"  Pairs with calibrated prob in [0.40, 0.50): {p_40_50_mask.sum():,} (True Positives: {y_val[p_40_50_mask].sum():,}, Hard Negatives: {(~y_val[p_40_50_mask].astype(bool)).sum():,})")
    print(f"  Empirical Positive Rate in [0.40, 0.50): {y_val[p_40_50_mask].mean()*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # QUESTION 2: Confidence-Gap Abstention Rule Analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("QUESTION 2: CONFIDENCE-GAP ABSTENTION IMPACT")
    print("=" * 85)
    print("Testing gap thresholds at tau = 0.50 (where borderline cases exist):")
    print(f"{'Gap':>6} | {'Suppressed':>10} | {'Holdout Prec':>12} {'Holdout Rec':>11} {'Holdout F0.5':>12}")
    print("-" * 55)
    for g in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        h_gap = run_eval(holdout_s1, tau=0.50, apply_guard=True, gap=g, apply_g11=True)
        print(f"{g:6.2f} | {h_gap['suppressed_by_gap']:10,d} | {h_gap['precision']:12.4f} {h_gap['recall']:11.4f} {h_gap['macro_f05']:12.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # QUESTION 3: Global 1:1 Bipartite Edge Qualification Confirmation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("QUESTION 3: GLOBAL 1:1 BIPARTITE QUALIFICATION CONFIRMATION")
    print("=" * 85)
    h_opt = run_eval(holdout_s1, tau=0.50, apply_guard=True, gap=0.0, apply_g11=True)
    h_nog11 = run_eval(holdout_s1, tau=0.50, apply_guard=True, gap=0.0, apply_g11=False)
    print(f"Holdout S1 entities: {len(holdout_s1):,}")
    print(f"Holdout true singletons: {sum(1 for sid in holdout_s1 if len(gt_map.get(sid, set())) == 0):,}")
    print(f"With Global 1:1 - Total candidate edges accepted: {h_opt['pred_count']:,} (Zero-match S1 entities: {len(holdout_s1) - sum(1 for sid in holdout_s1 if len(gt_map.get(sid, set())) > 0):,})")
    print(f"Without Global 1:1 - Total candidate edges accepted: {h_nog11['pred_count']:,}")
    print(f"Confirmation: ONLY edges with p >= tau are ever placed into the candidate pool. No S1 entity is ever forced into a match.")

    # ══════════════════════════════════════════════════════════════════════
    # QUESTION 4: Co-Location Guardrail on INDEPENDENT HOLDOUT Split
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("QUESTION 4: CO-LOCATION GUARDRAIL ON INDEPENDENT HOLDOUT SPLIT")
    print("=" * 85)
    h_with_guard = run_eval(holdout_s1, tau=0.50, apply_guard=True, gap=0.0, apply_g11=True)
    h_without_guard = run_eval(holdout_s1, tau=0.50, apply_guard=False, gap=0.0, apply_g11=True)
    print(f"Holdout evaluation (tau=0.50):")
    print(f"  WITHOUT Guardrail: Macro F0.5 = {h_without_guard['macro_f05']:.4f} | Prec = {h_without_guard['precision']:.4f} | Rec = {h_without_guard['recall']:.4f} | Singleton Acc = {h_without_guard['singleton_acc']*100:.2f}%")
    print(f"  WITH Guardrail:    Macro F0.5 = {h_with_guard['macro_f05']:.4f} | Prec = {h_with_guard['precision']:.4f} | Rec = {h_with_guard['recall']:.4f} | Singleton Acc = {h_with_guard['singleton_acc']*100:.2f}%")
    print(f"  Holdout Suppressions by Guardrail: {h_with_guard['suppressed_by_guard']} total")
    print(f"    - False Positives correctly blocked: {h_with_guard['guard_fp_blocked']} ({h_with_guard['guard_fp_blocked']/max(1, h_with_guard['suppressed_by_guard'])*100:.1f}%)")
    print(f"    - True Positives accidentally blocked: {h_with_guard['guard_tp_blocked']}")
    print(f"    - Net precision gain: +{(h_with_guard['precision'] - h_without_guard['precision'])*100:.2f}%")

if __name__ == "__main__":
    main()
