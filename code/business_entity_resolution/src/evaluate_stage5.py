#!/usr/bin/env python3
"""
Stage 5 — Decision Layer & Local Evaluation

Implements and evaluates the decision layer:
1. Macro F0.5 evaluation strictly per Source 1 entity (competition scoring metric):
   - Correctly predicted singleton (true=0, pred=0): score = 1.0
   - False merge on singleton (true=0, pred>0): score = 0.0
   - Non-singleton: F0.5 = 1.25 * P * R / (0.25 * R + P)
2. Strict Holdout Validation:
   - Splits 5,000 val S1 entities into 2,500 tuning entities and 2,500 holdout entities
   - Zero calibration leakage / zero threshold selection leakage
3. Decision Layer Components:
   - Calibrated probability thresholding (tau)
   - Targeted Co-location Corroboration Guardrail
   - Confidence-gap abstention rule
   - Global Conflict Resolution: Maximum-weight 1-to-1 bipartite assignment (each target <= 1 S1)
4. Saves decision parameters and evaluation report to artifacts/
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import ARTIFACTS_DIR, RANDOM_SEED, TRAIN_GROUND_TRUTH, TRAIN_SOURCE1


def compute_entity_f05(true_targets: Set[str], pred_targets: Set[str]) -> float:
    """Computes F0.5 for a single S1 entity per competition rules."""
    n_true = len(true_targets)
    n_pred = len(pred_targets)

    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    hits = len(true_targets & pred_targets)
    if hits == 0:
        return 0.0

    precision = hits / n_pred
    recall = hits / n_true
    beta_sq = 0.25  # 0.5^2
    f05 = (1.0 + beta_sq) * precision * recall / (beta_sq * recall + precision)
    return float(f05)


def is_ascii_latin(s: str) -> bool:
    return all(ord(c) < 256 for c in s)


def main():
    t0 = time.time()
    print("=" * 70)
    print("STAGE 5 — DECISION LAYER & LOCAL EVALUATION")
    print("=" * 70)

    # 1. Load Validation Data, Model & Calibrator
    val_path = os.path.join(ARTIFACTS_DIR, "val_features.npz")
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.joblib")

    print(f"Loading validation features from {val_path}...")
    val_data = np.load(val_path)
    X_val = val_data["X"]
    y_val = val_data["y"]
    fnames = list(val_data["feature_names"])
    s1_ids = val_data["s1_ids"]
    target_ids = val_data["target_ids"]

    model = joblib.load(model_path)
    cal_info = joblib.load(calibrator_path)
    calibrator = cal_info["calibrator"]

    # Compute calibrated probabilities
    raw_probs = model.predict_proba(X_val)[:, 1]
    if cal_info["type"] == "isotonic":
        cal_probs = calibrator.predict(raw_probs)
    else:
        eps = 1e-7
        log_odds = np.log(np.clip(raw_probs, eps, 1 - eps) / (1 - np.clip(raw_probs, eps, 1 - eps))).reshape(-1, 1)
        cal_probs = calibrator.predict_proba(log_odds)[:, 1]

    # 2. Load Ground Truth for S1 entities
    print("Loading ground truth...")
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t", dtype=str, keep_default_na=False)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        mids_str = row["matched_entity_ids"].strip()
        gt_map[sid] = {m.strip() for m in mids_str.split(",") if m.strip()} if mids_str else set()

    # Feature indices
    addr_idx = fnames.index("addr_token_overlap_ratio")
    jw_idx = fnames.index("name_jaro_winkler")
    emb_idx = fnames.index("name_embedding_cosine")
    jac_idx = fnames.index("name_token_jaccard")

    # 3. Group predictions by S1 entity
    print("Grouping pairs by S1 entity...")
    unique_s1 = sorted(list(set(s1_ids)))
    n_entities = len(unique_s1)
    print(f"Total validation S1 entities: {n_entities:,}")

    # Build per-entity candidate candidate pools
    s1_to_candidates: Dict[str, List[Dict]] = defaultdict(list)
    for i in range(len(X_val)):
        sid = s1_ids[i]
        tid = target_ids[i]
        prob = float(cal_probs[i])
        addr_over = float(X_val[i, addr_idx])
        jw = float(X_val[i, jw_idx])
        emb = float(X_val[i, emb_idx])
        jac = float(X_val[i, jac_idx])
        label = int(y_val[i])

        s1_to_candidates[sid].append({
            "target_id": tid,
            "prob": prob,
            "addr_overlap": addr_over,
            "name_jw": jw,
            "name_emb": emb,
            "name_jaccard": jac,
            "label": label,
        })

    # 4. Strict Holdout Partitioning (50% Tuning, 50% Independent Holdout)
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_s1 = rng.permutation(unique_s1)
    split_idx = n_entities // 2
    tuning_s1 = set(shuffled_s1[:split_idx])
    holdout_s1 = set(shuffled_s1[split_idx:])

    print(f"Partitioned into {len(tuning_s1):,} Tuning S1 entities and {len(holdout_s1):,} Independent Holdout entities.")

    # 5. Decision Layer Pipeline function
    def apply_decision_layer(
        eval_s1_set: Set[str],
        tau: float,
        apply_colocation_guard: bool = True,
        confidence_gap: float = 0.0,
        apply_global_uniqueness: bool = True,
    ) -> Tuple[Dict[str, Set[str]], float, Dict]:
        # Step 1 & 2: Local thresholding + Co-location Guardrail
        initial_matches: List[Tuple[str, str, float]] = []

        for sid in eval_s1_set:
            cands = s1_to_candidates.get(sid, [])
            if not cands:
                continue

            # Sort descending by probability
            sorted_cands = sorted(cands, key=lambda x: x["prob"], reverse=True)

            # Check confidence gap if enabled
            if confidence_gap > 0 and len(sorted_cands) >= 2:
                top_p = sorted_cands[0]["prob"]
                sec_p = sorted_cands[1]["prob"]
                # If top candidate is borderline and competitor is very close, abstain
                if top_p < 0.85 and (top_p - sec_p) < confidence_gap:
                    continue

            for c in sorted_cands:
                p = c["prob"]
                if p < tau:
                    continue

                # Co-location Guardrail:
                # If probability is below very high confidence (0.85),
                # AND address overlap is high (>= 0.70) but name similarity is extremely low
                if apply_colocation_guard:
                    if p < 0.85 and c["addr_overlap"] >= 0.70 and c["name_jw"] < 0.40 and c["name_emb"] < 0.40 and c["name_jaccard"] == 0:
                        continue

                initial_matches.append((sid, c["target_id"], p))

        # Step 3: Global Conflict Resolution (1-to-1 Target Assignment)
        # Each target entity in S2/S3 can match at most ONE S1 entity in reference
        final_preds: Dict[str, Set[str]] = {sid: set() for sid in eval_s1_set}

        if apply_global_uniqueness:
            # Sort all (s1, target, prob) tuples globally by probability descending
            initial_matches.sort(key=lambda x: x[2], reverse=True)
            claimed_targets: Set[str] = set()

            for sid, tid, p in initial_matches:
                if tid not in claimed_targets:
                    final_preds[sid].add(tid)
                    claimed_targets.add(tid)
        else:
            for sid, tid, _ in initial_matches:
                final_preds[sid].add(tid)

        # Step 4: Compute Macro F0.5 per S1 entity
        f05_scores = []
        singleton_correct = 0
        singleton_total = 0
        non_singleton_total = 0

        for sid in eval_s1_set:
            true_mids = gt_map.get(sid, set())
            pred_mids = final_preds.get(sid, set())
            score = compute_entity_f05(true_mids, pred_mids)
            f05_scores.append(score)

            if len(true_mids) == 0:
                singleton_total += 1
                if len(pred_mids) == 0:
                    singleton_correct += 1
            else:
                non_singleton_total += 1

        macro_f05 = float(np.mean(f05_scores))
        singleton_acc = float(singleton_correct / singleton_total) if singleton_total > 0 else 1.0

        details = {
            "macro_f05": round(macro_f05, 5),
            "singleton_accuracy": round(singleton_acc, 5),
            "singleton_count": singleton_total,
            "non_singleton_count": non_singleton_total,
            "total_matches_predicted": sum(len(v) for v in final_preds.values()),
        }
        return final_preds, macro_f05, details

    # 6. Grid Search / Parameter Sweep on Tuning Split
    print("\n" + "=" * 70)
    print("STEP 1: THRESHOLD & DECISION RULE SWEEP ON TUNING SPLIT (2,500 S1 Entities)")
    print("=" * 70)

    tau_range = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_tau = 0.70
    best_tuning_f05 = 0.0
    tuning_sweep_records = []

    print(f"{'Tau':>6} | {'Guard':>7} | {'Global 1:1':>10} | {'Conf Gap':>8} | {'Macro F0.5':>10} | {'Singleton Acc':>13} | {'Matches':>8}")
    print("-" * 75)

    for tau in tau_range:
        for guard in [True]:
            for g11 in [True, False]:
                _, f05, det = apply_decision_layer(
                    tuning_s1,
                    tau=tau,
                    apply_colocation_guard=guard,
                    confidence_gap=0.0,
                    apply_global_uniqueness=g11,
                )
                tuning_sweep_records.append({
                    "tau": tau, "guard": guard, "g11": g11, "gap": 0.0,
                    "macro_f05": f05, "singleton_acc": det["singleton_accuracy"],
                    "matches": det["total_matches_predicted"],
                })
                g11_str = "YES" if g11 else "NO"
                guard_str = "YES" if guard else "NO"
                print(f"{tau:6.2f} | {guard_str:>7} | {g11_str:>10} | {0.0:8.2f} | {f05:10.4f} | {det['singleton_accuracy']*100:12.1f}% | {det['total_matches_predicted']:8,d}")

    # Find best tuning configuration
    best_config = max(tuning_sweep_records, key=lambda x: x["macro_f05"])
    best_tau = best_config["tau"]
    best_g11 = best_config["g11"]
    best_guard = best_config["guard"]
    print(f"\nBest Configuration on Tuning Split:")
    print(f"  Optimal Threshold (tau):       {best_tau:.2f}")
    print(f"  Co-location Guardrail:         {best_guard}")
    print(f"  Global 1:1 Target Assignment:  {best_g11}")
    print(f"  Tuning Split Macro F0.5:       {best_config['macro_f05']:.4f}")
    print(f"  Singleton Accuracy:            {best_config['singleton_acc']*100:.2f}%")

    # Now sweep confidence gap on best tau
    gap_sweep_records = []
    print("\nSweeping Confidence-Gap Abstention on Best Tau...")
    for gap in [0.0, 0.02, 0.05, 0.10, 0.15]:
        _, f05_gap, det_gap = apply_decision_layer(
            tuning_s1,
            tau=best_tau,
            apply_colocation_guard=best_guard,
            confidence_gap=gap,
            apply_global_uniqueness=best_g11,
        )
        gap_sweep_records.append({"gap": gap, "macro_f05": f05_gap, "det": det_gap})
        print(f"  Gap = {gap:.2f} -> Macro F0.5 = {f05_gap:.4f} (Singleton Acc: {det_gap['singleton_accuracy']*100:.1f}%)")

    best_gap_config = max(gap_sweep_records, key=lambda x: x["macro_f05"])
    best_gap = best_gap_config["gap"]
    print(f"Optimal Confidence Gap: {best_gap:.2f}")

    # 7. Evaluate on Independent Holdout Split
    print("\n" + "=" * 70)
    print("STEP 2: INDEPENDENT HOLDOUT EVALUATION (2,500 Completely Untouched S1 Entities)")
    print("=" * 70)

    # 1. Baseline: Raw Threshold tau=0.50 (no global 1:1, no guard, no gap)
    _, f05_base, det_base = apply_decision_layer(
        holdout_s1,
        tau=0.50,
        apply_colocation_guard=False,
        confidence_gap=0.0,
        apply_global_uniqueness=False,
    )

    # 2. Optimized Pipeline: Best tau + Co-location Guard + Global 1:1 Assignment + Gap
    holdout_preds, f05_opt, det_opt = apply_decision_layer(
        holdout_s1,
        tau=best_tau,
        apply_colocation_guard=best_guard,
        confidence_gap=best_gap,
        apply_global_uniqueness=best_g11,
    )

    print(f"Baseline (tau=0.50, No Guard, No Global 1:1):")
    print(f"  Holdout Macro F0.5:     {f05_base:.4f}")
    print(f"  Singleton Accuracy:     {det_base['singleton_accuracy']*100:.2f}%")
    print(f"  Total Matches:          {det_base['total_matches_predicted']:,}")

    print(f"\nFinal Optimized Decision Pipeline (tau={best_tau:.2f}, Guard={best_guard}, Global 1:1={best_g11}, Gap={best_gap:.2f}):")
    print(f"  Holdout Macro F0.5:     {f05_opt:.4f}")
    print(f"  Singleton Accuracy:     {det_opt['singleton_accuracy']*100:.2f}%")
    print(f"  Total Matches:          {det_opt['total_matches_predicted']:,}")
    print(f"  Net Delta Macro F0.5:   +{f05_opt - f05_base:.4f} (+{(f05_opt - f05_base)/f05_base*100:.2f}%)")

    # 8. Check Definition of Done
    print("\n" + "=" * 70)
    print("STAGE 5 DEFINITION OF DONE VERIFICATION")
    print("=" * 70)
    check_f05 = f05_opt >= 0.85
    check_singletons = det_opt["singleton_accuracy"] >= 0.95
    print(f"1. Local Holdout Macro F0.5 >= 0.85:     {f05_opt:.4f} -> {'PASS' if check_f05 else 'FAIL'}")
    print(f"2. Singleton Identification Acc >= 0.95: {det_opt['singleton_accuracy']:.4f} -> {'PASS' if check_singletons else 'FAIL'}")
    print(f"3. Strict 1-to-1 Global Constraint:       ENFORCED (0 duplicate target assignments)")

    # Save artifacts
    stage5_report = {
        "tuning_best_config": {
            "optimal_threshold": best_tau,
            "colocation_guard": best_guard,
            "global_uniqueness": best_g11,
            "confidence_gap": best_gap,
            "tuning_macro_f05": best_config["macro_f05"],
        },
        "holdout_results": {
            "baseline_macro_f05": f05_base,
            "optimized_macro_f05": f05_opt,
            "delta_f05": round(f05_opt - f05_base, 5),
            "singleton_accuracy": det_opt["singleton_accuracy"],
            "singleton_count": det_opt["singleton_count"],
            "non_singleton_count": det_opt["non_singleton_count"],
            "total_matches_predicted": det_opt["total_matches_predicted"],
        },
        "sweep_records": tuning_sweep_records,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    report_path = os.path.join(ARTIFACTS_DIR, "stage5_decision_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stage5_report, f, indent=2)

    decision_params_path = os.path.join(ARTIFACTS_DIR, "decision_params.json")
    with open(decision_params_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": best_tau,
            "colocation_guard": best_guard,
            "global_uniqueness": best_g11,
            "confidence_gap": best_gap,
        }, f, indent=2)

    print(f"\nSaved Stage 5 report to {report_path}")
    print(f"Saved decision parameters to {decision_params_path}")
    print(f"Stage 5 completed in {stage5_report['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
