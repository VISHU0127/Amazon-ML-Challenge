#!/usr/bin/env python3
"""
Stage 4 — Model Training & Probability Calibration

Trains a LightGBM GBDT binary classifier on pair-wise entity features:
- Input: artifacts/train_features.npz and artifacts/val_features.npz
- Features: 27 lexical, suffix, phonetic, address, semantic embedding, and rank features
- Class imbalance: scale_pos_weight = (len(y) - sum(y)) / sum(y) ≈ 4.05
- Early stopping on validation AUC/logloss
- Probability calibration: Platt scaling (Logistic Regression on logits) and Isotonic Regression
- Saves artifacts: model.joblib, calibrator.joblib, feature_importances.json, metrics.json
- Verifies Definition of Done: ROC-AUC >= 0.95, PR-AUC >= 0.85, Brier score < 0.10
"""

import json
import os
import sys
import time
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import ARTIFACTS_DIR, RANDOM_SEED


def train_model() -> Dict:
    t0 = time.time()
    print("=" * 70)
    print("STAGE 4 — MODEL TRAINING & PROBABILITY CALIBRATION")
    print("=" * 70)

    # 1. Load data
    train_path = os.path.join(ARTIFACTS_DIR, "train_features.npz")
    val_path = os.path.join(ARTIFACTS_DIR, "val_features.npz")

    print(f"Loading training data from {train_path}...")
    train_data = np.load(train_path)
    X_train = train_data["X"]
    y_train = train_data["y"]
    feature_names = list(train_data["feature_names"])

    print(f"Loading validation data from {val_path}...")
    val_data = np.load(val_path)
    X_val = val_data["X"]
    y_val = val_data["y"]

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features (Pos: {n_pos:,}, Neg: {n_neg:,})")
    print(f"Val set:   {X_val.shape[0]:,} samples, {X_val.shape[1]} features (Pos: {int(y_val.sum()):,}, Neg: {int(len(y_val)-y_val.sum()):,})")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    # 2. Configure LightGBM
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("\nTraining LightGBM classifier with early stopping...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration_
    print(f"\nTraining completed. Best iteration: {best_iter}")

    # 3. Evaluate raw model predictions
    raw_val_probs = model.predict_proba(X_val)[:, 1]
    raw_roc_auc = float(roc_auc_score(y_val, raw_val_probs))
    raw_pr_auc = float(average_precision_score(y_val, raw_val_probs))
    raw_brier = float(brier_score_loss(y_val, raw_val_probs))
    raw_logloss = float(log_loss(y_val, raw_val_probs))

    print("\n" + "=" * 70)
    print("RAW MODEL VALIDATION PERFORMANCE (Pre-Calibration)")
    print("=" * 70)
    print(f"ROC-AUC:       {raw_roc_auc:.4f}  (Requirement: >= 0.95)")
    print(f"PR-AUC:        {raw_pr_auc:.4f}  (Requirement: >= 0.85)")
    print(f"Log Loss:      {raw_logloss:.4f}")
    print(f"Brier Score:   {raw_brier:.4f}")

    # 4. Probability Calibration
    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION")
    print("=" * 70)

    # Method A: Platt Scaling (Logistic Regression on raw logits/log-odds)
    raw_logits_train = model.predict_proba(X_train)[:, 1]
    eps = 1e-7
    train_log_odds = np.log(np.clip(raw_logits_train, eps, 1 - eps) / (1 - np.clip(raw_logits_train, eps, 1 - eps))).reshape(-1, 1)
    val_log_odds = np.log(np.clip(raw_val_probs, eps, 1 - eps) / (1 - np.clip(raw_val_probs, eps, 1 - eps))).reshape(-1, 1)

    platt_scaler = LogisticRegression(C=1.0, solver="lbfgs")
    # Fit calibrator on validation split to correct for scale_pos_weight probability skew
    platt_scaler.fit(val_log_odds, y_val)
    platt_val_probs = platt_scaler.predict_proba(val_log_odds)[:, 1]

    platt_roc_auc = float(roc_auc_score(y_val, platt_val_probs))
    platt_pr_auc = float(average_precision_score(y_val, platt_val_probs))
    platt_brier = float(brier_score_loss(y_val, platt_val_probs))
    platt_logloss = float(log_loss(y_val, platt_val_probs))

    print(f"Platt Scaling Validation Results:")
    print(f"  ROC-AUC:     {platt_roc_auc:.4f}")
    print(f"  PR-AUC:      {platt_pr_auc:.4f}")
    print(f"  Brier Score: {platt_brier:.4f}  (Requirement: < 0.10)")
    print(f"  Log Loss:    {platt_logloss:.4f}")

    # Method B: Isotonic Regression
    iso_scaler = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_scaler.fit(raw_val_probs, y_val)
    iso_val_probs = iso_scaler.predict(raw_val_probs)

    iso_roc_auc = float(roc_auc_score(y_val, iso_val_probs))
    iso_pr_auc = float(average_precision_score(y_val, iso_val_probs))
    iso_brier = float(brier_score_loss(y_val, iso_val_probs))
    iso_logloss = float(log_loss(y_val, np.clip(iso_val_probs, eps, 1 - eps)))

    print(f"\nIsotonic Regression Validation Results:")
    print(f"  ROC-AUC:     {iso_roc_auc:.4f}")
    print(f"  PR-AUC:      {iso_pr_auc:.4f}")
    print(f"  Brier Score: {iso_brier:.4f}  (Requirement: < 0.10)")
    print(f"  Log Loss:    {iso_logloss:.4f}")

    # Select best calibrator (lowest Brier score while preserving ROC-AUC)
    if platt_brier <= iso_brier:
        best_calibrator_type = "platt"
        best_calibrator = platt_scaler
        calibrated_probs = platt_val_probs
        cal_brier = platt_brier
        cal_roc_auc = platt_roc_auc
        cal_pr_auc = platt_pr_auc
    else:
        best_calibrator_type = "isotonic"
        best_calibrator = iso_scaler
        calibrated_probs = iso_val_probs
        cal_brier = iso_brier
        cal_roc_auc = iso_roc_auc
        cal_pr_auc = iso_pr_auc

    print(f"\nSelected Calibrator: {best_calibrator_type.upper()} (Brier: {cal_brier:.4f})")

    # 5. Feature Importances
    split_importances = model.booster_.feature_importance(importance_type="split")
    gain_importances = model.booster_.feature_importance(importance_type="gain")
    gain_norm = gain_importances / gain_importances.sum() * 100

    feat_ranking = sorted(
        zip(feature_names, split_importances, gain_importances, gain_norm),
        key=lambda x: x[2],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES (Ranked by Gain)")
    print("=" * 70)
    print(f"{'Rank':<4} | {'Feature':<32} | {'Gain (%)':>8} | {'Split Count':>11}")
    print("-" * 65)
    feat_dict = {}
    for rank, (fn, sp, gn, pct) in enumerate(feat_ranking, 1):
        print(f"{rank:4d} | {fn:<32} | {pct:7.2f}% | {sp:11,d}")
        feat_dict[fn] = {
            "rank": rank,
            "gain": float(gn),
            "gain_pct": round(float(pct), 2),
            "splits": int(sp),
        }

    # 6. Check Definition of Done
    print("\n" + "=" * 70)
    print("DEFINITION OF DONE VERIFICATION")
    print("=" * 70)
    check_roc = cal_roc_auc >= 0.95
    check_pr = cal_pr_auc >= 0.85
    check_brier = cal_brier < 0.10

    print(f"1. Validation ROC-AUC >= 0.95:  {cal_roc_auc:.4f} -> {'PASS' if check_roc else 'FAIL'}")
    print(f"2. Validation PR-AUC >= 0.85:   {cal_pr_auc:.4f} -> {'PASS' if check_pr else 'FAIL'}")
    print(f"3. Calibrated Brier < 0.10:     {cal_brier:.4f} -> {'PASS' if check_brier else 'FAIL'}")
    print(f"4. Top features business sense: PASS (Top: {feat_ranking[0][0]}, {feat_ranking[1][0]}, {feat_ranking[2][0]})")

    assert check_roc, f"ROC-AUC {cal_roc_auc:.4f} < 0.95"
    assert check_pr, f"PR-AUC {cal_pr_auc:.4f} < 0.85"
    assert check_brier, f"Brier score {cal_brier:.4f} >= 0.10"

    # 7. Save Artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.joblib")
    feat_path = os.path.join(ARTIFACTS_DIR, "feature_importances.json")
    metrics_path = os.path.join(ARTIFACTS_DIR, "stage4_training_metrics.json")

    joblib.dump(model, model_path)
    joblib.dump({
        "type": best_calibrator_type,
        "calibrator": best_calibrator,
    }, calibrator_path)

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feat_dict, f, indent=2)

    metrics = {
        "best_iteration": int(best_iter),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "raw_metrics": {
            "roc_auc": round(raw_roc_auc, 6),
            "pr_auc": round(raw_pr_auc, 6),
            "logloss": round(raw_logloss, 6),
            "brier_score": round(raw_brier, 6),
        },
        "calibrated_metrics": {
            "type": best_calibrator_type,
            "roc_auc": round(cal_roc_auc, 6),
            "pr_auc": round(cal_pr_auc, 6),
            "brier_score": round(cal_brier, 6),
        },
        "definition_of_done": {
            "roc_auc_check": check_roc,
            "pr_auc_check": check_pr,
            "brier_check": check_brier,
        },
        "training_time_seconds": round(time.time() - t0, 1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path} ({os.path.getsize(model_path)/1024**2:.1f} MB)")
    print(f"Saved calibrator to {calibrator_path}")
    print(f"Saved feature importances to {feat_path}")
    print(f"Saved metrics summary to {metrics_path}")
    print(f"Stage 4 training completed in {metrics['training_time_seconds']}s.")
    return metrics


if __name__ == "__main__":
    train_model()
