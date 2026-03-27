"""
Phase 3 — ToxPathGuard Model Training
---------------------------------------
Trains three independent XGBoost binary classifiers, one per
biological stress pathway assay:

    SR-p53  →  agent_p53.pkl   (DNA / Genotoxic Stress)
    SR-ARE  →  agent_are.pkl   (Oxidative Stress)
    SR-HSE  →  agent_hse.pkl   (Cellular / Heat-Shock Stress)

Each model is evaluated with PR-AUC as the primary metric because
the datasets are heavily imbalanced (5–17:1 negative/positive ratio).

Outputs written to models/:
    agent_p53.pkl
    agent_are.pkl
    agent_hse.pkl
    metrics.json     ← test-set scores + thresholds for Phase 4

Usage:
    python src/train.py --data data/processed --out models
"""

import os
import json
import argparse
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


# ── Configuration ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Assay → output model filename mapping.
# The file naming is deliberate: "agent_" prefix aligns with the
# three-checkpoint narrative you will use in the Phase 6 UI.
ASSAY_CONFIG = {
    "SR-p53": "agent_p53",
    "SR-ARE":  "agent_are",
    "SR-HSE":  "agent_hse",
}

# Train / test split ratio.
# 80/20 is standard. With ~6,000–7,000 samples this gives you
# ~1,200–1,400 test samples — enough for stable metric estimates.
TEST_SIZE = 0.20

# XGBoost base hyperparameters shared across all three models.
# These are deliberately conservative: no aggressive tuning, no
# grid search. The goal is a stable, reproducible baseline that
# runs in seconds on a laptop.
#
# n_estimators=300 with early_stopping_rounds=30 means the model
# stops adding trees when validation PR-AUC stops improving for
# 30 consecutive rounds. This prevents overfitting without needing
# a separate hyperparameter sweep.
XGBOOST_BASE_PARAMS = {
    "n_estimators":        300,
    "max_depth":           4,       # shallow trees reduce overfitting on small data
    "learning_rate":       0.05,    # slow learning → better generalisation
    "subsample":           0.8,     # row subsampling (like bagging)
    "colsample_bytree":    0.8,     # feature subsampling per tree
    "use_label_encoder":   False,
    "eval_metric":         "aucpr", # XGBoost's internal early-stopping metric
    "early_stopping_rounds": 30,
    "random_state":        RANDOM_SEED,
    "tree_method":         "hist",  # fast histogram-based algorithm; works offline
    "verbosity":           0,       # suppress XGBoost's own training logs
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_assay(data_dir: str, assay: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the pre-processed (X, y) for one assay from disk.
    The naming convention (hyphens → underscores) must match
    what features.py wrote in Phase 2.
    """
    safe_name = assay.replace("-", "_")
    path = os.path.join(data_dir, f"{safe_name}.npz")
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int8)


# ── Threshold Selection ────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds the probability threshold that maximises F1 score on the
    data you pass in (the validation split during training, or the
    test set here).

    Why this matters: XGBoost's default decision threshold is 0.5,
    but with severe class imbalance the optimal threshold is almost
    always much lower — sometimes 0.15 or 0.20. Using 0.5 would cause
    the model to miss most positives (toxics) even when the probability
    scores are well-calibrated.

    This threshold is saved to metrics.json and will be loaded by
    Phase 4's prediction layer to produce the risk labels.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more precision/recall value
    # than thresholds, so we trim the last value to align lengths.
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


# ── Metric Reporting ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


def report_metrics(
    assay: str,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes and prints a full evaluation summary for one assay.
    Returns a metrics dictionary to be serialised to metrics.json.
    """
    y_pred  = (y_prob >= threshold).astype(int)
    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision_val  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val     = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n  PR-AUC      : {pr_auc:.4f}   ← primary metric")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}   (at threshold {threshold:.3f})")
    print(f"  Precision   : {precision_val:.4f}")
    print(f"  Recall      : {recall_val:.4f}")
    print(f"\n  Confusion Matrix (threshold = {threshold:.3f}):")
    print(f"               Pred 0    Pred 1")
    print(f"  Actual 0  :  {tn:>6}    {fp:>6}   (TN, FP)")
    print(f"  Actual 1  :  {fn:>6}    {tp:>6}   (FN, TP)")
    print(f"\n  True Positives (toxics caught) : {tp}")
    print(f"  False Negatives (toxics missed): {fn}  ← keep this low")

    return {
        "assay":      assay,
        "pr_auc":     round(pr_auc, 4),
        "roc_auc":    round(roc_auc, 4),
        "f1":         round(f1, 4),
        "precision":  round(precision_val, 4),
        "recall":     round(recall_val, 4),
        "threshold":  round(threshold, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ── Model Training ─────────────────────────────────────────────────────────────

def train_one_model(
    assay: str,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray, float]:
    """
    Trains a single XGBoost classifier for one assay.

    The training procedure:
      1. Stratified 80/20 split — ensures the positive class is
         proportionally represented in both train and test sets.
         With 6% positives this is critical: a non-stratified split
         could accidentally put all positives in one partition.

      2. A further 10% of the training set is held out as a
         validation set for early stopping. This lets XGBoost
         stop adding trees before it overfits, without ever
         touching the test set.

      3. scale_pos_weight is computed from the training partition
         (not the full dataset) to avoid any subtle leakage of
         test-set statistics into training.

    Returns the fitted model, test probabilities, test labels,
    and the best threshold.
    """
    # ── Split: full → train+val / test ────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # ── Split: train+val → train / val (for early stopping) ───────────────────
    # We use 12.5% of the trainval set as validation, which gives roughly
    # a 70/10/20 train/val/test split of the full dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.125,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    # ── Class imbalance weight ─────────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / n_pos   # scale_pos_weight

    print(f"\n  Train : {len(y_train):,}  |  Val : {len(y_val):,}  |  Test : {len(y_test):,}")
    print(f"  Train positives : {n_pos:,}  |  Train negatives : {n_neg:,}")
    print(f"  scale_pos_weight: {spw:.1f}")

    # ── Instantiate and fit ────────────────────────────────────────────────────
    model = XGBClassifier(
        **XGBOOST_BASE_PARAMS,
        scale_pos_weight=spw,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_round = model.best_iteration
    print(f"  Best iteration  : {best_round} (early stopping at 30 rounds)")

    # ── Predict probabilities on the held-out test set ─────────────────────────
    # predict_proba returns [[p_neg, p_pos], ...]; we take the positive column.
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Find the optimal decision threshold ───────────────────────────────────
    threshold = find_best_threshold(y_test, y_prob)

    return model, y_prob, y_test, threshold


# ── Model Persistence ──────────────────────────────────────────────────────────

def save_model(model: XGBClassifier, out_dir: str, model_name: str) -> str:
    """
    Saves the trained model to disk using pickle.
    .pkl is chosen here because Phase 4's predict.py and Phase 6's
    Streamlit app will both just do pickle.load() — no extra
    dependencies, no format negotiation, no version friction.

    The XGBoost native .json format is more portable across XGBoost
    versions, but adds complexity to the load side. For a self-contained
    hackathon repo where train and predict run on the same machine,
    .pkl is strictly simpler.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def save_metrics(metrics_list: list[dict], out_dir: str) -> str:
    """
    Saves all three assays' metrics and optimal thresholds to a
    single JSON file. Phase 4 will load this file to look up the
    per-assay threshold instead of hardcoding 0.5.
    """
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training(data_dir: str, out_dir: str) -> None:

    print_section("ToxPathGuard — Phase 3: Model Training")

    all_metrics = []

    for assay, model_name in ASSAY_CONFIG.items():
        print_section(f"Assay: {assay}  →  {model_name}.pkl")

        # ── Load processed features ────────────────────────────────────────────
        X, y = load_assay(data_dir, assay)
        n_total = len(y)
        n_pos   = int(y.sum())
        n_neg   = n_total - n_pos
        print(f"  Total samples   : {n_total:,}")
        print(f"  Positives       : {n_pos:,}  ({n_pos/n_total:.1%})")
        print(f"  Negatives       : {n_neg:,}  ({n_neg/n_total:.1%})")

        # ── Train ──────────────────────────────────────────────────────────────
        model, y_prob, y_test, threshold = train_one_model(assay, X, y)

        # ── Evaluate ───────────────────────────────────────────────────────────
        metrics = report_metrics(assay, y_test, y_prob, threshold)
        all_metrics.append(metrics)

        # ── Save model ─────────────────────────────────────────────────────────
        model_path = save_model(model, out_dir, model_name)
        print(f"\n  Saved → {model_path}")

    # ── Save combined metrics ──────────────────────────────────────────────────
    metrics_path = save_metrics(all_metrics, out_dir)

    print_section("Training Complete")
    print(f"\n  {'Assay':<12} {'PR-AUC':>8} {'ROC-AUC':>9} {'F1':>7} {'Threshold':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*7} {'─'*11}")
    for m in all_metrics:
        print(
            f"  {m['assay']:<12} "
            f"{m['pr_auc']:>8.4f} "
            f"{m['roc_auc']:>9.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['threshold']:>11.4f}"
        )

    print(f"\n  Metrics saved → {metrics_path}")
    print(f"\n  All three models ready. Proceed to Phase 4 (prediction layer).")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 3 — Model Training"
    )
    parser.add_argument(
        "--data", type=str, default="data/processed",
        help="Directory containing the .npz feature files (default: data/processed)"
    )
    parser.add_argument(
        "--out", type=str, default="models",
        help="Output directory for trained models (default: models)"
    )
    args = parser.parse_args()
    run_training(args.data, args.out)
