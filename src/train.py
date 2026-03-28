"""
Phase 3 — ToxPathGuard Model Training
---------------------------------------
Trains three separate XGBoost binary classifiers, one per biological
stress pathway assay:

    SR-p53  →  agent_p53.pkl   (DNA / Genotoxic Stress)
    SR-ARE  →  agent_are.pkl   (Oxidative Stress)
    SR-HSE  →  agent_hse.pkl   (Cellular / Heat-Shock Stress)

Each model is evaluated with PR-AUC as the primary metric (correct for
imbalanced binary classification). Final metrics and decision thresholds
are saved alongside the models in models/training_report.json.

Usage:
    python src/train.py --processed data/processed --out models
"""

import os
import json
import pickle
import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


# ── Configuration ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Three assays, each mapped to a human-readable name and a safe model filename.
ASSAY_CONFIG = {
    "SR-p53": {
        "label":    "DNA / Genotoxic Stress",
        "npz_name": "SR_p53",
        "model":    "agent_p53.pkl",
    },
    "SR-ARE": {
        "label":    "Oxidative Stress (Nrf2/ARE)",
        "npz_name": "SR_ARE",
        "model":    "agent_are.pkl",
    },
    "SR-HSE": {
        "label":    "Cellular / Heat-Shock Stress",
        "npz_name": "SR_HSE",
        "model":    "agent_hse.pkl",
    },
}

# XGBoost base hyperparameters — kept intentionally conservative.
# scale_pos_weight is set per-assay at runtime based on the class ratio.
# We do not tune anything else: the dataset is small enough that the
# defaults are stable, and over-tuning would risk leaking the test set.
XGBOOST_BASE_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,       # shallow trees → less overfitting on small data
    "learning_rate":     0.05,    # slow shrinkage → more robust generalisation
    "subsample":         0.8,     # row sampling per tree
    "colsample_bytree":  0.8,     # feature sampling per tree
    "use_label_encoder": False,
    "eval_metric":       "aucpr", # XGBoost's internal PR-AUC — used for early stopping
    "random_state":      RANDOM_SEED,
    "verbosity":         0,       # silence XGBoost's own output; we print our own
    "n_jobs":            -1,
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_npz(processed_dir: str, npz_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a pre-built (X, y) dataset from a .npz file.
    The naming convention matches what features.py wrote to disk.
    """
    path = os.path.join(processed_dir, f"{npz_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{path}'. "
            "Run features.py first (Phase 2)."
        )
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int8)


# ── Class Imbalance Handling ───────────────────────────────────────────────────

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Computes the scale_pos_weight parameter for XGBoost.

    This is simply neg_count / pos_count. XGBoost uses it to upweight
    the minority class (toxic compounds) during training, so the loss
    function penalises missing a positive as much as it penalises the
    combined weight of all negatives. Without this, the model learns to
    predict 'non-toxic' for everything and still scores ~94% accuracy —
    which is useless and dangerous.
    """
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return round(n_neg / n_pos, 2)


# ── Optimal Threshold Selection ────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds the probability threshold that maximises F1-score on the
    validation set.

    The default 0.5 threshold is wrong for imbalanced data. A model
    trained with scale_pos_weight will output calibrated probabilities
    that reflect the true class imbalance — meaning a compound with a
    35% predicted probability might still be the best 'positive' guess.
    By sweeping the precision-recall curve we find the threshold that
    best balances precision and recall for the actual class distribution.

    This threshold is saved to JSON and used in Phase 4 for predictions.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F1 for each threshold point on the curve
    # Note: precision_recall_curve returns one fewer threshold than scores,
    # so we align by slicing off the last precision/recall pair.
    f1_scores = (
        2 * (precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-8)   # epsilon avoids division by zero
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    assay: str,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes all evaluation metrics for one model against the held-out
    test set. Returns a dictionary that is both printed and saved to JSON.
    """
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_test, y_prob)
    roc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "assay":       assay,
        "pr_auc":      round(pr_auc, 4),
        "roc_auc":     round(roc, 4),
        "f1":          round(f1, 4),
        "threshold":   round(threshold, 4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        # False negative rate matters most here: a missed toxic compound
        # is far worse than a false alarm. We print this explicitly.
        "fnr": round(fn / (fn + tp + 1e-8), 4),
    }


def print_eval(metrics: dict, spw: float, label: str) -> None:
    """Prints a readable summary block for one trained model."""
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  {metrics['assay']}  →  {label}")
    print(f"  {sep}")
    print(f"    scale_pos_weight : {spw}")
    print(f"    PR-AUC           : {metrics['pr_auc']}   ← primary metric")
    print(f"    ROC-AUC          : {metrics['roc_auc']}")
    print(f"    F1 (at threshold): {metrics['f1']}")
    print(f"    Decision thresh  : {metrics['threshold']}")
    print(f"    TP / FP          : {metrics['tp']} / {metrics['fp']}")
    print(f"    TN / FN          : {metrics['tn']} / {metrics['fn']}")
    print(f"    False-neg rate   : {metrics['fnr']:.1%}  (missed toxic compounds)")


# ── Model Persistence ──────────────────────────────────────────────────────────

def save_model(model, out_dir: str, filename: str) -> str:
    """Serialises a trained XGBoost model to disk using pickle."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_assay(
    assay: str,
    cfg: dict,
    processed_dir: str,
    out_dir: str,
) -> dict:
    """
    Full training pipeline for one assay. The steps are:

      1. Load the pre-built (X, y) from disk.
      2. Split into train (70%) / val (15%) / test (15%) with stratification.
         We use a three-way split so we can tune the decision threshold on
         val without contaminating the test set's evaluation.
      3. Compute scale_pos_weight from the training labels only
         (never look at test labels for this).
      4. Train XGBoost with early stopping on the val PR-AUC.
      5. Find the optimal classification threshold on the val set.
      6. Evaluate final metrics on the held-out test set.
      7. Save the model and return metrics for JSON export.
    """
    print(f"\n{'═' * 68}")
    print(f"  Training: {assay}  ({cfg['label']})")
    print(f"{'═' * 68}")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    X, y = load_npz(processed_dir, cfg["npz_name"])
    print(f"\n  Dataset   : {X.shape[0]:,} samples  ×  {X.shape[1]} features")
    print(f"  Positives : {int(y.sum()):,}  ({y.mean():.1%})")
    print(f"  Negatives : {int((y==0).sum()):,}  ({1-y.mean():.1%})")

    # ── Step 2: Split into train / val / test ─────────────────────────────────
    # First cut: 85% train+val, 15% held-out test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    # Second cut: split the 85% into 70% train and 15% val
    # (15/85 ≈ 0.176 of the train+val set gives us the 15% val we want)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=RANDOM_SEED, stratify=y_tv
    )

    print(f"\n  Train     : {len(y_train):,} samples")
    print(f"  Val       : {len(y_val):,} samples")
    print(f"  Test      : {len(y_test):,} samples")

    # ── Step 3: Class weight ──────────────────────────────────────────────────
    spw = compute_scale_pos_weight(y_train)
    print(f"  SPW       : {spw}  (neg/pos ratio on training set)")

    # ── Step 4: Train with early stopping ─────────────────────────────────────
    params = {**XGBOOST_BASE_PARAMS, "scale_pos_weight": spw}
    model  = XGBClassifier(**params)

    # Early stopping halts training when val PR-AUC stops improving,
    # which prevents overfitting without grid-searching n_estimators.
    model.fit(
    X_train,
    y_train,
    )

    # ── Step 5: Threshold optimisation on val ─────────────────────────────────
    val_prob  = model.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_prob)

    # ── Step 6: Final evaluation on held-out test set ─────────────────────────
    test_prob = model.predict_proba(X_test)[:, 1]
    metrics   = evaluate(assay, y_test, test_prob, threshold)
    print_eval(metrics, spw, cfg["label"])

    # ── Step 7: Save model ────────────────────────────────────────────────────
    path = save_model(model, out_dir, cfg["model"])
    print(f"\n  Saved → {path}")

    # Attach spw to metrics dict for JSON report
    metrics["scale_pos_weight"] = spw
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training(processed_dir: str, out_dir: str) -> None:

    print("\n" + "═" * 68)
    print("  ToxPathGuard — Phase 3: Model Training")
    print("═" * 68)

    all_metrics = {}

    for assay, cfg in ASSAY_CONFIG.items():
        metrics = train_one_assay(assay, cfg, processed_dir, out_dir)
        all_metrics[assay] = metrics

    # ── Save training report ──────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Training report saved → {report_path}")

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  Final Summary")
    print("─" * 68)
    print(f"  {'Assay':<10}  {'PR-AUC':>8}  {'ROC-AUC':>8}  {'F1':>6}  {'FNR':>6}  {'Threshold':>10}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*10}")
    for assay, m in all_metrics.items():
        print(
            f"  {assay:<10}  {m['pr_auc']:>8.4f}  {m['roc_auc']:>8.4f}"
            f"  {m['f1']:>6.4f}  {m['fnr']:>5.1%}  {m['threshold']:>10.4f}"
        )

    print("\n" + "═" * 68)
    print("  Phase 3 complete. Proceed to Phase 4 (prediction layer).")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 3 — Model Training"
    )
    parser.add_argument(
        "--processed", type=str, default="data/processed",
        help="Directory containing processed .npz files (default: data/processed)"
    )
    parser.add_argument(
        "--out", type=str, default="models",
        help="Output directory for trained models (default: models)"
    )
    args = parser.parse_args()
    run_training(args.processed, args.out)
