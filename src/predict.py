"""
Phase 4 — ToxPathGuard Prediction Layer
-----------------------------------------
Accepts a SMILES string and runs it through all three trained XGBoost
classifiers to produce a structured toxicity risk audit.

This module is the single integration point between the raw molecule
input and every downstream consumer (CLI, Streamlit UI, SHAP explainer).
Nothing above this layer should ever touch model files or featurisation
logic directly.

Usage (CLI test mode):
    python src/predict.py --smiles "CCO"
    python src/predict.py --smiles "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

# We deliberately import from our own features module so the featurisation
# path is exactly identical to what was used during training. Any drift
# between training features and prediction features would silently produce
# wrong probabilities — this import makes that impossible.
from features import smiles_to_mol, featurise_molecule, feature_names


# ── Configuration ──────────────────────────────────────────────────────────────

# Default paths — can be overridden if the project moves.
DEFAULT_MODELS_DIR    = "models"
DEFAULT_REPORT_PATH   = os.path.join(DEFAULT_MODELS_DIR, "training_report.json")

# Maps each assay to its model filename and human-readable pathway label.
# This is the single source of truth for agent identity in the prediction layer.
AGENT_CONFIG = {
    "SR-p53": {
        "agent_name":  "Agent 1 — DNA / Genotoxic Stress",
        "model_file":  "agent_p53.pkl",
    },
    "SR-ARE": {
        "agent_name":  "Agent 2 — Oxidative Stress",
        "model_file":  "agent_are.pkl",
    },
    "SR-HSE": {
        "agent_name":  "Agent 3 — Cellular / Heat-Shock Stress",
        "model_file":  "agent_hse.pkl",
    },
}

# Verdict thresholds applied to the *maximum* probability across all agents.
# These are narrative thresholds for the overall verdict, separate from the
# per-assay decision thresholds which come from training_report.json.
VERDICT_HIGH_RISK = 0.60
VERDICT_CAUTION   = 0.35


# ── Model and Threshold Loading ────────────────────────────────────────────────

def load_models(models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """
    Loads all three trained XGBoost classifiers from disk into a dictionary
    keyed by assay name. Raises a clear FileNotFoundError immediately if any
    model is missing, rather than failing silently mid-prediction.

    We load all three models once at startup rather than re-loading per
    prediction call. When the Streamlit app uses this module, it will call
    load_models() once outside the prediction function and pass the result
    in — this avoids re-reading files from disk on every user interaction.
    """
    models = {}
    for assay, cfg in AGENT_CONFIG.items():
        path = os.path.join(models_dir, cfg["model_file"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: '{path}'\n"
                f"Run src/train.py first (Phase 3) to generate trained models."
            )
        with open(path, "rb") as f:
            models[assay] = pickle.load(f)
    return models


def load_thresholds(report_path: str = DEFAULT_REPORT_PATH) -> dict:
    """
    Loads per-assay decision thresholds from the training report JSON.
    These thresholds were selected during training to maximise F1 on the
    validation set — they are the correct operating points for each model
    and must not be hardcoded or approximated.

    Returns a simple dict mapping assay name → float threshold, e.g.:
        { "SR-p53": 0.7621, "SR-ARE": 0.4822, "SR-HSE": 0.7991 }
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Training report not found: '{report_path}'\n"
            f"Run src/train.py first (Phase 3) to generate the report."
        )
    with open(report_path, "r") as f:
        report = json.load(f)

    thresholds = {}
    for assay in AGENT_CONFIG:
        if assay not in report:
            raise KeyError(
                f"Assay '{assay}' not found in training report. "
                f"The report may be from an incomplete training run."
            )
        thresholds[assay] = float(report[assay]["threshold"])
    return thresholds


# ── Risk Label Assignment ──────────────────────────────────────────────────────

def probability_to_risk_label(probability: float, threshold: float) -> str:
    """
    Converts a raw probability and its assay-specific decision threshold
    into a human-readable risk label.

    We use three zones rather than a binary yes/no because the threshold
    represents the F1-optimal operating point — compounds near the threshold
    are genuinely uncertain and deserve a 'Borderline' label rather than a
    confident classification in either direction. This is scientifically
    more honest and also makes the UI more interpretable.

    The zones are:
        probability >= threshold + 0.10  →  High Risk
        probability >= threshold - 0.10  →  Borderline
        probability <  threshold - 0.10  →  Low Risk
    """
    if probability >= threshold + 0.10:
        return "High Risk"
    elif probability >= threshold - 0.10:
        return "Borderline"
    else:
        return "Low Risk"


def compute_overall_verdict(agent_results: list[dict]) -> str:
    """
    Derives the overall audit verdict from the three per-agent results.

    The logic treats the three checkpoints as independent stress tests:
    if any single checkpoint returns a high-risk signal, the molecule
    fails the audit. The maximum probability across all three agents is
    the most conservative (safest) summary statistic for a screening tool.

    Verdict levels:
        High Risk  — at least one agent probability >= VERDICT_HIGH_RISK
        Caution    — at least one agent probability >= VERDICT_CAUTION
        Cleared    — all agent probabilities below VERDICT_CAUTION
    """
    max_prob = max(r["probability"] for r in agent_results)

    if max_prob >= VERDICT_HIGH_RISK:
        return "High Risk"
    elif max_prob >= VERDICT_CAUTION:
        return "Caution"
    else:
        return "Cleared"


# ── Core Prediction Logic ──────────────────────────────────────────────────────

def predict_single(
    smiles: str,
    models: dict,
    thresholds: dict,
) -> dict:
    """
    Runs the full three-agent toxicity audit for one SMILES string.

    The function is intentionally side-effect free — it receives all
    dependencies (models, thresholds) as arguments so it can be called
    from a Streamlit app, a test harness, or a CLI without any global
    state. The Streamlit app will load models once, then call this
    function on every new user input.

    Return structure:
    {
        "smiles":   str,
        "valid":    bool,
        "error":    str | None,
        "verdict":  str,          # "Cleared" | "Caution" | "High Risk" | "Invalid"
        "agents": [
            {
                "assay":          str,
                "agent_name":     str,
                "probability":    float,
                "threshold":      float,
                "risk_label":     str,
                "raw_prediction": int,   # 0 or 1 at the stored threshold
            },
            ...  (three entries, one per agent)
        ],
        "warnings": [ str, ... ]
    }
    """
    result = {
        "smiles":   smiles,
        "valid":    False,
        "error":    None,
        "verdict":  "Invalid",
        "agents":   [],
        "warnings": [],
    }

    # ── Step 1: Validate SMILES ────────────────────────────────────────────────
    mol = smiles_to_mol(smiles)
    if mol is None:
        result["error"] = (
            f"Invalid SMILES string: '{smiles}'. "
            "RDKit could not parse this molecule. "
            "Check for typos or unsupported notation."
        )
        return result

    result["valid"] = True

    # ── Step 2: Featurise ──────────────────────────────────────────────────────
    feature_vector = featurise_molecule(mol)

    if feature_vector is None:
        # This should not happen after a valid mol, but we handle it defensively.
        result["error"] = "Featurisation failed despite a valid SMILES. Check RDKit installation."
        result["valid"] = False
        return result

    # XGBoost expects shape (1, n_features) for a single sample.
    X = feature_vector.reshape(1, -1)

    # ── Step 3: Run each agent ─────────────────────────────────────────────────
    agent_results = []

    for assay, cfg in AGENT_CONFIG.items():
        model     = models[assay]
        threshold = thresholds[assay]

        # predict_proba returns [[prob_class_0, prob_class_1]]
        # We take index 1 — the probability of the toxic class.
        probability    = float(model.predict_proba(X)[0, 1])
        raw_prediction = int(probability >= threshold)
        risk_label     = probability_to_risk_label(probability, threshold)

        agent_results.append({
            "assay":          assay,
            "agent_name":     cfg["agent_name"],
            "probability":    round(probability, 4),
            "threshold":      round(threshold, 4),
            "risk_label":     risk_label,
            "raw_prediction": raw_prediction,
        })

    result["agents"]  = agent_results
    result["verdict"] = compute_overall_verdict(agent_results)
    return result


# ── Pretty Printer ─────────────────────────────────────────────────────────────

def print_prediction_result(result: dict) -> None:
    """
    Prints a readable audit report to stdout.
    Used in CLI test mode and for development debugging.
    The Streamlit UI will consume the raw dictionary instead.
    """
    sep = "─" * 68
    print(f"\n{'═' * 68}")
    print(f"  ToxPathGuard — Molecular Safety Audit")
    print(f"{'═' * 68}")
    print(f"  SMILES  : {result['smiles']}")
    print(f"  Valid   : {result['valid']}")

    if not result["valid"]:
        print(f"\n  ERROR   : {result['error']}")
        print(f"{'═' * 68}\n")
        return

    print(f"  Verdict : {result['verdict']}")

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  WARNING : {w}")

    print(f"\n  {sep}")
    print(f"  {'Agent':<40} {'Prob':>6}  {'Threshold':>10}  {'Label':>12}  {'Raw':>4}")
    print(f"  {sep}")

    for agent in result["agents"]:
        print(
            f"  {agent['agent_name']:<40} "
            f"{agent['probability']:>6.4f}  "
            f"{agent['threshold']:>10.4f}  "
            f"{agent['risk_label']:>12}  "
            f"{agent['raw_prediction']:>4}"
        )

    print(f"  {sep}")
    print(f"\n  Overall verdict: {result['verdict']}")
    print(f"{'═' * 68}\n")


# ── Public Loader (for Streamlit) ──────────────────────────────────────────────

def load_pipeline(
    models_dir:   str = DEFAULT_MODELS_DIR,
    report_path:  str = DEFAULT_REPORT_PATH,
) -> tuple[dict, dict]:
    """
    Convenience function for the Streamlit app.
    Loads models and thresholds once and returns both.
    The app should call this once at startup (outside any button callback)
    and store the result in st.session_state to avoid reloading on every run.

    Example usage in streamlit_app.py:
        if "pipeline" not in st.session_state:
            models, thresholds = load_pipeline()
            st.session_state.models = models
            st.session_state.thresholds = thresholds

        result = predict_single(smiles, st.session_state.models,
                                         st.session_state.thresholds)
    """
    models     = load_models(models_dir)
    thresholds = load_thresholds(report_path)
    return models, thresholds


# ── CLI Test Mode ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 4 — Prediction Layer Test"
    )
    parser.add_argument(
        "--smiles", type=str, required=True,
        help="SMILES string to audit (wrap in quotes if it contains brackets)"
    )
    parser.add_argument(
        "--models", type=str, default=DEFAULT_MODELS_DIR,
        help="Path to models directory (default: models/)"
    )
    parser.add_argument(
        "--report", type=str, default=DEFAULT_REPORT_PATH,
        help="Path to training_report.json (default: models/training_report.json)"
    )
    args = parser.parse_args()

    # Load once, then predict.
    print("Loading models and thresholds...")
    models, thresholds = load_pipeline(args.models, args.report)
    print("Pipeline loaded.\n")

    result = predict_single(args.smiles, models, thresholds)
    print_prediction_result(result)
