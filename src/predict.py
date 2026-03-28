"""
Phase 4.1 — ToxPathGuard Prediction Layer (Updated)
------------------------------------------------------
Integrates the three XGBoost agents with the structural alert module,
applicability-domain check, and the coordinator reasoning layer.

This is the single integration point for all downstream consumers
(CLI, Streamlit UI, SHAP explainer in Phase 5).

Usage:
    python src/predict.py --smiles "CCO"
    python src/predict.py --smiles "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
    python src/predict.py --smiles "c1ccc(N)cc1"   # aniline — aromatic amine
    python src/predict.py --smiles "this-is-not-a-molecule"
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

# ── Internal imports (same-directory modules) ──────────────────────────────────
# Running from the project root requires src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from features  import smiles_to_mol, featurise_molecule, feature_names
from alerts    import check_structural_alerts, has_any_alert
from domain    import check_domain, MORGAN_NBITS
from reasoner  import run_coordinator, SAFETY_THRESHOLDS
from explain   import explain_molecule


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODELS_DIR   = "models"
DEFAULT_REPORT_PATH  = os.path.join(DEFAULT_MODELS_DIR, "training_report.json")
DEFAULT_DOMAIN_NPZ   = os.path.join("data", "processed", "SR_p53.npz")

AGENT_CONFIG = {
    "SR-p53": {
        "agent_name": "Agent 1 — DNA / Genotoxic Stress",
        "model_file": "agent_p53.pkl",
    },
    "SR-ARE": {
        "agent_name": "Agent 2 — Oxidative Stress",
        "model_file": "agent_are.pkl",
    },
    "SR-HSE": {
        "agent_name": "Agent 3 — Cellular / Heat-Shock Stress",
        "model_file": "agent_hse.pkl",
    },
}


# ── Model and Threshold Loading ────────────────────────────────────────────────

def load_models(models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    models = {}
    for assay, cfg in AGENT_CONFIG.items():
        path = os.path.join(models_dir, cfg["model_file"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: '{path}'. Run src/train.py first."
            )
        with open(path, "rb") as f:
            models[assay] = pickle.load(f)
    return models


def load_thresholds(report_path: str = DEFAULT_REPORT_PATH) -> dict:
    """
    Loads training (F1-optimised) thresholds from the training report.
    Safety thresholds are defined in reasoner.py — they do not come from
    training_report.json because they are policy decisions, not statistical
    outputs of the training process.
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Training report not found: '{report_path}'. Run src/train.py first."
        )
    with open(report_path, "r") as f:
        report = json.load(f)

    return {
        assay: float(report[assay]["threshold"])
        for assay in AGENT_CONFIG
        if assay in report
    }


def load_pipeline(
    models_dir:  str = DEFAULT_MODELS_DIR,
    report_path: str = DEFAULT_REPORT_PATH,
) -> tuple[dict, dict]:
    """
    Loads models and thresholds once. The Streamlit app should call this
    once at startup and store results in st.session_state.
    """
    return load_models(models_dir), load_thresholds(report_path)


# ── Core Prediction ────────────────────────────────────────────────────────────

def predict_single(
    smiles:         str,
    models:         dict,
    thresholds:     dict,
    threshold_mode: str  = "safety",   # "safety" for demo, "training" for analysis
    domain_npz:     str  = DEFAULT_DOMAIN_NPZ,
    explain:        bool = True,
) -> dict:
    """
    Full multi-agent toxicity audit for one SMILES string.

    Returns a structured dict with all agent results, coordinator verdict,
    structural alerts, domain status, and reason summary.

    threshold_mode:
        "safety"   — uses sensitivity-oriented thresholds (SAFETY_THRESHOLDS).
                     Recommended for demo and pre-screening use.
        "training" — uses the F1-optimised thresholds from training_report.json.
                     Useful for comparing against Phase 3 metrics.
    """
    result = {
        "smiles":          smiles,
        "valid":           False,
        "error":           None,
        "verdict":         "Invalid",
        "threshold_mode":  threshold_mode,
        "agents":          [],
        "coordinator":     {},
        "structural_alerts": [],
        "domain":          {},
        "reason_summary":  "",
        "warnings":        [],
    }

    # ── Step 1: Validate SMILES ────────────────────────────────────────────────
    mol = smiles_to_mol(smiles)
    if mol is None:
        result["error"] = (
            f"Invalid SMILES: '{smiles}'. "
            "RDKit could not parse this molecule."
        )
        return result

    result["valid"] = True

    # ── Step 2: Featurise ──────────────────────────────────────────────────────
    feature_vector = featurise_molecule(mol)
    if feature_vector is None:
        result["error"] = "Featurisation failed despite valid SMILES."
        result["valid"] = False
        return result

    X        = feature_vector.reshape(1, -1)
    fp_bits  = feature_vector[:MORGAN_NBITS]   # first 1024 bits for domain check

    # ── Step 3: Structural alerts ──────────────────────────────────────────────
    alert_matches = check_structural_alerts(mol)
    result["structural_alerts"] = alert_matches

    # ── Step 4: Applicability domain check ────────────────────────────────────
    domain_result = check_domain(fp_bits, npz_path=domain_npz)
    result["domain"] = domain_result

    if domain_result["low_confidence"]:
        result["warnings"].append(domain_result["domain_warning"])

    # ── Step 5: Run each agent ─────────────────────────────────────────────────
    agent_results = []

    for assay, cfg in AGENT_CONFIG.items():
        model              = models[assay]
        training_threshold = thresholds[assay]
        safety_threshold   = SAFETY_THRESHOLDS.get(assay, 0.30)

        probability    = float(model.predict_proba(X)[0, 1])

        # The active threshold depends on the selected mode.
        active_threshold = (
            safety_threshold if threshold_mode == "safety"
            else training_threshold
        )
        raw_prediction = int(probability >= active_threshold)

        # Three-zone label around the active threshold.
        if probability >= active_threshold + 0.20:
            risk_label = "High Risk"
        elif probability >= active_threshold:
            risk_label = "Borderline"
        else:
            risk_label = "Low Risk"

        agent_results.append({
            "assay":               assay,
            "agent_name":          cfg["agent_name"],
            "probability":         round(probability, 4),
            "training_threshold":  round(training_threshold, 4),
            "safety_threshold":    round(safety_threshold, 4),
            "active_threshold":    round(active_threshold, 4),
            "threshold_mode":      threshold_mode,
            "raw_prediction":      raw_prediction,
            "risk_label":          risk_label,
        })

    result["agents"] = agent_results

    # ── Step 6: Coordinator reasoning ─────────────────────────────────────────
    coordinator = run_coordinator(
        agent_results  = agent_results,
        alert_matches  = alert_matches,
        domain_result  = domain_result,
        threshold_mode = threshold_mode,
    )

    result["coordinator"]    = coordinator
    result["verdict"]        = coordinator["verdict"]
    result["reason_summary"] = coordinator["reason_summary"]

    # ── Step 7: SHAP explanation ───────────────────────────────────────────────
    result["explanation"] = {}
    if explain and result["valid"]:
        agent_probs = {a["assay"]: a["probability"] for a in agent_results}
        result["explanation"] = explain_molecule(
            smiles      = smiles,
            models      = models,
            agent_probs = agent_probs,
        )

    return result


# ── Pretty Printer ─────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """Full formatted audit report for CLI output."""
    W = 70
    print(f"\n{'═' * W}")
    print(f"  ToxPathGuard — Multi-Pathway Molecular Safety Audit")
    print(f"{'═' * W}")
    print(f"  SMILES         : {result['smiles']}")
    print(f"  Valid          : {result['valid']}")
    print(f"  Threshold mode : {result['threshold_mode']}")

    if not result["valid"]:
        print(f"\n  ERROR: {result['error']}")
        print(f"{'═' * W}\n")
        return

    coord = result["coordinator"]
    print(f"  Verdict        : {result['verdict']}")
    print(f"  Primary concern: {coord.get('primary_concern', '—')}")

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  ⚠  {w}")

    # ── Agent table ────────────────────────────────────────────────────────────
    print(f"\n  {'─' * (W-2)}")
    print(f"  {'Agent':<40} {'Prob':>6}  {'S-Thresh':>9}  {'Label':>12}  {'Fire':>4}")
    print(f"  {'─' * (W-2)}")
    for a in result["agents"]:
        print(
            f"  {a['agent_name']:<40} "
            f"{a['probability']:>6.4f}  "
            f"{a['safety_threshold']:>9.4f}  "
            f"{a['risk_label']:>12}  "
            f"{a['raw_prediction']:>4}"
        )

    # ── Structural alerts ──────────────────────────────────────────────────────
    print(f"\n  Structural Alerts ({len(result['structural_alerts'])} matched):")
    if result["structural_alerts"]:
        for al in result["structural_alerts"]:
            print(f"    ✗ {al['name']}")
            print(f"      {al['description']}")
    else:
        print("    ✓ No structural toxicophores detected.")

    # ── Alert–model concordance ────────────────────────────────────────────────
    conc = coord.get("concordance_label")
    if conc:
        print(f"\n  Concordance    : {conc}")

    # ── Domain ────────────────────────────────────────────────────────────────
    dom = result["domain"]
    print(f"\n  Domain status  : {dom.get('domain_status', '—')}  "
          f"(max Tanimoto: {dom.get('max_similarity', 0):.3f})")

    # ── Reason summary ─────────────────────────────────────────────────────────
    print(f"\n  Summary:")
    # Word-wrap the summary at 65 characters for readability.
    words  = result["reason_summary"].split()
    line   = "    "
    for word in words:
        if len(line) + len(word) + 1 > 67:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n  Confidence: {coord.get('confidence_note', '')}")
    print(f"{'═' * W}\n")


# ── CLI Test Mode ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 4.1 — Molecular Safety Audit"
    )
    parser.add_argument("--smiles",  type=str, required=True)
    parser.add_argument("--models",  type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report",  type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--mode",
        type=str,
        default="safety",
        choices=["safety", "training"],
        help="Threshold mode: 'safety' (low FNR) or 'training' (F1-optimal)"
    )
    args = parser.parse_args()

    print("Loading pipeline...")
    models, thresholds = load_pipeline(args.models, args.report)
    print("Ready.\n")

    result = predict_single(
        smiles         = args.smiles,
        models         = models,
        thresholds     = thresholds,
        threshold_mode = args.mode,
    )
    print_result(result)
