"""
Phase 5 — ToxPathGuard Explainability Layer
---------------------------------------------
Computes SHAP values for each of the three XGBoost agents and
returns structured, human-readable feature contributions.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
score for a specific prediction. For XGBoost, SHAP values are computed
exactly using the tree structure — not approximated. This is the most
reliable explainability method for tree-based models.

What this module does NOT claim:
  - SHAP values explain causation, not just model behaviour.
  - High SHAP magnitude means the feature drove the model's score,
    not that the feature is biologically causal.
  - This is a decision-support tool. Communicate it as such.

Usage (standalone test):
    python src/explain.py --smiles "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
"""

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import shap
from features import smiles_to_mol, featurise_molecule, feature_names


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODELS_DIR = "models"

# How many top features to surface per direction (risk-driving / protective).
TOP_N = 8

# The 1030 feature names from Phase 2.
# 0–1023  : Morgan fingerprint bits
# 1024–1029 : physicochemical descriptors
FEATURE_NAMES = feature_names()

# Indices of the six physicochemical descriptors (non-fingerprint features).
# Named explicitly so the UI and summary text can distinguish them.
DESC_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RingCount"]
DESC_START  = 1024   # first descriptor index in the feature vector

AGENT_CONFIG = {
    "SR-p53": {"agent_name": "Agent 1 — DNA / Genotoxic Stress",  "model_file": "agent_p53.pkl"},
    "SR-ARE": {"agent_name": "Agent 2 — Oxidative Stress",         "model_file": "agent_are.pkl"},
    "SR-HSE": {"agent_name": "Agent 3 — Cellular / Heat-Shock Stress", "model_file": "agent_hse.pkl"},
}


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model(assay: str, models_dir: str = DEFAULT_MODELS_DIR):
    """Loads one XGBoost model from disk by assay name."""
    cfg  = AGENT_CONFIG[assay]
    path = os.path.join(models_dir, cfg["model_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: '{path}'. Run src/train.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models(models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    return {assay: load_model(assay, models_dir) for assay in AGENT_CONFIG}


# ── SHAP Computation ───────────────────────────────────────────────────────────

# Module-level explainer cache — built once per model per process.
# TreeExplainer construction is fast (~50ms per model) but there is no
# reason to rebuild it on every prediction call.
_explainer_cache: dict = {}


def get_explainer(assay: str, model) -> shap.TreeExplainer:
    """
    Returns a cached SHAP TreeExplainer for the given assay model.

    TreeExplainer is the correct choice for XGBoost — it uses the exact
    tree path algorithm rather than sampling-based approximations like
    KernelExplainer. Results are deterministic and fast (~1–5ms per sample).

    feature_perturbation="interventional" is the safer setting for
    correlated features (fingerprint bits are correlated by construction).
    """
    if assay not in _explainer_cache:
        _explainer_cache[assay] = shap.TreeExplainer(
            model,
            feature_perturbation="interventional",
        )
    return _explainer_cache[assay]


def compute_shap_values(
    assay:   str,
    model,
    X_row:   np.ndarray,
) -> np.ndarray:
    """
    Computes SHAP values for a single molecule (one row).

    X_row shape: (1, 1030) — must match training feature dimensionality exactly.

    Returns a 1-D array of length 1030 where each element is the SHAP
    contribution of that feature to the log-odds of the positive (toxic) class.
    Positive SHAP → pushes toward toxic.
    Negative SHAP → pushes toward non-toxic.

    We use shap_values[..., 1] because XGBoost's predict_proba returns
    [prob_class_0, prob_class_1] and SHAP mirrors this layout.
    """
    explainer   = get_explainer(assay, model)
    shap_output = explainer.shap_values(X_row)

    # XGBoost binary classifiers return either:
    #   - a single 2-D array (shape: n_samples × n_features) for the positive class, or
    #   - a list of two arrays [class_0_shap, class_1_shap]
    # We normalise both cases to a 1-D vector for the positive class.
    if isinstance(shap_output, list):
        # List format: index 1 is the positive class
        values = shap_output[1][0]
    else:
        values = shap_output[0]

    return values.astype(np.float32)


# ── Feature Contribution Extraction ───────────────────────────────────────────

def top_contributions(
    shap_values:    np.ndarray,
    feature_values: np.ndarray,
    n:              int = TOP_N,
) -> tuple[list[dict], list[dict]]:
    """
    Splits SHAP values into the top-N risk-driving features (positive SHAP)
    and top-N protective features (negative SHAP), sorted by magnitude.

    Each entry in the returned lists contains:
        feature_name  — human-readable name from features.py
        feature_value — the actual value of the feature for this molecule
        shap_value    — signed SHAP contribution
        direction     — "risk" or "protective"
        is_descriptor — True if the feature is a physicochemical descriptor,
                        False if it is a Morgan fingerprint bit

    Fingerprint bits and descriptors are pooled together. This is correct
    for XGBoost — it sees them as one flat feature vector and its SHAP
    values already account for their respective scales.
    """
    indices    = np.arange(len(shap_values))
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]   # highest |SHAP| first

    risk_features       = []
    protective_features = []

    for idx in sorted_idx:
        name        = FEATURE_NAMES[idx]
        shap_val    = float(shap_values[idx])
        feat_val    = float(feature_values[idx])
        is_desc     = idx >= DESC_START

        entry = {
            "feature_name":  name,
            "feature_value": round(feat_val, 4),
            "shap_value":    round(shap_val, 4),
            "is_descriptor": is_desc,
        }

        if shap_val > 0 and len(risk_features) < n:
            entry["direction"] = "risk"
            risk_features.append(entry)
        elif shap_val < 0 and len(protective_features) < n:
            entry["direction"] = "protective"
            protective_features.append(entry)

        if len(risk_features) >= n and len(protective_features) >= n:
            break

    return risk_features, protective_features


def descriptor_summary(shap_values: np.ndarray, feature_values: np.ndarray) -> list[dict]:
    """
    Returns the SHAP contributions for the six physicochemical descriptors only.
    These are always surfaced in the UI because judges and chemists understand
    LogP, TPSA, etc. — they are not opaque bit indices.
    """
    result = []
    for i, name in enumerate(DESC_NAMES):
        idx = DESC_START + i
        result.append({
            "feature_name":  name,
            "feature_value": round(float(feature_values[idx]), 4),
            "shap_value":    round(float(shap_values[idx]), 4),
            "direction":     "risk" if shap_values[idx] > 0 else "protective",
        })
    # Sort by |SHAP| descending so the most influential descriptor is first.
    return sorted(result, key=lambda x: abs(x["shap_value"]), reverse=True)


# ── Summary Text Builder ───────────────────────────────────────────────────────

def _feature_display_name(entry: dict) -> str:
    """
    Converts a feature dict into a short human-readable label.
    Fingerprint bits are labelled as 'structural pattern [bit N]' because
    their raw index means nothing to a non-ML audience.
    Descriptors use their chemical name directly.
    """
    if entry["is_descriptor"]:
        name  = entry["feature_name"]
        value = entry["feature_value"]
        # Add units / context for the descriptors judges know.
        if name == "LogP":
            return f"LogP ({value:.2f})"
        elif name == "MolWt":
            return f"Molecular Weight ({value:.1f} Da)"
        elif name == "TPSA":
            return f"TPSA ({value:.1f} Å²)"
        elif name == "HBD":
            return f"H-bond donors ({int(value)})"
        elif name == "HBA":
            return f"H-bond acceptors ({int(value)})"
        elif name == "RingCount":
            return f"Ring count ({int(value)})"
        return name
    else:
        # For fingerprint bits, strip the 'morgan_bit_' prefix.
        bit_idx = entry["feature_name"].replace("morgan_bit_", "bit ")
        return f"structural pattern [{bit_idx}]"


def build_agent_summary(
    agent_name:         str,
    probability:        float,
    risk_features:      list[dict],
    protective_features: list[dict],
    descriptor_contribs: list[dict],
) -> str:
    """
    Builds a one-paragraph natural-language explanation for one agent.
    Rule-based — no LLM, fully deterministic.
    """
    lines = []

    # ── Sentence 1: overall score ──────────────────────────────────────────────
    if probability > 0.70:
        severity = "a strong risk signal"
    elif probability > 0.35:
        severity = "a moderate risk signal"
    else:
        severity = "a low risk signal"

    lines.append(
        f"{agent_name} assigned {severity} (probability {probability:.1%})."
    )

    # ── Sentence 2: top risk drivers ──────────────────────────────────────────
    if risk_features:
        # Prefer descriptors in the summary if they appear — they're more legible.
        desc_risks = [f for f in risk_features if f["is_descriptor"]]
        fp_risks   = [f for f in risk_features if not f["is_descriptor"]]

        if desc_risks:
            top = _feature_display_name(desc_risks[0])
            lines.append(f"The primary risk driver from chemical properties was {top}.")
        if fp_risks:
            n_fp = len(fp_risks)
            lines.append(
                f"{n_fp} structural pattern(s) in the Morgan fingerprint "
                f"contributed positively to the risk score."
            )

    # ── Sentence 3: protective features ───────────────────────────────────────
    if protective_features:
        desc_prot = [f for f in protective_features if f["is_descriptor"]]
        if desc_prot:
            top_prot = _feature_display_name(desc_prot[0])
            lines.append(f"The most protective chemical property was {top_prot}.")

    # ── Sentence 4: descriptor highlights ─────────────────────────────────────
    high_impact_descs = [d for d in descriptor_contribs if abs(d["shap_value"]) > 0.01]
    if high_impact_descs:
        names = ", ".join(_feature_display_name({**d, "is_descriptor": True})
                          for d in high_impact_descs[:3])
        lines.append(f"Physicochemical properties with notable influence: {names}.")

    return " ".join(lines)


# ── Per-Agent Explanation ──────────────────────────────────────────────────────

def explain_agent(
    assay:          str,
    model,
    feature_vector: np.ndarray,
    probability:    float,
) -> dict:
    """
    Produces the full explanation block for one agent / assay.

    Args:
        assay:          e.g. "SR-p53"
        model:          loaded XGBoost model object
        feature_vector: 1-D float32 array of length 1030
        probability:    the model's predicted probability for the toxic class

    Returns a dict matching the Phase 5 output contract.
    """
    X_row = feature_vector.reshape(1, -1)
    shap_values = compute_shap_values(assay, model, X_row)

    risk_features, protective_features = top_contributions(
        shap_values, feature_vector, n=TOP_N
    )
    descriptor_contribs = descriptor_summary(shap_values, feature_vector)

    # All contributions sorted by |SHAP|, for a full ranked list.
    all_contributions = sorted(
        [
            {
                "feature_name":  FEATURE_NAMES[i],
                "feature_value": round(float(feature_vector[i]), 4),
                "shap_value":    round(float(shap_values[i]), 4),
                "is_descriptor": i >= DESC_START,
            }
            for i in range(len(shap_values))
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )[:TOP_N * 2]   # keep top 2×N for the UI to choose from

    cfg          = AGENT_CONFIG[assay]
    agent_name   = cfg["agent_name"]
    summary_text = build_agent_summary(
        agent_name, probability, risk_features, protective_features, descriptor_contribs
    )

    return {
        "assay":                 assay,
        "agent_name":            agent_name,
        "probability":           round(probability, 4),
        "shap_base_value":       None,   # set below after explainer call
        "top_risk_features":     risk_features,
        "top_protective_features": protective_features,
        "descriptor_contribs":   descriptor_contribs,
        "top_contributions":     all_contributions,
        "summary_text":          summary_text,
    }


# ── Full Molecule Explanation ──────────────────────────────────────────────────

def explain_molecule(
    smiles:       str,
    models:       dict,
    agent_probs:  dict[str, float],
    models_dir:   str = DEFAULT_MODELS_DIR,
) -> dict:
    """
    Runs the full explainability pipeline for one molecule across all three agents.

    Args:
        smiles:      the SMILES string (already validated upstream)
        models:      dict mapping assay → loaded XGBoost model
        agent_probs: dict mapping assay → predicted probability (from predict.py)
        models_dir:  path to models directory (unused if models are pre-loaded)

    Returns:
        {
            "smiles":          str,
            "agent_explanations": [ <per-agent explanation dict>, ... ],
            "overall_summary": str,
        }
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "smiles":             smiles,
            "agent_explanations": [],
            "overall_summary":    "Explanation unavailable: invalid SMILES.",
        }

    feature_vector = featurise_molecule(mol)
    if feature_vector is None:
        return {
            "smiles":             smiles,
            "agent_explanations": [],
            "overall_summary":    "Explanation unavailable: featurisation failed.",
        }

    agent_explanations = []
    for assay in AGENT_CONFIG:
        prob        = agent_probs.get(assay, 0.0)
        model       = models[assay]
        explanation = explain_agent(assay, model, feature_vector, prob)
        agent_explanations.append(explanation)

    overall_summary = _build_overall_summary(agent_explanations)

    return {
        "smiles":             smiles,
        "agent_explanations": agent_explanations,
        "overall_summary":    overall_summary,
    }


def _build_overall_summary(agent_explanations: list[dict]) -> str:
    """
    Combines the three per-agent summaries into a compact paragraph
    for the judge-facing summary panel.
    """
    if not agent_explanations:
        return "No explanation available."

    # Find the highest-probability agent.
    primary = max(agent_explanations, key=lambda e: e["probability"])

    # Collect all descriptor-level risk drivers across all agents.
    all_desc_risks = []
    for exp in agent_explanations:
        for f in exp["top_risk_features"]:
            if f["is_descriptor"]:
                all_desc_risks.append(f)

    lines = [
        f"Across all three pathway screens, the dominant risk signal came from "
        f"{primary['agent_name']} (probability {primary['probability']:.1%})."
    ]

    if all_desc_risks:
        # Deduplicate by feature name and pick the one with highest |SHAP|.
        seen = {}
        for f in all_desc_risks:
            name = f["feature_name"]
            if name not in seen or abs(f["shap_value"]) > abs(seen[name]["shap_value"]):
                seen[name] = f
        top_descs = sorted(seen.values(), key=lambda x: abs(x["shap_value"]), reverse=True)[:2]
        desc_names = " and ".join(
            _feature_display_name({**d, "is_descriptor": True}) for d in top_descs
        )
        lines.append(
            f"The most influential chemical properties across the audit were {desc_names}."
        )

    lines.append(
        "Structural pattern contributions from the Morgan fingerprint "
        "indicate that specific local chemical substructures — not just "
        "bulk properties — drove the model's assessment."
    )

    return " ".join(lines)


# ── Optional: SHAP Bar Chart ──────────────────────────────────────────────────

def save_shap_bar_chart(
    explanation:  dict,
    out_dir:      str = "assets",
    filename:     str | None = None,
) -> str | None:
    """
    Saves a horizontal bar chart of SHAP contributions for one agent
    to disk as a PNG. Returns the file path, or None if matplotlib
    is not available or the chart fails.

    This is intentionally wrapped in a broad try/except. If the chart
    fails — missing matplotlib, display issues on a headless server,
    any RDKit rendering issue — the prediction pipeline continues normally.
    The chart is a bonus, not a hard dependency.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for servers
        import matplotlib.pyplot as plt

        top_contribs = explanation.get("top_contributions", [])[:12]
        if not top_contribs:
            return None

        names  = [c["feature_name"].replace("morgan_bit_", "bit ") for c in top_contribs]
        values = [c["shap_value"] for c in top_contribs]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP contribution (positive = risk-driving)")
        ax.set_title(
            f"{explanation['agent_name']}\n"
            f"Probability: {explanation['probability']:.1%}",
            fontsize=10,
        )
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        assay_safe = explanation["assay"].replace("-", "_")
        fname      = filename or f"shap_{assay_safe}.png"
        out_path   = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    except Exception as e:
        # Log but never crash the pipeline over a chart.
        print(f"  [explain] SHAP bar chart skipped: {e}")
        return None


# ── Pretty Printer ─────────────────────────────────────────────────────────────

def print_explanation(explanation_result: dict) -> None:
    """Readable CLI output for one full molecule explanation."""
    W = 70
    print(f"\n{'═' * W}")
    print(f"  ToxPathGuard — SHAP Explanation Report")
    print(f"{'═' * W}")
    print(f"  SMILES: {explanation_result['smiles']}")

    for exp in explanation_result["agent_explanations"]:
        print(f"\n  {'─' * (W-2)}")
        print(f"  {exp['agent_name']}  (p = {exp['probability']:.4f})")
        print(f"  {'─' * (W-2)}")

        print(f"\n  Top risk-driving features:")
        for f in exp["top_risk_features"][:5]:
            tag = "[descriptor]" if f["is_descriptor"] else "[fingerprint]"
            print(f"    +{f['shap_value']:>7.4f}  {f['feature_name']:<28} {tag}")

        print(f"\n  Top protective features:")
        for f in exp["top_protective_features"][:5]:
            tag = "[descriptor]" if f["is_descriptor"] else "[fingerprint]"
            print(f"    {f['shap_value']:>8.4f}  {f['feature_name']:<28} {tag}")

        print(f"\n  Descriptor breakdown:")
        for d in exp["descriptor_contribs"]:
            bar = "▲" if d["shap_value"] > 0 else "▼"
            print(
                f"    {bar} {d['feature_name']:<12}"
                f"  value={d['feature_value']:>8.3f}"
                f"  SHAP={d['shap_value']:>+8.4f}"
            )

        print(f"\n  Summary: {exp['summary_text']}")

    print(f"\n{'─' * W}")
    print(f"  Overall: {explanation_result['overall_summary']}")
    print(f"{'═' * W}\n")


# ── CLI Test Mode ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 5 — SHAP Explainability Test"
    )
    parser.add_argument("--smiles",  type=str, required=True)
    parser.add_argument("--models",  type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--chart",   action="store_true",
                        help="Save SHAP bar charts to assets/")
    args = parser.parse_args()

    # Load models and thresholds via predict.py's loader
    sys.path.insert(0, os.path.dirname(__file__))
    from predict import load_pipeline, predict_single

    print("Loading pipeline...")
    models, thresholds = load_pipeline(args.models)
    print("Ready.\n")

    # Run prediction to get probabilities
    pred_result  = predict_single(args.smiles, models, thresholds)
    agent_probs  = {
        a["assay"]: a["probability"]
        for a in pred_result.get("agents", [])
    }

    # Run explanation
    exp_result = explain_molecule(args.smiles, models, agent_probs, args.models)
    print_explanation(exp_result)

    # Optionally save charts
    if args.chart:
        for exp in exp_result["agent_explanations"]:
            path = save_shap_bar_chart(exp)
            if path:
                print(f"  Chart saved → {path}")
