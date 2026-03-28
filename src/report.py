"""
Phase 7 — ToxPathGuard Audit Report Generator
------------------------------------------------
Converts a full prediction result dictionary into a clean,
structured JSON report suitable for download or archiving.

Usage:
    from report import generate_report
    report_json = generate_report(prediction_result)
"""

import json
from datetime import datetime


def generate_report(result: dict) -> str:
    """
    Takes the full dict returned by predict_single() and produces
    a clean, self-contained JSON report string.

    Strips internal objects (model handles, numpy arrays) so the
    output is safely serialisable and human-readable.

    Returns a formatted JSON string ready for st.download_button().
    """
    agents_summary = []
    for agent in result.get("agents", []):
        agents_summary.append({
            "assay":              agent.get("assay"),
            "agent_name":        agent.get("agent_name"),
            "probability":       agent.get("probability"),
            "safety_threshold":  agent.get("safety_threshold"),
            "training_threshold":agent.get("training_threshold"),
            "risk_label":        agent.get("risk_label"),
            "raw_prediction":    agent.get("raw_prediction"),
        })

    # Extract per-agent SHAP top features if explanation is present
    explanation_summary = {}
    exp = result.get("explanation", {})
    for agent_exp in exp.get("agent_explanations", []):
        assay = agent_exp.get("assay", "unknown")
        explanation_summary[assay] = {
            "top_risk_features": [
                {
                    "feature": f["feature_name"],
                    "shap":    f["shap_value"],
                    "type":    "descriptor" if f["is_descriptor"] else "fingerprint",
                }
                for f in agent_exp.get("top_risk_features", [])[:5]
            ],
            "top_protective_features": [
                {
                    "feature": f["feature_name"],
                    "shap":    f["shap_value"],
                    "type":    "descriptor" if f["is_descriptor"] else "fingerprint",
                }
                for f in agent_exp.get("top_protective_features", [])[:5]
            ],
            "descriptor_contribs": agent_exp.get("descriptor_contribs", []),
            "summary_text":       agent_exp.get("summary_text", ""),
        }

    coordinator = result.get("coordinator", {})

    report = {
        "metadata": {
            "tool":        "ToxPathGuard — Multi-Pathway Molecular Safety Audit",
            "version":     "1.0.0-hackathon",
            "generated":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "disclaimer":  (
                "This report is produced by a pre-screening prototype trained on Tox21 "
                "stress-response assays. It is not a clinical diagnostic tool and must not "
                "be used for regulatory or medical decision-making."
            ),
        },
        "molecule": {
            "smiles":  result.get("smiles"),
            "valid":   result.get("valid"),
        },
        "audit": {
            "verdict":             result.get("verdict"),
            "threshold_mode":      result.get("threshold_mode"),
            "primary_concern":     coordinator.get("primary_concern"),
            "concordance_label":   coordinator.get("concordance_label"),
            "confidence_level":    coordinator.get("confidence_level", "—"),
            "reason_summary":      coordinator.get("reason_summary"),
            "confidence_note":     coordinator.get("confidence_note"),
        },
        "agents": agents_summary,
        "structural_alerts": [
            {
                "name":        al.get("name"),
                "description": al.get("description"),
            }
            for al in result.get("structural_alerts", [])
        ],
        "domain": {
            "status":         result.get("domain", {}).get("domain_status"),
            "max_similarity": result.get("domain", {}).get("max_similarity"),
            "warning":        result.get("domain", {}).get("domain_warning"),
            "low_confidence": result.get("domain", {}).get("low_confidence"),
        },
        "explanation": explanation_summary,
        "audit_trace": result.get("audit_trace", []),
        "warnings": result.get("warnings", []),
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
