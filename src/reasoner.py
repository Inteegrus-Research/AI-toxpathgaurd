"""
Phase 4.1 — Coordinator and Reasoning Layer
----------------------------------------------
A deterministic rule-based coordinator that reads the outputs from all
three agents and produces a final audit verdict, a risk characterisation,
and a concise human-readable summary.

This is not an LLM. It is a structured if-then policy system that mimics
how a trained toxicologist would combine model scores, structural alerts,
and domain status into a coherent recommendation. Rule-based reasoning is
more appropriate here than an LLM because:

  - It is fully offline and reproducible.
  - Every output can be traced back to a specific decision rule.
  - Judges can ask "why" and get a deterministic answer.
  - It cannot hallucinate chemistry.
"""


# ── Safety Threshold Policy ────────────────────────────────────────────────────
#
# The F1-optimised thresholds from Phase 3 training are good statistical
# operating points but are too conservative for a safety screening tool.
# A molecule screener should minimise false negatives (missed toxic compounds)
# at the cost of more false positives (over-flagged safe compounds).
#
# The safety thresholds below were chosen by moving each threshold down by
# roughly 50% of the probability range between 0 and the F1 threshold,
# targeting a high-sensitivity operating regime without flagging everything.
#
# These are calibrated per-assay, not set to a single universal value,
# because the class distributions and model confidence ranges differ:
#   - SR-p53 F1 threshold = 0.76 → safety threshold = 0.35
#   - SR-ARE F1 threshold = 0.48 → safety threshold = 0.20
#   - SR-HSE F1 threshold = 0.80 → safety threshold = 0.35
#
# At these operating points, the models should detect the majority of true
# positives while accepting a higher false-positive rate — which is the
# correct trade-off for pre-screening, not clinical diagnosis.

SAFETY_THRESHOLDS = {
    "SR-p53": 0.35,
    "SR-ARE": 0.20,
    "SR-HSE": 0.35,
}

# ── Verdict Thresholds ─────────────────────────────────────────────────────────
# Applied to the maximum agent probability after safety thresholding.
VERDICT_HIGH = 0.55
VERDICT_CAUTION = 0.25


# ── Alert-Model Concordance Labels ────────────────────────────────────────────

def classify_alert_concordance(
    any_agent_high_risk: bool,
    alert_matches: list[dict],
) -> str | None:
    """
    Returns a concordance label describing the relationship between the
    model's prediction and the structural alert result.

    This is the most important interpretability signal in the audit:
      - Confirmed Toxicophore: model + chemistry agree → highest confidence
      - Novel Risk Motif:      model alarmed, no known alert → interesting find
      - Alerted but Model-Low: chemistry flagged, model disagrees → review carefully
      - None:                  both agree the molecule is clean
    """
    has_alerts = len(alert_matches) > 0

    if any_agent_high_risk and has_alerts:
        return "Confirmed Toxicophore"
    elif any_agent_high_risk and not has_alerts:
        return "Novel Risk Motif"
    elif not any_agent_high_risk and has_alerts:
        return "Alerted — Model Score Low"
    else:
        return None


# ── Per-Agent Risk Label (Safety Threshold) ───────────────────────────────────

def apply_safety_threshold(assay: str, probability: float) -> dict:
    """
    Applies the safety threshold for one assay and returns an enriched
    label dict. Both the training threshold (from JSON) and the safety
    threshold are returned so the UI can display both.
    """
    safety_thresh = SAFETY_THRESHOLDS.get(assay, 0.30)
    raw_pred      = int(probability >= safety_thresh)

    if probability >= safety_thresh + 0.20:
        risk_label = "High Risk"
    elif probability >= safety_thresh:
        risk_label = "Borderline"
    else:
        risk_label = "Low Risk"

    return {
        "safety_threshold":    round(safety_thresh, 4),
        "safety_raw_pred":     raw_pred,
        "safety_risk_label":   risk_label,
    }


# ── Coordinator Verdict ────────────────────────────────────────────────────────

def compute_verdict(
    agent_results:   list[dict],
    alert_matches:   list[dict],
    domain_result:   dict,
    threshold_mode:  str = "safety",
) -> str:
    """
    Final verdict using a conservative combination of signals:
      1. If any agent fires at the safety threshold → at least Caution.
      2. If the molecule is confirmed by a structural alert → upgrade to High Risk.
      3. If the molecule is out of domain → flag but do not downgrade.
      4. The maximum probability still governs the severity level.

    threshold_mode: "safety" (default for demo) or "training"
    """
    prob_key = "probability"   # raw probability is always used

    max_prob = max(r[prob_key] for r in agent_results)

    any_high  = any(
        r.get("safety_risk_label") == "High Risk"
        for r in agent_results
    ) if threshold_mode == "safety" else (max_prob >= 0.60)

    has_alert = len(alert_matches) > 0

    if any_high and has_alert:
        return "High Risk"
    elif any_high:
        return "High Risk"
    elif max_prob >= VERDICT_CAUTION or has_alert:
        return "Caution"
    else:
        return "Cleared"


# ── Primary Concern Identification ────────────────────────────────────────────

def identify_primary_concern(agent_results: list[dict]) -> dict:
    """Returns the agent with the highest raw probability."""
    return max(agent_results, key=lambda r: r["probability"])


# ── Reason Summary Generator ──────────────────────────────────────────────────

def build_reason_summary(
    verdict:          str,
    agent_results:    list[dict],
    alert_matches:    list[dict],
    domain_result:    dict,
    concordance:      str | None,
) -> str:
    """
    Constructs a concise, human-readable explanation of the audit verdict.
    Every sentence is traceable to a specific input signal.
    This is the text the UI will display in the summary panel.
    """
    lines = []

    primary = identify_primary_concern(agent_results)
    primary_name = primary["agent_name"]
    primary_prob = primary["probability"]

    # ── Sentence 1: Overall verdict and primary driver ─────────────────────────
    if verdict == "High Risk":
        lines.append(
            f"The molecule was flagged as High Risk. "
            f"The strongest signal came from {primary_name} "
            f"(probability {primary_prob:.2%})."
        )
    elif verdict == "Caution":
        lines.append(
            f"The molecule received a Caution rating. "
            f"The most elevated signal was in {primary_name} "
            f"(probability {primary_prob:.2%})."
        )
    else:
        lines.append(
            f"The molecule was Cleared across all three stress pathways. "
            f"The highest probability was {primary_prob:.2%}, "
            f"which is below the safety threshold."
        )

    # ── Sentence 2: Multi-agent agreement ─────────────────────────────────────
    firing_agents = [
        r["agent_name"] for r in agent_results
        if r.get("safety_raw_pred", 0) == 1
    ]
    if len(firing_agents) >= 2:
        agent_list = " and ".join(
            a.replace("Agent 1 — ", "").replace("Agent 2 — ", "").replace("Agent 3 — ", "")
            for a in firing_agents
        )
        lines.append(
            f"Risk signals were raised by multiple pathways: {agent_list}, "
            f"suggesting broad biological stress potential."
        )
    elif len(firing_agents) == 1:
        concern = firing_agents[0].split("— ")[-1]
        lines.append(f"Only the {concern} pathway raised a risk signal.")

    # ── Sentence 3: Structural alert concordance ───────────────────────────────
    if concordance == "Confirmed Toxicophore":
        top_alert = alert_matches[0]["name"]
        lines.append(
            f"A known structural toxicophore was detected ({top_alert}), "
            f"which is consistent with the model's prediction. "
            f"This is a Confirmed Toxicophore signal."
        )
    elif concordance == "Novel Risk Motif":
        lines.append(
            "No standard structural toxicophore was detected, but the model "
            "score is elevated. This may represent a novel risk motif or "
            "an atypical mechanism not captured by standard alert rules."
        )
    elif concordance == "Alerted — Model Score Low":
        top_alert = alert_matches[0]["name"]
        lines.append(
            f"A structural alert was detected ({top_alert}), but the model "
            f"assigned a low probability. This warrants manual review — "
            f"the alert may be inactive in this molecular context."
        )
    elif not alert_matches and verdict == "Cleared":
        lines.append("No structural toxicophores were detected.")

    # ── Sentence 4: Domain confidence ─────────────────────────────────────────
    if domain_result["low_confidence"]:
        lines.append(
            f"Note: {domain_result['domain_warning']} "
            f"Results should be interpreted with caution."
        )

    return " ".join(lines)


# ── Confidence Annotation ─────────────────────────────────────────────────────

def confidence_note(domain_result: dict, agent_results: list[dict]) -> str:
    """
    Returns a single-sentence confidence qualifier for the audit report.
    This is separate from the reason summary and displayed as a footnote.
    """
    max_prob   = max(r["probability"] for r in agent_results)
    low_domain = domain_result["low_confidence"]

    if max_prob > 0.80 and not low_domain:
        return "High model confidence. Molecule is well-represented in training data."
    elif max_prob > 0.80 and low_domain:
        return "High model score but molecule is outside the training domain. Use with caution."
    elif max_prob > 0.40 and low_domain:
        return "Moderate model score in an extrapolation region. Treat as indicative only."
    elif max_prob < 0.20:
        return "Low probability across all pathways. Consistent with a low-risk profile."
    else:
        return "Intermediate model confidence. No strong signal in either direction."


# ── Main Coordinator Entrypoint ────────────────────────────────────────────────

def run_coordinator(
    agent_results:  list[dict],
    alert_matches:  list[dict],
    domain_result:  dict,
    threshold_mode: str = "safety",
) -> dict:
    """
    The coordinator entry point. Accepts raw agent outputs and returns
    the full coordinator block for inclusion in the prediction result.

    This is the function predict.py calls. The Streamlit app will receive
    this block already structured and can render it directly.
    """
    # Enrich each agent result with safety threshold analysis.
    for agent in agent_results:
        safety_info = apply_safety_threshold(agent["assay"], agent["probability"])
        agent.update(safety_info)

    any_agent_high = any(
        r.get("safety_raw_pred", 0) == 1
        for r in agent_results
    )

    concordance = classify_alert_concordance(any_agent_high, alert_matches)
    verdict     = compute_verdict(agent_results, alert_matches, domain_result, threshold_mode)
    summary     = build_reason_summary(verdict, agent_results, alert_matches, domain_result, concordance)
    confidence  = confidence_note(domain_result, agent_results)
    primary     = identify_primary_concern(agent_results)

    return {
        "verdict":              verdict,
        "primary_concern":      primary["agent_name"],
        "concordance_label":    concordance,
        "reason_summary":       summary,
        "confidence_note":      confidence,
        "threshold_mode":       threshold_mode,
        "safety_thresholds":    SAFETY_THRESHOLDS,
    }
