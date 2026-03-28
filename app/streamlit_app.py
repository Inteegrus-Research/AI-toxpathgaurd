"""
Phase 6 — ToxPathGuard Streamlit UI
--------------------------------------
A fully offline, single-process Streamlit dashboard for the
ToxPathGuard multi-pathway molecular safety audit system.

Run from the project root:
    streamlit run app/streamlit_app.py

All prediction, explanation, and reasoning logic lives in src/.
This file only orchestrates layout and display.
"""

import os
import sys
import io
import base64

import streamlit as st
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
# Must happen before any src/ imports.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict  import load_pipeline, predict_single
from explain  import explain_molecule, save_shap_bar_chart

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxPathGuard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
# Aesthetic direction: dark analytical console — think "molecular diagnostics
# terminal." Monospace data, amber risk signals, teal clear signals.
# We inject CSS once, at the top, before any components render.

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0b0e14;
    --bg-secondary:  #131720;
    --bg-card:       #1a1f2e;
    --bg-card-hover: #1e2538;
    --border:        #2a3048;
    --text-primary:  #e8eaf0;
    --text-secondary:#8b93a8;
    --text-mono:     #a8b4c8;
    --accent-amber:  #f5a623;
    --accent-teal:   #4dd9ac;
    --accent-red:    #e05c5c;
    --accent-blue:   #5b8dee;
    --accent-orange: #f0874a;
    --font-body:     'IBM Plex Sans', sans-serif;
    --font-mono:     'IBM Plex Mono', monospace;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-primary); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 14px !important;
    border-radius: 6px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(91,141,238,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: var(--accent-blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Section headers ── */
h1, h2, h3 { font-family: var(--font-body) !important; font-weight: 600 !important; }
h1 { font-size: 1.6rem !important; letter-spacing: -0.02em !important; }
h3 { font-size: 1.0rem !important; color: var(--text-secondary) !important; font-weight: 400 !important; }

/* ── Custom card blocks (rendered via st.markdown) ── */
.tpg-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.tpg-card-accent-red    { border-left: 4px solid var(--accent-red); }
.tpg-card-accent-amber  { border-left: 4px solid var(--accent-amber); }
.tpg-card-accent-teal   { border-left: 4px solid var(--accent-teal); }
.tpg-card-accent-blue   { border-left: 4px solid var(--accent-blue); }

.verdict-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-family: var(--font-mono);
}
.badge-highrisk  { background: rgba(224,92,92,0.18);  color: var(--accent-red); }
.badge-caution   { background: rgba(245,166,35,0.18); color: var(--accent-amber); }
.badge-cleared   { background: rgba(77,217,172,0.18); color: var(--accent-teal); }
.badge-invalid   { background: rgba(139,147,168,0.12); color: var(--text-secondary); }

.prob-bar-wrap { margin: 0.4rem 0 0.2rem; }
.prob-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    width: 100%;
}
.prob-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.mono { font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-mono); }
.label-sm { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--text-secondary); }
.value-lg { font-size: 1.6rem; font-weight: 600; font-family: var(--font-mono); }

.alert-row { display: flex; align-items: flex-start; gap: 0.6rem; margin: 0.4rem 0; }
.alert-icon { color: var(--accent-amber); font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
.alert-name { font-weight: 600; font-size: 0.88rem; color: var(--accent-amber); }
.alert-desc { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }

.shap-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
}
.shap-name { width: 160px; flex-shrink: 0; font-family: var(--font-mono); color: var(--text-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.shap-bar-risk      { height: 10px; background: var(--accent-red);  border-radius: 3px; }
.shap-bar-protect   { height: 10px; background: var(--accent-teal); border-radius: 3px; }
.shap-val { width: 60px; flex-shrink: 0; text-align: right; font-family: var(--font-mono); color: var(--text-secondary); }

.domain-pill {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    font-weight: 600;
}
.domain-in   { background: rgba(77,217,172,0.12); color: var(--accent-teal); }
.domain-edge { background: rgba(245,166,35,0.12); color: var(--accent-amber); }
.domain-out  { background: rgba(224,92,92,0.12);  color: var(--accent-red); }

hr.tpg-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

.disclaimer-box {
    background: rgba(91,141,238,0.08);
    border: 1px solid rgba(91,141,238,0.25);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

/* ── Streamlit component cleanup ── */
[data-testid="stMetric"]         { background: var(--bg-card); border-radius: 8px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"]    { color: var(--text-secondary) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"]    { color: var(--text-primary) !important; font-family: var(--font-mono) !important; }
div[data-testid="stExpander"]    { background: var(--bg-card); border: 1px solid var(--border) !important; border-radius: 8px; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────

EXAMPLE_MOLECULES = {
    "Ethanol (safe)":          "CCO",
    "Paracetamol (borderline)":"CC(=O)Nc1ccc(O)cc1",
    "Aniline (alerted)":       "c1ccc(N)cc1",
    "Benzo[a]pyrene (high risk)": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "Aspirin":                 "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine":                "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
}

VERDICT_CONFIG = {
    "High Risk": {"badge": "badge-highrisk",  "icon": "🔴", "color": "#e05c5c", "card": "tpg-card-accent-red"},
    "Caution":   {"badge": "badge-caution",   "icon": "🟡", "color": "#f5a623", "card": "tpg-card-accent-amber"},
    "Cleared":   {"badge": "badge-cleared",   "icon": "🟢", "color": "#4dd9ac", "card": "tpg-card-accent-teal"},
    "Invalid":   {"badge": "badge-invalid",   "icon": "⚪", "color": "#8b93a8", "card": "tpg-card-accent-blue"},
}

MODELS_DIR   = os.path.join(ROOT, "models")
REPORT_PATH  = os.path.join(ROOT, "models", "training_report.json")
DOMAIN_NPZ   = os.path.join(ROOT, "data", "processed", "SR_p53.npz")
ASSETS_DIR   = os.path.join(ROOT, "assets")


# ── Resource loading (cached — runs once per session) ──────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def get_pipeline():
    """
    Loads all three XGBoost models and thresholds once.
    st.cache_resource keeps this alive across reruns —
    models are never re-read from disk during a demo.
    """
    return load_pipeline(MODELS_DIR, REPORT_PATH)


# ── Molecule rendering ─────────────────────────────────────────────────────────

def mol_to_image_bytes(smiles: str, size: tuple = (380, 280)) -> bytes | None:
    """
    Renders a 2D molecule image using RDKit and returns raw PNG bytes.
    Returns None if rendering fails — the UI will show a text fallback.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ── HTML rendering helpers ─────────────────────────────────────────────────────

def verdict_badge_html(verdict: str) -> str:
    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["Invalid"])
    return f'<span class="verdict-badge {cfg["badge"]}">{cfg["icon"]} {verdict}</span>'


def prob_bar_html(probability: float, color: str) -> str:
    pct = int(probability * 100)
    return f"""
    <div class="prob-bar-wrap">
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>"""


def risk_color(risk_label: str) -> str:
    return {
        "High Risk":   "#e05c5c",
        "Borderline":  "#f5a623",
        "Low Risk":    "#4dd9ac",
    }.get(risk_label, "#8b93a8")


def domain_pill_html(domain_status: str) -> str:
    cls = {
        "In Domain":       "domain-in",
        "Edge of Domain":  "domain-edge",
        "Out of Domain":   "domain-out",
    }.get(domain_status, "domain-edge")
    return f'<span class="domain-pill {cls}">{domain_status}</span>'


def shap_bar_html(
    features: list[dict],
    direction: str,
    max_abs: float,
    max_width: int = 120,
) -> str:
    """Renders an inline SHAP horizontal mini-bar for a feature list."""
    bar_class = "shap-bar-risk" if direction == "risk" else "shap-bar-protect"
    rows = []
    for f in features[:6]:
        name   = f["feature_name"].replace("morgan_bit_", "bit ")
        val    = abs(f["shap_value"])
        width  = int((val / (max_abs + 1e-9)) * max_width)
        shap_s = f"{f['shap_value']:+.3f}"
        tag    = "★" if f["is_descriptor"] else "·"
        rows.append(f"""
        <div class="shap-row">
            <span class="shap-name" title="{f['feature_name']}">{tag} {name}</span>
            <div class="{bar_class}" style="width:{width}px;"></div>
            <span class="shap-val">{shap_s}</span>
        </div>""")
    return "\n".join(rows)


# ── Agent card renderer ────────────────────────────────────────────────────────

def render_agent_card(agent: dict, explanation: dict | None) -> None:
    """
    Renders one agent's results as a styled card inside a Streamlit column.
    Draws probability bar, risk label, threshold, and SHAP mini-bars.
    """
    prob       = agent["probability"]
    risk_label = agent["risk_label"]
    color      = risk_color(risk_label)
    fired      = agent["raw_prediction"] == 1
    fired_icon = "⚡ FIRED" if fired else "— clear"

    # Agent name cleaned for display
    display_name = agent["agent_name"].split("— ")[-1]

    st.markdown(f"""
    <div class="tpg-card {'tpg-card-accent-red' if fired else 'tpg-card-accent-blue'}">
        <div class="label-sm">{agent['agent_name'].split(' — ')[0]}</div>
        <div style="font-size:1rem; font-weight:600; margin:0.3rem 0 0.5rem;">{display_name}</div>
        <div class="value-lg" style="color:{color};">{prob:.1%}</div>
        {prob_bar_html(prob, color)}
        <div style="display:flex; justify-content:space-between; margin-top:0.4rem;">
            <span class="mono">{verdict_badge_html(risk_label)}</span>
            <span class="mono" style="font-size:0.75rem; color:{'#e05c5c' if fired else '#8b93a8'};">{fired_icon}</span>
        </div>
        <hr class="tpg-divider"/>
        <div style="display:flex; justify-content:space-between;">
            <span class="label-sm">Safety threshold</span>
            <span class="mono">{agent['safety_threshold']:.4f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
            <span class="label-sm">Training threshold</span>
            <span class="mono">{agent['training_threshold']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP mini-bars if explanation is available
    if explanation:
        risk_feats = explanation.get("top_risk_features", [])
        prot_feats = explanation.get("top_protective_features", [])
        all_feats  = risk_feats + prot_feats
        max_abs    = max((abs(f["shap_value"]) for f in all_feats), default=1e-9)

        with st.expander("SHAP explanation", expanded=False):
            if risk_feats:
                st.markdown('<div class="label-sm" style="margin-bottom:0.4rem;">Risk drivers</div>', unsafe_allow_html=True)
                st.markdown(shap_bar_html(risk_feats, "risk", max_abs), unsafe_allow_html=True)
            if prot_feats:
                st.markdown('<div class="label-sm" style="margin:0.6rem 0 0.4rem;">Protective features</div>', unsafe_allow_html=True)
                st.markdown(shap_bar_html(prot_feats, "protect", max_abs), unsafe_allow_html=True)

            st.markdown("<hr class='tpg-divider'/>", unsafe_allow_html=True)
            st.markdown('<div class="label-sm" style="margin-bottom:0.4rem;">Descriptor breakdown  ★</div>', unsafe_allow_html=True)
            for d in explanation.get("descriptor_contribs", []):
                arrow = "▲" if d["shap_value"] > 0 else "▼"
                col_s = "#e05c5c" if d["shap_value"] > 0 else "#4dd9ac"
                st.markdown(
                    f'<div class="shap-row">'
                    f'<span class="shap-name">{arrow} {d["feature_name"]}</span>'
                    f'<span class="mono">{d["feature_value"]:.3f}</span>'
                    f'<span class="shap-val" style="color:{col_s};">{d["shap_value"]:+.4f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if explanation.get("summary_text"):
                st.markdown(
                    f'<div style="margin-top:0.8rem; font-size:0.80rem; '
                    f'color:#8b93a8; line-height:1.5;">{explanation["summary_text"]}</div>',
                    unsafe_allow_html=True,
                )


# ── Main app layout ────────────────────────────────────────────────────────────

def main():

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.3rem; font-weight:700; letter-spacing:-0.02em; '
            'margin-bottom:0.2rem;">⚗️ ToxPathGuard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.78rem; color:#8b93a8; margin-bottom:1.5rem; line-height:1.6;">'
            'Multi-pathway molecular safety audit across three independent biological stress checkpoints.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">Example molecules</div>', unsafe_allow_html=True)

        selected_example = None
        for label, smi in EXAMPLE_MOLECULES.items():
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                selected_example = smi

        st.markdown("<hr class='tpg-divider'/>", unsafe_allow_html=True)

        st.markdown('<div class="label-sm">Threshold mode</div>', unsafe_allow_html=True)
        threshold_mode = st.radio(
            label="threshold_mode_radio",
            options=["safety", "training"],
            index=0,
            format_func=lambda x: "Safety (low FNR)" if x == "safety" else "Training (F1-optimal)",
            label_visibility="collapsed",
        )

        st.markdown("<hr class='tpg-divider'/>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.70rem; color:#5a6075; line-height:1.6;">'
            '<b>Models:</b> XGBoost · Tox21 assays<br>'
            '<b>Features:</b> Morgan FP (r=2, 1024-bit) + 6 descriptors<br>'
            '<b>Explainability:</b> SHAP TreeExplainer<br>'
            '<b>Training PR-AUC:</b> 0.387 / 0.520 / 0.383'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Main header ────────────────────────────────────────────────────────────
    st.markdown(
        '<h1>ToxPathGuard <span style="font-weight:300; color:#5b8dee;">·</span> '
        'Molecular Safety Audit</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-box">'
        '⚠️  <strong>Pre-screening prototype only.</strong> '
        'This tool is not a clinical diagnostic instrument. Predictions are based on '
        'Tox21 stress-response assays and are intended for early-stage decision support '
        'by trained professionals. Do not use for regulatory or medical decisions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Input section ──────────────────────────────────────────────────────────
    input_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")

    with input_col:
        # If an example was clicked, pre-fill the text input via session state
        if selected_example and "smiles_input" not in st.session_state:
            st.session_state["smiles_input"] = selected_example
        elif selected_example:
            st.session_state["smiles_input"] = selected_example

        smiles = st.text_input(
            label="SMILES string",
            key="smiles_input",
            placeholder="Enter SMILES string, e.g.  CCO  or  c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
            label_visibility="collapsed",
        )

    with btn_col:
        run_audit = st.button("Run Audit", use_container_width=True)

    if not smiles and not run_audit:
        st.markdown(
            '<div style="text-align:center; color:#3d4460; padding:3rem 0; font-size:0.9rem;">'
            'Enter a SMILES string above or choose an example from the sidebar.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if not smiles:
        st.warning("Please enter a SMILES string.")
        return

    # ── Load pipeline (cached) ─────────────────────────────────────────────────
    try:
        models, thresholds = get_pipeline()
    except FileNotFoundError as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    # ── Run prediction ─────────────────────────────────────────────────────────
    with st.spinner("Running multi-pathway audit..."):
        result = predict_single(
            smiles         = smiles,
            models         = models,
            thresholds     = thresholds,
            threshold_mode = threshold_mode,
            domain_npz     = DOMAIN_NPZ,
            explain        = True,
        )

    # ── Invalid SMILES ─────────────────────────────────────────────────────────
    if not result["valid"]:
        st.markdown(f"""
        <div class="tpg-card tpg-card-accent-red">
            <div class="label-sm">Validation error</div>
            <div style="margin-top:0.5rem; font-size:0.9rem;">
                {result.get('error', 'Could not parse SMILES string.')}
            </div>
            <div style="margin-top:0.7rem; font-size:0.78rem; color:#8b93a8;">
                Check for typos, unsupported notation, or paste a valid SMILES string.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Unpack result ──────────────────────────────────────────────────────────
    agents         = result.get("agents", [])
    coordinator    = result.get("coordinator", {})
    alerts         = result.get("structural_alerts", [])
    domain         = result.get("domain", {})
    explanation    = result.get("explanation", {})
    agent_exps     = {
        e["assay"]: e
        for e in explanation.get("agent_explanations", [])
    }
    verdict        = result["verdict"]
    vcfg           = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["Invalid"])

    # ── Top verdict banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="tpg-card {vcfg['card']}" style="margin-top:1rem;">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.8rem;">
            <div>
                <div class="label-sm">Final audit verdict</div>
                <div style="margin-top:0.4rem; font-size:1.7rem; font-weight:700;
                            font-family:'IBM Plex Mono',monospace; color:{vcfg['color']};">
                    {vcfg['icon']} {verdict}
                </div>
                <div style="margin-top:0.3rem; font-size:0.82rem; color:#8b93a8;">
                    Primary concern: <strong>{coordinator.get('primary_concern', '—')}</strong>
                </div>
            </div>
            <div style="text-align:right;">
                <div class="label-sm">Concordance</div>
                <div style="margin-top:0.3rem; font-size:0.9rem; font-weight:600;
                            color:{vcfg['color']};">
                    {coordinator.get('concordance_label') or '—'}
                </div>
                <div style="margin-top:0.4rem;">
                    {domain_pill_html(domain.get('domain_status', '—'))}
                    <span class="mono" style="font-size:0.75rem; margin-left:0.5rem;">
                        Tanimoto: {domain.get('max_similarity', 0):.3f}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reason summary ─────────────────────────────────────────────────────────
    if coordinator.get("reason_summary"):
        with st.expander("Coordinator reasoning", expanded=True):
            st.markdown(
                f'<div style="font-size:0.88rem; line-height:1.7; color:#c8cfe0;">'
                f'{coordinator["reason_summary"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if coordinator.get("confidence_note"):
                st.markdown(
                    f'<div style="margin-top:0.6rem; font-size:0.78rem; color:#5a6480; '
                    f'font-style:italic;">{coordinator["confidence_note"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # ── Three-column agent layout ──────────────────────────────────────────────
    st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">Agent results — three biological stress pathways</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    for col, agent in zip([col1, col2, col3], agents):
        with col:
            render_agent_card(agent, agent_exps.get(agent["assay"]))

    st.markdown("<div style='margin:0.5rem 0;'></div>", unsafe_allow_html=True)

    # ── Bottom section: molecule + alerts + SHAP ───────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    # ── Left: molecule image + structural alerts ───────────────────────────────
    with left_col:
        st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">Molecule structure</div>', unsafe_allow_html=True)

        img_bytes = mol_to_image_bytes(smiles)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
        else:
            st.markdown(
                '<div class="tpg-card" style="text-align:center; color:#5a6480; padding:2rem;">'
                'Molecule rendering unavailable.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="mono" style="font-size:0.72rem; word-break:break-all; '
            f'color:#4a5270; margin-top:0.3rem;">{smiles}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin:0.8rem 0;'></div>", unsafe_allow_html=True)

        # Structural alerts
        st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">Structural alerts</div>', unsafe_allow_html=True)
        if alerts:
            for al in alerts:
                st.markdown(f"""
                <div class="tpg-card tpg-card-accent-amber" style="padding:0.8rem 1rem;">
                    <div class="alert-row">
                        <span class="alert-icon">⚠</span>
                        <div>
                            <div class="alert-name">{al['name']}</div>
                            <div class="alert-desc">{al['description']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tpg-card" style="padding:0.8rem 1rem;">
                <span style="color:#4dd9ac; font-size:0.85rem;">✓ No structural toxicophores detected.</span>
            </div>
            """, unsafe_allow_html=True)

        # Domain warning
        if domain.get("low_confidence"):
            st.markdown(f"""
            <div class="tpg-card tpg-card-accent-amber" style="margin-top:0.6rem; padding:0.8rem 1rem;">
                <div class="label-sm">Domain warning</div>
                <div style="font-size:0.80rem; color:#f5a623; margin-top:0.3rem; line-height:1.5;">
                    {domain.get('domain_warning', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right: overall SHAP summary + charts ──────────────────────────────────
    with right_col:
        st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">SHAP feature importance — across all agents</div>', unsafe_allow_html=True)

        # Overall SHAP explanation summary text
        if explanation.get("overall_summary"):
            st.markdown(
                f'<div class="tpg-card" style="font-size:0.84rem; color:#a8b4c8; line-height:1.65;">'
                f'{explanation["overall_summary"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # SHAP bar chart images (saved to assets/ during prediction)
        # We try to load them; if missing, we fall back to inline text.
        chart_shown = False
        chart_tabs  = [a["agent_name"].split("— ")[-1] for a in agents]

        if chart_tabs:
            tabs = st.tabs(chart_tabs)
            for tab, agent in zip(tabs, agents):
                with tab:
                    # Try loading pre-saved chart from assets/
                    safe_assay = agent["assay"].replace("-", "_")
                    chart_path = os.path.join(ASSETS_DIR, f"shap_{safe_assay}.png")

                    if os.path.exists(chart_path):
                        try:
                            st.image(chart_path, use_container_width=True)
                            chart_shown = True
                        except Exception:
                            chart_shown = False

                    if not chart_shown:
                        # Fallback: generate and show chart inline
                        exp_block = agent_exps.get(agent["assay"])
                        if exp_block:
                            saved = save_shap_bar_chart(exp_block, out_dir=ASSETS_DIR)
                            if saved and os.path.exists(saved):
                                try:
                                    st.image(saved, use_container_width=True)
                                    chart_shown = True
                                except Exception:
                                    pass

                    if not chart_shown:
                        # Text-only fallback — always safe
                        exp_block = agent_exps.get(agent["assay"])
                        if exp_block:
                            rf = exp_block.get("top_risk_features", [])
                            pf = exp_block.get("top_protective_features", [])
                            all_f = sorted(rf + pf, key=lambda x: abs(x["shap_value"]), reverse=True)
                            max_s = max((abs(f["shap_value"]) for f in all_f), default=1e-9)
                            st.markdown(
                                shap_bar_html(rf[:5],  "risk",    max_s),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                shap_bar_html(pf[:5], "protect", max_s),
                                unsafe_allow_html=True,
                            )

        # Descriptor contribution table across all agents
        st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="label-sm" style="margin-bottom:0.6rem;">Physicochemical descriptors — SHAP contributions</div>', unsafe_allow_html=True)

        # Build a compact descriptor table showing value + per-agent SHAP
        desc_names = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RingCount"]
        if agent_exps:
            first_exp   = next(iter(agent_exps.values()))
            desc_values = {d["feature_name"]: d["feature_value"] for d in first_exp.get("descriptor_contribs", [])}

            header_html = (
                '<div style="display:grid; grid-template-columns: 120px 80px repeat(3, 70px); '
                'gap:0 0.5rem; font-size:0.70rem; font-family:var(--font-mono); '
                'color:#5a6480; padding:0 0.2rem; margin-bottom:0.3rem;">'
                '<span>Property</span><span>Value</span>'
                '<span>Agent 1</span><span>Agent 2</span><span>Agent 3</span>'
                '</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)

            for desc in desc_names:
                val_str = f"{desc_values.get(desc, 0.0):.3g}"
                cells   = [f'<span>{val_str}</span>']
                for assay in ["SR-p53", "SR-ARE", "SR-HSE"]:
                    exp_b = agent_exps.get(assay, {})
                    shaps = {d["feature_name"]: d["shap_value"] for d in exp_b.get("descriptor_contribs", [])}
                    sv    = shaps.get(desc, 0.0)
                    col_s = "#e05c5c" if sv > 0.02 else ("#4dd9ac" if sv < -0.02 else "#5a6480")
                    cells.append(f'<span style="color:{col_s}; text-align:right;">{sv:+.3f}</span>')

                row_html = (
                    '<div style="display:grid; grid-template-columns: 120px 80px repeat(3, 70px); '
                    'gap:0 0.5rem; font-size:0.78rem; font-family:var(--font-mono); '
                    'padding:0.2rem 0.2rem; border-bottom:1px solid #1e2538;">'
                    f'<span style="color:#a8b4c8;">{desc}</span>'
                    + "".join(cells) +
                    '</div>'
                )
                st.markdown(row_html, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin:2rem 0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; font-size:0.70rem; color:#2e3450;">'
        'ToxPathGuard · XGBoost + SHAP + RDKit · Tox21 dataset · '
        'Fully offline · Hackathon prototype — not a clinical tool'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    main()
