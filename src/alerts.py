"""
Phase 4.1 — Structural Alert Detection
----------------------------------------
Lightweight SMARTS-based toxicophore screening using RDKit.

This is not a comprehensive toxicophore library. It is a high-signal,
low-noise set of patterns that cover the most well-established structural
alerts in early-stage safety screening. The goal is to support the model
predictions with interpretable chemical evidence, not to replace them.

Each alert is a named (pattern, description) pair. When a molecule
matches, the alert name and description are surfaced in the audit output.
"""

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ── Alert Registry ─────────────────────────────────────────────────────────────
#
# Format: (name, SMARTS, short_description)
#
# Curation principle:
#   - Every pattern here appears in published Ames/genotoxicity screening
#     literature or standard regulatory alert databases (e.g. Derek, DEREK,
#     Lhasa, ICH M7 guidance).
#   - We keep only patterns with high specificity. False alerts in a demo
#     destroy credibility faster than missed ones.
#   - Patterns are listed from most-severe to least-severe signal.

STRUCTURAL_ALERTS = [
    (
        "Polycyclic Aromatic Hydrocarbon (PAH)",
        "c1ccc2ccccc2c1",                      # naphthalene core and larger PAHs
        "Multi-ring aromatic scaffold. PAHs are well-established "
        "genotoxic carcinogens via reactive metabolite formation.",
    ),
    (
        "Aromatic Amine",
        "Nc1ccccc1",
        "Primary amine attached to aromatic ring. Aromatic amines are "
        "a major class of genotoxic carcinogens; associated with bladder cancer.",
    ),
    (
        "Nitro Group (aromatic)",
        "[N+](=O)[O-]",
        "Nitro group. Aromatic nitro compounds can be reduced to reactive "
        "hydroxylamines and nitroso intermediates, which are genotoxic.",
    ),
    (
        "Quinone / Para-Quinone",
        "O=C1C=CC(=O)C=C1",
        "Quinone scaffold. Quinones are Michael acceptors and redox-cycling "
        "agents — major drivers of oxidative stress (SR-ARE pathway).",
    ),
    (
        "Michael Acceptor (alpha-beta unsaturated carbonyl)",
        "[CH]=[CH]C(=O)",
        "Alpha,beta-unsaturated carbonyl. Reacts covalently with cellular "
        "nucleophiles (GSH, DNA, proteins). Electrophilic stress activator.",
    ),
    (
        "Epoxide",
        "C1OC1",
        "Epoxide ring. Highly reactive electrophile. Epoxides are often "
        "formed as reactive metabolites of PAHs and are directly genotoxic.",
    ),
    (
        "Alkyl Halide",
        "[CH2][F,Cl,Br,I]",
        "Alkyl halide. Reactive alkylating agents that can directly "
        "modify DNA bases, causing mutations.",
    ),
    (
        "Azo Compound",
        "[N]=[N]",
        "Azo linkage. Can be reduced to aromatic amines in vivo, "
        "generating genotoxic metabolites.",
    ),
    (
        "Hydrazine",
        "[NH]-[NH2]",
        "Hydrazine or hydrazide. Associated with hepatotoxicity and "
        "genotoxicity via reactive intermediate formation.",
    ),
    (
        "N-Nitroso",
        "[N]-N=O",
        "N-nitroso group. Highly potent alkylating agents. "
        "N-nitrosamines are among the most well-characterised carcinogens.",
    ),
    (
        "Acyl Halide",
        "C(=O)[F,Cl,Br,I]",
        "Acyl halide. Extremely reactive electrophile. Rarely a drug "
        "but flags reactive intermediates or synthetic impurities.",
    ),
]

# Pre-compile all SMARTS patterns once at import time.
# This avoids repeated compilation on every prediction call.
_COMPILED_ALERTS = []
for name, smarts, desc in STRUCTURAL_ALERTS:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        # This should never happen with the patterns above, but we guard it
        # so a bad SMARTS string does not silently drop an alert.
        import warnings
        warnings.warn(f"[alerts.py] Failed to compile SMARTS for '{name}': {smarts}")
    else:
        _COMPILED_ALERTS.append((name, pattern, desc))


# ── Ring Count Safety Check ────────────────────────────────────────────────────

_MIN_PAH_RINGS = 3   # Three or more fused rings → flag as potential PAH risk

def _has_pah_ring_system(mol) -> bool:
    """
    Secondary check for large fused ring systems that may not match the
    naphthalene SMARTS exactly (e.g. coronene, benzo[a]pyrene extensions).
    Uses ring count as a fast heuristic.
    """
    ring_info = mol.GetRingInfo()
    return ring_info.NumRings() >= _MIN_PAH_RINGS


# ── Public API ─────────────────────────────────────────────────────────────────

def check_structural_alerts(mol) -> list[dict]:
    """
    Runs the molecule against the full structural alert registry.

    Returns a list of matched alerts. Each matched alert is a dict with:
        name        — short name of the toxicophore
        description — one-sentence mechanistic explanation
        smarts      — the SMARTS string that matched

    Returns an empty list if no alerts match.
    """
    if mol is None:
        return []

    matches = []
    seen_names = set()   # prevent duplicate entries from multiple matches of the same pattern

    for name, pattern, desc in _COMPILED_ALERTS:
        if name in seen_names:
            continue
        if mol.HasSubstructMatch(pattern):
            matches.append({
                "name":        name,
                "description": desc,
            })
            seen_names.add(name)

    # Supplement the PAH SMARTS check with the ring-count heuristic.
    # This catches larger fused systems the SMARTS may undercount.
    pah_name = "Polycyclic Aromatic Hydrocarbon (PAH)"
    if _has_pah_ring_system(mol) and pah_name not in seen_names:
        matches.append({
            "name":        pah_name + " [ring-count heuristic]",
            "description": "Three or more rings detected. Large fused ring systems "
                           "are a strong structural indicator of PAH-like genotoxic risk.",
        })

    return matches


def has_any_alert(alert_matches: list[dict]) -> bool:
    """Simple predicate — True if at least one structural alert was matched."""
    return len(alert_matches) > 0
