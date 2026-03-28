"""
Phase 4.1 — Applicability Domain Check
-----------------------------------------
Provides a lightweight, fast estimate of whether a query molecule falls
within the chemical space covered by the training data.

Method: Tanimoto similarity against a stratified sample of training
Morgan fingerprints. We use the first 1024 columns of the processed .npz
files (which are the Morgan fingerprint bits from Phase 2) rather than
re-reading the original CSV. This keeps the check fully offline and fast.

The Tanimoto coefficient for binary fingerprints is:
    Tc = |A ∩ B| / |A ∪ B|

We take the maximum Tanimoto across the reference sample (max-sim).
Max-sim is the standard applicability-domain metric in QSAR literature:
it tells you how similar the query is to its nearest training neighbour.

Confidence zones:
    max_sim >= 0.40  →  Within domain
    max_sim >= 0.25  →  Borderline / edge of domain
    max_sim <  0.25  →  Outside domain (extrapolation risk)
"""

import os
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────

# Number of reference fingerprints to sample from training data.
# 300 is enough to catch most in-domain molecules quickly.
# More does not improve coverage meaningfully for Tox21 scale data.
REFERENCE_SAMPLE_SIZE = 300

MORGAN_NBITS = 1024   # must match features.py

# Similarity thresholds for the three domain zones.
DOMAIN_IN        = 0.40
DOMAIN_EDGE      = 0.25

# Which processed .npz file to sample from for the reference set.
# We use SR_p53 because it has the most usable rows and is a fair
# representation of the full Tox21 chemical space.
_DEFAULT_NPZ = os.path.join("data", "processed", "SR_p53.npz")


# ── Reference Set Building ─────────────────────────────────────────────────────

_reference_fps: np.ndarray | None = None   # module-level cache


def _load_reference_fps(npz_path: str = _DEFAULT_NPZ) -> np.ndarray:
    """
    Loads a random sample of Morgan fingerprint bit-vectors from the
    training .npz. The first MORGAN_NBITS columns of X are the fingerprint.
    Caches the result so it is only loaded once per process.
    """
    global _reference_fps

    if _reference_fps is not None:
        return _reference_fps

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Reference fingerprint file not found: '{npz_path}'.\n"
            "Run features.py (Phase 2) to generate processed datasets."
        )

    data = np.load(npz_path)
    X    = data["X"]  # shape: (n_samples, 1030)

    # Extract the fingerprint portion only (first 1024 columns).
    fps = X[:, :MORGAN_NBITS].astype(np.uint8)

    # Stratified random sample — reproducible seed so the domain check
    # gives consistent results across prediction calls.
    rng = np.random.default_rng(seed=42)
    n   = min(REFERENCE_SAMPLE_SIZE, len(fps))
    idx = rng.choice(len(fps), size=n, replace=False)

    _reference_fps = fps[idx]
    return _reference_fps


# ── Tanimoto Computation ───────────────────────────────────────────────────────

def _tanimoto_max(query_fp: np.ndarray, reference_fps: np.ndarray) -> float:
    """
    Computes the maximum Tanimoto similarity between a query fingerprint
    and a matrix of reference fingerprints using vectorised NumPy operations.

    For binary bit vectors:
        intersection = dot(query, ref_i)
        union        = |query| + |ref_i| - intersection
        Tc           = intersection / union

    This runs in ~1ms for 300 reference fingerprints on a modern CPU.
    """
    query = query_fp.astype(np.float32)                   # shape: (1024,)
    ref   = reference_fps.astype(np.float32)              # shape: (n_ref, 1024)

    # Intersection: number of bits set in both query and each reference.
    intersections = ref.dot(query)                        # shape: (n_ref,)

    # Union: |A| + |B| - |A ∩ B|
    query_count = query.sum()
    ref_counts  = ref.sum(axis=1)                         # shape: (n_ref,)
    unions      = query_count + ref_counts - intersections

    # Avoid division by zero for all-zero fingerprints (degenerate case).
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.where(unions > 0, intersections / unions, 0.0)

    return float(similarities.max())


# ── Public API ─────────────────────────────────────────────────────────────────

def check_domain(
    mol_fp_bits: np.ndarray,
    npz_path: str = _DEFAULT_NPZ,
) -> dict:
    """
    Runs the applicability-domain check for one molecule.

    Args:
        mol_fp_bits: 1-D uint8 or float32 array of Morgan fingerprint bits,
                     length MORGAN_NBITS (1024). This is the first 1024
                     elements of the featurise_molecule() output vector.
        npz_path:    Path to a processed .npz file for reference sampling.

    Returns:
        {
            "max_similarity":  float,   # Tanimoto to nearest training neighbour
            "domain_status":   str,     # "In Domain" | "Edge of Domain" | "Out of Domain"
            "domain_warning":  str,     # Human-readable message for the UI
            "low_confidence":  bool,    # True if outside or at edge of domain
        }
    """
    reference_fps = _load_reference_fps(npz_path)
    max_sim       = _tanimoto_max(mol_fp_bits[:MORGAN_NBITS], reference_fps)
    max_sim       = round(max_sim, 4)

    if max_sim >= DOMAIN_IN:
        status   = "In Domain"
        warning  = f"Molecule is within the training domain (max Tanimoto: {max_sim:.2f})."
        low_conf = False
    elif max_sim >= DOMAIN_EDGE:
        status   = "Edge of Domain"
        warning  = (
            f"Molecule is at the edge of the training domain "
            f"(max Tanimoto: {max_sim:.2f}). Predictions may be less reliable."
        )
        low_conf = True
    else:
        status   = "Out of Domain"
        warning  = (
            f"Low confidence: molecule is outside the training domain "
            f"(max Tanimoto: {max_sim:.2f}). Treat predictions as indicative only."
        )
        low_conf = True

    return {
        "max_similarity": max_sim,
        "domain_status":  status,
        "domain_warning": warning,
        "low_confidence": low_conf,
    }
