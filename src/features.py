"""
Phase 2 — ToxPathGuard Feature Engineering
--------------------------------------------
Converts the Tox21 CSV into model-ready feature matrices for three
biological stress pathway assays:

    SR-p53  →  DNA / Genotoxic Stress
    SR-ARE  →  Oxidative Stress
    SR-HSE  →  Cellular / Heat-Shock Stress Response

Output: one .npz file per assay saved to data/processed/
Each file contains:
    X  — float32 feature matrix  (n_samples × n_features)
    y  — int8 label vector       (n_samples,)

Usage:
    python src/features.py --data data/tox21.csv --out data/processed
"""

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem

# Silence RDKit's stderr — we handle errors explicitly
RDLogger.DisableLog("rdApp.*")


# ── Configuration ──────────────────────────────────────────────────────────────

# The three assay targets we are building classifiers for.
TARGET_ASSAYS = ["SR-p53", "SR-ARE", "SR-HSE"]

# Morgan fingerprint settings.
# 1024 bits at radius 2 is the right choice here — explained at the bottom.
MORGAN_RADIUS = 2
MORGAN_NBITS  = 1024


# ── Molecule Parsing ───────────────────────────────────────────────────────────

def smiles_to_mol(smiles: str):
    """
    Converts a SMILES string to an RDKit Mol object.
    Returns None if the string is missing, empty, or chemically invalid.
    This is the single point of failure handling — every downstream
    function can safely assume it receives a valid Mol or None.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


# ── Feature Extraction ─────────────────────────────────────────────────────────

def morgan_fingerprint(mol) -> np.ndarray:
    """
    Computes a folded Morgan (circular) fingerprint as a bit vector.
    Radius 2 captures roughly the same neighbourhood as ECFP4 —
    two bonds out from each atom — which is the de facto standard
    for small-molecule similarity and activity modelling.
    Returns a 1-D float32 array of length MORGAN_NBITS.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=MORGAN_RADIUS,
        nBits=MORGAN_NBITS,
        useChirality=False,   # chirality adds noise for this task at this scale
    )
    return np.array(fp, dtype=np.float32)


def physicochemical_descriptors(mol) -> np.ndarray:
    """
    Computes six interpretable physicochemical properties:

        MolWt  — molecular weight; controls absorption and metabolism
        LogP   — octanol-water partition; proxy for membrane permeability
        TPSA   — topological polar surface area; proxy for membrane crossing
        HBD    — hydrogen bond donors; affects solubility and permeability
        HBA    — hydrogen bond acceptors; same
        RingCount — number of rings; rough proxy for structural complexity

    These six features give the model chemical context that fingerprint
    bits alone cannot easily encode. Each one is a real number, not a
    binary flag, so the model can learn thresholds (e.g. LogP > 5 is
    a Lipinski red flag).

    Returns a 1-D float32 array of length 6.
    """
    return np.array([
        Descriptors.MolWt(mol),                          # molecular weight
        Descriptors.MolLogP(mol),                        # calculated LogP
        rdMolDescriptors.CalcTPSA(mol),                  # TPSA
        rdMolDescriptors.CalcNumHBD(mol),                # H-bond donors
        rdMolDescriptors.CalcNumHBA(mol),                # H-bond acceptors
        rdMolDescriptors.CalcNumRings(mol),              # ring count
    ], dtype=np.float32)


def featurise_molecule(mol) -> np.ndarray:
    """
    Combines the Morgan fingerprint and the physicochemical descriptors
    into a single flat feature vector.
    Total length: MORGAN_NBITS + 6  (default: 1030 features)
    Returns None if mol is None — the caller filters these out.
    """
    if mol is None:
        return None
    fp   = morgan_fingerprint(mol)
    desc = physicochemical_descriptors(mol)
    return np.concatenate([fp, desc])


def feature_names() -> list[str]:
    """
    Returns a human-readable list of feature names aligned with the
    output of featurise_molecule(). Used in SHAP plots in Phase 5.
    """
    fp_names   = [f"morgan_bit_{i}" for i in range(MORGAN_NBITS)]
    desc_names = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RingCount"]
    return fp_names + desc_names


# ── Dataset Assembly ───────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Loads the Tox21 CSV, converts every SMILES to an RDKit Mol,
    and drops rows where the SMILES is invalid.
    Adds a 'mol' column for downstream use.
    """
    df = pd.read_csv(csv_path)
    df["mol"] = df["smiles"].apply(smiles_to_mol)

    n_before = len(df)
    df = df[df["mol"].notna()].reset_index(drop=True)
    n_after  = len(df)

    print(f"  Loaded {n_before:,} rows — kept {n_after:,} after SMILES validation "
          f"(dropped {n_before - n_after})")
    return df


def build_assay_dataset(
    df: pd.DataFrame,
    assay_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Builds (X, y) for a single assay column.

    The steps are:
      1. Filter to rows where the assay label is not NaN.
         (Each assay has different missingness, so each produces a
         different number of usable rows — this is expected.)
      2. Extract features for every molecule in that filtered set.
      3. Convert labels to integer (0 or 1).

    Returns:
        X     — float32 feature matrix (n_samples × n_features)
        y     — int8 label vector (n_samples,)
        names — list of feature names (same for every assay)
    """
    # Step 1: keep only rows with a real label for this assay
    mask   = df[assay_col].notna()
    subset = df[mask].reset_index(drop=True)

    # Step 2: featurise every molecule in the subset
    feature_vectors = [featurise_molecule(mol) for mol in subset["mol"]]

    # Guard: if any featurisation returned None (should not happen
    # after SMILES cleaning, but we are defensive here)
    valid = [i for i, v in enumerate(feature_vectors) if v is not None]
    if len(valid) < len(subset):
        dropped = len(subset) - len(valid)
        print(f"    WARNING [{assay_col}]: dropped {dropped} rows where "
              f"featurisation failed despite valid SMILES.")
        feature_vectors = [feature_vectors[i] for i in valid]
        subset = subset.iloc[valid].reset_index(drop=True)

    X = np.stack(feature_vectors, axis=0).astype(np.float32)
    y = subset[assay_col].astype(int).values.astype(np.int8)

    return X, y, feature_names()


# ── Sanity Check ───────────────────────────────────────────────────────────────

def run_sanity_check(
    assay: str,
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
) -> None:
    """
    Prints a compact summary for one assay dataset.
    Checks that:
      - X and y have the same number of rows (alignment),
      - feature count matches expected length,
      - label values are strictly 0 or 1 (no leakage of NaN-converted floats),
      - there are no NaN values in X.
    """
    n_samples, n_features = X.shape
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    expected_features = MORGAN_NBITS + 6

    print(f"\n  [{assay}]")
    print(f"    Samples         : {n_samples:,}")
    print(f"    Features        : {n_features}  (expected {expected_features})")
    print(f"    Positives       : {n_pos:,}  ({n_pos/n_samples:.1%})")
    print(f"    Negatives       : {n_neg:,}  ({n_neg/n_samples:.1%})")
    print(f"    scale_pos_weight: {n_neg/n_pos:.1f}  ← use this in Phase 3")
    print(f"    Label/row match : {'OK' if len(y) == n_samples else 'MISMATCH — STOP'}")
    print(f"    NaN in X        : {'NONE — clean' if not np.isnan(X).any() else 'FOUND — investigate'}")
    print(f"    Label values    : {set(y.tolist())}  (expected {{0, 1}})")
    print(f"    Feature preview : {names[:4]} ... {names[-3:]}")


# ── Save and Load ──────────────────────────────────────────────────────────────

def save_dataset(out_dir: str, assay: str, X: np.ndarray, y: np.ndarray) -> str:
    """
    Saves (X, y) as a compressed NumPy archive (.npz).
    The filename encodes the assay name so Phase 3 can load by name.
    Returns the path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Replace the hyphen in assay names (e.g. SR-p53 → SR_p53)
    # so the filename is safe across operating systems
    safe_name = assay.replace("-", "_")
    path = os.path.join(out_dir, f"{safe_name}.npz")
    np.savez_compressed(path, X=X, y=y)
    return path


def load_dataset(out_dir: str, assay: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a saved assay dataset from disk.
    This is the function Phase 3 (train.py) will call — defining it here
    keeps the loading contract in one place.
    """
    safe_name = assay.replace("-", "_")
    path = os.path.join(out_dir, f"{safe_name}.npz")
    data = np.load(path)
    return data["X"], data["y"]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_feature_engineering(csv_path: str, out_dir: str) -> None:

    print("\n" + "─" * 72)
    print("  ToxPathGuard — Phase 2: Feature Engineering")
    print("─" * 72)

    # ── 1. Load and clean the dataset ─────────────────────────────────────────
    print("\n[1/3] Loading and cleaning SMILES...")
    df = load_and_clean(csv_path)

    # ── 2. Build per-assay feature matrices ───────────────────────────────────
    print("\n[2/3] Building feature matrices for each assay...")
    for assay in TARGET_ASSAYS:
        print(f"\n  Processing {assay}...")
        X, y, names = build_assay_dataset(df, assay)
        path = save_dataset(out_dir, assay, X, y)
        print(f"  Saved → {path}  (X: {X.shape}, y: {y.shape})")

    # ── 3. Sanity check ────────────────────────────────────────────────────────
    print("\n[3/3] Sanity check — reloading from disk and verifying...")
    for assay in TARGET_ASSAYS:
        X, y = load_dataset(out_dir, assay)
        run_sanity_check(assay, X, y, feature_names())

    print("\n" + "─" * 72)
    print("  Phase 2 complete. Proceed to Phase 3 (model training).")
    print("─" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 2 — Feature Engineering"
    )
    parser.add_argument(
        "--data", type=str, default="data/tox21.csv",
        help="Path to the Tox21 CSV (default: data/tox21.csv)"
    )
    parser.add_argument(
        "--out", type=str, default="data/processed",
        help="Output directory for processed .npz files (default: data/processed)"
    )
    args = parser.parse_args()
    run_feature_engineering(args.data, args.out)
