"""
Phase 1 — ToxPathGuard Data Audit
----------------------------------
Validates the Tox21 CSV dataset and produces a go/no-go recommendation
for the three chosen assay columns before any feature engineering begins.

Usage:
    python src/audit.py --data data/tox21.csv

Or set DATA_PATH at the top of the script if you prefer not to use CLI args.
"""

import sys
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit's noisy stderr warnings during SMILES validation
RDLogger.DisableLog("rdApp.*")

# ── Configuration ──────────────────────────────────────────────────────────────

# The three assay columns we are evaluating as biological stress checkpoints.
# Change these here if the audit reveals a better alternative.
TARGET_ASSAYS = {
    "SR-p53": "DNA / Genotoxic Stress",
    "SR-ARE": "Oxidative Stress (Nrf2/ARE pathway)",
    "SR-HSE": "Cellular / Heat-Shock Stress Response",
}

# Thresholds that determine a go/no-go recommendation per assay.
# If positive rate falls below MIN_POSITIVE_RATE, we flag it as too sparse.
# If missing rate exceeds MAX_MISSING_RATE, we flag it as too incomplete.
MIN_POSITIVE_RATE = 0.04   # 4 % minimum positives
MAX_MISSING_RATE  = 0.60   # 60 % maximum allowed NaNs


# ── SMILES Validation ──────────────────────────────────────────────────────────

def validate_smiles(smiles_series: pd.Series) -> pd.Series:
    """
    Returns a boolean Series — True where RDKit can parse the SMILES,
    False where it cannot (invalid, missing, or empty).
    """
    def is_valid(smi):
        if not isinstance(smi, str) or not smi.strip():
            return False
        mol = Chem.MolFromSmiles(smi)
        return mol is not None

    return smiles_series.apply(is_valid)


# ── Per-Assay Statistics ───────────────────────────────────────────────────────

def compute_assay_stats(df: pd.DataFrame, valid_mask: pd.Series) -> list[dict]:
    """
    For each target assay, computes:
      - total rows in the full dataset,
      - rows that have a non-NaN label AND a valid SMILES (usable rows),
      - count of positives (label == 1) among usable rows,
      - positive rate,
      - missing rate across all rows,
      - a go/no-go verdict.
    """
    stats = []
    total_rows = len(df)

    for col, pathway_name in TARGET_ASSAYS.items():

        if col not in df.columns:
            # The column doesn't even exist in this CSV — hard fail.
            stats.append({
                "assay":         col,
                "pathway":       pathway_name,
                "status":        "COLUMN MISSING",
                "total_rows":    total_rows,
                "usable_rows":   0,
                "positives":     0,
                "positive_rate": None,
                "missing_rate":  1.0,
                "verdict":       "NO-GO — column not found in dataset",
            })
            continue

        # Usable rows: valid SMILES AND a real label (not NaN)
        has_label = df[col].notna()
        usable_mask = valid_mask & has_label
        usable_df = df.loc[usable_mask, col]

        n_usable   = int(usable_mask.sum())
        n_positive = int((usable_df == 1).sum())
        n_missing  = int(df[col].isna().sum())

        pos_rate     = n_positive / n_usable if n_usable > 0 else 0.0
        missing_rate = n_missing / total_rows

        # Determine go/no-go
        if n_usable == 0:
            verdict = "NO-GO — zero usable rows"
        elif pos_rate < MIN_POSITIVE_RATE:
            verdict = f"CAUTION — positive rate {pos_rate:.1%} is very low"
        elif missing_rate > MAX_MISSING_RATE:
            verdict = f"CAUTION — {missing_rate:.1%} of rows have missing labels"
        else:
            verdict = "GO"

        stats.append({
            "assay":         col,
            "pathway":       pathway_name,
            "total_rows":    total_rows,
            "usable_rows":   n_usable,
            "positives":     n_positive,
            "positive_rate": pos_rate,
            "missing_rate":  missing_rate,
            "verdict":       verdict,
        })

    return stats


# ── Printing Helpers ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_assay_table(stats: list[dict]) -> None:
    """Prints a readable summary table for the three assays."""
    header = f"{'Assay':<12} {'Usable':>8} {'Positives':>10} {'Pos Rate':>10} {'Missing':>10}  Verdict"
    print(header)
    print("─" * len(header))
    for s in stats:
        pos_str     = f"{s['positive_rate']:.1%}" if s['positive_rate'] is not None else "N/A"
        missing_str = f"{s['missing_rate']:.1%}"
        print(
            f"{s['assay']:<12} "
            f"{s['usable_rows']:>8,} "
            f"{s['positives']:>10,} "
            f"{pos_str:>10} "
            f"{missing_str:>10}  "
            f"{s['verdict']}"
        )


# ── Main Audit Routine ─────────────────────────────────────────────────────────

def run_audit(csv_path: str) -> None:

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print_section("Loading dataset")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{csv_path}'.")
        print("Place your Tox21 CSV at data/tox21.csv or pass --data <path>.")
        sys.exit(1)

    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"File loaded    : {csv_path}")

    # ── 2. Column overview ─────────────────────────────────────────────────────
    print_section("Column names")
    for col in df.columns:
        print(f"  {col}")

    # ── 3. Identify the SMILES column ──────────────────────────────────────────
    # Tox21 CSVs use 'smiles' (lowercase). We check common variants.
    smiles_col = None
    for candidate in ["smiles", "SMILES", "Smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break

    if smiles_col is None:
        print("\nERROR: No SMILES column found. Cannot continue.")
        sys.exit(1)

    print_section(f"SMILES validation  (column: '{smiles_col}')")
    valid_mask   = validate_smiles(df[smiles_col])
    n_valid      = int(valid_mask.sum())
    n_invalid    = len(df) - n_valid
    invalid_rate = n_invalid / len(df)

    print(f"  Valid SMILES   : {n_valid:,}  ({1 - invalid_rate:.1%})")
    print(f"  Invalid SMILES : {n_invalid:,}  ({invalid_rate:.1%})")

    if invalid_rate > 0.05:
        print("  WARNING: more than 5 % of SMILES are invalid — check your CSV source.")

    # ── 4. Global missing values ───────────────────────────────────────────────
    print_section("Missing values across all columns (top 15 by count)")
    missing_counts = df.isnull().sum().sort_values(ascending=False).head(15)
    for col_name, count in missing_counts.items():
        pct = count / len(df)
        print(f"  {col_name:<25} {count:>6,}  ({pct:.1%})")

    # ── 5. Per-assay audit ─────────────────────────────────────────────────────
    print_section("Assay-level audit for the three chosen checkpoints")
    stats = compute_assay_stats(df, valid_mask)
    print_assay_table(stats)

    # ── 6. Pathway details (verbose) ───────────────────────────────────────────
    print_section("Detailed breakdown per assay")
    for s in stats:
        print(f"\n  [{s['assay']}]  →  {s['pathway']}")
        print(f"    Usable rows    : {s['usable_rows']:,}")
        print(f"    Positives      : {s['positives']:,}")
        print(f"    Positive rate  : {s['positive_rate']:.2%}" if s['positive_rate'] is not None else "    Positive rate  : N/A")
        print(f"    Missing labels : {s['missing_rate']:.2%}")
        print(f"    Verdict        : {s['verdict']}")

    # ── 7. Final recommendation ────────────────────────────────────────────────
    print_section("Overall go / no-go recommendation")
    all_go    = all(s["verdict"] == "GO" for s in stats)
    any_no_go = any(s["verdict"].startswith("NO-GO") for s in stats)

    if all_go:
        print("  ALL THREE ASSAYS PASS — proceed to Phase 2 (feature engineering).")
    elif any_no_go:
        print("  AT LEAST ONE ASSAY FAILED — revise assay mapping before Phase 2.")
        print("  Suggested replacements to consider:")
        print("    DNA stress    : NR-p53  (nuclear receptor pathway, also genotoxic signal)")
        print("    Oxidative     : SR-ATAD5 (DNA damage / replication stress, Nrf2-adjacent)")
        print("    Cellular      : SR-MMP  (mitochondrial membrane potential, broader stress)")
    else:
        print("  MIXED RESULTS — some assays have low positive rates (see CAUTION flags).")
        print("  You can still proceed, but set scale_pos_weight carefully in Phase 3.")
        print("  Assays with < 6 % positives will require aggressive imbalance handling.")

    print()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 1 — Tox21 dataset audit"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/tox21.csv",
        help="Path to the Tox21 CSV file (default: data/tox21.csv)",
    )
    args = parser.parse_args()
    run_audit(args.data)
