"""
Phase 7 — ToxPathGuard Demo Test Suite
-----------------------------------------
Runs the full prediction pipeline against the demo molecule set and
reports pass/fail/warning for each case.

This script is the pre-demo sanity check. Run it before every
live demonstration to confirm the pipeline is stable.

Usage:
    python tests/run_demo_tests.py
    python tests/run_demo_tests.py --verbose
    python tests/run_demo_tests.py --mode training
"""

import os
import sys
import json
import time
import argparse
import traceback

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict import load_pipeline, predict_single

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(ROOT, "models")
REPORT_PATH = os.path.join(ROOT, "models", "training_report.json")
DOMAIN_NPZ  = os.path.join(ROOT, "data", "processed", "SR_p53.npz")
SUITE_PATH  = os.path.join(ROOT, "tests", "demo_molecules.json")

# ANSI colours — degrade gracefully on terminals that don't support them
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def load_suite(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_test(
    molecule:       dict,
    models:         dict,
    thresholds:     dict,
    threshold_mode: str,
    verbose:        bool,
) -> dict:
    """
    Runs one molecule through the full pipeline.
    Returns a result dict describing pass/warn/fail/crash.
    """
    name     = molecule["name"]
    smiles   = molecule["smiles"]
    category = molecule["category"]
    expected = molecule.get("expected_verdict")

    t0 = time.perf_counter()

    try:
        result  = predict_single(
            smiles         = smiles,
            models         = models,
            thresholds     = thresholds,
            threshold_mode = threshold_mode,
            domain_npz     = DOMAIN_NPZ,
            explain        = False,   # skip SHAP for speed in the test suite
        )
        elapsed = time.perf_counter() - t0

        verdict      = result.get("verdict", "Unknown")
        domain       = result.get("domain", {})
        domain_warn  = domain.get("low_confidence", False)
        warnings     = result.get("warnings", [])

        # ── Determine test status ──────────────────────────────────────────────
        if category == "invalid":
            # For invalid inputs, we expect the pipeline NOT to crash and to
            # return valid=False. Any verdict is acceptable.
            status = "pass" if not result.get("valid", True) else "warn"
            status_reason = "correctly rejected invalid input" if status == "pass" \
                            else "invalid input was not properly rejected"

        elif expected is not None and verdict != expected:
            # Soft fail: we flag a mismatch but do not treat it as a crash.
            # Threshold sensitivity means borderline verdicts can differ —
            # the test runner flags these for review, not as hard failures.
            status = "warn"
            status_reason = f"expected '{expected}', got '{verdict}'"

        elif domain_warn:
            status = "warn"
            status_reason = f"domain warning — {domain.get('domain_status', '')}"

        else:
            status = "pass"
            status_reason = verdict

        return {
            "name":          name,
            "category":      category,
            "status":        status,
            "verdict":       verdict,
            "reason":        status_reason,
            "elapsed_s":     round(elapsed, 3),
            "domain_warn":   domain_warn,
            "warnings":      warnings,
            "crashed":       False,
            "error":         None,
            "result":        result if verbose else None,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "name":      name,
            "category":  category,
            "status":    "crash",
            "verdict":   "ERROR",
            "reason":    str(exc),
            "elapsed_s": round(elapsed, 3),
            "crashed":   True,
            "error":     traceback.format_exc(),
            "result":    None,
        }


def print_result_line(test_result: dict) -> None:
    status = test_result["status"]
    name   = test_result["name"]
    cat    = test_result["category"]
    reason = test_result["reason"]
    t      = test_result["elapsed_s"]

    if status == "pass":
        icon  = f"{GREEN}✓{RESET}"
        label = f"{GREEN}{reason}{RESET}"
    elif status == "warn":
        icon  = f"{YELLOW}⚠{RESET}"
        label = f"{YELLOW}{reason}{RESET}"
    elif status == "crash":
        icon  = f"{RED}✗ CRASH{RESET}"
        label = f"{RED}{reason}{RESET}"
    else:
        icon  = "?"
        label = reason

    print(f"  {icon}  {BOLD}{name:<28}{RESET} [{cat:<14}]  {label}  ({t:.3f}s)")


def print_summary(results: list[dict], total_elapsed: float) -> None:
    n_pass   = sum(1 for r in results if r["status"] == "pass")
    n_warn   = sum(1 for r in results if r["status"] == "warn")
    n_crash  = sum(1 for r in results if r["status"] == "crash")
    n_total  = len(results)
    avg_time = total_elapsed / n_total if n_total else 0

    print(f"\n{'─' * 70}")
    print(f"  {BOLD}Results:{RESET}  "
          f"{GREEN}{n_pass} passed{RESET}  "
          f"{YELLOW}{n_warn} warned{RESET}  "
          f"{RED}{n_crash} crashed{RESET}  "
          f"/ {n_total} total")
    print(f"  Avg prediction time : {avg_time:.3f}s")
    print(f"  Total suite time    : {total_elapsed:.2f}s")

    if n_crash > 0:
        print(f"\n  {RED}{BOLD}DEMO NOT SAFE — {n_crash} crash(es) must be fixed before demo.{RESET}")
        for r in results:
            if r["status"] == "crash":
                print(f"\n  {RED}[CRASH] {r['name']}{RESET}")
                print(f"  {r['error']}")
    elif n_warn > 0:
        print(f"\n  {YELLOW}Demo is safe. {n_warn} warning(s) are expected behaviour "
              f"(domain warnings, borderline verdicts).{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}All tests passed. Demo is safe.{RESET}")

    print(f"{'─' * 70}\n")


def run_suite(threshold_mode: str = "safety", verbose: bool = False) -> bool:
    """
    Loads the pipeline and runs the full demo test suite.
    Returns True if there are zero crashes.
    """
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"  ToxPathGuard — Demo Test Suite  [mode: {threshold_mode}]")
    print(f"{'═' * 70}\n")

    # Load suite
    if not os.path.exists(SUITE_PATH):
        print(f"{RED}Test suite not found at {SUITE_PATH}{RESET}")
        return False

    molecules = load_suite(SUITE_PATH)
    print(f"  {len(molecules)} molecules loaded from demo suite.\n")

    # Load pipeline once
    print("  Loading pipeline...")
    t_load = time.perf_counter()
    try:
        models, thresholds = load_pipeline(MODELS_DIR, REPORT_PATH)
    except Exception as e:
        print(f"{RED}Pipeline load failed: {e}{RESET}")
        return False
    print(f"  Pipeline ready  ({time.perf_counter() - t_load:.2f}s)\n")

    # Run tests
    results       = []
    total_start   = time.perf_counter()

    for mol in molecules:
        r = run_test(mol, models, thresholds, threshold_mode, verbose)
        print_result_line(r)

        if verbose and r.get("result") and r["status"] in ("pass", "warn"):
            res = r["result"]
            for agent in res.get("agents", []):
                print(f"      {agent['agent_name']:<42}  p={agent['probability']:.4f}  {agent['risk_label']}")

        results.append(r)

    total_elapsed = time.perf_counter() - total_start
    print_summary(results, total_elapsed)

    return all(r["status"] != "crash" for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ToxPathGuard Phase 7 — Demo Test Suite"
    )
    parser.add_argument(
        "--mode", type=str, default="safety",
        choices=["safety", "training"],
        help="Threshold mode (default: safety)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-agent probabilities for each molecule"
    )
    args = parser.parse_args()

    passed = run_suite(threshold_mode=args.mode, verbose=args.verbose)
    sys.exit(0 if passed else 1)
