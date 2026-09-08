"""
Run Robustness and Grouped-OOD Diagnostics Audit across all 15 frozen slices.

Loads diagnostic slices from results/gemini/ROBUSTNESS_DIAGNOSTIC_SLICES.json
and evaluates predictions against the locked baseline anchor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels
from gemini.metrics import compute_metrics


def run_diagnostics(preds_path: Path | None = None):
    labels, _ = get_canonical_labels()

    diag = json.load(open(ROOT / "results/gemini/ROBUSTNESS_DIAGNOSTIC_SLICES.json", "r", encoding="utf-8"))
    slices = diag["slices"]

    if preds_path is None:
        preds_path = ROOT / "cache/gemini/nested_cv_v5_oof_preds.json"

    preds_eval = json.load(open(preds_path, "r", encoding="utf-8"))
    preds_base = json.load(open(ROOT / "results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json", "r", encoding="utf-8"))

    print("=" * 90)
    print(f"ROBUSTNESS & GROUPED-OOD DIAGNOSTIC AUDIT: {preds_path.name}")
    print("=" * 90)
    print(f"{'Slice Name':22s} | {'Count':5s} | {'Baseline R@5':12s} | {'Model R@5':10s} | {'Net Delta':10s} | {'Hits Base':9s} | {'Hits Model':10s}")
    print("-" * 90)

    all_non_negative = True
    results = {}

    for s_name, qids in slices.items():
        valid_qids = [q for q in qids if labels.get(q)]
        m_base = compute_metrics({q: preds_base[q][:5] for q in valid_qids}, labels, valid_qids)
        m_eval = compute_metrics({q: preds_eval[q][:5] for q in valid_qids}, labels, valid_qids)

        r_base = m_base["recall_at_5"]
        r_eval = m_eval["recall_at_5"]
        delta = r_eval - r_base

        h_base = int(round(r_base * len(valid_qids)))
        h_eval = int(round(r_eval * len(valid_qids)))

        status = "GAIN" if delta > 0 else ("TIE" if delta == 0 else "LOSS")
        if delta < 0:
            all_non_negative = False

        results[s_name] = {
            "count": len(valid_qids),
            "baseline_r5": r_base,
            "model_r5": r_eval,
            "delta": delta,
            "status": status,
        }

        print(f"{s_name:22s} | {len(valid_qids):5d} | {r_base:12.6f} | {r_eval:10.6f} | {delta:+10.6f} | {h_base:4d}/{len(valid_qids)} | {h_eval:4d}/{len(valid_qids)} ({status})")

    print("=" * 90)
    print(f"All diagnostic slices non-negative: {all_non_negative}")
    return results, all_non_negative


if __name__ == "__main__":
    preds_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_diagnostics(preds_file)
