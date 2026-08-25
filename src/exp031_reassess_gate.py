"""Reassess EXP-031 with baseline-selected folds treated as neutral."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp012b_core import atomic_json, sha256_file


SCHEMA = "legalir.exp031_gate_reassessment.v1"
BASELINE = "unaccented_base"


def reassess(report: Mapping[str, Any]) -> dict[str, Any]:
    models = sorted({str(row["model"]) for row in report["per_fold"]})
    results = {}
    for model in models:
        rows = [row for row in report["per_fold"] if row["model"] == model]
        interventions = [row for row in rows if row["selected_variant"] != BASELINE]
        mean_recall = sum(float(row["delta_recall@5"]) for row in rows) / len(rows)
        mean_precision = sum(float(row["delta_precision@5"]) for row in rows) / len(rows)
        positive = sum(float(row["delta_recall@5"]) > 0 for row in interventions)
        required_positive = max(1, len(interventions) - 1) if interventions else 0
        worst = min(float(row["delta_recall@5"]) for row in rows)
        passed = bool(
            len(rows) == 5 and mean_recall >= 0.003 and mean_precision >= 0.0
            and worst >= -0.005 and positive >= required_positive
        )
        results[model] = {
            "passed": passed,
            "completed_outer_folds": len(rows),
            "neutral_baseline_folds": len(rows) - len(interventions),
            "intervention_folds": len(interventions),
            "positive_intervention_folds": positive,
            "required_positive_intervention_folds": required_positive,
            "mean_delta_recall@5": mean_recall,
            "mean_delta_precision@5": mean_precision,
            "worst_outer_delta_recall@5": worst,
        }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(row["passed"] for row in results.values()) else "REJECTED",
        "models": results,
        "note": "Baseline-selected folds are neutral; reassessment does not alter scores or model selection.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.report.read_text(encoding="utf-8"))
    result = reassess(source)
    result["source_report"] = str(args.report.resolve())
    result["source_report_sha256"] = sha256_file(args.report)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
