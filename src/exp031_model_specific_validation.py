"""Fresh outer-heldout validation of model-specific EXP-030 capsule policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp012b_core import sha256_file
from exp030_legal_evidence_routing import (
    CAPSULE_CONFIGS,
    LABEL_POLICY,
    _worker,
    atomic_json,
    canonical_answers,
    paired_bootstrap_delta,
    read_jsonl,
    score_metrics,
)


SCHEMA = "legalir.exp031_model_specific_validation.v1"
SEED = 31031
BASELINE = "unaccented_base"
MODELS = ("bge_m3", "gte")
CANONICAL_VARIANTS = tuple(CAPSULE_CONFIGS)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def label_blind_sample(qids: Sequence[str], *, outer: str, limit: int) -> set[str]:
    """Choose evaluation queries without consulting answers or candidate ranks."""
    ordered = sorted({str(qid) for qid in qids}, key=lambda qid: (_hash([SEED, outer, qid]), qid))
    return set(ordered[:limit])


def select_policies(gate: Mapping[str, Any], *, minimum_delta: float = 0.003) -> dict[str, dict[str, Any]]:
    """Select independently inside each outer fold; baseline is a valid safe choice."""
    rows = list(gate["rows"])
    outers = sorted({str(row["outer"]) for row in rows})
    policies: dict[str, dict[str, Any]] = {}
    for outer in outers:
        policies[outer] = {}
        for model in MODELS:
            candidates = []
            for variant in CANONICAL_VARIANTS:
                if variant == BASELINE:
                    continue
                selected = [
                    row for row in rows
                    if row["outer"] == outer and row["model"] == model and row["variant"] == variant
                ]
                expected_inner = len(outers) - 1
                if len(selected) != expected_inner:
                    raise ValueError(f"incomplete inner coverage: {outer}/{model}/{variant}")
                delta_recall = sum(float(row["delta_recall@5"]) for row in selected) / len(selected)
                delta_precision = sum(
                    float(row["metrics"]["precision@5"]) - float(row["baseline"]["precision@5"])
                    for row in selected
                ) / len(selected)
                candidates.append({
                    "variant": variant,
                    "inner_folds": len(selected),
                    "mean_delta_recall@5": delta_recall,
                    "mean_delta_precision@5": delta_precision,
                })
            eligible = [
                row for row in candidates
                if row["mean_delta_recall@5"] >= minimum_delta and row["mean_delta_precision@5"] >= 0.0
            ]
            if eligible:
                chosen = sorted(
                    eligible,
                    key=lambda row: (
                        -row["mean_delta_recall@5"], -row["mean_delta_precision@5"],
                        CANONICAL_VARIANTS.index(row["variant"]),
                    ),
                )[0]
                reason = "inner_only_gate_pass"
            else:
                chosen = {
                    "variant": BASELINE, "inner_folds": len(outers) - 1,
                    "mean_delta_recall@5": 0.0, "mean_delta_precision@5": 0.0,
                }
                reason = "safe_baseline_no_variant_passed"
            policies[outer][model] = {**chosen, "reason": reason, "candidates": candidates}
    return policies


def _marker_path(root: Path, name: str) -> Path:
    return root / "jobs" / name / "_SUCCESS.json"


def run_job(
    *, root: Path, name: str, fingerprint: str, output: Path, model: str, capsules: Path,
    qids: set[str], variant: str, device: str, local_only: bool, resume: bool,
) -> dict[str, Any]:
    marker = _marker_path(root, name)
    if resume and marker.exists():
        payload = _json(marker)
        if payload.get("fingerprint") == fingerprint and (output / "scores.jsonl").exists():
            return {**payload, "state": "SKIPPED_RESUME"}
    marker.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = _worker(
            command="score-worker", output_dir=output, model=model, capsules=capsules,
            qids=qids, device=device, local_only=local_only, capsule_variant=variant,
        )
        payload = {
            "schema_version": SCHEMA, "state": "SUCCESS", "fingerprint": fingerprint,
            "started_at": started, "finished_at": time.time(), "result": result,
        }
        atomic_json(marker, payload)
        return payload
    except Exception as error:
        payload = {
            "schema_version": SCHEMA, "state": "FAILED_MODEL", "fingerprint": fingerprint,
            "started_at": started, "finished_at": time.time(), "error": repr(error),
        }
        atomic_json(marker.parent / "_FAILED.json", payload)
        return payload


def evaluate(
    *, exp030_root: Path, cache_root: Path, train: Path, folds_path: Path, preprocessing: Path,
    output_root: Path, sample_per_outer: int, device: str, local_only: bool, resume: bool,
) -> dict[str, Any]:
    gate_path = exp030_root / "bounded_gate.json"
    gate = _json(gate_path)
    answers, label_stats = canonical_answers(
        train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl",
    )
    if label_stats["policy"] != LABEL_POLICY or label_stats["evaluable_queries"] != 6991:
        raise ValueError("canonical label contract drift")
    folds = {str(key): [str(qid) for qid in values] for key, values in _json(folds_path).items()}
    policies = select_policies(gate)
    source_fingerprint = _hash({
        "gate": sha256_file(gate_path), "train": sha256_file(train), "folds": sha256_file(folds_path),
        "code": sha256_file(Path(__file__)), "exp030_code": sha256_file(Path(__file__).with_name("exp030_legal_evidence_routing.py")),
        "label_fingerprint": label_stats["label_fingerprint"], "sample_per_outer": sample_per_outer,
        "policies": policies,
    })
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "run_manifest.json", {
        "schema_version": SCHEMA, "fingerprint": source_fingerprint, "policies": policies,
        "sample_rule": "label_blind_sha256_v1", "sample_per_outer": sample_per_outer,
        "label_fingerprint": label_stats["label_fingerprint"],
    })

    jobs = []
    for outer in sorted(folds):
        eligible = [qid for qid in folds[outer] if answers[qid]]
        sample = label_blind_sample(eligible, outer=outer, limit=sample_per_outer)
        atomic_json(output_root / "samples" / f"{outer}.json", sorted(sample))
        capsule = cache_root / "canonical_cascade" / label_stats["label_fingerprint"][:16] / outer / "capsules" / "capsules.jsonl"
        if not capsule.exists():
            raise FileNotFoundError(capsule)
        for model in MODELS:
            variants = [BASELINE]
            selected_variant = str(policies[outer][model]["variant"])
            if selected_variant != BASELINE:
                variants.append(selected_variant)
            for variant in variants:
                name = f"heldout-{outer}-{model}-{variant}"
                output = output_root / "heldout" / outer / model / variant
                fingerprint = _hash([source_fingerprint, outer, model, variant, sorted(sample)])
                result = run_job(
                    root=output_root, name=name, fingerprint=fingerprint, output=output, model=model,
                    capsules=capsule, qids=sample, variant=variant, device=device,
                    local_only=local_only, resume=resume,
                )
                jobs.append({"outer": outer, "model": model, "variant": variant, **result})
                atomic_json(output_root / "RUN_STATUS.json", {
                    "schema_version": SCHEMA, "fingerprint": source_fingerprint,
                    "completed_jobs": sum(row["state"] in {"SUCCESS", "SKIPPED_RESUME"} for row in jobs),
                    "total_jobs": sum(1 + (policies[o][m]["variant"] != BASELINE) for o in folds for m in MODELS),
                    "jobs": jobs,
                })

    per_fold = []
    for outer in sorted(folds):
        for model in MODELS:
            selected = str(policies[outer][model]["variant"])
            baseline_path = output_root / "heldout" / outer / model / BASELINE / "scores.jsonl"
            selected_path = output_root / "heldout" / outer / model / selected / "scores.jsonl"
            if not baseline_path.exists() or not selected_path.exists():
                continue
            baseline_metrics = score_metrics(baseline_path, answers)
            selected_metrics = score_metrics(selected_path, answers)
            delta = paired_bootstrap_delta(baseline_path, selected_path, answers, samples=5000, seed=SEED)
            per_fold.append({
                "outer": outer, "model": model, "selected_variant": selected,
                "baseline": baseline_metrics, "selected": selected_metrics,
                "delta_precision@5": selected_metrics["precision@5"] - baseline_metrics["precision@5"],
                **delta,
            })

    aggregate = {}
    for model in MODELS:
        rows = [row for row in per_fold if row["model"] == model]
        aggregate[model] = {
            "completed_folds": len(rows),
            "mean_baseline_recall@5": sum(row["baseline"]["recall@5"] for row in rows) / len(rows) if rows else None,
            "mean_selected_recall@5": sum(row["selected"]["recall@5"] for row in rows) / len(rows) if rows else None,
            "mean_delta_recall@5": sum(row["delta_recall@5"] for row in rows) / len(rows) if rows else None,
            "mean_delta_precision@5": sum(row["delta_precision@5"] for row in rows) / len(rows) if rows else None,
            "positive_outer_folds": sum(row["delta_recall@5"] > 0 for row in rows),
            "worst_outer_delta": min((row["delta_recall@5"] for row in rows), default=None),
        }
        value = aggregate[model]
        value["passed"] = bool(
            len(rows) == len(folds) and value["mean_delta_recall@5"] >= 0.003
            and value["mean_delta_precision@5"] >= 0.0 and value["positive_outer_folds"] >= 4
            and value["worst_outer_delta"] >= -0.005
        )
    report = {
        "schema_version": SCHEMA, "status": "PASS" if all(aggregate[m]["passed"] for m in MODELS) else "REJECTED",
        "fingerprint": source_fingerprint, "policies": policies, "jobs": jobs,
        "per_fold": per_fold, "aggregate": aggregate,
        "scope": "fresh label-blind bounded outer-heldout confirmation; not full OOF and not fine-tuning evidence",
    }
    atomic_json(output_root / "REPORT.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp030-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-per-outer", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate(
        exp030_root=args.exp030_root, cache_root=args.cache_root, train=args.train, folds_path=args.folds,
        preprocessing=args.preprocessing, output_root=args.output_root, sample_per_outer=args.sample_per_outer,
        device=args.device, local_only=args.local_only, resume=args.resume,
    )
    print(json.dumps({"status": report["status"], "report": str((args.output_root / 'REPORT.json').resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
