"""EXP-031: read-only diagnostics for EXP-030 capsule/model interactions.

This module never rescores a model.  It consumes the completed bounded screen,
reconstructs per-query rank movement, and tests score aggregation hypotheses
from the already persisted per-view logits.  Any promising rule remains a
development-screen result until evaluated on untouched outer-heldout queries.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from exp030_legal_evidence_routing import (
    LABEL_POLICY,
    accent_fold,
    atomic_json,
    canonical_answers,
    query_signals,
    read_jsonl,
)


SCHEMA = "legalir.exp031_capsule_diagnostics.v1"
BASELINE = "unaccented_base"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return sum(rows) / len(rows) if rows else None


def _top(scores: Mapping[str, float], k: int = 5) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:k]]


def _score_map(row: Mapping[str, Any], reducer: Callable[[Sequence[float]], float] | None = None) -> dict[str, float]:
    result = {}
    for item in row["scores"]:
        values = [float(value) for value in item.get("view_scores", [])]
        result[str(item["doc_id"])] = float(item["score"]) if reducer is None else reducer(values)
    return result


def _rank_map(doc_ids: Sequence[str]) -> dict[str, int]:
    return {str(doc_id): rank for rank, doc_id in enumerate(doc_ids, 1)}


def fuse_rankings(
    original: Sequence[str], reranked: Sequence[str], *, method: str, weight: float = 0.5, rrf_k: int = 32,
) -> list[str]:
    """Fuse two complete, identical candidate permutations deterministically."""
    if len(original) != len(reranked) or set(original) != set(reranked):
        raise ValueError("rank fusion requires identical candidate membership")
    left, right = _rank_map(original), _rank_map(reranked)
    if method == "borda":
        score = {doc_id: -((1.0 - weight) * left[doc_id] + weight * right[doc_id]) for doc_id in left}
    elif method == "rrf":
        score = {
            doc_id: (1.0 - weight) / (rrf_k + left[doc_id]) + weight / (rrf_k + right[doc_id])
            for doc_id in left
        }
    else:
        raise ValueError(f"unknown fusion method: {method}")
    return sorted(left, key=lambda doc_id: (-score[doc_id], left[doc_id], doc_id))


def _recall_precision(top: Sequence[str], gold: set[str]) -> tuple[float, float]:
    hits = len(set(top) & gold)
    return hits / len(gold), hits / 5.0


def _query_bucket(query: str, gold: set[str]) -> dict[str, str]:
    signals = query_signals(query)
    active = [name.removeprefix("explicit_") for name, value in signals.items() if value]
    token_count = len(accent_fold(query).split())
    return {
        "signal": "+".join(active) if active else "none",
        "length": "short" if token_count <= 8 else ("medium" if token_count <= 16 else "long"),
        "gold_count": str(min(len(gold), 3)),
    }


def _load_queries(train: Path) -> dict[str, str]:
    data = _json(train)
    return {str(qid): str(row["question"]) for qid, row in data.items()}


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["doc_id"]): row for row in read_jsonl(path)}


def _load_job_rows(root: Path, gate: Mapping[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    keys = {
        (str(row["outer"]), str(row["inner"]), str(row["model"]), str(row["variant"]))
        for row in gate["rows"]
    }
    keys |= {(outer, inner, model, BASELINE) for outer, inner, model, _ in keys}
    result = {}
    for key in sorted(keys):
        outer, inner, model, variant = key
        path = root / "bounded_screen" / outer / inner / model / variant / "scores.jsonl"
        rows = {str(row["qid"]): row for row in read_jsonl(path)}
        if len(rows) != 64:
            raise ValueError(f"expected 64 score rows at {path}, got {len(rows)}")
        result[key] = rows
    return result


def _aggregate_slices(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[field])].append(row)
    return {
        key: {
            "queries": len(rows),
            "delta_recall@5": _mean(float(row["delta_recall@5"]) for row in rows),
            "delta_precision@5": _mean(float(row["delta_precision@5"]) for row in rows),
            "wins": sum(float(row["delta_recall@5"]) > 0 for row in rows),
            "losses": sum(float(row["delta_recall@5"]) < 0 for row in rows),
        }
        for key, rows in sorted(grouped.items())
    }


def rank_movement(
    *, job_rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    queries: Mapping[str, str], answers: Mapping[str, set[str]], metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (outer, inner, model, variant), variants in sorted(job_rows.items()):
        if variant == BASELINE:
            continue
        baselines = job_rows[(outer, inner, model, BASELINE)]
        if set(baselines) != set(variants):
            raise ValueError(f"qid mismatch for {outer}/{inner}/{model}/{variant}")
        for qid in sorted(variants):
            gold = answers[qid]
            baseline_top = _top(_score_map(baselines[qid]))
            variant_top = _top(_score_map(variants[qid]))
            base_recall, base_precision = _recall_precision(baseline_top, gold)
            recall, precision = _recall_precision(variant_top, gold)
            bucket = _query_bucket(queries[qid], gold)
            entered = set(variant_top) - set(baseline_top)
            left = set(baseline_top) - set(variant_top)
            grouped[(model, variant)].append({
                "qid": qid,
                "outer": outer,
                "inner": inner,
                "delta_recall@5": recall - base_recall,
                "delta_precision@5": precision - base_precision,
                "top5_churn": len(entered),
                "entered_gold": len(entered & gold),
                "left_gold": len(left & gold),
                "entered_verified_title": sum(metadata.get(doc_id, {}).get("official_title", {}).get("status") == "VERIFIED" for doc_id in entered),
                "entered_typed_scope": sum(bool(metadata.get(doc_id, {}).get("typed_scope")) for doc_id in entered),
                **bucket,
            })

    report = {}
    for (model, variant), rows in sorted(grouped.items()):
        report[f"{model}/{variant}"] = {
            "queries": len(rows),
            "delta_recall@5": _mean(float(row["delta_recall@5"]) for row in rows),
            "delta_precision@5": _mean(float(row["delta_precision@5"]) for row in rows),
            "wins": sum(float(row["delta_recall@5"]) > 0 for row in rows),
            "losses": sum(float(row["delta_recall@5"]) < 0 for row in rows),
            "mean_top5_churn": _mean(float(row["top5_churn"]) for row in rows),
            "entered_gold": sum(int(row["entered_gold"]) for row in rows),
            "left_gold": sum(int(row["left_gold"]) for row in rows),
            "entered_verified_title_rate": (
                sum(int(row["entered_verified_title"]) for row in rows)
                / max(1, sum(int(row["top5_churn"]) for row in rows))
            ),
            "entered_typed_scope_rate": (
                sum(int(row["entered_typed_scope"]) for row in rows)
                / max(1, sum(int(row["top5_churn"]) for row in rows))
            ),
            "by_signal": _aggregate_slices(rows, "signal"),
            "by_length": _aggregate_slices(rows, "length"),
            "by_gold_count": _aggregate_slices(rows, "gold_count"),
        }
    return report


def _reducers() -> dict[str, Callable[[Sequence[float]], float]]:
    reducers: dict[str, Callable[[Sequence[float]], float]] = {
        "base_only": lambda values: values[0],
        "mean": lambda values: sum(values) / len(values),
        "max": max,
        "logsumexp_mean": lambda values: max(values) + math.log(sum(math.exp(value - max(values)) for value in values) / len(values)),
    }
    for weight in (0.25, 0.5, 0.75):
        reducers[f"residual_{weight:.2f}"] = lambda values, weight=weight: values[0] + weight * (max(values) - values[0])
    for penalty in (0.05, 0.1, 0.2, 0.5):
        reducers[f"penalized_max_{penalty:.2f}"] = lambda values, penalty=penalty: max(values) - penalty * math.log(len(values))
    return reducers


def aggregation_screen(
    *, job_rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    answers: Mapping[str, set[str]], variant: str = "multi_view",
) -> dict[str, Any]:
    per_job: dict[tuple[str, str, str, str], dict[str, float]] = {}
    view_count_stats: dict[str, Counter[int]] = defaultdict(Counter)
    first_view_mismatches = 0
    comparisons = 0
    reducers = _reducers()
    for (outer, inner, model, current_variant), rows in sorted(job_rows.items()):
        if current_variant != variant:
            continue
        baseline_rows = job_rows[(outer, inner, model, BASELINE)]
        both_rows = job_rows[(outer, inner, model, "both_base")]
        for qid, row in rows.items():
            both_map = _score_map(both_rows[qid])
            for item in row["scores"]:
                values = [float(value) for value in item["view_scores"]]
                view_count_stats[model][len(values)] += 1
                comparisons += 1
                if not math.isclose(values[0], both_map[str(item["doc_id"])], rel_tol=0.0, abs_tol=2e-5):
                    first_view_mismatches += 1
        for name, reducer in reducers.items():
            recalls, precisions, base_recalls, base_precisions = [], [], [], []
            for qid in sorted(rows):
                gold = answers[qid]
                recall, precision = _recall_precision(_top(_score_map(rows[qid], reducer)), gold)
                base_recall, base_precision = _recall_precision(_top(_score_map(baseline_rows[qid])), gold)
                recalls.append(recall)
                precisions.append(precision)
                base_recalls.append(base_recall)
                base_precisions.append(base_precision)
            per_job[(outer, inner, model, name)] = {
                "recall@5": float(_mean(recalls)),
                "precision@5": float(_mean(precisions)),
                "delta_recall@5": float(_mean(r - b for r, b in zip(recalls, base_recalls))),
                "delta_precision@5": float(_mean(p - b for p, b in zip(precisions, base_precisions))),
            }

    methods = {}
    for model in sorted({key[2] for key in per_job}):
        for name in reducers:
            rows = [(key, value) for key, value in per_job.items() if key[2] == model and key[3] == name]
            outer_delta = {
                outer: float(_mean(value["delta_recall@5"] for key, value in rows if key[0] == outer))
                for outer in sorted({key[0] for key, _ in rows})
            }
            deltas = list(outer_delta.values())
            methods[f"{model}/{name}"] = {
                "jobs": len(rows),
                "recall@5": _mean(value["recall@5"] for _, value in rows),
                "precision@5": _mean(value["precision@5"] for _, value in rows),
                "delta_recall@5": _mean(value["delta_recall@5"] for _, value in rows),
                "delta_precision@5": _mean(value["delta_precision@5"] for _, value in rows),
                "outer_delta_recall@5": outer_delta,
                "positive_outer_folds": sum(value > 0 for value in deltas),
                "worst_outer_delta": min(deltas),
            }
    return {
        "variant": variant,
        "first_view_matches_both_base": first_view_mismatches == 0,
        "first_view_mismatches": first_view_mismatches,
        "comparisons": comparisons,
        "view_count_distribution": {model: dict(sorted(counts.items())) for model, counts in view_count_stats.items()},
        "methods": methods,
        "warning": "Exploratory reuse of bounded discovery rows; requires fresh outer-heldout confirmation.",
    }


def fusion_screen(
    *, job_rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    answers: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Explore LambdaMART/reranker rank fusion on bounded discovery rows."""
    configurations = [("borda", weight, 0) for weight in (0.25, 0.5, 0.75)]
    configurations += [("rrf", weight, rrf_k) for rrf_k in (10, 32, 60) for weight in (0.25, 0.5, 0.75)]
    configurations += [("window", float(window), 0) for window in (8, 10, 16, 32)]
    per_job: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    oracle_rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    preranker_rows: list[tuple[float, float]] = []
    for (outer, inner, model, variant), rows in sorted(job_rows.items()):
        if variant == BASELINE:
            for qid, row in rows.items():
                original = [str(item["doc_id"]) for item in row["scores"]]
                preranker_rows.append(_recall_precision(original[:5], answers[qid]))
        for method, value, rrf_k in configurations:
            recalls, precisions = [], []
            name = f"{method}_{int(value) if method == 'window' else f'{value:.2f}'}"
            if method == "rrf":
                name += f"_k{rrf_k}"
            for qid, row in rows.items():
                original = [str(item["doc_id"]) for item in row["scores"]]
                reranked = _top(_score_map(row), k=len(original))
                if method == "window":
                    window = int(value)
                    rerank_map = _score_map(row)
                    combined = sorted(original[:window], key=lambda doc_id: (-rerank_map[doc_id], original.index(doc_id), doc_id)) + original[window:]
                else:
                    combined = fuse_rankings(original, reranked, method=method, weight=value, rrf_k=rrf_k)
                recall, precision = _recall_precision(combined[:5], answers[qid])
                recalls.append(recall)
                precisions.append(precision)
                if method == "borda" and value == 0.5:
                    union = set(original[:5]) | set(reranked[:5])
                    oracle_rows[(model, variant)].append(len(union & answers[qid]) / len(answers[qid]))
            per_job[(outer, inner, model, variant, name)] = {
                "recall@5": float(_mean(recalls)), "precision@5": float(_mean(precisions)),
            }

    summary = {}
    for model, variant, name in sorted({(key[2], key[3], key[4]) for key in per_job}):
        rows = [(key, value) for key, value in per_job.items() if key[2:] == (model, variant, name)]
        outer_recall = {
            outer: float(_mean(value["recall@5"] for key, value in rows if key[0] == outer))
            for outer in sorted({key[0] for key, _ in rows})
        }
        summary[f"{model}/{variant}/{name}"] = {
            "recall@5": _mean(value["recall@5"] for _, value in rows),
            "precision@5": _mean(value["precision@5"] for _, value in rows),
            "outer_recall@5": outer_recall,
            "worst_outer_recall@5": min(outer_recall.values()),
        }
    return {
        "preranker": {
            "recall@5": _mean(row[0] for row in preranker_rows),
            "precision@5": _mean(row[1] for row in preranker_rows),
            "rows": len(preranker_rows),
        },
        "top5_union_oracle": {
            f"{model}/{variant}": _mean(values) for (model, variant), values in sorted(oracle_rows.items())
        },
        "methods": summary,
        "warning": "Exploratory reuse of bounded discovery rows; select inside each outer fold and confirm on fresh heldout rows.",
    }


def title_scope_coverage(
    *, job_rows: Mapping[tuple[str, str, str, str], Mapping[str, Mapping[str, Any]]],
    answers: Mapping[str, set[str]], metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    qids = sorted({qid for rows in job_rows.values() for qid in rows})
    candidate_ids = {
        str(item["doc_id"])
        for key, rows in job_rows.items() if key[3] == BASELINE
        for row in rows.values() for item in row["scores"]
    }
    gold_ids = {doc_id for qid in qids for doc_id in answers[qid] if doc_id in candidate_ids}
    def summarize(doc_ids: set[str]) -> dict[str, Any]:
        return {
            "documents": len(doc_ids),
            "verified_title": sum(metadata.get(doc_id, {}).get("official_title", {}).get("status") == "VERIFIED" for doc_id in doc_ids),
            "typed_scope": sum(bool(metadata.get(doc_id, {}).get("typed_scope")) for doc_id in doc_ids),
            "relations": sum(bool(metadata.get(doc_id, {}).get("relations")) for doc_id in doc_ids),
        }
    return {"bounded_candidates": summarize(candidate_ids), "retained_gold_documents": summarize(gold_ids)}


def analyze(*, exp030_root: Path, train: Path, preprocessing: Path, metadata_path: Path, output: Path) -> dict[str, Any]:
    gate = _json(exp030_root / "bounded_gate.json")
    answers, label_stats = canonical_answers(
        train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl",
    )
    if label_stats["policy"] != LABEL_POLICY or label_stats["evaluable_queries"] != 6991:
        raise ValueError("canonical label contract drift")
    queries = _load_queries(train)
    metadata = _load_metadata(metadata_path)
    jobs = _load_job_rows(exp030_root, gate)
    report = {
        "schema_version": SCHEMA,
        "source_experiment": str(exp030_root.resolve()),
        "label_fingerprint": label_stats["label_fingerprint"],
        "jobs": len(jobs),
        "score_rows": sum(len(rows) for rows in jobs.values()),
        "rank_movement": rank_movement(job_rows=jobs, queries=queries, answers=answers, metadata=metadata),
        "aggregation_screen": aggregation_screen(job_rows=jobs, answers=answers),
        "fusion_screen": fusion_screen(job_rows=jobs, answers=answers),
        "metadata_coverage": title_scope_coverage(job_rows=jobs, answers=answers, metadata=metadata),
        "interpretation_contract": {
            "development_only": True,
            "may_override_exp030_gate": False,
            "requires_fresh_outer_heldout_confirmation": True,
        },
    }
    atomic_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp030-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(
        exp030_root=args.exp030_root, train=args.train, preprocessing=args.preprocessing,
        metadata_path=args.metadata, output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "jobs": report["jobs"], "score_rows": report["score_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
