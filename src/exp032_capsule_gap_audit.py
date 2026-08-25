"""Audit gaps between EXP-030 capsule theory and its realised renderer."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from exp012b_core import atomic_json, sha256_file
from exp030_legal_evidence_routing import (
    build_candidate_views,
    canonical_answers,
    enrich_capsules,
    lexical_view_relevance,
    query_signals,
    read_jsonl,
    read_selected_jsonl,
    select_views,
    write_jsonl,
)


SCHEMA = "legalir.exp032_capsule_gap_audit.v1"


def _rates(counts: Counter[str], prefix: str) -> dict[str, float | int]:
    total = counts[f"{prefix}_n"]
    fields = ("scope", "relations", "title", "path_any", "path_all", "bm25", "ev1", "ev2", "aux")
    return {"documents": total, **{field: counts[f"{prefix}_{field}"] / total for field in fields}, "mean_selected_views": counts[f"{prefix}_view_sum"] / total}


def _length_summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(values), "p50": statistics.median(values),
        "p90": ordered[int(0.90 * (len(values) - 1))],
        "p99": ordered[int(0.99 * (len(values) - 1))], "max": max(values),
    }


def audit(
    *, results_root: Path, capsule_root: Path, metadata_dir: Path, v3_dir: Path,
    train: Path, preprocessing: Path, model_id: str, output: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    answers, label_stats = canonical_answers(train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    overlap: dict[str, list[float]] = defaultdict(list)
    truncation_examples = []
    materialized_root = output.parent / "enriched_samples"

    for sample_path in sorted((results_root / "samples").glob("fold_*.json")):
        outer = sample_path.stem
        qids = set(json.loads(sample_path.read_text(encoding="utf-8")))
        source_capsules = capsule_root / outer / "capsules" / "capsules.jsonl"
        enriched_dir = materialized_root / outer
        capsules = enriched_dir / "capsules.jsonl"
        if not (enriched_dir / "_SUCCESS.json").exists():
            subset = enriched_dir.parent / f"{outer}.source.jsonl"
            write_jsonl(subset, read_selected_jsonl(source_capsules, qids))
            enrich_capsules(
                capsules=subset, metadata_dir=metadata_dir, v3_dir=v3_dir,
                output_dir=enriched_dir,
            )
        for row in read_selected_jsonl(capsules, qids):
            qid, query = str(row["qid"]), str(row["query"])
            gold = answers[qid]
            signal_counts["queries"] += 1
            signal_counts.update(key for key, value in query_signals(query).items() if value)
            for candidate in row["candidates"]:
                group = "gold" if str(candidate["doc_id"]) in gold else "negative"
                evidence = list(candidate.get("evidence", []))
                counts[f"{group}_n"] += 1
                counts[f"{group}_scope"] += bool(candidate.get("typed_scope"))
                counts[f"{group}_relations"] += bool(candidate.get("header_relations"))
                counts[f"{group}_title"] += candidate.get("official_title", {}).get("status") == "VERIFIED"
                counts[f"{group}_path_any"] += any(item.get("structural_path") for item in evidence)
                counts[f"{group}_path_all"] += bool(evidence) and all(item.get("structural_path") for item in evidence)
                counts[f"{group}_bm25"] += any(item.get("provenance") != "e5" for item in evidence)
                counts[f"{group}_ev1"] += len(evidence) == 1
                counts[f"{group}_ev2"] += len(evidence) >= 2
                overlap[group].append(lexical_view_relevance(query, " ".join(str(item.get("raw_text") or "") for item in evidence)))

                views = build_candidate_views(query, candidate, identity_variant="both", include_structure=True, include_scope=True, include_relations=True)
                selected = select_views(query, views)
                counts[f"{group}_aux"] += len(selected) > 1
                counts[f"{group}_view_sum"] += len(selected)
                if group != "gold":
                    continue
                for view in views:
                    text = str(view["text"])
                    full_tokens = len(tokenizer(query, text, add_special_tokens=True, truncation=False)["input_ids"])
                    lengths[f"gold_{view['kind']}_full"].append(full_tokens)
                    marker = "[BẰNG CHỨNG TRẢ LỜI]"
                    prefix = text.split(marker, 1)[0] + marker if marker in text else text
                    prefix_tokens = len(tokenizer(query, prefix, add_special_tokens=True, truncation=False)["input_ids"])
                    lengths[f"gold_{view['kind']}_prefix"].append(prefix_tokens)
                    if view["kind"] != "base":
                        counts["gold_aux_views"] += 1
                        counts["gold_aux_answer_fully_cut"] += prefix_tokens >= 512
                        counts["gold_aux_truncated"] += full_tokens > 512
                        if prefix_tokens >= 512 and len(truncation_examples) < 12:
                            truncation_examples.append({
                                "qid": qid, "doc_id": str(candidate["doc_id"]), "kind": view["kind"],
                                "prefix_tokens": prefix_tokens, "full_tokens": full_tokens,
                                "scope_entries": len(candidate.get("typed_scope", [])),
                            })

    report = {
        "schema_version": SCHEMA,
        "source_results_sha256": sha256_file(results_root / "REPORT.json"),
        "label_fingerprint": label_stats["label_fingerprint"],
        "query_signals": dict(signal_counts),
        "coverage": {"gold": _rates(counts, "gold"), "negative": _rates(counts, "negative")},
        "evidence_lexical_relevance": {key: {"mean": statistics.mean(values), "median": statistics.median(values)} for key, values in overlap.items()},
        "token_lengths": {key: _length_summary(values) for key, values in sorted(lengths.items())},
        "auxiliary_truncation": {
            "gold_auxiliary_views": counts["gold_aux_views"],
            "truncated_over_512_rate": counts["gold_aux_truncated"] / max(1, counts["gold_aux_views"]),
            "answer_marker_after_512_rate": counts["gold_aux_answer_fully_cut"] / max(1, counts["gold_aux_views"]),
            "examples": truncation_examples,
        },
        "implementation_findings": {
            "typed_scope_is_not_optional_when_present": "select_views returns base plus the sole applicability auxiliary even without an explicit scope signal",
            "aggregation": "candidate-level max over a variable number of views",
            "passage_supervision": "document labels supervise capsules without answer-passage labels",
            "in_document_selection": "upstream E5/BM25 evidence is reused; no second-stage search across every chunk of the shortlisted document",
        },
    }
    atomic_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(
        results_root=args.results_root, capsule_root=args.capsule_root,
        metadata_dir=args.metadata_dir, v3_dir=args.v3_dir,
        train=args.train, preprocessing=args.preprocessing,
        model_id=args.model_id, output=args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "coverage": report["coverage"], "auxiliary_truncation": report["auxiliary_truncation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
