"""Build a read-only, query-level audit from the completed EXP-112 OOF run.

This script never writes into EXP-112. It reads the immutable OOF predictions,
canonical labels, and frozen-source SQLite store, then writes EXP-final evidence.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "exp112_task_adaptive_retrieval"
OUT = ROOT / "results" / "exp_final_retrieval" / "error_audit"
SOURCE_DB = ROOT / "cache" / "exp112_task_adaptive_retrieval" / "sources.sqlite"
CONTEXT = ROOT / "public_test_dataset" / "selected-contexts"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def fold_text(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()))


def tokens(value: str) -> set[str]:
    return {t for t in fold_text(value).split() if len(t) > 1}


def best_excerpt(question: str, passage: str, width: int = 420) -> str:
    q = tokens(question)
    clean = re.sub(r"\s+", " ", passage).strip()
    if len(clean) <= width:
        return clean
    candidates = []
    step = max(80, width // 2)
    for start in range(0, len(clean), step):
        piece = clean[start : start + width]
        score = len(tokens(piece) & q) / max(1, len(q))
        candidates.append((score, -start, piece))
    return max(candidates)[2]


def context(doc_id: str) -> dict:
    path = CONTEXT / f"context_{doc_id}.json"
    if not path.exists():
        return {"id": doc_id, "passage": "", "link": "", "missing": True}
    return read(path)


def source_rows(db: sqlite3.Connection, qid: str) -> dict[str, list[dict]]:
    result = {}
    for source, payload in db.execute("SELECT source,payload FROM sources WHERE q=?", (qid,)):
        result[source] = json.loads(payload)
    return result


def rank_map(rows: list[dict]) -> dict[str, int]:
    return {str(row["doc_id"]): int(row.get("rank", i)) for i, row in enumerate(rows, 1)}


def bucket(rank: int | None) -> str:
    if rank is None:
        return "outside_union"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 50:
        return "11-50"
    if rank <= 100:
        return "51-100"
    return "101+"


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    import exp109b_encoder_complementarity as old

    train = read(ROOT / "public_test_dataset" / "train.json")
    canonical, label_audit = old.canonical_labels()
    metadata = old.build_parent_text_metadata()
    assignment_frequency = Counter(d for gold in canonical.values() for d in gold)

    db = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    summary = Counter()
    source_topk = {s: Counter() for s in ("e5", "lal", "bm25", "trigram", "jina")}
    records = []
    per_query = []

    for fold in range(5):
        predictions = read(OLD / "outer" / f"fold_{fold}" / "PREDICTIONS.json")
        for qid, pred in predictions.items():
            gold = set(canonical.get(qid, set()))
            if not gold:
                continue
            order = list(map(str, pred["order"]))
            final_ranks = {d: i for i, d in enumerate(order, 1)}
            sources = source_rows(db, qid)
            source_ranks = {s: rank_map(sources.get(s, [])) for s in source_topk}
            top5 = set(order[:5])
            q_recall = len(top5 & gold) / len(gold)
            per_query.append({"qid": qid, "fold": fold, "gold_count": len(gold), "recall@5": q_recall})
            for doc_id in sorted(gold):
                fr = final_ranks.get(doc_id)
                ranks = {s: source_ranks[s].get(doc_id) for s in source_topk}
                best_source_rank = min((r for r in ranks.values() if r is not None), default=None)
                b = bucket(fr)
                summary[f"final_{b}"] += 1
                if fr is None:
                    summary["candidate_miss"] += 1
                elif fr > 5:
                    summary["ranker_miss"] += 1
                if fr is not None and fr > 5 and best_source_rank is not None and best_source_rank <= 5:
                    summary["fusion_harm_source_top5_to_final_out"] += 1
                if fr is not None and fr <= 5 and (best_source_rank is None or best_source_rank > 5):
                    summary["fusion_rescue_all_sources_out_to_final_top5"] += 1
                for s, r in ranks.items():
                    for k in (1, 5, 10, 50, 100, 200, 500):
                        if r is not None and r <= k:
                            source_topk[s][k] += 1
                records.append(
                    {
                        "qid": qid,
                        "fold": fold,
                        "question": train[qid]["question"],
                        "gold_count": len(gold),
                        "gold_doc": doc_id,
                        "gold_frequency": assignment_frequency[doc_id],
                        "final_rank": fr,
                        "final_bucket": b,
                        "source_ranks": ranks,
                        "best_source_rank": best_source_rank,
                        "top5": order[:5],
                        "top5_scores": pred.get("scores", [])[:5],
                        "parent_chunk_count": metadata.get(doc_id, {}).get("parent_chunk_count"),
                        "parent_token_length": metadata.get(doc_id, {}).get("parent_token_length"),
                    }
                )

    assignment_total = len(records)
    strata = {}
    for field, groups in {
        "gold_frequency": [("unseen_elsewhere", lambda r: r["gold_frequency"] == 1), ("repeated", lambda r: r["gold_frequency"] > 1)],
        "gold_count": [("single", lambda r: r["gold_count"] == 1), ("multi", lambda r: r["gold_count"] > 1)],
        "document_length": [("short_le_512", lambda r: (r["parent_token_length"] or 0) <= 512),
                            ("medium_513_2048", lambda r: 512 < (r["parent_token_length"] or 0) <= 2048),
                            ("long_gt_2048", lambda r: (r["parent_token_length"] or 0) > 2048)],
    }.items():
        strata[field] = {}
        for name, predicate in groups:
            rows = [r for r in records if predicate(r)]
            strata[field][name] = {
                "gold_assignments": len(rows),
                "top5": sum((r["final_rank"] or 10**9) <= 5 for r in rows),
                "assignment_recall@5": sum((r["final_rank"] or 10**9) <= 5 for r in rows) / max(1, len(rows)),
                "outside_union": sum(r["final_rank"] is None for r in rows),
            }

    # Pick diverse, decision-relevant misses. Direct evidence is embedded so the
    # report can be inspected without another multi-GB corpus scan.
    classes = {
        "fusion_harm": lambda r: r["final_rank"] is not None and r["final_rank"] > 5 and (r["best_source_rank"] or 10**9) <= 5,
        "near_boundary": lambda r: r["final_rank"] is not None and 6 <= r["final_rank"] <= 10,
        "deep_ranker_miss": lambda r: r["final_rank"] is not None and 11 <= r["final_rank"] <= 50,
        "candidate_miss": lambda r: r["final_rank"] is None,
        "multi_gold_miss": lambda r: r["gold_count"] > 1 and (r["final_rank"] is None or r["final_rank"] > 5),
    }
    samples = {}
    for name, predicate in classes.items():
        candidates = sorted((r for r in records if predicate(r)), key=lambda r: (r["fold"], r["qid"], r["gold_doc"]))[:12]
        enriched = []
        for row in candidates:
            gold_ctx = context(row["gold_doc"])
            false = []
            for doc_id in row["top5"][:3]:
                ctx = context(doc_id)
                false.append({"doc_id": doc_id, "link": ctx.get("link", ""), "excerpt": best_excerpt(row["question"], ctx.get("passage", ""))})
            enriched.append(row | {"gold_link": gold_ctx.get("link", ""),
                                   "gold_excerpt": best_excerpt(row["question"], gold_ctx.get("passage", "")),
                                   "top_false_positive_evidence": false})
        samples[name] = enriched

    report = {
        "status": "COMPLETE_READ_ONLY_ERROR_AUDIT",
        "source_experiment": "EXP-112 five-fold OOF",
        "metric_unit_note": "Counts below are gold assignments; macro query recall remains the official selection unit.",
        "canonical_label_audit": label_audit,
        "queries": len(per_query),
        "gold_assignments": assignment_total,
        "summary": dict(summary),
        "source_gold_assignment_coverage": {
            s: {f"recall@{k}": v / assignment_total for k, v in counts.items()} for s, counts in source_topk.items()
        },
        "strata": strata,
        "samples": samples,
    }
    write(OUT / "ERROR_AUDIT.json", report)
    write(OUT / "GOLD_ASSIGNMENTS.json", records)
    write(OUT / "QUERY_RECALL.json", per_query)
    print(json.dumps({k: report[k] for k in ("status", "queries", "gold_assignments", "summary", "source_gold_assignment_coverage", "strata")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
