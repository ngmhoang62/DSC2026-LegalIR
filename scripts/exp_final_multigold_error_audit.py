"""Audit recoverable multi-gold errors at the top-five boundary."""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_final.relations import amendment_relation, ascii_words, authority  # noqa: E402
from exp_final.slate import token_set  # noqa: E402

OUT = ROOT / "results/exp_final_retrieval/multigold_error_audit"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def blend(systems, weights=(.3, .4, .3), k=10, depth=64):
    out = {}
    for qid in systems[0]:
        score = {}
        for system, weight in zip(systems, weights):
            for rank, doc in enumerate(system[qid][:depth], 1):
                score[doc] = score.get(doc, 0.) + weight / (k + rank)
        out[qid] = sorted(score, key=lambda doc: (-score[doc], doc))
    return out


def main():
    import exp109b_encoder_complementarity as old

    labels, _ = old.canonical_labels()
    folds = read(ROOT / "cache/cv_folds.json")
    train = read(ROOT / "public_test_dataset/train.json")
    xgb = read(ROOT / "results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json")
    lgb = read(ROOT / "results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json")
    profile = read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json")
    base = blend([xgb, lgb, profile])
    db_path = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    titles = {str(doc): json.loads(payload).get("retrieval_name", "") for doc, payload in db.execute("select doc,payload from documents")}
    db.close()
    fold_of = {qid: fold for fold in range(5) for qid in folds[f"fold_{fold}"]}
    rows = []
    rank_counter = Counter()
    relationship_counter = Counter()
    intent_counter = Counter()
    for qid, gold in labels.items():
        if len(gold) <= 1 or qid not in base:
            continue
        order = base[qid]
        ranks = {doc: rank for rank, doc in enumerate(order, 1)}
        top_gold = [doc for doc in order[:5] if doc in gold]
        missed = sorted((doc for doc in gold if ranks.get(doc, 10_000) > 5), key=lambda doc: ranks.get(doc, 10_000))
        if not top_gold or not missed:
            continue
        for doc in missed:
            rank = ranks.get(doc, 10_000)
            rank_counter[str(min(rank, 11)) if rank <= 10 else "outside10"] += 1
        nearest = missed[0]
        nearest_rank = ranks.get(nearest, 10_000)
        relations = []
        for existing in top_gold:
            flag, reason = amendment_relation(titles.get(nearest, ""), titles.get(existing, ""))
            if flag:
                relations.append(reason)
        relation = relations[0] if relations else "none"
        relationship_counter[relation] += 1
        words = ascii_words(train[qid]["question"])
        markers = {
            "and_or": sum(word in {"va", "hoac", "dong", "thoi"} for word in words),
            "citation": sum(word in {"dieu", "khoan", "diem", "nghi", "dinh", "thong", "tu", "luat"} for word in words),
            "numbers": sum(word.isdigit() for word in words),
            "question_tokens": len(words),
        }
        intent = "compound_marker" if markers["and_or"] else "no_compound_marker"
        intent_counter[intent] += 1
        top5 = []
        for rank, candidate in enumerate(order[:10], 1):
            ct = token_set(titles.get(candidate, ""))
            top5.append({
                "rank": rank,
                "doc": candidate,
                "gold": candidate in gold,
                "title": titles.get(candidate, ""),
                "authority": authority(titles.get(candidate, "")),
                "query_title_overlap": len(token_set(train[qid]["question"]) & ct),
            })
        rows.append({
            "qid": qid,
            "fold": fold_of[qid],
            "question": train[qid]["question"],
            "gold_count": len(gold),
            "gold_in_top5": len(top_gold),
            "nearest_missed_rank": nearest_rank,
            "nearest_missed_doc": nearest,
            "nearest_relation_to_retrieved_gold": relation,
            "markers": markers,
            "ranking": top5,
        })
    rows.sort(key=lambda row: (row["nearest_missed_rank"], -row["markers"]["and_or"], row["qid"]))
    near = [row for row in rows if row["nearest_missed_rank"] <= 10]
    summary = {
        "status": "COMPLETE_MULTIGOLD_ERROR_AUDIT",
        "queries_with_partial_gold_and_missing_gold": len(rows),
        "recoverable_within_top10": len(near),
        "nearest_missed_rank_counts": dict(rank_counter),
        "relation_counts": dict(relationship_counter),
        "intent_marker_counts": dict(intent_counter),
        "mean_question_tokens_near": float(np.mean([r["markers"]["question_tokens"] for r in near])) if near else 0.,
        "mean_question_tokens_all": float(np.mean([r["markers"]["question_tokens"] for r in rows])) if rows else 0.,
    }
    write(OUT / "SUMMARY.json", summary)
    write(OUT / "RECOVERABLE_CASES.json", near)
    write(OUT / "ALL_PARTIAL_MULTIGOLD_CASES.json", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nRepresentative recoverable cases:", flush=True)
    for row in near[:20]:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
