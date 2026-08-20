import json
import lightgbm as lgb
from pathlib import Path
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
records = list(read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')))

def _source(candidate, name):
    v = candidate.get("sources", {}).get(name, {})
    return float(v.get("rank", 999)), float(v.get("score", 0.0)), v.get("best_chunk_id"), v.get("neighbor_qids", [])

def build_advanced_features(records):
    type_codes = {name: i for i, name in enumerate(sorted({str(r.get("document_type", "unknown")) for rec in records for r in rec["candidates"]}))}
    out = {}
    
    for rec in records:
        qid = str(rec["qid"])
        query_words = len(str(rec["query"]).split())
        cands = rec["candidates"]
        
        bge_ranked = sorted([_source(c, "bge")[1] for c in cands if _source(c, "bge")[0] < 999], reverse=True)
        bm25_ranked = sorted([_source(c, "bm25")[1] for c in cands if _source(c, "bm25")[0] < 999], reverse=True)
        
        best_bge, second_bge = (bge_ranked + [0.0, 0.0])[:2]
        best_bm25, second_bm25 = (bm25_ranked + [0.0, 0.0])[:2]
        
        rows = []
        for c in cands:
            bge_rank, bge_score, bge_chunk, _ = _source(c, "bge")
            bm25_rank, bm25_score, bm25_chunk, _ = _source(c, "bm25")
            memory_rank, memory_score, _, memory_neighbors = _source(c, "memory")
            
            doc_len = float(c.get("document_length", 0))
            chunk_cnt = float(c.get("chunk_count", 0))
            art_cnt = float(c.get("article_count", 0))
            
            # Reciprocal ranks
            rr_bge = 0.0 if bge_rank >= 999 else 1.0 / bge_rank
            rr_bm25 = 0.0 if bm25_rank >= 999 else 1.0 / bm25_rank
            rr_mem = 0.0 if memory_rank >= 999 else 1.0 / memory_rank
            
            # RRF combined score
            rrf_score = (1.0 / (60 + bge_rank)) + (1.0 / (60 + bm25_rank)) + (1.0 / (60 + memory_rank) if memory_rank < 999 else 0.0)
            
            # Exact chunk agreement
            same_best_chunk = float(bge_chunk is not None and bm25_chunk is not None and bge_chunk == bm25_chunk)
            
            row = {
                "qid": qid,
                "doc_id": str(c["doc_id"]),
                "bge_rank": bge_rank,
                "bge_score": bge_score,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_score,
                "memory_rank": memory_rank,
                "memory_score": memory_score,
                "memory_neighbors_count": float(len(memory_neighbors)),
                
                "bge_reciprocal": rr_bge,
                "bm25_reciprocal": rr_bm25,
                "memory_reciprocal": rr_mem,
                "rrf_score": rrf_score,
                
                "channel_count": float(c["channel_count"]),
                "best_source_rank": float(c["best_source_rank"]),
                "same_best_chunk": same_best_chunk,
                
                "bge_bm25_overlap": float(bge_rank < 999 and bm25_rank < 999),
                "bge_memory_overlap": float(bge_rank < 999 and memory_rank < 999),
                "bm25_memory_overlap": float(bm25_rank < 999 and memory_rank < 999),
                "all_channel_agreement": float(bge_rank < 999 and bm25_rank < 999 and memory_rank < 999),
                
                "candidate_rank": float(c["rank"]),
                "query_tokens": float(query_words),
                "document_length": doc_len,
                "document_chunks": chunk_cnt,
                "article_count": art_cnt,
                "chunk_density": chunk_cnt / max(doc_len / 1000.0, 1.0),
                "avg_article_length": doc_len / max(art_cnt, 1.0),
                
                "fallback": float(str(c.get("parse_mode")) == "fallback"),
                "hierarchy_available": float(c.get("hierarchy_available", 0)),
                
                "bge_margin": float(best_bge - second_bge),
                "bm25_margin": float(best_bm25 - second_bm25),
                "bge_diff_from_best": float(best_bge - bge_score),
                "bm25_diff_from_best": float(best_bm25 - bm25_score),
                "bge_normalized": float(bge_score / max(best_bge, 1e-5)),
                "bm25_normalized": float(bm25_score / max(best_bm25, 1e-5)),
                
                "bge_x_bm25_score": float(bge_score * bm25_score),
                "rank_disagreement": float(abs(bge_rank - bm25_rank)) if bge_rank < 999 and bm25_rank < 999 else 999.0,
                "min_rank": float(min(bge_rank, bm25_rank, memory_rank)),
                "document_type_code": float(type_codes.get(str(c.get("document_type", "unknown")), -1))
            }
            rows.append(row)
        out[qid] = rows
    return out

print("Building advanced features...")
by_qid = build_advanced_features(records)
feature_cols = [col for col in list(by_qid.values())[0][0].keys() if col not in ("qid", "doc_id")]
print(f"Total features: {len(feature_cols)} -> {feature_cols}")

# Train 5-Fold OOF
predictions = {}
fold_metrics = {}

for fold_name, heldout_values in sorted(folds.items()):
    heldout = [str(v) for v in heldout_values]
    training = sorted(set(by_qid) - set(heldout))
    
    train_rows = [row for qid in training for row in by_qid[qid]]
    X_train = np.asarray([[row[col] for col in feature_cols] for row in train_rows], dtype=np.float32)
    y_train = np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in train_rows], dtype=np.int32)
    group_train = [len(by_qid[qid]) for qid in training]
    
    val_rows = [row for qid in heldout for row in by_qid[qid]]
    X_val = np.asarray([[row[col] for col in feature_cols] for row in val_rows], dtype=np.float32)
    y_val = np.asarray([int(row["doc_id"] in answers[row["qid"]]) for row in val_rows], dtype=np.int32)
    group_val = [len(by_qid[qid]) for qid in heldout]
    
    model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.03,
        n_estimators=450,
        num_leaves=63,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
        verbosity=-1
    )
    model.fit(
        X_train, y_train, group=group_train,
        eval_set=[(X_val, y_val)], eval_group=[group_val],
        eval_at=[5], eval_metric="ndcg",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    offset = 0
    fold_preds = {}
    val_preds = model.predict(X_val)
    for qid in heldout:
        cnt = len(by_qid[qid])
        doc_ids = [row["doc_id"] for row in by_qid[qid]]
        scores = dict(zip(doc_ids, val_preds[offset:offset+cnt]))
        fold_preds[qid] = rank_ids(scores)
        predictions[qid] = [{"doc_id": doc_id, "rank": rank, "lambda_score": scores[doc_id], "fold": fold_name}
                            for rank, doc_id in enumerate(rank_ids(scores), 1)]
        offset += cnt
        
    gold_heldout = {qid: answers[qid] for qid in heldout}
    fold_m = evaluate_rankings(fold_preds, gold_heldout, ks=(5, 10, 24))
    fold_metrics[fold_name] = fold_m
    print(f"[{fold_name}] Recall@5: {fold_m['recall@5']:.4f} | Prec@5: {fold_m['precision@5']:.4f} | Recall@24: {fold_m['recall@24']:.4f}")

all_ranks = {qid: [row["doc_id"] for row in values] for qid, values in predictions.items()}
overall_m = evaluate_rankings(all_ranks, answers, ks=(5, 10, 20, 24, 32))
print("\n=======================================================")
print(f"5-FOLD OOF OVERALL METRICS:")
print(f"Recall@5:  {overall_m['recall@5']:.5f}")
print(f"Prec@5:    {overall_m['precision@5']:.5f}")
print(f"Recall@10: {overall_m['recall@10']:.5f}")
print(f"Recall@24: {overall_m['recall@24']:.5f}")
print(f"Recall@32: {overall_m['recall@32']:.5f}")
print("=======================================================")
