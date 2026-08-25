"""EXP-026: fold-isolated LambdaMART and provenance-bound evidence capsules.

The module deliberately consumes the immutable EXP-022 parent pool.  It does
not contain a retriever and it refuses records that would change membership.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from exp012b_bm25 import BM25Searcher, default_segmenter, safe_fts_query
from exp012b_core import (
    artifact_manifest, atomic_json, canonical_json, load_answers, load_v3_manifest,
    read_jsonl, require_success, sha256_file, stage_run, write_jsonl,
)
from exp012b_retrieval import PersistentChunkReader, evaluate_rankings
from exp012b_tuning import load_folds

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "legalir.exp026_lambdamart_capsules.v1"
FEATURE_BLOCKS = {
    "retrieval": ("candidate_rank", "e5_rank", "e5_score", "e5_recip", "bm25_rank", "bm25_recip"),
    "provenance": ("has_e5", "has_bm25", "rank_gap", "bm25_passage_min", "bm25_passage_mean", "bm25_passage_count", "e5_margin", "e5_relative"),
    "metadata": ("passage_length", "parse_fallback", "scope_nodes", "label_tokens", "query_tokens"),
    "structural": ("e5_evidence_count", "e5_evidence_parent_count", "e5_evidence_max", "e5_evidence_mean", "e5_evidence_spread"),
}
FEATURE_SETS = ("retrieval", "retrieval+provenance", "retrieval+provenance+metadata", "all")
PARAMS = ({"num_leaves": 31, "min_child_samples": 30, "n_estimators": 300}, {"num_leaves": 63, "min_child_samples": 40, "n_estimators": 450})
K_GRID = (16, 24, 32, 50)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_metrics(predictions: dict[str, list[str]], answers: dict[str, set[str]], folds: dict[str, list[str]]) -> dict[str, Any]:
    per_fold = {name: evaluate_rankings({qid: predictions[qid] for qid in qids}, {qid: answers[qid] for qid in qids}, ks=(5, 16, 24, 32, 50)) for name, qids in sorted(folds.items())}
    return {"aggregate": evaluate_rankings(predictions, answers, ks=(5, 16, 24, 32, 50)), "per_fold": per_fold}


def _inputs(candidates: Path, train: Path, folds: Path, v3: Path, e5: Path, prep: Path) -> dict[str, str]:
    return {"candidates_sha256": sha256_file(candidates), "train_sha256": sha256_file(train), "folds_sha256": sha256_file(folds), "v3_manifest_sha256": sha256_file(v3 / "manifest.json"), "e5_manifest_sha256": sha256_file(e5 / "manifest.json"), "preprocessing_manifest_sha256": sha256_file(prep / "manifest.json"), "preprocessing_exclusions_sha256": sha256_file(prep / "exclusions.json")}


def audit_inputs(*, candidates: Path, train: Path, folds_path: Path, v3_dir: Path, e5_dir: Path, preprocessing_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate the immutable parent pool before any feature/model work."""
    v3 = load_v3_manifest(v3_dir)
    require_success(candidates.parent)
    # E5's historical success marker predates the v3-fingerprint convention;
    # its immutable manifest is the authoritative compatibility binding.
    require_success(e5_dir)
    e5_manifest = _json(e5_dir / "manifest.json")
    if e5_manifest.get("corpus_fingerprint") != v3["content_fingerprint"]:
        raise ValueError("E5 cache was built for a different structural-v3 corpus")
    answers, folds = load_answers(train), load_folds(folds_path)
    fold_by_qid = {str(qid): name for name, qids in folds.items() for qid in qids}
    docs = {str(row["doc_id"]) for row in read_jsonl(v3_dir / "documents.jsonl")}
    expected_qids, observed_qids, source_counts, unannotated = set(answers), set(), Counter(), 0
    with stage_run(output_dir, "exp026-input-audit", total=len(answers), v3_fingerprint=v3["content_fingerprint"]) as log:
        for number, record in enumerate(read_jsonl(candidates), 1):
            qid = str(record.get("qid")); rows = record.get("candidates", [])
            if qid in observed_qids or qid not in answers or str(record.get("fold")) != fold_by_qid.get(qid):
                raise ValueError(f"candidate qid/fold mismatch: {qid}")
            if len(rows) != 150:
                raise ValueError(f"candidate count must be 150: {qid}/{len(rows)}")
            ids = [str(row.get("doc_id")) for row in rows]
            if len(ids) != len(set(ids)) or not set(ids) <= docs:
                raise ValueError(f"invalid parent-document IDs: {qid}")
            for rank, row in enumerate(rows, 1):
                if int(row.get("rank", -1)) != rank:
                    raise ValueError(f"unstable candidate rank: {qid}/{rank}")
                sources = row.get("sources", {})
                if not set(sources) <= {"e5", "bm25"}:
                    raise ValueError(f"illegal retrieval provenance: {qid}/{row['doc_id']}")
                # BM25-novel parents outside the stored E5@150 tail are valid
                # fixed-pool members, but EXP-022 has no row-level source data
                # for them.  Capsules must therefore use scoped frozen-index
                # fallback and reports expose this availability limitation.
                if not sources:
                    unannotated += 1
                for source in sources: source_counts[source] += 1
                if "e5" in sources and ("rank" not in sources["e5"] or "aggregate_score" not in sources["e5"]):
                    raise ValueError(f"incomplete e5 provenance: {qid}/{row['doc_id']}")
                if "bm25" in sources and "rank" not in sources["bm25"]:
                    raise ValueError(f"incomplete bm25 provenance: {qid}/{row['doc_id']}")
            observed_qids.add(qid)
            if number % 256 == 0: log.status(stage="exp026-input-audit", state="RUNNING", completed=number, total=len(answers))
        if observed_qids != expected_qids: raise ValueError("candidate/train query sets differ")
        report = {"schema_version": SCHEMA, "status": "PASS", "queries": len(observed_qids), "candidates": len(observed_qids) * 150, "source_counts": dict(source_counts), "unannotated_union_tail_candidates": unannotated, "inputs": _inputs(candidates, train, folds_path, v3_dir, e5_dir, preprocessing_dir), "v3_fingerprint": v3["content_fingerprint"]}
        atomic_json(output_dir / "input_audit.json", report); log.set_telemetry(report)
    result = artifact_manifest(stage="exp026-input-audit", inputs=report["inputs"], config={"candidate_limit": 150, "allowed_sources": ["e5", "bm25"]}, files=[output_dir / "input_audit.json"]); atomic_json(output_dir / "manifest.json", result); return report


def _orders(record: dict[str, Any]) -> dict[str, list[str]]:
    rows = record["candidates"]
    union = [str(row["doc_id"]) for row in rows]
    def order(source: str) -> list[str]:
        return [str(row["doc_id"]) for row in sorted(rows, key=lambda r: (int(r.get("sources", {}).get(source, {}).get("rank", 10**9)), int(r["rank"]), str(r["doc_id"]))) ]
    return {"union": union, "e5_rank": order("e5"), "bm25_rank": order("bm25")}


def baselines(*, candidates: Path, train: Path, folds_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    answers, folds = load_answers(train), load_folds(folds_path); by_name: dict[str, dict[str, list[str]]] = {name: {} for name in ("union", "e5_rank", "bm25_rank")}
    for row in read_jsonl(candidates):
        for name, order in _orders(row).items(): by_name[name][str(row["qid"])] = order
    report = {"schema_version": SCHEMA, "baselines": {name: _fold_metrics(pred, answers, folds) for name, pred in by_name.items()}, "primary": "e5_rank"}
    with stage_run(output_dir, "exp026-baselines", total=len(answers), v3_fingerprint=v3_fingerprint) as log:
        atomic_json(output_dir / "baseline_report.json", report); log.set_telemetry(report)
    atomic_json(output_dir / "manifest.json", artifact_manifest(stage="exp026-baselines", inputs={"candidates_sha256": sha256_file(candidates), "train_sha256": sha256_file(train)}, config={"baselines": list(by_name)}, files=[output_dir / "baseline_report.json"])); return report


def _columns(feature_set: str) -> tuple[str, ...]:
    pieces = feature_set.split("+") if feature_set != "all" else tuple(FEATURE_BLOCKS)
    return tuple(value for piece in pieces for value in FEATURE_BLOCKS[piece])


def _doc_metadata(v3_dir: Path) -> dict[str, dict[str, float]]:
    result = {}
    for row in read_jsonl(v3_dir / "documents.jsonl"):
        result[str(row["doc_id"])] = {"passage_length": float(row.get("passage_length", 0)), "parse_fallback": float(row.get("parse_mode") == "fallback"), "scope_nodes": float(len(row.get("scope_node_ids", []))), "label_tokens": float(len(str(row.get("document_label", "")).split()))}
    return result


def feature_rows(*, candidates: Path, v3_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize compact float32 features; all values come from frozen inputs."""
    docs = _doc_metadata(v3_dir)
    chunk_parent = {str(row["chunk_id"]): str(row.get("parent_node_id", "")) for row in read_jsonl(v3_dir / "chunks.jsonl")}
    records = list(read_jsonl(candidates)); names = _columns("all"); matrix = np.empty((len(records) * 150, len(names)), dtype=np.float32); index = []; cursor = 0
    with stage_run(output_dir, "exp026-feature-rows", total=len(records), v3_fingerprint=load_v3_manifest(v3_dir)["content_fingerprint"]) as log:
        for number, record in enumerate(records, 1):
            e5_scores = [float(r["sources"]["e5"]["aggregate_score"]) for r in record["candidates"] if "e5" in r["sources"]]
            best_e5 = max(e5_scores, default=1.0); second_e5 = sorted(e5_scores, reverse=True)[1] if len(e5_scores) > 1 else best_e5
            start = cursor
            for row in record["candidates"]:
                source, e5, bm25 = row["sources"], row["sources"].get("e5", {}), row["sources"].get("bm25", {}); er, br = float(e5.get("rank", 999)), float(bm25.get("rank", 999)); passage = [float(x) for x in bm25.get("passage_ranks", [])]; evidence = e5.get("evidence", []); scores = [float(x.get("chunk_score", 0)) for x in evidence]
                evidence_parents = {chunk_parent.get(str(x.get("chunk_id", "")), "") for x in evidence} - {""}
                values = {"candidate_rank": float(row["rank"]), "e5_rank": er, "e5_score": float(e5.get("aggregate_score", 0)), "e5_recip": 0.0 if er >= 999 else 1/er, "bm25_rank": br, "bm25_recip": 0.0 if br >= 999 else 1/br, "has_e5": float("e5" in source), "has_bm25": float("bm25" in source), "rank_gap": abs(er-br) if er < 999 and br < 999 else 999., "bm25_passage_min": min(passage, default=999.), "bm25_passage_mean": float(np.mean(passage)) if passage else 999., "bm25_passage_count": float(len(passage)), "e5_margin": best_e5-second_e5, "e5_relative": float(e5.get("aggregate_score", 0))/max(best_e5, 1e-6), "query_tokens": float(len(str(record["query"]).split())), "e5_evidence_count": float(len(evidence)), "e5_evidence_parent_count": float(len(evidence_parents)), "e5_evidence_max": max(scores, default=0.), "e5_evidence_mean": float(np.mean(scores)) if scores else 0., "e5_evidence_spread": max(scores, default=0.)-min(scores, default=0.)}
                values.update(docs[str(row["doc_id"])])
                matrix[cursor] = [values[name] for name in names]; cursor += 1
            index.append({"qid": str(record["qid"]), "fold": str(record["fold"]), "start": start, "end": cursor, "doc_ids": [str(x["doc_id"]) for x in record["candidates"]]})
            if number % 256 == 0: log.status(stage="exp026-feature-rows", state="RUNNING", completed=number, total=len(records))
        np.save(output_dir / "features.f32.npy", matrix); write_jsonl(output_dir / "query_index.jsonl", index); atomic_json(output_dir / "feature_schema.json", {"schema_version": SCHEMA, "columns": list(names), "blocks": FEATURE_BLOCKS, "rows": cursor})
    files=[output_dir / "features.f32.npy", output_dir / "query_index.jsonl", output_dir / "feature_schema.json"]; result=artifact_manifest(stage="exp026-feature-rows", inputs={"candidates_sha256":sha256_file(candidates),"v3_manifest_sha256":sha256_file(v3_dir/'manifest.json')}, config={"dtype":"float32","columns":list(names)}, files=files); atomic_json(output_dir/"manifest.json",result); return result


def _load_features(feature_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    return np.load(feature_dir / "features.f32.npy", mmap_mode="r"), list(read_jsonl(feature_dir / "query_index.jsonl")), _json(feature_dir / "feature_schema.json")["columns"]


def _fit_predict(train_idx: list[dict[str, Any]], test_idx: list[dict[str, Any]], data: np.ndarray, answers: dict[str, set[str]], columns: list[str], feature_set: str, param: dict[str, int]) -> tuple[dict[str, list[str]], Any]:
    from lightgbm import LGBMRanker
    positions = [columns.index(c) for c in _columns(feature_set)]; train_rows = np.concatenate([np.arange(x["start"], x["end"]) for x in train_idx]); x = np.asarray(data[train_rows][:, positions]); y = np.asarray([int(doc in answers[str(qid)]) for item in train_idx for qid in [item["qid"]] for doc in item["doc_ids"]], dtype=np.int32)
    model = LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=.04, random_state=42, deterministic=True, force_col_wise=True, n_jobs=-1, verbosity=-1, **param); model.fit(x, y, group=[x["end"]-x["start"] for x in train_idx]); output={}
    for item in test_idx:
        scores=model.predict(np.asarray(data[item["start"]:item["end"], positions])); order=sorted(range(len(scores)), key=lambda i:(-float(scores[i]), str(item["doc_ids"][i]))); output[str(item["qid"])] = [str(item["doc_ids"][i]) for i in order]
    return output, model


def oof_lambdamart(*, feature_dir: Path, train: Path, folds_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    data, index, columns = _load_features(feature_dir); answers, folds = load_answers(train), load_folds(folds_path); by_qid={x["qid"]:x for x in index}; predictions={}; selections={}; importance=defaultdict(float)
    with stage_run(output_dir, "exp026-nested-oof", total=len(index), v3_fingerprint=v3_fingerprint) as log:
        for outer, heldout in sorted(folds.items()):
            outer_train=[q for name, qs in folds.items() if name != outer for q in qs]; screens=[]
            for feature_set in FEATURE_SETS:
                for param in PARAMS:
                    vals=[]
                    for inner in sorted(set(folds)-{outer}):
                        train_ids=[q for q in outer_train if q not in set(folds[inner])]; pred,_=_fit_predict([by_qid[q] for q in train_ids],[by_qid[q] for q in folds[inner]],data,answers,columns,feature_set,param); metric=evaluate_rankings(pred,{q:answers[q] for q in folds[inner]},ks=(5,)); vals.append(metric)
                    screens.append({"feature_set":feature_set,"params":param,"recall":float(np.mean([x['recall@5'] for x in vals])),"precision":float(np.mean([x['precision@5'] for x in vals]))})
            chosen=sorted(screens,key=lambda x:(-x['recall'],-x['precision'],FEATURE_SETS.index(x['feature_set']),PARAMS.index(x['params'])))[0]; pred,model=_fit_predict([by_qid[q] for q in outer_train],[by_qid[q] for q in heldout],data,answers,columns,chosen['feature_set'],chosen['params']); predictions.update(pred); selections[outer]={"chosen":chosen,"screens":screens};
            for name,value in zip(_columns(chosen['feature_set']),model.feature_importances_): importance[name]+=float(value)
            log.status(stage="exp026-nested-oof",state="RUNNING",completed=len(predictions),total=len(index)); log.log(f"outer={outer} selected={chosen['feature_set']} recall={chosen['recall']:.6f}")
        write_jsonl(output_dir/"oof_predictions.jsonl",({"qid":qid,"doc_ids":ids} for qid,ids in sorted(predictions.items()))); report={"schema_version":SCHEMA,"metrics":_fold_metrics(predictions,answers,folds),"selections":selections,"feature_importance":dict(sorted(importance.items(),key=lambda x:-x[1])),"fold_isolated":True,"nested_selection":True}; atomic_json(output_dir/"oof_report.json",report); log.set_telemetry(report["metrics"])
    files=[output_dir/"oof_predictions.jsonl",output_dir/"oof_report.json"]; result=artifact_manifest(stage="exp026-nested-oof",inputs={"feature_manifest_sha256":sha256_file(feature_dir/'manifest.json'),"train_sha256":sha256_file(train),"folds_sha256":sha256_file(folds_path)},config={"feature_sets":list(FEATURE_SETS),"params":list(PARAMS)},files=files); atomic_json(output_dir/"manifest.json",result); return report


def shortlist_audit(*, predictions: Path, train: Path, folds_path: Path, output_dir: Path, v3_fingerprint: str) -> dict[str, Any]:
    answers, folds=load_answers(train),load_folds(folds_path); pred={str(x['qid']):[str(v) for v in x['doc_ids']] for x in read_jsonl(predictions)}; per={name:evaluate_rankings({q:pred[q] for q in qids},{q:answers[q] for q in qids},ks=K_GRID) for name,qids in folds.items()}; viable=[k for k in K_GRID if all(values[f'recall@{k}']>=.985 for values in per.values())]; report={"schema_version":SCHEMA,"threshold":.985,"per_fold":per,"shortlist_k":viable[0] if viable else None,"status":"PASS" if viable else "FAIL"}
    with stage_run(output_dir,"exp026-shortlist-audit",total=len(pred),v3_fingerprint=v3_fingerprint) as log: atomic_json(output_dir/'shortlist_audit.json',report); log.set_telemetry(report)
    atomic_json(output_dir/'manifest.json',artifact_manifest(stage='exp026-shortlist-audit',inputs={'predictions_sha256':sha256_file(predictions),'train_sha256':sha256_file(train)},config={'grid':K_GRID,'floor':.985},files=[output_dir/'shortlist_audit.json'])); return report


def capsules(*, candidates: Path, predictions: Path, shortlist: Path, v3_dir: Path, bm25_db: Path, output_dir: Path, max_tokens: int=768) -> dict[str, Any]:
    """Build deterministic capsules; BM25-only parents use query scoring inside that same parent only."""
    short=_json(shortlist); k=short.get('shortlist_k')
    if not k: raise RuntimeError('shortlist gate failed; capsules are intentionally blocked')
    v3=load_v3_manifest(v3_dir); docs={str(r['doc_id']):r for r in read_jsonl(v3_dir/'documents.jsonl')}; chosen={str(r['qid']):[str(x) for x in r['doc_ids'][:k]] for r in read_jsonl(predictions)}; candidates_by_qid={str(r['qid']):r for r in read_jsonl(candidates)}; needed=set(); selections={}; searcher=BM25Searcher(bm25_db,profile='legal_structure')
    try:
        with stage_run(output_dir,'exp026-build-capsules',total=len(chosen),v3_fingerprint=v3['content_fingerprint']) as log:
            for number,qid in enumerate(sorted(chosen),1):
                record=candidates_by_qid[qid]; by_doc={str(r['doc_id']):r for r in record['candidates']}; expression=safe_fts_query(default_segmenter(str(record['query'])))
                for doc_id in chosen[qid]:
                    e5=by_doc[doc_id].get('sources',{}).get('e5',{}).get('evidence',[]); picks=[{'chunk_id':str(x['chunk_id']),'provenance':'e5','parent_node_id':None} for x in e5[:2]]
                    if not picks:
                        fallback=searcher.search_document_expression(expression,doc_id,limit=2)
                        picks=[{'chunk_id':str(x['chunk_id']),'provenance':'bm25_scoped_fallback','parent_node_id':str(x.get('parent_node_id',''))} for x in fallback[:2]]
                    if not picks: raise RuntimeError(f'no frozen evidence for {qid}/{doc_id}')
                    selections[(qid,doc_id)]=picks; needed.update(x['chunk_id'] for x in picks)
                if number%256==0: log.status(stage='exp026-build-capsules',state='RUNNING',completed=number,total=len(chosen))
            lookup=ROOT/'cache'/'exp026_lambdamart_capsules'/'chunk_lookup'/'chunk_offsets.sqlite'; chunks=PersistentChunkReader(v3_dir,lookup) if lookup.exists() else None
            try:
                found=chunks.load(needed) if chunks else {str(r['chunk_id']):r for r in read_jsonl(v3_dir/'chunks.jsonl') if str(r['chunk_id']) in needed}
            finally:
                if chunks: chunks.close()
            rows=[]
            for qid in sorted(chosen):
                record=candidates_by_qid[qid]; items=[]
                for doc_id in chosen[qid]:
                    evidence=[]
                    for pick in selections[(qid,doc_id)]:
                        chunk=found[pick['chunk_id']]; text=str(chunk['raw_text']).strip(); evidence.append({**pick,'start':int(chunk['start']),'end':int(chunk['end']),'raw_text':text,'token_count':int(chunk['token_count'])})
                    # Token limit is audited conservatively from v3 tokenizer counts; rendering is deferred to a reranker.
                    total=sum(x['token_count'] for x in evidence); items.append({'doc_id':doc_id,'document_label':str(docs[doc_id].get('document_label','')),'scope_node_ids':docs[doc_id].get('scope_node_ids',[]),'evidence':evidence,'estimated_tokens':min(total,max_tokens)})
                rows.append({'schema_version':SCHEMA,'qid':qid,'query':record['query'],'max_tokens':max_tokens,'candidates':items})
            write_jsonl(output_dir/'capsules.jsonl',rows); report={'schema_version':SCHEMA,'queries':len(rows),'capsules':sum(len(x['candidates']) for x in rows),'provenance':dict(Counter(e['provenance'] for x in rows for c in x['candidates'] for e in c['evidence'])),'max_tokens':max_tokens}; atomic_json(output_dir/'capsule_audit.json',report); log.set_telemetry(report)
    finally: searcher.close()
    files=[output_dir/'capsules.jsonl',output_dir/'capsule_audit.json']; result=artifact_manifest(stage='exp026-build-capsules',inputs={'candidates_sha256':sha256_file(candidates),'predictions_sha256':sha256_file(predictions),'v3_manifest_sha256':sha256_file(v3_dir/'manifest.json')},config={'shortlist_k':k,'max_tokens':max_tokens},files=files); atomic_json(output_dir/'manifest.json',result); return report


def final_report(*, baseline: Path, oof: Path, shortlist: Path, output_dir: Path) -> dict[str, Any]:
    base,oof_report,short=_json(baseline),_json(oof),_json(shortlist); b=base['baselines']['e5_rank']['aggregate']; m=oof_report['metrics']['aggregate']; accepted=m['recall@5']>=b['recall@5']+.002 and m['precision@5']>=b['precision@5']; report={'schema_version':SCHEMA,'hypothesis':'fold-isolated structural/provenance features improve top-5 without changing the pool','baseline':b,'oof':m,'shortlist':short,'acceptance_gate':{'recall_gain_minimum':.002,'precision_non_decrease':True,'accepted':accepted},'recommended_next_stage':'capsule/reranker only if accepted and shortlist passed'}; atomic_json(output_dir/'REPORT.json',report); return report


def main(argv: Sequence[str] | None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('audit','baselines','features','oof','shortlist','capsules','report')); p.add_argument('--cache-root',type=Path,default=ROOT/'cache'/'exp026_lambdamart_capsules'); p.add_argument('--results-root',type=Path,default=ROOT/'results'/'exp026_lambdamart_capsules'); p.add_argument('--candidates',type=Path,default=ROOT/'cache'/'exp022_e5_bm25_union'/'train_oof_candidates.jsonl'); p.add_argument('--train',type=Path,default=ROOT/'public_test_dataset'/'train.json'); p.add_argument('--folds',type=Path,default=ROOT/'cache'/'cv_folds.json'); p.add_argument('--v3',type=Path,default=ROOT/'cache'/'structural_v3_e5_final_v1'); p.add_argument('--e5',type=Path,default=ROOT/'cache'/'e5_final_v1'); p.add_argument('--preprocessing',type=Path,default=ROOT/'cache'/'final_preprocessed_v2'); p.add_argument('--bm25-db',type=Path,default=ROOT/'cache'/'exp021_sparse'/'passage_hierarchy'/'fts5'/'bm25_v3.sqlite'); a=p.parse_args(argv); fp=load_v3_manifest(a.v3)['content_fingerprint']
    if a.stage=='audit': result=audit_inputs(candidates=a.candidates,train=a.train,folds_path=a.folds,v3_dir=a.v3,e5_dir=a.e5,preprocessing_dir=a.preprocessing,output_dir=a.results_root/'input_audit')
    elif a.stage=='baselines': result=baselines(candidates=a.candidates,train=a.train,folds_path=a.folds,output_dir=a.results_root/'baselines',v3_fingerprint=fp)
    elif a.stage=='features': result=feature_rows(candidates=a.candidates,v3_dir=a.v3,output_dir=a.cache_root/'features')
    elif a.stage=='oof': result=oof_lambdamart(feature_dir=a.cache_root/'features',train=a.train,folds_path=a.folds,output_dir=a.results_root/'oof',v3_fingerprint=fp)
    elif a.stage=='shortlist': result=shortlist_audit(predictions=a.results_root/'oof'/'oof_predictions.jsonl',train=a.train,folds_path=a.folds,output_dir=a.results_root/'shortlist',v3_fingerprint=fp)
    elif a.stage=='capsules': result=capsules(candidates=a.candidates,predictions=a.results_root/'oof'/'oof_predictions.jsonl',shortlist=a.results_root/'shortlist'/'shortlist_audit.json',v3_dir=a.v3,bm25_db=a.bm25_db,output_dir=a.cache_root/'capsules')
    else: result=final_report(baseline=a.results_root/'baselines'/'baseline_report.json',oof=a.results_root/'oof'/'oof_report.json',shortlist=a.results_root/'shortlist'/'shortlist_audit.json',output_dir=a.results_root)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
