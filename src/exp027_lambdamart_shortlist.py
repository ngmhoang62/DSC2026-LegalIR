"""EXP-027: provenance repair sidecar and retained-gold shortlist evaluation.

EXP-022 is immutable.  This stage joins an auditable BM25 sidecar onto its
fixed parent pool without reordering or rewriting any candidate record.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from exp012b_core import artifact_manifest, atomic_json, load_answers, load_v3_manifest, read_jsonl, sha256_file, stage_run, write_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp021_sparse_depth_tune import _cascade, _iter_evidence, _rankings
from exp026_lambdamart_capsules import FEATURE_BLOCKS, FEATURE_SETS, PARAMS, _columns, _doc_metadata, _fit_predict, capsules as build_capsules

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "legalir.exp027_lambdamart_shortlist.v1"
K_GRID = (16, 24, 32, 50, 64, 80, 100)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def retained_answers(train: Path, exclusions: Path, impact: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    all_answers = load_answers(train)
    excluded = {str(row["doc_id"]) for row in _json(exclusions)}
    retained = {qid: gold - excluded for qid, gold in all_answers.items()}
    removed = sum(len(all_answers[qid] - retained[qid]) for qid in all_answers)
    impacted = list(read_jsonl(impact))
    expected = sum(len(row.get("intentionally_excluded_gold_ids", [])) for row in impacted)
    if removed != expected:
        raise ValueError(f"retained-gold impact mismatch: {removed}/{expected}")
    return retained, {"excluded_document_ids": len(excluded), "removed_gold_occurrences": removed, "affected_queries": len(impacted)}


def build_sidecar(*, candidates: Path, evidence_dir: Path, tuning_report: Path, folds_path: Path, output_dir: Path) -> dict[str, Any]:
    """Recover full cascade BM25 ranks/evidence for the fixed EXP-022 parents."""
    folds = load_folds(folds_path); fold_for = {qid: name for name, qids in folds.items() for qid in qids}
    selected = _json(tuning_report)["selected_by_candidate_budget"]["50"]
    fixed = {str(row["qid"]): row for row in read_jsonl(candidates)}
    rows: dict[str, dict[str, Any]] = {}
    recovered = 0
    with stage_run(output_dir, "exp027-build-provenance-sidecar", total=len(fixed)) as log:
        for position, raw in enumerate(_iter_evidence(evidence_dir / "shards"), 1):
            qid = str(raw["qid"])
            if qid not in fixed:
                raise ValueError(f"raw sparse query absent from fixed pool: {qid}")
            config = selected[fold_for[qid]]
            first, rrf = _rankings(raw["evidence"], int(config["depth"]), int(config["parent_rrf_k"]))
            sparse = _cascade(first, rrf, int(config["fusion_rrf_k"]), int(config["head_cutoff"]))
            rank_by_doc = {doc_id: rank for rank, doc_id in enumerate(sparse, 1)}
            passage_by_doc = {str(doc_id): [int(rank) for rank in ranks] for doc_id, ranks in raw["evidence"]}
            values = []
            for candidate in fixed[qid]["candidates"]:
                doc_id, sources = str(candidate["doc_id"]), candidate["sources"]
                bm25 = None
                if doc_id in rank_by_doc:
                    bm25 = {"rank": rank_by_doc[doc_id], "passage_ranks": passage_by_doc.get(doc_id, [])}
                    old = sources.get("bm25")
                    if old is not None and (int(old["rank"]) != bm25["rank"] or list(map(int, old.get("passage_ranks", []))) != bm25["passage_ranks"]):
                        raise ValueError(f"existing BM25 provenance mismatch: {qid}/{doc_id}")
                if not sources and bm25 is None:
                    raise ValueError(f"unannotated candidate is not a sparse BM25 parent: {qid}/{doc_id}")
                if not sources:
                    recovered += 1
                values.append({"doc_id": doc_id, "bm25": bm25})
            rows[qid] = {"schema_version": SCHEMA, "qid": qid, "candidate_ids": [str(c["doc_id"]) for c in fixed[qid]["candidates"]], "bm25": values}
            if position % 256 == 0: log.status(stage="exp027-build-provenance-sidecar", state="RUNNING", completed=position, total=len(fixed))
        if set(rows) != set(fixed): raise ValueError("raw sparse evidence query set mismatch")
        write_jsonl(output_dir / "provenance_sidecar.jsonl", (rows[qid] for qid in sorted(rows)))
        report = {"schema_version": SCHEMA, "status": "PASS", "queries": len(rows), "recovered_unannotated_candidates": recovered, "candidate_membership_sha256": sha256_file(candidates), "sparse_evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"), "sparse_tuning_report_sha256": sha256_file(tuning_report)}
        atomic_json(output_dir / "sidecar_audit.json", report); log.set_telemetry(report)
    files = [output_dir / "provenance_sidecar.jsonl", output_dir / "sidecar_audit.json"]
    manifest = artifact_manifest(stage="exp027-build-provenance-sidecar", inputs={"candidates_sha256": sha256_file(candidates), "evidence_manifest_sha256": sha256_file(evidence_dir / "manifest.json"), "tuning_report_sha256": sha256_file(tuning_report)}, config={"candidate_limit": 150, "sparse_budget": 50}, files=files)
    atomic_json(output_dir / "manifest.json", manifest); return report


def _merged_records(candidates: Path, sidecar: Path) -> list[dict[str, Any]]:
    side = {str(row["qid"]): row for row in read_jsonl(sidecar)}; merged = []
    for record in read_jsonl(candidates):
        qid = str(record["qid"]); supplemental = side.get(qid)
        if supplemental is None or supplemental["candidate_ids"] != [str(c["doc_id"]) for c in record["candidates"]]:
            raise ValueError(f"sidecar membership/order mismatch: {qid}")
        bm25_by_doc = {str(row["doc_id"]): row.get("bm25") for row in supplemental["bm25"]}
        cloned = {key: value for key, value in record.items() if key != "candidates"}; cloned["candidates"] = []
        for candidate in record["candidates"]:
            row = {**candidate, "sources": dict(candidate["sources"])}; doc_id = str(row["doc_id"]); bm25 = bm25_by_doc[doc_id]
            if bm25 is not None: row["sources"]["bm25"] = bm25
            if not row["sources"]: raise ValueError(f"still no provenance: {qid}/{doc_id}")
            cloned["candidates"].append(row)
        merged.append(cloned)
    return merged


def feature_rows(*, candidates: Path, sidecar: Path, v3_dir: Path, output_dir: Path) -> dict[str, Any]:
    docs = _doc_metadata(v3_dir)
    chunk_parent = {str(row["chunk_id"]): str(row.get("parent_node_id", "")) for row in read_jsonl(v3_dir / "chunks.jsonl")}
    records = _merged_records(candidates, sidecar); names = _columns("all"); matrix = np.empty((len(records) * 150, len(names)), dtype=np.float32); index=[]; cursor=0
    with stage_run(output_dir, "exp027-feature-rows", total=len(records), v3_fingerprint=load_v3_manifest(v3_dir)["content_fingerprint"]) as log:
        for number, record in enumerate(records, 1):
            e5_scores = [float(c["sources"]["e5"]["aggregate_score"]) for c in record["candidates"] if "e5" in c["sources"]]; best=max(e5_scores, default=1.0); second=sorted(e5_scores, reverse=True)[1] if len(e5_scores)>1 else best; start=cursor
            for candidate in record["candidates"]:
                source=candidate["sources"]; e5=source.get("e5", {}); bm25=source.get("bm25", {}); er=float(e5.get("rank",999)); br=float(bm25.get("rank",999)); passage=[float(x) for x in bm25.get("passage_ranks",[])]; evidence=e5.get("evidence",[]); scores=[float(x.get("chunk_score",0)) for x in evidence]; parents={chunk_parent.get(str(x.get("chunk_id","")),"") for x in evidence}-{ "" }
                value={"candidate_rank":float(candidate["rank"]),"e5_rank":er,"e5_score":float(e5.get("aggregate_score",0)),"e5_recip":0 if er>=999 else 1/er,"bm25_rank":br,"bm25_recip":0 if br>=999 else 1/br,"has_e5":float("e5" in source),"has_bm25":float("bm25" in source),"rank_gap":abs(er-br) if er<999 and br<999 else 999.,"bm25_passage_min":min(passage,default=999.),"bm25_passage_mean":float(np.mean(passage)) if passage else 999.,"bm25_passage_count":float(len(passage)),"e5_margin":best-second,"e5_relative":float(e5.get("aggregate_score",0))/max(best,1e-6),"query_tokens":float(len(str(record["query"]).split())),"e5_evidence_count":float(len(evidence)),"e5_evidence_parent_count":float(len(parents)),"e5_evidence_max":max(scores,default=0.),"e5_evidence_mean":float(np.mean(scores)) if scores else 0.,"e5_evidence_spread":max(scores,default=0.)-min(scores,default=0.)}; value.update(docs[str(candidate["doc_id"])]); matrix[cursor]=[value[x] for x in names]; cursor+=1
            index.append({"qid":str(record["qid"]),"fold":str(record["fold"]),"start":start,"end":cursor,"doc_ids":[str(c["doc_id"]) for c in record["candidates"]]})
            if number%256==0: log.status(stage="exp027-feature-rows",state="RUNNING",completed=number,total=len(records))
        np.save(output_dir/"features.f32.npy",matrix); write_jsonl(output_dir/"query_index.jsonl",index); atomic_json(output_dir/"feature_schema.json",{"schema_version":SCHEMA,"columns":list(names),"blocks":FEATURE_BLOCKS,"rows":cursor})
    files=[output_dir/"features.f32.npy",output_dir/"query_index.jsonl",output_dir/"feature_schema.json"]; manifest=artifact_manifest(stage="exp027-feature-rows",inputs={"candidates_sha256":sha256_file(candidates),"sidecar_sha256":sha256_file(sidecar),"v3_manifest_sha256":sha256_file(v3_dir/"manifest.json")},config={"dtype":"float32","columns":list(names)},files=files); atomic_json(output_dir/"manifest.json",manifest); return manifest


def _load_features(path: Path) -> tuple[np.ndarray,list[dict[str,Any]],list[str]]:
    return np.load(path/"features.f32.npy",mmap_mode="r"),list(read_jsonl(path/"query_index.jsonl")),_json(path/"feature_schema.json")["columns"]


def _metrics(pred: dict[str,list[str]], answers: dict[str,set[str]], folds: dict[str,list[str]]) -> dict[str,Any]:
    return {"aggregate":evaluate_rankings(pred,answers,ks=(5,*K_GRID)),"per_fold":{name:evaluate_rankings({q:pred[q] for q in qids},{q:answers[q] for q in qids},ks=(5,*K_GRID)) for name,qids in sorted(folds.items())}}


def baselines(*, candidates: Path, sidecar: Path, answers: dict[str,set[str]], folds: dict[str,list[str]], output_dir: Path, v3_fingerprint: str) -> dict[str,Any]:
    records=_merged_records(candidates,sidecar); predictions={"union":{},"e5_rank":{},"bm25_rank":{}}
    for record in records:
        rows=record["candidates"]; qid=str(record["qid"]); predictions["union"][qid]=[str(x["doc_id"]) for x in rows]
        for source,name in (("e5","e5_rank"),("bm25","bm25_rank")):
            predictions[name][qid]=[str(x["doc_id"]) for x in sorted(rows,key=lambda x:(int(x["sources"].get(source,{}).get("rank",10**9)),int(x["rank"]),str(x["doc_id"])))]
    report={"schema_version":SCHEMA,"denominator":"retained_gold","baselines":{name:_metrics(value,answers,folds) for name,value in predictions.items()},"primary":"e5_rank"}
    with stage_run(output_dir,"exp027-baselines",total=len(records),v3_fingerprint=v3_fingerprint) as log: atomic_json(output_dir/"baseline_report.json",report); log.set_telemetry(report)
    atomic_json(output_dir/"manifest.json",artifact_manifest(stage="exp027-baselines",inputs={"candidates_sha256":sha256_file(candidates),"sidecar_sha256":sha256_file(sidecar)},config={"denominator":"retained_gold"},files=[output_dir/"baseline_report.json"])); return report


def oof(*, feature_dir: Path, answers: dict[str,set[str]], folds: dict[str,list[str]], output_dir: Path, v3_fingerprint: str) -> dict[str,Any]:
    data,index,columns=_load_features(feature_dir); by_qid={x["qid"]:x for x in index}; predictions={}; selections={}; importance=defaultdict(float)
    with stage_run(output_dir,"exp027-nested-oof",total=len(index),v3_fingerprint=v3_fingerprint) as log:
        for outer,heldout in sorted(folds.items()):
            outer_train=[q for name,values in folds.items() if name!=outer for q in values]; screens=[]
            for feature_set in FEATURE_SETS:
                for param in PARAMS:
                    screen=[]
                    for inner in sorted(set(folds)-{outer}):
                        train_ids=[q for q in outer_train if q not in set(folds[inner])]; pred,_=_fit_predict([by_qid[q] for q in train_ids],[by_qid[q] for q in folds[inner]],data,answers,columns,feature_set,param); screen.append(evaluate_rankings(pred,{q:answers[q] for q in folds[inner]},ks=(5,)))
                    screens.append({"feature_set":feature_set,"params":param,"recall":float(np.mean([x["recall@5"] for x in screen])),"precision":float(np.mean([x["precision@5"] for x in screen]))})
            chosen=sorted(screens,key=lambda x:(-x["recall"],-x["precision"],FEATURE_SETS.index(x["feature_set"]),PARAMS.index(x["params"])))[0]; pred,model=_fit_predict([by_qid[q] for q in outer_train],[by_qid[q] for q in heldout],data,answers,columns,chosen["feature_set"],chosen["params"]); predictions.update(pred); selections[outer]={"chosen":chosen,"screens":screens}
            for name,value in zip(_columns(chosen["feature_set"]),model.feature_importances_): importance[name]+=float(value)
            log.status(stage="exp027-nested-oof",state="RUNNING",completed=len(predictions),total=len(index)); log.log(f"outer={outer} selected={chosen['feature_set']} recall={chosen['recall']:.6f}")
        write_jsonl(output_dir/"oof_predictions.jsonl",({"qid":qid,"doc_ids":values} for qid,values in sorted(predictions.items()))); report={"schema_version":SCHEMA,"denominator":"retained_gold","metrics":_metrics(predictions,answers,folds),"selections":selections,"feature_importance":dict(sorted(importance.items(),key=lambda x:-x[1])),"fold_isolated":True,"nested_selection":True}; atomic_json(output_dir/"oof_report.json",report); log.set_telemetry(report["metrics"])
    files=[output_dir/"oof_predictions.jsonl",output_dir/"oof_report.json"]; atomic_json(output_dir/"manifest.json",artifact_manifest(stage="exp027-nested-oof",inputs={"feature_manifest_sha256":sha256_file(feature_dir/"manifest.json")},config={"feature_sets":list(FEATURE_SETS),"params":list(PARAMS),"denominator":"retained_gold"},files=files)); return report


def shortlist(*, predictions: Path, answers: dict[str,set[str]], folds: dict[str,list[str]], output_dir: Path, v3_fingerprint: str) -> dict[str,Any]:
    pred={str(row["qid"]):[str(x) for x in row["doc_ids"]] for row in read_jsonl(predictions)}; per={name:evaluate_rankings({q:pred[q] for q in qids},{q:answers[q] for q in qids},ks=K_GRID) for name,qids in folds.items()}; viable=[k for k in K_GRID if all(value[f"recall@{k}"]>=.985 for value in per.values())]; k=viable[0] if viable else None; report={"schema_version":SCHEMA,"denominator":"retained_gold","threshold":.985,"grid":list(K_GRID),"per_fold":per,"shortlist_k":k,"pairs":len(pred)*k if k else None,"shortfall_at_100":{name:.985-value["recall@100"] for name,value in per.items()},"status":"PASS" if k else "FAIL"}
    with stage_run(output_dir,"exp027-shortlist-audit",total=len(pred),v3_fingerprint=v3_fingerprint) as log: atomic_json(output_dir/"shortlist_audit.json",report); log.set_telemetry(report)
    atomic_json(output_dir/"manifest.json",artifact_manifest(stage="exp027-shortlist-audit",inputs={"predictions_sha256":sha256_file(predictions)},config={"grid":list(K_GRID),"floor":.985,"denominator":"retained_gold"},files=[output_dir/"shortlist_audit.json"])); return report


def report(*, retained: dict[str,Any], baseline: Path, oof_path: Path, shortlist_path: Path, output_dir: Path, capsule_audit: Path | None = None) -> dict[str,Any]:
    payload={"schema_version":SCHEMA,"retained_label_impact":retained,"baselines":_json(baseline),"oof":_json(oof_path),"shortlist":_json(shortlist_path),"capsules":_json(capsule_audit) if capsule_audit and capsule_audit.exists() else None,"exp026_status":"diagnostic_only_due_to_missing_BM25_provenance_and_all_gold_shortlist_denominator"}; atomic_json(output_dir/"REPORT.json",payload); return payload


def main(argv: Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("sidecar","features","baselines","oof","shortlist","capsules","report")); p.add_argument("--cache-root",type=Path,default=ROOT/"cache"/"exp027_lambdamart_shortlist"); p.add_argument("--results-root",type=Path,default=ROOT/"results"/"exp027_lambdamart_shortlist"); p.add_argument("--candidates",type=Path,default=ROOT/"cache"/"exp022_e5_bm25_union"/"train_oof_candidates.jsonl"); p.add_argument("--evidence",type=Path,default=ROOT/"cache"/"exp021_sparse"/"depth_tune"/"raw4096_evidence"); p.add_argument("--tuning-report",type=Path,default=ROOT/"results"/"exp021_sparse"/"depth_rrf_tuning"/"tuning_report.json"); p.add_argument("--train",type=Path,default=ROOT/"public_test_dataset"/"train.json"); p.add_argument("--folds",type=Path,default=ROOT/"cache"/"cv_folds.json"); p.add_argument("--v3",type=Path,default=ROOT/"cache"/"structural_v3_e5_final_v1"); p.add_argument("--preprocessing",type=Path,default=ROOT/"cache"/"final_preprocessed_v2"); p.add_argument("--bm25-db",type=Path,default=ROOT/"cache"/"exp021_sparse"/"passage_hierarchy"/"fts5"/"bm25_v3.sqlite"); a=p.parse_args(argv); answers,impact=retained_answers(a.train,a.preprocessing/"exclusions.json",a.preprocessing/"train_label_impact.jsonl"); folds=load_folds(a.folds); fp=load_v3_manifest(a.v3)["content_fingerprint"]; sidecar=a.cache_root/"provenance"/"provenance_sidecar.jsonl"
    if a.stage=="sidecar": result=build_sidecar(candidates=a.candidates,evidence_dir=a.evidence,tuning_report=a.tuning_report,folds_path=a.folds,output_dir=a.cache_root/"provenance")
    elif a.stage=="features": result=feature_rows(candidates=a.candidates,sidecar=sidecar,v3_dir=a.v3,output_dir=a.cache_root/"features")
    elif a.stage=="baselines": result=baselines(candidates=a.candidates,sidecar=sidecar,answers=answers,folds=folds,output_dir=a.results_root/"baselines",v3_fingerprint=fp)
    elif a.stage=="oof": result=oof(feature_dir=a.cache_root/"features",answers=answers,folds=folds,output_dir=a.results_root/"oof",v3_fingerprint=fp)
    elif a.stage=="shortlist": result=shortlist(predictions=a.results_root/"oof"/"oof_predictions.jsonl",answers=answers,folds=folds,output_dir=a.results_root/"shortlist",v3_fingerprint=fp)
    elif a.stage=="capsules": result=build_capsules(candidates=a.candidates,predictions=a.results_root/"oof"/"oof_predictions.jsonl",shortlist=a.results_root/"shortlist"/"shortlist_audit.json",v3_dir=a.v3,bm25_db=a.bm25_db,output_dir=a.cache_root/"capsules")
    else: result=report(retained=impact,baseline=a.results_root/"baselines"/"baseline_report.json",oof_path=a.results_root/"oof"/"oof_report.json",shortlist_path=a.results_root/"shortlist"/"shortlist_audit.json",output_dir=a.results_root,capsule_audit=a.cache_root/"capsules"/"capsule_audit.json")
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
