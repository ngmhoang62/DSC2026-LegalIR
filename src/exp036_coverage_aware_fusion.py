"""EXP-036: fold-isolated, coverage-aware fusion over frozen E5 and BM25.

This module deliberately works in the *source union* only.  It first proves
the 150+150 parent universe has sufficient coverage; no learned model is fit
when that ceiling fails.  The resulting files are self-describing and are not
inputs to any reranker.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, write_jsonl
from exp012b_tuning import load_folds
from exp030_legal_evidence_routing import LABEL_POLICY, canonical_answers

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="legalir.exp036_coverage_aware_fusion.v1"
POLICY="canonical_duplicate_alias_drop_empty_passage_v1"
KS=(5,16,24,32,40,50,64); SEED=2105; UNIVERSE_FLOOR=.992; K32_FLOOR=.985
FEATURES=("e5_rank","bm25_rank","e5_score","has_e5","has_bm25","e5_percentile","bm25_percentile","e5_z","rank_gap","e5_gap","bm25_gap","source_agreement","title_overlap","parse_fallback","passage_log","scope_nodes","label_tokens","query_tokens","e5_evidence_count","e5_evidence_max","e5_evidence_mean","bm25_passage_min","bm25_passage_count","source_visibility")

def _j(p:Path)->Any: return json.loads(p.read_text(encoding="utf-8"))
def _digest(v:Any)->str: return hashlib.sha256(canonical_json(v).encode()).hexdigest()
def _rank(ids:list[str], gold:str)->int:
    try:return ids.index(gold)+1
    except ValueError:return 10**6
def _fold_name(value:str)->str:
    value=str(value)
    return value if value.startswith("fold_") else f"fold_{value}"
def _stable(rows:Iterable[tuple[str,float]])->list[str]: return [d for d,_ in sorted(rows,key=lambda x:(-float(x[1]),str(x[0])))]
def evaluate_rankings(rankings,answers,qids,ks=KS):
    totals={f"recall@{k}":0. for k in ks}; totals.update({f"precision@{k}":0. for k in ks}); totals["mrr@5"]=0.; n=0
    for q in qids:
        gold=answers[str(q)]
        if not gold: continue
        pred=rankings[str(q)]; n+=1
        for k in ks:
            totals[f"recall@{k}"]+=len(set(pred[:k])&gold)/len(gold); totals[f"precision@{k}"]+=len(set(pred[:k])&gold)/k
        first=next((i for i,d in enumerate(pred[:5],1) if d in gold),None); totals["mrr@5"]+=0 if first is None else 1/first
    return {k:v/max(n,1) for k,v in totals.items()}
def _fold_metrics(r:dict[str,list[str]], a:dict[str,set[str]], f:dict[str,list[str]], qids:Iterable[str]|None=None)->dict[str,Any]:
    q=list(qids) if qids is not None else sorted(a)
    return {"aggregate":evaluate_rankings(r,a,q,KS),"per_fold":{x:evaluate_rankings(r,a,ids,KS) for x,ids in sorted(f.items()) if set(ids)&set(q)}}

def _load_e5(path:Path)->dict[str,dict[str,dict[str,Any]]]:
    out={}
    for row in read_jsonl(path):
        q=str(row["qid"]); vals={}
        for rank,c in enumerate(row["candidates"][:150],1):
            d=str(c["doc_id"]); vals[d]={"rank":rank,"score":float(c.get("aggregate_score",0)),"evidence":c.get("evidence",[])}
        out[q]=vals
    return out
def _load_bm25(shards:Path)->dict[str,dict[str,dict[str,Any]]]:
    out={}
    for path in sorted(shards.glob("evidence_*.jsonl")):
        for row in read_jsonl(path):
            q=str(row["qid"])
            if q in out: raise RuntimeError(f"duplicate BM25 QID: {q}")
            vals={}
            for rank,item in enumerate(row.get("evidence",[])[:150],1):
                d=str(item[0]); vals[d]={"rank":rank,"passages":list(item[1]) if len(item)>1 else []}
            out[q]=vals
    return out
def _tokens(t:str)->set[str]: return {x for x in t.lower().split() if x}

def audit(*,train:Path,folds_path:Path,preprocessing:Path,v3:Path,e5_rankings:Path,bm25_shards:Path,out:Path)->dict[str,Any]:
    if LABEL_POLICY!=POLICY: raise RuntimeError("canonical label policy drift")
    answers,stats=canonical_answers(train,preprocessing/"exclusions.json",preprocessing/"train_label_impact.jsonl")
    if stats["evaluable_queries"]!=6991: raise RuntimeError("unexpected canonical denominator")
    folds={str(k):list(map(str,v)) for k,v in load_folds(folds_path).items()}; fold_of={q:f for f,qs in folds.items() for q in qs}
    query={str(q):str(x["question"]) for q,x in _j(train).items()}; docs={str(x["doc_id"]):x for x in read_jsonl(v3/"documents.jsonl")}
    e5,bm25=_load_e5(e5_rankings),_load_bm25(bm25_shards)
    if set(e5)!=set(answers) or set(bm25)!=set(answers): raise RuntimeError("source rankings and canonical QIDs differ")
    records=[]; source_e5={}; source_bm={}; universe={}
    for q in sorted(answers):
        eu,bu=e5[q],bm25[q]; ids=sorted(set(eu)|set(bu),key=lambda d:(min(eu.get(d,{"rank":999})["rank"],bu.get(d,{"rank":999})["rank"]),str(d)))
        if len(ids)>300: raise RuntimeError("source union exceeds 300 parents")
        source_e5[q]=_stable((d,151-e["rank"]) for d,e in eu.items()); source_bm[q]=_stable((d,151-e["rank"]) for d,e in bu.items()); universe[q]=ids
        es=np.array([x["score"] for x in eu.values()],dtype=np.float64); mean=float(es.mean()) if len(es) else 0.; sd=max(float(es.std()),1e-6); top=sorted(es,reverse=True)
        qtok=_tokens(query[q]); rows=[]
        for d in ids:
            er=eu.get(d); br=bu.get(d); meta=docs[d]; ev=(er or {}).get("evidence",[]); ps=(br or {}).get("passages",[])
            title=_tokens(str(meta.get("document_label",""))); erank=float(er["rank"]) if er else 151.; brank=float(br["rank"]) if br else 151.; score=float(er["score"]) if er else 0.
            evscores=[float(x.get("score",x.get("chunk_score",0))) for x in ev]
            vals=(erank,brank,score,float(er is not None),float(br is not None),erank/150.,brank/150.,(score-mean)/sd,abs(erank-brank), (top[0]-score if er and top else 0.), 1./brank if br else 0.,float(er is not None and br is not None),len(qtok&title)/max(len(qtok|title),1),float(meta.get("parse_mode")=="fallback"),math.log1p(float(meta.get("passage_length",0))),float(len(meta.get("scope_node_ids",[]))),float(len(str(meta.get("document_label","")).split())),float(len(query[q].split())),float(len(ev)),max(evscores,default=0.),float(np.mean(evscores)) if evscores else 0.,float(min(ps)) if ps else 999.,float(len(ps)),float((er is not None)+(br is not None)))
            rows.append({"doc_id":d,"features":[float(x) for x in vals],"e5_rank":None if not er else er["rank"],"bm25_rank":None if not br else br["rank"]})
        records.append({"qid":q,"fold":fold_of[q],"query":query[q],"candidates":rows})
    out.mkdir(parents=True,exist_ok=True); write_jsonl(out/"universe.jsonl",records); atomic_json(out/"feature_schema.json",{"schema_version":SCHEMA,"columns":FEATURES})
    # Pool coverage is evaluated over the complete (<=300) universe.  The
    # source-oracle @32 below is a ranking diagnostic, never the ceiling.
    def coverage(qids:list[str])->float:
        evaluable=[q for q in qids if answers[q]]
        if not evaluable: raise RuntimeError("fold has no evaluable canonical queries")
        return float(np.mean([len(set(universe[q])&answers[q])/len(answers[q]) for q in evaluable]))
    ceiling={name:{"universe_recall":coverage(qs),"source_oracle_top32":evaluate_rankings(universe,answers,qs,(32,))["recall@32"],"curves":evaluate_rankings(universe,answers,qs,KS)} for name,qs in sorted(folds.items())}; e5m=_fold_metrics(source_e5,answers,folds); bm=_fold_metrics(source_bm,answers,folds)
    blind=[]
    for q,g in answers.items():
        for d in g:
            if _rank(source_e5[q],d)>100 and _rank(source_bm[q],d)>50 and _rank(universe[q],d)<=300: blind.append([q,d])
    ok=all(x["universe_recall"]>=UNIVERSE_FLOOR for x in ceiling.values())
    report={"schema_version":SCHEMA,"status":"PASS" if ok else "REJECTED_SOURCE_CEILING","label_policy":POLICY,"label_stats":stats,"universe":{"max_parents":max(map(len,universe.values())),"mean_parents":float(np.mean([len(x) for x in universe.values()])),"per_fold_ceiling":ceiling,"blind_gold_recovered_from_e5_100_bm25_50":len(blind)},"source_baselines":{"e5":e5m,"bm25":bm},"inputs":{"train":sha256_file(train),"folds":sha256_file(folds_path),"e5":sha256_file(e5_rankings),"v3_manifest":sha256_file(v3/"manifest.json")},"fingerprint":_digest(records)}
    atomic_json(out/"AUDIT.json",report); atomic_json(out/"RUN_STATUS.json",{"status":report["status"],"stage":"audit"}); atomic_json(out/"_SUCCESS.json",{"schema_version":SCHEMA,"status":report["status"]}); return report

def _rrf(rec:dict[str,Any],constant:int=32)->np.ndarray:
    return np.asarray([(.65/(constant+(r["e5_rank"] or 10**6)))+(.35/(constant+(r["bm25_rank"] or 10**6))) for r in rec["candidates"]],dtype=np.float64)
def _calibrated_rrf(rec:dict[str,Any])->np.ndarray:
    """Registered EXP-034 calibrated RRF: alpha=.65, rank constant=32.

    EXP-034's ``selected_by_outer`` stores parent aggregation, not RRF
    coefficients.  Parsing it as alpha/k would silently corrupt this baseline.
    """
    a,k=.65,32
    return np.asarray([(a/(k+(r["e5_rank"] or 10**6)))+((1-a)/(k+(r["bm25_rank"] or 10**6))) for r in rec["candidates"]],dtype=np.float64)
def _guarded(scores:np.ndarray,rec:dict[str,Any])->np.ndarray:
    # Guard only source heads; remaining candidates retain learned ordering.
    boost=np.zeros(len(scores));
    for i,r in enumerate(rec["candidates"]):
        if min(r["e5_rank"] or 999,r["bm25_rank"] or 999)<=3: boost[i]=1e6
    return scores+boost
def _train_predict(train_records:list[dict[str,Any]],test_records:list[dict[str,Any]],answers:dict[str,set[str]])->dict[str,np.ndarray]:
    try: import lightgbm as lgb
    except ImportError as e: raise RuntimeError("EXP-036 training requires the preinstalled LightGBM runtime") from e
    x=np.asarray([r["features"] for z in train_records for r in z["candidates"]],dtype=np.float32); y=np.asarray([float(r["doc_id"] in answers[z["qid"]]) for z in train_records for r in z["candidates"]]); groups=[len(z["candidates"]) for z in train_records]
    model=lgb.LGBMRanker(objective="lambdarank",learning_rate=.03,n_estimators=400,num_leaves=31,min_child_samples=20,reg_lambda=1,random_state=SEED,n_jobs=1,deterministic=True,verbosity=-1)
    model.fit(x,y,group=groups); return {z["qid"]:model.predict(np.asarray([r["features"] for r in z["candidates"]],dtype=np.float32)) for z in test_records}
def _orders(records:list[dict[str,Any]],scores:dict[str,np.ndarray],guard:bool=False)->dict[str,list[str]]:
    out={}
    for z in records:
        v=_guarded(scores[z["qid"]],z) if guard else scores[z["qid"]]
        out[z["qid"]]=_stable((r["doc_id"],float(s)) for r,s in zip(z["candidates"],v))
    return out

def run_fold(*,audit_dir:Path,out:Path,outer:str,resume:bool=False)->dict[str,Any]:
    audit_report=_j(audit_dir/"AUDIT.json")
    if audit_report["status"]!="PASS": raise RuntimeError("source ceiling gate failed; learned fusion is forbidden")
    answers,_=canonical_answers(ROOT/"public_test_dataset"/"train.json",ROOT/"cache"/"final_preprocessed_v2"/"exclusions.json",ROOT/"cache"/"final_preprocessed_v2"/"train_label_impact.jsonl")
    folds={str(k):list(map(str,v)) for k,v in load_folds(ROOT/"cache"/"cv_folds.json").items()}; outer=_fold_name(outer)
    if outer not in folds: raise ValueError(f"unknown outer fold: {outer}")
    records=list(read_jsonl(audit_dir/"universe.jsonl")); by={r["qid"]:r for r in records}; held=folds[outer]; train=[r for r in records if r["qid"] not in set(held)]; test=[by[q] for q in held]
    fold_out=out/outer; target=fold_out/"rankings.jsonl"
    if resume and target.exists(): return _j(fold_out/"REPORT.json")
    # Inner cross-fit chooses policy without using outer heldout labels.
    inner=[]
    for inner_fold in sorted(set(r["fold"] for r in train)):
        va=[r for r in train if r["fold"]==inner_fold]; tr=[r for r in train if r["fold"]!=inner_fold]; pred=_train_predict(tr,va,answers)
        policies={"learned":_orders(va,pred),"guardrail":_orders(va,pred,True),"rrf":_orders(va,{r["qid"]:_calibrated_rrf(r) for r in va})}
        inner.append({n:evaluate_rankings(v,answers,[r["qid"] for r in va],KS) for n,v in policies.items()})
    def key(n:str):
        m={k:float(np.mean([x[n][k] for x in inner])) for k in ("recall@32","recall@16","recall@5","mrr@5")}; return (m["recall@32"],m["recall@16"],m["recall@5"],m["mrr@5"])
    policy=max(("rrf","learned","guardrail"),key=key); pred=_train_predict(train,test,answers) if policy!="rrf" else {r["qid"]:_calibrated_rrf(r) for r in test}; ranking=_orders(test,pred,policy=="guardrail")
    fold_out.mkdir(parents=True,exist_ok=True); write_jsonl(target,({"qid":q,"doc_ids":ranking[q],"policy":policy,"fold":str(outer)} for q in sorted(ranking)))
    metric=evaluate_rankings(ranking,answers,held,KS); report={"schema_version":SCHEMA,"outer_fold":outer,"policy":policy,"inner_selection":{n:key(n) for n in ("rrf","learned","guardrail")},"metrics":metric,"heldout_qids_fingerprint":_digest(sorted(held)),"fold_isolation":{"heldout_excluded_from_training":True,"heldout_excluded_from_selection":True}}
    atomic_json(fold_out/"REPORT.json",report); atomic_json(fold_out/"_SUCCESS.json",report); return report

def report(*,audit_dir:Path,out:Path)->dict[str,Any]:
    answers,_=canonical_answers(ROOT/"public_test_dataset"/"train.json",ROOT/"cache"/"final_preprocessed_v2"/"exclusions.json",ROOT/"cache"/"final_preprocessed_v2"/"train_label_impact.jsonl"); folds={str(k):list(map(str,v)) for k,v in load_folds(ROOT/"cache"/"cv_folds.json").items()}; rankings={}
    for f in folds:
        p=out/f/"rankings.jsonl"
        if p.exists(): rankings.update({str(x["qid"]):list(map(str,x["doc_ids"])) for x in read_jsonl(p)})
    metrics=_fold_metrics(rankings,answers,folds,rankings) if rankings else {}
    complete=set(rankings)==set(answers); per=metrics.get("per_fold",{})
    baseline=_j(ROOT/"results"/"exp034_shallow_retrieval"/"calibration"/"REPORT.json")["best_rrf"]
    bper=baseline["per_fold"]
    # all-gold incomplete rate is a query-level metric, separate from the
    # average per-gold recall reported above.
    def incomplete(qids): return float(np.mean([not answers[q] <= set(rankings[q][:32]) for q in qids]))
    multi=[q for q,g in answers.items() if len(g)>1]
    # Baseline rankings are provenance-complete and provide the comparison.
    brank={str(x["qid"]):list(map(str,x["doc_ids"])) for x in read_jsonl(ROOT/"results"/"exp034_shallow_retrieval"/"calibration"/"rrf_rankings.jsonl")}
    multi_new=float(np.mean([not answers[q] <= set(rankings[q][:32]) for q in multi])) if complete else None
    multi_old=float(np.mean([not answers[q] <= set(brank[q][:32]) for q in multi]))
    accepted=complete and all(x["recall@32"]>=K32_FLOOR and x["recall@5"]>=bper[f]["recall@5"]-.002 for f,x in per.items()) and metrics["aggregate"]["recall@5"]>=baseline["metrics"]["recall@5"] and multi_new is not None and multi_new<=multi_old*.8
    early_fail=bool("fold_0" in per and per["fold_0"]["recall@32"]<K32_FLOOR)
    status="ACCEPTED" if accepted else ("REJECTED_EARLY_FOLD0" if early_fail else ("REJECTED" if complete else "INCOMPLETE"))
    result={"schema_version":SCHEMA,"status":status,"ranking_metrics":metrics,"candidate_source_ceiling":_j(audit_dir/"AUDIT.json")["universe"]["per_fold_ceiling"],"official_exp034_rrf":baseline,"early_stop":{"triggered":early_fail,"reason":"fold_0 Recall@32 below fixed promotion floor" if early_fail else None},"multi_gold_incomplete_at32":{"new":multi_new,"baseline":multi_old,"relative_reduction":None if multi_new is None else 1-multi_new/max(multi_old,1e-12)}}
    out.mkdir(parents=True,exist_ok=True); atomic_json(out/"REPORT.json",result); atomic_json(out/"RUN_STATUS.json",{"status":status,"stage":"report"});
    if status!="INCOMPLETE": atomic_json(out/"_SUCCESS.json",result)
    return result

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("audit","fixture","run-fold","overnight","report")); p.add_argument("--resume",action="store_true"); p.add_argument("--outer",default="0"); a=p.parse_args()
    audit_dir=ROOT/"cache"/"exp036_coverage_aware_fusion"; out=ROOT/"results"/"exp036_coverage_aware_fusion"
    kw=dict(train=ROOT/"public_test_dataset"/"train.json",folds_path=ROOT/"cache"/"cv_folds.json",preprocessing=ROOT/"cache"/"final_preprocessed_v2",v3=ROOT/"cache"/"structural_v3_e5_final_v1",e5_rankings=ROOT/"cache"/"exp021_e5_dense_candidates"/"config_rankings"/"top2_mean.jsonl",bm25_shards=ROOT/"cache"/"exp021_sparse"/"depth_tune"/"raw4096_evidence"/"shards",out=audit_dir)
    if a.stage in ("audit","fixture"): result=audit(**kw)
    elif a.stage=="run-fold": result=run_fold(audit_dir=audit_dir,out=out,outer=a.outer,resume=a.resume)
    elif a.stage=="overnight":
        ar=audit(**kw)
        if ar["status"]=="PASS":
            for f in map(str,range(5)): run_fold(audit_dir=audit_dir,out=out,outer=f,resume=True)
        result=report(audit_dir=audit_dir,out=out)
    else: result=report(audit_dir=audit_dir,out=out)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__=="__main__": main()
