"""EXP-035: immutable-label retrieval error cohort and evidence-pack audit."""
from __future__ import annotations

import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, write_jsonl
from exp012b_tuning import load_folds
from exp030_legal_evidence_routing import LABEL_POLICY, canonical_answers

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp035_retrieval_error_adjudication.v1"
POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
TAGS = ("valid_direct_gold", "multi_gold_secondary_or_background", "underspecified_query",
        "suspected_label_semantic_mismatch", "source_generation_miss", "fusion_displacement",
        "structural_or_evidence_failure", "unresolved")

def _json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def _hash(value: Any) -> str: return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
def _rank(ids: list[str], gold: str) -> int:
    try: return ids.index(gold) + 1
    except ValueError: return 10**6
def _family(text: str) -> str:
    text=text.lower()
    for name, pat in (("explicit_locator",r"\b(điều|khoản|điểm)\s+\d+"),("definition",r"thế nào là|là gì|được hiểu|khái niệm"),("sanction",r"xử phạt|mức phạt|truy cứu"),("procedure",r"thủ tục|hồ sơ|thời hạn|bao lâu"),("rights",r"điều kiện|quyền|nghĩa vụ|được phép")):
        if re.search(pat,text): return name
    return "other"

def _rankings(path: Path) -> dict[str,list[str]]:
    return {str(x["qid"]): [str(v) for v in x["doc_ids"]] for x in read_jsonl(path)}

def _tag(row: dict[str,Any]) -> str:
    """Conservative deterministic starter tag; human review may only refine it."""
    if int(row["gold_count"]) > 1 and int(row["rrf_rank"]) <= 32: return "multi_gold_secondary_or_background"
    if int(row["e5_rank_exp022"]) > 100 and int(row["bm25_rank_exp022"]) > 50: return "source_generation_miss"
    if min(int(row["e5_rank_exp022"]),int(row["bm25_rank_exp022"])) <= 32 and int(row["rrf_rank"]) > 32: return "fusion_displacement"
    if row.get("parse_mode") == "fallback": return "structural_or_evidence_failure"
    if _family(str(row["question"])) == "other" and len(str(row["question"]).split()) <= 10: return "underspecified_query"
    return "unresolved"

def audit(*, train: Path, folds_path: Path, preprocessing: Path, docs_path: Path, rrf_path: Path,
          error_cases: Path, output: Path, control_per_error: int=1) -> dict[str,Any]:
    if LABEL_POLICY != POLICY: raise RuntimeError("canonical label policy drift")
    labels, stats = canonical_answers(train, preprocessing/"exclusions.json", preprocessing/"train_label_impact.jsonl")
    if stats["evaluable_queries"] != 6991: raise RuntimeError("unexpected canonical denominator")
    folds={k:list(map(str,v)) for k,v in load_folds(folds_path).items()}; fold_for={q:k for k,v in folds.items() for q in v}
    queries={str(q):str(v["question"]) for q,v in _json(train).items()}; docs={str(x["doc_id"]):x for x in read_jsonl(docs_path)}
    rrf=_rankings(rrf_path); source_rows=defaultdict(list)
    for row in read_jsonl(error_cases): source_rows[str(row["qid"])].append(dict(row))
    official={q for q,g in labels.items() if g and any(_rank(rrf[q],d)>32 for d in g)}
    shadow={q for q,rows in source_rows.items() if any(int(x["exp102_rank"])>32 for x in rows)}
    errors=official|shadow
    # Matched controls: deterministic same fold/family/gold count, then first lexical qid.
    candidates=defaultdict(list)
    for q,g in labels.items():
        if not g or q in errors: continue
        key=(fold_for[q],_family(queries[q]),len(g)); candidates[key].append(q)
    controls=[]
    for q in sorted(errors):
        key=(fold_for[q],_family(queries[q]),len(labels[q])); controls.extend(candidates[key][:control_per_error])
    controls=sorted(set(controls)-errors)
    cohort=[]
    for role,qids in (("error",sorted(errors)),("control",controls)):
        for q in qids:
            by_gold={x["gold_doc_id"]:x for x in source_rows[q]}
            gold_rows=[]
            for gold in sorted(labels[q]):
                prior=by_gold.get(gold,{})
                record={"gold_doc_id":gold,"rrf_rank":_rank(rrf[q],gold),"e5_rank_exp022":prior.get("e5_rank_exp022",10**6),"bm25_rank_exp022":prior.get("bm25_rank_exp022",10**6),"exp101_rank":prior.get("exp101_rank"),"exp102_rank":prior.get("exp102_rank"),"exp103_rank":prior.get("exp103_rank"),"document_label":docs[gold].get("document_label",""),"parse_mode":docs[gold].get("parse_mode","missing"),"passage_length":docs[gold].get("passage_length",0)}
                gold_rows.append(record)
            base={"schema_version":SCHEMA,"qid":q,"role":role,"fold":fold_for[q],"question":queries[q],"query_family":_family(queries[q]),"gold_count":len(labels[q]),"golds":gold_rows}
            # Starter tag is immutable diagnostic metadata; reviewers append reviewed_tag/reviewer_note.
            flat={"question":queries[q],"gold_count":len(labels[q]),"rrf_rank":max(x["rrf_rank"] for x in gold_rows),"e5_rank_exp022":min(x["e5_rank_exp022"] for x in gold_rows),"bm25_rank_exp022":min(x["bm25_rank_exp022"] for x in gold_rows),"parse_mode":"fallback" if any(x["parse_mode"]=="fallback" for x in gold_rows) else "structured"}
            base["starter_tag"]=_tag(flat); base["reviewed_tag"]=None; base["reviewer_note"]=None
            base["top_rrf_candidates"]=rrf[q][:32]; cohort.append(base)
    output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/"evidence_pack.jsonl",cohort)
    report={"schema_version":SCHEMA,"status":"PASS","label_policy":LABEL_POLICY,"label_stats":stats,"cohort":{"error_queries":len(errors),"official_rrf_errors":len(official),"exp102_shadow_errors":len(shadow),"controls":len(controls),"rows":len(cohort)},"tags":list(TAGS),"exp102":"observational_only","inputs":{"train_sha256":sha256_file(train),"folds_sha256":sha256_file(folds_path),"rrf_sha256":sha256_file(rrf_path),"error_cases_sha256":sha256_file(error_cases),"docs_sha256":sha256_file(docs_path)},"cohort_fingerprint":_hash([{k:v for k,v in row.items() if k not in ("reviewed_tag","reviewer_note")} for row in cohort])}
    atomic_json(output/"REPORT.json",report); atomic_json(output/"RUN_STATUS.json",{"status":"COMPLETE","stage":"audit","rows":len(cohort)})
    atomic_json(output/"_SUCCESS.json",{"schema_version":SCHEMA,"report_sha256":sha256_file(output/"REPORT.json")}); return report

def main(argv: Iterable[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("audit","fixture","run-fold","overnight","report")); p.add_argument("--output",type=Path,default=ROOT/"results"/"exp035_retrieval_error_adjudication"); p.add_argument("--resume",action="store_true"); a=p.parse_args(argv)
    if a.stage not in ("audit","fixture","report"): raise SystemExit("EXP-035 is audit-only; stage is intentionally blocked")
    result=audit(train=ROOT/"public_test_dataset"/"train.json",folds_path=ROOT/"cache"/"cv_folds.json",preprocessing=ROOT/"cache"/"final_preprocessed_v2",docs_path=ROOT/"cache"/"structural_v3_e5_final_v1"/"documents.jsonl",rrf_path=ROOT/"results"/"exp034_shallow_retrieval"/"calibration"/"rrf_rankings.jsonl",error_cases=ROOT/"results"/"retrieval_error_analysis_101_103"/"gold_rank_cases.jsonl",output=a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
