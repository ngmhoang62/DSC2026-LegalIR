"""EXP-029: fold-isolated reranker benchmark with a failure-isolated runner.

The module deliberately does not retrieve documents.  It consumes EXP-022's
immutable 150-parent pool and creates a separate K=64 capsule set for *each*
outer fold: outer-train queries are ranked by an inner-cross-fitted LambdaMART,
while outer-heldout queries are ranked by a model fitted only on outer-train.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import (artifact_manifest, atomic_json, canonical_json,
    load_v3_manifest, read_jsonl, require_success, sha256_file, write_jsonl)
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp026_lambdamart_capsules import capsules as build_capsules
from exp027_lambdamart_shortlist import retained_answers
from exp028_lambdamart_shortlist import _fit, _load_features, _predict

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "legalir.exp029_nested_lora_reranker.v1"
SEED, K, MAX_LENGTH, SCREEN_QUERIES = 2029, 64, 512, 64
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "epochs": 3, "effective_batch": 32}
LEGAL_INSTRUCTION = "Given a Vietnamese legal-information need, retrieve legal passages that authoritatively answer the query."
JINA_CITATION = """@misc{nasika2026jinarerankerv35, title={jina-reranker-v3.5: Hybrid-Attention Listwise Reranking with Self-Distillation for Domain-Robust Retrieval}, author={Christina Nasika and Feng Wang and Antonis Minas Krasakis and Han Xiao}, year={2026}, eprint={2607.18152}, archivePrefix={arXiv}, primaryClass={cs.CL}, url={https://arxiv.org/abs/2607.18152}}"""

@dataclass(frozen=True)
class ModelSpec:
    key: str; model_id: str; kind: str; lora_targets: tuple[str, ...] = ()

MODELS = (
    ModelSpec("vietnamese", "AITeamVN/Vietnamese_Reranker", "pair", ("query", "key", "value", "dense")),
    ModelSpec("bge_m3", "BAAI/bge-reranker-v2-m3", "pair", ("query", "key", "value", "dense")),
    ModelSpec("gte", "Alibaba-NLP/gte-multilingual-reranker-base", "pair", ("query", "key", "value", "dense")),
    ModelSpec("mmarco", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "pair", ("query", "key", "value", "dense")),
    ModelSpec("qwen3", "Qwen/Qwen3-Reranker-0.6B", "qwen", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ModelSpec("jina", "jinaai/jina-reranker-v3.5", "jina"),
)
MODEL_BY_KEY = {x.key: x for x in MODELS}

def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _hash(value: Any) -> str: return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()
def _now() -> float: return time.time()
def _qid_set(path: Path) -> set[str]:
    payload=_json(path)
    return {str(x) for x in (payload if isinstance(payload, list) else [payload])}

def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {"cache": args.cache_root, "results": args.results_root, "candidates": args.candidates,
      "sidecar": args.sidecar, "features": args.features, "train": args.train, "folds": args.folds,
      "preprocessing": args.preprocessing, "v3": args.v3, "bm25_db": args.bm25_db}

def _state(root: Path, event: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    event = {"schema_version": SCHEMA, "at": _now(), **event}
    with (root / "state.jsonl").open("a", encoding="utf8", newline="\n") as h: h.write(canonical_json(event) + "\n")
    atomic_json(root / "RUN_STATUS.json", event)

def _job(root: Path, name: str, fingerprint: str, fn: Callable[[], dict[str, Any]], *, retries: int = 0,
         isolate: str | None = None, resume: bool = False) -> dict[str, Any]:
    """Transactional runner primitive.  It never accepts a stale success marker."""
    out = root / "jobs" / name; success, failed = out / "_SUCCESS.json", out / "_FAILED.json"
    if resume and success.exists() and _json(success).get("fingerprint") == fingerprint:
        _state(root, {"job": name, "state": "SKIPPED_RESUME", "fingerprint": fingerprint}); return _json(success)
    out.mkdir(parents=True, exist_ok=True); success.unlink(missing_ok=True); failed.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        _state(root, {"job": name, "state": "RUNNING", "attempt": attempt, "fingerprint": fingerprint})
        try:
            payload = fn(); marker = {"fingerprint": fingerprint, "finished_at": _now(), "payload": payload}
            atomic_json(success, marker); _state(root, {"job": name, "state": "SUCCESS", "fingerprint": fingerprint}); return marker
        except Exception as error:
            detail = {"fingerprint": fingerprint, "attempt": attempt, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "finished_at": _now()}
            atomic_json(failed, detail); _state(root, {"job": name, "state": "RETRY" if attempt < retries else (isolate or "FAILED"), **detail})
    if isolate: return {"state": isolate, "error": _json(failed)["error"]}
    raise RuntimeError(_json(failed)["error"])

def audit(*, paths: dict[str, Path], output: Path) -> dict[str, Any]:
    require_success(paths["candidates"].parent); require_success(paths["sidecar"].parent); require_success(paths["features"])
    v3 = load_v3_manifest(paths["v3"]); data, index, columns = _load_features(paths["features"])
    candidates = {str(x["qid"]): x for x in read_jsonl(paths["candidates"])}; folds = load_folds(paths["folds"])
    answers, impact = retained_answers(paths["train"], paths["preprocessing"] / "exclusions.json", paths["preprocessing"] / "train_label_impact.jsonl")
    folded = {str(q) for qs in folds.values() for q in qs}
    if set(candidates) != set(answers) or folded != set(answers) or len(index) != len(answers) or data.shape[0] != len(answers) * 150:
        raise ValueError("EXP-029 train/fold/candidate/feature membership mismatch")
    for item in index:
        qid = str(item["qid"]); ids = [str(x["doc_id"]) for x in candidates[qid]["candidates"]]
        if item["doc_ids"] != ids or int(item["end"]) - int(item["start"]) != 150: raise ValueError(f"feature order mismatch: {qid}")
    report = {"schema_version": SCHEMA, "status": "PASS", "queries": len(answers), "pairs": len(answers)*K,
      "candidate_limit": 150, "shortlist_k": K, "columns": columns, "retained_label_impact": impact,
      "v3_fingerprint": v3["content_fingerprint"], "inputs": {k: sha256_file(p if p.is_file() else p / "manifest.json") for k,p in paths.items() if k not in {"cache","results","bm25_db"}}}
    output.mkdir(parents=True, exist_ok=True); atomic_json(output / "input_audit.json", report); atomic_json(output / "manifest.json", artifact_manifest(stage="exp029-audit", inputs=report["inputs"], config={"k":K}, files=[output / "input_audit.json"])); return report

def _outer_predictions(outer: str, folds: dict[str, list[str]], by_qid: dict[str, dict[str, Any]], data: np.ndarray,
                       answers: dict[str, set[str]], columns: list[str]) -> dict[str, list[str]]:
    """The critical anti-leakage split used to build one capsule inventory per outer fold."""
    params = {"num_leaves": 31, "min_child_samples": 30, "n_estimators": 300, "reg_lambda": 0.0}
    outer_train = [q for f, qs in folds.items() if f != outer for q in qs]; result: dict[str, list[str]] = {}
    for inner in sorted(set(folds) - {outer}):
        valid = set(folds[inner]); train_ids = [q for q in outer_train if q not in valid]
        model, positions = _fit([by_qid[q] for q in train_ids], data, answers, columns, "all", params)
        result.update(_predict(model, positions, [by_qid[q] for q in folds[inner]], data))
    model, positions = _fit([by_qid[q] for q in outer_train], data, answers, columns, "all", params)
    result.update(_predict(model, positions, [by_qid[q] for q in folds[outer]], data))
    if set(result) != set(by_qid): raise ValueError(f"incomplete outer cascade: {outer}")
    return result

def build_cascade(
    *, paths: dict[str, Path], output: Path,
    answers_override: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Build the historical EXP-029 cascade or an explicitly supplied label view.

    ``answers_override`` is intentionally opt-in so later experiments can fix a
    label policy without silently changing EXP-029's historical semantics.
    Callers using an override must bind ``output`` to their own fingerprinted
    namespace.
    """
    data,index,columns = _load_features(paths["features"]); by_qid={str(x["qid"]):x for x in index}; folds=load_folds(paths["folds"])
    answers = (
        {str(qid): set(gold) for qid, gold in answers_override.items()}
        if answers_override is not None
        else retained_answers(
            paths["train"], paths["preprocessing"] / "exclusions.json",
            paths["preprocessing"] / "train_label_impact.jsonl",
        )[0]
    )
    if set(answers) != set(by_qid):
        raise ValueError("cascade answer/feature qid membership mismatch")
    output.mkdir(parents=True, exist_ok=True); completed={}
    for outer in sorted(folds):
        odir=output / outer
        existing=odir / "capsules" / "capsules.jsonl"
        if (odir / "capsules" / "_SUCCESS.json").exists() and existing.exists():
            completed[outer] = {"capsules": str(existing.resolve()), "queries": len(by_qid), "resumed": True}; continue
        pred=_outer_predictions(outer,folds,by_qid,data,answers,columns)
        write_jsonl(odir / "predictions.jsonl", ({"qid":q,"doc_ids":ids} for q,ids in sorted(pred.items())))
        atomic_json(odir / "shortlist_audit.json", {"schema_version":SCHEMA,"status":"PASS","shortlist_k":K,"outer":outer,"fold_isolated":True})
        build_capsules(candidates=paths["candidates"], predictions=odir / "predictions.jsonl", shortlist=odir / "shortlist_audit.json", v3_dir=paths["v3"], bm25_db=paths["bm25_db"], output_dir=odir / "capsules")
        completed[outer] = {"capsules": str((odir / "capsules" / "capsules.jsonl").resolve()), "queries": len(pred)}
    atomic_json(output / "cascade_report.json", {"schema_version":SCHEMA,"status":"PASS","per_outer":completed,"k":K}); return completed

def _heading_chain(candidate: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    ids = candidate.get("scope_node_ids", []); labels=[]
    for node_id in ids:
        row=nodes.get(str(node_id));
        if row and row.get("heading_text"): labels.append(str(row["heading_text"]))
    return " > ".join(dict.fromkeys(labels))

def render_candidate(query: str, candidate: dict[str, Any], *, structural: bool, tokenizer: Any | None = None, max_length: int = MAX_LENGTH,
                     nodes: dict[str, dict[str, Any]] | None = None) -> str:
    prefix = "[Văn bản] " + str(candidate.get("document_label", ""))
    if structural: prefix += "\n[Ngữ cảnh] " + _heading_chain(candidate, nodes or {})
    evidence = "\n".join(str(x.get("raw_text", "")) for x in candidate.get("evidence", []))
    text = prefix + "\n[Nội dung] " + evidence
    if tokenizer is None: return text
    # Reserve pair special tokens by measuring the document alone and truncate deterministically.
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max(1, max_length - 32))["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

def _pair_scores(spec: ModelSpec, model: Any, tokenizer: Any, query: str, docs: list[str], device: str) -> list[float]:
    import torch
    if spec.kind == "qwen":
        yes = tokenizer("yes", add_special_tokens=False)["input_ids"][-1]; no = tokenizer("no", add_special_tokens=False)["input_ids"][-1]
        prompts=[f"<Instruct>: {LEGAL_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}\n<|im_start|>assistant\n" for doc in docs]
        batch=tokenizer(prompts,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt"); batch={k:v.to(device) for k,v in batch.items()}
        with torch.inference_mode(): logits=model(**batch).logits[:, -1, [no,yes]]; return torch.softmax(logits,dim=-1)[:,1].float().cpu().tolist()
    batch=tokenizer([query]*len(docs),docs,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
    # GTE's remote implementation has a one-entry type embedding table.  Its
    # own forward generates zero type ids when this field is absent; passing
    # tokenizer pair ids risks an out-of-range CUDA embedding index.
    if spec.key == "gte": batch.pop("token_type_ids", None)
    batch={k:v.to(device) for k,v in batch.items()}
    with torch.inference_mode(): return model(**batch).logits.reshape(-1).float().cpu().tolist()

def load_scorer(spec: ModelSpec, *, device: str, local_only: bool = False) -> tuple[Any, Any]:
    if spec.kind == "jina":
        module_cache=ROOT / "cache" / "exp029_nested_lora_reranker" / "hf_modules"
        module_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(module_cache)
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    if local_only:
        # Transformers/PEFT may otherwise probe adapter_config.json remotely
        # despite local_files_only=True.  A benchmark preflight must be fully
        # deterministic once model snapshots have been cached.
        os.environ["HF_HUB_OFFLINE"] = "1"
    if spec.kind == "jina":
        return AutoModel.from_pretrained(spec.model_id, trust_remote_code=True, local_files_only=local_only, dtype="auto").to(device).eval(), None
    tok=AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True, local_files_only=local_only, padding_side="left" if spec.kind=="qwen" else "right")
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    factory=AutoModelForCausalLM if spec.kind == "qwen" else AutoModelForSequenceClassification
    model=factory.from_pretrained(spec.model_id, trust_remote_code=True, local_files_only=local_only, torch_dtype="auto")
    if spec.key == "gte":
        # The model's remote code registers this non-persistent RoPE buffer, but
        # Transformers 5 leaves it uninitialized after loading.  Its intended
        # value is deterministic arange(max_position_embeddings); without this
        # repair the first RoPE lookup is out of bounds (and becomes a CUDA
        # device-side assert on GPU).
        import torch
        embeddings=model.new.embeddings
        embeddings.position_ids=torch.arange(int(model.config.max_position_embeddings), dtype=torch.long)
    model=model.to(device).eval()
    return model,tok

def score_capsules(*, spec: ModelSpec, capsules: Path, output: Path, structural: bool, device: str, local_only: bool = False,
                   resume: bool = False, qids: set[str] | None = None) -> dict[str, Any]:
    records=[x for x in read_jsonl(capsules) if qids is None or str(x["qid"]) in qids]; nodes={}
    model,tok=load_scorer(spec,device=device,local_only=local_only); output.mkdir(parents=True,exist_ok=True); path=output/"scores.jsonl"; existing={str(x["qid"]):x for x in read_jsonl(path)} if resume and path.exists() else {}
    started=time.perf_counter()
    try:
        for n,row in enumerate(records,1):
            qid=str(row["qid"])
            if qid in existing: continue
            docs=[render_candidate(str(row["query"]),c,structural=structural,tokenizer=tok,nodes=nodes) for c in row["candidates"]]
            if spec.kind=="jina":
                values=model.rerank(str(row["query"]),docs); scores=[0.0]*len(docs)
                for value in values: scores[int(value["index"] if isinstance(value,dict) else value.index)]=float(value["relevance_score"] if isinstance(value,dict) else value.relevance_score)
            else: scores=[]
            if spec.kind!="jina":
                for start in range(0,len(docs),8): scores.extend(_pair_scores(spec,model,tok,str(row["query"]),docs[start:start+8],device))
            existing[qid]={"schema_version":SCHEMA,"qid":qid,"model":spec.key,"structural":structural,"scores":[{"doc_id":str(c["doc_id"]),"score":float(s)} for c,s in zip(row["candidates"],scores)]}
            if n % 64 == 0: write_jsonl(path,(existing[k] for k in sorted(existing)))
        write_jsonl(path,(existing[k] for k in sorted(existing)))
    finally:
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    return {"queries":len(existing),"seconds":time.perf_counter()-started,"scores":str(path.resolve())}

def _worker(command: str, *, spec: ModelSpec, capsules: Path, output: Path, qids: set[str], device: str,
            local_only: bool, structural: bool = True, adapter: Path | None = None) -> dict[str, Any]:
    """Run exactly one CUDA action in a fresh interpreter.

    A device-side assert irreversibly poisons CUDA in its process.  Keeping the
    scheduler CPU-only makes model/fold failure isolation real rather than just
    an exception handler around a poisoned context.
    """
    output.mkdir(parents=True, exist_ok=True); qids_path=output / "worker_qids.json"; atomic_json(qids_path, sorted(qids))
    argv=[sys.executable, str(Path(__file__).resolve()), command, "--model", spec.key, "--capsules", str(capsules), "--output", str(output), "--qids-json", str(qids_path), "--device", device]
    if local_only: argv.append("--local-only")
    if structural: argv.append("--structural")
    if adapter is not None: argv += ["--adapter", str(adapter)]
    env=dict(os.environ); env["CUDA_LAUNCH_BLOCKING"]="1"
    with (output/"worker.stdout.log").open("w",encoding="utf8") as out, (output/"worker.stderr.log").open("w",encoding="utf8") as err:
        completed=subprocess.run(argv,cwd=str(ROOT),env=env,stdout=out,stderr=err,check=False)
    if completed.returncode:
        tail=(output/"worker.stderr.log").read_text(encoding="utf8",errors="replace")[-6000:]
        raise RuntimeError(f"{command} worker exit={completed.returncode}: {tail}")
    return _json(output/"worker_result.json")

def isolated_score(**kwargs: Any) -> dict[str, Any]: return _worker("score-worker", **kwargs)
def isolated_lora_score(**kwargs: Any) -> dict[str, Any]: return _worker("score-lora-worker", **kwargs)
def isolated_train(**kwargs: Any) -> dict[str, Any]: return _worker("train-worker", structural=True, **kwargs)

def _prediction_metrics(scores: Path, answers: dict[str,set[str]]) -> dict[str,float]:
    pred={str(x["qid"]):[str(y["doc_id"]) for y in sorted(x["scores"],key=lambda y:(-float(y["score"]),str(y["doc_id"])))[:5]] for x in read_jsonl(scores)}
    return evaluate_rankings(pred,{q:answers[q] for q in pred},ks=(5,))

def choose_screen(rows: list[dict[str,Any]]) -> dict[str,Any]:
    if not rows: raise ValueError("empty screen")
    return sorted(rows,key=lambda x:(-x["recall@5"],-x["precision@5"],x["seconds_per_query"], [m.key for m in MODELS].index(x["model"])))[0]

def preflight(*, output: Path, device: str, local_only: bool) -> dict[str,Any]:
    result={"schema_version":SCHEMA,"models":{},"lora":LORA}
    for spec in MODELS:
        try:
            model,tok=load_scorer(spec,device=device,local_only=local_only)
            names=[name for name,_ in model.named_modules()]; missing=[x for x in spec.lora_targets if not any(name.endswith(x) for name in names)]
            # Jina lacks a published local fine-tuning API: require an actual differentiable output, never assume it.
            # Pair encoders have a validated pairwise ranking loss below.  Qwen's
            # yes/no template is zero-shot-only until its backward fixture passes.
            eligible=spec.kind=="pair" and not missing
            result["models"][spec.key]={"state":"ELIGIBLE" if eligible else "ZERO_SHOT_ONLY","missing_targets":missing,"model_id":spec.model_id}
            del model; gc.collect()
        except Exception as error: result["models"][spec.key]={"state":"FAILED_MODEL","error":f"{type(error).__name__}: {error}"}
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    atomic_json(output/"preflight.json",result); return result

def _hard_groups(capsules: Iterable[dict[str,Any]], answers: dict[str,set[str]], seed: int = SEED) -> list[dict[str,Any]]:
    rng=np.random.default_rng(seed); result=[]
    for row in capsules:
        qid=str(row["qid"]); positives=[x for x in row["candidates"] if str(x["doc_id"]) in answers[qid]]; negatives=[x for x in row["candidates"] if str(x["doc_id"]) not in answers[qid]]
        for pos in positives:
            if negatives:
                picks=list(negatives[:3])+list(negatives[8:32][:2])+list(negatives[32:])[-1:]
                while len(picks)<8: picks.append(negatives[int(rng.integers(len(negatives)))])
                result.append({"qid":qid,"query":row["query"],"pos":pos,"neg":picks[:8]})
    return result

def train_lora(*, spec: ModelSpec, capsules: Path, train_qids: set[str], answers: dict[str,set[str]], output: Path, device: str, local_only: bool) -> dict[str,Any]:
    if spec.kind == "jina": raise RuntimeError("Jina is zero-shot-only until its native backward fixture is validated")
    import torch
    from peft import LoraConfig, get_peft_model
    model,tok=load_scorer(spec,device=device,local_only=local_only); names=[n for n,_ in model.named_modules()]
    missing=[x for x in spec.lora_targets if not any(n.endswith(x) for n in names)]
    if missing: raise RuntimeError(f"unsafe LoRA target modules: {missing}")
    model.config.use_cache=False; model.gradient_checkpointing_enable()
    model=get_peft_model(model,LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"], lora_dropout=LORA["lora_dropout"], bias="none", task_type="SEQ_CLS", target_modules=list(spec.lora_targets))); model.train()
    records=[x for x in read_jsonl(capsules) if str(x["qid"]) in train_qids]; groups=_hard_groups(records,answers); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); step=0
    for epoch in range(LORA["epochs"]):
        for group in groups:
            docs=[render_candidate(group["query"],group["pos"],structural=True,tokenizer=tok)]+[render_candidate(group["query"],x,structural=True,tokenizer=tok) for x in group["neg"]]
            if spec.kind=="qwen": raise RuntimeError("Qwen LoRA train requires a validated yes/no differentiable template fixture")
            batch=tok([group["query"]]*len(docs),docs,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt"); batch={k:v.to(device) for k,v in batch.items()}
            logits=model(**batch).logits.reshape(-1); loss=-torch.log_softmax(logits,dim=0)[0]/LORA["effective_batch"]; loss.backward(); step+=1
            if step%LORA["effective_batch"]==0: optimizer.step(); optimizer.zero_grad()
    if step%LORA["effective_batch"]: optimizer.step(); optimizer.zero_grad()
    output.mkdir(parents=True,exist_ok=True); model.save_pretrained(output/"adapter"); tok.save_pretrained(output/"adapter")
    report={"schema_version":SCHEMA,"model":spec.key,"groups":len(groups),"steps":step,"lora":LORA}; atomic_json(output/"train_report.json",report); return report

def score_lora(*, spec: ModelSpec, capsules: Path, adapter: Path, output: Path, qids: set[str], device: str, local_only: bool) -> dict[str, Any]:
    """Score a saved pair-encoder adapter; checkpoint reload is part of evaluation."""
    from peft import PeftModel
    model, tok = load_scorer(spec, device=device, local_only=local_only)
    model = PeftModel.from_pretrained(model, str(adapter)).to(device).eval()
    output.mkdir(parents=True,exist_ok=True)
    records=[x for x in read_jsonl(capsules) if str(x["qid"]) in qids]; rows=[]
    try:
        for pos,row in enumerate(records,1):
            docs=[render_candidate(str(row["query"]),c,structural=True,tokenizer=tok) for c in row["candidates"]]; scores=[]
            for start in range(0,len(docs),8): scores.extend(_pair_scores(spec,model,tok,str(row["query"]),docs[start:start+8],device))
            rows.append({"schema_version":SCHEMA,"qid":str(row["qid"]),"model":spec.key,"adapter":str(adapter),"scores":[{"doc_id":str(c["doc_id"]),"score":float(s)} for c,s in zip(row["candidates"],scores)]})
            if pos % 64 == 0: write_jsonl(output/"scores.jsonl",rows)
        write_jsonl(output/"scores.jsonl",rows)
    finally:
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    return {"queries":len(rows),"scores":str((output/"scores.jsonl").resolve())}

def report(*, root: Path, output: Path) -> dict[str,Any]:
    events=list(read_jsonl(root/"state.jsonl")) if (root/"state.jsonl").exists() else []
    failures=[x for x in events if x.get("state") in {"FAILED_MODEL","FAILED_OUTER","FAILED","BLOCKED"}]
    payload={"schema_version":SCHEMA,"status":"PARTIAL" if failures else "COMPLETE","events":events,"failures":failures,"jina_citation":JINA_CITATION,
      "promotion_gate":{"min_recall_gain":.002,"precision_non_decrease":True,"max_fold_recall_loss":.005}}
    output.mkdir(parents=True,exist_ok=True); atomic_json(output/"REPORT.json",payload); return payload

def overnight(args: argparse.Namespace) -> dict[str,Any]:
    paths=_paths(args); root=args.results_root; fp=_hash({k:str(v) for k,v in paths.items()}|{"schema":SCHEMA,"lora":LORA}); atomic_json(root/"run_manifest.json",{"schema_version":SCHEMA,"fingerprint":fp,"paths":{k:str(v) for k,v in paths.items()},"models":[x.__dict__ for x in MODELS]})
    _job(root,"audit",fp,lambda:audit(paths=paths,output=root/"audit"),resume=args.resume)
    _job(root,"cascade",fp,lambda:build_cascade(paths=paths,output=args.cache_root/"cascade"),resume=args.resume)
    pre_marker=_job(root,"preflight",fp,lambda:preflight(output=root/"preflight",device=args.device,local_only=args.local_only),resume=args.resume)
    pre=pre_marker["payload"]
    answers,_=retained_answers(args.train,args.preprocessing/"exclusions.json",args.preprocessing/"train_label_impact.jsonl"); folds=load_folds(args.folds); screens=defaultdict(list)
    unavailable={key for key,value in pre["models"].items() if value.get("state")=="FAILED_MODEL"}
    for outer in sorted(folds):
        capsule=args.cache_root/"cascade"/outer/"capsules"/"capsules.jsonl"
        # Selection consumes only the four inner validation folds contained in
        # outer-train.  The outer heldout fold is untouched until after winner
        # selection has been written and fingerprinted.
        for spec in MODELS:
            if spec.key in unavailable: continue
            per_inner=[]
            for inner in sorted(set(folds)-{outer}):
                sample=set(sorted(folds[inner])[:SCREEN_QUERIES]); out=root/"screen"/outer/inner/spec.key
                marker=_job(root,f"screen-{outer}-{inner}-{spec.key}",_hash([fp,outer,inner,spec.key,"inner-only"]),lambda s=spec,c=capsule,o=out,q=sample:isolated_score(spec=s,capsules=c,output=o,structural=True,device=args.device,local_only=args.local_only,qids=q),retries=2,isolate="FAILED_MODEL",resume=args.resume)
                if marker.get("state")=="FAILED_MODEL":
                    unavailable.add(spec.key); per_inner=[]; break
                metric=_prediction_metrics(out/"scores.jsonl",answers); per_inner.append({**metric,"seconds":marker["payload"]["seconds"],"queries":len(sample),"inner":inner})
            if len(per_inner)==len(folds)-1:
                screens[outer].append({"model":spec.key,"recall@5":float(np.mean([x["recall@5"] for x in per_inner])),"precision@5":float(np.mean([x["precision@5"] for x in per_inner])),"seconds_per_query":sum(x["seconds"] for x in per_inner)/sum(x["queries"] for x in per_inner),"per_inner":per_inner})
    winners={outer:choose_screen(rows) for outer,rows in screens.items() if rows}; atomic_json(root/"selection.json",{"schema_version":SCHEMA,"winners":winners,"screen":screens,"unavailable_models":sorted(unavailable)})
    # The long stages are intentionally individual jobs: a broken adapter/fold cannot stop the remaining folds.
    for outer,winner in sorted(winners.items()):
        spec=MODEL_BY_KEY[winner["model"]]; capsule=args.cache_root/"cascade"/outer/"capsules"/"capsules.jsonl"; held=set(folds[outer]); z=root/"zero_shot"/outer/spec.key
        _job(root,f"zero-{outer}",_hash([fp,outer,spec.key,"ES"]),lambda s=spec,c=capsule,o=z,q=held:isolated_score(spec=s,capsules=c,output=o,structural=True,device=args.device,local_only=args.local_only,qids=q),retries=2,isolate="FAILED_OUTER",resume=args.resume)
        _job(root,f"zero-e-{outer}",_hash([fp,outer,spec.key,"E"]),lambda s=spec,c=capsule,o=root/"zero_shot_e"/outer/spec.key,q=held:isolated_score(spec=s,capsules=c,output=o,structural=False,device=args.device,local_only=args.local_only,qids=q),retries=2,isolate="FAILED_OUTER",resume=args.resume)
        if pre["models"].get(spec.key,{}).get("state") == "ELIGIBLE":
            train=set(q for f,qs in folds.items() if f!=outer for q in qs)
            trained=_job(root,f"train-{outer}",_hash([fp,outer,spec.key,"lora"]),lambda s=spec,c=capsule,t=train,o=root/"lora"/outer/spec.key:isolated_train(spec=s,capsules=c,output=o,qids=t,device=args.device,local_only=args.local_only),isolate="FAILED_OUTER",resume=args.resume)
            if trained.get("state") != "FAILED_OUTER":
                _job(root,f"score-lora-{outer}",_hash([fp,outer,spec.key,"adapter-score"]),lambda s=spec,c=capsule,q=held,o=root/"lora_scores"/outer/spec.key,a=root/"lora"/outer/spec.key/"adapter":isolated_lora_score(spec=s,capsules=c,adapter=a,output=o,qids=q,device=args.device,local_only=args.local_only),isolate="FAILED_OUTER",resume=args.resume)
    return report(root=root,output=root)

def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("audit","build-cascade","preflight","zero-shot-screen","select","evaluate-zero-shot","train-lora","evaluate-lora","overnight","report","score-worker","score-lora-worker","train-worker")); p.add_argument("--cache-root",type=Path,default=ROOT/"cache"/"exp029_nested_lora_reranker"); p.add_argument("--results-root",type=Path,default=ROOT/"results"/"exp029_nested_lora_reranker"); p.add_argument("--candidates",type=Path,default=ROOT/"cache"/"exp022_e5_bm25_union"/"train_oof_candidates.jsonl"); p.add_argument("--sidecar",type=Path,default=ROOT/"cache"/"exp027_lambdamart_shortlist"/"provenance"/"provenance_sidecar.jsonl"); p.add_argument("--features",type=Path,default=ROOT/"cache"/"exp027_lambdamart_shortlist"/"features"); p.add_argument("--train",type=Path,default=ROOT/"public_test_dataset"/"train.json"); p.add_argument("--folds",type=Path,default=ROOT/"cache"/"cv_folds.json"); p.add_argument("--preprocessing",type=Path,default=ROOT/"cache"/"final_preprocessed_v2"); p.add_argument("--v3",type=Path,default=ROOT/"cache"/"structural_v3_e5_final_v1"); p.add_argument("--bm25-db",type=Path,default=ROOT/"cache"/"exp021_sparse"/"passage_hierarchy"/"fts5"/"bm25_v3.sqlite"); p.add_argument("--device",default="cuda"); p.add_argument("--local-only",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--model",choices=tuple(MODEL_BY_KEY)); p.add_argument("--capsules",type=Path); p.add_argument("--output",type=Path); p.add_argument("--qids-json",type=Path); p.add_argument("--adapter",type=Path); p.add_argument("--structural",action="store_true"); a=p.parse_args(argv); paths=_paths(a)
    if a.stage=="audit": result=audit(paths=paths,output=a.results_root/"audit")
    elif a.stage=="build-cascade": result=build_cascade(paths=paths,output=a.cache_root/"cascade")
    elif a.stage=="preflight": result=preflight(output=a.results_root/"preflight",device=a.device,local_only=a.local_only)
    elif a.stage=="report": result=report(root=a.results_root,output=a.results_root)
    elif a.stage=="score-worker":
        if not all((a.model,a.capsules,a.output,a.qids_json)): raise SystemExit("score worker arguments missing")
        result=score_capsules(spec=MODEL_BY_KEY[a.model],capsules=a.capsules,output=a.output,structural=a.structural,device=a.device,local_only=a.local_only,qids=_qid_set(a.qids_json))
        atomic_json(a.output/"worker_result.json",result)
    elif a.stage=="score-lora-worker":
        if not all((a.model,a.capsules,a.output,a.qids_json,a.adapter)): raise SystemExit("LoRA score worker arguments missing")
        result=score_lora(spec=MODEL_BY_KEY[a.model],capsules=a.capsules,adapter=a.adapter,output=a.output,qids=_qid_set(a.qids_json),device=a.device,local_only=a.local_only)
        atomic_json(a.output/"worker_result.json",result)
    elif a.stage=="train-worker":
        if not all((a.model,a.capsules,a.output,a.qids_json)): raise SystemExit("train worker arguments missing")
        answers,_=retained_answers(a.train,a.preprocessing/"exclusions.json",a.preprocessing/"train_label_impact.jsonl")
        result=train_lora(spec=MODEL_BY_KEY[a.model],capsules=a.capsules,train_qids=_qid_set(a.qids_json),answers=answers,output=a.output,device=a.device,local_only=a.local_only)
        atomic_json(a.output/"worker_result.json",result)
    elif a.stage in {"zero-shot-screen","select","evaluate-zero-shot","train-lora","evaluate-lora"}:
        raise SystemExit(f"{a.stage} is scheduled by overnight; run overnight --resume to preserve its frozen run manifest")
    else: result=overnight(a)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
