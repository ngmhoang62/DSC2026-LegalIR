"""EXP-108: nested atomic condition-aware residual reranker.

This experiment owns every generated artifact.  It consumes the immutable
EXP-022 E5+BM25 union as retrieval provenance, selects RRF weights only on the
outer calibration fold, renders one source-exact atomic package per candidate,
and trains a six-negative residual BGE reranker.  No held-out labels are used
before the final outer evaluation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exp030_legal_evidence_routing import canonical_answers

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp108.atomic_condition_residual.v1"
MODEL_ID = "BAAI/bge-reranker-v2-m3"
SEED = 108_2026
K = 50
RRF_K = 32
MAX_LENGTH = 512
MAX_NEGS = 6
WEIGHTS = ((.55, .45), (.65, .35), (.75, .25))
CALIBRATION = {"fold_0": "fold_4", "fold_1": "fold_0", "fold_2": "fold_1", "fold_3": "fold_2", "fold_4": "fold_3"}
TOKEN_RE = re.compile(r"(?u)\b\w+\b")
LAW_RE = re.compile(r"(?i)\b(?:luật|nghị\s*định|thông\s*tư|quyết\s*định)?\s*(?:số\s*)?\d{1,4}/\d{2,4}/[A-ZĐ-]{2,}(?:-[A-ZĐ]+)*\b")
CITATION_RE = re.compile(r"(?i)\b(?:điều|khoản|điểm)\s+(?:\d+[a-z]?|[a-zđ])\b")
NUMBER_RE = re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:%|phần\s*trăm|đồng|triệu|tỷ|ngày|tháng|năm|giờ|phút|km|m2|m²|kg|tấn)\b")
CONDITION_RE = re.compile(r"(?i)(?:^|[.;,:])\s*((?:nếu|khi|trừ|đối\s+với)\b[^.;]{3,180})")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32); std = float(arr.std())
    return (arr - arr.mean()) / (std if std > 1e-5 else 1.0)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lexical_score(query: str, text: str) -> float:
    q = Counter(tokens(query)); d = Counter(tokens(text))
    if not q or not d: return 0.0
    overlap = sum(min(count, d.get(term, 0)) for term, count in q.items())
    return overlap / math.sqrt(sum(q.values()) * sum(d.values()))


def evaluate(rankings: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], ks: Sequence[int] = (1, 5, 16, 50)) -> dict[str, float]:
    recalls = {k: [] for k in ks}; precisions = []; reciprocal = []
    for qid, gold in answers.items():
        if not gold or qid not in rankings: continue
        ranked = list(map(str, rankings[qid]))
        for k in ks: recalls[k].append(len(set(ranked[:k]) & gold) / len(gold))
        precisions.append(len(set(ranked[:5]) & gold) / 5.0)
        first = next((i for i, doc in enumerate(ranked[:5], 1) if doc in gold), None)
        reciprocal.append(1.0 / first if first else 0.0)
    return {**{f"recall@{k}": float(np.mean(value)) for k, value in recalls.items()}, "precision@5": float(np.mean(precisions)), "mrr@5": float(np.mean(reciprocal)), "evaluable_queries": len(reciprocal)}


def metric_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    return (metrics["recall@50"], metrics["recall@16"], metrics["recall@5"], metrics["mrr@5"])


def rrf_candidates(candidates: Sequence[Mapping[str, Any]], dense_weight: float, bm25_weight: float, k: int = K) -> list[dict[str, Any]]:
    ranked = []
    for original in candidates:
        source = original.get("sources", {}); dense = source.get("e5", {}); sparse = source.get("bm25", {})
        dr = dense.get("rank"); br = sparse.get("rank")
        score = (dense_weight / (RRF_K + int(dr)) if dr else 0.0) + (bm25_weight / (RRF_K + int(br)) if br else 0.0)
        ranked.append({"doc_id": str(original["doc_id"]), "dense_rank": int(dr) if dr else None,
            "dense_score": float(dense["aggregate_score"]) if dense.get("aggregate_score") is not None else None,
            "dense_evidence": list(dense.get("evidence", [])),
            "bm25_rank": int(br) if br else None, "bm25_score": float(sparse["score"]) if sparse.get("score") is not None else None,
            "rrf_score": float(score), "source_rank": int(original.get("rank", 10**9))})
    ranked.sort(key=lambda row: (-row["rrf_score"], row["source_rank"], row["doc_id"]))
    output = ranked[:k]
    for rank, row in enumerate(output, 1): row["rank"] = rank; row["stage1_score"] = row["rrf_score"]
    if len(output) != k or len({row["doc_id"] for row in output}) != k: raise ValueError("candidate K/uniqueness violation")
    return output


def parse_conditions(question: str) -> list[dict[str, Any]]:
    found: list[tuple[str, str, float]] = []
    for kind, pattern, weight in (("law", LAW_RE, 2.5), ("citation", CITATION_RE, 2.5), ("number", NUMBER_RE, 2.0), ("condition", CONDITION_RE, 1.2)):
        for match in pattern.finditer(question):
            text = (match.group(1) if match.lastindex else match.group(0)).strip()
            found.append((kind, text, weight))
    unique = []; seen = set()
    for kind, text, weight in sorted(found, key=lambda value: (question.lower().find(value[1].lower()), value[0])):
        key = re.sub(r"\s+", " ", text.lower())
        if key not in seen: unique.append({"condition_id": f"c{len(unique):02d}", "kind": kind, "text": text, "weight": weight, "confidence": "high"}); seen.add(key)
    return unique or [{"condition_id": "c00", "kind": "query", "text": question, "weight": 1.0, "confidence": "fallback"}]


def condition_coverage(conditions: Sequence[Mapping[str, Any]], text: str) -> set[str]:
    normalized = " ".join(tokens(text)); covered = set()
    for condition in conditions:
        terms = tokens(str(condition["text"])); exact = " ".join(terms) in normalized
        ratio = sum(term in normalized.split() for term in set(terms)) / max(1, len(set(terms)))
        threshold = 1.0 if condition["kind"] in {"law", "citation", "number"} else .65
        if exact or ratio >= threshold: covered.add(str(condition["condition_id"]))
    return covered


def unit_context_text(unit: Mapping[str, Any]) -> str:
    breadcrumb = " > ".join(str(value) for value in unit.get("breadcrumb", []) if value)
    return f"{breadcrumb}\n{unit.get('raw_text', '')}".strip()


def select_atomic_units(units: Sequence[Mapping[str, Any]], question: str, conditions: Sequence[Mapping[str, Any]], selector: str = "condition", evidence_budget: int = 320) -> list[dict[str, Any]]:
    scored = []
    for unit in units:
        search_text = unit_context_text(unit); lexical = lexical_score(question, search_text); dense = float(unit.get("dense_score", 0.0))
        relevance = .5 / (RRF_K + 1) if len(units) == 1 else 0.0
        scored.append({**dict(unit), "lexical_score": lexical, "dense_score": dense, "coverage": sorted(condition_coverage(conditions, search_text))})
    dense_order = {id(row): rank for rank, row in enumerate(sorted(scored, key=lambda r: (-r["dense_score"], r["node_id"])), 1)}
    lexical_order = {id(row): rank for rank, row in enumerate(sorted(scored, key=lambda r: (-r["lexical_score"], r["node_id"])), 1)}
    for row in scored: row["hybrid_score"] = .5/(RRF_K+dense_order[id(row)]) + .5/(RRF_K+lexical_order[id(row)])
    scored.sort(key=lambda row: (-row["hybrid_score"], row["token_count"], row["node_id"]))
    if not scored: return []
    if selector != "condition": return [scored[0]]
    weights = {str(c["condition_id"]): float(c["weight"]) for c in conditions}
    # Coverage-first 1/2-unit knapsack. Include the best relevance and coverage
    # units so a decisive low-ranked legal condition cannot be pruned early.
    coverage_ranked = sorted(scored, key=lambda row: (-sum(weights.get(cid, 0.0) for cid in row["coverage"]), -row["hybrid_score"], row["token_count"], row["node_id"]))
    pool=[]; seen=set()
    for row in scored[:12] + coverage_ranked[:12]:
        if row["node_id"] not in seen: pool.append(row); seen.add(row["node_id"])
    choices=[]
    for left_index,left in enumerate(pool):
        choices.append([left])
        for right in pool[left_index+1:]:
            if ranges_overlap((int(left["start"]),int(left["end"])),(int(right["start"]),int(right["end"]))): continue
            if int(left["token_count"])+int(right["token_count"]) <= evidence_budget*2:
                choices.append([left,right])
    def objective(choice: Sequence[Mapping[str,Any]]) -> tuple[float,float,float,int,str]:
        covered=set().union(*(set(row["coverage"]) for row in choice)); coverage=sum(weights.get(cid,0.0) for cid in covered)
        relevance=sum(float(row["hybrid_score"]) for row in choice); token_cost=sum(int(row["token_count"]) for row in choice)
        return coverage,relevance,-float(token_cost),-len(choice),"|".join(str(row["node_id"]) for row in choice)
    return list(max(choices,key=objective))


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def choose_negatives(candidates: Sequence[Mapping[str, Any]], gold: set[str], epoch: int, policy: str) -> list[dict[str, Any]]:
    negatives = [dict(row) for row in candidates if str(row["doc_id"]) not in gold]
    if policy == "top6": return negatives[:MAX_NEGS]
    chosen: list[dict[str, Any]] = []
    def take(pool: Sequence[Mapping[str, Any]], count: int, rotate: bool = False) -> None:
        available = [dict(row) for row in pool if str(row["doc_id"]) not in {str(x["doc_id"]) for x in chosen}]
        if rotate and available: available = available[epoch % len(available):] + available[:epoch % len(available)]
        chosen.extend(available[:count])
    take(negatives[:8], 2); take([r for r in negatives if 9 <= int(r["rank"]) <= 24], 2, True)
    deep = [r for r in negatives if 25 <= int(r["rank"]) <= 50]
    take(sorted(deep, key=lambda r: (-float(r.get("lexical_score", 0.0)), int(r["rank"]))), 1, True)
    confusers = sorted(negatives, key=lambda r: (-float(r.get("condition_score", 0.0)), int(r["rank"])))
    take(confusers, 1, True); take(negatives, MAX_NEGS-len(chosen), True)
    ids = [str(row["doc_id"]) for row in chosen]
    if len(ids) != len(set(ids)) or set(ids) & gold: raise ValueError("invalid negative group")
    return chosen[:MAX_NEGS]


def confidence_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores = np.asarray([float(row["stage1_score"]) for row in rows], dtype=np.float64)
    top = scores[:10]; probability = np.exp(zscore(top)); probability /= probability.sum()
    dense = {str(r["doc_id"]) for r in sorted(rows, key=lambda x: (x["dense_rank"] is None, x["dense_rank"] or 10**9))[:5]}
    sparse = {str(r["doc_id"]) for r in sorted(rows, key=lambda x: (x["bm25_rank"] is None, x["bm25_rank"] or 10**9))[:5]}
    return {"margin_5_6": float(scores[4]-scores[5]), "entropy_top10": float(-(probability*np.log(probability+1e-12)).sum()), "agreement_top5": len(dense & sparse)/5.0, "gap_top5_rest": float(scores[:5].mean()-scores[5:].mean())}


def residual_rank(rows: Sequence[Mapping[str, Any]], ce_scores: Mapping[str, float], alpha: float) -> list[str]:
    anchor = zscore([float(row["stage1_score"]) for row in rows]); ce = zscore([float(ce_scores[str(row["doc_id"])]) for row in rows])
    score = (1-alpha)*anchor + alpha*ce
    return [str(rows[i]["doc_id"]) for i in sorted(range(len(rows)), key=lambda i: (-float(score[i]), str(rows[i]["doc_id"])))]


@dataclass
class Paths:
    cache: Path; results: Path; train: Path; folds: Path; preprocess: Path; structural: Path; union: Path; queries: Path; e5: Path


class RunLogger:
    def __init__(self, paths: Paths, run_id: str):
        self.paths = paths; self.run_id = run_id; self.root = paths.results / "logs" / run_id; self.root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"exp108.{run_id}"); self.logger.setLevel(logging.INFO); self.logger.handlers.clear()
        handler = logging.FileHandler(self.root / "overnight.log", encoding="utf-8"); handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); self.logger.addHandler(handler)
    def event(self, stage: str, message: str, terminal: bool = True, **extra: Any) -> None:
        payload = {"stage": stage, "message": message, **extra}; self.logger.info(json.dumps(payload, ensure_ascii=False))
        with (self.root / f"{stage}.log").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"timestamp": time.time(), **payload}, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        for handler in self.logger.handlers: handler.flush()
        if terminal: print(f"[{stage}] {message}", flush=True)
    def status(self, stage: str, state: str, **extra: Any) -> None:
        write_json(self.paths.results / "RUN_STATUS.json", {"schema_version": SCHEMA, "run_id": self.run_id, "phase": stage, "state": state, "updated_at": time.time(), **extra})


def artifact(directory: Path, stage: str, report: Mapping[str, Any], files: Sequence[Path]) -> str:
    fingerprint = stable_hash(report); write_json(directory / "REPORT.json", {**dict(report), "fingerprint": fingerprint})
    all_files = [directory / "REPORT.json", *files]
    def label(path: Path) -> str:
        try: return str(path.relative_to(directory))
        except ValueError: return str(path.resolve())
    write_json(directory / "manifest.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint, "files": {label(path): sha256(path) for path in all_files}})
    write_json(directory / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint})
    return fingerprint


def load_data(paths: Paths) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, set[str]], dict[str, str]]:
    train = read_json(paths.train); folds = read_json(paths.folds)
    all_answers, _ = canonical_answers(paths.train, paths.preprocess / "exclusions.json", paths.preprocess / "train_label_impact.jsonl")
    answers = {qid: gold for qid, gold in all_answers.items() if gold}
    questions = {str(qid): str(value["question"]) for qid, value in train.items()}
    return train, folds, answers, questions


def audit_inputs(paths: Paths, log: RunLogger) -> dict[str, Any]:
    log.status("audit-inputs", "RUNNING"); train, folds, answers, _ = load_data(paths)
    required = [paths.train, paths.folds, paths.preprocess/"exclusions.json", paths.structural/"nodes.jsonl", paths.structural/"documents.jsonl", paths.structural/"manifest.json", paths.union, paths.queries/"train_queries.f32.npy", paths.queries/"train_query_ids.json", paths.e5/"embeddings.f16.npy", paths.e5/"chunk_ids.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    disk = shutil.disk_usage(paths.cache.parent).free
    membership = [str(qid) for name in sorted(folds) for qid in folds[name]]
    report = {"schema_version": SCHEMA, "status": "PASS" if not missing and len(answers)==6991 and len(train)-len(answers)==9 and len(membership)==len(set(membership))==7000 and disk>=50*1024**3 else "REJECTED_INPUT_OR_PREFLIGHT", "canonical_evaluable": len(answers), "non_evaluable": len(train)-len(answers), "fold_members": len(membership), "missing": missing, "free_disk_bytes": disk, "hashes": {"folds": sha256(paths.folds), "structural_manifest": sha256(paths.structural/"manifest.json"), "union": sha256(paths.union)}}
    out = paths.results/"audit-inputs"; artifact(out, "audit-inputs", report, []); log.status("audit-inputs", report["status"]); log.event("audit-inputs", report["status"])
    if report["status"] != "PASS": raise RuntimeError(report["status"])
    return report


def load_union(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["qid"]): list(row["candidates"]) for row in iter_jsonl(path)}


def build_candidates(paths: Paths, log: RunLogger, outers: Sequence[str]) -> dict[str, Any]:
    log.status("build-candidates", "RUNNING"); _, folds, answers, _ = load_data(paths); union = load_union(paths.union); summaries = {}
    for outer in outers:
        calibration = CALIBRATION[outer]; qids = list(map(str, folds[calibration])); local_answers = {qid: answers[qid] for qid in qids if qid in answers}
        trials = []
        for dense, sparse in WEIGHTS:
            rankings = {qid: [r["doc_id"] for r in rrf_candidates(union[qid], dense, sparse)] for qid in qids}
            metrics = evaluate(rankings, local_answers); trials.append({"dense_weight": dense, "bm25_weight": sparse, "metrics": metrics})
        winner = max(trials, key=lambda row: (*metric_key(row["metrics"]), -abs(row["dense_weight"]-.65)))
        dense, sparse = winner["dense_weight"], winner["bm25_weight"]
        output = paths.cache/"candidates"/outer/"candidates.jsonl"
        count = write_jsonl(output, ({"qid": qid, "outer": outer, "calibration_fold": calibration, "candidates": rrf_candidates(union[qid], dense, sparse)} for qid in sorted(union)))
        report = {"schema_version": SCHEMA, "status": "PASS", "outer": outer, "calibration_fold": calibration, "queries": count, "selected": winner, "trials": trials}
        out = paths.results/"candidates"/outer; artifact(out, "build-candidates", report, [output]); summaries[outer] = report; log.event("build-candidates", f"{outer}: {dense:.2f}/{sparse:.2f}, calibration R50={winner['metrics']['recall@50']:.6f}")
    report = {"schema_version": SCHEMA, "status": "PASS", "outers": summaries}; out=paths.results/"candidates"; artifact(out,"build-candidates",report,[]); log.status("build-candidates","PASS"); return report


def build_unit_inventory(paths: Paths, log: RunLogger) -> dict[str, Any]:
    log.status("build-units", "RUNNING"); output = paths.cache/"units"/"units.jsonl"; document_meta = {str(row["doc_id"]): row for row in iter_jsonl(paths.structural/"documents.jsonl")}; node_by_id = {}; selected=[]; counts=Counter()
    for row in iter_jsonl(paths.structural/"nodes.jsonl"):
        node_by_id[str(row["node_id"])] = row
        if row["kind"] in {"point","clause","article","fallback"}: selected.append(row)
    children = Counter(str(row["parent_id"]) for row in selected if row.get("parent_id"))
    def rows() -> Iterator[dict[str, Any]]:
        for row in selected:
            kind=str(row["kind"])
            if kind=="article" and children[str(row["node_id"])]>0: continue
            if kind=="clause" and children[str(row["node_id"])]>0: continue
            ancestry=[]; current=row
            for _ in range(8):
                ancestry.append(str(current.get("heading_text") or f"{current['kind']} {current.get('label','')}").strip())
                parent=current.get("parent_id")
                if not parent or str(parent) not in node_by_id: break
                current=node_by_id[str(parent)]
            text=str(row["raw_text"]); counts[kind]+=1
            yield {"node_id":str(row["node_id"]),"doc_id":str(row["doc_id"]),"kind":kind,"parent_id":row.get("parent_id"),"start":int(row["start"]),"end":int(row["end"]),"breadcrumb":list(reversed([x for x in ancestry if x])),"raw_text_hash":hashlib.sha256(text.encode()).hexdigest(),"raw_text":text,"estimated_tokens":max(1,len(tokens(text))),"document_label":str(document_meta.get(str(row["doc_id"]),{}).get("document_label",row["doc_id"]))}
    count=write_jsonl(output,rows()); report={"schema_version":SCHEMA,"status":"PASS" if count else "REJECTED_UNIT_GATE","units":count,"kinds":dict(counts)}; out=paths.results/"units"; artifact(out,"build-units",report,[output]); log.status("build-units",report["status"]); log.event("build-units",f"{report['status']}: {count:,} leaf atomic units")
    if report["status"]!="PASS": raise RuntimeError(report["status"])
    return report


def parse_chunk_span(chunk_id: str) -> tuple[int, int] | None:
    parts = chunk_id.split(":")
    try: return int(parts[-3]), int(parts[-2])
    except (ValueError, IndexError): return None


def tokenizer_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def source_window(tokenizer: Any, text: str, budget: int, query: str) -> tuple[int, int, str, bool]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) <= budget: return 0, len(text), text, False
    query_terms = [term for term in tokens(query) if len(term) >= 3]
    lower = text.lower(); anchors = [lower.find(term) for term in query_terms if lower.find(term) >= 0]
    center_char = int(np.median(anchors)) if anchors else len(text)//2
    center_token = min(range(len(offsets)), key=lambda i: abs(int(offsets[i][0])-center_char))
    left = max(0, min(center_token-budget//2, len(ids)-budget)); right = min(len(ids), left+budget)
    start = int(offsets[left][0]); end = int(offsets[right-1][1])
    return start, end, text[start:end], True


def render_package(label: str, units: Sequence[Mapping[str, Any]]) -> str:
    blocks = [f"[VĂN BẢN] {label}"]
    for index, unit in enumerate(units, 1):
        breadcrumb = " > ".join(str(value) for value in unit.get("breadcrumb", []) if value)
        blocks.extend([f"[ĐƯỜNG DẪN {index}] {breadcrumb}", f"[BẰNG CHỨNG {index}]", str(unit["selected_text"])])
    return "\n".join(blocks)


def budget_package(tokenizer: Any, question: str, chosen: Sequence[Mapping[str, Any]], evidence_budget: int = 320, metadata_budget: int = 64) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    query_tokens = tokenizer_count(tokenizer, question); available = min(evidence_budget, MAX_LENGTH-query_tokens-metadata_budget-4)
    available = max(128, available)
    selected=[]; remaining=available
    for position, raw in enumerate(chosen[:2]):
        if remaining < 24: break
        allocation = remaining if position==0 else min(remaining, max(64, remaining))
        local_start, local_end, text, fallback = source_window(tokenizer, str(raw["raw_text"]), allocation, question)
        row={key:value for key,value in raw.items() if key!="raw_text"}; row.update({"selected_text":text,"selected_start":int(raw["start"])+local_start,"selected_end":int(raw["start"])+local_end,"atomic_window_fallback":fallback,"selected_tokens":tokenizer_count(tokenizer,text)})
        selected.append(row); remaining-=row["selected_tokens"]
    if not selected: raise ValueError("empty evidence package")
    package=render_package(str(chosen[0].get("document_label",chosen[0]["doc_id"])),selected)
    pair=tokenizer(question,package,add_special_tokens=True,truncation=False)
    while len(pair["input_ids"])>MAX_LENGTH and selected:
        last=selected[-1]; excess=len(pair["input_ids"])-MAX_LENGTH; new_budget=max(16,int(last["selected_tokens"])-excess-4)
        source=str(chosen[len(selected)-1]["raw_text"]); local_start,local_end,text,_=source_window(tokenizer,source,new_budget,question)
        last.update({"selected_text":text,"selected_start":int(chosen[len(selected)-1]["start"])+local_start,"selected_end":int(chosen[len(selected)-1]["start"])+local_end,"atomic_window_fallback":True,"selected_tokens":tokenizer_count(tokenizer,text)})
        package=render_package(str(chosen[0].get("document_label",chosen[0]["doc_id"])),selected); pair=tokenizer(question,package,add_special_tokens=True,truncation=False)
        if new_budget==16 and len(pair["input_ids"])>MAX_LENGTH:
            if len(selected)==2: selected.pop()
            else: raise ValueError("metadata/query cannot fit pair budget")
            package=render_package(str(chosen[0].get("document_label",chosen[0]["doc_id"])),selected); pair=tokenizer(question,package,add_special_tokens=True,truncation=False)
    audit={"pair_tokens":len(pair["input_ids"]),"query_tokens":query_tokens,"evidence_tokens":sum(int(x["selected_tokens"]) for x in selected),"units":len(selected),"window_fallbacks":sum(bool(x["atomic_window_fallback"]) for x in selected)}
    return package, selected, audit


def load_units_by_doc(path: Path) -> dict[str, list[dict[str, Any]]]:
    output=defaultdict(list)
    for row in iter_jsonl(path): output[str(row["doc_id"])].append(row)
    return output


def attach_dense(unit: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> float:
    scores=[]; span=(int(unit["start"]),int(unit["end"]))
    for chunk in evidence:
        chunk_span=parse_chunk_span(str(chunk.get("chunk_id","")))
        if chunk_span and ranges_overlap(span,chunk_span): scores.append(float(chunk.get("chunk_score",0.0)))
    return max(scores,default=0.0)


def load_dense_index(paths: Paths) -> tuple[np.memmap, dict[str,int], dict[str,list[tuple[tuple[int,int],int]]]]:
    query_ids=list(map(str,read_json(paths.queries/"train_query_ids.json"))); query_index={qid:index for index,qid in enumerate(query_ids)}
    queries=np.load(paths.queries/"train_queries.f32.npy",mmap_mode="r")
    embeddings=np.load(paths.e5/"embeddings.f16.npy",mmap_mode="r"); by_doc=defaultdict(list)
    for index,row in enumerate(iter_jsonl(paths.e5/"chunk_ids.jsonl")):
        span=parse_chunk_span(str(row["chunk_id"]))
        if span: by_doc[str(row["doc_id"])].append((span,index))
    if len(embeddings)!=sum(len(value) for value in by_doc.values()): raise ValueError("chunk embedding/index mismatch")
    return (queries,query_index,{"embeddings":embeddings,"by_doc":dict(by_doc)})


def score_parent_chunks(query_vector: np.ndarray, doc_ids: Sequence[str], dense_index: Mapping[str,Any]) -> dict[str,list[tuple[tuple[int,int],float]]]:
    by_doc=dense_index["by_doc"]; embeddings=dense_index["embeddings"]; flat=[]; ownership=[]
    for doc in doc_ids:
        for span,index in by_doc.get(doc,[]): flat.append(index); ownership.append((doc,span))
    if not flat: return {}
    matrix=np.asarray(embeddings[np.asarray(flat,dtype=np.int64)],dtype=np.float32); scores=matrix@np.asarray(query_vector,dtype=np.float32)
    result=defaultdict(list)
    for (doc,span),score in zip(ownership,scores): result[doc].append((span,float(score)))
    return result


def atomic_dense_score(unit: Mapping[str,Any], chunks: Sequence[tuple[tuple[int,int],float]]) -> float:
    span=(int(unit["start"]),int(unit["end"])); values=[score for chunk_span,score in chunks if ranges_overlap(span,chunk_span)]
    return max(values,default=max((score for _,score in chunks),default=0.0)-1.0)


def build_evidence(paths: Paths, log: RunLogger, outer: str) -> dict[str, Any]:
    log.status("build-evidence","RUNNING",outer=outer); _,_,answers,questions=load_data(paths)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,local_files_only=True,trust_remote_code=True,use_fast=True)
    units_by_doc=load_units_by_doc(paths.cache/"units"/"units.jsonl"); queries,query_index,dense_index=load_dense_index(paths)
    source=paths.cache/"candidates"/outer/"candidates.jsonl"; output=paths.cache/"evidence"/outer/"sidecars.jsonl"
    counters=Counter(); anchor_total=anchor_covered=retainable_total=retainable_covered=gold_pairs=0; started=time.time()
    def rows() -> Iterator[dict[str,Any]]:
        nonlocal anchor_total,anchor_covered,retainable_total,retainable_covered,gold_pairs
        for index,row in enumerate(iter_jsonl(source),1):
            qid=str(row["qid"]); question=questions[qid]; conditions=parse_conditions(question); packages=[]; doc_ids=[str(candidate["doc_id"]) for candidate in row["candidates"]]
            chunk_scores=score_parent_chunks(queries[query_index[qid]],doc_ids,dense_index)
            for candidate in row["candidates"]:
                doc=str(candidate["doc_id"]); options=[]
                for unit in units_by_doc.get(doc,[]):
                    copy=dict(unit); copy["token_count"]=int(copy.get("estimated_tokens",1)); copy["dense_score"]=atomic_dense_score(copy,chunk_scores.get(doc,[])); options.append(copy)
                if not options: raise ValueError(f"no atomic units for {doc}")
                chosen=select_atomic_units(options,question,conditions,"condition"); package,selected,audit=budget_package(tokenizer,question,chosen)
                coverage=condition_coverage(conditions,package); total_weight=sum(float(c["weight"]) for c in conditions); covered_weight=sum(float(c["weight"]) for c in conditions if str(c["condition_id"]) in coverage)
                side={key:value for key,value in candidate.items() if key!="dense_evidence"}; side.update({"selected_units":[{key:value for key,value in unit.items() if key not in {"selected_text"}} for unit in selected],"rendered_hash":hashlib.sha256(package.encode()).hexdigest(),"condition_coverage":sorted(coverage),"condition_score":covered_weight/max(total_weight,1e-8),"lexical_score":max(float(x.get("lexical_score",0)) for x in selected),"pair_tokens":audit["pair_tokens"]})
                packages.append(side); counters.update({"pairs":1,"two_units":audit["units"]==2,"window_fallbacks":audit["window_fallbacks"]})
                if doc in answers.get(qid,set()):
                    gold_pairs+=1; exact=[c for c in conditions if c["kind"] in {"law","citation","number"}]; exact_ids={str(c["condition_id"]) for c in exact}; parent_coverage=set().union(*(condition_coverage(exact,unit_context_text(unit)) for unit in options)) if exact else set()
                    anchor_total+=len(exact); anchor_covered+=len(exact_ids&coverage); retainable_total+=len(parent_coverage); retainable_covered+=len(parent_coverage&coverage)
            if index%200==0: log.event("build-evidence",f"{outer}: {index}/7000 queries, ETA {(time.time()-started)/index*(7000-index)/60:.1f} min",terminal=index%1000==0)
            yield {"qid":qid,"outer":outer,"conditions":conditions,"candidates":packages}
    count=write_jsonl(output,rows()); retention=anchor_covered/anchor_total if anchor_total else 1.0; relative=retainable_covered/retainable_total if retainable_total else 1.0
    structural_ok=count==7000 and counters["pairs"]==350000; evidence_ok=relative>=.90 and retention>=.80
    status="PASS" if structural_ok and evidence_ok else "REJECTED_EVIDENCE_GATE"
    report={"schema_version":SCHEMA,"status":status,"outer":outer,"queries":count,"pairs":counters["pairs"],"two_unit_pairs":counters["two_units"],"window_fallbacks":counters["window_fallbacks"],"gold_pairs":gold_pairs,"exact_anchor_total":anchor_total,"exact_anchor_covered":anchor_covered,"exact_anchor_retention":retention,"retainable_anchor_total":retainable_total,"retainable_anchor_covered":retainable_covered,"retainable_anchor_retention":relative,"gates":{"structural":structural_ok,"absolute_anchor_retention_ge_080":retention>=.80,"relative_retainable_anchor_retention_ge_090":relative>=.90},"dense_selector":"all_in_parent_structural_chunks_e5_dot_v1","zero_silent_truncation":True}
    out=paths.results/"evidence"/outer; artifact(out,"build-evidence",report,[output]); log.status("build-evidence",status,outer=outer); log.event("build-evidence",f"{outer}: {status}, absolute={retention:.4f}, retainable={relative:.4f}")
    if status!="PASS": raise RuntimeError(status)
    return report


def load_sidecars(path: Path, qids: set[str] | None=None) -> dict[str,list[dict[str,Any]]]:
    result={}
    for row in iter_jsonl(path):
        qid=str(row["qid"])
        if qids is None or qid in qids: result[qid]=list(row["candidates"])
    return result


def load_selected_raw(paths: Paths, ids: set[str]) -> dict[str,str]:
    result={}
    for row in iter_jsonl(paths.cache/"units"/"units.jsonl"):
        if str(row["node_id"]) in ids: result[str(row["node_id"])]=str(row["raw_text"])
    if set(result)!=ids: raise ValueError(f"missing selected atomic nodes: {len(ids-set(result))}")
    return result


def materialize_package(candidate: Mapping[str,Any], raw: Mapping[str,str]) -> str:
    selected=[]
    for unit in candidate["selected_units"]:
        text=raw[str(unit["node_id"])]
        base_start=int(unit["start"]); left=int(unit["selected_start"])-base_start; right=int(unit["selected_end"])-base_start
        selected.append({**unit,"selected_text":text[left:right]})
    package=render_package(str(selected[0].get("document_label",candidate["doc_id"])),selected)
    if hashlib.sha256(package.encode()).hexdigest()!=candidate["rendered_hash"]: raise ValueError("rendered package hash mismatch")
    return package


def make_model(device: torch.device, adapter: Path | None=None) -> tuple[nn.Module,Any]:
    tokenizer=AutoTokenizer.from_pretrained(adapter or MODEL_ID,local_files_only=True,trust_remote_code=True,use_fast=True)
    base=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,num_labels=1,local_files_only=True,trust_remote_code=True)
    base.config.use_cache=False; base.gradient_checkpointing_enable()
    if adapter is not None: model=PeftModel.from_pretrained(base,adapter,is_trainable=True)
    else: model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,target_modules=["query","key","value","dense"],lora_dropout=.05,bias="none",task_type="SEQ_CLS"))
    return model.to(device),tokenizer


def score_pairs(model: nn.Module, tokenizer: Any, pairs: Sequence[tuple[str,str]], device: torch.device, batch_size: int=8, grad: bool=False) -> torch.Tensor | list[float]:
    values=[]; context=contextlib.nullcontext() if grad else torch.inference_mode()
    with context:
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size]; encoded=tokenizer([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=False,return_tensors="pt").to(device)
            if int(encoded["input_ids"].shape[1])>MAX_LENGTH: raise ValueError("silent truncation guard")
            logits=model(**encoded).logits.squeeze(-1); values.append(logits if grad else logits.float().cpu())
    joined=torch.cat(values) if values else torch.empty(0)
    return joined if grad else joined.tolist()


def score_rows(model: nn.Module, tokenizer: Any, rows: Mapping[str,Sequence[Mapping[str,Any]]], questions: Mapping[str,str], raw: Mapping[str,str], device: torch.device, log: RunLogger, stage: str) -> dict[str,dict[str,float]]:
    model.eval(); result={}; started=time.time()
    for index,qid in enumerate(sorted(rows),1):
        candidates=rows[qid]; pairs=[(questions[qid],materialize_package(row,raw)) for row in candidates]
        scores=score_pairs(model,tokenizer,pairs,device)
        result[qid]={str(row["doc_id"]):float(score) for row,score in zip(candidates,scores)}
        if index%200==0: log.event(stage,f"{index}/{len(rows)} query groups, ETA {(time.time()-started)/index*(len(rows)-index)/60:.1f} min",terminal=index%1000==0)
    return result


def selected_node_ids(rows: Mapping[str,Sequence[Mapping[str,Any]]]) -> set[str]:
    return {str(unit["node_id"]) for candidates in rows.values() for candidate in candidates for unit in candidate["selected_units"]}


def preflight(paths: Paths, log: RunLogger) -> dict[str,Any]:
    log.status("preflight","RUNNING"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); result={"schema_version":SCHEMA,"device":str(device),"model":MODEL_ID,"fp32_optimizer_preflight":"NO_CUDA"}
    if device.type=="cuda":
        model,tokenizer=make_model(device); model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5)
        sample=tokenizer(["điều kiện cấp phép"]*4,["[VĂN BẢN] kiểm tra\n[BẰNG CHỨNG] nội dung pháp luật"]*4,padding=True,return_tensors="pt").to(device)
        try:
            optimizer.zero_grad(); model(**sample).logits.mean().backward(); optimizer.step(); free,total=torch.cuda.mem_get_info(device); ratio=free/total
            result.update({"free_vram":int(free),"total_vram":int(total),"headroom":ratio,"fp32_optimizer_preflight":"PASS" if ratio>=.10 else "INSUFFICIENT_HEADROOM"})
        except torch.OutOfMemoryError: result["fp32_optimizer_preflight"]="OOM"
        finally: del model,optimizer; torch.cuda.empty_cache()
    result["status"]="PASS" if result["fp32_optimizer_preflight"]=="PASS" else "REJECTED_INPUT_OR_PREFLIGHT"; out=paths.results/"preflight"; artifact(out,"preflight",result,[]); log.status("preflight",result["status"]); log.event("preflight",f"{result['status']}: {result['fp32_optimizer_preflight']}")
    if result["status"]!="PASS": raise RuntimeError(result["status"])
    return result


def metrics_split(rankings: Mapping[str,Sequence[str]], answers: Mapping[str,set[str]]) -> dict[str,Any]:
    single={qid:gold for qid,gold in answers.items() if len(gold)==1}; multi={qid:gold for qid,gold in answers.items() if len(gold)>1}
    return {"overall":evaluate(rankings,answers),"single":evaluate(rankings,single),"multi":evaluate(rankings,multi)}


def frozen_screen(paths: Paths, log: RunLogger, outer: str) -> dict[str,Any]:
    _,folds,answers,questions=load_data(paths); calibration=CALIBRATION[outer]; qids=set(map(str,folds[calibration])); local={qid:answers[qid] for qid in qids if qid in answers}
    rows=load_sidecars(paths.cache/"evidence"/outer/"sidecars.jsonl",qids); raw=load_selected_raw(paths,selected_node_ids(rows)); device=torch.device("cuda")
    model,tokenizer=make_model(device); scores=score_rows(model,tokenizer,rows,questions,raw,device,log,"frozen-screen")
    baseline={qid:[str(r["doc_id"]) for r in value] for qid,value in rows.items()}; trials=[]
    for alpha in (0.05,.10,.15,.20):
        ranking={qid:residual_rank(value,scores[qid],alpha) for qid,value in rows.items()}; split=metrics_split(ranking,local); trials.append({"alpha":alpha,"metrics":split})
    base=metrics_split(baseline,local); winner=max(trials,key=lambda x:(x["metrics"]["overall"]["recall@5"],x["metrics"]["multi"]["recall@5"],x["metrics"]["overall"]["precision@5"],-x["alpha"]))
    passed=winner["metrics"]["overall"]["recall@5"]>=base["overall"]["recall@5"] and winner["metrics"]["overall"]["mrr@5"]>=base["overall"]["mrr@5"]-.002 and winner["metrics"]["multi"]["recall@5"]>=base["multi"]["recall@5"]
    report={"schema_version":SCHEMA,"status":"PASS" if passed else "REJECTED_SELECTOR_SCREEN","outer":outer,"calibration_fold":calibration,"baseline":base,"selected":winner,"trials":trials}; out=paths.results/"selector-screen"/outer; artifact(out,"frozen-screen",report,[]); write_json(paths.cache/"selector-screen"/outer/"scores.json",scores); log.event("frozen-screen",f"{outer}: {report['status']}, R5 {base['overall']['recall@5']:.4f} -> {winner['metrics']['overall']['recall@5']:.4f}")
    del model; torch.cuda.empty_cache()
    if not passed: raise RuntimeError(report["status"])
    return report


def residual_loss(anchor: torch.Tensor, ce: torch.Tensor, labels: torch.Tensor, alpha: float=.25, temperature: float=.08) -> torch.Tensor:
    # tanh makes the residual scale group-independent, so training and K50 inference use the identical transform.
    total=(1-alpha)*anchor+alpha*torch.tanh(ce); positives=torch.nonzero(labels>.5,as_tuple=False).flatten(); negatives=torch.nonzero(labels<=.5,as_tuple=False).flatten()
    if not len(positives) or not len(negatives): raise ValueError("group requires positives and negatives")
    losses=[]
    for index in positives: losses.append(F.cross_entropy(torch.cat([total[index:index+1],total[negatives]]).unsqueeze(0)/temperature,torch.zeros(1,dtype=torch.long,device=total.device)))
    return torch.stack(losses).mean()


def residual_rank_bounded(rows: Sequence[Mapping[str,Any]], ce_scores: Mapping[str,float], alpha: float) -> list[str]:
    anchor=zscore([float(row["stage1_score"]) for row in rows]); ce=np.tanh(np.asarray([float(ce_scores[str(row["doc_id"])]) for row in rows],dtype=np.float32)); score=(1-alpha)*anchor+alpha*ce
    return [str(rows[i]["doc_id"]) for i in sorted(range(len(rows)),key=lambda i:(-float(score[i]),str(rows[i]["doc_id"])))]


def save_checkpoint(model: nn.Module, tokenizer: Any, optimizer: Any, output: Path, epoch: int, position: int, policy: str) -> None:
    adapter=output/"adapter"; adapter.mkdir(parents=True,exist_ok=True); model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    torch.save({"optimizer":optimizer.state_dict(),"epoch":epoch,"position":position,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],"numpy_rng":np.random.get_state(),"python_rng":random.getstate()},output/"training_state.pt")
    write_json(output/"checkpoint.json",{"schema_version":SCHEMA,"epoch":epoch,"position":position,"policy":policy})


def train_model(paths: Paths, log: RunLogger, rows: Mapping[str,Sequence[Mapping[str,Any]]], qids: Sequence[str], answers: Mapping[str,set[str]], questions: Mapping[str,str], policy: str, output: Path, resume: bool=True, epochs: int=2) -> dict[str,Any]:
    device=torch.device("cuda"); raw=load_selected_raw(paths,selected_node_ids({qid:rows[qid] for qid in qids if qid in rows})); checkpoint=output/"checkpoint.json"; start_epoch=start_position=0
    if resume and checkpoint.exists():
        state_meta=read_json(checkpoint); model,tokenizer=make_model(device,output/"adapter"); state=torch.load(output/"training_state.pt",map_location="cpu",weights_only=False); start_epoch=int(state_meta["epoch"]); start_position=int(state_meta["position"])
    else: model,tokenizer=make_model(device); state=None
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=.01)
    if state is not None: optimizer.load_state_dict(state["optimizer"]); torch.set_rng_state(state["torch_rng"]); random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    ordered=sorted(map(str,qids)); steps=skipped=0; started=time.time(); model.train()
    for epoch in range(start_epoch,epochs):
        epoch_order=ordered.copy(); random.Random(SEED+epoch).shuffle(epoch_order); begin=start_position if epoch==start_epoch else 0
        for position in range(begin,len(epoch_order)):
            qid=epoch_order[position]; candidates=rows[qid]; gold=answers.get(qid,set()); positives=[dict(row) for row in candidates if str(row["doc_id"]) in gold]
            if not positives: skipped+=1; continue
            negatives=choose_negatives(candidates,gold,epoch,policy); group=positives+negatives; pairs=[(questions[qid],materialize_package(row,raw)) for row in group]
            logits=score_pairs(model,tokenizer,pairs,device,batch_size=max(1,min(8,len(pairs))),grad=True); labels=torch.tensor([1.0]*len(positives)+[0.0]*len(negatives),device=device)
            anchor=torch.tensor(zscore([float(row["stage1_score"]) for row in group]),device=device); loss=residual_loss(anchor,logits,labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); steps+=1
            if (position+1)%250==0:
                save_checkpoint(model,tokenizer,optimizer,output,epoch,position+1,policy); free,total=torch.cuda.mem_get_info(device); log.event("train",f"{policy} ep {epoch+1}/{epochs} {position+1}/{len(epoch_order)}, loss={loss.item():.4f}, free={free/1024**3:.2f}GB, ETA={(time.time()-started)/max(1,steps)*(len(epoch_order)*(epochs-epoch)-position-1)/60:.1f} min")
        save_checkpoint(model,tokenizer,optimizer,output,epoch+1,0,policy); start_position=0
    return {"steps":steps,"skipped_no_positive":skipped,"epochs":epochs,"policy":policy,"residual_transform":"tanh_raw_ce_v1"}


def score_adapter(paths: Paths, log: RunLogger, adapter: Path, rows: Mapping[str,Sequence[Mapping[str,Any]]], questions: Mapping[str,str], stage: str) -> dict[str,dict[str,float]]:
    device=torch.device("cuda"); raw=load_selected_raw(paths,selected_node_ids(rows)); model,tokenizer=make_model(device,adapter); model.eval(); result=score_rows(model,tokenizer,rows,questions,raw,device,log,stage); del model; torch.cuda.empty_cache(); return result


def choose_confidence_policy(rows: Mapping[str,Sequence[Mapping[str,Any]]], scores: Mapping[str,Mapping[str,float]], answers: Mapping[str,set[str]]) -> dict[str,Any]:
    signals={qid:confidence_features(value) for qid,value in rows.items()}; names=("margin_5_6","entropy_top10","agreement_top5","gap_top5_rest")
    matrix=np.asarray([[signals[qid][name] for name in names] for qid in sorted(rows)],dtype=np.float64); normalized=np.column_stack([zscore(matrix[:,i]) for i in range(matrix.shape[1])]); confidence=normalized[:,0]-normalized[:,1]+normalized[:,2]+normalized[:,3]; by_qid={qid:float(value) for qid,value in zip(sorted(rows),confidence)}
    trials=[]
    for low,high in ((0,.15),(0,.20),(.05,.20),(.05,.30),(.10,.30)):
        for quantile in (.50,.70):
            threshold=float(np.quantile(confidence,quantile)); ranking={qid:residual_rank_bounded(value,scores[qid],low if by_qid[qid]>=threshold else high) for qid,value in rows.items()}; trials.append({"kind":"confidence","alpha_low":low,"alpha_high":high,"quantile":quantile,"threshold":threshold,"metrics":metrics_split(ranking,answers)})
    for alpha in (.15,.20,.25,.30):
        ranking={qid:residual_rank_bounded(value,scores[qid],alpha) for qid,value in rows.items()}; trials.append({"kind":"global","alpha":alpha,"metrics":metrics_split(ranking,answers)})
    winner=max(trials,key=lambda x:(x["metrics"]["overall"]["recall@5"],x["metrics"]["multi"]["recall@5"],x["metrics"]["overall"]["precision@5"],x["metrics"]["overall"]["mrr@5"],x["metrics"]["overall"]["recall@1"],-float(x.get("alpha_high",x.get("alpha",0)))))
    return {"selected":winner,"trials":trials,"confidence_features":names,"confidence_by_qid":by_qid}


def apply_policy(rows: Mapping[str,Sequence[Mapping[str,Any]]], scores: Mapping[str,Mapping[str,float]], policy: Mapping[str,Any]) -> tuple[dict[str,list[str]],dict[str,float]]:
    if policy["kind"]=="global":
        alphas={qid:float(policy["alpha"]) for qid in rows}
    else:
        raw={qid:confidence_features(value) for qid,value in rows.items()}; names=("margin_5_6","entropy_top10","agreement_top5","gap_top5_rest")
        matrix=np.asarray([[raw[qid][name] for name in names] for qid in sorted(rows)]); normalized=np.column_stack([zscore(matrix[:,i]) for i in range(4)]); values=normalized[:,0]-normalized[:,1]+normalized[:,2]+normalized[:,3]; threshold=float(np.quantile(values,float(policy["quantile"])))
        alphas={qid:float(policy["alpha_low"] if value>=threshold else policy["alpha_high"]) for qid,value in zip(sorted(rows),values)}
    return {qid:residual_rank_bounded(value,scores[qid],alphas[qid]) for qid,value in rows.items()},alphas


def calibrate_outer(paths: Paths, log: RunLogger, outer: str, resume: bool=True) -> dict[str,Any]:
    _,folds,answers,questions=load_data(paths); calibration=CALIBRATION[outer]; calibration_qids=set(map(str,folds[calibration])); train_qids=[str(qid) for name in sorted(folds) if name not in {outer,calibration} for qid in folds[name]]
    rows=load_sidecars(paths.cache/"evidence"/outer/"sidecars.jsonl"); calibration_rows={qid:rows[qid] for qid in calibration_qids}; local_answers={qid:answers[qid] for qid in calibration_qids if qid in answers}; trials=[]
    for policy in ("top6","curriculum"):
        model_out=paths.cache/"calibration"/outer/policy; fit=train_model(paths,log,rows,train_qids,answers,questions,policy,model_out,resume)
        score_file=model_out/"scores.json"
        if resume and score_file.exists(): scores=read_json(score_file)
        else: scores=score_adapter(paths,log,model_out/"adapter",calibration_rows,questions,f"score-{policy}"); write_json(score_file,scores)
        choice=choose_confidence_policy(calibration_rows,scores,local_answers); trials.append({"negative_policy":policy,"fit":fit,"fusion":choice["selected"],"fusion_trials":choice["trials"]})
    winner=max(trials,key=lambda x:(x["fusion"]["metrics"]["overall"]["recall@5"],x["fusion"]["metrics"]["multi"]["recall@5"],x["fusion"]["metrics"]["overall"]["precision@5"],x["fusion"]["metrics"]["overall"]["recall@1"],x["fusion"]["metrics"]["overall"]["mrr@5"],x["negative_policy"]=="top6"))
    report={"schema_version":SCHEMA,"status":"PASS","outer":outer,"calibration_fold":calibration,"selected_negative_policy":winner["negative_policy"],"selected_fusion":winner["fusion"],"trials":trials}; out=paths.results/"calibration"/outer; artifact(out,"calibrate-outer",report,[]); log.event("calibrate",f"{outer}: {winner['negative_policy']} + {winner['fusion']['kind']}, R5={winner['fusion']['metrics']['overall']['recall@5']:.4f}"); return report


def train_outer(paths: Paths, log: RunLogger, outer: str, resume: bool=True) -> dict[str,Any]:
    _,folds,answers,questions=load_data(paths); qids=[str(qid) for name in sorted(folds) if name!=outer for qid in folds[name]]; rows=load_sidecars(paths.cache/"evidence"/outer/"sidecars.jsonl"); calibration=read_json(paths.results/"calibration"/outer/"REPORT.json"); policy=str(calibration["selected_negative_policy"])
    output=paths.cache/"outer-models"/outer; fit=train_model(paths,log,rows,qids,answers,questions,policy,output,resume); report={"schema_version":SCHEMA,"status":"PASS","outer":outer,"negative_policy":policy,"fusion":calibration["selected_fusion"],"fit":fit}; out=paths.results/"train"/outer; artifact(out,"train-outer",report,[]); return report


def evaluate_outer(paths: Paths, log: RunLogger, outer: str, resume: bool=True) -> dict[str,Any]:
    _,folds,answers,questions=load_data(paths); qids=set(map(str,folds[outer])); local={qid:answers[qid] for qid in qids if qid in answers}; rows=load_sidecars(paths.cache/"evidence"/outer/"sidecars.jsonl",qids); score_file=paths.cache/"outer-scores"/f"{outer}.json"
    if resume and score_file.exists(): scores=read_json(score_file)
    else: scores=score_adapter(paths,log,paths.cache/"outer-models"/outer/"adapter",rows,questions,f"score-{outer}"); write_json(score_file,scores)
    training=read_json(paths.results/"train"/outer/"REPORT.json"); prediction,alphas=apply_policy(rows,scores,training["fusion"]); baseline={qid:[str(row["doc_id"]) for row in value] for qid,value in rows.items()}; final=metrics_split(prediction,local); base=metrics_split(baseline,local); delta=final["overall"]["recall@5"]-base["overall"]["recall@5"]
    passed=final["overall"]["recall@5"]>=.95 and delta>=.01 and final["multi"]["recall@5"]>=.78 and final["overall"]["recall@1"]>=base["overall"]["recall@1"]-.005 and final["overall"]["mrr@5"]>=base["overall"]["mrr@5"]-.005 and final["overall"]["precision@5"]>=base["overall"]["precision@5"]
    transitions=Counter()
    for qid,gold in local.items():
        before=len(set(baseline[qid][:5])&gold)/len(gold); after=len(set(prediction[qid][:5])&gold)/len(gold); transitions["rescued" if after>before else "harmed" if after<before else "unchanged"]+=1
    report={"schema_version":SCHEMA,"status":"PASS_TARGET_096" if final["overall"]["recall@5"]>=.96 and passed else "PASS_FOLD_GATE" if passed else "REJECTED_FOLD0_AMBITIOUS_GATE" if outer=="fold_0" else "COMPLETE","outer":outer,"metrics":final,"baseline":base,"delta_recall@5":delta,"transitions":dict(transitions),"alpha_distribution":dict(Counter(map(str,alphas.values()))),"predictions":prediction}
    out=paths.results/"evaluation"/outer; artifact(out,"evaluate-outer",report,[]); log.event("evaluate",f"{outer}: {report['status']}, R5={final['overall']['recall@5']:.4f}, delta={delta:+.4f}"); return report


def evaluate_oof(paths: Paths, log: RunLogger) -> dict[str,Any]:
    _,folds,answers,_=load_data(paths); prediction={}; baseline={}; per_fold={}
    for outer in sorted(folds):
        report=read_json(paths.results/"evaluation"/outer/"REPORT.json"); prediction.update(report["predictions"]); qids=set(map(str,folds[outer])); rows=load_sidecars(paths.cache/"evidence"/outer/"sidecars.jsonl",qids); baseline.update({qid:[str(row["doc_id"]) for row in value] for qid,value in rows.items()}); per_fold[outer]={"metrics":report["metrics"],"baseline":report["baseline"],"delta_recall@5":report["delta_recall@5"]}
    final=metrics_split(prediction,answers); base=metrics_split(baseline,answers); delta=final["overall"]["recall@5"]-base["overall"]["recall@5"]; deltas=[x["delta_recall@5"] for x in per_fold.values()]
    passed=final["overall"]["recall@5"]>=.95 and delta>=.01 and sum(x>0 for x in deltas)>=4 and min(deltas)>=-.002 and final["overall"]["precision@5"]>=base["overall"]["precision@5"] and final["multi"]["recall@5"]>=.78 and final["overall"]["mrr@5"]>=base["overall"]["mrr@5"]-.002
    report={"schema_version":SCHEMA,"status":"PASS_PROMOTION_GATE" if passed else "REJECTED_OOF_GATE","metrics":final,"baseline":base,"delta_recall@5":delta,"folds":per_fold}; out=paths.results/"oof"; artifact(out,"evaluate-oof",report,[]); log.event("oof",f"{report['status']}: R5={final['overall']['recall@5']:.4f}, delta={delta:+.4f}"); return report


def stage_complete(directory: Path) -> bool:
    report_path=directory/"REPORT.json"
    if not (directory/"_SUCCESS.json").exists() or not report_path.exists(): return False
    status=str(read_json(report_path).get("status",""))
    return status=="PASS" or status.startswith("PASS_") or status in {"COMPLETE","PASS_PROMOTION_GATE"}


def run_outer(paths: Paths, log: RunLogger, outer: str, resume: bool) -> dict[str,Any]:
    if not (resume and stage_complete(paths.results/"evidence"/outer)): build_evidence(paths,log,outer)
    if not (resume and stage_complete(paths.results/"selector-screen"/outer)): frozen_screen(paths,log,outer)
    if not (resume and stage_complete(paths.results/"calibration"/outer)): calibrate_outer(paths,log,outer,resume)
    if not (resume and stage_complete(paths.results/"train"/outer)): train_outer(paths,log,outer,resume)
    if resume and stage_complete(paths.results/"evaluation"/outer): return read_json(paths.results/"evaluation"/outer/"REPORT.json")
    return evaluate_outer(paths,log,outer,resume)


def overnight(paths: Paths, log: RunLogger, resume: bool) -> dict[str,Any]:
    log.event("overnight",f"run_id={log.run_id}; detailed log={log.root/'overnight.log'}")
    if not (resume and stage_complete(paths.results/"audit-inputs")): audit_inputs(paths,log)
    if not (resume and stage_complete(paths.results/"units")): build_unit_inventory(paths,log)
    if not (resume and stage_complete(paths.results/"candidates")): build_candidates(paths,log,tuple(CALIBRATION))
    if not (resume and stage_complete(paths.results/"preflight")): preflight(paths,log)
    fold0=run_outer(paths,log,"fold_0",resume)
    if fold0["status"] not in {"PASS_FOLD_GATE","PASS_TARGET_096"}:
        log.status("overnight",fold0["status"],outer="fold_0"); return fold0
    for outer in ("fold_1","fold_2","fold_3","fold_4"): run_outer(paths,log,outer,resume)
    report=evaluate_oof(paths,log); log.status("overnight",report["status"]); return report


def default_paths(args: argparse.Namespace) -> Paths:
    return Paths(cache=args.cache_root,results=args.results_root,train=args.train,folds=args.folds,preprocess=args.preprocess,structural=args.structural,union=args.union,queries=args.queries,e5=args.e5)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=("overnight","stage","status")); parser.add_argument("stage_name",nargs="?"); parser.add_argument("--outer",choices=tuple(CALIBRATION)); parser.add_argument("--resume",action="store_true"); parser.add_argument("--run-id",default=time.strftime("%Y%m%d-%H%M%S")); parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--cache-root",type=Path,default=ROOT/"cache"/"exp108_atomic_condition_reranker"); parser.add_argument("--results-root",type=Path,default=ROOT/"results"/"exp108_atomic_condition_reranker"); parser.add_argument("--train",type=Path,default=ROOT/"public_test_dataset"/"train.json"); parser.add_argument("--folds",type=Path,default=ROOT/"cache"/"cv_folds.json"); parser.add_argument("--preprocess",type=Path,default=ROOT/"cache"/"final_preprocessed_v2"); parser.add_argument("--structural",type=Path,default=ROOT/"cache"/"structural_v3_e5_final_v1"); parser.add_argument("--union",type=Path,default=ROOT/"cache"/"exp022_e5_bm25_union"/"train_oof_candidates.jsonl"); parser.add_argument("--queries",type=Path,default=ROOT/"cache"/"exp021_e5_dense_candidates"/"query_embeddings"); parser.add_argument("--e5",type=Path,default=ROOT/"cache"/"e5_final_v1")
    args=parser.parse_args(); paths=default_paths(args); set_seed(); log=RunLogger(paths,args.run_id)
    if args.command=="status":
        value=read_json(paths.results/"RUN_STATUS.json") if (paths.results/"RUN_STATUS.json").exists() else {"state":"NOT_STARTED"}; print(json.dumps(value,ensure_ascii=False,indent=2)); return 0
    if args.dry_run: print(json.dumps({"schema":SCHEMA,"paths":{key:str(value) for key,value in vars(paths).items()},"outer":args.outer},ensure_ascii=False,indent=2)); return 0
    try:
        if args.command=="overnight": report=overnight(paths,log,args.resume)
        else:
            name=args.stage_name
            if name=="audit-inputs": report=audit_inputs(paths,log)
            elif name=="build-units": report=build_unit_inventory(paths,log)
            elif name=="build-candidates": report=build_candidates(paths,log,(args.outer,) if args.outer else tuple(CALIBRATION))
            elif name=="preflight": report=preflight(paths,log)
            elif name=="build-evidence" and args.outer: report=build_evidence(paths,log,args.outer)
            elif name=="frozen-screen" and args.outer: report=frozen_screen(paths,log,args.outer)
            elif name=="calibrate" and args.outer: report=calibrate_outer(paths,log,args.outer,args.resume)
            elif name=="train" and args.outer: report=train_outer(paths,log,args.outer,args.resume)
            elif name=="evaluate" and args.outer: report=evaluate_outer(paths,log,args.outer,args.resume)
            elif name=="evaluate-oof": report=evaluate_oof(paths,log)
            else: parser.error("unknown stage or missing --outer")
        return 0 if not str(report.get("status","")).startswith("REJECTED") else 2
    except KeyboardInterrupt:
        log.status(args.stage_name or args.command,"INTERRUPTED"); return 130
    except Exception as exc:
        log.logger.exception("fatal pipeline error"); log.status(args.stage_name or args.command,"FAILED",error=repr(exc)); print(f"FAILED: {exc}; log={log.root/'overnight.log'}",file=sys.stderr,flush=True); return 1


if __name__=="__main__": raise SystemExit(main())
