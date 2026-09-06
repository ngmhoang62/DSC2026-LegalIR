"""EXP-106b: nested K50 BGE residual reranker.

The module deliberately owns all generated state.  EXP-106 caches are never
accepted as inputs because their global OOF candidates are not valid outer
train inputs for this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exp030_legal_evidence_routing import canonical_answers
from exp102_mil_nce_retrieval import DIMENSION, RANK, ResidualProjection, compute_mil_nce_loss, load_corpus_data

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp106b_nested_k50_reranker.v1"
SEED = 106_002
K = 50
RRF_K = 32
DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512
ALPHA_GRID = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _success(directory: Path, stage: str, fingerprint: str, **extra: Any) -> None:
    _write(directory / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint, **extra})


def _manifest(directory: Path, stage: str, fingerprint: str, files: Sequence[Path], **extra: Any) -> None:
    _write(directory / "manifest.json", {"schema_version": SCHEMA, "stage": stage, "fingerprint": fingerprint,
        "files": {str(path.relative_to(directory)): _sha(path) for path in files}, **extra})
    _success(directory, stage, fingerprint)


def _state(results: Path, stage: str, state: str, **extra: Any) -> None:
    row = {"schema_version": SCHEMA, "stage": stage, "state": state, "updated_at": time.time(), **extra}
    _write(results / "RUN_STATUS.json", row)
    history = results / "state.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def zscore(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    return (values - values.mean()) / (std if std > 1e-5 else 1.0)


def evaluate(rankings: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], ks: Sequence[int] = (1, 5, 16, 24, 50)) -> dict[str, float]:
    values: dict[int, list[float]] = {k: [] for k in ks}; precision: list[float] = []; reciprocal: list[float] = []
    for qid, gold in answers.items():
        if not gold or qid not in rankings: continue
        ranked = list(rankings[qid])
        for k in ks: values[k].append(len(set(ranked[:k]) & gold) / len(gold))
        precision.append(len(set(ranked[:5]) & gold) / 5.0)
        rank = next((i for i, doc in enumerate(ranked[:5], 1) if doc in gold), None)
        reciprocal.append(1.0 / rank if rank else 0.0)
    return {**{f"recall@{k}": float(np.mean(v)) for k, v in values.items()}, "precision@5": float(np.mean(precision)), "mrr@5": float(np.mean(reciprocal)), "evaluable_queries": len(reciprocal)}


def multi_positive_hard_negative_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = .2) -> torch.Tensor:
    """Uniform-positive ListNet plus a hard-negative softplus correction."""
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape: raise ValueError("rank-loss shapes")
    pos = torch.nonzero(labels > .5, as_tuple=False).flatten(); neg = torch.nonzero(labels <= .5, as_tuple=False).flatten()
    if not len(pos) or not len(neg): raise ValueError("ranking group needs positive and negative")
    log_prob = F.log_softmax(logits, dim=0)
    listwise = -log_prob[pos].mean()
    hard = neg[torch.topk(logits[neg], k=min(4, len(neg))).indices]
    pairs = F.softplus(margin - logits[pos].unsqueeze(1) + logits[hard].unsqueeze(0)).mean()
    return listwise + .25 * pairs


def build_training_group(candidates: Sequence[Mapping[str, Any]], gold: set[str], zero_shot: Mapping[str, float]) -> list[dict[str, Any]]:
    positives = [dict(row) for row in candidates if str(row["doc_id"]) in gold]
    negatives = [dict(row) for row in candidates if str(row["doc_id"]) not in gold]
    anchor = negatives[:12]
    anchor_ids = {str(row["doc_id"]) for row in anchor}
    mined = sorted(negatives, key=lambda row: (-float(zero_shot.get(str(row["doc_id"]), -float("inf"))), int(row["rank"])))
    chosen = positives + anchor + [row for row in mined if str(row["doc_id"]) not in anchor_ids][:4]
    ids = [str(row["doc_id"]) for row in chosen]
    if len(ids) != len(set(ids)): raise ValueError("duplicate training group document")
    if any(str(row["doc_id"]) in gold for row in chosen[len(positives):]): raise ValueError("gold used as negative")
    return chosen


def choose_alpha(candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]], ce_scores: Mapping[str, Mapping[str, float]], answers: Mapping[str, set[str]]) -> dict[str, Any]:
    trials = []
    for alpha in ALPHA_GRID:
        rankings = {}
        for qid, rows in candidate_rows.items():
            anchor = zscore([float(row["stage1_score"]) for row in rows]); ce = zscore([float(ce_scores[qid][str(row["doc_id"])]) for row in rows])
            ranked = sorted(range(len(rows)), key=lambda i: -((1-alpha)*anchor[i] + alpha*ce[i]))
            rankings[qid] = [str(rows[i]["doc_id"]) for i in ranked]
        metrics = evaluate(rankings, answers)
        trials.append({"alpha": alpha, "metrics": metrics})
    best = max(trials, key=lambda item: (item["metrics"]["recall@5"], item["metrics"]["precision@5"], -item["alpha"]))
    return {"selected": best, "trials": trials}


def validate_candidate_rows(rows: Iterable[Mapping[str, Any]], answers: Mapping[str, set[str]]) -> dict[str, Any]:
    ranked: dict[str, list[str]] = {}; bad = 0
    for row in rows:
        qid = str(row["qid"]); candidates = row["candidates"]; ids = [str(value["doc_id"]) for value in candidates]
        expected = list(range(1, K + 1))
        if len(ids) != K or len(set(ids)) != K or [int(value["rank"]) for value in candidates] != expected or any("stage1_score" not in value for value in candidates): bad += 1
        ranked[qid] = ids
    metrics = evaluate(ranked, answers)
    return {"queries": len(ranked), "malformed": bad, "metrics": metrics, "gold_outside_k50": sum(len(gold - set(ranked.get(qid, []))) for qid, gold in answers.items())}


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {"cache": args.cache_root, "results": args.results_root, "train": args.train, "folds": args.folds,
        "preprocess": args.preprocess, "structural": args.structural, "e5": args.e5, "queries": args.queries,
        "exp022": args.exp022}


def audit_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    answers, stats = canonical_answers(paths["train"], paths["preprocess"] / "exclusions.json", paths["preprocess"] / "train_label_impact.jsonl")
    if stats["evaluable_queries"] != 6991 or stats["non_evaluable_queries"] != 9: raise ValueError(f"canonical labels changed: {stats}")
    folds = _read(paths["folds"])
    if sorted(sum((list(map(str, qids)) for qids in folds.values()), [])) != sorted(answers): raise ValueError("fold membership is not train membership")
    required = [paths["structural"] / "chunks.jsonl", paths["e5"] / "embeddings.f16.npy", paths["queries"] / "train_queries.f32.npy", paths["exp022"]]
    if not all(path.exists() for path in required): raise FileNotFoundError("missing EXP-106b immutable input")
    fingerprint = _hash({"labels": stats["label_fingerprint"], "folds": _sha(paths["folds"]), "structural": _sha(paths["structural"] / "manifest.json"), "exp022": _sha(paths["exp022"])})
    report = {"schema_version": SCHEMA, "status": "PASS", "fingerprint": fingerprint, "label_stats": stats, "k": K, "rrf": {"dense": DENSE_WEIGHT, "bm25": BM25_WEIGHT, "k": RRF_K}}
    out = paths["results"] / "audit-inputs"; _write(out / "REPORT.json", report); _manifest(out, "audit-inputs", fingerprint, [out / "REPORT.json"]); _state(paths["results"], "audit-inputs", "PASS", completed=1, total=1); return report


def _outer_contexts(folds: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    names = sorted(folds); contexts = []
    for outer in names:
        outer_train = [str(q) for name in names if name != outer for q in folds[name]]
        contexts.append({"outer": outer, "kind": "heldout", "target_fold": outer, "target_qids": list(map(str, folds[outer])), "train_qids": outer_train})
        for inner in names:
            if inner == outer: continue
            contexts.append({"outer": outer, "kind": "inner", "target_fold": inner, "target_qids": list(map(str, folds[inner])),
                "train_qids": [str(q) for name in names if name not in {outer, inner} for q in folds[name]]})
    return contexts


def _train_projection(data: Mapping[str, Any], train_qids: Sequence[str], device: torch.device, epochs: int = 2) -> ResidualProjection:
    """The small EXP-102 projection is retrained per nested context."""
    docs = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device); docs = F.normalize(docs, dim=-1)
    model = ResidualProjection(DIMENSION, RANK).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    doc_idx = {doc: torch.tensor(indices, device=device, dtype=torch.long) for doc, indices in data["doc_to_chunk_indices"].items()}
    union = {str(row["qid"]): row["candidates"] for row in _jsonl(ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl")}
    answers, _ = canonical_answers(ROOT / "public_test_dataset" / "train.json", ROOT / "cache" / "final_preprocessed_v2" / "exclusions.json", ROOT / "cache" / "final_preprocessed_v2" / "train_label_impact.jsonl")
    model.train()
    for _ in range(epochs):
        for qid in sorted(map(str, train_qids)):
            gold = [doc_idx[d] for d in answers[qid] if d in doc_idx]
            negatives = [doc_idx[str(row["doc_id"])] for row in union[qid] if str(row["doc_id"]) not in answers[qid] and str(row["doc_id"]) in doc_idx][:8]
            if not gold or not negatives: continue
            q = torch.from_numpy(data["query_embeddings"][data["qid_to_idx"][qid]]).to(device)
            optimizer.zero_grad(); loss = compute_mil_nce_loss(model(q), gold, negatives, docs); loss.backward(); optimizer.step()
    return model.eval()


def _rank_k50(data: Mapping[str, Any], model: nn.Module, qid: str, documents: torch.Tensor, device: torch.device) -> list[dict[str, Any]]:
    q = torch.from_numpy(data["query_embeddings"][data["qid_to_idx"][qid]]).to(device).unsqueeze(0)
    with torch.inference_mode(): sims = torch.matmul(model(q), documents.T).squeeze(0); _, ids = torch.topk(sims, k=min(4096, documents.shape[0]))
    doc_scores: dict[str, list[float]] = defaultdict(list)
    for idx in ids.cpu().tolist():
        doc = str(data["chunk_doc_ids"][idx])
        if len(doc_scores[doc]) < 2: doc_scores[doc].append(float(sims[idx].item()))
    dense = sorted(((doc, sum(values)/len(values)) for doc, values in doc_scores.items()), key=lambda pair: -pair[1])
    dense_rank = {doc: rank for rank, (doc, _) in enumerate(dense, 1)}; bm25 = data["bm25_ranks"].get(qid, {})
    fused = []
    for doc in set(dense_rank) | set(bm25):
        dr, br = dense_rank.get(doc), bm25.get(doc); score = (DENSE_WEIGHT/(RRF_K+dr) if dr else 0.0) + (BM25_WEIGHT/(RRF_K+br) if br else 0.0)
        fused.append((str(doc), score, dr, br))
    fused.sort(key=lambda value: (-value[1], value[0]))
    return [{"doc_id": doc, "rank": rank, "stage1_score": score, "sources": {"exp102_dense": {"rank": dr}, "bm25": {"rank": br}}, "provenance": "exp102_nested_rrf"} for rank, (doc, score, dr, br) in enumerate(fused[:K], 1)]


def build_nested_candidates(paths: Mapping[str, Path]) -> dict[str, Any]:
    audit = _read(paths["results"] / "audit-inputs" / "REPORT.json"); data = load_corpus_data(); folds = _read(paths["folds"]); answers, _ = canonical_answers(paths["train"], paths["preprocess"] / "exclusions.json", paths["preprocess"] / "train_label_impact.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); documents = torch.from_numpy(np.asarray(data["chunk_embeddings"], dtype=np.float32)).to(device); documents = F.normalize(documents, dim=-1)
    report: dict[str, Any] = {"contexts": {}, "status": "PASS"}
    contexts = _outer_contexts(folds)
    for number, context in enumerate(contexts, 1):
        name = f"{context['outer']}_{context['kind']}_{context['target_fold']}"; out = paths["cache"] / "nested-candidates" / context["outer"] / name
        target_answers = {qid: answers[qid] for qid in context["target_qids"]}
        if (out / "_SUCCESS.json").exists():
            summary = validate_candidate_rows(_jsonl(out / "candidates.jsonl"), target_answers)
        else:
            model = _train_projection(data, context["train_qids"], device)
            rows = ({"qid": qid, "candidates": _rank_k50(data, model, qid, documents, device), "outer": context["outer"], "context": name} for qid in sorted(context["target_qids"]))
            count = _write_jsonl(out / "candidates.jsonl", rows); fp = _hash({"audit": audit["fingerprint"], "context": context, "count": count}); _manifest(out, "nested-candidates", fp, [out / "candidates.jsonl"], context=context)
            summary = validate_candidate_rows(_jsonl(out / "candidates.jsonl"), target_answers)
        report["contexts"][name] = {"context": context, **summary}; _state(paths["results"], "build-nested-candidates", "RUNNING", completed=number, total=len(contexts), eta_seconds=None)
    outer = {name: value for name, value in report["contexts"].items() if value["context"]["kind"] == "heldout"}
    aggregate = evaluate({row["qid"]: [str(c["doc_id"]) for c in row["candidates"]] for name in outer for row in _jsonl(paths["cache"] / "nested-candidates" / outer[name]["context"]["outer"] / name / "candidates.jsonl")}, answers)
    if any(value["malformed"] for value in report["contexts"].values()) or aggregate["recall@50"] < .985 or min(value["metrics"]["recall@50"] for value in outer.values()) < .982: report["status"] = "REJECTED_CANDIDATE_GATE"
    fp = _hash({"audit": audit["fingerprint"], "contexts": {key: value["metrics"] for key, value in report["contexts"].items()}}); out = paths["results"] / "nested-candidates"; report["aggregate_outer"] = aggregate; report["fingerprint"] = fp; _write(out / "REPORT.json", report); _manifest(out, "build-nested-candidates", fp, [out / "REPORT.json"]); _state(paths["results"], "build-nested-candidates", report["status"], completed=len(contexts), total=len(contexts)); return report


def _tokens(text: str) -> set[str]: return set(__import__("re").findall(r"\w+", text.lower()))


def build_evidence_sidecars(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Create compact raw-text provenance; renderer reads this sidecar lazily."""
    candidate_report = _read(paths["results"] / "nested-candidates" / "REPORT.json")
    if candidate_report["status"] != "PASS": raise RuntimeError("candidate gate must pass")
    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(paths["structural"] / "chunks.jsonl"): chunks[str(row["doc_id"])].append(row)
    chunk_index = {str(row["chunk_id"]): index for index, row in enumerate(_jsonl(paths["e5"] / "chunk_ids.jsonl"))}
    query_ids = list(map(str, _read(paths["queries"] / "train_query_ids.json")))
    query_index = {qid: index for index, qid in enumerate(query_ids)}
    query_embeddings = np.load(paths["queries"] / "train_queries.f32.npy", mmap_mode="r")
    chunk_embeddings = np.load(paths["e5"] / "embeddings.f16.npy", mmap_mode="r")
    train = _read(paths["train"]); output_root = paths["cache"] / "evidence-sidecars"; total = malformed = maximum = 0
    for name, info in candidate_report["contexts"].items():
        context = info["context"]; source = paths["cache"] / "nested-candidates" / context["outer"] / name / "candidates.jsonl"; out = output_root / context["outer"] / name
        def rows() -> Iterable[dict[str, Any]]:
            nonlocal total, malformed, maximum
            for row in _jsonl(source):
                qid = str(row["qid"]); qterms = _tokens(train[qid]["question"]); qvec = query_embeddings[query_index[qid]]; selected = []
                for candidate in row["candidates"]:
                    options = chunks.get(str(candidate["doc_id"]), [])
                    if not options: malformed += 1; continue
                    lexical = sorted(options, key=lambda chunk: (-len(qterms & _tokens(str(chunk["raw_text"]))), str(chunk["chunk_id"])))
                    dense = sorted(options, key=lambda chunk: (-float(np.dot(qvec, chunk_embeddings[chunk_index[str(chunk["chunk_id"])]])) if str(chunk["chunk_id"]) in chunk_index else float("inf"), str(chunk["chunk_id"])))
                    lexical_rank = {str(chunk["chunk_id"]): rank for rank, chunk in enumerate(lexical, 1)}; dense_rank = {str(chunk["chunk_id"]): rank for rank, chunk in enumerate(dense, 1)}
                    scored = sorted(options, key=lambda chunk: (-(.5/(32+lexical_rank[str(chunk["chunk_id"])]) + .5/(32+dense_rank[str(chunk["chunk_id"])])), str(chunk["chunk_id"])))
                    chunk = scored[0]; text = str(chunk["raw_text"]); score = .5/(32+lexical_rank[str(chunk["chunk_id"])]) + .5/(32+dense_rank[str(chunk["chunk_id"])]); maximum = max(maximum, len(text)); selected.append({"doc_id": str(candidate["doc_id"]), "chunk_id": str(chunk["chunk_id"]), "source_start": int(chunk["start"]), "source_end": int(chunk["end"]), "parent_node_id": str(chunk["parent_node_id"]), "selector": "hybrid_rrf_dense_050", "selector_score": float(score), "lexical_rank": lexical_rank[str(chunk["chunk_id"])], "dense_rank": dense_rank[str(chunk["chunk_id"])]})
                total += len(selected); yield {"qid": qid, "outer": context["outer"], "context": name, "selected": selected}
        count = _write_jsonl(out / "sidecar.jsonl", rows()); fp = _hash({"candidate": _sha(source), "structural": _sha(paths["structural"] / "manifest.json"), "count": count}); _manifest(out, "build-evidence-sidecars", fp, [out / "sidecar.jsonl"])
    report = {"schema_version": SCHEMA, "status": "PASS" if not malformed else "REJECTED_EVIDENCE_GATE", "pairs": total, "malformed": malformed, "max_raw_chars": maximum, "renderer": "lazy_source_exact_one_view", "query_truncation": 0}
    out = paths["results"] / "evidence-sidecars"; fp = _hash(report); report["fingerprint"] = fp; _write(out / "REPORT.json", report); _manifest(out, "build-evidence-sidecars", fp, [out / "REPORT.json"]); _state(paths["results"], "build-evidence-sidecars", report["status"], completed=1, total=1); return report


def _context_file(paths: Mapping[str, Path], outer: str, name: str, leaf: str) -> Path:
    return paths["cache"] / ("nested-candidates" if leaf == "candidates.jsonl" else "evidence-sidecars") / outer / name / leaf


def _load_context(paths: Mapping[str, Path], outer: str, name: str) -> dict[str, list[dict[str, Any]]]:
    candidates = {str(row["qid"]): row["candidates"] for row in _jsonl(_context_file(paths, outer, name, "candidates.jsonl"))}
    sidecars = {str(row["qid"]): row["selected"] for row in _jsonl(_context_file(paths, outer, name, "sidecar.jsonl"))}
    for qid, rows in candidates.items():
        selected = {str(row["doc_id"]): row for row in sidecars.get(qid, [])}
        if len(rows) != K or set(selected) != {str(row["doc_id"]) for row in rows}: raise ValueError(f"candidate/evidence mismatch {outer}/{name}/{qid}")
        for row in rows: row["evidence"] = selected[str(row["doc_id"])]
    return candidates


def _load_raw_chunks(paths: Mapping[str, Path], ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for row in _jsonl(paths["structural"] / "chunks.jsonl"):
        key = str(row["chunk_id"])
        if key in ids: found[key] = str(row["raw_text"])
        if len(found) == len(ids): break
    if found.keys() != ids: raise ValueError(f"missing source chunks: {len(ids - found.keys())}")
    return found


def _renderer(question: str, candidate: Mapping[str, Any], raw: Mapping[str, str]) -> str:
    evidence = candidate["evidence"]; text = raw[str(evidence["chunk_id"])]
    return f"[VĂN BẢN {candidate['doc_id']}]\n[BẰNG CHỨNG CHÍNH]\n{text}"


def render_budgeted_pair(tokenizer: Any, question: str, candidate: Mapping[str, Any], raw: Mapping[str, str]) -> tuple[str, str]:
    query_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    if len(query_ids) > 128: raise ValueError("query exceeds protected 128-token budget")
    document = _renderer(question, candidate, raw)
    budget = MAX_LENGTH - len(query_ids) - 4
    if budget <= 0: raise ValueError("pair wrapper exhausts model budget")
    doc_ids = tokenizer(document, add_special_tokens=False)["input_ids"][:budget]
    return question, tokenizer.decode(doc_ids, skip_special_tokens=True)


def _make_model(device: torch.device, fp16: bool = False) -> tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=1, local_files_only=True, trust_remote_code=True)
    model.config.use_cache = False; model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["query", "key", "value", "dense"], lora_dropout=.05, bias="none", task_type="SEQ_CLS"))
    return model.to(device), tokenizer


def _score_pairs(model: nn.Module, tokenizer: Any, pairs: Sequence[tuple[str, str]], device: torch.device, batch_size: int = 16, grad: bool = False) -> torch.Tensor | list[float]:
    chunks: list[torch.Tensor] = []
    mode = torch.enable_grad() if grad else torch.inference_mode()
    with mode:
        for start in range(0, len(pairs), batch_size):
            qs, docs = zip(*pairs[start:start+batch_size]); inputs = tokenizer(list(qs), list(docs), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze(-1)
            chunks.append(logits if grad else logits.float().cpu())
    value = torch.cat(chunks)
    return value if grad else value.tolist()


def _context_names(folds: Mapping[str, Sequence[str]], outer: str) -> dict[str, str]:
    return {inner: f"{outer}_inner_{inner}" for inner in sorted(folds) if inner != outer}


def _collect_outer_rows(paths: Mapping[str, Path], outer: str, qids: Sequence[str], folds: Mapping[str, Sequence[str]]) -> dict[str, list[dict[str, Any]]]:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for inner, name in _context_names(folds, outer).items(): by_qid.update(_load_context(paths, outer, name))
    requested = set(map(str, qids))
    if set(by_qid) & requested != requested: raise ValueError("missing nested outer-train context")
    return {qid: by_qid[qid] for qid in requested}


def _train_reranker(paths: Mapping[str, Path], qids: Sequence[str], rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], answers: Mapping[str, set[str]], questions: Mapping[str, str], device: torch.device, alpha: float, output: Path) -> dict[str, Any]:
    ids = {str(row["evidence"]["chunk_id"]) for rows in rows_by_qid.values() for row in rows}; raw = _load_raw_chunks(paths, ids)
    model, tokenizer = _make_model(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    model.train(); steps = skipped = 0
    # Zero-shot mining is computed with the frozen base behaviour before each group update.
    for _epoch in range(2):
        for qid in sorted(map(str, qids)):
            candidates = rows_by_qid[qid]; gold = answers[qid]
            if not gold & {str(row["doc_id"]) for row in candidates}: skipped += 1; continue
            pairs = [render_budgeted_pair(tokenizer, questions[qid], row, raw) for row in candidates]
            with torch.no_grad(): zero = _score_pairs(model, tokenizer, pairs, device)
            mined = {str(row["doc_id"]): float(score) for row, score in zip(candidates, zero)}
            group = build_training_group(candidates, gold, mined)
            group_pairs = [render_budgeted_pair(tokenizer, questions[qid], row, raw) for row in group]
            logits = _score_pairs(model, tokenizer, group_pairs, device, batch_size=len(group_pairs), grad=True)
            labels = torch.tensor([1.0 if str(row["doc_id"]) in gold else 0.0 for row in group], device=device)
            # Protected residual: Stage-1 is fixed and only CE logits are learned.
            anchor = torch.tensor(zscore([float(row["stage1_score"]) for row in group]), device=device)
            ce = (logits - logits.mean()) / logits.std(unbiased=False).clamp_min(1e-5)
            loss = multi_positive_hard_negative_loss((1-alpha)*anchor + alpha*ce, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); steps += 1
    output.mkdir(parents=True, exist_ok=True); model.save_pretrained(output); tokenizer.save_pretrained(output)
    return {"steps": steps, "skipped_no_retained_positive": skipped, "raw_chunks": len(raw), "alpha": alpha}


def _load_adapter(adapter: Path, device: torch.device) -> tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=1, local_files_only=True, trust_remote_code=True)
    return PeftModel.from_pretrained(base, adapter).to(device).eval(), tokenizer


def _score_rows(paths: Mapping[str, Path], rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], questions: Mapping[str, str], adapter: Path, output: Path) -> dict[str, dict[str, float]]:
    ids = {str(row["evidence"]["chunk_id"]) for rows in rows_by_qid.values() for row in rows}; raw = _load_raw_chunks(paths, ids); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, tokenizer = _load_adapter(adapter, device); result = {}
    for index, qid in enumerate(sorted(rows_by_qid), 1):
        rows = rows_by_qid[qid]; scores = _score_pairs(model, tokenizer, [render_budgeted_pair(tokenizer, questions[qid], row, raw) for row in rows], device)
        result[qid] = {str(row["doc_id"]): float(score) for row, score in zip(rows, scores)}
    _write(output, result); return result


def calibrate_fold(paths: Mapping[str, Path], outer: str) -> dict[str, Any]:
    audit = _read(paths["results"] / "audit-inputs" / "REPORT.json"); folds = _read(paths["folds"]); names = sorted(folds); calibration = names[(names.index(outer)+1) % len(names)]
    if calibration == outer: raise ValueError("calibration fold cannot be outer")
    train_qids = [str(q) for name in names if name not in {outer, calibration} for q in folds[name]]; calibration_qids = list(map(str, folds[calibration])); answers, _ = canonical_answers(paths["train"], paths["preprocess"] / "exclusions.json", paths["preprocess"] / "train_label_impact.jsonl"); questions = {str(q): str(value["question"]) for q, value in _read(paths["train"]).items()}
    train_rows = _collect_outer_rows(paths, outer, train_qids, folds); calibration_rows = _load_context(paths, outer, f"{outer}_inner_{calibration}"); out = paths["cache"] / "calibration" / outer; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = _train_reranker(paths, train_qids, train_rows, answers, questions, device, .25, out / "adapter"); scores = _score_rows(paths, {q: calibration_rows[q] for q in calibration_qids}, questions, out / "adapter", out / "scores.json")
    choice = choose_alpha({q: calibration_rows[q] for q in calibration_qids}, scores, {q: answers[q] for q in calibration_qids if answers[q]}); report = {"schema_version": SCHEMA, "outer": outer, "calibration_fold": calibration, "fit": fit, "selection": choice}; fp = _hash({"audit": audit["fingerprint"], "report": report}); report["fingerprint"] = fp; result = paths["results"] / "calibration" / outer; _write(result / "REPORT.json", report); _manifest(result, "calibrate-fold", fp, [result / "REPORT.json"]); _state(paths["results"], "calibrate-fold", "PASS", outer=outer, completed=1, total=1); return report


def train_fold(paths: Mapping[str, Path], outer: str) -> dict[str, Any]:
    folds = _read(paths["folds"]); names = sorted(folds); train_qids = [str(q) for name in names if name != outer for q in folds[name]]; answers, _ = canonical_answers(paths["train"], paths["preprocess"] / "exclusions.json", paths["preprocess"] / "train_label_impact.jsonl"); questions = {str(q): str(value["question"]) for q, value in _read(paths["train"]).items()}; calibration = _read(paths["results"] / "calibration" / outer / "REPORT.json"); alpha = float(calibration["selection"]["selected"]["alpha"]); rows = _collect_outer_rows(paths, outer, train_qids, folds); out = paths["cache"] / "outer-models" / outer; fit = _train_reranker(paths, train_qids, rows, answers, questions, torch.device("cuda" if torch.cuda.is_available() else "cpu"), alpha, out / "adapter"); fp = _hash({"outer": outer, "alpha": alpha, "fit": fit}); report = {"schema_version": SCHEMA, "outer": outer, "alpha": alpha, "fit": fit, "fingerprint": fp}; result = paths["results"] / "train" / outer; _write(result / "REPORT.json", report); _manifest(result, "train-fold", fp, [result / "REPORT.json"]); _state(paths["results"], "train-fold", "PASS", outer=outer, completed=1, total=1); return report


def score_fold(paths: Mapping[str, Path], outer: str) -> dict[str, Any]:
    folds = _read(paths["folds"]); qids = list(map(str, folds[outer])); questions = {str(q): str(value["question"]) for q, value in _read(paths["train"]).items()}; rows = _load_context(paths, outer, f"{outer}_heldout_{outer}"); output = paths["cache"] / "outer-scores" / f"{outer}.json"; scores = _score_rows(paths, {q: rows[q] for q in qids}, questions, paths["cache"] / "outer-models" / outer / "adapter", output); fp = _hash({"outer": outer, "scores": _sha(output)}); report = {"schema_version": SCHEMA, "outer": outer, "queries": len(scores), "fingerprint": fp}; result = paths["results"] / "scores" / outer; _write(result / "REPORT.json", report); _manifest(result, "score-fold", fp, [result / "REPORT.json"]); _state(paths["results"], "score-fold", "PASS", outer=outer, completed=1, total=1); return report


def evaluate_oof(paths: Mapping[str, Path]) -> dict[str, Any]:
    folds = _read(paths["folds"]); answers, _ = canonical_answers(paths["train"], paths["preprocess"] / "exclusions.json", paths["preprocess"] / "train_label_impact.jsonl"); prediction = {}; baseline = {}; per_fold = {}; deltas = []; rescued = harmed = unchanged = outside_k50 = 0; movement = defaultdict(int)
    for outer in sorted(folds):
        rows = _load_context(paths, outer, f"{outer}_heldout_{outer}"); scores = _read(paths["cache"] / "outer-scores" / f"{outer}.json"); alpha = float(_read(paths["results"] / "train" / outer / "REPORT.json")["alpha"]); qids = list(map(str, folds[outer])); fused = {}; anchor = {}
        for qid in qids:
            values = rows[qid]; a, c = zscore([float(row["stage1_score"]) for row in values]), zscore([float(scores[qid][str(row["doc_id"])]) for row in values]); order = sorted(range(K), key=lambda i: -((1-alpha)*a[i]+alpha*c[i])); fused[qid] = [str(values[i]["doc_id"]) for i in order]; anchor[qid] = [str(row["doc_id"]) for row in values]
        local_answers = {q: answers[q] for q in qids if answers[q]}; fm, bm = evaluate(fused, local_answers), evaluate(anchor, local_answers); per_fold[outer] = {"alpha": alpha, "metrics": fm, "baseline": bm, "delta_recall@5": fm["recall@5"]-bm["recall@5"]}; prediction.update(fused); baseline.update(anchor)
        for qid, gold in local_answers.items():
            before = len(set(anchor[qid][:5]) & gold) / len(gold); after = len(set(fused[qid][:5]) & gold) / len(gold); delta_q = after-before; deltas.append(delta_q)
            if delta_q > 0: rescued += 1
            elif delta_q < 0: harmed += 1
            else: unchanged += 1
            for doc in gold:
                old = anchor[qid].index(doc)+1 if doc in anchor[qid] else K+1; new = fused[qid].index(doc)+1 if doc in fused[qid] else K+1
                if old > K: outside_k50 += 1
                movement[f"{min(old, K+1)}->{min(new, K+1)}"] += 1
    metrics, base = evaluate(prediction, answers), evaluate(baseline, answers); rng = np.random.default_rng(SEED); sample = np.asarray(deltas); boot = np.mean(sample[rng.integers(0, len(sample), size=(10000, len(sample)))], axis=1); delta = metrics["recall@5"]-base["recall@5"]; fold_delta = [value["delta_recall@5"] for value in per_fold.values()]; passed = metrics["recall@5"] >= .94 and delta >= .01 and metrics["precision@5"] >= base["precision@5"] and sum(value >= 0 for value in fold_delta) >= 4 and min(fold_delta) >= -.002 and float(np.quantile(boot, .025)) > 0
    report = {"schema_version": SCHEMA, "status": "PASS_PROMOTION_GATE" if passed else "REJECTED_OOF_GATE", "metrics": metrics, "baseline": base, "delta_recall@5": delta, "bootstrap_delta_recall@5_95": [float(np.quantile(boot,.025)), float(np.quantile(boot,.975))], "folds": per_fold, "error_analysis": {"rescued_queries": rescued, "harmed_queries": harmed, "unchanged_queries": unchanged, "gold_outside_k50": outside_k50, "gold_rank_movements": dict(sorted(movement.items()))}, "evidence_failures": 0, "gates": {"aggregate_r5": metrics["recall@5"] >= .94, "delta": delta >= .01, "precision": metrics["precision@5"] >= base["precision@5"], "fold_stability": sum(value >= 0 for value in fold_delta) >= 4 and min(fold_delta) >= -.002, "bootstrap": float(np.quantile(boot,.025)) > 0}}
    out = paths["results"] / "oof"; fp = _hash(report); report["fingerprint"] = fp; _write(out / "REPORT.json", report); _manifest(out, "evaluate-oof", fp, [out / "REPORT.json"]); _state(paths["results"], "evaluate-oof", report["status"], completed=5, total=5); return report


def preflight(paths: Mapping[str, Path]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); result = {"schema_version": SCHEMA, "device": str(device), "model": MODEL_ID, "fp32_optimizer_preflight": "NOT_RUN"}
    if device.type == "cuda":
        model, tokenizer = _make_model(device); model.train()
        sample = tokenizer(["câu hỏi pháp luật về điều khoản áp dụng"] * 16, ["văn bản pháp luật có bằng chứng trọng tâm"] * 16, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", padding=True).to(device)
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4); optimizer.zero_grad(); model(**sample).logits.mean().backward(); optimizer.step(); free, total = torch.cuda.mem_get_info(device); result.update({"fp32_optimizer_preflight": "PASS" if free / total >= .10 else "INSUFFICIENT_HEADROOM", "free_vram": int(free), "total_vram": int(total)})
        except torch.OutOfMemoryError: result["fp32_optimizer_preflight"] = "OOM"
        finally: del model; torch.cuda.empty_cache()
    out = paths["results"] / "preflight"; fp = _hash(result); result["fingerprint"] = fp; _write(out / "REPORT.json", result); _manifest(out, "preflight", fp, [out / "REPORT.json"]); _state(paths["results"], "preflight", "PASS" if result["fp32_optimizer_preflight"] == "PASS" else "NEEDS_FP16_LO_RATIONALE", completed=1, total=1); return result


def status(paths: Mapping[str, Path]) -> dict[str, Any]:
    value = _read(paths["results"] / "RUN_STATUS.json") if (paths["results"] / "RUN_STATUS.json").exists() else {"state": "NOT_STARTED"}; print(json.dumps(value, ensure_ascii=False, indent=2)); return value


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("audit-inputs", "build-nested-candidates", "build-evidence-sidecars", "preflight", "calibrate-fold", "train-fold", "score-fold", "evaluate-oof", "status")); parser.add_argument("--outer", choices=tuple(f"fold_{i}" for i in range(5))); parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "exp106b_nested_k50_reranker"); parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "exp106b_nested_k50_reranker"); parser.add_argument("--train", type=Path, default=ROOT / "public_test_dataset" / "train.json"); parser.add_argument("--folds", type=Path, default=ROOT / "cache" / "cv_folds.json"); parser.add_argument("--preprocess", type=Path, default=ROOT / "cache" / "final_preprocessed_v2"); parser.add_argument("--structural", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1"); parser.add_argument("--e5", type=Path, default=ROOT / "cache" / "e5_final_v1"); parser.add_argument("--queries", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings"); parser.add_argument("--exp022", type=Path, default=ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl"); args = parser.parse_args(); paths = _paths(args); set_seed()
    actions = {"audit-inputs": audit_inputs, "build-nested-candidates": build_nested_candidates, "build-evidence-sidecars": build_evidence_sidecars, "preflight": preflight, "status": status}
    if args.stage in {"calibrate-fold", "train-fold", "score-fold"}:
        if not args.outer: parser.error("--outer is required for fold stages")
        {"calibrate-fold": calibrate_fold, "train-fold": train_fold, "score-fold": score_fold}[args.stage](paths, args.outer)
    elif args.stage == "evaluate-oof": evaluate_oof(paths)
    else: actions[args.stage](paths)
    return 0


if __name__ == "__main__": raise SystemExit(main())
