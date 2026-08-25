"""Small, reproducible model/tokenizer screen over a fixed v3 chunk fixture.

This is intentionally a development screen, not a five-fold evaluation.  It
never reads test labels and it does not mutate the structural-v3 cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache" / "exp015_model_screen"
DEFAULT_RESULTS = ROOT / "results" / "exp015_model_screen"
V3_DIR = ROOT / "cache" / "structural_v3"
TRAIN_PATH = ROOT / "public_test_dataset" / "train.json"
FOLDS_PATH = ROOT / "cache" / "cv_folds.json"
FIXTURE_SEED = 15015
FIXTURE_QUERIES = 128
BACKGROUND_DOCUMENTS = 384
CHUNKS_PER_DOCUMENT = 8


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    backend: str  # sentence_transformers or last_token_auto
    max_length: int
    query_prefix: str
    document_prefix: str
    trust_remote_code: bool = False


LEGAL_INSTRUCTION = (
    "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question"
)
E5_INSTRUCT_PREFIX = f"Instruct: {LEGAL_INSTRUCTION}\nQuery: "

MODELS: dict[str, ModelSpec] = {
    "bge_m3": ModelSpec("bge_m3", "BAAI/bge-m3", "sentence_transformers", 512, "", ""),
    "vnlegal_lal": ModelSpec(
        "vnlegal_lal", "darklethelong/vnlegal-lal", "last_token_auto", 2048,
        f"Instruct: {LEGAL_INSTRUCTION}\nQuery: ", "",
    ),
    "vietnamese_legal_embedding": ModelSpec(
        "vietnamese_legal_embedding", "bqbbao6/vietnamese-legal-embedding",
        "sentence_transformers", 512, "query: ", "passage: ", True,
    ),
    "vietlegal_harrier_0_6b": ModelSpec(
        "vietlegal_harrier_0_6b", "mainguyen9/vietlegal-harrier-0.6b",
        "sentence_transformers", 512, E5_INSTRUCT_PREFIX, "",
    ),
    "multilingual_e5_large_instruct": ModelSpec(
        "multilingual_e5_large_instruct", "intfloat/multilingual-e5-large-instruct",
        "sentence_transformers", 512, E5_INSTRUCT_PREFIX, "",
    ),
    "qwen3_embedding_0_6b": ModelSpec(
        "qwen3_embedding_0_6b", "Qwen/Qwen3-Embedding-0.6B", "last_token_auto", 2048,
        E5_INSTRUCT_PREFIX, "",
    ),
    "vietlegal_e5": ModelSpec(
        "vietlegal_e5", "mainguyen9/vietlegal-e5", "sentence_transformers", 512,
        "query: ", "passage: ",
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_train() -> dict[str, dict[str, Any]]:
    raw = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("train.json must be a qid mapping")
    return {str(qid): row for qid, row in raw.items()}


def answer_ids(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer", [])
    if isinstance(answer, dict):
        answer = answer.get("answer", answer.get("documents", []))
    if not isinstance(answer, list):
        raise ValueError("Unexpected answer field")
    return [str(value) for value in answer]


def fixture_paths(cache_dir: Path) -> dict[str, Path]:
    root = cache_dir / "fixture"
    return {
        "root": root,
        "queries": root / "queries.jsonl",
        "chunks": root / "chunks.jsonl",
        "manifest": root / "manifest.json",
        "success": root / "_SUCCESS.json",
    }


def build_fixture(cache_dir: Path, *, force: bool = False) -> dict[str, Any]:
    paths = fixture_paths(cache_dir)
    if paths["success"].exists() and not force:
        return json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if any(path.exists() for key, path in paths.items() if key not in {"root", "success"}) and not force:
        raise RuntimeError("Partial fixture exists; use --force-fixture only after inspecting it")

    train = load_train()
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    fold_qids = [str(qid) for qid in folds["fold_0"]]
    v3_manifest = json.loads((V3_DIR / "manifest.json").read_text(encoding="utf-8"))

    # First pass obtains the available document universe without materialising 1.2GB of chunks.
    document_ids: set[str] = set()
    for row in read_jsonl(V3_DIR / "documents.jsonl"):
        document_ids.add(str(row["doc_id"]))
    eligible = [
        qid for qid in fold_qids
        if qid in train and answer_ids(train[qid]) and set(answer_ids(train[qid])).issubset(document_ids)
    ]
    ordered_qids = sorted(eligible, key=lambda qid: digest({"seed": FIXTURE_SEED, "qid": qid}))
    selected_qids = ordered_qids[:FIXTURE_QUERIES]
    if len(selected_qids) != FIXTURE_QUERIES:
        raise RuntimeError(f"Only {len(selected_qids)} eligible fold_0 questions; expected {FIXTURE_QUERIES}")

    positive_docs = {doc_id for qid in selected_qids for doc_id in answer_ids(train[qid])}
    remaining_docs = sorted(document_ids - positive_docs, key=lambda doc_id: digest({"seed": FIXTURE_SEED, "doc_id": doc_id}))
    background_docs = remaining_docs[:BACKGROUND_DOCUMENTS]
    chosen_docs = positive_docs | set(background_docs)

    # An equal document cap prevents long documents from receiving a larger maximum-score lottery.
    per_doc: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for chunk in read_jsonl(V3_DIR / "chunks.jsonl"):
        doc_id = str(chunk["doc_id"])
        if doc_id in chosen_docs:
            per_doc[doc_id].append((digest({"seed": FIXTURE_SEED, "chunk": chunk["chunk_id"]}), chunk))
    missing_chunk_docs = sorted(doc_id for doc_id in chosen_docs if not per_doc.get(doc_id))
    if missing_chunk_docs:
        raise RuntimeError(f"Fixture documents without chunks: {missing_chunk_docs[:10]}")
    chosen_chunks: list[dict[str, Any]] = []
    for doc_id in sorted(chosen_docs):
        entries = sorted(per_doc[doc_id], key=lambda pair: pair[0])[:CHUNKS_PER_DOCUMENT]
        chosen_chunks.extend(chunk for _, chunk in entries)
    chosen_chunks.sort(key=lambda row: str(row["chunk_id"]))

    queries = [
        {"qid": qid, "question": str(train[qid]["question"]), "answer_doc_ids": answer_ids(train[qid])}
        for qid in selected_qids
    ]
    fixture_doc_ids = {str(row["doc_id"]) for row in chosen_chunks}
    oracle_missing = {
        row["qid"]: sorted(set(row["answer_doc_ids"]) - fixture_doc_ids)
        for row in queries
        if set(row["answer_doc_ids"]) - fixture_doc_ids
    }
    if oracle_missing:
        raise RuntimeError(f"Fixture is missing answer documents: {oracle_missing}")

    paths["root"].mkdir(parents=True, exist_ok=True)
    with paths["queries"].open("w", encoding="utf-8", newline="\n") as handle:
        for row in queries:
            handle.write(canonical_json(row) + "\n")
    with paths["chunks"].open("w", encoding="utf-8", newline="\n") as handle:
        for row in chosen_chunks:
            handle.write(canonical_json(row) + "\n")
    manifest = {
        "experiment": "exp015_model_screen",
        "purpose": "development_screen_only_not_oof",
        "corpus_policy": "frozen_structural_v3_unfiltered",
        "v3_content_fingerprint": v3_manifest["content_fingerprint"],
        "fixture_seed": FIXTURE_SEED,
        "source_fold": "fold_0",
        "query_count": len(queries),
        "positive_document_count": len(positive_docs),
        "background_document_count": len(background_docs),
        "document_count": len(fixture_doc_ids),
        "chunk_count": len(chosen_chunks),
        "chunks_per_document_cap": CHUNKS_PER_DOCUMENT,
        "query_ids_sha256": digest(selected_qids),
        "chunk_ids_sha256": digest([row["chunk_id"] for row in chosen_chunks]),
        "queries_sha256": sha256_file(paths["queries"]),
        "chunks_sha256": sha256_file(paths["chunks"]),
    }
    write_json(paths["manifest"], manifest)
    write_json(paths["success"], {"manifest_sha256": sha256_file(paths["manifest"])})
    return manifest


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError("Non-finite or zero-norm embeddings")
    return vectors / norms


class SentenceTransformerEncoder:
    def __init__(self, spec: ModelSpec, device: str, allow_download: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = device
        self.model = SentenceTransformer(
            spec.repo_id, device=device, trust_remote_code=spec.trust_remote_code,
            local_files_only=not allow_download,
        )
        self.model.max_seq_length = spec.max_length
        if device.startswith("cuda"):
            self.model.half()
        self.tokenizer = self.model.tokenizer

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return l2_normalize(np.asarray(self.model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )))


class LastTokenEncoder:
    def __init__(self, spec: ModelSpec, device: str, allow_download: bool) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.device = device
        self.torch = torch
        kwargs = {"trust_remote_code": spec.trust_remote_code, "local_files_only": not allow_download}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, **kwargs)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(spec.repo_id, dtype=dtype, **kwargs).to(device).eval()

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=self.spec.max_length, return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            # Qwen-family model cards specify last-token pooling.  The mask form is robust to
            # both left and right padding rather than assuming a particular tokenizer default.
            positions = encoded["attention_mask"].sum(dim=1) - 1
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                positions = encoded["attention_mask"].shape[1] - 1 - (encoded["attention_mask"].sum(dim=1) == 0).long()
            pooled = hidden[range(hidden.shape[0]), positions]
            batches.append(pooled.float().cpu().numpy())
        return l2_normalize(np.concatenate(batches, axis=0))


def create_encoder(spec: ModelSpec, device: str, allow_download: bool) -> Any:
    if spec.backend == "sentence_transformers":
        return SentenceTransformerEncoder(spec, device, allow_download)
    if spec.backend == "last_token_auto":
        return LastTokenEncoder(spec, device, allow_download)
    raise ValueError(f"Unknown backend: {spec.backend}")


def encode_oom_safe(encoder: Any, texts: Sequence[str], batch_size: int, device: str) -> tuple[np.ndarray, int]:
    current = batch_size
    while True:
        try:
            return encoder.encode(texts, current), current
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or current <= 1:
                raise
            current = max(1, current // 2)
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()


def fixture_rows(cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = fixture_paths(cache_dir)
    if not paths["success"].exists():
        raise RuntimeError("Fixture missing. Run --stage build-fixture first.")
    return list(read_jsonl(paths["queries"])), list(read_jsonl(paths["chunks"])), json.loads(paths["manifest"].read_text(encoding="utf-8"))


def evaluate(queries: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]], q_vec: np.ndarray, d_vec: np.ndarray) -> dict[str, Any]:
    doc_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(chunks):
        doc_to_indices[str(row["doc_id"])].append(index)
    doc_ids = sorted(doc_to_indices)
    score_matrix = q_vec @ d_vec.T
    ranks: list[int] = []
    per_query: list[dict[str, Any]] = []
    for q_index, query in enumerate(queries):
        scores = [(doc_id, float(np.max(score_matrix[q_index, indices]))) for doc_id, indices in doc_to_indices.items()]
        scores.sort(key=lambda item: (-item[1], item[0]))
        target = set(query["answer_doc_ids"])
        rank = next(index for index, (doc_id, _) in enumerate(scores, start=1) if doc_id in target)
        ranks.append(rank)
        per_query.append({"qid": query["qid"], "first_relevant_rank": rank, "top_5": [doc_id for doc_id, _ in scores[:5]]})
    metrics = {f"recall_at_{k}": float(np.mean([rank <= k for rank in ranks])) for k in (5, 20, 100)}
    metrics["mrr_at_5"] = float(np.mean([1.0 / rank if rank <= 5 else 0.0 for rank in ranks]))
    metrics["median_first_relevant_rank"] = float(np.median(ranks))
    return {"metrics": metrics, "per_query": per_query, "document_count": len(doc_ids)}


def model_paths(cache_dir: Path, results_dir: Path, key: str) -> tuple[Path, Path, Path]:
    return cache_dir / "models" / key, results_dir / f"{key}.json", results_dir / f"{key}.embeddings.npz"


def run_model(cache_dir: Path, results_dir: Path, key: str, *, device: str, batch_size: int, allow_download: bool, force: bool) -> dict[str, Any]:
    if key not in MODELS:
        raise ValueError(f"Unknown model key {key}; choices: {', '.join(MODELS)}")
    spec = MODELS[key]
    model_dir, report_path, embedding_path = model_paths(cache_dir, results_dir, key)
    if report_path.exists() and embedding_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    queries, chunks, fixture = fixture_rows(cache_dir)
    chunk_texts = [spec.document_prefix + str(row["retrieval_text"]) for row in chunks]
    query_texts = [spec.query_prefix + str(row["question"]) for row in queries]
    start = time.perf_counter()
    encoder = create_encoder(spec, device, allow_download)
    chunk_counts = encoder.token_counts(chunk_texts)
    query_counts = encoder.token_counts(query_texts)
    chunk_vectors, effective_batch = encode_oom_safe(encoder, chunk_texts, batch_size, device)
    query_vectors, effective_batch = encode_oom_safe(encoder, query_texts, effective_batch, device)
    elapsed = time.perf_counter() - start
    result = evaluate(queries, chunks, query_vectors, chunk_vectors)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embedding_path, query_vectors=query_vectors.astype(np.float32), chunk_vectors=chunk_vectors.astype(np.float32))
    try:
        import torch
        peak_memory_mib = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.startswith("cuda") else None
    except Exception:
        peak_memory_mib = None
    report = {
        "experiment": "exp015_model_screen",
        "interpretation": "development_screen_only_not_oof",
        "model": {"key": spec.key, "repo_id": spec.repo_id, "backend": spec.backend, "max_length": spec.max_length,
                  "query_prefix": spec.query_prefix, "document_prefix": spec.document_prefix, "trust_remote_code": spec.trust_remote_code},
        "fixture": fixture,
        "runtime": {"device": device, "initial_batch_size": batch_size, "effective_batch_size": effective_batch,
                    "elapsed_seconds": round(elapsed, 3), "texts_per_second": round((len(chunk_texts) + len(query_texts)) / elapsed, 3),
                    "peak_cuda_memory_mib": peak_memory_mib},
        "tokenization": {"chunk_tokens_p50": float(np.percentile(chunk_counts, 50)), "chunk_tokens_p95": float(np.percentile(chunk_counts, 95)),
                         "chunk_truncated": int(sum(count > spec.max_length for count in chunk_counts)), "query_truncated": int(sum(count > spec.max_length for count in query_counts))},
        "embeddings": {"dimension": int(chunk_vectors.shape[1]), "chunk_vectors_sha256": hashlib.sha256(chunk_vectors.tobytes()).hexdigest()},
        **result,
    }
    write_json(report_path, report)
    write_json(model_dir / "manifest.json", {"report": str(report_path), "embedding": str(embedding_path), "report_sha256": sha256_file(report_path)})
    del encoder, chunk_vectors, query_vectors
    gc.collect()
    if device.startswith("cuda"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


def write_summary(results_dir: Path) -> Path:
    reports = []
    for key in MODELS:
        path = results_dir / f"{key}.json"
        if path.exists():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    reports.sort(key=lambda row: (-row["metrics"]["recall_at_5"], row["model"]["key"]))
    lines = ["# EXP-015 model/tokenizer screen", "", "> Development screen only (one fixed fold-0 fixture); not an OOF or final-model claim.", "",
             "| Rank | Model | R@5 | R@20 | R@100 | MRR@5 | Dim | sec | texts/s | trunc. chunks |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(reports, start=1):
        metrics, runtime, tok = row["metrics"], row["runtime"], row["tokenization"]
        lines.append(f"| {index} | `{row['model']['key']}` | {metrics['recall_at_5']:.4f} | {metrics['recall_at_20']:.4f} | {metrics['recall_at_100']:.4f} | {metrics['mrr_at_5']:.4f} | {row['embeddings']['dimension']} | {runtime['elapsed_seconds']:.1f} | {runtime['texts_per_second']:.1f} | {tok['chunk_truncated']} |")
    lines.extend(["", "Every model uses its documented query/document formatting and native token budget; corpus rows, cosine scoring, max-over-chunks parent aggregation, and fixture are identical."])
    summary = results_dir / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def preflight() -> dict[str, Any]:
    required = [V3_DIR / "manifest.json", V3_DIR / "documents.jsonl", V3_DIR / "chunks.jsonl", TRAIN_PATH, FOLDS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    result = {"python": platform.python_version(), "missing_inputs": missing, "models": {key: spec.repo_id for key, spec in MODELS.items()}}
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "build-fixture", "run-model", "run-all", "summarize"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), help="Required with --stage run-model")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face model download if model is not cached")
    parser.add_argument("--force-fixture", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run a completed per-model screen")
    args = parser.parse_args()
    if args.stage == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.stage == "build-fixture":
        print(json.dumps(build_fixture(args.cache_dir, force=args.force_fixture), ensure_ascii=False, indent=2))
    elif args.stage == "run-model":
        if not args.model:
            parser.error("--model is required with --stage run-model")
        print(json.dumps(run_model(args.cache_dir, args.results_dir, args.model, device=args.device, batch_size=args.batch_size, allow_download=args.allow_download, force=args.force), ensure_ascii=False, indent=2))
    elif args.stage == "run-all":
        for key in MODELS:
            print(f"[EXP015] {key}", flush=True)
            run_model(args.cache_dir, args.results_dir, key, device=args.device, batch_size=args.batch_size, allow_download=args.allow_download, force=args.force)
        print(write_summary(args.results_dir))
    else:
        print(write_summary(args.results_dir))


if __name__ == "__main__":
    main()
