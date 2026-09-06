"""EXP-034: fold-isolated shallow retrieval with a query-side projection.

The document/chunk encoder is immutable.  This experiment learns only a small
residual projection over cached VietLegal-E5 query vectors and evaluates either
a fixed K=32 pool or, after that gate fails, a nested adaptive K policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

from exp012b_core import (
    atomic_json,
    canonical_json,
    content_hash,
    read_jsonl,
    require_success,
    sha256_file,
    stage_run,
    write_jsonl,
)
from exp012b_tuning import load_folds
from exp030_legal_evidence_routing import LABEL_POLICY, canonical_answers


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp034_shallow_retrieval.v1"
EXPECTED_LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
SEED = 2034
DIMENSION = 1024
RANK = 32
KS = (5, 16, 24, 32, 40, 50, 64)
TIERS = (16, 24, 32, 40, 50, 64)
CHUNK_DEPTH = 4096
MAX_OUTPUT = 64
EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
GRAD_CLIP = 1.0
FIXED_FLOOR = 0.985
RRF_TOLERANCE = 0.002
ADAPTIVE_MEAN_K_MAX = 40.0
AGGREGATIONS = ("max", "top2_mean")
BETAS = (0.0, 0.005, 0.01)
GAMMAS = (0.0, 0.005)
RRF_GRID = tuple((alpha, constant) for alpha in (0.35, 0.5, 0.65, 0.8) for constant in (0, 10, 32, 60))


def require_label_policy() -> None:
    if LABEL_POLICY != EXPECTED_LABEL_POLICY:
        raise RuntimeError(f"EXP-034 refuses canonical label policy {LABEL_POLICY!r}")


def crossfit_splits(folds: Mapping[str, Sequence[str]], outer: str) -> list[tuple[str, list[str], list[str]]]:
    if outer not in folds:
        raise ValueError(f"unknown outer fold: {outer}")
    outer_heldout = set(map(str, folds[outer])); result = []
    train_folds = [name for name in sorted(folds) if name != outer]
    for inner in train_folds:
        train_qids = [str(qid) for name in train_folds if name != inner for qid in folds[name]]
        eval_qids = list(map(str, folds[inner]))
        if outer_heldout & (set(train_qids) | set(eval_qids)) or set(train_qids) & set(eval_qids):
            raise RuntimeError("fold isolation violation")
        result.append((inner, train_qids, eval_qids))
    return result


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_qid_seed(qid: str) -> int:
    return int.from_bytes(hashlib.sha256(str(qid).encode("utf-8")).digest()[:8], "big")


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _write_manifest(*, output_dir: Path, stage: str, inputs: Mapping[str, Any],
                    config: Mapping[str, Any], files: Sequence[Path]) -> dict[str, Any]:
    hashes = {path.name: sha256_file(path) for path in files}
    manifest = {
        "schema_version": SCHEMA, "stage": stage, "inputs": dict(inputs), "config": dict(config),
        "artifact_sha256": hashes, "content_fingerprint": content_hash({"inputs": inputs, "config": config, "files": hashes}),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class AggregationConfig:
    aggregation: str
    beta: float
    gamma: float

    @property
    def key(self) -> str:
        return f"{self.aggregation}__b{self.beta:g}__g{self.gamma:g}"

    def as_dict(self) -> dict[str, Any]:
        return {"aggregation": self.aggregation, "beta": self.beta, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AggregationConfig":
        aggregation = str(value["aggregation"])
        beta, gamma = float(value["beta"]), float(value["gamma"])
        if aggregation not in AGGREGATIONS or beta not in BETAS or gamma not in GAMMAS:
            raise ValueError(f"unknown aggregation policy: {value}")
        return cls(aggregation, beta, gamma)


AGGREGATION_GRID = tuple(AggregationConfig(a, b, g) for a in AGGREGATIONS for b in BETAS for g in GAMMAS)


class ResidualProjection(nn.Module):
    """Identity-initialized low-rank residual query projection."""

    def __init__(self, dimension: int = DIMENSION, rank: int = RANK) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(query + self.up(torch.nn.functional.gelu(self.down(query))), p=2, dim=-1)


def lse_pairwise_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor,
                      temperature: float = TEMPERATURE) -> torch.Tensor:
    """Mean smooth pairwise loss; every positive receives its own gradient."""
    if positive_scores.ndim != 1 or negative_scores.ndim != 1:
        raise ValueError("positive and negative scores must be vectors")
    if not positive_scores.numel() or not negative_scores.numel():
        raise ValueError("pairwise loss needs at least one positive and one negative")
    margins = (negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)) / temperature
    zeros = torch.zeros((margins.shape[0], 1), dtype=margins.dtype, device=margins.device)
    return torch.logsumexp(torch.cat((zeros, margins), dim=1), dim=1).mean()


def parent_score(scores: torch.Tensor, aggregation: str) -> torch.Tensor:
    if scores.ndim != 1 or not scores.numel():
        raise ValueError("parent score needs a non-empty score vector")
    values = torch.topk(scores, min(2, scores.numel())).values
    if aggregation == "max":
        return values[0]
    if aggregation == "top2_mean":
        return values.mean()
    raise ValueError(f"unknown aggregation: {aggregation}")


def aggregate_hits(hits: Iterable[tuple[int, str, str, float]], chunk_counts: Mapping[str, int],
                   config: AggregationConfig, limit: int = MAX_OUTPUT) -> list[dict[str, Any]]:
    """Aggregate score-sorted chunk hits into deterministic parent rankings."""
    by_doc: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for index, doc_id, kind, score in hits:
        if len(by_doc[str(doc_id)]) < 2:
            by_doc[str(doc_id)].append((int(index), str(kind), float(score)))
    rows: list[dict[str, Any]] = []
    for doc_id, evidence in by_doc.items():
        values = [item[2] for item in evidence]
        base = values[0] if config.aggregation == "max" else sum(values) / len(values)
        score = base - config.beta * math.log1p(int(chunk_counts[doc_id]))
        if evidence[0][1] == "article":
            score += config.gamma
        rows.append({
            "doc_id": doc_id,
            "score": float(score),
            "evidence": [{"chunk_index": index, "kind": kind, "score": value} for index, kind, value in evidence],
        })
    rows.sort(key=lambda row: (-float(row["score"]), str(row["doc_id"])))
    return rows[:limit]


def evaluate_rankings(rankings: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]],
                      qids: Iterable[str], ks: Sequence[int] = KS) -> dict[str, float]:
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"precision@{k}": 0.0 for k in ks})
    totals["mrr@5"] = 0.0
    count = 0
    for raw_qid in qids:
        qid = str(raw_qid); gold = answers[qid]
        if not gold:
            continue
        prediction = list(map(str, rankings[qid])); count += 1
        for k in ks:
            overlap = len(set(prediction[:k]) & gold)
            totals[f"recall@{k}"] += overlap / len(gold)
            totals[f"precision@{k}"] += overlap / k
        first = next((rank for rank, doc_id in enumerate(prediction[:5], 1) if doc_id in gold), None)
        totals["mrr@5"] += 0.0 if first is None else 1.0 / first
    return {key: value / max(count, 1) for key, value in totals.items()}


def choose_aggregation(config_rankings: Mapping[str, Mapping[str, Sequence[str]]], answers: Mapping[str, set[str]],
                       train_qids: Sequence[str]) -> AggregationConfig:
    def key(config: AggregationConfig) -> tuple[float, float, float, int]:
        metric = evaluate_rankings(config_rankings[config.key], answers, train_qids)
        return metric["recall@32"], metric["recall@5"], metric["mrr@5"], -AGGREGATION_GRID.index(config)
    return max(AGGREGATION_GRID, key=key)


def mine_negative_ids(*, dense_ranking: Sequence[str], bm25_ranking: Sequence[str], gold: set[str],
                      all_docs: Sequence[str], seed: int) -> list[str]:
    rng = random.Random(seed); selected: list[str] = []; seen = set(gold)
    def take(values: Iterable[str], count: int) -> None:
        for raw in values:
            doc_id = str(raw)
            if doc_id in seen:
                continue
            seen.add(doc_id); selected.append(doc_id)
            if len(selected) >= count:
                break
    take(dense_ranking, 3)
    dense_count = len(selected)
    target = dense_count + 3
    for raw in bm25_ranking:
        doc_id = str(raw)
        if doc_id not in seen:
            seen.add(doc_id); selected.append(doc_id)
        if len(selected) >= target:
            break
    shuffled = list(map(str, all_docs)); rng.shuffle(shuffled)
    take(shuffled, 8)
    if len(selected) != 8 or set(selected) & gold:
        raise RuntimeError("negative mining could not produce eight clean documents")
    return selected


def required_tier(ranking: Sequence[str], gold: set[str]) -> int:
    if not gold:
        raise ValueError("required tier is undefined for a query without canonical gold")
    positions = [rank for rank, doc_id in enumerate(ranking[: TIERS[-1]], 1) if str(doc_id) in gold]
    worst = max(positions) if len(positions) == len(gold) else TIERS[-1]
    return next(tier for tier in TIERS if worst <= tier)


def apply_tier_offset(predicted_tier_indices: Sequence[int], offset: int) -> list[int]:
    if offset < 0 or offset >= len(TIERS):
        raise ValueError("adaptive tier offset is outside the supported range")
    result = [TIERS[min(max(int(index), 0) + offset, len(TIERS) - 1)] for index in predicted_tier_indices]
    if any(value not in TIERS for value in result):
        raise RuntimeError("malformed adaptive policy output")
    return result


def calibrate_tier_offset(*, predicted_indices: Sequence[int], rankings: Sequence[Sequence[str]],
                          gold: Sequence[set[str]], fold_names: Sequence[str], floor: float = FIXED_FLOOR) -> int:
    if not (len(predicted_indices) == len(rankings) == len(gold) == len(fold_names)):
        raise ValueError("adaptive calibration arrays differ in length")
    for offset in range(len(TIERS)):
        budgets = apply_tier_offset(predicted_indices, offset)
        recalls: dict[str, list[float]] = defaultdict(list)
        for budget, ranking, labels, fold in zip(budgets, rankings, gold, fold_names):
            if labels:
                recalls[str(fold)].append(len(set(ranking[:budget]) & labels) / len(labels))
        if recalls and all(sum(values) / len(values) >= floor for values in recalls.values()):
            return offset
    raise RuntimeError("no conservative adaptive offset satisfies every inner fold")


class RetrievalData:
    """Validated, immutable view over EXP-021/E5 and structural metadata."""

    def __init__(self, *, e5_dir: Path, query_dir: Path, v3_dir: Path, train: Path,
                 folds_path: Path, preprocessing: Path) -> None:
        require_label_policy()
        self.e5_dir, self.query_dir, self.v3_dir = e5_dir, query_dir, v3_dir
        self.e5_manifest = _json(e5_dir / "manifest.json")
        self.v3_manifest = _json(v3_dir / "manifest.json")
        self.query_manifest = _json(query_dir / "manifest.json")
        self.answers, self.label_stats = canonical_answers(
            train, preprocessing / "exclusions.json", preprocessing / "train_label_impact.jsonl"
        )
        train_payload = _json(train)
        self.queries = {str(qid): str(row["question"]) for qid, row in train_payload.items()}
        self.folds = {name: list(map(str, values)) for name, values in load_folds(folds_path).items()}
        self.fold_for = {qid: name for name, qids in self.folds.items() for qid in qids}
        self.query_ids = list(map(str, _json(query_dir / "train_query_ids.json")))
        self.query_index = {qid: index for index, qid in enumerate(self.query_ids)}
        self.query_matrix = np.load(query_dir / "train_queries.f32.npy", mmap_mode="r")
        ids = list(read_jsonl(e5_dir / "chunk_ids.jsonl"))
        self.chunk_ids = [str(row["chunk_id"]) for row in ids]
        self.chunk_docs = [str(row["doc_id"]) for row in ids]
        self.chunk_kinds = [chunk_id.split(":", 2)[1] for chunk_id in self.chunk_ids]
        self.doc_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, doc_id in enumerate(self.chunk_docs): self.doc_to_indices[doc_id].append(index)
        self.chunk_counts = {doc_id: len(values) for doc_id, values in self.doc_to_indices.items()}
        self.all_docs = sorted(self.doc_to_indices)
        self.doc_metadata = {str(row["doc_id"]): row for row in read_jsonl(v3_dir / "documents.jsonl")}
        self._validate()

    def _validate(self) -> None:
        if not (self.e5_dir / "_SUCCESS.json").exists() or not (self.v3_dir / "_SUCCESS.json").exists():
            raise RuntimeError("E5 or structural cache lacks _SUCCESS.json")
        if self.e5_manifest.get("corpus_fingerprint") != self.v3_manifest.get("content_fingerprint"):
            raise RuntimeError("E5 and structural corpus fingerprints differ")
        if self.e5_manifest.get("dimension") != DIMENSION or self.query_manifest.get("dimension") != DIMENSION:
            raise RuntimeError("EXP-034 requires 1024-dimensional E5 embeddings")
        if self.query_manifest.get("e5_cache_fingerprint") != self.e5_manifest.get("cache_fingerprint"):
            raise RuntimeError("query embeddings do not belong to the E5 cache")
        if self.query_matrix.shape != (len(self.query_ids), DIMENSION) or self.query_matrix.dtype != np.float32:
            raise RuntimeError("malformed cached query matrix")
        if len(self.chunk_ids) != int(self.e5_manifest.get("chunks", -1)):
            raise RuntimeError("chunk IDs and E5 manifest differ")
        expected = set(self.answers)
        if set(self.query_ids) != expected or set(self.fold_for) != expected or set(self.queries) != expected:
            raise RuntimeError("train, folds, answers, and query cache have different QIDs")
        if set(self.chunk_counts) != set(self.doc_metadata):
            raise RuntimeError("structural documents and E5 chunk parents differ")

    def query_vectors(self, qids: Sequence[str], device: torch.device) -> torch.Tensor:
        values = np.array([self.query_matrix[self.query_index[str(qid)]] for qid in qids], dtype=np.float32)
        return torch.from_numpy(values).to(device)

    def document_tensor(self, device: torch.device) -> torch.Tensor:
        matrix = np.array(np.load(self.e5_dir / self.e5_manifest["embedding_file"], mmap_mode="r"), dtype=np.float32)
        if matrix.shape != (len(self.chunk_ids), DIMENSION):
            raise RuntimeError("malformed document embedding matrix")
        return torch.from_numpy(matrix).to(device)


def _exact_records(*, data: RetrievalData, qids: Sequence[str], model: ResidualProjection,
                   documents: torch.Tensor, configs: Sequence[AggregationConfig], device: torch.device,
                   batch_size: int = 32, limit: int = MAX_OUTPUT) -> dict[str, list[dict[str, Any]]]:
    output = {config.key: [] for config in configs}; model.eval()
    with torch.inference_mode():
        for start in range(0, len(qids), batch_size):
            batch_qids = list(qids[start:start + batch_size])
            query = model(data.query_vectors(batch_qids, device))
            scores, indices = torch.topk(query @ documents.T, CHUNK_DEPTH, dim=1)
            for qid, row_scores, row_indices in zip(batch_qids, scores.cpu().numpy(), indices.cpu().numpy()):
                hits = [(int(index), data.chunk_docs[int(index)], data.chunk_kinds[int(index)], float(score))
                        for score, index in zip(row_scores, row_indices)]
                for config in configs:
                    candidates = aggregate_hits(hits, data.chunk_counts, config, limit=limit)
                    output[config.key].append({"schema_version": SCHEMA, "qid": qid, "candidates": candidates})
    return output


def _exact_rankings_by_config(*, data: RetrievalData, qids: Sequence[str], model: ResidualProjection,
                              documents: torch.Tensor, configs: Sequence[AggregationConfig],
                              device: torch.device, batch_size: int = 32,
                              limit: int = MAX_OUTPUT) -> dict[str, dict[str, list[str]]]:
    """Score once and retain only doc IDs for the full calibration grid."""
    output: dict[str, dict[str, list[str]]] = {config.key: {} for config in configs}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(qids), batch_size):
            batch_qids = list(qids[start:start + batch_size])
            query = model(data.query_vectors(batch_qids, device))
            scores, indices = torch.topk(query @ documents.T, CHUNK_DEPTH, dim=1)
            for qid, row_scores, row_indices in zip(batch_qids, scores.cpu().numpy(), indices.cpu().numpy()):
                hits = [(int(index), data.chunk_docs[int(index)], data.chunk_kinds[int(index)], float(score))
                        for score, index in zip(row_scores, row_indices)]
                for config in configs:
                    output[config.key][qid] = [item["doc_id"] for item in aggregate_hits(
                        hits, data.chunk_counts, config, limit=limit
                    )]
    return output


def _rankings(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {str(row["qid"]): [str(item["doc_id"]) for item in row["candidates"]] for row in records}


def _load_bm25(sidecar: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_jsonl(sidecar):
        values = [(int(item["bm25"]["rank"]), str(item["doc_id"])) for item in row["bm25"] if item.get("bm25")]
        result[str(row["qid"])] = [doc_id for _, doc_id in sorted(values)]
    return result


def _rrf_rank(e5: Sequence[str], bm25: Sequence[str], alpha: float, constant: int, limit: int = MAX_OUTPUT) -> list[str]:
    er = {str(doc): rank for rank, doc in enumerate(e5, 1)}; br = {str(doc): rank for rank, doc in enumerate(bm25, 1)}
    docs = set(er) | set(br)
    return sorted(docs, key=lambda doc: (
        -(alpha / (constant + er[doc]) if doc in er else 0.0)
        - ((1.0 - alpha) / (constant + br[doc]) if doc in br else 0.0), str(doc)
    ))[:limit]


def audit_inputs(*, data: RetrievalData, train: Path, folds: Path, preprocessing: Path,
                 sidecar: Path, output_dir: Path) -> dict[str, Any]:
    with stage_run(output_dir, "exp034-input-audit", total=1, v3_fingerprint=data.v3_manifest["content_fingerprint"]) as log:
        require_success(sidecar.parent)
        expected_e5_hashes = data.e5_manifest.get("artifact_sha256", {})
        for name, expected in expected_e5_hashes.items():
            actual = sha256_file(data.e5_dir / name)
            if actual != expected: raise RuntimeError(f"E5 artifact hash mismatch: {name}")
        for name, expected in data.query_manifest.get("artifacts", {}).items():
            if sha256_file(data.query_dir / name) != expected:
                raise RuntimeError(f"query artifact hash mismatch: {name}")
        expected_documents = data.v3_manifest.get("artifact_sha256", {}).get("documents.jsonl")
        if not expected_documents or sha256_file(data.v3_dir / "documents.jsonl") != expected_documents:
            raise RuntimeError("structural document metadata hash mismatch")
        report = {
            "schema_version": SCHEMA, "status": "PASS", "label_policy": LABEL_POLICY,
            "label_stats": data.label_stats, "queries": len(data.answers), "chunks": len(data.chunk_ids),
            "documents": len(data.all_docs), "dimension": DIMENSION,
            "inputs": {
                "train_sha256": sha256_file(train), "folds_sha256": sha256_file(folds),
                "preprocessing_manifest_sha256": sha256_file(preprocessing / "manifest.json"),
                "e5_manifest_sha256": sha256_file(data.e5_dir / "manifest.json"),
                "v3_manifest_sha256": sha256_file(data.v3_dir / "manifest.json"),
                "query_manifest_sha256": sha256_file(data.query_dir / "manifest.json"),
                "bm25_sidecar_sha256": sha256_file(sidecar), "code_sha256": sha256_file(Path(__file__)),
            },
        }
        atomic_json(output_dir / "REPORT.json", report)
        _write_manifest(output_dir=output_dir, stage="exp034-input-audit", inputs=report["inputs"],
                        config={"label_policy": LABEL_POLICY, "dimension": DIMENSION}, files=(output_dir / "REPORT.json",))
        log.set_telemetry({"queries": len(data.answers), "chunks": len(data.chunk_ids)})
    return report


def calibrate(*, data: RetrievalData, sidecar: Path, output_dir: Path, device: torch.device,
              query_limit: int | None = None) -> dict[str, Any]:
    qids = sorted(data.answers)
    if query_limit is not None:
        qids = sorted(qid for fold in sorted(data.folds) for qid in data.folds[fold][:query_limit])
    documents = data.document_tensor(device); identity = ResidualProjection().to(device)
    with stage_run(output_dir, "exp034-calibration", total=len(qids), v3_fingerprint=data.v3_manifest["content_fingerprint"]) as log:
        ranking_by_config = _exact_rankings_by_config(
            data=data, qids=qids, model=identity, documents=documents,
            configs=AGGREGATION_GRID, device=device,
        )
        selected: dict[str, AggregationConfig] = {}
        fold_metrics: dict[str, Any] = {}
        for outer, heldout in sorted(data.folds.items()):
            train_qids = [qid for qid in qids if qid not in set(heldout)]
            if not train_qids or not set(heldout) & set(qids): continue
            selected[outer] = choose_aggregation(ranking_by_config, data.answers, train_qids)
            fold_qids = [qid for qid in heldout if qid in set(qids)]
            fold_metrics[outer] = evaluate_rankings(ranking_by_config[selected[outer].key], data.answers, fold_qids)
        # Provenance-complete RRF uses dense ranks from the fixed union plus EXP-027's repaired BM25 ranks.
        fixed_rows = {str(row["qid"]): row for row in read_jsonl(ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl") if str(row["qid"]) in set(qids)}
        bm25 = _load_bm25(sidecar)
        e5 = {qid: [str(item["doc_id"]) for item in sorted(row["candidates"], key=lambda x: int(x.get("sources", {}).get("e5", {}).get("rank", 10**9))) if "e5" in item.get("sources", {})] for qid, row in fixed_rows.items()}
        rrf_metrics: dict[str, Any] = {}
        rrf_rankings: dict[str, dict[str, list[str]]] = {}
        for alpha, constant in RRF_GRID:
            key = f"a{alpha:g}_k{constant}"; values = {qid: _rrf_rank(e5[qid], bm25[qid], alpha, constant) for qid in qids}
            rrf_rankings[key] = values; rrf_metrics[key] = evaluate_rankings(values, data.answers, qids)
        selected_rrf: dict[str, str] = {}; best_rrf_per_fold: dict[str, Any] = {}; chosen_rrf: dict[str, list[str]] = {}
        qid_set = set(qids)
        for fold, heldout in sorted(data.folds.items()):
            fold_qids = [qid for qid in heldout if qid in qid_set]
            train_qids = [qid for qid in qids if qid not in set(heldout)]
            if not fold_qids or not train_qids:
                continue
            train_metrics = {key: evaluate_rankings(values, data.answers, train_qids) for key, values in rrf_rankings.items()}
            selected_rrf[fold] = max(
                sorted(train_metrics),
                key=lambda key: (train_metrics[key]["recall@5"], train_metrics[key]["recall@32"],
                                 train_metrics[key]["mrr@5"], -sorted(train_metrics).index(key)),
            )
            best_rrf_per_fold[fold] = evaluate_rankings(rrf_rankings[selected_rrf[fold]], data.answers, fold_qids)
            chosen_rrf.update({qid: rrf_rankings[selected_rrf[fold]][qid] for qid in fold_qids})
        chosen_rankings = {qid: ranking_by_config[selected[data.fold_for[qid]].key][qid] for qid in qids}
        write_jsonl(output_dir / "selected_rankings.jsonl", ({"schema_version": SCHEMA, "qid": qid, "doc_ids": chosen_rankings[qid]} for qid in qids))
        write_jsonl(output_dir / "rrf_rankings.jsonl", ({"schema_version": SCHEMA, "qid": qid, "doc_ids": chosen_rrf[qid]} for qid in qids))
        e5_metrics = evaluate_rankings(chosen_rankings, data.answers, qids)
        bm25_metrics = evaluate_rankings(bm25, data.answers, qids)
        report = {
            "schema_version": SCHEMA, "status": "PASS" if query_limit is None else "SMOKE_ONLY",
            "queries": len(qids), "aggregation_grid": [config.as_dict() for config in AGGREGATION_GRID],
            "selected_by_outer": {fold: config.as_dict() for fold, config in selected.items()},
            "fold_metrics": fold_metrics,
            "baselines": {"e5": e5_metrics, "bm25": bm25_metrics},
            "best_rrf": {"config": "fold_isolated", "selected_by_outer": selected_rrf,
                         "metrics": evaluate_rankings(chosen_rrf, data.answers, qids), "per_fold": best_rrf_per_fold},
            "rrf_grid_metrics": rrf_metrics, "smoke_limit": query_limit,
        }
        atomic_json(output_dir / "REPORT.json", report)
        _write_manifest(output_dir=output_dir, stage="exp034-calibration",
                        inputs={"e5_cache_fingerprint": data.e5_manifest["cache_fingerprint"], "label_fingerprint": data.label_stats["label_fingerprint"], "sidecar_sha256": sha256_file(sidecar)},
                        config={"aggregation_grid": [value.as_dict() for value in AGGREGATION_GRID], "rrf_grid": list(RRF_GRID), "smoke_limit": query_limit},
                        files=(output_dir / "REPORT.json", output_dir / "selected_rankings.jsonl", output_dir / "rrf_rankings.jsonl"))
        log.set_telemetry({"queries": len(qids), "rrf_selection": "fold_isolated"})
    return report


def _top_indices_for_doc(query: torch.Tensor, documents: torch.Tensor, indices: Sequence[int], count: int = 2) -> list[int]:
    idx = torch.tensor(list(indices), dtype=torch.long, device=documents.device)
    with torch.inference_mode(): values = documents.index_select(0, idx) @ query
    positions = torch.topk(values, min(count, values.numel())).indices
    return [int(idx[int(position)].item()) for position in positions]


def _mine_epoch(*, data: RetrievalData, train_qids: Sequence[str], model: ResidualProjection,
                documents: torch.Tensor, bm25: Mapping[str, Sequence[str]], config: AggregationConfig,
                device: torch.device, epoch: int, output_path: Path) -> dict[str, Any]:
    exact = _exact_records(data=data, qids=train_qids, model=model, documents=documents,
                           configs=(config,), device=device, limit=64)[config.key]
    dense = _rankings(exact); rows = []
    model.eval()
    with torch.inference_mode():
        projected = model(data.query_vectors(list(train_qids), device))
        for qid, query in zip(train_qids, projected):
            positive = [_top_indices_for_doc(query, documents, data.doc_to_indices[doc_id]) for doc_id in sorted(data.answers[qid])]
            negative_ids = mine_negative_ids(dense_ranking=dense[qid], bm25_ranking=bm25[qid], gold=data.answers[qid],
                                             all_docs=data.all_docs, seed=SEED + epoch * 1_000_003 + _stable_qid_seed(qid))
            negative = [_top_indices_for_doc(query, documents, data.doc_to_indices[doc_id]) for doc_id in negative_ids]
            rows.append({"schema_version": SCHEMA, "qid": qid, "positive": positive, "negative": negative,
                         "negative_doc_ids": negative_ids})
    write_jsonl(output_path, rows)
    return {str(row["qid"]): row for row in rows}


def _score_selected_parent(query: torch.Tensor, documents: torch.Tensor, indices: Sequence[int],
                           data: RetrievalData, config: AggregationConfig) -> torch.Tensor:
    idx = torch.tensor(list(indices), dtype=torch.long, device=documents.device)
    values = documents.index_select(0, idx) @ query
    score = parent_score(values, config.aggregation)
    doc_id = data.chunk_docs[int(indices[0])]
    score = score - config.beta * math.log1p(data.chunk_counts[doc_id])
    if data.chunk_kinds[int(indices[0])] == "article": score = score + config.gamma
    return score


def train_projection(*, data: RetrievalData, train_qids: Sequence[str], eval_qids: Sequence[str],
                     bm25: Mapping[str, Sequence[str]], config: AggregationConfig, output_dir: Path,
                     device: torch.device, resume: bool, smoke: bool = False) -> dict[str, Any]:
    requested_train_qids = sorted(map(str, train_qids))
    eval_qids = sorted(map(str, eval_qids))
    if set(requested_train_qids) & set(eval_qids): raise ValueError("train/eval QIDs overlap")
    train_qids = [qid for qid in requested_train_qids if data.answers[qid]]
    if not train_qids or not eval_qids: raise ValueError("train/eval split is empty")
    _set_seed(); documents = data.document_tensor(device); model = ResidualProjection().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    checkpoint = output_dir / "checkpoint.pt"; start_epoch = 0
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        expected = _hash({"train": train_qids, "eval": eval_qids, "config": config.as_dict()})
        if saved.get("split_fingerprint") != expected: raise RuntimeError("resume checkpoint split/config mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start_epoch = int(saved["epoch"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    epochs = 1 if smoke else EPOCHS
    with stage_run(output_dir, "exp034-train-projection", total=epochs, v3_fingerprint=data.v3_manifest["content_fingerprint"]) as log:
        for epoch in range(start_epoch, epochs):
            mining_path = output_dir / f"mining_epoch_{epoch}.jsonl"
            mined = _mine_epoch(data=data, train_qids=train_qids, model=model, documents=documents, bm25=bm25,
                                config=config, device=device, epoch=epoch, output_path=mining_path)
            order = list(train_qids); random.Random(SEED + epoch).shuffle(order); model.train(); losses = []
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start:start + BATCH_SIZE]; optimizer.zero_grad(set_to_none=True); query = model(data.query_vectors(batch, device)); batch_loss = []
                for vector, qid in zip(query, batch):
                    row = mined[qid]
                    positives = torch.stack([_score_selected_parent(vector, documents, values, data, config) for values in row["positive"]])
                    negatives = torch.stack([_score_selected_parent(vector, documents, values, data, config) for values in row["negative"]])
                    batch_loss.append(lse_pairwise_loss(positives, negatives))
                loss = torch.stack(batch_loss).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); losses.append(float(loss.detach().cpu()))
            payload = {
                "schema_version": SCHEMA, "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "split_fingerprint": _hash({"train": train_qids, "eval": eval_qids, "config": config.as_dict()}),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "mean_loss": float(np.mean(losses)),
            }
            _atomic_torch_save(checkpoint, payload); log.status(stage="exp034-train-projection", state="RUNNING", completed=epoch + 1, total=epochs, emit_log=True)
        records = _exact_records(data=data, qids=eval_qids, model=model, documents=documents, configs=(config,), device=device, limit=MAX_OUTPUT)[config.key]
        for row in records: row["fold"] = data.fold_for[row["qid"]]
        write_jsonl(output_dir / "predictions.jsonl", records); rankings = _rankings(records); metrics = evaluate_rankings(rankings, data.answers, eval_qids)
        report = {
            "schema_version": SCHEMA, "status": "SMOKE_ONLY" if smoke else "PASS", "train_qids": len(train_qids),
            "excluded_non_evaluable_train_qids": len(requested_train_qids) - len(train_qids),
            "eval_qids": len(eval_qids), "train_qids_sha256": _hash(train_qids), "eval_qids_sha256": _hash(eval_qids),
            "aggregation": config.as_dict(), "hyperparameters": {"rank": RANK, "epochs": epochs, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "temperature": TEMPERATURE, "seed": SEED},
            "metrics": metrics, "checkpoint_sha256": sha256_file(checkpoint),
        }
        atomic_json(output_dir / "REPORT.json", report); log.set_telemetry({"metrics": metrics})
        _write_manifest(output_dir=output_dir, stage="exp034-train-projection",
                        inputs={"e5_cache_fingerprint": data.e5_manifest["cache_fingerprint"], "label_fingerprint": data.label_stats["label_fingerprint"],
                                "train_qids_sha256": report["train_qids_sha256"], "eval_qids_sha256": report["eval_qids_sha256"]},
                        config={"aggregation": config.as_dict(), **report["hyperparameters"]},
                        files=(checkpoint, output_dir / "predictions.jsonl", output_dir / "REPORT.json"))
    return report


def _router_features(row: Mapping[str, Any], query: str, bm25_ranking: Sequence[str]) -> list[float]:
    candidates = list(row["candidates"]); scores = np.array([float(item["score"]) for item in candidates], dtype=np.float64)
    if len(scores) < TIERS[-1]: raise ValueError("router ranking has fewer than 64 documents")
    bm = set(map(str, bm25_ranking)); features = [float(len(query.split())), float(scores.mean()), float(scores.std())]
    for k in (5, 16, 24, 32, 40, 50):
        features.extend((float(scores[k - 1]), float(scores[k - 1] - scores[k]), float(np.mean(scores[:k]))))
    features.extend((float(scores[0] - scores[-1]), float(np.polyfit(np.arange(64), scores[:64], 1)[0])))
    for k in (16, 32, 64): features.append(float(sum(str(item["doc_id"]) in bm for item in candidates[:k])) / k)
    evidence = candidates[0].get("evidence", [])
    features.append(float(evidence[0]["score"] - evidence[-1]["score"]) if len(evidence) > 1 else 0.0)
    features.append(float(evidence[0].get("kind") == "article") if evidence else 0.0)
    return features


def _adaptive_fold_impl(*, data: RetrievalData, outer: str, fixed_dir: Path,
                        calibration: Mapping[str, Any], sidecar: Path, output_dir: Path,
                        device: torch.device, resume: bool, smoke: bool = False) -> dict[str, Any]:
    bm25 = _load_bm25(sidecar)
    config = AggregationConfig.from_dict(calibration["selected_by_outer"][outer]); inner_records: list[dict[str, Any]] = []
    for inner, train_qids, eval_qids in crossfit_splits(data.folds, outer):
        if smoke: train_qids, eval_qids = train_qids[:32], eval_qids[:16]
        inner_dir = output_dir / "inner" / inner
        train_projection(data=data, train_qids=train_qids, eval_qids=eval_qids, bm25=bm25, config=config,
                         output_dir=inner_dir, device=device, resume=resume, smoke=smoke)
        inner_records.extend(read_jsonl(inner_dir / "predictions.jsonl"))
    X, y, rankings, gold, fold_names, qids = [], [], [], [], [], []
    for row in inner_records:
        qid = str(row["qid"])
        if not data.answers[qid]:
            continue
        ranking = [str(item["doc_id"]) for item in row["candidates"]]
        X.append(_router_features(row, data.queries[qid], bm25[qid])); y.append(TIERS.index(required_tier(ranking, data.answers[qid])))
        rankings.append(ranking); gold.append(data.answers[qid]); fold_names.append(data.fold_for[qid]); qids.append(qid)
    router = lgb.LGBMClassifier(objective="multiclass", num_class=len(TIERS), n_estimators=200, learning_rate=0.05,
                               num_leaves=15, min_child_samples=30, reg_lambda=2.0, random_state=SEED,
                               n_jobs=1, verbosity=-1)
    router.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
    predicted = router.predict(np.asarray(X, dtype=np.float32)).astype(int).tolist()
    # Smoke validates the router/data path only; its tiny sample is not allowed
    # to make or block a metric claim.
    inner_recall_at_64: dict[str, float] = {}
    for fold in sorted(set(fold_names)):
        values = [len(set(ranking[:64]) & labels) / len(labels)
                  for ranking, labels, name in zip(rankings, gold, fold_names) if name == fold]
        inner_recall_at_64[fold] = float(np.mean(values))
    inner_gate_pass = True
    try:
        offset = calibrate_tier_offset(predicted_indices=predicted, rankings=rankings, gold=gold,
                                       fold_names=fold_names, floor=0.0 if smoke else FIXED_FLOOR)
    except RuntimeError:
        # Failure even at the maximum supported tier is an expected rejection,
        # not an infrastructure failure.  Emit K=64 diagnostics and stop expansion.
        inner_gate_pass = False
        offset = len(TIERS) - 1
    outer_records = list(read_jsonl(fixed_dir / outer / "predictions.jsonl")); outer_X = []
    for row in outer_records:
        qid = str(row["qid"]); outer_X.append(_router_features(row, data.queries[qid], bm25[qid]))
    outer_pred = router.predict(np.asarray(outer_X, dtype=np.float32)).astype(int).tolist(); budgets = apply_tier_offset(outer_pred, offset)
    output_rows = []; recalls = []
    for row, budget in zip(outer_records, budgets):
        qid = str(row["qid"]); ranking = [str(item["doc_id"]) for item in row["candidates"]]
        if data.answers[qid]:
            recalls.append(len(set(ranking[:budget]) & data.answers[qid]) / len(data.answers[qid]))
        output_rows.append({"schema_version": SCHEMA, "qid": qid, "selected_k": budget, "doc_ids": ranking[:budget]})
    output_dir.mkdir(parents=True, exist_ok=True); write_jsonl(output_dir / "adaptive_predictions.jsonl", output_rows)
    model_path = output_dir / "router.txt"; router.booster_.save_model(str(model_path))
    distribution = Counter(map(str, budgets)); report = {
        "schema_version": SCHEMA,
        "status": "SMOKE_ONLY" if smoke else "PASS" if inner_gate_pass else "REJECTED_INNER_GATE",
        "outer": outer, "inner_crossfit": True, "inner_gate_pass": inner_gate_pass,
        "inner_recall@64_by_fold": inner_recall_at_64,
        "router_train_qids": len(qids), "offset": offset if inner_gate_pass else None,
        "diagnostic_applied_offset": offset, "retained_recall@selected_k": float(np.mean(recalls)),
        "mean_k": float(np.mean(budgets)), "p50_k": float(np.percentile(budgets, 50)), "p90_k": float(np.percentile(budgets, 90)),
        "max_k": int(max(budgets)), "tier_distribution": dict(sorted(distribution.items())), "pairs_saved_vs_64": int(sum(64 - k for k in budgets)),
        "router_sha256": sha256_file(model_path),
    }
    atomic_json(output_dir / "REPORT.json", report)
    _write_manifest(output_dir=output_dir, stage="exp034-adaptive-fold",
                    inputs={"label_fingerprint": data.label_stats["label_fingerprint"], "outer": outer},
                    config={"tiers": list(TIERS), "floor": FIXED_FLOOR, "inner_crossfit": True},
                    files=(output_dir / "adaptive_predictions.jsonl", model_path, output_dir / "REPORT.json"))
    return report


def adaptive_fold(*, data: RetrievalData, outer: str, fixed_dir: Path,
                  calibration: Mapping[str, Any], sidecar: Path, output_dir: Path,
                  device: torch.device, resume: bool, smoke: bool = False) -> dict[str, Any]:
    with stage_run(output_dir, "exp034-adaptive-fold", total=4,
                   v3_fingerprint=data.v3_manifest["content_fingerprint"]) as log:
        report = _adaptive_fold_impl(
            data=data, outer=outer, fixed_dir=fixed_dir, calibration=calibration,
            sidecar=sidecar, output_dir=output_dir, device=device, resume=resume, smoke=smoke,
        )
        log.set_telemetry({
            "outer": outer, "retained_recall": report["retained_recall@selected_k"],
            "mean_k": report["mean_k"], "max_k": report["max_k"],
        })
    return report


def _fixed_gate(report: Mapping[str, Any], rrf_fold_metric: Mapping[str, Any]) -> bool:
    return float(report["metrics"]["recall@32"]) >= FIXED_FLOOR and float(report["metrics"]["recall@5"]) >= float(rrf_fold_metric["recall@5"]) - RRF_TOLERANCE


def _adaptive_gate(report: Mapping[str, Any]) -> bool:
    return (report.get("status") == "PASS" and bool(report.get("inner_gate_pass", True))
            and float(report["retained_recall@selected_k"]) >= FIXED_FLOOR
            and float(report["mean_k"]) <= ADAPTIVE_MEAN_K_MAX and int(report["max_k"]) <= TIERS[-1])


def _fixed_final_gate(*, reports: Mapping[str, Mapping[str, Any]],
                      predictions: Mapping[str, Sequence[str]], data: RetrievalData,
                      calibration: Mapping[str, Any]) -> bool:
    if len(reports) != len(data.folds):
        return False
    global_metrics = evaluate_rankings(predictions, data.answers, sorted(predictions))
    rrf = calibration["best_rrf"]
    return (
        all(float(row["metrics"]["recall@32"]) >= FIXED_FLOOR for row in reports.values())
        and float(global_metrics["recall@5"]) >= float(rrf["metrics"]["recall@5"])
        and all(float(reports[fold]["metrics"]["recall@5"])
                >= float(rrf["per_fold"][fold]["recall@5"]) - RRF_TOLERANCE
                for fold in reports)
    )


def _breakdowns(*, data: RetrievalData, predictions: Mapping[str, Sequence[str]], budgets: Mapping[str, int],
                rrf_rankings: Mapping[str, Sequence[str]], sidecar: Path) -> dict[str, Any]:
    """Occurrence-level diagnostics for every evaluable predicted query."""
    fixed_rows = {str(row["qid"]): row for row in read_jsonl(ROOT / "cache" / "exp022_e5_bm25_union" / "train_oof_candidates.jsonl") if str(row["qid"]) in predictions}
    bm25 = _load_bm25(sidecar); all_counts = np.asarray(list(data.chunk_counts.values()), dtype=np.int64)
    decile_edges = np.percentile(all_counts, np.arange(10, 100, 10))
    counters: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for qid, prediction in predictions.items():
        budget = int(budgets[qid]); chosen = set(map(str, prediction[:budget])); gold = data.answers[qid]
        e5_docs = {str(item["doc_id"]) for item in fixed_rows[qid]["candidates"] if "e5" in item.get("sources", {})}
        bm_docs = set(bm25[qid]); base_rank = {str(doc): rank for rank, doc in enumerate(rrf_rankings[qid], 1)}
        for doc_id in gold:
            hit = int(doc_id in chosen)
            source = "both" if doc_id in e5_docs and doc_id in bm_docs else "e5_only" if doc_id in e5_docs else "bm25_only" if doc_id in bm_docs else "neither"
            parse = str(data.doc_metadata[doc_id].get("parse_mode", "unknown"))
            decile = f"d{min(int(np.searchsorted(decile_edges, data.chunk_counts[doc_id], side='right')) + 1, 10)}"
            count_group = "single_gold" if len(gold) == 1 else "multi_gold"
            rank = base_rank.get(doc_id)
            rank_group = "outside64" if rank is None else "1-16" if rank <= 16 else "17-32" if rank <= 32 else "33-50" if rank <= 50 else "51-64"
            for family, value in (("source", source), ("parse_mode", parse), ("chunk_count_decile", decile),
                                  ("gold_cardinality", count_group), ("rrf_rank_bucket", rank_group)):
                counters[family][value][0] += hit; counters[family][value][1] += 1
    return {
        family: {value: {"hits": counts[0], "occurrences": counts[1], "recall": counts[0] / counts[1]}
                 for value, counts in sorted(values.items())}
        for family, values in sorted(counters.items())
    }


def build_report(*, data: RetrievalData, calibration: Mapping[str, Any], fixed_root: Path,
                 adaptive_root: Path, sidecar: Path, output_path: Path) -> dict[str, Any]:
    fixed = {fold: _json(fixed_root / fold / "REPORT.json") for fold in sorted(data.folds) if (fixed_root / fold / "REPORT.json").exists()}
    adaptive = {fold: _json(adaptive_root / fold / "REPORT.json") for fold in sorted(data.folds) if (adaptive_root / fold / "REPORT.json").exists()}
    rrf_per_fold = calibration.get("best_rrf", {}).get("per_fold", {})
    fixed_predictions: dict[str, list[str]] = {}
    for fold in fixed:
        fixed_predictions.update(_rankings(list(read_jsonl(fixed_root / fold / "predictions.jsonl"))))
    fixed_global = evaluate_rankings(fixed_predictions, data.answers, sorted(fixed_predictions)) if fixed_predictions else {}
    fixed_accepted = _fixed_final_gate(
        reports=fixed, predictions=fixed_predictions, data=data, calibration=calibration
    )
    adaptive_accepted = (
        len(adaptive) == len(data.folds)
        and all(float(row["retained_recall@selected_k"]) >= FIXED_FLOOR for row in adaptive.values())
        and float(np.mean([row["mean_k"] for row in adaptive.values()])) <= ADAPTIVE_MEAN_K_MAX
        and max(int(row["max_k"]) for row in adaptive.values()) == TIERS[-1]
    )
    final_predictions: dict[str, list[str]] = {}; final_budgets: dict[str, int] = {}
    if fixed_accepted:
        final_predictions = fixed_predictions; final_budgets = {qid: 32 for qid in final_predictions}
    elif adaptive_accepted:
        for fold in adaptive:
            for row in read_jsonl(adaptive_root / fold / "adaptive_predictions.jsonl"):
                final_predictions[str(row["qid"])] = list(map(str, row["doc_ids"])); final_budgets[str(row["qid"])] = int(row["selected_k"])
    elif adaptive:
        for fold in adaptive:
            for row in read_jsonl(adaptive_root / fold / "adaptive_predictions.jsonl"):
                final_predictions[str(row["qid"])] = list(map(str, row["doc_ids"])); final_budgets[str(row["qid"])] = int(row["selected_k"])
    elif fixed_predictions:
        final_predictions = fixed_predictions; final_budgets = {qid: 32 for qid in final_predictions}
    rrf_rankings = {str(row["qid"]): list(map(str, row["doc_ids"])) for row in read_jsonl(output_path.parent / "calibration" / "rrf_rankings.jsonl")} if (output_path.parent / "calibration" / "rrf_rankings.jsonl").exists() else {}
    breakdowns = _breakdowns(data=data, predictions=final_predictions, budgets=final_budgets, rrf_rankings=rrf_rankings, sidecar=sidecar) if final_predictions and set(final_predictions) <= set(rrf_rankings) else {}
    report = {
        "schema_version": SCHEMA, "status": "ACCEPTED_FIXED" if fixed_accepted else "ACCEPTED_ADAPTIVE" if adaptive_accepted else "REJECTED",
        "label_policy": LABEL_POLICY, "label_fingerprint": data.label_stats["label_fingerprint"],
        "calibration": {"best_rrf": calibration.get("best_rrf"), "selected_by_outer": calibration.get("selected_by_outer")},
        "fixed": {"per_fold": fixed, "global_metrics": fixed_global}, "adaptive": adaptive,
        "breakdowns": breakdowns, "breakdowns_diagnostic_only": not (fixed_accepted or adaptive_accepted),
        "acceptance": {"fixed": fixed_accepted, "adaptive": adaptive_accepted, "fixed_floor": FIXED_FLOOR, "adaptive_mean_k_max": ADAPTIVE_MEAN_K_MAX},
        "scope": {"heavy_reranker": "not_run", "public_submission": "not_run", "corpus_rebuild": "not_run"},
    }
    atomic_json(output_path, report)
    _write_manifest(output_dir=output_path.parent, stage="exp034-report",
                    inputs={"label_fingerprint": data.label_stats["label_fingerprint"], "calibration_fingerprint": _hash(calibration)},
                    config={"fixed_floor": FIXED_FLOOR, "adaptive_mean_k_max": ADAPTIVE_MEAN_K_MAX}, files=(output_path,))
    return report


def _run_status(path: Path, **values: Any) -> None:
    atomic_json(path, {"schema_version": SCHEMA, "pid": os.getpid(), "updated_at": time.time(), **values})


def overnight(*, data: RetrievalData, cache_root: Path, results_root: Path, sidecar: Path,
              device: torch.device, resume: bool) -> dict[str, Any]:
    status_path = results_root / "RUN_STATUS.json"; results_root.mkdir(parents=True, exist_ok=True)
    try:
        _run_status(status_path, state="RUNNING", phase="audit")
        audit_inputs(data=data, train=ROOT / "public_test_dataset" / "train.json", folds=ROOT / "cache" / "cv_folds.json",
                     preprocessing=ROOT / "cache" / "final_preprocessed_v2", sidecar=sidecar, output_dir=results_root / "audit")
        _run_status(status_path, state="RUNNING", phase="calibration")
        calibration_path = results_root / "calibration" / "REPORT.json"
        cached_calibration = _json(calibration_path) if resume and calibration_path.exists() else None
        calibration = cached_calibration if cached_calibration and cached_calibration.get("status") == "PASS" else calibrate(data=data, sidecar=sidecar, output_dir=results_root / "calibration", device=device)
        bm25 = _load_bm25(sidecar); fixed_root = cache_root / "fixed"; adaptive_root = results_root / "adaptive"

        def train_outer(fold: str) -> dict[str, Any]:
            config = AggregationConfig.from_dict(calibration["selected_by_outer"][fold])
            train_qids = [qid for name, qids in data.folds.items() if name != fold for qid in qids]
            return train_projection(data=data, train_qids=train_qids, eval_qids=data.folds[fold], bm25=bm25, config=config,
                                    output_dir=fixed_root / fold, device=device, resume=resume)

        _run_status(status_path, state="RUNNING", phase="fixed-fold-0"); fold0 = train_outer("fold_0")
        rrf_rows = {str(row["qid"]): list(map(str, row["doc_ids"])) for row in read_jsonl(results_root / "calibration" / "rrf_rankings.jsonl")}
        rrf_fold0 = evaluate_rankings(rrf_rows, data.answers, data.folds["fold_0"])
        need_adaptive = not _fixed_gate(fold0, rrf_fold0)
        if not need_adaptive:
            for fold in sorted(data.folds):
                if fold == "fold_0": continue
                _run_status(status_path, state="RUNNING", phase=f"fixed-{fold}"); train_outer(fold)
            fixed_reports = {fold: _json(fixed_root / fold / "REPORT.json") for fold in data.folds}
            fixed_predictions: dict[str, list[str]] = {}
            for fold in data.folds:
                fixed_predictions.update(_rankings(list(read_jsonl(fixed_root / fold / "predictions.jsonl"))))
            need_adaptive = not _fixed_final_gate(
                reports=fixed_reports, predictions=fixed_predictions, data=data, calibration=calibration
            )
        if need_adaptive:
            _run_status(status_path, state="RUNNING", phase="adaptive-fold-0")
            adaptive0 = adaptive_fold(data=data, outer="fold_0", fixed_dir=fixed_root, calibration=calibration, sidecar=sidecar,
                                      output_dir=adaptive_root / "fold_0", device=device, resume=resume)
            if _adaptive_gate(adaptive0):
                for fold in sorted(data.folds):
                    if fold == "fold_0": continue
                    if not (fixed_root / fold / "_SUCCESS.json").exists():
                        _run_status(status_path, state="RUNNING", phase=f"fixed-for-adaptive-{fold}"); train_outer(fold)
                    _run_status(status_path, state="RUNNING", phase=f"adaptive-{fold}")
                    adaptive_fold(data=data, outer=fold, fixed_dir=fixed_root, calibration=calibration, sidecar=sidecar,
                                  output_dir=adaptive_root / fold, device=device, resume=resume)
        final = build_report(data=data, calibration=calibration, fixed_root=fixed_root, adaptive_root=adaptive_root, sidecar=sidecar,
                             output_path=results_root / "REPORT.json")
        _run_status(status_path, state="COMPLETE" if final["status"].startswith("ACCEPTED") else "REJECTED", phase="report", result=final["status"])
        return final
    except BaseException as error:
        _run_status(status_path, state="FAILED", phase="exception", error=f"{type(error).__name__}: {error}")
        raise


def _paths(args: argparse.Namespace) -> tuple[RetrievalData, Path, Path, Path, torch.device]:
    data = RetrievalData(e5_dir=args.e5_dir, query_dir=args.query_dir, v3_dir=args.v3_dir, train=args.train,
                         folds_path=args.folds, preprocessing=args.preprocessing)
    return data, args.cache_root, args.results_root, args.sidecar, torch.device(args.device)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(__import__("sys").stdout, "reconfigure"): __import__("sys").stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", choices=("audit", "calibrate", "train-fold", "adaptive-fold", "overnight", "report"))
    parser.add_argument("--e5-dir", type=Path, default=ROOT / "cache" / "e5_final_v1")
    parser.add_argument("--query-dir", type=Path, default=ROOT / "cache" / "exp021_e5_dense_candidates" / "query_embeddings")
    parser.add_argument("--v3-dir", type=Path, default=ROOT / "cache" / "structural_v3_e5_final_v1")
    parser.add_argument("--train", type=Path, default=ROOT / "public_test_dataset" / "train.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "cache" / "cv_folds.json")
    parser.add_argument("--preprocessing", type=Path, default=ROOT / "cache" / "final_preprocessed_v2")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "cache" / "exp027_lambdamart_shortlist" / "provenance" / "provenance_sidecar.jsonl")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "exp034_shallow_retrieval")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "exp034_shallow_retrieval")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--outer", choices=tuple(f"fold_{i}" for i in range(5)))
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args(argv); data, cache_root, results_root, sidecar, device = _paths(args)
    if args.stage == "audit": result = audit_inputs(data=data, train=args.train, folds=args.folds, preprocessing=args.preprocessing, sidecar=sidecar, output_dir=results_root / "audit")
    elif args.stage == "calibrate":
        calibration_dir = results_root / "smoke" / "calibration" if args.smoke_limit is not None else results_root / "calibration"
        result = calibrate(data=data, sidecar=sidecar, output_dir=calibration_dir, device=device, query_limit=args.smoke_limit)
    elif args.stage == "train-fold":
        if not args.outer: raise SystemExit("--outer is required")
        smoke = args.smoke_limit is not None
        calibration_dir = results_root / "smoke" / "calibration" if smoke else results_root / "calibration"
        calibration = _json(calibration_dir / "REPORT.json"); config = AggregationConfig.from_dict(calibration["selected_by_outer"][args.outer]); bm25 = _load_bm25(sidecar)
        train_qids = [qid for fold, qids in data.folds.items() if fold != args.outer for qid in qids]; eval_qids = data.folds[args.outer]
        if smoke: train_qids, eval_qids = train_qids[:args.smoke_limit], eval_qids[:max(8, args.smoke_limit // 2)]
        train_dir = cache_root / "smoke" / "fixed" / args.outer if smoke else cache_root / "fixed" / args.outer
        result = train_projection(data=data, train_qids=train_qids, eval_qids=eval_qids, bm25=bm25, config=config, output_dir=train_dir, device=device, resume=args.resume, smoke=smoke)
    elif args.stage == "adaptive-fold":
        if not args.outer: raise SystemExit("--outer is required")
        smoke = args.smoke_limit is not None
        fixed_dir = cache_root / "smoke" / "fixed" if smoke else cache_root / "fixed"
        calibration_dir = results_root / "smoke" / "calibration" if smoke else results_root / "calibration"
        adaptive_dir = results_root / "smoke" / "adaptive" / args.outer if smoke else results_root / "adaptive" / args.outer
        result = adaptive_fold(data=data, outer=args.outer, fixed_dir=fixed_dir, calibration=_json(calibration_dir / "REPORT.json"), sidecar=sidecar, output_dir=adaptive_dir, device=device, resume=args.resume, smoke=smoke)
    elif args.stage == "overnight": result = overnight(data=data, cache_root=cache_root, results_root=results_root, sidecar=sidecar, device=device, resume=args.resume)
    else: result = build_report(data=data, calibration=_json(results_root / "calibration" / "REPORT.json"), fixed_root=cache_root / "fixed", adaptive_root=results_root / "adaptive", sidecar=sidecar, output_path=results_root / "REPORT.json")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
