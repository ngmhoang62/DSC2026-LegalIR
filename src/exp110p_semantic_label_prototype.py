"""EXP-110P: fold-safe semantic label-prototype/cache specialist.

The module is deliberately independent of the EXP-109C late-interaction
implementation.  It contains the small, auditable core used by the Colab
notebook: canonical labels, strict contexts, frozen E5 similarity, prototype
features, protected residuals, metrics/gates, and verified shard I/O.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA = "legalir.exp110p_semantic_label_prototype.v1"
LABEL_POLICY = "canonical_duplicate_alias_drop_empty_passage_v1"
LABEL_FINGERPRINT = "9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9"
FOLD0 = "fold_0"
INNER_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
EXPECTED_QIDS = 7000
EXPECTED_EVALUABLE = 6991
DIMENSION = 1024
MISSING_SENTINEL = -2.0
SOURCES = ("vietlegal_e5", "vnlegal_lal", "bm25")
# The locked EXP-109B LambdaMART matrix had a fixed Harrier block even though
# the cached-fusion pilot routed only E5, LAL and BM25 rankings.  Keep the
# omitted source as explicit all-zero/missing columns; removing it changes the
# model's column indices and makes anchor reproduction invalid.
ANCHOR_FEATURE_SOURCES = ("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal", "bm25")
SOFT_POLICY_GRID = tuple((n, beta, gamma) for n in (16, 32, 64) for beta in (5.0, 10.0, 20.0) for gamma in (0.0, 0.5, 1.0))
RESIDUAL_CONFIG_GRID = tuple((alpha, quantile) for alpha in (0.05, 0.10, 0.20, 0.30) for quantile in (0.50, 0.70, 0.85))
CURVE_KS = (1, 3, 5, 10, 16, 20, 50)

SOURCE_FEATURE_NAMES = tuple(
    f"{alias}_{suffix}"
    for alias in ("e5", "harrier", "lal", "bm25")
    for suffix in ("score", "rank", "recip", "z", "margin_rank1", "margin_rank5", "margin_rank10", "present")
)
BASE_FEATURE_NAMES = SOURCE_FEATURE_NAMES + (
    "source_agreement_top5",
    "source_agreement_top10",
    "source_agreement_top20",
    "dense_top1_score",
    "dense_top2_score",
    "dense_top1_top2_gap",
    "parent_chunk_count",
    "parent_token_length",
    "query_token_length",
)
PROTOTYPE_FEATURE_NAMES = (
    "proto_seen",
    "proto_similarity_missing",
    "proto_support_count",
    "proto_log_support_count",
    "proto_max_similarity",
    "proto_second_similarity",
    "proto_top2_mean",
    "proto_top3_mean",
    "proto_max_minus_second",
    "proto_max_minus_top3_mean",
    "proto_centroid_cosine",
    "proto_max_minus_centroid",
    "proto_support_dispersion",
    "proto_soft_vote_raw",
    "proto_soft_vote_cardinality_norm",
    "proto_soft_vote_frequency_norm",
    "proto_winning_document_margin",
    "proto_vote_entropy",
    "nearest_support_similarity",
    "nearest_minus_second_support_similarity",
    "top_neighbor_document_agreement",
    "prototype_winner_margin",
    "prototype_seen_candidate_fraction",
    "source_agreement_top5",
    "anchor_rank1_margin",
    "anchor_rank5_margin",
)
AUGMENTED_FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    name for name in PROTOTYPE_FEATURE_NAMES if name not in {"source_agreement_top5"}
)
FORBIDDEN_FEATURE_TOKENS = ("doc_id", "label", "target", "nearest", "gold", "answer", "query_id")


def validate_feature_contract(names: Sequence[str]) -> None:
    """Fail closed unless the frozen EXP-109B anchor column order is intact."""
    if tuple(names) != BASE_FEATURE_NAMES:
        raise ValueError("LambdaMART feature order/schema is not the locked EXP-109B contract")
    if len(names) != len(set(names)):
        raise ValueError("LambdaMART feature names must be unique")
    if any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS) for name in names):
        raise ValueError("forbidden LambdaMART feature")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n" if pretty else canonical_json(value) + "\n"
    atomic_text(path, body)


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as handle:
            for record in records:
                handle.write(canonical_json(dict(record)) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _canonical_answers(
    train: Mapping[str, Mapping[str, Any]],
    exclusions_rows: Sequence[Mapping[str, Any]],
    impact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    exclusions = {str(row["doc_id"]): row for row in exclusions_rows}
    if len(exclusions) != len(exclusions_rows):
        raise ValueError("duplicate document id in exclusions")
    declared = {
        str(row["query_id"]): {str(x) for x in row.get("intentionally_excluded_gold_ids", [])}
        for row in impact_rows
    }
    if len(declared) != len(impact_rows):
        raise ValueError("duplicate query id in label impact")
    answers: dict[str, set[str]] = {}
    observed: dict[str, set[str]] = {}
    duplicate_occurrences = 0
    empty_occurrences = 0
    for raw_qid, row in train.items():
        qid = str(raw_qid)
        gold: set[str] = set()
        removed: set[str] = set()
        for raw_doc_id in row.get("answer", []):
            doc_id = str(raw_doc_id)
            exclusion = exclusions.get(doc_id)
            if exclusion is None:
                gold.add(doc_id)
                continue
            removed.add(doc_id)
            reasons = {str(reason) for reason in exclusion.get("reasons", [])}
            replacement = exclusion.get("duplicate_retained_id")
            if reasons == {"exact_duplicate_raw_passage"}:
                if not replacement or str(replacement) in exclusions:
                    raise ValueError(f"invalid duplicate alias: {qid}/{doc_id}")
                gold.add(str(replacement))
                duplicate_occurrences += 1
            elif reasons == {"empty_passage"}:
                empty_occurrences += 1
            else:
                raise ValueError(f"unsupported exclusion policy: {qid}/{doc_id}/{sorted(reasons)}")
        answers[qid] = gold
        if removed:
            observed[qid] = removed
    if observed != declared:
        raise ValueError("canonical-gold impact sidecar mismatch")
    non_evaluable = sorted(qid for qid, values in answers.items() if not values)
    stats = {
        "policy": LABEL_POLICY,
        "queries": len(answers),
        "evaluable_queries": len(answers) - len(non_evaluable),
        "non_evaluable_queries": len(non_evaluable),
        "non_evaluable_qids": non_evaluable,
        "canonicalized_duplicate_occurrences": duplicate_occurrences,
        "dropped_empty_occurrences": empty_occurrences,
        "assignment_count": sum(len(values) for values in answers.values()),
        "observed_gold_documents": len({doc for values in answers.values() for doc in values}),
        "repeated_gold_documents": sum(
            count >= 2 for count in Counter(doc for values in answers.values() for doc in values).values()
        ),
        "label_fingerprint": content_hash({qid: sorted(values) for qid, values in sorted(answers.items())}),
    }
    return answers, stats


def canonical_labels(train_path: Path, exclusions_path: Path, impact_path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    raw = read_json(train_path)
    if not isinstance(raw, dict):
        raise ValueError("train must be a JSON object")
    impact_raw = read_json(impact_path) if Path(impact_path).suffix.lower() == ".json" else None
    impact_rows = impact_raw if isinstance(impact_raw, list) else list(read_jsonl(impact_path))
    return _canonical_answers(raw, read_json(exclusions_path), impact_rows)


def load_folds(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("folds must be a JSON object")
    # Preserve `cv_folds.json` order for EXP-109B anchor reproduction.  The
    # original LGBM ranking groups were fitted in that order; sorting changes
    # the row/group stream and therefore the deterministic histogram model.
    folds = {str(name): [str(qid) for qid in values] for name, values in raw.items()}
    fold_for: dict[str, str] = {}
    for name, qids in folds.items():
        for qid in qids:
            if qid in fold_for:
                raise ValueError(f"duplicate qid in folds: {qid}")
            fold_for[qid] = name
    return folds, fold_for


def validate_fold_partition(folds: Mapping[str, Sequence[str]], train_qids: Iterable[str], *, outer: str = FOLD0) -> None:
    train = {str(qid) for qid in train_qids}
    members = [str(qid) for qids in folds.values() for qid in qids]
    if len(members) != len(set(members)) or set(members) != train:
        raise ValueError("folds are not an exactly-once partition of train")
    if outer not in folds:
        raise ValueError(f"missing outer fold: {outer}")


def normalize_fp32(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(~np.isfinite(array)) or np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("embedding contains non-finite or zero vector")
    return array / norms


def renormalize_frozen(values: np.ndarray) -> np.ndarray:
    return normalize_fp32(np.asarray(values, dtype=np.float16).astype(np.float32, copy=False))


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(normalize_fp32(left) @ normalize_fp32(right).T, dtype=np.float32)


def stable_order(scores: Sequence[float], doc_ids: Sequence[str], limit: int | None = None) -> list[int]:
    if len(scores) != len(doc_ids):
        raise ValueError("scores/doc_ids length mismatch")
    order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), str(doc_ids[i])))
    return order if limit is None else order[:limit]


def stable_q_order(scores: Sequence[float], qids: Sequence[str], limit: int | None = None) -> list[int]:
    return stable_order(scores, qids, limit)


def query_zscore(values: Mapping[str, float] | Sequence[float]) -> dict[str, float] | np.ndarray:
    if isinstance(values, Mapping):
        keys = list(values)
        array = np.asarray([float(values[key]) for key in keys], dtype=np.float32)
        normalized = _zscore_array(array)
        return {key: float(value) for key, value in zip(keys, normalized)}
    return _zscore_array(np.asarray(values, dtype=np.float32))


def _zscore_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std()) if values.size else 0.0
    return np.zeros_like(values, dtype=np.float32) if std <= 1e-12 else (values - values.mean()) / std


def _source_alias(source: str) -> str:
    return {
        "vietlegal_e5": "e5",
        "vietlegal_harrier_0_6b": "harrier",
        "vnlegal_lal": "lal",
        "bm25": "bm25",
    }.get(str(source), str(source))


def _ranking_items(source_rows: Mapping[str, Any], source: str) -> Sequence[Mapping[str, Any]]:
    values = source_rows.get("sources", source_rows).get(source, []) if isinstance(source_rows, Mapping) else []
    if not isinstance(values, Sequence):
        raise ValueError(f"invalid source ranking: {source}")
    return values


def source_scalar_features(source: str, ranking: Sequence[Mapping[str, Any]], doc_id: str, *, depth: int = 50) -> dict[str, float]:
    alias = _source_alias(source)
    # Candidate membership is capped by `depth`, but the frozen EXP-109B
    # feature contract computes score moments/margins and presence from its
    # complete cached top-500 source list.  Do not truncate this context.
    full = list(ranking)
    scores = {str(item["doc_id"]): float(item.get("score", 0.0)) for item in full}
    ranks = {str(item["doc_id"]): int(item.get("rank", index + 1)) for index, item in enumerate(full)}
    present = str(doc_id) in ranks
    score = scores.get(str(doc_id), 0.0)
    rank = ranks.get(str(doc_id), depth + 1)
    # EXP-109B computes these source moments in float64 before the final
    # matrix is cast to float32.  Match that order exactly for anchor parity.
    array = np.asarray(list(scores.values()), dtype=np.float64)
    mean = float(array.mean()) if len(array) else 0.0
    std = float(array.std()) if len(array) else 0.0
    ordered = sorted(scores.values(), reverse=True)

    def margin(cutoff: int) -> float:
        reference = ordered[min(cutoff - 1, len(ordered) - 1)] if ordered else 0.0
        return float(reference - score) if present else 0.0

    return {
        f"{alias}_score": score,
        f"{alias}_rank": float(rank),
        f"{alias}_recip": 1.0 / rank if present else 0.0,
        f"{alias}_z": (score - mean) / std if present and std > 1e-12 else 0.0,
        f"{alias}_margin_rank1": margin(1),
        f"{alias}_margin_rank5": margin(5),
        f"{alias}_margin_rank10": margin(10),
        f"{alias}_present": float(present),
    }


def build_source_feature_record(
    qid: str,
    doc_id: str,
    source_rows: Mapping[str, Any],
    *,
    depth: int = 50,
    query_token_length: int = 0,
    parent_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"qid": str(qid), "doc_id": str(doc_id)}
    parent_metadata = parent_metadata or {}
    # `ANCHOR_FEATURE_SOURCES` is intentionally wider than `SOURCES`: the
    # former is the frozen EXP-109B feature contract, the latter is the actual
    # three-source candidate union of its cached pilot.
    for source in ANCHOR_FEATURE_SOURCES:
        record.update(source_scalar_features(source, _ranking_items(source_rows, source), doc_id, depth=depth))
    rank_maps = {
        source: {str(item["doc_id"]): int(item.get("rank", i + 1)) for i, item in enumerate(_ranking_items(source_rows, source))}
        for source in SOURCES
    }
    for cutoff, name in ((5, "source_agreement_top5"), (10, "source_agreement_top10"), (20, "source_agreement_top20")):
        record[name] = float(sum(ranks.get(str(doc_id), depth + 1) <= cutoff for ranks in rank_maps.values()))
    dense_scores = sorted(
        [
            float(item.get("score", 0.0))
            for source in SOURCES
            if source != "bm25"
            for item in _ranking_items(source_rows, source)
            if str(item.get("doc_id")) == str(doc_id)
        ],
        reverse=True,
    )
    record["dense_top1_score"] = dense_scores[0] if dense_scores else 0.0
    record["dense_top2_score"] = dense_scores[1] if len(dense_scores) > 1 else 0.0
    record["dense_top1_top2_gap"] = record["dense_top1_score"] - record["dense_top2_score"]
    record["query_token_length"] = float(query_token_length)
    record["parent_chunk_count"] = float(parent_metadata.get("parent_chunk_count", 0))
    record["parent_token_length"] = float(parent_metadata.get("parent_token_length", 0))
    return record


def candidate_union(source_rows: Mapping[str, Any], *, depth: int = 50) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for source in SOURCES:
        for item in _ranking_items(source_rows, source)[:depth]:
            doc_id = str(item["doc_id"])
            if doc_id not in seen:
                seen.add(doc_id)
                result.append(doc_id)
    return result


def source_feature_names() -> tuple[str, ...]:
    return BASE_FEATURE_NAMES


def source_feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    missing = set(BASE_FEATURE_NAMES) - set(record)
    if missing:
        raise ValueError(f"missing source feature(s): {sorted(missing)}")
    return np.asarray([float(record[name]) for name in BASE_FEATURE_NAMES], dtype=np.float32)


def augmented_feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    missing = set(AUGMENTED_FEATURE_NAMES) - set(record)
    if missing:
        raise ValueError(f"missing augmented feature(s): {sorted(missing)}")
    return np.asarray([float(record[name]) for name in AUGMENTED_FEATURE_NAMES], dtype=np.float32)


def combine_source_and_prototype_features(
    source_record: Mapping[str, Any],
    prototype_record: Mapping[str, float],
) -> dict[str, Any]:
    combined = dict(source_record)
    combined.update({name: float(value) for name, value in prototype_record.items() if name in PROTOTYPE_FEATURE_NAMES})
    return combined


@dataclass(frozen=True)
class SupportBank:
    support_qids: tuple[str, ...]
    vectors: np.ndarray
    support_by_doc: Mapping[str, np.ndarray]
    centroids: Mapping[str, np.ndarray]
    support_fingerprint: str
    labels_by_qid: Mapping[str, tuple[str, ...]]
    support_fold_names: tuple[str, ...] = ()

    def provenance(self) -> dict[str, Any]:
        return {
            "support_fold_names": list(self.support_fold_names),
            "support_query_count": len(self.support_qids),
            "self_excluded": True,
            "heldout_excluded": True,
            "support_fingerprint": self.support_fingerprint,
        }

    def sidecar(self) -> dict[str, list[str]]:
        return {self.support_fingerprint: list(self.support_qids)}


def build_support_bank(
    support_qids: Sequence[str],
    embeddings: np.ndarray,
    qid_to_row: Mapping[str, int],
    answers: Mapping[str, set[str]],
    *,
    forbidden_qids: Iterable[str] = (),
    support_fold_names: Iterable[str] = (),
) -> SupportBank:
    qids = tuple(sorted(str(qid) for qid in support_qids))
    forbidden = {str(qid) for qid in forbidden_qids}
    overlap = sorted(set(qids) & forbidden)
    if overlap:
        raise ValueError(f"forbidden qids entered support bank: {overlap[:5]}")
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate support qid")
    missing = [qid for qid in qids if qid not in qid_to_row or qid not in answers]
    if missing:
        raise ValueError(f"support qid missing embedding/label: {missing[:5]}")
    vectors = normalize_fp32(np.asarray(embeddings)[[qid_to_row[qid] for qid in qids]])
    by_doc: dict[str, list[int]] = defaultdict(list)
    for index, qid in enumerate(qids):
        for doc_id in sorted(answers[qid]):
            by_doc[str(doc_id)].append(index)
    support_by_doc = {doc: np.asarray(indices, dtype=np.int32) for doc, indices in sorted(by_doc.items())}
    centroids = {
        doc: normalize_fp32(vectors[indices].mean(axis=0, keepdims=True))[0]
        for doc, indices in support_by_doc.items()
    }
    labels_by_qid = {qid: tuple(sorted(str(doc) for doc in answers[qid])) for qid in qids}
    return SupportBank(qids, vectors, support_by_doc, centroids, content_hash({"support_qids": qids}), labels_by_qid, tuple(sorted({str(name) for name in support_fold_names})))


def inner_qids(folds: Mapping[str, Sequence[str]], *, outer: str = FOLD0) -> list[str]:
    if outer not in folds:
        raise ValueError(f"unknown outer fold: {outer}")
    return sorted(str(qid) for name, qids in folds.items() if name != outer for qid in qids)


def support_qids_for_context(
    folds: Mapping[str, Sequence[str]],
    *,
    outer: str = FOLD0,
    heldout: str | None = None,
    target_qid: str | None = None,
) -> list[str]:
    if heldout == outer:
        raise ValueError("outer Fold 0 cannot be an inner heldout context")
    excluded = {outer}
    if heldout:
        excluded.add(str(heldout))
    qids = [str(qid) for name, values in folds.items() if name not in excluded for qid in values]
    if target_qid is not None:
        qids = [qid for qid in qids if qid != str(target_qid)]
    return sorted(qids)


def _missing_features() -> dict[str, float]:
    values = {name: MISSING_SENTINEL for name in PROTOTYPE_FEATURE_NAMES}
    values.update({"proto_seen": 0.0, "proto_similarity_missing": 1.0, "proto_support_count": 0.0, "proto_log_support_count": 0.0,
                   "proto_soft_vote_raw": 0.0, "proto_soft_vote_cardinality_norm": 0.0, "proto_soft_vote_frequency_norm": 0.0,
                   "proto_winning_document_margin": 0.0, "proto_vote_entropy": 0.0, "nearest_support_similarity": MISSING_SENTINEL,
                   "nearest_minus_second_support_similarity": MISSING_SENTINEL, "top_neighbor_document_agreement": 0.0,
                   "prototype_winner_margin": 0.0, "prototype_vote_entropy": 0.0, "prototype_seen_candidate_fraction": 0.0,
                   "source_agreement_top5": 0.0, "anchor_rank1_margin": 0.0, "anchor_rank5_margin": 0.0})
    return values


def _vote_entropy(votes: Mapping[str, float]) -> float:
    positive = np.asarray([max(0.0, float(value)) for value in votes.values()], dtype=np.float64)
    total = float(positive.sum())
    if total <= 0 or len(positive) <= 1:
        return 0.0
    p = positive[positive > 0] / total
    return float(-np.sum(p * np.log(p)) / max(math.log(len(p)), 1e-12))


def _winning_margin(votes: Mapping[str, float]) -> tuple[float, str | None]:
    ordered = sorted(((float(value), str(doc)) for doc, value in votes.items()), key=lambda item: (-item[0], item[1]))
    if not ordered:
        return 0.0, None
    return float(ordered[0][0] - (ordered[1][0] if len(ordered) > 1 else 0.0)), ordered[0][1]


def validate_soft_policy(policy: tuple[int, float, float]) -> tuple[int, float, float]:
    n, beta, gamma = int(policy[0]), float(policy[1]), float(policy[2])
    if (n, beta, gamma) not in SOFT_POLICY_GRID:
        raise ValueError(f"soft policy is outside bounded grid: {(n, beta, gamma)}")
    return n, beta, gamma


def prototype_feature_records(
    q_vector: np.ndarray,
    candidate_doc_ids: Sequence[str],
    bank: SupportBank,
    *,
    soft_policy: tuple[int, float, float] = (64, 10.0, 0.5),
    source_rows: Mapping[str, Any] | None = None,
    anchor_scores: Mapping[str, float] | None = None,
    source_depth: int = 50,
) -> dict[str, dict[str, float]]:
    n_neighbours, beta, gamma = validate_soft_policy(soft_policy)
    candidates = [str(doc_id) for doc_id in candidate_doc_ids]
    q = normalize_fp32(np.asarray(q_vector, dtype=np.float32).reshape(1, -1))[0]
    if not len(bank.support_qids):
        return {doc: _missing_features() for doc in candidates}
    similarities = np.asarray(bank.vectors @ q, dtype=np.float32)
    neighbour_order = stable_q_order(similarities, bank.support_qids, min(n_neighbours, len(bank.support_qids)))
    nearest = float(similarities[neighbour_order[0]])
    second_nearest = float(similarities[neighbour_order[1]]) if len(neighbour_order) > 1 else MISSING_SENTINEL
    raw_votes: defaultdict[str, float] = defaultdict(float)
    card_votes: defaultdict[str, float] = defaultdict(float)
    for index in neighbour_order:
        support_qid = bank.support_qids[index]
        labels = list(bank.labels_by_qid.get(support_qid, ()))
        affinity = math.exp(float(beta) * (float(similarities[index]) - 1.0))
        cardinality = max(len(labels), 1)
        for doc_id in labels:
            raw_votes[doc_id] += affinity
            card_votes[doc_id] += affinity / cardinality
    counts = {doc: len(indices) for doc, indices in bank.support_by_doc.items()}
    freq_votes = {doc: value / (max(counts.get(doc, 1), 1) ** gamma) for doc, value in card_votes.items()}
    winner_margin, winner_doc = _winning_margin(freq_votes)
    entropy = _vote_entropy(freq_votes)
    seen_count = sum(doc in bank.support_by_doc for doc in candidates)
    top_neighbour_agreement = 0.0
    if neighbour_order:
        label_counts: Counter[str] = Counter()
        for index in neighbour_order:
            for doc_id in bank.labels_by_qid.get(bank.support_qids[index], ()):
                label_counts[doc_id] += 1
        top_neighbour_agreement = max(label_counts.values(), default=0) / len(neighbour_order)
    anchor_scores = {str(k): float(v) for k, v in (anchor_scores or {}).items()}
    ordered_anchor = sorted(anchor_scores.values(), reverse=True)
    # Source ranks are query-level invariants.  Building them per candidate made
    # the Colab nested policy screen needlessly quadratic in candidate count.
    source_ranks = {
        source: {str(item["doc_id"]): int(item.get("rank", i + 1)) for i, item in enumerate(_ranking_items(source_rows, source)[:source_depth])}
        for source in SOURCES
    } if source_rows else {}
    record: dict[str, dict[str, float]] = {}
    for doc_id in candidates:
        values = _missing_features()
        indices = bank.support_by_doc.get(doc_id)
        if indices is not None and len(indices):
            local = np.sort(similarities[indices])[::-1]
            max_similarity = float(local[0])
            second = float(local[1]) if len(local) > 1 else MISSING_SENTINEL
            top2 = float(np.mean(local[: min(2, len(local))]))
            top3 = float(np.mean(local[: min(3, len(local))]))
            centroid_cos = float(bank.centroids[doc_id] @ q)
            centroid_values = np.asarray(bank.vectors[indices] @ bank.centroids[doc_id], dtype=np.float32)
            values.update({
                "proto_seen": 1.0,
                "proto_similarity_missing": 0.0,
                "proto_support_count": float(len(indices)),
                "proto_log_support_count": float(math.log1p(len(indices))),
                "proto_max_similarity": max_similarity,
                "proto_second_similarity": second,
                "proto_top2_mean": top2,
                "proto_top3_mean": top3,
                "proto_max_minus_second": max_similarity - second if second != MISSING_SENTINEL else 0.0,
                "proto_max_minus_top3_mean": max_similarity - top3,
                "proto_centroid_cosine": centroid_cos,
                "proto_max_minus_centroid": max_similarity - centroid_cos,
                "proto_support_dispersion": float(np.var(centroid_values)) if len(centroid_values) > 1 else 0.0,
            })
        values.update({
            "proto_soft_vote_raw": float(raw_votes.get(doc_id, 0.0)),
            "proto_soft_vote_cardinality_norm": float(card_votes.get(doc_id, 0.0)),
            "proto_soft_vote_frequency_norm": float(freq_votes.get(doc_id, 0.0)),
            "proto_winning_document_margin": winner_margin,
            "proto_vote_entropy": entropy,
            "nearest_support_similarity": nearest,
            "nearest_minus_second_support_similarity": nearest - second_nearest if second_nearest != MISSING_SENTINEL else 0.0,
            "top_neighbor_document_agreement": float(top_neighbour_agreement),
            "prototype_winner_margin": winner_margin,
            "prototype_vote_entropy": entropy,
            "prototype_seen_candidate_fraction": seen_count / len(candidates) if candidates else 0.0,
        })
        if source_ranks:
            values["source_agreement_top5"] = float(sum(ranks.get(doc_id, source_depth + 1) <= 5 for ranks in source_ranks.values()))
        if ordered_anchor:
            values["anchor_rank1_margin"] = float(ordered_anchor[0] - anchor_scores.get(doc_id, 0.0))
            values["anchor_rank5_margin"] = float((ordered_anchor[min(4, len(ordered_anchor) - 1)] - anchor_scores.get(doc_id, 0.0)))
        record[doc_id] = values
    return record


def prototype_views(feature_records: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    docs = sorted(feature_records)
    p1 = {doc: float(values["proto_max_similarity"]) for doc, values in feature_records.items()}
    p2 = {doc: float(values["proto_centroid_cosine"]) for doc, values in feature_records.items()}
    p3 = {doc: float(values["proto_soft_vote_frequency_norm"]) for doc, values in feature_records.items()}
    p4_z = {
        doc: float(np.mean([query_zscore(p1)[doc], query_zscore(p2)[doc], query_zscore(p3)[doc]]))
        for doc in docs
    }
    return {
        "P1_max_exemplar": p1,
        "P2_normalized_centroid": p2,
        "P3_soft_cache_vote": p3,
        "P4_fixed_normalized_combination": p4_z,
    }


def rank_scores(scores: Mapping[str, float], *, limit: int | None = None) -> list[str]:
    return [str(doc) for doc in sorted(scores, key=lambda doc: (-float(scores[doc]), str(doc)))[:limit]]


def confidence_gate_weight(
    *,
    nearest_similarity: float,
    prototype_winner_margin: float,
    prototype_vote_entropy: float,
    seen_candidate_fraction: float,
    threshold: float,
    anchor_confidence: float | None = None,
    anchor_high_confidence_threshold: float | None = None,
) -> float:
    if seen_candidate_fraction <= 0.0 or nearest_similarity < threshold:
        return 0.0
    if anchor_confidence is not None and anchor_high_confidence_threshold is not None and anchor_confidence >= anchor_high_confidence_threshold:
        return 0.0
    margin_factor = float(np.clip(prototype_winner_margin / 0.05, 0.0, 1.0))
    entropy_factor = float(np.clip(1.0 - prototype_vote_entropy, 0.0, 1.0))
    return float(np.clip(seen_candidate_fraction * max(margin_factor, 0.25) * max(entropy_factor, 0.25), 0.0, 1.0))


def protected_residual_scores(
    anchor_scores: Mapping[str, float],
    prototype_scores: Mapping[str, float],
    *,
    alpha: float,
    gate_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    docs = sorted(set(anchor_scores) | set(prototype_scores))
    anchor_z = query_zscore({doc: float(anchor_scores.get(doc, 0.0)) for doc in docs})
    proto_z = query_zscore({doc: float(prototype_scores.get(doc, MISSING_SENTINEL)) for doc in docs})
    weights = gate_weights or {doc: 1.0 for doc in docs}
    return {doc: float(anchor_z[doc] + float(alpha) * float(weights.get(doc, 0.0)) * proto_z[doc]) for doc in docs}


def _prediction_metrics(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> dict[str, Any]:
    evaluable = [str(qid) for qid in qids if answers.get(str(qid))]
    metrics: dict[str, Any] = {"evaluable_queries": len(evaluable)}
    if not evaluable:
        return metrics
    for k in CURVE_KS:
        metrics[f"recall@{k}"] = float(np.mean([len(set(predictions[qid][:k]) & answers[qid]) / len(answers[qid]) for qid in evaluable]))
    metrics["precision@5"] = float(np.mean([len(set(predictions[qid][:5]) & answers[qid]) / 5.0 for qid in evaluable]))
    reciprocal = []
    for qid in evaluable:
        first = next((rank for rank, doc in enumerate(predictions[qid], 1) if doc in answers[qid]), None)
        reciprocal.append(1.0 / first if first is not None and first <= 5 else 0.0)
    metrics["mrr@5"] = float(np.mean(reciprocal))
    multi = [qid for qid in evaluable if len(answers[qid]) >= 2]
    single = [qid for qid in evaluable if len(answers[qid]) == 1]
    metrics["multi_gold_recall@5"] = float(np.mean([len(set(predictions[qid][:5]) & answers[qid]) / len(answers[qid]) for qid in multi])) if multi else 0.0
    metrics["single_gold_recall@5"] = float(np.mean([len(set(predictions[qid][:5]) & answers[qid]) / len(answers[qid]) for qid in single])) if single else 0.0
    metrics["cardinality"] = {"single_gold": {"count": len(single), "recall@5": metrics["single_gold_recall@5"]}, "multi_gold": {"count": len(multi), "recall@5": metrics["multi_gold_recall@5"]}}
    return metrics


def prediction_metrics(predictions: Mapping[str, Sequence[str]], answers: Mapping[str, set[str]], qids: Sequence[str]) -> dict[str, Any]:
    checked: dict[str, list[str]] = {}
    expected = {str(qid) for qid in qids}
    if set(predictions) != expected:
        raise ValueError("prediction qid coverage mismatch")
    for qid in sorted(expected):
        values = [str(doc) for doc in predictions[qid]]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate document in prediction: {qid}")
        checked[qid] = values
    return _prediction_metrics(checked, answers, sorted(expected))


def deterministic_bootstrap(deltas: Sequence[float], *, samples: int = 10_000, seed: int = 110) -> dict[str, float]:
    values = np.asarray(list(deltas), dtype=np.float64)
    if not len(values):
        return {"samples": int(samples), "mean": 0.0, "lower": 0.0, "upper": 0.0, "seed": seed}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[draws].mean(axis=1)
    return {"samples": int(samples), "mean": float(values.mean()), "lower": float(np.quantile(means, 0.025)), "upper": float(np.quantile(means, 0.975)), "seed": int(seed)}


def paired_delta_bootstrap(
    anchor: Mapping[str, Sequence[str]],
    winner: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
    *,
    k: int = 5,
    samples: int = 10_000,
    seed: int = 110,
) -> tuple[list[float], dict[str, float]]:
    deltas = [len(set(winner[qid][:k]) & answers[qid]) / len(answers[qid]) - len(set(anchor[qid][:k]) & answers[qid]) / len(answers[qid]) for qid in qids if answers.get(qid)]
    return deltas, deterministic_bootstrap(deltas, samples=samples, seed=seed)


def breakdown_seen_frequency(
    predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
    bank: SupportBank,
) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for qid in qids:
        gold = answers.get(str(qid), set())
        if not gold:
            continue
        for doc in sorted(gold):
            count = len(bank.support_by_doc.get(doc, ()))
            bucket = "unseen" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4+"
            buckets[bucket].append(float(doc in set(predictions[str(qid)][:5])))
    return {key: {"gold_occurrences": len(values), "recall@5_occurrence": float(np.mean(values)) if values else 0.0} for key, values in sorted(buckets.items())}


def prediction_comparison(
    anchor: Mapping[str, Sequence[str]],
    winner: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Return per-query wins/losses/ties and Top-k gold movement."""
    wins = losses = ties = 0
    gold_into = gold_out = 0
    for raw_qid in qids:
        qid = str(raw_qid)
        gold = answers.get(qid, set())
        if not gold:
            continue
        anchor_hits = len(set(anchor[qid][:k]) & gold)
        winner_hits = len(set(winner[qid][:k]) & gold)
        if winner_hits > anchor_hits:
            wins += 1
        elif winner_hits < anchor_hits:
            losses += 1
        else:
            ties += 1
        anchor_top = set(anchor[qid][:k])
        winner_top = set(winner[qid][:k])
        gold_into += len((winner_top - anchor_top) & gold)
        gold_out += len((anchor_top - winner_top) & gold)
    return {"k": int(k), "wins": wins, "losses": losses, "ties": ties, "gold_into_top_k": gold_into, "gold_out_of_top_k": gold_out}


def choice_oracle_predictions(
    arm_predictions: Mapping[str, Mapping[str, Sequence[str]]],
    answers: Mapping[str, set[str]],
    qids: Sequence[str],
    *,
    k: int = 5,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Compute the explicitly label-dependent per-query arm oracle."""
    tie_order = ("A", "C", "B", "D")
    chosen: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    for raw_qid in qids:
        qid = str(raw_qid)
        gold = answers.get(qid, set())
        if not gold:
            continue
        available = [name for name in tie_order if name in arm_predictions and qid in arm_predictions[name]]
        if not available:
            raise ValueError(f"choice oracle has no arm for {qid}")
        arm = max(available, key=lambda name: (len(set(arm_predictions[name][qid][:k]) & gold), -tie_order.index(name)))
        chosen[qid] = [str(doc) for doc in arm_predictions[arm][qid]]
        counts[arm] += 1
    return chosen, {"label_dependent": True, "selection_scope": "per-query oracle", "k": int(k), "chosen_arm_counts": dict(sorted(counts.items()))}


def exact_normalized_duplicate_groups(embeddings: np.ndarray, qids: Sequence[str]) -> list[list[str]]:
    """Group byte-identical normalized FP32 query vectors deterministically."""
    vectors = normalize_fp32(np.asarray(embeddings, dtype=np.float32))
    if len(vectors) != len(qids):
        raise ValueError("embedding/qid count mismatch")
    groups: defaultdict[bytes, list[str]] = defaultdict(list)
    for qid, vector in zip(map(str, qids), vectors):
        groups[vector.tobytes()].append(qid)
    return [sorted(values) for values in sorted(groups.values(), key=lambda values: (values[0], len(values))) if len(values) > 1]


def prototype_coverage_report(
    feature_records_by_qid: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Summarize missing/seen prototype features without exposing support IDs."""
    candidates = [features for rows in feature_records_by_qid.values() for features in rows.values()]
    seen = sum(float(row.get("proto_seen", 0.0)) > 0.5 for row in candidates)
    return {
        "queries": len(feature_records_by_qid),
        "candidate_documents": len(candidates),
        "seen_candidate_documents": int(seen),
        "missing_candidate_documents": int(len(candidates) - seen),
        "seen_candidate_fraction": float(seen / len(candidates)) if candidates else 0.0,
    }


def nearest_similarity_breakdown(
    embeddings: np.ndarray,
    qids: Sequence[str],
    qid_to_row: Mapping[str, int],
    support_by_qid: Mapping[str, Sequence[str]],
    *,
    bands: Sequence[tuple[str, float, float]] = (("lt_0.30", -math.inf, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, math.inf)),
) -> dict[str, Any]:
    """Bucket target queries by nearest support cosine in their strict context."""
    vectors = normalize_fp32(np.asarray(embeddings, dtype=np.float32))
    result: dict[str, list[str]] = {name: [] for name, _low, _high in bands}
    missing: list[str] = []
    for raw_qid in qids:
        qid = str(raw_qid)
        support = [str(value) for value in support_by_qid.get(qid, ())]
        if not support:
            missing.append(qid)
            continue
        support_vectors = vectors[[qid_to_row[value] for value in support]]
        nearest = float(np.max(support_vectors @ vectors[qid_to_row[qid]]))
        for name, low, high in bands:
            if low <= nearest < high or (math.isinf(high) and nearest >= low):
                result[name].append(qid)
                break
    return {"bands": {name: {"queries": len(values), "qids": sorted(values)} for name, values in result.items()}, "no_support": {"queries": len(missing), "qids": sorted(missing)}}


def strict_inner_gate(
    anchor_metrics: Mapping[str, Any],
    winner_metrics: Mapping[str, Any],
    *,
    per_fold_delta: Mapping[str, float],
    bootstrap: Mapping[str, float],
) -> dict[str, Any]:
    delta = float(winner_metrics.get("recall@5", 0.0)) - float(anchor_metrics.get("recall@5", 0.0))
    multi_delta = float(winner_metrics.get("multi_gold_recall@5", 0.0)) - float(anchor_metrics.get("multi_gold_recall@5", 0.0))
    mrr_delta = float(winner_metrics.get("mrr@5", 0.0)) - float(anchor_metrics.get("mrr@5", 0.0))
    recall1_delta = float(winner_metrics.get("recall@1", 0.0)) - float(anchor_metrics.get("recall@1", 0.0))
    checks = {
        "delta_recall@5_ge_0.005": delta >= 0.005,
        "paired_bootstrap_lower_gt_0": float(bootstrap.get("lower", -math.inf)) > 0.0,
        "at_least_3_of_4_folds_improve": sum(float(value) > 0.0 for value in per_fold_delta.values()) >= 3,
        "worst_fold_delta_ge_minus_0.002": min(per_fold_delta.values(), default=-math.inf) >= -0.002,
        "multi_gold_non_decrease": multi_delta >= 0.0,
        "precision_non_decrease": float(winner_metrics.get("precision@5", -math.inf)) >= float(anchor_metrics.get("precision@5", math.inf)),
        "mrr_safety": mrr_delta >= -0.001,
        "recall1_safety": recall1_delta >= -0.002,
    }
    passed = all(checks.values())
    return {
        "status": "PASS_EXP110P_STRICT_INNER_GATE" if passed else "REJECTED_EXP110P_STRICT_INNER_GATE",
        "pass": passed,
        "checks": checks,
        "delta_recall@5": delta,
        "multi_gold_delta": multi_delta,
        "mrr_delta": mrr_delta,
        "recall1_delta": recall1_delta,
        "per_fold_delta": dict(sorted(per_fold_delta.items())),
        "bootstrap": dict(bootstrap),
        "fold0_read": False,
        "claim_boundary": "strict cross-fit F1-F4 inner only; not Fold-0/public Recall",
    }


def select_nested_arm(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Choose an arm using only the supplied training-context metrics."""
    order = {"A": 0, "C": 1, "B": 2, "D": 3}

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, float, float, int]:
        name, metrics = item
        return (
            float(metrics.get("recall@5", -math.inf)),
            float(metrics.get("multi_gold_recall@5", -math.inf)),
            float(metrics.get("precision@5", -math.inf)),
            float(metrics.get("mrr@5", -math.inf)),
            float(metrics.get("recall@1", -math.inf)),
            -order.get(name, 99),
        )

    if not candidates:
        raise ValueError("no arms supplied")
    return max(candidates.items(), key=key)[0]


def select_nested_residual_config(candidates: Mapping[tuple[float, float], Mapping[str, Any]]) -> tuple[float, float]:
    """Select C's alpha/threshold quantile from training-context metrics only."""
    if not candidates:
        raise ValueError("no residual configurations supplied")

    def key(item: tuple[tuple[float, float], Mapping[str, Any]]) -> tuple[float, float, float, float, float, float, float]:
        (alpha, quantile), metrics = item
        return (
            float(metrics.get("recall@5", -math.inf)),
            float(metrics.get("multi_gold_recall@5", -math.inf)),
            float(metrics.get("precision@5", -math.inf)),
            float(metrics.get("mrr@5", -math.inf)),
            float(metrics.get("recall@1", -math.inf)),
            -float(alpha),
            -float(quantile),
        )

    return tuple(max(candidates.items(), key=key)[0])  # type: ignore[return-value]


def nested_context_provenance(
    folds: Mapping[str, Sequence[str]],
    *,
    outer: str,
    heldout: str,
    target_qid: str | None = None,
) -> dict[str, Any]:
    support = support_qids_for_context(folds, outer=outer, heldout=heldout, target_qid=target_qid)
    if any(str(qid) in set(folds.get(outer, ())) for qid in support):
        raise ValueError("outer fold leaked into support context")
    if any(str(qid) in set(folds.get(heldout, ())) for qid in support):
        raise ValueError("heldout fold leaked into support context")
    return {"target_qid": str(target_qid) if target_qid is not None else None, "support_fold_names": [name for name in sorted(folds) if name not in {outer, heldout}], "support_query_count": len(support), "self_excluded": target_qid is not None, "heldout_excluded": True, "support_fingerprint": content_hash({"support_qids": support})}


def make_lambdamart_training_rows(
    qids: Sequence[str],
    source_rows_by_qid: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, set[str]],
    *,
    depth: int = 50,
    parent_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    query_token_lengths: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[str, str]]]:
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    identities: list[tuple[str, str]] = []
    parent_metadata = parent_metadata or {}
    query_token_lengths = query_token_lengths or {}
    for qid in (str(x) for x in qids):
        row = source_rows_by_qid[qid]
        docs = candidate_union(row, depth=depth)
        if not docs:
            continue
        groups.append(len(docs))
        for doc_id in docs:
            feature = build_source_feature_record(qid, doc_id, row, depth=depth, query_token_length=query_token_lengths.get(qid, 0), parent_metadata=parent_metadata.get(doc_id))
            matrices.append(source_feature_vector(feature))
            labels.append(float(doc_id in answers.get(qid, set())))
            identities.append((qid, doc_id))
    if not matrices:
        raise ValueError("empty LambdaMART rows")
    return np.vstack(matrices).astype(np.float32), np.asarray(labels, dtype=np.float32), groups, identities


def fit_lambdamart(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[int],
    config: Mapping[str, Any],
    *,
    feature_names: Sequence[str] | None = None,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised only without optional Colab dependency
        raise RuntimeError("LightGBM is required for anchor reproduction") from exc
    names = tuple(feature_names or BASE_FEATURE_NAMES)
    if names == BASE_FEATURE_NAMES:
        validate_feature_contract(names)
    elif names != AUGMENTED_FEATURE_NAMES or len(names) != len(set(names)):
        raise ValueError("LambdaMART feature order/schema is not an approved EXP-110P arm contract")
    # Match EXP-109B's sklearn wrapper and parameter names exactly.  Using
    # lgb.train or `min_data_in_leaf` directly can produce a different model
    # from the locked pilot under otherwise identical inputs.
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[5],
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_data_in_leaf"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["num_boost_round"]),
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        deterministic=True,
        random_state=109,
        verbosity=-1,
    )
    model.fit(
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        group=list(map(int, groups)),
        feature_name=list(names),
    )
    return model


def bounded_lambdamart_configs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return a small deterministic neighborhood around a locked anchor config."""
    base = {str(key): value for key, value in config.items()}
    candidates: list[dict[str, Any]] = [dict(base)]
    for key, factor in (("num_boost_round", 0.75), ("num_boost_round", 1.25), ("num_leaves", 0.75), ("num_leaves", 1.25)):
        if key not in base:
            continue
        value = int(base[key])
        changed = dict(base)
        changed[key] = max(1, int(round(value * factor)))
        candidates.append(changed)
    unique: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for candidate in candidates:
        fingerprint = canonical_json(candidate)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(candidate)
    return tuple(unique)


def make_augmented_training_rows(
    qids: Sequence[str],
    source_rows_by_qid: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, set[str]],
    embeddings: np.ndarray,
    qid_to_row: Mapping[str, int],
    *,
    support_qids_for_qid: Mapping[str, Sequence[str]],
    policy: tuple[int, float, float],
    outer_qids: Iterable[str] = (),
    depth: int = 50,
    expansion: bool = False,
    parent_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    query_token_lengths: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[str, str]]]:
    """Build B/D rows with leave-one-query-out support provenance.

    The caller supplies a context for every row.  A target query is never
    silently added to its own support context; this makes the invariant
    inspectable instead of relying on a global prototype bank.
    """
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    identities: list[tuple[str, str]] = []
    forbidden = set(map(str, outer_qids))
    parent_metadata = parent_metadata or {}
    query_token_lengths = query_token_lengths or {}
    for qid in sorted(str(value) for value in qids):
        support = [str(value) for value in support_qids_for_qid.get(qid, ()) if str(value) != qid]
        if forbidden & set(support):
            raise ValueError(f"outer qid leaked into augmented support: {qid}")
        bank = build_support_bank(support, embeddings, qid_to_row, answers, forbidden_qids=forbidden)
        base = candidate_union(source_rows_by_qid[qid], depth=depth)
        all_docs = sorted(set(base) | (set(bank.support_by_doc) if expansion else set()))
        if not all_docs:
            continue
        proto = prototype_feature_records(embeddings[qid_to_row[qid]], all_docs, bank, soft_policy=policy, source_rows=source_rows_by_qid[qid], source_depth=depth)
        groups.append(len(all_docs))
        for doc_id in all_docs:
            source = build_source_feature_record(
                qid, doc_id, source_rows_by_qid[qid], depth=depth,
                query_token_length=query_token_lengths.get(qid, 0),
                parent_metadata=parent_metadata.get(doc_id),
            )
            combined = combine_source_and_prototype_features(source, proto[doc_id])
            matrices.append(augmented_feature_vector(combined))
            labels.append(float(doc_id in answers.get(qid, set())))
            identities.append((qid, doc_id))
    if not matrices:
        raise ValueError("empty augmented rows")
    return np.vstack(matrices).astype(np.float32), np.asarray(labels, dtype=np.float32), groups, identities


def rank_model_predictions(
    model: Any,
    qids: Sequence[str],
    source_rows_by_qid: Mapping[str, Mapping[str, Any]],
    embeddings: np.ndarray,
    qid_to_row: Mapping[str, int],
    *,
    support_qids: Mapping[str, Sequence[str]],
    answers: Mapping[str, set[str]],
    policy: tuple[int, float, float],
    outer_qids: Iterable[str] = (),
    depth: int = 50,
    expansion: bool = False,
    parent_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    query_token_lengths: Mapping[str, int] | None = None,
) -> dict[str, list[str]]:
    forbidden = set(map(str, outer_qids))
    predictions: dict[str, list[str]] = {}
    parent_metadata = parent_metadata or {}
    query_token_lengths = query_token_lengths or {}
    for qid in sorted(str(value) for value in qids):
        support = [str(value) for value in support_qids.get(qid, ()) if str(value) != qid]
        bank = build_support_bank(support, embeddings, qid_to_row, answers, forbidden_qids=forbidden)
        base = candidate_union(source_rows_by_qid[qid], depth=depth)
        docs = sorted(set(base) | (set(bank.support_by_doc) if expansion else set()))
        proto = prototype_feature_records(embeddings[qid_to_row[qid]], docs, bank, soft_policy=policy, source_rows=source_rows_by_qid[qid], source_depth=depth)
        X = np.vstack([
            augmented_feature_vector(combine_source_and_prototype_features(
                build_source_feature_record(
                    qid, doc, source_rows_by_qid[qid], depth=depth,
                    query_token_length=query_token_lengths.get(qid, 0),
                    parent_metadata=parent_metadata.get(doc),
                ),
                proto[doc],
            ))
            for doc in docs
        ])
        scores = np.asarray(model.predict(X), dtype=np.float32)
        predictions[qid] = [docs[index] for index in stable_order(scores, docs)]
    return predictions


def reproduce_anchor_predictions(
    source_rows_by_qid: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, set[str]],
    folds: Mapping[str, Sequence[str]],
    configs_by_heldout_fold: Mapping[str, Mapping[str, Any]],
    *,
    outer: str = FOLD0,
    depth: int = 50,
    parent_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    query_token_lengths: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict[str, Any]]:
    if outer in configs_by_heldout_fold or outer in folds and outer in configs_by_heldout_fold:
        raise ValueError("Fold 0 config/prediction is forbidden in inner anchor reproduction")
    predictions: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    fold_meta: dict[str, Any] = {}
    parent_metadata = parent_metadata or {}
    query_token_lengths = query_token_lengths or {}
    inner = [name for name in sorted(folds) if name != outer]
    for heldout in inner:
        if heldout not in configs_by_heldout_fold:
            raise ValueError(f"missing locked config for {heldout}")
        train_qids = [str(qid) for name in inner if name != heldout for qid in folds[name]]
        valid_qids = [str(qid) for qid in folds[heldout]]
        X, y, groups, _ = make_lambdamart_training_rows(
            train_qids, source_rows_by_qid, answers, depth=depth,
            parent_metadata=parent_metadata, query_token_lengths=query_token_lengths,
        )
        model = fit_lambdamart(X, y, groups, configs_by_heldout_fold[heldout], feature_names=BASE_FEATURE_NAMES)
        valid_rows: list[np.ndarray] = []
        valid_ids: list[tuple[str, str]] = []
        for qid in valid_qids:
            docs = candidate_union(source_rows_by_qid[qid], depth=depth)
            values = [source_feature_vector(build_source_feature_record(
                qid, doc, source_rows_by_qid[qid], depth=depth,
                query_token_length=query_token_lengths.get(qid, 0), parent_metadata=parent_metadata.get(doc),
            )) for doc in docs]
            valid_rows.extend(values)
            valid_ids.extend((qid, doc) for doc in docs)
        scores = np.asarray(model.predict(np.vstack(valid_rows)), dtype=np.float32)
        by_qid: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (qid, doc), score in zip(valid_ids, scores):
            by_qid[qid].append((doc, float(score)))
        for qid in valid_qids:
            ordered = sorted(by_qid[qid], key=lambda item: (-item[1], item[0]))
            predictions[qid] = [doc for doc, _score in ordered]
            raw_scores[qid] = {doc: score for doc, score in ordered}
        fold_meta[heldout] = {"train_qids": len(train_qids), "validation_qids": len(valid_qids), "config": dict(configs_by_heldout_fold[heldout]), "feature_contract": list(BASE_FEATURE_NAMES)}
    if any(qid in set(map(str, folds.get(outer, ()))) for qid in predictions):
        raise ValueError("Fold 0 appeared in reproduced inner anchor")
    return predictions, raw_scores, {"folds": fold_meta, "fold0_read": False, "depth": depth}


def load_anchor_prediction_rows(path: Path, *, outer: str = FOLD0) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    predictions: dict[str, list[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        if qid in predictions:
            raise ValueError(f"duplicate anchor qid: {qid}")
        docs = row.get("prediction", row.get("documents", row.get("ranking")))
        if not isinstance(docs, list):
            raise ValueError(f"anchor row lacks prediction list: {qid}")
        values = [str(item.get("doc_id") if isinstance(item, Mapping) else item) for item in docs]
        if len(values) != len(set(values)):
            raise ValueError(f"anchor has duplicate doc IDs: {qid}")
        predictions[qid] = values
        raw = row.get("raw_scores", {})
        raw_scores[qid] = {str(doc): float(score) for doc, score in raw.items()} if isinstance(raw, Mapping) else {}
    return predictions, raw_scores


def validate_inner_only_qids(qids: Iterable[str], folds: Mapping[str, Sequence[str]], *, outer: str = FOLD0) -> None:
    outer_qids = {str(qid) for qid in folds.get(outer, ())}
    overlap = sorted({str(qid) for qid in qids} & outer_qids)
    if overlap:
        raise ValueError(f"Fold 0 qids in inner artifact: {overlap[:5]}")


def write_similarity_shards(
    embeddings: np.ndarray,
    qids: Sequence[str],
    output_dir: Path,
    *,
    block_size: int = 256,
    forbidden_qids: Iterable[str] = (),
    resume: bool = True,
) -> dict[str, Any]:
    qids = tuple(str(qid) for qid in qids)
    forbidden = set(map(str, forbidden_qids))
    if forbidden & set(qids):
        raise ValueError("forbidden qid in similarity cache")
    vectors = normalize_fp32(np.asarray(embeddings))
    if vectors.shape[0] != len(qids):
        raise ValueError("embedding/qid count mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = content_hash({"qids": qids, "shape": list(vectors.shape), "dtype": "float32", "diagonal": "-inf"})
    receipts: list[dict[str, Any]] = []
    for start in range(0, len(qids), int(block_size)):
        stop = min(start + int(block_size), len(qids))
        path = output_dir / f"similarity-{start:06d}-{stop:06d}.npy"
        receipt_path = path.with_suffix(".json")
        valid = False
        if resume and path.exists() and receipt_path.exists():
            saved = read_json(receipt_path)
            valid = (
                saved.get("fingerprint") == fingerprint
                and saved.get("sha256") == sha256_file(path)
                and saved.get("shape") == [stop - start, len(qids)]
                and saved.get("dtype") == "float32"
            )
        if not valid:
            block = np.asarray(vectors[start:stop] @ vectors.T, dtype=np.float32)
            diagonal = np.arange(start, stop)
            block[np.arange(stop - start), diagonal] = -np.inf
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, block, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            atomic_json(receipt_path, {"schema_version": SCHEMA, "fingerprint": fingerprint, "start": start, "stop": stop, "shape": list(block.shape), "dtype": "float32", "sha256": sha256_file(path)})
        receipts.append({"name": path.name, "start": start, "stop": stop, "sha256": sha256_file(path)})
    manifest = {"schema_version": SCHEMA, "stage": "similarity-cache", "qids": list(qids), "shape": [len(qids), len(qids)], "block_size": int(block_size), "fingerprint": fingerprint, "shards": receipts, "fold0_read": False}
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def load_verified_similarity_shard(path: Path, receipt_path: Path, *, expected_fingerprint: str) -> np.ndarray:
    receipt = read_json(receipt_path)
    if receipt.get("fingerprint") != expected_fingerprint or receipt.get("sha256") != sha256_file(path):
        raise ValueError(f"corrupt or stale similarity shard: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32:
        raise ValueError("similarity shard must be FP32")
    expected_shape = receipt.get("shape")
    if expected_shape is not None and list(values.shape) != list(expected_shape):
        raise ValueError("similarity shard shape mismatch")
    return values


def gpu_similarity_block(left: np.ndarray, right: np.ndarray, *, device: str = "cuda") -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for optional GPU path") from exc
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    with torch.inference_mode():
        return (torch.as_tensor(normalize_fp32(left), device=device) @ torch.as_tensor(normalize_fp32(right), device=device).T).float().cpu().numpy()


def similarity_parity(cpu: np.ndarray, gpu: np.ndarray, qids: Sequence[str], *, top_k: int = 64) -> dict[str, Any]:
    cpu = np.asarray(cpu, dtype=np.float32)
    gpu = np.asarray(gpu, dtype=np.float32)
    if cpu.shape != gpu.shape:
        raise ValueError("CPU/GPU similarity shape mismatch")
    max_abs_error = float(np.max(np.abs(cpu - gpu))) if cpu.size else 0.0
    identity = all(stable_q_order(cpu[row], qids, top_k) == stable_q_order(gpu[row], qids, top_k) for row in range(cpu.shape[0]))
    return {"max_absolute_error": max_abs_error, "top64_identity": identity, "pass": max_abs_error <= 1e-5 and identity}


def verify_input_manifest(bundle_root: Path, manifest: Mapping[str, Any], required_files: Sequence[str]) -> dict[str, Any]:
    bundle_root = Path(bundle_root)
    files = manifest.get("files", {})
    if isinstance(files, list):
        files = {str(item["path"]): item for item in files}
    missing = [rel for rel in required_files if rel not in files]
    if missing:
        raise ValueError(f"manifest missing required files: {missing}")
    checked: list[dict[str, Any]] = []
    for rel in required_files:
        entry = files[rel]
        path = bundle_root / rel
        if not path.exists() or not path.is_file():
            raise ValueError(f"missing bundle file: {path}")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_bytes != int(entry["bytes"]) or observed_sha != str(entry["sha256"]):
            raise ValueError(f"bundle hash mismatch: {rel}")
        checked.append({"path": rel, "bytes": observed_bytes, "sha256": observed_sha})
    if manifest.get("fold0_predictions_included", True) is not False:
        raise ValueError("input manifest does not prove fold0_predictions_included=false")
    return {"status": "PASS", "files": checked, "manifest_fingerprint": content_hash(manifest)}


def resource_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        import psutil
        memory = psutil.virtual_memory()
        result["ram_total_bytes"] = int(memory.total)
        result["ram_available_bytes"] = int(memory.available)
        result["ram_used_percent"] = float(memory.percent)
    except ImportError:
        result["psutil"] = "unavailable"
    try:
        import torch
        result["gpu_available"] = bool(torch.cuda.is_available())
        if result["gpu_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:  # pragma: no cover
        result["gpu_available"] = False
        result["gpu_error"] = type(exc).__name__
    return result


def projected_resource_gate(snapshot: Mapping[str, Any], *, projected_peak_bytes: int, projected_cpu_seconds: float, drive_free_bytes: int) -> dict[str, Any]:
    total = int(snapshot.get("ram_total_bytes", 0))
    available = int(snapshot.get("ram_available_bytes", 0))
    checks = {
        "available_ram_ge_4_gib": available >= 4 * 1024**3,
        "drive_free_ge_5_gib": int(drive_free_bytes) >= 5 * 1024**3,
        "projected_peak_le_70_percent": total > 0 and int(projected_peak_bytes) <= 0.70 * total,
        "projected_cpu_le_3_hours": float(projected_cpu_seconds) <= 3 * 3600,
    }
    return {"status": "PASS_COLAB_RESOURCE_GATE" if all(checks.values()) else "REJECTED_COLAB_RESOURCE_GATE", "checks": checks, "projected_peak_bytes": int(projected_peak_bytes), "projected_cpu_seconds": float(projected_cpu_seconds), "drive_free_bytes": int(drive_free_bytes)}


__all__ = [name for name in globals() if not name.startswith("_")]
