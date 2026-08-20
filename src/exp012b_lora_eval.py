"""Held-out adapter scoring, OOF evaluation, and guarded submission export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

from exp012b_core import atomic_json, canonical_json, load_answers, read_jsonl, sha256_file
from exp012b_retrieval import evaluate_rankings
from exp012b_reranker import (
    _load_reranker,
    aggregate_ce_documents,
    fuse_stage1_and_ce,
    score_evidence_records,
    unload_cuda,
)
from exp012b_tuning import load_folds, nested_tune_zero_shot


def _compact_stage1_path(source_path: Path, logger: Any | None = None) -> Path:
    """Materialize doc/rank-only candidates once; reuse across every evaluation."""
    compact_path = source_path.parent / "stage1_compact.jsonl"
    marker_path = source_path.parent / "stage1_compact.manifest.json"
    source_manifest_path = source_path.parent / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists() else {}
    )
    source_hash = source_manifest.get("artifact_sha256", {}).get(source_path.name)
    if source_hash is None:
        source_hash = sha256_file(source_path)
    if compact_path.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("source_sha256") == source_hash:
            return compact_path
    if logger is not None:
        logger.log(f"phase=build-compact-stage1 source={source_path.name}")
    temporary = compact_path.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n", buffering=1024 * 1024) as output:
        for count, row in enumerate(read_jsonl(source_path), start=1):
            output.write(
                canonical_json(
                    {
                        "qid": str(row["qid"]),
                        "candidates": [
                            {"doc_id": str(candidate["doc_id"]), "rank": int(candidate["rank"])}
                            for candidate in row["candidates"]
                        ],
                    }
                ) + "\n"
            )
            if logger is not None and count % 512 == 0:
                logger.status(
                    stage="evaluation", state="RUNNING", phase="build-compact-stage1",
                    completed=count, total=source_manifest.get("queries"),
                )
    temporary.replace(compact_path)
    atomic_json(
        marker_path,
        {
            "schema_version": "legalir.exp012b_v3.stage1_compact.v1",
            "source_sha256": source_hash,
            "queries": count,
            "sha256": sha256_file(compact_path),
        },
    )
    return compact_path


def _load_stage1(
    source_path: Path, qids: set[str] | None = None, logger: Any | None = None
) -> dict[str, list[dict[str, Any]]]:
    compact_path = _compact_stage1_path(source_path, logger)
    return {
        str(row["qid"]): row["candidates"]
        for row in read_jsonl(compact_path)
        if qids is None or str(row["qid"]) in qids
    }


def _gate_passed(report: Mapping[str, Any]) -> bool:
    gate = report.get("promotion_gate")
    if isinstance(gate, Mapping):
        return bool(gate.get("overall_pass"))
    return bool(gate)


def load_adapter(adapter_dir: Path, *, device: str, merge: bool = False):
    from peft import PeftModel

    base_name = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ]
    model, tokenizer = _load_reranker(base_name, device)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False).to(device)
    if merge:
        # PEFT hooks add measurable overhead for every pair. Safe merge folds
        # LoRA deltas into the in-memory base weights; no merged checkpoint is
        # written and the adapter on disk remains untouched.
        model = model.merge_and_unload(safe_merge=True)
    model.eval()
    return model, tokenizer


def score_lora_fold(
    *,
    fold: int,
    evidence_path: Path,
    folds_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    device: str,
    resume: bool = False,
    merge_adapter: bool = False,
    batch_size: int = 24,
    token_cache_dir: Path | None = None,
) -> dict[str, Any]:
    qids = set(load_folds(folds_path)[f"fold_{fold}"])
    return score_evidence_records(
        evidence_path,
        output_dir,
        model_name=str(adapter_dir),
        device=device,
        resume=resume,
        batch_size=batch_size,
        qids_filter=qids,
        stage_name="score-lora-fold",
        model_loader=lambda: load_adapter(adapter_dir, device=device, merge=merge_adapter),
        execution_metadata={"merge_adapter": merge_adapter},
        token_cache_dir=token_cache_dir,
    )


def evaluate_lora_fold(
    *,
    fold: int,
    candidates_path: Path,
    scores_path: Path,
    train_path: Path,
    folds_path: Path,
    zero_shot_metrics_path: Path,
    output_path: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    fold_name = f"fold_{fold}"
    heldout = load_folds(folds_path)[fold_name]
    heldout_set = set(heldout)
    candidates = _load_stage1(candidates_path, heldout_set, logger)
    scores = [row for row in read_jsonl(scores_path) if row["qid"] in heldout_set]
    zero_shot = json.loads(zero_shot_metrics_path.read_text(encoding="utf-8"))
    config = zero_shot["chosen_config"][fold_name]
    ce = aggregate_ce_documents(scores, gamma=float(config["gamma"]))
    predictions = {
        qid: [
            row["doc_id"]
            for row in fuse_stage1_and_ce(
                candidates[qid], ce[qid], ce_weight=float(config["ce_weight"]), limit=5
            )
        ]
        for qid in heldout
    }
    answers = load_answers(train_path)
    metrics = evaluate_rankings(predictions, {qid: answers[qid] for qid in heldout}, ks=(5,))
    baseline = zero_shot["fold_metrics"][fold_name]
    result = {
        "fold": fold,
        "metrics": metrics,
        "zero_shot_metrics": baseline,
        "recall_delta": metrics["recall@5"] - baseline["recall@5"],
        "precision_delta": metrics["precision@5"] - baseline["precision@5"],
        "continuation_gate": (
            metrics["recall@5"] - baseline["recall@5"] >= 0.005
            and metrics["precision@5"] - baseline["precision@5"] >= -0.001
        ),
        "predictions": predictions,
    }
    atomic_json(output_path, result)
    return result


def evaluate_oof(
    fold_result_paths: list[Path],
    train_path: Path,
    output_path: Path,
    *,
    candidates_path: Path | None = None,
    fold_score_paths: list[Path] | None = None,
    folds_path: Path | None = None,
    zero_shot_metrics_path: Path | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    predictions: dict[str, list[str]] = {}
    fold_metrics: dict[str, Any] = {}
    for path in fold_result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        fold_metrics[f"fold_{result['fold']}"] = result["metrics"]
        overlap = set(predictions) & set(result["predictions"])
        if overlap:
            raise ValueError(f"OOF query overlap: {sorted(overlap)[:5]}")
        predictions.update(result["predictions"])
    answers = load_answers(train_path)
    if set(predictions) != set(answers):
        raise ValueError(f"OOF coverage mismatch: {len(predictions)}/{len(answers)}")
    chosen_config = None
    # Once all fold-isolated logits exist, tune the aggregation/fusion knobs on
    # the other four OOF folds. No query is scored or tuned with an adapter that
    # trained on that query.
    if candidates_path is not None or fold_score_paths is not None or folds_path is not None:
        if candidates_path is None or fold_score_paths is None or folds_path is None:
            raise ValueError("OOF retuning requires candidates, every fold score file, and folds")
        if logger is not None:
            logger.log("phase=load-oof-artifacts")
        stage1 = _load_stage1(candidates_path, logger=logger)
        score_rows = [
            {"qid": row["qid"], "doc_id": row["doc_id"], "score": row["score"]}
            for path in fold_score_paths
            for row in read_jsonl(path)
        ]
        tuned = nested_tune_zero_shot(stage1, score_rows, answers, load_folds(folds_path))
        predictions = tuned["predictions"]
        fold_metrics = tuned["fold_metrics"]
        chosen_config = tuned["chosen_config"]
    metrics = evaluate_rankings(predictions, answers, ks=(5,))
    if zero_shot_metrics_path is None:
        raise ValueError("OOF promotion requires the zero-shot baseline report")
    zero_shot = json.loads(zero_shot_metrics_path.read_text(encoding="utf-8"))["metrics"]
    recall_delta = metrics["recall@5"] - float(zero_shot["recall@5"])
    precision_delta = metrics["precision@5"] - float(zero_shot["precision@5"])
    checks = {
        "recall_gain_pass": recall_delta >= 0.005,
        "precision_non_regression_pass": precision_delta >= -0.001,
    }
    gate = {**checks, "overall_pass": all(checks.values())}
    result = {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "chosen_config": chosen_config,
        "promotion_gate": gate,
        "zero_shot_metrics": zero_shot,
        "zero_shot_delta": {"recall@5": recall_delta, "precision@5": precision_delta},
        "target_088": metrics["recall@5"] >= 0.88,
        "stretch_090": metrics["recall@5"] >= 0.90,
    }
    atomic_json(output_path, result)
    return result


def write_submission(
    predictions: Mapping[str, list[str]],
    output_dir: Path,
    *,
    promotion_report: Path,
) -> tuple[Path, Path]:
    report = json.loads(promotion_report.read_text(encoding="utf-8"))
    if not _gate_passed(report):
        raise RuntimeError("Public submission blocked: OOF promotion gate did not pass")
    normalized = {}
    for qid, documents in predictions.items():
        unique = list(dict.fromkeys(map(str, documents)))[:5]
        if not 1 <= len(unique) <= 5:
            raise ValueError(f"Invalid submission cardinality for {qid}: {len(unique)}")
        normalized[str(qid)] = {"answer": unique}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "submission.json"
    json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")
    return json_path, zip_path


def build_public_submission(
    *,
    candidates_path: Path,
    scores_path: Path,
    zero_shot_metrics_path: Path,
    oof_report_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    candidates = _load_stage1(candidates_path)
    oof_report = json.loads(oof_report_path.read_text(encoding="utf-8"))
    tuning = oof_report.get("chosen_config")
    if not tuning:
        tuning = json.loads(zero_shot_metrics_path.read_text(encoding="utf-8"))["chosen_config"]
    # Use the modal fold configuration; deterministic lexical tie-break.
    serialized = [json.dumps(value, sort_keys=True) for value in tuning.values()]
    selected_key = sorted(set(serialized), key=lambda key: (-serialized.count(key), key))[0]
    config = json.loads(selected_key)
    ce = aggregate_ce_documents(read_jsonl(scores_path), gamma=float(config["gamma"]))
    predictions = {
        qid: [
            row["doc_id"]
            for row in fuse_stage1_and_ce(
                candidate_rows, ce[qid], ce_weight=float(config["ce_weight"]), limit=5
            )
        ]
        for qid, candidate_rows in candidates.items()
    }
    return write_submission(predictions, output_dir, promotion_report=oof_report_path)
