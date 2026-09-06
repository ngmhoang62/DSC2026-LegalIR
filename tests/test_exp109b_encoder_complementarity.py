from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exp109b_encoder_complementarity as exp  # noqa: E402


def test_model_contracts_are_locked_and_allowed_only() -> None:
    assert tuple(exp.MODEL_SPECS) == ("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal")
    assert exp.MODEL_SPECS["vietlegal_e5"].query_prefix == "query: "
    assert exp.MODEL_SPECS["vietlegal_e5"].document_prefix == "passage: "
    assert exp.MODEL_SPECS["vietlegal_harrier_0_6b"].query_prefix.startswith("Instruct: Given a Vietnamese legal question")
    assert exp.MODEL_SPECS["vnlegal_lal"].max_length == 2048
    assert exp.MODEL_SPECS["vnlegal_lal"].pooling == "last_non_padding"
    assert all(spec.dimension == 1024 for spec in exp.MODEL_SPECS.values())


def test_prepare_texts_does_not_change_structural_document_text() -> None:
    spec = exp.MODEL_SPECS["vietlegal_harrier_0_6b"]
    assert exp.prepare_texts(["văn bản"], spec, is_query=False) == ["văn bản"]
    assert exp.prepare_texts(["văn bản"], spec, is_query=True)[0].startswith("Instruct:")


def test_last_non_padding_pooling_handles_left_and_right_padding() -> None:
    right_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    left_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
    assert exp.last_non_padding_positions(right_mask).tolist() == [2, 1]
    assert exp.last_non_padding_positions(left_mask).tolist() == [4, 4]
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    assert torch.equal(exp.pool_last_non_padding(hidden, right_mask), torch.stack([hidden[0, 2], hidden[1, 1]]))
    assert torch.equal(exp.pool_last_non_padding(hidden, left_mask), torch.stack([hidden[0, 4], hidden[1, 4]]))
    with pytest.raises(ValueError):
        exp.last_non_padding_positions(torch.zeros((1, 3), dtype=torch.long))


def test_fp16_vectors_are_cast_then_renormalized() -> None:
    original = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    values = exp.renormalize_fp16(original.astype(np.float16))
    assert values.dtype == np.float32
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
    assert values[0, 0] == pytest.approx(0.6, abs=2e-4)


def test_top2_parent_scorer_matches_independent_reference() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b", "b", "b", "c"])
    documents = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    actual = exp.compute_parent_scores(query, documents, index, chunk_block_size=2)
    expected = exp.reference_parent_scores(query[0], documents, index).unsqueeze(0)
    assert torch.allclose(actual, expected)
    assert actual[0].tolist() == pytest.approx([0.5, 1.0, 0.0])
    actual.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_reference_scorer_preserves_fp32_documents_without_requantization() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b"])
    documents = np.asarray([[1.00031, 0.00012], [0.30123, 0.95387], [0.50019, 0.86588]], dtype=np.float32)
    query = torch.tensor([0.90117, 0.43346], dtype=torch.float32)
    actual = exp.reference_parent_scores(query, documents, index).detach().cpu().numpy()
    expected = exp.numpy_top2_parent_scores(query.detach().cpu().numpy(), documents, index)
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_secondary_aggregations_are_diagnostics_only_and_finite() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "a", "b"])
    documents = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float16)
    diagnostics = exp.numpy_secondary_parent_scores(np.asarray([1.0, 0.0]), documents, index)
    assert set(diagnostics) == {"max", "normalized_logsumexp"}
    assert diagnostics["max"].tolist() == pytest.approx([1.0, 1.0])
    assert np.isfinite(diagnostics["normalized_logsumexp"]).all()


def test_noncontiguous_parent_mapping_and_one_chunk_are_exact() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["a", "b", "a", "c", "b"])
    assert index.noncontiguous
    documents = torch.randn(5, 9, dtype=torch.float64)
    query = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    actual = exp.compute_parent_scores(query, documents, index, chunk_block_size=2)
    expected = torch.stack([exp.reference_parent_scores(row, documents, index) for row in query])
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    actual.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_numpy_scorer_uses_top2_and_stable_parent_ties() -> None:
    index = exp.CorpusIndex.from_chunk_doc_ids(["doc-b", "doc-a", "doc-a"])
    documents = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float16)
    values = exp.numpy_top2_parent_scores(np.asarray([1.0, 0.0], dtype=np.float32), documents, index)
    assert values.tolist() == pytest.approx([1.0, 1.0])
    assert exp.stable_rank(values, index.doc_ids) == ["doc-a", "doc-b"]


def test_stable_rank_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError):
        exp.stable_rank([float("nan")], ["a"])


def test_canonical_labels_match_frozen_policy() -> None:
    answers, stats = exp.canonical_labels()
    assert len(answers) == 7000
    assert stats["evaluable_queries"] == 6991
    assert stats["non_evaluable_queries"] == 9
    assert stats["affected_queries"] == 13
    assert stats["canonicalized_duplicate_occurrences"] == 2
    assert stats["dropped_empty_occurrences"] == 11
    assert stats["label_fingerprint"] == "9bdf9593b61fe3423d1f1a819ac9fb3e8d7225e6003da0afb840c1f5853fd4c9"


def test_folds_have_exactly_one_membership_and_nested_partitions_exclude_outer() -> None:
    folds, fold_for = exp.load_folds()
    assert set(folds) == set(exp.FOLD_NAMES)
    assert len(fold_for) == 7000
    parts = exp.nested_partitions({"fold_0": ["a", "b"], "fold_1": ["c"], "fold_2": ["d"]}, "fold_0")
    assert len(parts) == 2
    assert all(not set(part["train_qids"]) & {"a", "b"} for part in parts)
    assert {qid for part in parts for qid in part["validation_qids"]} == {"c", "d"}


def test_metrics_use_multi_gold_fraction_not_hit_any_only() -> None:
    answers = {"q1": {"a"}, "q2": {"a", "b"}}
    predictions = {"q1": ["a"], "q2": ["b", "x", "z"]}
    metrics = exp.evaluate_rankings(predictions, answers, ["q1", "q2"])
    assert metrics["recall@5"] == pytest.approx(0.75)
    assert metrics["multi_gold_recall@5"] == pytest.approx(0.5)
    assert metrics["precision@5"] == pytest.approx(2.0 / 10.0)


def _bounded_rows() -> list[dict[str, object]]:
    rows = []
    for i in range(12):
        qid = f"q{i}"
        rows.append({
            "qid": qid,
            "fold": "fold_1" if i < 6 else "fold_2",
            "candidate_doc_ids": [f"d{j}" for j in range(40)],
            "gold_doc_ids": [f"g{i}"],
            "role": "control" if i == 0 else "error",
            "flags": {"bounded_oracle_fixture_not_recall": True, "gold_force_included": True, "may_not_be_reported_as_full_corpus_recall": True},
        })
    return rows


def test_bounded_metrics_keep_oracle_warning_and_count_rescues_losses() -> None:
    rows = _bounded_rows()
    base = {str(row["qid"]): [f"x{j}" for j in range(5)] + [str(row["gold_doc_ids"][0])] for row in rows}
    alt = {str(row["qid"]): ([str(row["gold_doc_ids"][0])] + [f"x{j}" for j in range(5)]) for row in rows}
    metrics = exp.bounded_model_metrics(rows, e5_rankings=base, candidate_rankings=alt, source_generation_rankings=alt)
    assert len(metrics["rescued_top5_qids"]) == 12
    assert metrics["net_top5_rescues"] == 12
    assert metrics["control_exit_rate"] == 0.0
    assert metrics["flags"]["bounded_oracle_fixture_not_recall"] is True
    assert metrics["flags"]["gold_force_included"] is True
    gate = exp.bounded_gate(metrics, included_folds=["fold_1", "fold_2"], source_generation_miss_count=12)
    assert gate["checks"]["at_least_10_top5_rescues"] is True
    assert "control_exit_rate_le_0_02" not in gate["checks"]
    assert gate["diagnostics"]["not_a_bounded_pass_check"] is True
    assert gate["minimum_source_generation_additions"] == 8


def test_bounded_loader_reads_manifest_not_success_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "CACHE_ROOT", tmp_path)
    fixture_dir = tmp_path / "bounded_fixture" / "fold_0"
    rows = [{
        "qid": "q1",
        "fold": "fold_1",
        "candidate_doc_ids": ["d1"],
        "gold_doc_ids": ["d1"],
        "flags": {
            "bounded_oracle_fixture_not_recall": True,
            "gold_force_included": True,
            "may_not_be_reported_as_full_corpus_recall": True,
        },
    }]
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "retrieval_text": "text"}]
    exp.write_jsonl_atomic(fixture_dir / "queries.jsonl", rows)
    exp.write_jsonl_atomic(fixture_dir / "chunks.jsonl", chunks)
    fingerprint = "fixture-fingerprint"
    exp.atomic_json(fixture_dir / "manifest.json", {
        "content_fingerprint": fingerprint,
        "bounded_oracle_fixture_not_recall": True,
        "gold_force_included": True,
        "may_not_be_reported_as_full_corpus_recall": True,
        "queries_sha256": exp.sha256_file(fixture_dir / "queries.jsonl"),
        "chunks_sha256": exp.sha256_file(fixture_dir / "chunks.jsonl"),
    })
    exp.write_success(fixture_dir, stage="bounded-fixture", fingerprint=fingerprint)
    loaded_rows, loaded_chunks, index, candidates = exp._load_bounded_data("fold_0")
    assert loaded_rows == rows
    assert loaded_chunks == chunks
    assert index.doc_ids == ["d1"]
    assert candidates == {"q1": [0]}


def test_model_selection_is_lal_only_after_gate_repair() -> None:
    def item(rescues: list[str]) -> dict[str, object]:
        return {"gate": {"pass": True}, "metrics": {"rescued_top5_qids": rescues, "net_top5_rescues": len(rescues), "source_generation_top32_additions": 10, "control_exit_rate": 0.0, "normalized_rank_improvement": {"lower": 0.1}}, "estimated_encode_seconds": 1}
    observed = exp.choose_models_from_bounded({"vietlegal_harrier_0_6b": item(["a", "b", "c", "d", "e"]), "vnlegal_lal": item(["f", "g", "h", "i", "j"])})
    assert observed["status"] == "PASS_LAL_ONLY_GATE_REPAIR"
    assert observed["selected_models"] == ["vnlegal_lal"]


def test_lal_failure_is_hard_stop() -> None:
    failed = {"gate": {"pass": False}, "metrics": {}}
    result = exp.choose_models_from_bounded({"vietlegal_harrier_0_6b": {"gate": {"pass": True}, "metrics": {}}, "vnlegal_lal": failed})
    assert result["status"] == "REJECTED_BOUNDED_COMPLEMENTARITY"
    assert result["selected_models"] == []


def test_verified_bounded_rankings_reject_hash_or_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "lal.rankings.jsonl"
    exp.write_jsonl_atomic(path, [{"qid": "q1", "documents": ["d1", "d2"]}])
    observed = exp._read_verified_bounded_rankings(
        path,
        expected_sha256=exp.sha256_file(path),
        expected_qids={"q1"},
        candidate_doc_ids={"q1": {"d1", "d2"}},
    )
    assert observed == {"q1": ["d1", "d2"]}
    with pytest.raises(RuntimeError, match="SHA-256"):
        exp._read_verified_bounded_rankings(path, expected_sha256="wrong", expected_qids={"q1"}, candidate_doc_ids={"q1": {"d1", "d2"}})


def test_bootstrap_is_deterministic() -> None:
    values = [0.1, 0.2, -0.1, 0.4]
    assert exp.deterministic_bootstrap(values) == exp.deterministic_bootstrap(values)


def test_rrf_simplex_and_missing_source_reference() -> None:
    weights = exp.simplex_weights(("vietlegal_e5", "bm25"))
    assert weights
    assert all(sum(item.values()) == pytest.approx(1.0) for item in weights)
    assert all(item["vietlegal_e5"] >= 0.3 and item["bm25"] >= 0.1 for item in weights)
    rankings = {"vietlegal_e5": ["a", "b"], "bm25": ["b", "c"]}
    scores = exp.weighted_rrf_scores(rankings, {"vietlegal_e5": 0.5, "bm25": 0.5}, rrf_k=10)
    assert scores["a"] == pytest.approx(0.5 / 11)
    assert scores["b"] == pytest.approx(0.5 / 12 + 0.5 / 11)
    assert exp.weighted_rrf_ranking(rankings, {"vietlegal_e5": 0.5, "bm25": 0.5}, rrf_k=10)[0] == "b"


def test_candidate_depth_selection_records_outer_train_provenance() -> None:
    folds = {"fold_0": ["q0"], "fold_1": ["q1"], "fold_2": ["q2"]}
    answers = {qid: {"gold"} for qid in ("q0", "q1", "q2")}
    source = {
        "e5": {qid: ["noise"] * 100 + ["gold"] for qid in answers},
        "bm25": {qid: ["gold"] for qid in answers},
    }
    result = exp.select_candidate_depth(source, answers, folds, heldout="fold_0")
    assert result["fold_0"]["heldout_excluded"] == "fold_0"
    assert result["fold_0"]["selection_qids"] == 2
    assert result["fold_0"]["depth"] in (100, 200, 500)


def test_feature_contract_has_no_label_or_id_features() -> None:
    exp.validate_feature_contract(exp.ALLOWED_FEATURES)
    assert not any(any(token in name.lower() for token in exp.FORBIDDEN_FEATURE_TOKENS) for name in exp.ALLOWED_FEATURES)
    source_maps = {name: {} for name in ("vietlegal_e5", "vietlegal_harrier_0_6b", "vnlegal_lal", "bm25")}
    for source in source_maps:
        source_maps[source] = {"d": (1, 0.5)} if source == "vietlegal_e5" else {}
    matrix, names = exp.build_lambdamart_features("q", ["d"], source_maps, {"d": {"parent_chunk_count": 2, "parent_token_length": 20}}, query_token_length=7, depth=50)
    assert matrix.shape == (1, len(exp.ALLOWED_FEATURES))
    assert tuple(names) == exp.ALLOWED_FEATURES


def test_lambdamart_grid_and_baseline_feature_schema_are_fixed() -> None:
    assert len(exp.lambdamart_hyperparameter_grid()) == 16
    source_maps = {"vietlegal_e5": {"d": (1, 0.5)}, "bm25": {"d": (1, 0.2)}}
    matrix, names = exp.build_lambdamart_features("q", ["d"], source_maps, {"d": {"parent_chunk_count": 1, "parent_token_length": 4}}, query_token_length=2, depth=50)
    assert matrix.shape == (1, len(exp.ALLOWED_FEATURES))
    assert tuple(names) == exp.ALLOWED_FEATURES
    assert matrix[0, list(exp.ALLOWED_FEATURES).index("harrier_present")] == 0.0


def test_oom_fallback_is_ordered_and_stops_at_success() -> None:
    calls: list[int] = []
    def operation(batch: int) -> str:
        calls.append(batch)
        if batch > 2:
            raise RuntimeError("CUDA out of memory")
        return "ok"
    effective, value, attempts = exp.oom_safe_batch(operation)
    assert effective == 2 and value == "ok"
    assert calls == [8, 4, 2]
    assert [item["status"] for item in attempts] == ["OOM", "OOM", "PASS"]


def test_shard_receipt_and_corruption_are_fail_closed(tmp_path: Path) -> None:
    vectors = np.ones((2, exp.DIMENSION), dtype=np.float32)
    path = tmp_path / "shard.npz"
    fingerprint = "f" * 64
    receipt = exp._save_embedding_shard(path, ids=["c1", "c2"], parent_ids=["d", "d"], vectors=vectors, token_counts=np.asarray([2, 3]), truncation_count=0, manifest_fingerprint=fingerprint)
    assert receipt["count"] == 2
    assert exp.validate_embedding_shard(path, expected_fingerprint=fingerprint)["count"] == 2
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError):
        exp.validate_embedding_shard(path, expected_fingerprint=fingerprint)


def test_query_embedding_receipt_is_fingerprint_and_shape_checked(tmp_path: Path) -> None:
    path = tmp_path / "queries.npz"
    fingerprint = "q" * 64
    np.savez_compressed(
        path,
        query_ids=np.asarray(["q0"], dtype="U"),
        vectors=np.ones((1, exp.DIMENSION), dtype=np.float16),
        token_counts=np.asarray([2], dtype=np.int32),
        truncation_count=np.asarray([0], dtype=np.int64),
        manifest_fingerprint=np.asarray([fingerprint], dtype="U"),
    )
    assert exp.validate_query_embeddings(path, expected_fingerprint=fingerprint, expected_count=1)["count"] == 1
    with pytest.raises(RuntimeError):
        exp.validate_query_embeddings(path, expected_fingerprint="x" * 64, expected_count=1)


def test_nested_rrf_inner_predictions_exclude_outer_fold() -> None:
    folds = {f"fold_{index}": [f"q{index}"] for index in range(5)}
    answers = {f"q{index}": {f"g{index}"} for index in range(5)}
    rankings = {
        "vietlegal_e5": {qid: [next(iter(gold)), "noise"] for qid, gold in answers.items()},
        "bm25": {qid: ["noise", next(iter(gold))] for qid, gold in answers.items()},
    }
    result = exp.nested_rrf_screen(
        rankings,
        answers,
        folds,
        outer="fold_0",
        included_sources=("vietlegal_e5", "bm25"),
        candidate_depth=100,
    )
    assert set(result["inner_predictions"]) == {"q1", "q2", "q3", "q4"}
    assert "q0" not in result["inner_predictions"]
    assert set(result["heldout_predictions"]) == {"q0"}
    assert result["chosen"]["metrics"] == result["inner_metrics"]
    assert result["selection_scope"] == "outer_train_inner_cv"


def test_cached_pilot_gate_is_fold0_free_and_enforces_safety_metrics() -> None:
    folds = {f"fold_{index}": [f"q{index}"] for index in range(1, 5)}
    answers = {f"q{index}": {f"g{index}"} for index in range(5)}
    baseline = {f"q{index}": ["n1", "n2", "n3", "n4", "n5", f"g{index}"] for index in range(1, 5)}
    candidate = {f"q{index}": [f"g{index}", "noise"] for index in range(1, 5)}
    gate = exp._pilot_gate(baseline, candidate, answers, folds)
    assert gate["qids"] == 4
    assert gate["pass"] is True
    assert set(gate["per_fold_delta_recall@5"]) == set(folds)
    assert "q0" not in baseline and "q0" not in candidate


def test_simplex_weights_support_lal_bm25_without_e5() -> None:
    weights = exp.simplex_weights(("vnlegal_lal", "bm25"))
    assert weights
    assert all(sum(item.values()) == pytest.approx(1.0) for item in weights)


def test_success_marker_is_fingerprint_checked(tmp_path: Path) -> None:
    directory = tmp_path / "stage"
    exp.write_success(directory, stage="test", fingerprint="abc")
    assert exp.require_success(directory, "abc")["status"] == "PASS"
    with pytest.raises(RuntimeError):
        exp.require_success(directory, "different")
