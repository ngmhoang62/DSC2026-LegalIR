from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import exp109c_latent_condition_late_interaction as exp  # noqa: E402


def test_marker_and_special_token_contract() -> None:
    assert exp.marker_text("abc", "query") == "[QueryMarker] abc"
    assert exp.marker_text("abc", "document") == "[DocumentMarker] abc"
    assert exp.strip_special_token_positions([0, 4, 5, 2], [1, 1, 1, 1], [0, 2]).tolist() == [1, 2]
    with pytest.raises(ValueError):
        exp.jina_marker("other")


def test_projection_and_normalization_shape() -> None:
    hidden = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1
    projection = np.eye(4, dtype=np.float32)
    values = exp.project_and_normalize(hidden, projection, dimensions=4)
    assert values.shape == (2, 3, 4)
    assert np.allclose(np.linalg.norm(values, axis=-1), 1.0)
    with pytest.raises(ValueError):
        exp.project_and_normalize(hidden, np.eye(3, dtype=np.float32), dimensions=4)


def test_numpy_torch_maxsim_parity() -> None:
    rng = np.random.default_rng(109)
    query = exp.l2_normalize(rng.normal(size=(4, 8)).astype(np.float32))
    document = exp.l2_normalize(rng.normal(size=(7, 8)).astype(np.float32))
    value, matches = exp.numpy_maxsim(query, document)
    if exp.torch is not None:
        torch_value, torch_matches = exp.torch_maxsim(exp.torch.tensor(query), exp.torch.tensor(document))
        assert float(torch_value) == pytest.approx(value, abs=1e-5)
        assert torch_matches.detach().numpy() == pytest.approx(matches, abs=1e-5)


def test_int8_row_scale_parity_is_renormalized() -> None:
    rng = np.random.default_rng(42)
    values = exp.l2_normalize(rng.normal(size=(96, 128)).astype(np.float32))
    packed, scales = exp.quantize_rows(values)
    restored = exp.dequantize_rows(packed, scales)
    assert packed.dtype == np.int8
    assert scales.dtype == np.float16
    assert np.allclose(np.linalg.norm(restored, axis=1), 1.0, atol=2e-5)
    assert np.mean(np.abs(values - restored)) < 0.003


def test_idf_anchor_selection_keeps_prefix_digit_tail_and_is_deterministic() -> None:
    ids = list(range(12))
    mask = [1] * len(ids)
    tokens = ["prefix", "heading", "điều", "x", "2024", "x", "x", "x", "x", "x", "x", "tail"]
    first = exp.select_anchor_indices(ids, mask, maximum=6, idf={index: float(index) for index in ids}, structural_prefix_count=2, token_texts=tokens, tail_tokens=1)
    second = exp.select_anchor_indices(ids, mask, maximum=6, idf={index: float(index) for index in ids}, structural_prefix_count=2, token_texts=tokens, tail_tokens=1)
    assert first.tolist() == second.tolist()
    assert {0, 1, 2, 4, 11}.issubset(set(first.tolist()))
    assert len(first) == 6
    assert first.tolist() == sorted(first.tolist())


def test_anchor_selection_bounds_many_numeric_pieces_without_dropping_hard_anchors() -> None:
    ids = list(range(160)); selected = exp.select_anchor_indices(ids, [1] * len(ids), maximum=96, idf={value: float(value) for value in ids}, structural_prefix_count=2, token_texts=[f"{value}" for value in ids], tail_tokens=8)
    assert len(selected) == 96
    assert {0, 1, 152, 153, 154, 155, 156, 157, 158, 159}.issubset(set(selected.tolist()))


def test_length_policy_is_label_free_and_deterministic() -> None:
    result = exp.choose_length_by_coverage([10, 50, 60, 130], [64, 96, 128], .75)
    assert result["selected"] == 64
    assert result["truncated_count"] == 1
    assert result["curve"] == {"64": .75, "96": .75, "128": .75}


def _meta(chunk_id: str, start: int, end: int, tokens: int = 4) -> dict[str, object]:
    return {"chunk_id": chunk_id, "source_start": start, "source_end": end, "retained_tokens": tokens}


def test_two_chunk_union_covers_disjoint_conditions_and_tie_breaks() -> None:
    vectors = [
        np.asarray([.9, .1, .1, .1], dtype=np.float32),
        np.asarray([.1, .9, .1, .1], dtype=np.float32),
        np.asarray([.9, .1, .1, .1], dtype=np.float32),
    ]
    top1, second, union = exp.select_two_chunks(vectors, [_meta("c1", 0, 10), _meta("c2", 20, 30), _meta("c3", 0, 10)])
    assert top1 == 0
    assert second == 1
    assert union.tolist() == pytest.approx([.9, .9, .1, .1])
    features = exp.latent_condition_features(vectors, [_meta("c1", 0, 10), _meta("c2", 20, 30), _meta("c3", 0, 10)])
    assert features["li_two_chunk_incremental_gain"] > 0
    assert features["li_selected_chunk_count"] == 2


def test_parent_length_lottery_diagnostics_and_feature_schema() -> None:
    vectors = [np.asarray([.8, .4], dtype=np.float32), np.asarray([.4, .8], dtype=np.float32)]
    features = exp.latent_condition_features(vectors, [_meta("a", 0, 2, 2), _meta("b", 3, 5, 2)], parent_token_count=4)
    record = {name: features[name] for name in exp.LATE_FEATURES}
    record["li_exact_rank_within_candidate_pool"] = 1
    assert exp.build_late_feature_vector({**{name: 0.0 for name in exp.SCALAR_109B_FEATURES}, **record}).shape == (len(exp.ALLOWED_FEATURES),)
    exp.validate_feature_contract(exp.ALLOWED_FEATURES)
    with pytest.raises(ValueError):
        exp.validate_feature_contract(exp.ALLOWED_FEATURES + ("doc_id",))


def test_candidate_union_unique_rank_provenance_and_depth() -> None:
    sources = {
        "vietlegal_e5": [{"doc_id": "b", "rank": 1, "score": .9}, {"doc_id": "a", "rank": 2, "score": .8}],
        "vnlegal_lal": [{"doc_id": "a", "rank": 1, "score": .7}, {"doc_id": "c", "rank": 2, "score": .6}],
        "bm25": [{"doc_id": "d", "rank": 1, "score": .5}],
    }
    rows = exp.build_candidate_union(sources, depth=2)
    assert [row["doc_id"] for row in rows] == ["a", "b", "d", "c"]
    assert rows[0]["sources"]["vietlegal_e5"]["rank"] == 2
    assert len({row["doc_id"] for row in rows}) == 4


def test_fold0_candidate_validation_preserves_variable_d100_source_union() -> None:
    sources = {
        "vietlegal_e5": [{"doc_id": "a", "rank": 1}, {"doc_id": "b", "rank": 2}],
        "vnlegal_lal": [{"doc_id": "b", "rank": 1}, {"doc_id": "c", "rank": 2}],
        "bm25": [{"doc_id": "d", "rank": 1}],
    }
    union = exp.build_candidate_union(sources, depth=100)
    scores = [{"doc_id": row["doc_id"], "candidate_rank": row["candidate_rank"], "features": {name: 0.0 for name in exp.ALLOWED_FEATURES}} for row in reversed(union)]
    ids, digest = exp._validate_fold0_score_row({"qid": "q", "scores": scores}, sources)
    assert ids == [row["doc_id"] for row in union]
    assert digest == exp.content_hash(ids)
    assert len(ids) == 4  # No post-union truncation to per-source depth.
    scores[0]["candidate_rank"] = 99
    with pytest.raises(exp.GateRejected, match="candidate ranks differ"):
        exp._validate_fold0_score_row({"qid": "q", "scores": scores}, sources)


def test_final_frozen_config_policy_is_deterministic() -> None:
    def selected(config: dict[str, float | int], recall: float) -> dict[str, object]:
        return {"selected": {"config": config, "metrics": {"recall@5": recall, "precision@5": .2, "multi_gold_recall@5": .7, "mrr@5": .8}}}
    small = {"num_leaves": 7, "min_data_in_leaf": 50, "learning_rate": .03, "num_boost_round": 200}
    large = {"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": .03, "num_boost_round": 200}
    tie = {"fold_1": selected(large, .93), "fold_2": selected(small, .93), "fold_3": selected(large, .93), "fold_4": selected(small, .93)}
    result = exp.select_final_frozen_config(tie)
    assert result["config"] == small
    majority = dict(tie); majority["fold_4"] = selected(large, .90)
    assert exp.select_final_frozen_config(majority)["config"] == large


def test_frozen_fold0_evaluation_rejects_before_any_label_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "RESULTS_ROOT", tmp_path)
    exp.atomic_json(tmp_path / "FROZEN_WINNER_LOCK.json", {"status": "PASS_FROZEN_WINNER_LOCK", "lock_fingerprint": "fixture"})
    monkeypatch.setattr(exp, "canonical_labels", lambda: (_ for _ in ()).throw(AssertionError("labels must not be read")))
    with pytest.raises(exp.GateRejected, match="FOLD0_PREDICTION_LOCK"):
        exp.evaluate_fold0_frozen(authorize=True)


def test_candidate_ceiling_gate_uses_outer_train_only() -> None:
    folds = {"fold_0": ["q0"], "fold_1": ["q1"], "fold_2": ["q2"], "fold_3": ["q3"], "fold_4": ["q4"]}
    answers = {qid: {"gold"} for qid in folds for qid in folds[qid]}
    ranking = {qid: {"vietlegal_e5": [{"doc_id": "gold", "rank": 1}], "vnlegal_lal": [], "bm25": []} for qid in answers}
    report = exp.candidate_ceiling_report(ranking, answers, folds, outer="fold_0")
    assert "q0" not in report["curves"]["200"]["per_inner_fold"]
    assert report["gate"]["pass"] is True
    assert report["claim_boundary"].startswith("candidate coverage")


def test_candidate_ceiling_helper_is_pure_and_anchor_budget_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "RESULTS_ROOT", tmp_path / "results")
    folds = {f"fold_{index}": [f"q{index}"] for index in range(5)}
    answers = {qid: {"gold"} for qids in folds.values() for qid in qids}
    sources = {qid: {"vietlegal_e5": [{"doc_id": "gold", "rank": 1}], "vnlegal_lal": [], "bm25": []} for qid in answers}
    exp.candidate_ceiling_report(sources, answers, folds, outer="fold_0")
    assert not (tmp_path / "results" / "CANDIDATE_CEILING_REPORT.json").exists()
    with pytest.raises(exp.GateRejected, match="mandatory anchors"):
        exp.select_anchor_indices([1, 2, 3, 4], [1, 1, 1, 1], maximum=2, structural_prefix_count=2, token_texts=["a", "b", "3", "tail"], tail_tokens=1)


def test_unfingerprinted_candidate_gate_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "RESULTS_ROOT", tmp_path)
    exp.atomic_json(tmp_path / "CANDIDATE_CEILING_REPORT.json", {"status": "PASS_CANDIDATE_CEILING", "outer": "fold_0", "input_fingerprints": {}})
    with pytest.raises(exp.GateRejected, match="current verified source manifests"):
        exp.require_current_candidate_report()


def test_parent_index_and_bounded_scorer() -> None:
    index = exp.ParentIndex.from_chunk_doc_ids(["b", "a", "a"])
    assert index.doc_ids == ["b", "a"]
    assert index.indices("a") == [1, 2]
    query = np.asarray([[1., 0.]], dtype=np.float32)
    chunks = [(np.asarray([[1., 0.]], dtype=np.float32), _meta("a1", 0, 1, 1)), (np.asarray([[0., 1.]], dtype=np.float32), _meta("a2", 2, 3, 1))]
    result = exp.score_parent_from_chunks(query, chunks)
    assert result["selected_chunk_ids"] == ["a1", "a2"]


def test_numeric_diagnostic_accumulator_ignores_descriptive_fields() -> None:
    total: dict[str, float] = {}
    exp.accumulate_numeric_diagnostics(total, "approximate", {"chunks": 12, "wall_seconds": .25, "tier": "approximate_medoids", "ok": True})
    assert total == {"approximate_chunks": 12.0, "approximate_wall_seconds": .25}


def test_d100_futility_reject_requires_all_precommitted_checks() -> None:
    current_2048 = {"delta_recall@5": .0031377343, "bootstrap": {"upper": .00986145, "lower": -.00350448}, "choice_oracle_gain": .014995925, "folds_non_negative": 2, "multi_gold_delta": -.00339147, "worst_fold_delta": -.0029296875}
    decision = exp.d100_futility_decision(current_2048, target_query_count=2048)
    assert decision["status"] == "AMBIGUOUS_D100_FUTILITY_2048"
    assert decision["action"] == "run_next_precommitted_budget"
    only_multi_negative = {**current_2048, "choice_oracle_gain": .016}
    assert exp.d100_futility_decision(only_multi_negative, target_query_count=2048)["status"] == "AMBIGUOUS_D100_FUTILITY_2048"
    only_oracle_low = {**current_2048, "multi_gold_delta": .001}
    assert exp.d100_futility_decision(only_oracle_low, target_query_count=2048)["status"] == "AMBIGUOUS_D100_FUTILITY_2048"
    all_adverse = {"delta_recall@5": 0.0, "bootstrap": {"upper": .004, "lower": -.01}, "choice_oracle_gain": .014, "folds_non_negative": 1, "multi_gold_delta": -.001, "worst_fold_delta": -.02}
    assert exp.d100_futility_decision(all_adverse, target_query_count=2048)["status"] == "REJECTED_D100_FUTILITY_2048"
    strong = {**current_2048, "delta_recall@5": .004, "folds_non_negative": 3, "multi_gold_delta": 0.0, "choice_oracle_gain": .020}
    assert exp.d100_futility_decision(strong, target_query_count=2048)["status"] == "CONTINUE_D100_FUTILITY_2048"


@pytest.mark.skipif(exp.torch is None or not exp.torch.cuda.is_available(), reason="CUDA is required for batched scorer parity")
def test_batched_chunk_maxsim_preserves_negative_matches_padding_and_features() -> None:
    # Padding is deliberately longer than either chunk.  A zero-padding bug
    # would beat the valid negative match for the second query token.
    query = exp.l2_normalize(np.asarray([[1., 0., 0.], [0., -1., 0.]], dtype=np.float32))
    chunks = [
        (exp.l2_normalize(np.asarray([[1., 0., 0.]], dtype=np.float32)), _meta("one", 0, 1, 1)),
        (exp.l2_normalize(np.asarray([[0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]], dtype=np.float32)), _meta("three", 2, 5, 3)),
    ]
    reference = exp.score_parent_from_chunks_cuda(query, chunks, device="cuda")
    matches, diagnostics = exp.batched_chunk_matches_cuda(query, chunks, policy={"max_chunks": 8, "max_document_tokens": 32, "max_intermediate_bytes": 1024 * 1024}, device="cuda")
    actual = exp.score_parent_from_match_vectors(matches, [meta for _document, meta in chunks])
    assert diagnostics["chunks"] == 2
    for expected, observed in zip(reference["chunk_scores"], actual["chunk_scores"]):
        assert observed == pytest.approx(expected, abs=1e-6)
    for name in exp.LATE_FEATURES:
        assert actual["features"][name] == pytest.approx(reference["features"][name], abs=1e-6)
    assert actual["selected_chunk_ids"] == reference["selected_chunk_ids"]


def test_fidelity_and_gate() -> None:
    metrics = exp.fidelity_metrics([.9, .8, .7, .6, .5], [.9, .8, .7, .6, .5], positive_pairs=[(0, 1)])
    assert exp.fidelity_gate(metrics, anchors=96)["pass"] is True
    assert metrics["top5_agreement"] == 1.0
    failed = exp.fidelity_gate({"mean_absolute_error": .02, "top5_agreement": .9, "positive_ordering_retention": .5, "no_monotonic_error_explosion": True}, anchors=128)
    assert failed["status"] == "REJECTED_COMPRESSION_FIDELITY_GATE"


def test_fidelity_metrics_rank_only_within_each_query_and_reference_ordering() -> None:
    # A global top-5 would compare unrelated MaxSim scales from q1/q2.  The
    # grouped contract instead catches the reversal in q2 and compares pair
    # ordering against the full-token reference, not against relevance alone.
    full = [.9, .8, .2, .1]
    compressed = [.9, .8, .1, .2]
    metrics = exp.fidelity_metrics(full, compressed, qids=["q1", "q1", "q2", "q2"], doc_ids=["g1", "n1", "g2", "n2"], gold_by_qid={"q1": {"g1"}, "q2": {"g2"}})
    assert metrics["top5_agreement"] == 1.0
    assert metrics["pairwise_sign_agreement"] == .5
    assert metrics["conditional_positive_order_retention"] == .5
    assert metrics["false_flips_vs_reference"] == 1


def test_rescue_anchor_selectors_are_deterministic_and_bounded() -> None:
    values = exp.l2_normalize(np.arange(400, dtype=np.float32).reshape(100, 4) + 1)
    position = exp._position_stratified_indices(100, 16, [0, 1, 95, 99])
    medoids = exp._diversity_medoids(values, 16, [0, 1, 95, 99])
    assert len(position) <= 16 and len(medoids) == 16
    assert {0, 1, 95, 99}.issubset(set(position))
    assert {0, 1, 95, 99}.issubset(set(medoids))


def test_config_e_incremental_medoids_match_reference_and_keep_128d_values() -> None:
    rng = np.random.default_rng(109)
    values = exp.l2_normalize(rng.normal(size=(320, exp.CONFIG_E_STORAGE_DIMENSION)).astype(np.float32))
    geometry = exp.l2_normalize(values[:, :exp.CONFIG_E_SELECTION_DIMENSION])
    mandatory = exp.config_e_mandatory_indices(len(values))
    reference = exp._diversity_medoids_reference(geometry, exp.CONFIG_E_ANCHORS, mandatory)
    actual = exp.config_e_anchor_indices(values)
    assert actual.tolist() == reference.tolist()
    assert len(actual) == exp.CONFIG_E_ANCHORS
    assert values[actual].shape == (exp.CONFIG_E_ANCHORS, exp.CONFIG_E_STORAGE_DIMENSION)


def test_streaming_parent_maxsim_matches_concatenated_reference() -> None:
    rng = np.random.default_rng(109)
    query = rng.normal(size=(7, exp.DIMENSION)).astype(np.float32)
    chunks = [rng.normal(size=(3, exp.DIMENSION)).astype(np.float32), rng.normal(size=(11, exp.DIMENSION)).astype(np.float32)]
    expected = exp.numpy_maxsim(query, np.concatenate(chunks, axis=0))
    actual = exp.streaming_parent_maxsim(query, chunks)
    assert actual[0] == pytest.approx(expected[0], abs=1e-6)
    assert actual[1].tolist() == pytest.approx(expected[1].tolist(), abs=1e-6)


def test_e4_exact_union_is_deterministic_unique_and_label_free() -> None:
    approximate = {"d3": .3, "d2": .2, "d1": .1, "d4": .4}
    selected = exp.exact_refinement_set(approximate, ["d1", "d2", "d2", "outside"], top_k=16)
    assert selected == ["d4", "d3", "d2", "d1"]
    assert len(selected) == len(set(selected))
    with pytest.raises(ValueError, match="locks"):
        exp.exact_refinement_set(approximate, [], top_k=20)
    cascade = exp.cascade_refined_ranking(approximate, {"d1": .9})
    assert cascade[0] == "d1"


def test_multi_positive_loss_does_not_make_positives_compete() -> None:
    one = exp.multi_positive_loss([1.0], [.0, -.1])
    two = exp.multi_positive_loss([1.0, 1.0], [.0, -.1])
    assert two == pytest.approx(one)


def test_six_negative_curriculum_is_in_pool_and_excludes_gold() -> None:
    rows = [{"doc_id": f"d{i}", "candidate_rank": i + 1, "anchor_rank": i + 1, "source_presence": 2 if i == 6 else 1, "late_coverage": float(i)} for i in range(40)]
    selected = exp.select_six_negatives(rows, ["d0", "d10"], epoch=1)
    assert len(selected) == len(set(selected)) <= 6
    assert "d0" not in selected and "d10" not in selected
    assert set(selected) <= {row["doc_id"] for row in rows}


def test_three_seed_ensemble_and_identity_adapter() -> None:
    source = {"a": 2., "b": 1., "c": 0.}
    result = exp.ensemble_seed_scores({109: source, 110: source, 111: source})
    assert result["a"] > result["b"] > result["c"]
    if exp.torch is not None:
        assert exp.adapter_identity_score_parity() < 1e-6
    with pytest.raises(ValueError):
        exp.ensemble_seed_scores({109: source}, seeds=(109,))


def test_evaluated_prediction_report_requires_complete_unique_predictions() -> None:
    answers = {"q1": {"a"}, "q2": {"b"}}
    report = exp.evaluated_prediction_report(stage="fixture", predictions={"q1": ["a", "x"], "q2": ["x", "b"]}, answers=answers, qids=["q1", "q2"], anchor_predictions={"q1": ["x", "a"], "q2": ["x", "b"]})
    assert report["metrics"]["recall@5"] == 1.0
    assert report["fold_predictions_validated"] is True
    with pytest.raises(exp.GateRejected, match="qid coverage"):
        exp.evaluated_prediction_report(stage="fixture", predictions={"q1": ["a"]}, answers=answers, qids=["q1", "q2"])
    with pytest.raises(exp.GateRejected, match="duplicate"):
        exp.evaluated_prediction_report(stage="fixture", predictions={"q1": ["a", "a"], "q2": ["b"]}, answers=answers, qids=["q1", "q2"])


def test_locked_fold0_runner_accepts_complete_prediction_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(exp, "canonical_labels", lambda: ({"q0": {"gold"}}, {}))
    monkeypatch.setattr(exp, "load_folds", lambda: ({"fold_0": ["q0"]}, {}))
    exp.atomic_json(tmp_path / "FINAL_INNER_GATE.json", {"status": "PASS_FINAL_INNER_GATE"})
    exp.atomic_json(tmp_path / "FOLD0_PREDICTIONS.json", {"predictions": {"q0": ["gold"]}})
    report = exp.locked_fold0(authorize=True)
    assert report["status"] == "PASS_TARGET_098"


def test_shard_write_resume_and_corruption_rejection(tmp_path: Path) -> None:
    vectors = np.ones((3, exp.DIMENSION), dtype=np.int8)
    scales = np.ones((3,), dtype=np.float16)
    passages = [{"chunk_id": "c1", "doc_id": "d1", "token_start": 0, "token_end": 1}]
    receipt = exp.save_index_shard(tmp_path, 0, vectors, scales, passages, fingerprint="f" * 64)
    assert exp.verify_index_shard(tmp_path, receipt, fingerprint="f" * 64)["count"] == 1
    (tmp_path / receipt["vectors"]).write_bytes((tmp_path / receipt["vectors"]).read_bytes() + b"x")
    with pytest.raises(exp.GateRejected):
        exp.verify_index_shard(tmp_path, receipt, fingerprint="f" * 64)


def test_dual_shard_write_resume_and_full_store_loading(tmp_path: Path) -> None:
    approx = np.ones((2, exp.DIMENSION), dtype=np.int8); full = np.ones((4, exp.DIMENSION), dtype=np.int8)
    scales = np.ones((2,), dtype=np.float16); full_scales = np.ones((4,), dtype=np.float16)
    passages = [{"chunk_id": "c1", "doc_id": "d1", "token_start": 0, "token_end": 2, "full_token_start": 0, "full_token_end": 4}]
    receipt = exp.save_dual_index_shard(tmp_path, 0, approx, scales, full, full_scales, passages, fingerprint="a" * 64)
    assert receipt["dual_store"] is True
    assert exp.verify_index_shard(tmp_path, receipt, fingerprint="a" * 64)["full_token_vectors"] == 4


def test_authorization_barriers_and_status_do_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXP109C_ALLOW_PREFLIGHT_GPU", raising=False)
    with pytest.raises(exp.GateRejected, match="explicit authorization"):
        exp.resource_preflight(authorize=False)
    assert exp.status()["namespace"] == exp.NAMESPACE


def test_canonical_label_contract_is_current() -> None:
    answers, stats = exp.canonical_labels()
    assert len(answers) == 7000
    assert stats["evaluable_queries"] == 6991
    assert stats["non_evaluable_queries"] == 9
    assert stats["label_fingerprint"] == exp.LABEL_FINGERPRINT
