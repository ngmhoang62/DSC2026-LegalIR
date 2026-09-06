from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import exp110p_semantic_label_prototype as core
import exp110p_prepare_colab_bundle as exporter


def test_canonical_duplicate_alias_and_empty_label_policy(tmp_path: Path) -> None:
    train = {
        "q1": {"question": "one", "answer": ["dup", "a"]},
        "q2": {"question": "two", "answer": ["empty"]},
        "q3": {"question": "three", "answer": ["b", "c"]},
    }
    exclusions = [
        {"doc_id": "dup", "duplicate_retained_id": "a", "reasons": ["exact_duplicate_raw_passage"]},
        {"doc_id": "empty", "duplicate_retained_id": None, "reasons": ["empty_passage"]},
    ]
    impact = [
        {"query_id": "q1", "intentionally_excluded_gold_ids": ["dup"]},
        {"query_id": "q2", "intentionally_excluded_gold_ids": ["empty"]},
    ]
    train_path = tmp_path / "train.json"
    exclusions_path = tmp_path / "exclusions.json"
    impact_path = tmp_path / "impact.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    exclusions_path.write_text(json.dumps(exclusions), encoding="utf-8")
    impact_path.write_text(json.dumps(impact), encoding="utf-8")
    answers, stats = core.canonical_labels(train_path, exclusions_path, impact_path)
    assert answers == {"q1": {"a"}, "q2": set(), "q3": {"b", "c"}}
    assert stats["queries"] == 3
    assert stats["evaluable_queries"] == 2
    assert stats["assignment_count"] == 3
    assert stats["canonicalized_duplicate_occurrences"] == 1
    assert stats["dropped_empty_occurrences"] == 1


def test_context_and_support_bank_are_fold_safe() -> None:
    folds = {"fold_0": ["q0"], "fold_1": ["q1", "q2"], "fold_2": ["q3"], "fold_3": ["q4"], "fold_4": ["q5"]}
    assert core.inner_qids(folds) == ["q1", "q2", "q3", "q4", "q5"]
    assert core.support_qids_for_context(folds, heldout="fold_2", target_qid="q1") == ["q2", "q4", "q5"]
    with pytest.raises(ValueError, match="forbidden"):
        core.build_support_bank(["q0"], np.eye(4, dtype=np.float32), {f"q{i}": i for i in range(4)}, {f"q{i}": {"d"} for i in range(4)}, forbidden_qids={"q0"})


def _small_bank() -> tuple[core.SupportBank, np.ndarray]:
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    answers = {"q1": {"a"}, "q2": {"b", "c"}, "q3": {"c"}}
    bank = core.build_support_bank(["q1", "q2", "q3"], embeddings, {"q1": 0, "q2": 1, "q3": 2}, answers, support_fold_names=["fold_1", "fold_2"])
    return bank, embeddings


def test_multigold_soft_vote_centroid_and_missing_sentinel() -> None:
    bank, embeddings = _small_bank()
    features = core.prototype_feature_records(embeddings[0], ["a", "b", "c", "unseen"], bank, soft_policy=(16, 5.0, 0.0))
    assert features["a"]["proto_seen"] == 1.0
    assert features["b"]["proto_seen"] == 1.0
    assert features["c"]["proto_support_count"] == 2.0
    assert features["c"]["proto_soft_vote_raw"] > features["b"]["proto_soft_vote_raw"]
    assert features["unseen"]["proto_seen"] == 0.0
    assert features["unseen"]["proto_similarity_missing"] == 1.0
    assert features["unseen"]["proto_max_similarity"] == core.MISSING_SENTINEL
    assert features["a"]["proto_centroid_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert bank.provenance()["support_fold_names"] == ["fold_1", "fold_2"]
    expected_b = np.exp(5.0 * (float(np.dot(embeddings[0], embeddings[1]) / np.linalg.norm(embeddings[1])) - 1.0)) / 2.0
    assert features["b"]["proto_soft_vote_cardinality_norm"] == pytest.approx(expected_b, rel=1e-6)


def test_cosine_and_stable_tie_and_zscore() -> None:
    values = core.cosine_matrix(np.asarray([[1.0, 0.0]], dtype=np.float16), np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float16))
    assert values.dtype == np.float32
    assert values[0].tolist() == pytest.approx([1.0, 1.0])
    assert core.stable_order([0.5, 0.5], ["b", "a"]) == [1, 0]
    assert np.allclose(core.query_zscore([2.0, 2.0]), [0.0, 0.0])


def test_protected_residual_preserves_base_documents() -> None:
    scores = core.protected_residual_scores({"a": 2.0, "b": 1.0}, {"b": 2.0, "c": 3.0}, alpha=0.1, gate_weights={"a": 0.0, "b": 1.0, "c": 1.0})
    assert set(scores) == {"a", "b", "c"}
    assert scores["a"] != scores["b"]
    assert core.confidence_gate_weight(nearest_similarity=0.2, prototype_winner_margin=0.2, prototype_vote_entropy=0.1, seen_candidate_fraction=1.0, threshold=0.3) == 0.0


def test_similarity_shard_resume_and_corrupt_receipt_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(110)
    vectors = rng.normal(size=(7, 4)).astype(np.float32)
    manifest = core.write_similarity_shards(vectors, [f"q{i}" for i in range(7)], tmp_path, block_size=3)
    assert manifest["fold0_read"] is False
    shard = tmp_path / manifest["shards"][0]["name"]
    receipt = shard.with_suffix(".json")
    loaded = core.load_verified_similarity_shard(shard, receipt, expected_fingerprint=manifest["fingerprint"])
    assert loaded.dtype == np.float32
    receipt.write_text(json.dumps({"fingerprint": manifest["fingerprint"], "sha256": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        core.load_verified_similarity_shard(shard, receipt, expected_fingerprint=manifest["fingerprint"])


def test_bootstrap_and_strict_gate() -> None:
    anchor = {"recall@5": 0.900, "recall@1": 0.700, "mrr@5": 0.800, "precision@5": 0.200, "multi_gold_recall@5": 0.700}
    winner = {"recall@5": 0.906, "recall@1": 0.699, "mrr@5": 0.800, "precision@5": 0.200, "multi_gold_recall@5": 0.701}
    bootstrap = core.deterministic_bootstrap([0.006] * 100, samples=1000, seed=110)
    gate = core.strict_inner_gate(anchor, winner, per_fold_delta={f"fold_{i}": 0.006 for i in range(1, 5)}, bootstrap=bootstrap)
    assert gate["pass"] is True
    assert gate["status"] == "PASS_EXP110P_STRICT_INNER_GATE"
    assert core.deterministic_bootstrap([0.1, -0.1], samples=50, seed=7) == core.deterministic_bootstrap([0.1, -0.1], samples=50, seed=7)
    bad = core.strict_inner_gate(anchor, anchor, per_fold_delta={f"fold_{i}": 0.0 for i in range(1, 5)}, bootstrap={"lower": 0.0})
    assert bad["status"] == "REJECTED_EXP110P_STRICT_INNER_GATE"


def test_nested_selection_tie_prefers_simple_arm() -> None:
    metrics = {name: {"recall@5": 0.9, "multi_gold_recall@5": 0.7, "precision@5": 0.2, "mrr@5": 0.8, "recall@1": 0.6} for name in ("A", "B", "C", "D")}
    assert core.select_nested_arm(metrics) == "A"


def test_nested_residual_grid_comparison_oracle_and_duplicate_audit() -> None:
    configs = {(0.05, 0.50): {"recall@5": 0.9}, (0.10, 0.70): {"recall@5": 0.9, "multi_gold_recall@5": 0.8}}
    assert core.select_nested_residual_config(configs) == (0.10, 0.70)
    answers = {"q1": {"a"}, "q2": {"b"}, "q3": {"a", "b"}}
    anchor = {"q1": ["a"], "q2": ["x"], "q3": ["a"]}
    winner = {"q1": ["x"], "q2": ["b"], "q3": ["b"]}
    comparison = core.prediction_comparison(anchor, winner, answers, ["q1", "q2", "q3"], k=1)
    assert comparison["wins"] == 1 and comparison["losses"] == 1 and comparison["ties"] == 1 and comparison["gold_into_top_k"] == 2
    oracle, meta = core.choice_oracle_predictions({"A": anchor, "C": winner}, answers, ["q1", "q2", "q3"])
    assert oracle["q2"] == ["b"] and meta["label_dependent"] is True
    groups = core.exact_normalized_duplicate_groups(np.asarray([[1, 0], [2, 0], [0, 1]], dtype=np.float32), ["q1", "q2", "q3"])
    assert groups == [["q1", "q2"]]


def test_bounded_lambdamart_configs_stay_near_locked_config() -> None:
    candidates = core.bounded_lambdamart_configs({"num_boost_round": 100, "num_leaves": 16, "seed": 109})
    assert candidates[0]["num_boost_round"] == 100
    assert candidates[0]["num_leaves"] == 16
    assert len(candidates) == len({core.canonical_json(item) for item in candidates})


def test_anchor_feature_schema_matches_exp109b_with_explicit_missing_harrier() -> None:
    source_rows = {
        "sources": {
            "vietlegal_e5": [{"doc_id": "d1", "rank": 1, "score": 0.9}, {"doc_id": "d2", "rank": 2, "score": 0.7}],
            "vnlegal_lal": [{"doc_id": "d2", "rank": 1, "score": 0.8}],
            "bm25": [{"doc_id": "d1", "rank": 1, "score": 3.0}],
        }
    }
    record = core.build_source_feature_record(
        "q1", "d1", source_rows, depth=50, query_token_length=4,
        parent_metadata={"parent_chunk_count": 2, "parent_token_length": 11},
    )
    assert tuple(core.source_feature_names()) == core.BASE_FEATURE_NAMES
    assert len(core.BASE_FEATURE_NAMES) == 41
    core.validate_feature_contract(core.BASE_FEATURE_NAMES)
    assert len(core.AUGMENTED_FEATURE_NAMES) == len(set(core.AUGMENTED_FEATURE_NAMES))
    assert record["harrier_present"] == 0.0
    assert record["harrier_rank"] == 51.0
    assert record["dense_top1_score"] == 0.9
    assert record["dense_top2_score"] == 0.0
    assert core.source_feature_vector(record).shape == (41,)


def test_notebook_is_parseable_and_has_stop_contract() -> None:
    notebook_path = Path(__file__).parents[1] / "notebooks" / "exp110p_semantic_label_prototype_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["cell_type"] == "code"
    assert "drive.mount('/content/drive', force_remount=True)" in "".join(notebook["cells"][0]["source"])
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for token in ("INPUT_MANIFEST.json", "fold0_read", "Fold 0 has not been read", "Explicit user authorization is required", "similarity", "prototype", "reproduce_anchor_predictions", "RESIDUAL_CONFIG_GRID", "support_sidecar"):
        assert token in text
    assert "D:\\Study" not in text
    assert all(cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_exporter_refuses_to_fabricate_missing_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "DEFAULT_ANCHOR_CANDIDATES", ())
    with pytest.raises(RuntimeError, match="refusing to fabricate"):
        exporter.export_bundle(tmp_path)
