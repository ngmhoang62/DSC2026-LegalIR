from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp034_shallow_retrieval import (
    AggregationConfig,
    ResidualProjection,
    RetrievalData,
    _adaptive_gate,
    _exact_records,
    aggregate_hits,
    apply_tier_offset,
    calibrate_tier_offset,
    canonical_answers,
    crossfit_splits,
    lse_pairwise_loss,
    mine_negative_ids,
    required_tier,
    require_label_policy,
    train_projection,
)


class ProjectionTests(unittest.TestCase):
    def test_zero_initialized_projection_is_identity_after_normalization(self):
        model = ResidualProjection(dimension=4, rank=2)
        query = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
        expected = torch.nn.functional.normalize(query, p=2, dim=-1)
        torch.testing.assert_close(model(query), expected)

    def test_multi_positive_loss_gives_every_positive_a_gradient(self):
        positives = torch.tensor([0.7, 0.2], requires_grad=True)
        negatives = torch.tensor([0.6, 0.1])
        loss = lse_pairwise_loss(positives, negatives)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(positives.grad < 0))


class AggregationTests(unittest.TestCase):
    def test_deterministic_tie_and_length_penalty(self):
        hits = [(0, "long", "article", 0.8), (1, "short", "article", 0.8)]
        rows = aggregate_hits(hits, {"long": 99, "short": 1}, AggregationConfig("max", 0.01, 0.0))
        self.assertEqual([row["doc_id"] for row in rows], ["short", "long"])
        tied = aggregate_hits(hits, {"long": 1, "short": 1}, AggregationConfig("max", 0.0, 0.0))
        self.assertEqual([row["doc_id"] for row in tied], ["long", "short"])

    def test_article_bonus_only_uses_best_evidence_kind(self):
        hits = [(0, "a", "context", 0.8), (1, "b", "article", 0.799)]
        rows = aggregate_hits(hits, {"a": 1, "b": 1}, AggregationConfig("max", 0.0, 0.005))
        self.assertEqual(rows[0]["doc_id"], "b")


class LabelAndMiningTests(unittest.TestCase):
    def test_canonical_labels_alias_duplicate_and_drop_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.json").write_text(json.dumps({"q": {"answer": ["old", "blank", "keep"]}}), encoding="utf-8")
            (root / "exclusions.json").write_text(json.dumps([
                {"doc_id": "old", "reasons": ["exact_duplicate_raw_passage"], "duplicate_retained_id": "new"},
                {"doc_id": "blank", "reasons": ["empty_passage"], "duplicate_retained_id": None},
            ]), encoding="utf-8")
            (root / "impact.jsonl").write_text(json.dumps({"query_id": "q", "intentionally_excluded_gold_ids": ["old", "blank"]}) + "\n", encoding="utf-8")
            answers, stats = canonical_answers(root / "train.json", root / "exclusions.json", root / "impact.jsonl")
            self.assertEqual(answers, {"q": {"new", "keep"}})
            self.assertEqual(stats["canonicalized_duplicate_occurrences"], 1)
            self.assertEqual(stats["dropped_empty_occurrences"], 1)

    def test_negative_mining_excludes_all_gold_and_is_deterministic(self):
        kwargs = dict(dense_ranking=["g", "d1", "d2", "d3"], bm25_ranking=["g2", "d2", "b1", "b2", "b3"],
                      gold={"g", "g2"}, all_docs=[f"r{i}" for i in range(20)], seed=2034)
        first = mine_negative_ids(**kwargs); second = mine_negative_ids(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertFalse(set(first) & {"g", "g2"})


class AdaptiveTests(unittest.TestCase):
    def test_crossfit_never_exposes_outer_heldout(self):
        folds = {f"fold_{index}": [f"q{index}"] for index in range(5)}
        for _, train_qids, eval_qids in crossfit_splits(folds, "fold_0"):
            self.assertNotIn("q0", train_qids)
            self.assertNotIn("q0", eval_qids)
            self.assertFalse(set(train_qids) & set(eval_qids))

    def test_required_tier_accounts_for_all_gold(self):
        ranking = [str(index) for index in range(64)]
        self.assertEqual(required_tier(ranking, {"1", "30"}), 32)
        self.assertEqual(required_tier(ranking, {"1", "missing"}), 64)
        with self.assertRaises(ValueError):
            required_tier(ranking, set())

    def test_conservative_offset_satisfies_each_inner_fold(self):
        rankings = [["x", "g"] + [str(i) for i in range(62)], ["x"] * 20 + ["g"] + [str(i) for i in range(43)]]
        offset = calibrate_tier_offset(predicted_indices=[0, 0], rankings=rankings, gold=[{"g"}, {"g"}],
                                       fold_names=["fold_1", "fold_2"], floor=1.0)
        self.assertEqual(offset, 1)
        self.assertEqual(apply_tier_offset([0, 0], offset), [24, 24])

    def test_malformed_offset_fails_closed(self):
        with self.assertRaises(ValueError): apply_tier_offset([0], -1)
        with self.assertRaises(ValueError): apply_tier_offset([0], 6)

    def test_malformed_label_policy_fails_closed(self):
        import exp034_shallow_retrieval as module
        previous = module.LABEL_POLICY
        try:
            module.LABEL_POLICY = "unexpected_policy"
            with self.assertRaises(RuntimeError):
                require_label_policy()
        finally:
            module.LABEL_POLICY = previous

    def test_rejected_inner_gate_cannot_promote(self):
        report = {"status": "REJECTED_INNER_GATE", "inner_gate_pass": False,
                  "retained_recall@selected_k": 1.0, "mean_k": 16.0, "max_k": 16}
        self.assertFalse(_adaptive_gate(report))


class SyntheticRetrievalTests(unittest.TestCase):
    def test_exact_retrieval_is_deterministic(self):
        class FakeData:
            chunk_docs = ["a", "b", "a", "c"]
            chunk_kinds = ["article", "article", "context", "fallback"]
            chunk_counts = {"a": 2, "b": 1, "c": 1}
            def query_vectors(self, qids, device):
                return torch.tensor([[1.0, 0.0] for _ in qids], device=device)
        data = FakeData(); documents = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.0, 1.0]])
        model = ResidualProjection(dimension=2, rank=1)
        config = AggregationConfig("max", 0.0, 0.0)
        import exp034_shallow_retrieval as module
        previous = module.CHUNK_DEPTH
        try:
            module.CHUNK_DEPTH = 4
            first = _exact_records(data=data, qids=["q"], model=model, documents=documents, configs=(config,), device=torch.device("cpu"), limit=3)
            second = _exact_records(data=data, qids=["q"], model=model, documents=documents, configs=(config,), device=torch.device("cpu"), limit=3)
        finally:
            module.CHUNK_DEPTH = previous
        self.assertEqual(first, second)
        self.assertEqual([row["doc_id"] for row in first[config.key][0]["candidates"]], ["a", "b", "c"])

    def test_manifest_binding_mismatch_is_rejected(self):
        data = RetrievalData.__new__(RetrievalData)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data.e5_dir = root / "e5"; data.v3_dir = root / "v3"
            data.e5_dir.mkdir(); data.v3_dir.mkdir()
            (data.e5_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
            (data.v3_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
            data.e5_manifest = {"corpus_fingerprint": "wrong", "dimension": 1024, "chunks": 1, "cache_fingerprint": "e5"}
            data.v3_manifest = {"content_fingerprint": "right"}
            data.query_manifest = {"dimension": 1024, "e5_cache_fingerprint": "e5"}
            data.query_matrix = np.zeros((1, 1024), dtype=np.float32); data.query_ids = ["q"]
            data.chunk_ids = ["c"]; data.chunk_counts = {"d": 1}; data.doc_metadata = {"d": {}}
            data.answers = {"q": {"d"}}; data.fold_for = {"q": "fold_0"}; data.queries = {"q": "question"}
            with self.assertRaises(RuntimeError): data._validate()

    def test_resume_reproduces_checkpoint_predictions_and_metrics_bytes(self):
        class FakeData:
            def __init__(self):
                generator = torch.Generator().manual_seed(17)
                matrix = torch.randn(12, 1024, generator=generator)
                self.documents = torch.nn.functional.normalize(matrix, dim=1)
                self.qids = ["q1", "q2", "q3"]
                self.query_map = {
                    "q1": self.documents[0].clone(), "q2": self.documents[1].clone(),
                    "q3": self.documents[2].clone(),
                }
                self.chunk_docs = [f"d{i}" for i in range(12)]
                self.chunk_kinds = ["article"] * 12
                self.chunk_counts = {doc: 1 for doc in self.chunk_docs}
                self.doc_to_indices = {doc: [index] for index, doc in enumerate(self.chunk_docs)}
                self.all_docs = list(self.chunk_docs)
                self.answers = {qid: {f"d{index}"} for index, qid in enumerate(self.qids)}
                self.answers["q2"] = set()
                self.fold_for = {qid: "fold_0" for qid in self.qids}
                self.e5_manifest = {"cache_fingerprint": "synthetic"}
                self.v3_manifest = {"content_fingerprint": "synthetic-v3"}
                self.label_stats = {"label_fingerprint": "synthetic-labels"}

            def query_vectors(self, qids, device):
                return torch.stack([self.query_map[qid] for qid in qids]).to(device)

            def document_tensor(self, device):
                return self.documents.to(device)

        import exp034_shallow_retrieval as module
        data = FakeData(); bm25 = {qid: list(data.all_docs) for qid in data.qids}
        previous = module.CHUNK_DEPTH
        try:
            module.CHUNK_DEPTH = 12
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                kwargs = dict(data=data, train_qids=["q1", "q2"], eval_qids=["q3"], bm25=bm25,
                              config=AggregationConfig("max", 0.0, 0.0), output_dir=output,
                              device=torch.device("cpu"), smoke=True)
                report = train_projection(**kwargs, resume=False)
                self.assertEqual(report["excluded_non_evaluable_train_qids"], 1)
                first = {name: (output / name).read_bytes() for name in ("checkpoint.pt", "predictions.jsonl", "REPORT.json")}
                train_projection(**kwargs, resume=True)
                second = {name: (output / name).read_bytes() for name in first}
                self.assertEqual(first, second)
        finally:
            module.CHUNK_DEPTH = previous


if __name__ == "__main__":
    unittest.main()
