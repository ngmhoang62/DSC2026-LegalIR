import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp013_core import fuse_rankings, stable_topk, validate_candidate_record
from exp013_candidates import build_query_memory
from exp013_ranker import _document_spans, build_feature_rows
from exp013_late_interaction import (
    dequantize_rows,
    document_prototypes,
    maxsim,
    quantize_rows,
    retrieve_prototype_documents,
    select_anchor_indices,
)


class Exp013PrimitiveTests(unittest.TestCase):
    def test_anchor_selection_keeps_rare_tokens_and_is_deterministic(self):
        ids = [1, 1, 1, 2, 3, 4]
        selected = select_anchor_indices(ids, [1] * len(ids), maximum=3)
        self.assertEqual(selected.tolist(), [3, 4, 5])

    def test_int8_row_quantisation_has_small_cosine_error(self):
        rng = np.random.default_rng(42)
        original = rng.normal(size=(48, 64)).astype(np.float32)
        packed, scales = quantize_rows(original)
        restored = dequantize_rows(packed, scales)
        cosine = np.sum(original * restored, axis=1) / (
            np.linalg.norm(original, axis=1) * np.linalg.norm(restored, axis=1)
        )
        self.assertGreater(float(cosine.min()), 0.999)

    def test_maxsim_prefers_matching_document(self):
        query = np.array([[1., 0.], [0., 1.]])
        matching = np.array([[1., 0.], [0., 1.]])
        wrong = np.array([[0., 1.], [0., 1.]])
        self.assertGreater(maxsim(query, matching), maxsim(query, wrong))

    def test_prototype_retrieval_stable_ties(self):
        vectors = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], dtype=np.float16)
        rows = retrieve_prototype_documents(np.array([[1., 0.]]), vectors,
            np.array([0, 2, 4]), ["b", "a"], limit=2)
        self.assertEqual([row["doc_id"] for row in rows], ["a", "b"])

    def test_prototypes_bounded(self):
        values = np.eye(30, dtype=np.float32)
        self.assertEqual(document_prototypes(values, maximum=24).shape, (24, 30))

    def test_rrf_and_candidate_contract(self):
        fused = fuse_rankings({"bm25": [{"doc_id": "b", "rank": 1}], "colbert": [{"doc_id": "a", "rank": 1}]})
        self.assertEqual([row["doc_id"] for row in fused], ["a", "b"])
        validate_candidate_record({"qid": "q", "candidates": fused})
        with self.assertRaises(ValueError):
            validate_candidate_record({"qid": "q", "candidates": [{"doc_id": "a", "rank": 2}]})

    def test_query_memory_never_uses_same_fold_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            folds = root / "folds.json"
            train.write_text(json.dumps({
                "q1": {"question": "thủ tục xe máy", "answer": ["d1"]},
                "q2": {"question": "thủ tục xe máy", "answer": ["d2"]},
                "q3": {"question": "thuế thu nhập", "answer": ["d3"]},
            }, ensure_ascii=False), encoding="utf-8")
            folds.write_text(json.dumps({"fold_0": ["q1", "q2"], "fold_1": ["q3"]}), encoding="utf-8")
            output = root / "memory"
            build_query_memory(split="train", train_path=train, query_path=train, folds_path=folds,
                output_dir=output, v3_fingerprint="fixture", neighbors=2, documents=10)
            records = {row["qid"]: row for row in map(json.loads, (output / "train_rankings.jsonl").read_text(encoding="utf-8").splitlines())}
            for qid in ("q1", "q2"):
                neighbors = [neighbor for result in records[qid]["rankings"] for neighbor in result["neighbor_qids"]]
                self.assertNotIn("q1", neighbors)
                self.assertNotIn("q2", neighbors)

    def test_exact_document_spans_and_feature_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passages = root / "passages.jsonl"
            passages.write_text("\n".join(json.dumps(value) for value in [
                {"doc_id": "d1", "token_start": 0, "token_end": 2},
                {"doc_id": "d1", "token_start": 2, "token_end": 4},
                {"doc_id": "d2", "token_start": 4, "token_end": 5},
            ]) + "\n", encoding="utf-8")
            self.assertEqual(_document_spans(passages)["d1"], (0, 4, 2))
            candidates = root / "candidates.jsonl"
            exact = root / "exact.jsonl"
            features = root / "features.jsonl"
            candidates.write_text(json.dumps({"qid": "q", "candidates": [{"doc_id": "d1", "rank": 1, "rrf_score": .1, "best_source_rank": 1, "sources": {}}]}) + "\n", encoding="utf-8")
            exact.write_text(json.dumps({"qid": "q", "query_tokens": 3, "scores": [{"doc_id": "d1", "exact_maxsim": .9, "exact_rank": 1, "document_tokens": 4}]}) + "\n", encoding="utf-8")
            self.assertEqual(build_feature_rows(candidates, exact, features), 1)
            row = json.loads(features.read_text(encoding="utf-8"))
            self.assertEqual(row["exact_maxsim"], .9)
            self.assertEqual(row["bm25_rank"], 999.0)


if __name__ == "__main__":
    unittest.main()
