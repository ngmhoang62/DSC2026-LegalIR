from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp021_e5_dense_candidates import AGGREGATIONS, aggregate_chunk_hits, nested_select


class DenseAggregationTests(unittest.TestCase):
    def test_aggregate_by_parent_and_preserve_top_evidence(self):
        hits = [
            ("c1", "doc-a", 0.9), ("c2", "doc-b", 0.85),
            ("c3", "doc-a", 0.8), ("c4", "doc-a", 0.7),
            ("c5", "doc-b", 0.6),
        ]
        output = aggregate_chunk_hits(hits)
        self.assertEqual(set(output), set(AGGREGATIONS))
        self.assertEqual(output["max"][0]["doc_id"], "doc-a")
        doc_a = next(row for row in output["top2_mean"] if row["doc_id"] == "doc-a")
        self.assertEqual([row["chunk_id"] for row in doc_a["evidence"]], ["c1", "c3", "c4"])
        self.assertAlmostEqual(doc_a["aggregate_score"], 0.85)

    def test_deterministic_doc_tie_break(self):
        output = aggregate_chunk_hits([("a", "doc-z", 0.5), ("b", "doc-a", 0.5)])
        self.assertEqual([row["doc_id"] for row in output["max"]], ["doc-a", "doc-z"])
        self.assertEqual([row["rank"] for row in output["max"]], [1, 2])

    def test_nested_selection_never_uses_heldout_fold(self):
        folds = {"fold_0": ["q0"], "fold_1": ["q1"]}
        answers = {"q0": {"a"}, "q1": {"b"}}
        rankings = {
            "max": {"q0": ["a"], "q1": ["x"]},
            "top2_mean": {"q0": ["x"], "q1": ["b"]},
            "top4_mean": {"q0": ["x"], "q1": ["x"]},
            "logsumexp": {"q0": ["x"], "q1": ["x"]},
        }
        chosen, oof, metrics = nested_select(rankings, answers, folds)
        self.assertEqual(chosen["fold_0"], "top2_mean")
        self.assertEqual(chosen["fold_1"], "max")
        self.assertEqual(oof["q0"], ["x"])
        self.assertEqual(oof["q1"], ["x"])
        self.assertEqual(set(metrics), set(folds))


if __name__ == "__main__":
    unittest.main()
