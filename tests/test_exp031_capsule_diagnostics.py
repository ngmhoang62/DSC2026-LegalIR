from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp031_capsule_diagnostics import _query_bucket, _reducers, _recall_precision, _top, fuse_rankings


class CapsuleDiagnosticsTests(unittest.TestCase):
    def test_deterministic_top5_tie_break_and_metrics(self) -> None:
        scores = {str(index): 1.0 for index in range(7)}
        self.assertEqual(_top(scores), ["0", "1", "2", "3", "4"])
        self.assertEqual(_recall_precision(_top(scores), {"1", "6"}), (0.5, 0.2))

    def test_residual_aggregation_never_changes_single_view(self) -> None:
        for name, reducer in _reducers().items():
            self.assertEqual(reducer([2.5]), 2.5, name)

    def test_max_has_variable_view_count_bias_but_penalty_corrects_it(self) -> None:
        reducers = _reducers()
        self.assertGreater(reducers["max"]([0.0, 0.2]), reducers["max"]([0.1]))
        self.assertLess(reducers["penalized_max_0.50"]([0.0, 0.2]), reducers["penalized_max_0.50"]([0.1]))

    def test_query_bucket_uses_vietnamese_accent_folding(self) -> None:
        bucket = _query_bucket("Đối tượng nào áp dụng khoản 2 Điều 5?", {"1"})
        self.assertEqual(bucket["signal"], "subject+article")
        self.assertEqual(bucket["gold_count"], "1")

    def test_rank_fusion_requires_same_membership_and_is_deterministic(self) -> None:
        original = ["a", "b", "c", "d"]
        reranked = ["d", "c", "b", "a"]
        self.assertEqual(fuse_rankings(original, reranked, method="borda", weight=0.0), original)
        self.assertEqual(fuse_rankings(original, reranked, method="borda", weight=1.0), reranked)
        with self.assertRaises(ValueError):
            fuse_rankings(original, ["a", "b", "c", "x"], method="rrf")

if __name__ == "__main__":
    unittest.main()
