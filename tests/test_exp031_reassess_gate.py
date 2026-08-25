from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp031_reassess_gate import reassess


class ReassessGateTests(unittest.TestCase):
    def test_neutral_baseline_folds_do_not_make_pass_impossible(self) -> None:
        rows = []
        for fold, variant, delta in ((0, "unaccented_base", 0.0), (1, "multi_view", 0.01), (2, "unaccented_base", 0.0), (3, "multi_view", 0.01), (4, "multi_view", -0.004)):
            rows.append({"model": "bge_m3", "outer": f"fold_{fold}", "selected_variant": variant, "delta_recall@5": delta, "delta_precision@5": 0.0})
        result = reassess({"per_fold": rows})
        self.assertTrue(result["models"]["bge_m3"]["passed"])
        self.assertEqual(result["models"]["bge_m3"]["positive_intervention_folds"], 2)


if __name__ == "__main__":
    unittest.main()
