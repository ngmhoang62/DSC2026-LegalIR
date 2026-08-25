import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from exp026_lambdamart_capsules import FEATURE_BLOCKS, _columns
from exp028_lambdamart_shortlist import FLOOR, K_GRID, PARAM_GRID, _permuted, choose_screen


def screen(feature_set, params, recalls):
    viable = [k for k in K_GRID if all(value >= FLOOR for value in recalls[k])]
    k = viable[0] if viable else 100
    return {"feature_set": feature_set, "params": params, "per_inner": {}, "viable_k": viable[0] if viable else None,
            "diagnostic_k": k, "worst_inner_recall": min(recalls[k]), "mean_inner_recall": sum(recalls[k]) / len(recalls[k])}


class Exp028Tests(unittest.TestCase):
    def test_parameter_grid_is_24(self):
        self.assertEqual(len(PARAM_GRID), 24)

    def test_feature_schema_is_24_columns(self):
        self.assertEqual(len(_columns("all")), 24)
        self.assertEqual(set(FEATURE_BLOCKS), {"retrieval", "provenance", "metadata", "structural"})

    def test_strict_gate_rejects_mean_only_k(self):
        bad = {k: [.98, .98, .98, .98] for k in K_GRID}; bad[32] = [.99, .99, .99, .97]; bad[50] = [.99] * 4; bad[64] = [.99] * 4; bad[80] = [.99] * 4; bad[100] = [.99] * 4
        good = {k: [.98, .98, .98, .98] for k in K_GRID}; good[24] = [.985, .986, .987, .988]; good[32] = [.99] * 4; good[50] = [.99] * 4; good[64] = [.99] * 4; good[80] = [.99] * 4; good[100] = [.99] * 4
        chosen = choose_screen([screen("retrieval", PARAM_GRID[0], bad), screen("all", PARAM_GRID[1], good)])
        self.assertEqual(chosen["feature_set"], "all")
        self.assertEqual(chosen["viable_k"], 24)

    def test_smallest_viable_k_and_deterministic_tie_break(self):
        low = {k: [.98, .98, .98, .98] for k in K_GRID}; low[50] = [.985] * 4; low[64] = [.99] * 4; low[80] = [.99] * 4; low[100] = [.99] * 4
        equal = {k: [.98, .98, .98, .98] for k in K_GRID}; equal[50] = [.985] * 4; equal[64] = [.99] * 4; equal[80] = [.99] * 4; equal[100] = [.99] * 4
        selected = choose_screen([screen("all", PARAM_GRID[1], equal), screen("retrieval", PARAM_GRID[0], low)])
        self.assertEqual(selected["viable_k"], 50)
        self.assertEqual(selected["feature_set"], "retrieval")

    def test_within_query_permutation_is_reproducible(self):
        data = np.arange(24, dtype=np.float32).reshape(6, 4); item = {"start": 1, "end": 5}
        one, two = _permuted(data, item, 2, 2028), _permuted(data, item, 2, 2028)
        self.assertTrue(np.array_equal(one, two))
        self.assertTrue(np.array_equal(one[:, [0, 1, 3]], data[1:5, [0, 1, 3]]))
        self.assertEqual(sorted(one[:, 2]), sorted(data[1:5, 2]))


if __name__ == "__main__":
    unittest.main()
