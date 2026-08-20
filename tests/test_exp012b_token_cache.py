from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp012b_core import atomic_json, canonical_json
from exp012b_token_cache import TokenCacheReader


class TokenCacheTests(unittest.TestCase):
    def test_packed_ids_round_trip_and_qid_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.asarray([1, 2, 3, 7, 8], dtype=np.uint32)
            values.tofile(root / "input_ids.u32")
            rows = [
                {"qid": "q1", "doc_id": "d1", "chunk_id": "c1", "bundle_hash": "h1", "offset": 0, "length": 3},
                {"qid": "q2", "doc_id": "d2", "chunk_id": "c2", "bundle_hash": "h2", "offset": 3, "length": 2},
            ]
            with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(canonical_json(row) + "\n")
            atomic_json(root / "manifest.json", {"counts": {"tokens": 5, "queries": 2}})
            records = list(TokenCacheReader(root).query_records({"q2"}))
            self.assertEqual([row["qid"] for row in records], ["q2"])
            self.assertEqual(records[0]["pairs"][0]["input_ids"].tolist(), [7, 8])


if __name__ == "__main__":
    unittest.main()
