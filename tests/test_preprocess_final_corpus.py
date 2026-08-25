from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_final_corpus import (
    build_preprocessed_corpus,
    derive_name_from_link,
    normalize_retrieval_name,
)


class FinalPreprocessingTests(unittest.TestCase):
    def test_url_name_derivation_and_normalization(self):
        self.assertEqual(derive_name_from_link("https://x/a/Van-ban--01.aspx?x=1"), "Van-ban--01")
        self.assertEqual(normalize_retrieval_name("Van-ban--01   "), "Van ban 01")

    def test_empty_and_duplicate_policy_and_label_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            docs = [
                {"id": 1, "link": "https://x/Luat-A.aspx", "passage": "Nội dung"},
                {"id": 2, "name": "Luật-B", "link": "https://x/b", "passage": ""},
                {"id": 121575, "name": "Trùng", "link": "https://x/c", "passage": "Nội dung trùng"},
                {"id": 84226, "name": "Giữ", "link": "https://x/d", "passage": "Nội dung trùng"},
            ]
            for doc in docs:
                (raw / f"context_{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            train = root / "train.json"
            train.write_text(json.dumps({"q": {"question": "?", "answer": ["1", "2", "121575"]}}), encoding="utf-8")
            manifest = build_preprocessed_corpus(raw, train, root / "out")
            self.assertEqual(manifest["retained_context_count"], 2)
            self.assertEqual(manifest["raw_missing_name_count"], 1)
            self.assertEqual(manifest["retained_derived_name_count"], 1)
            self.assertEqual(manifest["train_label_impact"]["gold_id_occurrences"], 2)
            record = json.loads((root / "out" / "contexts" / "context_1.json").read_text(encoding="utf-8"))
            self.assertEqual(record["derived_name"], "Luat-A")
            self.assertEqual(record["retrieval_name"], "Luat A")
            self.assertFalse((root / "out" / "contexts" / "context_2.json").exists())
            self.assertTrue((root / "out" / "_SUCCESS.json").exists())


if __name__ == "__main__":
    unittest.main()
