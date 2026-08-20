from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_structural_chunks_v3 import audit_cache
from structural_chunker_v3 import build_corpus, parse_document


class WhitespaceTokenizer:
    name_or_path = "test-whitespace-tokenizer"

    @staticmethod
    def _offsets(text: str):
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def token_length(self, text: str) -> int:
        return len(self._offsets(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        offsets = self._offsets(text)
        if len(offsets) <= max_tokens:
            return text.strip()
        return text[: offsets[max_tokens - 1][1]].strip() if max_tokens else ""

    def windows(self, text: str, max_tokens: int, overlap: int):
        offsets = self._offsets(text)
        if not offsets:
            return []
        result = []
        start = 0
        stride = max_tokens - overlap
        while start < len(offsets):
            end = min(start + max_tokens, len(offsets))
            result.append((offsets[start][0], offsets[end - 1][1]))
            if end == len(offsets):
                break
            start += stride
        return result


class BoundaryMergeTokenizer(WhitespaceTokenizer):
    """Simulate a tokenizer whose prefix/content concatenation costs +1."""

    def token_length(self, text: str) -> int:
        base = super().token_length(text)
        marker = "[Nội dung]"
        if marker in text and text.split(marker, 1)[1].strip():
            return base + 1
        return base


class StructuralParserTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WhitespaceTokenizer()

    def parse(self, passage: str, **extra):
        document = {"id": 123, "name": "Luật thử nghiệm", "passage": passage}
        return parse_document(
            document,
            self.tokenizer,
            max_tokens=extra.get("max_tokens", 384),
            token_window=extra.get("token_window", 352),
            token_overlap=extra.get("token_overlap", 32),
        )

    def test_standard_hierarchy_and_inline_reference(self):
        passage = (
            "LỜI NÓI ĐẦU\n\nChương I\nQUY ĐỊNH CHUNG\n\nMục 1. PHẠM VI\n\n"
            "Điều 1. Phạm vi điều chỉnh\nNội dung theo Điều 99 của luật khác.\n"
            "1. Khoản thứ nhất.\na) Điểm a.\nb) Điểm b.\n"
            "2. Khoản thứ hai.\n\nĐiều 2\nNội dung điều hai."
        )
        document, nodes, chunks = self.parse(passage)
        kinds = [node["kind"] for node in nodes]
        self.assertEqual(document["parse_mode"], "structured")
        self.assertEqual(kinds.count("article"), 2)
        self.assertIn("chapter", kinds)
        self.assertIn("section", kinds)
        self.assertIn("clause", kinds)
        self.assertIn("point", kinds)
        self.assertEqual(len(document["scope_node_ids"]), 1)
        self.assertFalse(any(node["label"] == "99" for node in nodes))
        for node in nodes:
            if not node["synthetic"]:
                self.assertEqual(passage[node["start"] : node["end"]], node["raw_text"])
        for chunk in chunks:
            self.assertLessEqual(chunk["token_count"], 384)
            self.assertEqual(passage[chunk["start"] : chunk["end"]], chunk["raw_text"])

    def test_repeated_article_labels_have_unique_offset_ids(self):
        passage = "Điều 1. Bản gốc\nNội dung.\n\nĐiều 1. Bản sửa đổi\nNội dung khác."
        _, nodes, _ = self.parse(passage)
        articles = [node for node in nodes if node["kind"] == "article"]
        self.assertEqual([node["label"] for node in articles], ["1", "1"])
        self.assertEqual(len({node["node_id"] for node in articles}), 2)

    def test_long_article_prefers_clauses_then_token_windows(self):
        long_leaf = " ".join(f"từ{i}" for i in range(80))
        passage = (
            "Điều 1. Điều dài\n"
            "1. khoản ngắn một hai ba.\n"
            "2. " + long_leaf + "\n"
            "3. khoản cuối bốn năm sáu."
        )
        _, _, chunks = self.parse(
            passage, max_tokens=24, token_window=16, token_overlap=4
        )
        self.assertGreater(len(chunks), 3)
        self.assertIn("oversized_leaf_window", {chunk["split_reason"] for chunk in chunks})
        self.assertTrue(all(chunk["token_count"] <= 24 for chunk in chunks))
        windows = [chunk for chunk in chunks if chunk["split_reason"] == "oversized_leaf_window"]
        self.assertTrue(any(windows[i + 1]["start"] < windows[i]["end"] for i in range(len(windows) - 1)))

    def test_token_window_accounts_for_non_additive_boundary_cost(self):
        tokenizer = BoundaryMergeTokenizer()
        passage = "Điều 1. Điều dài\n1. " + " ".join(f"từ{i}" for i in range(80))
        _, _, chunks = parse_document(
            {"id": 77, "name": "Luật", "passage": passage},
            tokenizer,
            max_tokens=24,
            token_window=16,
            token_overlap=4,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["token_count"] <= 24 for chunk in chunks))

    def test_fallback_annex_missing_name_and_empty_passage(self):
        passage = "PHỤ LỤC\n\nBảng dữ liệu không có cấu trúc Điều."
        document = {"id": 9, "link": "https://example.test/legal/annex", "passage": passage}
        record, nodes, chunks = parse_document(document, self.tokenizer)
        self.assertEqual(record["parse_mode"], "fallback")
        self.assertTrue(any(node["kind"] == "annex" for node in nodes))
        self.assertTrue(chunks)

        empty_record, empty_nodes, empty_chunks = parse_document(
            {"id": 10, "link": "https://example.test/empty", "passage": ""},
            self.tokenizer,
        )
        self.assertTrue(empty_record["synthetic"])
        self.assertTrue(empty_nodes[0]["synthetic"])
        self.assertTrue(empty_chunks[0]["raw_text"])
        self.assertIsNone(empty_chunks[0]["start"])

    def test_parse_is_deterministic(self):
        passage = "Chương I\nTÊN CHƯƠNG\nĐiều 1. Nội dung\n1. Một khoản."
        first = self.parse(passage)
        second = self.parse(passage)
        self.assertEqual(first, second)


class CorpusAuditIntegrationTests(unittest.TestCase):
    def test_build_twice_and_audit(self):
        tokenizer = WhitespaceTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            docs = [
                {"id": 1, "name": "Luật A", "passage": "Điều 1. Phạm vi\nNội dung A."},
                {"id": 2, "link": "https://x/b", "passage": "Bảng không có điều."},
                {"id": 3, "link": "https://x/c", "passage": ""},
            ]
            for doc in docs:
                (contexts / f"context_{doc['id']}.json").write_text(
                    json.dumps(doc, ensure_ascii=False), encoding="utf-8"
                )
            train = root / "train.json"
            train.write_text(
                json.dumps({"q": {"question": "Câu hỏi", "answer": ["1", "2", "3"]}}),
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            first_manifest = build_corpus(contexts, first, tokenizer, max_tokens=32, token_window=24, token_overlap=4)
            second_manifest = build_corpus(contexts, second, tokenizer, max_tokens=32, token_window=24, token_overlap=4)
            self.assertEqual(first_manifest["content_fingerprint"], second_manifest["content_fingerprint"])
            report = audit_cache(
                first,
                contexts,
                train,
                root / "audit",
                tokenizer,
                expected_documents=3,
                compare_cache=second,
            )
            self.assertEqual(report["status"], "PASS", report["hard_error_samples"])
            self.assertEqual(report["counts"]["ground_truth_documents_represented"], 3)


if __name__ == "__main__":
    unittest.main()
