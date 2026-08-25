from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp033_in_document_evidence_routing import (
    MARKERS, _audit_token_lengths, explicit_scope_spans, protected_residual, reciprocal_hybrid,
    render_capsule_v2, select_evidence, select_mmr_evidence,
    strict_scope_heading_kind,
)


class Tokenizer:
    def __call__(self, query, document, **kwargs):
        return {"input_ids": [0] * (len(query.split()) + len(document.split()) + 3)}


class Exp033Tests(unittest.TestCase):
    def test_scope_length_audit_suppresses_misleading_model_warning(self):
        class AuditTokenizer:
            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {"length": [546]}
        tokenizer = AuditTokenizer()
        self.assertEqual(_audit_token_lengths(tokenizer, ["long scope"]), [546])
        self.assertFalse(tokenizer.kwargs["verbose"])
        self.assertFalse(tokenizer.kwargs["truncation"])

    def test_scope_heading_is_strict_and_body_wrap_is_not_a_heading(self):
        self.assertEqual(strict_scope_heading_kind("Điều 1. Phạm vi điều chỉnh"), "scope_of_regulation")
        self.assertEqual(strict_scope_heading_kind("Điều 1. Đối tượng áp dụng và phạm vi điều chỉnh"), "combined")
        self.assertEqual(strict_scope_heading_kind("Điều 2. Phạm vi, đối tượng lấy phiếu tín nhiệm"), "combined")
        self.assertIsNone(strict_scope_heading_kind("Điều 6. Tiêu chuẩn, định mức sử dụng"))
        self.assertEqual(strict_scope_heading_kind("Điều 2. Đối tượng áp\r\n\ndụng\nNghị định này áp dụng"), "applicable_subjects")
        self.assertEqual(strict_scope_heading_kind("Điều 1. Phạm vi điều chỉnh\nNội dung có nhắc đối tượng áp dụng"), "scope_of_regulation")
        node = {
            "start": 100,
            "raw_text": "Điều 6. Tiêu chuẩn\nViệc xác định cơ quan thuộc\r\n\nđối tượng áp dụng tiêu chuẩn tại khoản 1.",
        }
        self.assertEqual(explicit_scope_spans(node), [])

    def test_embedded_combined_scope_is_source_exact(self):
        raw = (
            "Điều 3. Tổ chức thực hiện\n\nI. QUY ĐỊNH CHUNG\n\n"
            "1. Phạm vi điều chỉnh: Tiêu chuẩn này quy định A.\n\n"
            "2. Đối tượng áp dụng: Tổ chức, cá nhân B.\n\n"
            "3. Giải thích từ ngữ\n\nNội dung C."
        )
        spans = explicit_scope_spans({"start": 1000, "raw_text": raw})
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["kind"], "combined")
        self.assertEqual(spans[0]["source_start"], 1000 + raw.index("1. Phạm vi"))
        self.assertEqual(raw[spans[0]["relative_start"]:spans[0]["relative_end"]], spans[0]["raw_text"])
        self.assertNotIn("3. Giải thích", spans[0]["raw_text"])

    def test_embedded_heading_without_declarative_content_is_rejected(self):
        toc = (
            "2.1.2. Phạm vi điều chỉnh\n\n"
            "2.1.3. Đối tượng áp dụng\n\n"
            "2.2. Vai trò và đặc điểm của pháp luật hành nghề y dược"
        )
        self.assertEqual(explicit_scope_spans({"start": 0, "raw_text": toc}), [])
        self.assertEqual(explicit_scope_spans({"start": 0, "raw_text": "1. Phạm vi áp dụng\n\n1.2 Phân loại"}), [])

    def test_selector_is_deterministic_and_respects_redundancy(self):
        rows = [{"chunk_id": "b"}, {"chunk_id": "a"}, {"chunk_id": "c"}]
        selected = select_evidence(rows, [1., 1., .5], redundancy={("a", "b"): .95})
        self.assertEqual([row["chunk_id"] for row in selected], ["a", "c"])

    def test_mmr_uses_relevance_head_and_nonredundant_secondary(self):
        rows = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
        vectors = __import__("numpy").array([[1., 0.], [.95, .3122499], [0., 1.]], dtype="float32")
        selected = select_mmr_evidence(rows, [1., .99, .7], vectors, lambda_value=.70)
        self.assertEqual([row["chunk_id"] for row in selected], ["a", "c"])
        self.assertLess(selected[1]["redundancy"], .90)

    def test_hybrid_is_deterministic(self):
        dense = [{"chunk_id": "b"}, {"chunk_id": "a"}]
        sparse = [{"chunk_id": "a"}, {"chunk_id": "c"}]
        self.assertEqual([row["chunk_id"] for row in reciprocal_hybrid(dense, sparse, dense_weight=.5)], ["a", "b", "c"])

    def test_capsule_one_view_vietnamese_answer_first_and_pair_budget(self):
        primary = {"chunk_id": "p", "raw_text": "Điều 2. Nội dung trọng tâm. " * 20}
        rendered = render_capsule_v2(query="Ai được áp dụng", document={"raw_text": "", "document_label": "Thong tu 1"}, primary=primary, secondary=None, ancestry=[{"kind": "article", "heading_text": "Điều 2"}], scope=None, tokenizer=Tokenizer(), max_length=80)
        self.assertTrue(rendered["one_view"])
        self.assertLessEqual(rendered["pair_tokens"], 80)
        self.assertIn("[BẰNG CHỨNG CHÍNH]", rendered["text"])
        self.assertIn("Điều 2.", rendered["text"])
        self.assertTrue(set(marker for marker in MARKERS if marker in rendered["text"]))

    def test_protected_residual_has_fallback_and_exact_top_five(self):
        original = [str(value) for value in range(64)]
        evidence = {value: 0. for value in original}
        anchor = {value: float(64 - int(value)) for value in original}
        unchanged = protected_residual(original, evidence, anchor, alpha=.1, window=16, tau=1.)
        self.assertEqual(unchanged[:5], original[:5])
        evidence["10"] = 100.
        changed = protected_residual(original, evidence, anchor, alpha=.75, window=16, tau=0.)
        self.assertEqual(len(changed[:5]), 5)
        self.assertEqual(set(changed), set(original))


if __name__ == "__main__":
    unittest.main()
