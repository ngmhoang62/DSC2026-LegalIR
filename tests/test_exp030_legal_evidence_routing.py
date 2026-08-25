import sys
import json
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from exp030_legal_evidence_routing import (
    MODEL_BY_KEY,
    VIETNAMESE_MARKERS,
    aggregate_view_scores,
    _job,
    build_candidate_views,
    canonical_answers,
    classify_scope_node,
    configure_hf_modules_cache,
    capsule_gate,
    discover_lora_targets,
    extract_official_title,
    extract_relations,
    format_structural_path,
    qwen_format_instruction,
    paired_bootstrap_delta,
    query_signals,
    read_selected_jsonl,
    score_query_record,
    select_hard_negatives,
    select_views,
    scope_display_span,
    stratified_sample,
    truncate_pair_document,
    training_attempts,
    write_indexed_jsonl,
)


def candidate(doc_id="1", *, title=True, scope=True):
    return {
        "doc_id": doc_id,
        "document_label": "Thong tu 17 2022 TT BGTVT sua doi van tai",
        "official_title": {
            "status": "VERIFIED" if title else "MISSING",
            "display_text": "Sửa đổi quy định về vận tải bằng xe ô tô",
        },
        "typed_scope": ([{
            "kind": "applicable_subjects",
            "raw_text": "Điều 2. Đối tượng áp dụng Doanh nghiệp vận tải.",
        }] if scope else []),
        "header_relations": [{"raw_text": "sửa đổi Thông tư 12/2020/TT-BGTVT"}],
        "evidence": [{
            "raw_text": "Doanh nghiệp phải cấp lệnh vận chuyển cho lái xe.",
            "structural_path": "Chương II > Điều 25 > Khoản 1",
            "relations": [],
        }],
    }


class SimpleTokenizer:
    def __call__(self, *args, **kwargs):
        if len(args) == 2:
            tokens = (str(args[0]) + " " + str(args[1])).split()
        else:
            tokens = str(args[0]).split()
        return {"input_ids": list(range(len(tokens) + (2 if kwargs.get("add_special_tokens", True) else 0)))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{value}" for value in ids)


class Exp030Tests(unittest.TestCase):
    def test_remote_code_cache_is_workspace_writable_and_bound_after_import(self):
        path = configure_hf_modules_cache()
        import transformers.dynamic_module_utils as dynamic_module_utils
        self.assertTrue(path.is_dir())
        self.assertEqual(Path(dynamic_module_utils.HF_MODULES_CACHE).resolve(), path.resolve())
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        self.assertEqual(probe.read_text(encoding="utf-8"), "ok")
        probe.unlink()

    def test_canonical_answers_alias_duplicates_and_drop_only_empty_passages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            exclusions = root / "exclusions.json"
            impact = root / "impact.jsonl"
            train.write_text(json.dumps({
                "duplicate": {"answer": ["old"]},
                "empty": {"answer": ["blank"]},
                "mixed": {"answer": ["blank", "kept"]},
            }), encoding="utf-8")
            exclusions.write_text(json.dumps([
                {"doc_id": "old", "reasons": ["exact_duplicate_raw_passage"], "duplicate_retained_id": "new"},
                {"doc_id": "blank", "reasons": ["empty_passage"], "duplicate_retained_id": None},
            ]), encoding="utf-8")
            impact.write_text("\n".join(json.dumps(row) for row in [
                {"query_id": "duplicate", "intentionally_excluded_gold_ids": ["old"]},
                {"query_id": "empty", "intentionally_excluded_gold_ids": ["blank"]},
                {"query_id": "mixed", "intentionally_excluded_gold_ids": ["blank"]},
            ]) + "\n", encoding="utf-8")
            answers, stats = canonical_answers(train, exclusions, impact)
            self.assertEqual(answers["duplicate"], {"new"})
            self.assertEqual(answers["empty"], set())
            self.assertEqual(answers["mixed"], {"kept"})
            self.assertEqual(stats["canonicalized_duplicate_occurrences"], 1)
            self.assertEqual(stats["dropped_empty_occurrences"], 2)
            self.assertEqual(stats["evaluable_queries"], 2)
            self.assertEqual(stats["non_evaluable_qids"], ["empty"])

    def test_real_canonical_label_accounting(self):
        root = Path(__file__).resolve().parents[1]
        answers, stats = canonical_answers(
            root / "public_test_dataset" / "train.json",
            root / "cache" / "final_preprocessed_v2" / "exclusions.json",
            root / "cache" / "final_preprocessed_v2" / "train_label_impact.jsonl",
        )
        self.assertEqual(stats["queries"], 7000)
        self.assertEqual(stats["evaluable_queries"], 6991)
        self.assertEqual(stats["non_evaluable_queries"], 9)
        self.assertEqual(stats["canonicalized_duplicate_occurrences"], 2)
        self.assertEqual(stats["dropped_empty_occurrences"], 11)
        self.assertEqual(answers["127798"], {"280171"})
        self.assertEqual(answers["130058"], {"280171"})

    def test_title_is_exact_source_span_and_accented(self):
        passage = (
            "BỘ GIAO THÔNG VẬN TẢI\nSố: 17/2022/TT-BGTVT\n\nTHÔNG TƯ\n"
            "SỬA ĐỔI QUY ĐỊNH VỀ VẬN TẢI BẰNG XE Ô TÔ\n\nCăn cứ Luật Giao thông đường bộ..."
        )
        result = extract_official_title(passage, "Thong tu 17 2022 TT BGTVT sua doi quy dinh van tai")
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(passage[result["start"]:result["end"]], result["raw_text"])
        self.assertIn("VẬN TẢI", result["display_text"])

    def test_title_never_diacritizes_slug(self):
        result = extract_official_title("THONG TU\nSUA DOI QUY DINH\nDieu 1. Noi dung", "Thong tu sua doi")
        self.assertEqual(result["status"], "MISSING")

    def test_scope_requires_heading_prefix(self):
        correct = {"heading_text": "Điều 2. Đối tượng áp dụng", "raw_text": "Điều 2. Đối tượng áp dụng\nTổ chức, cá nhân..."}
        false_positive = {"heading_text": "Điều 36. Trách nhiệm thi hành", "raw_text": "Điều 36. Bộ trưởng chịu trách nhiệm trong phạm vi cả nước."}
        combined = {"heading_text": "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng", "raw_text": "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng"}
        self.assertEqual(classify_scope_node(correct), "applicable_subjects")
        self.assertIsNone(classify_scope_node(false_positive))
        self.assertEqual(classify_scope_node(combined), "combined")

    def test_scope_combined_reverse_order_and_subheading_exact_offsets(self):
        combined = {
            "heading_text": "Điều 1. Đối tượng và phạm vi áp dụng",
            "raw_text": "Điều 1. Đối tượng và phạm vi áp dụng\nQuy định chung.",
        }
        raw = (
            "Điều 1. Quy định chung\n"
            "1. Đối tượng áp dụng: tổ chức, cá nhân vận tải.\n"
            "2. Trách nhiệm thực hiện: theo quy định này."
        )
        node = {"heading_text": "Điều 1. Quy định chung", "raw_text": raw}
        self.assertEqual(classify_scope_node(combined), "combined")
        self.assertEqual(classify_scope_node(node), "applicable_subjects")
        span = scope_display_span(node, "applicable_subjects")
        self.assertEqual(raw[span["relative_start"]:span["relative_end"]], span["raw_text"])
        self.assertTrue(span["raw_text"].startswith("1. Đối tượng áp dụng"))
        self.assertNotIn("2. Trách nhiệm", span["raw_text"])

    def test_structural_path_uses_ancestry(self):
        path = format_structural_path([
            {"kind": "document", "heading_text": ""},
            {"kind": "chapter", "heading_text": "Chương II"},
            {"kind": "article", "heading_text": "Điều 25"},
            {"kind": "clause", "label": "1", "heading_text": ""},
        ])
        self.assertEqual(path, "Chương II > Điều 25 > Khoản 1")

    def test_views_use_only_vietnamese_markers(self):
        views = build_candidate_views("Doanh nghiệp nào được áp dụng?", candidate(), identity_variant="both")
        rendered = "\n".join(view["text"] for view in views)
        for marker in VIETNAMESE_MARKERS[:7]:
            if marker in {"[PHẠM VI ĐIỀU CHỈNH]"}:
                continue
            self.assertIn(marker, rendered)
        self.assertNotIn("[ANSWER EVIDENCE]", rendered)
        self.assertNotIn("[STRUCTURAL PATH]", rendered)

    def test_router_keeps_base_and_handles_unknown_query(self):
        views = build_candidate_views("Một câu hỏi chưa có taxonomy", candidate(), identity_variant="both")
        selected = select_views("Một câu hỏi chưa có taxonomy", views)
        self.assertEqual(selected[0]["kind"], "base")
        self.assertGreaterEqual(len(selected), 2)
        self.assertFalse(any(query_signals("Một câu hỏi chưa có taxonomy").values()))

    def test_actual_pair_budget(self):
        text, length = truncate_pair_document(SimpleTokenizer(), "truy vấn pháp luật", " ".join(["nội dung"] * 100), 20)
        self.assertLessEqual(length, 20)
        self.assertTrue(text)

    def test_qwen_uses_card_fields_and_vietnamese_document(self):
        prompt = qwen_format_instruction("Ai được áp dụng?", "[VĂN BẢN] Luật A")
        self.assertIn("<Instruct>:", prompt)
        self.assertIn("<Query>: Ai được áp dụng?", prompt)
        self.assertIn("<Document>: [VĂN BẢN]", prompt)

    def test_jina_listwise_mapping_and_order(self):
        class Jina:
            max_words = 0
            def rerank(self, query, documents):
                self.max_words = max(self.max_words, *(len(document.split()) for document in documents))
                return [
                    {"index": index, "relevance_score": float(len(documents) - index), "document": document}
                    for index, document in reversed(list(enumerate(documents)))
                ]
        model = Jina()
        candidates = [candidate(str(i), scope=False) for i in range(64)]
        candidates[0]["evidence"][0]["raw_text"] = "nội dung " * 1000
        row = {"qid": "q", "query": "Doanh nghiệp vận tải", "candidates": candidates}
        result = score_query_record(
            spec=MODEL_BY_KEY["jina"], model=model, tokenizer=SimpleTokenizer(), row=row, device="cpu",
        )
        self.assertEqual(len(result["scores"]), 64)
        self.assertGreater(result["scores"][0]["score"], result["scores"][-1]["score"])
        self.assertLessEqual(model.max_words, 512)

    def test_hard_negatives_are_unique_and_rotate(self):
        row = {"qid": "q", "query": "q", "candidates": [candidate(str(i), scope=i % 2 == 0) for i in range(12)]}
        first = select_hard_negatives(row, {"0"}, epoch=0)
        second = select_hard_negatives(row, {"0"}, epoch=1)
        ids = [item["candidate"]["doc_id"] for item in first["negatives"]]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertNotEqual(
            [item["candidate"]["doc_id"] for item in first["negatives"]],
            [item["candidate"]["doc_id"] for item in second["negatives"]],
        )

    def test_stratified_sample_is_deterministic(self):
        rows = []
        answers = {}
        for index in range(20):
            qid = str(index)
            rows.append({"qid": qid, "query": "câu hỏi " + ("dài " * index), "candidates": [candidate(str(i), scope=i == 0) for i in range(64)]})
            answers[qid] = {"0"}
        self.assertEqual(stratified_sample(rows, answers, limit=7), stratified_sample(rows, answers, limit=7))
        self.assertEqual(len(stratified_sample(rows, answers, limit=7)), 7)

    def test_empty_retained_gold_is_excluded_from_sampling_and_bootstrap(self):
        rows = [
            {"qid": "empty", "query": "q", "candidates": [candidate(str(i)) for i in range(64)]},
            {"qid": "valid", "query": "q", "candidates": [candidate(str(i)) for i in range(64)]},
        ]
        answers = {"empty": set(), "valid": {"0"}}
        self.assertEqual(stratified_sample(rows, answers, limit=2), {"valid"})
        score_rows = [
            {"qid": qid, "scores": [{"doc_id": str(i), "score": float(64 - i)} for i in range(64)]}
            for qid in ("empty", "valid")
        ]
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            variant = Path(directory) / "variant.jsonl"
            from exp012b_core import write_jsonl
            write_jsonl(baseline, score_rows)
            write_jsonl(variant, score_rows)
            result = paired_bootstrap_delta(baseline, variant, answers, samples=20)
            self.assertEqual(result["queries"], 1)
            self.assertEqual(result["delta_recall@5"], 0.0)

    def test_indexed_jsonl_seeks_only_requested_qids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capsules.jsonl"
            index = Path(directory) / "capsules.index.sqlite"
            rows = [{"qid": str(value), "payload": "có dấu " * (value + 1)} for value in range(5)]
            self.assertEqual(write_indexed_jsonl(path, rows, index), 5)
            selected = list(read_selected_jsonl(path, {"1", "4"}))
            self.assertEqual([row["qid"] for row in selected], ["1", "4"])
            self.assertEqual(selected[1]["payload"], rows[4]["payload"])

    def test_relation_offsets_and_aggregation(self):
        text = "Văn bản này sửa đổi Thông tư 12/2020/TT-BGTVT và nội dung khác."
        relations = extract_relations(text)
        self.assertTrue(relations)
        relation = relations[0]
        self.assertEqual(text[relation["start"]:relation["end"]], relation["raw_text"])
        self.assertEqual(aggregate_view_scores([1.0, 2.0], method="max"), 2.0)

    def test_lora_targets_are_attention_only(self):
        class Model:
            def named_modules(self):
                return [
                    ("encoder.layer.0.attention.self.query", object()),
                    ("encoder.layer.0.attention.self.key", object()),
                    ("encoder.layer.0.attention.self.value", object()),
                    ("classifier.dense", object()),
                ]
        contract = discover_lora_targets(MODEL_BY_KEY["bge_m3"], Model())
        self.assertTrue(contract["eligible"])
        self.assertNotIn("classifier.dense", contract["matched_modules"])

    def test_gte_fused_qkv_lora_targets(self):
        class Model:
            def named_modules(self):
                return [
                    ("new.encoder.layer.0.attention.qkv_proj", object()),
                    ("new.encoder.layer.0.attention.o_proj", object()),
                    ("new.classifier", object()),
                ]
        contract = discover_lora_targets(MODEL_BY_KEY["gte"], Model())
        self.assertTrue(contract["eligible"])
        self.assertEqual(contract["targets"], ["qkv_proj", "o_proj"])

    def test_training_policy_prefers_full_fp32_then_lora(self):
        attempts = training_attempts(
            {"full_finetune_candidate": True, "lora": {"state": "ELIGIBLE"}},
            bitsandbytes=True,
        )
        self.assertEqual(
            attempts,
            [("full", "fp32"), ("full", "fp16"), ("lora", "fp32"), ("lora", "fp16"), ("lora", "qlora")],
        )

    def test_gate_and_scheduler_resume_are_strict(self):
        passing = capsule_gate([{"delta_recall@5": 0.004}] * 4 + [{"delta_recall@5": -0.001}], bounded=True)
        failing = capsule_gate([{"delta_recall@5": 0.01}, {"delta_recall@5": -0.006}], bounded=True)
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            first = _job(root, "a", "fp", lambda: (calls.append(1) or {"ok": True}), resume=False)
            second = _job(root, "a", "fp", lambda: (calls.append(2) or {"ok": False}), resume=True)
            isolated = _job(root, "b", "fp2", lambda: (_ for _ in ()).throw(ValueError("x")), resume=False, isolate="FAILED_MODEL")
            self.assertEqual(first["payload"], second["payload"])
            self.assertEqual(calls, [1])
            self.assertEqual(isolated["state"], "FAILED_MODEL")


if __name__ == "__main__":
    unittest.main()
