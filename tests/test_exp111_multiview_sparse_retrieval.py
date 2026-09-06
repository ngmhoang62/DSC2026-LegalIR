"""Fast regression tests for EXP-111 primitives and isolation contracts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "src" / "exp111_multiview_sparse_retrieval.py"
SPEC = importlib.util.spec_from_file_location("exp111", MODULE)
assert SPEC and SPEC.loader
exp111 = importlib.util.module_from_spec(SPEC)
sys.modules["exp111"] = exp111
SPEC.loader.exec_module(exp111)


class Exp111UnitTests(unittest.TestCase):
    def test_json_file_reads_are_explicit_utf8(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn(".read_text())", source)

    def test_surface_normalizes_nfc_and_casefold(self) -> None:
        self.assertEqual(exp111.surface("Điều 33/2023-NĐ-CP"), ["điều", "33", "2023", "nđ", "cp"])

    def test_fts_or_is_quoted_and_operator_safe(self) -> None:
        expression = exp111.fts_or(["or", "near", "alpha_1"])
        self.assertEqual(expression, '"or" OR "near" OR "alpha_1"')
        self.assertEqual(exp111.fts_or([]), '"__exp111_no_token__"')

    def test_exact_fts_sequence_preserves_order_and_normalizes_nfd(self) -> None:
        tokens = exp111.exact_fts_token_sequence("Điều 33/2023")
        self.assertEqual(tokens, exp111.exact_fts_token_sequence("Điều 33/2023"))
        self.assertEqual(tokens[1:], ["33", "2023"])

    def test_phrase_df_uses_window_rows_and_real_fts_terms(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "i.sqlite"; conn = sqlite3.connect(path)
            conn.execute("CREATE VIRTUAL TABLE local384 USING fts5(text, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE local384_vocab USING fts5vocab(local384, 'row')")
            for _ in range(10): conn.execute("INSERT INTO local384(text) VALUES(?)", ("điều 33",))
            conn.execute("INSERT INTO local384(text) VALUES(?)", ("điều hiếm",)); conn.commit(); conn.close()
            expression = exp111.phrase_expression(path, "local384", ["Điều", "hiếm"], 2)
            self.assertNotEqual(expression, '"__exp111_no_token__"')
            self.assertIn("hiếm", expression)

    def test_reused_phrase_connection_matches_public_phrase_function(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "i.sqlite"; conn = sqlite3.connect(path)
            conn.execute("CREATE VIRTUAL TABLE local384 USING fts5(text, tokenize='unicode61 tokenchars _')")
            conn.execute("CREATE VIRTUAL TABLE local384_vocab USING fts5vocab(local384, 'row')")
            conn.execute("INSERT INTO local384(text) VALUES(?)", ("điều hiếm",)); conn.commit(); conn.close()
            expected = exp111.phrase_expression(path, "local384", ["Điều", "hiếm"], 2)
            reader = sqlite3.connect(path)
            try:
                with exp111.ExactFtsSession() as tokenizer:
                    actual = exp111.phrase_expression_terms(reader, "local384", tokenizer.sequence("Điều hiếm"), 2)
            finally:
                reader.close()
            self.assertEqual(actual, expected)

    def test_citation_positive_and_negative(self) -> None:
        self.assertEqual(exp111.citation_tokens("Theo Nghị định 33/2023/NĐ-CP."), ["cite_33_2023_nđ_cp"])
        self.assertEqual(exp111.citation_tokens("33/2023 là tỉ lệ, không có cơ quan."), [])

    def test_windows_cover_source_and_offsets(self) -> None:
        text = " ".join(f"w{number}" for number in range(1000))
        records = list(exp111.windows("d", text, 384, 96))
        self.assertEqual(records[0]["word_start"], 0)
        self.assertEqual(records[-1]["word_end"], 1000)
        self.assertTrue(all(left["word_end"] > right["word_start"] for left, right in zip(records, records[1:])))
        self.assertTrue(all(text[row["char_start"]:row["char_end"]] for row in records))
        self.assertEqual(len({row["window_id"] for row in records}), len(records))

    def test_windows_are_deterministic(self) -> None:
        text = "a b c d e f g h"
        self.assertEqual(list(exp111.windows("x", text, 4, 1)), list(exp111.windows("x", text, 4, 1)))

    def test_window_offsets_round_trip_nfd_source(self) -> None:
        text = "Die\u0302u 33/2023/NĐ-CP và quy định"
        record = next(exp111.windows("nfd", text, 8, 1))
        self.assertEqual(record["raw_text_hash"], hashlib.sha256(text[record["char_start"]:record["char_end"]].encode("utf-8")).hexdigest())

    def test_v0_exp021_cascade_uses_fused_head_and_rrf_tail(self) -> None:
        hits = [
            {"doc_id": "a", "parent_node_id": "a1", "chunk_id": "a1", "rank": 1, "score": -4.0},
            {"doc_id": "b", "parent_node_id": "b1", "chunk_id": "b1", "rank": 2, "score": -3.0},
            {"doc_id": "b", "parent_node_id": "b2", "chunk_id": "b2", "rank": 3, "score": -2.0},
        ]
        result = exp111.aggregate_v0_exp021(hits, {"depth": 3, "parent_rrf_k": 3, "fusion_rrf_k": 5, "head_cutoff": 1})
        self.assertEqual([row["doc_id"] for row in result], ["a", "b"])
        self.assertGreater(result[0]["raw_score"], 0.0)

    def test_nonredundant_second_window_does_not_double_count_overlap(self) -> None:
        rows = [
            {"doc_id": "a", "score": -10.0, "unit_rank": 1, "unit_id": "a:0", "word_start": 0, "word_end": 384},
            {"doc_id": "a", "score": -9.0, "unit_rank": 2, "unit_id": "a:1", "word_start": 96, "word_end": 480},
            {"doc_id": "a", "score": -8.0, "unit_rank": 3, "unit_id": "a:2", "word_start": 500, "word_end": 884},
        ]
        result = exp111.aggregate_units(rows, top=5, nonredundant=True, second_lambda=.6)[0]
        self.assertEqual(result["second_score"], 8.0)
        self.assertAlmostEqual(result["raw_score"], 14.8)

    def test_parent_dedup_and_tie_break(self) -> None:
        rows = [
            {"doc_id": "b", "score": -1.0, "unit_rank": 1, "unit_id": "b", "word_start": 0, "word_end": 1},
            {"doc_id": "a", "score": -1.0, "unit_rank": 1, "unit_id": "a", "word_start": 0, "word_end": 1},
        ]
        self.assertEqual(exp111.ranked_docs(exp111.aggregate_units(rows, top=5)), ["a", "b"])

    def test_robust_z_edges(self) -> None:
        self.assertEqual(exp111.robust_z([]), [])
        self.assertEqual(exp111.robust_z([1.0, 1.0]), [0.0, 0.0])
        self.assertEqual(len(exp111.robust_z([1.0, 2.0, 3.0])), 3)

    def test_hierarchical_rrf_prevents_view_multiplicity(self) -> None:
        duplicate = exp111.hierarchical_rrf({"v2_w384": ["a"], "v2_w512": ["a"], "v3_parent": ["b"]})
        self.assertEqual(duplicate[0], "a")
        # A local family gives one max contribution, not two summed votes.
        self.assertAlmostEqual(1 / 21, 1 / 21)

    def test_candidate_union_is_top50_per_source_not_total50(self) -> None:
        sources = {f"s{number}": [f"{number}-{rank}" for rank in range(50)] for number in range(3)}
        union = list(dict.fromkeys(doc for docs in sources.values() for doc in docs[:50]))
        self.assertEqual(len(union), 150)

    def test_candidate_coverage_counts_each_qid_gold_once(self) -> None:
        rankings = {"a": {"q": ["gold"]}, "b": {"q": ["gold"]}}
        coverage, sets = exp111.candidate_coverage(rankings, {"q": {"gold"}}, ["q"], 50)
        self.assertEqual(coverage["candidate_coverage@50"], 1.0)
        self.assertEqual(sets["q"], {"gold"})

    def test_streaming_union_coverage_is_a_set_not_concatenated_ranking(self) -> None:
        # Concatenating v0[:100] then another source loses V0's rank 101--200
        # when measured at 200.  A top-K/source ceiling must retain it.
        source = {"v0_control": [f"d{number}" for number in range(200)], "other": [f"x{number}" for number in range(100)]}
        coverage = exp111.CandidateCoverageSums(); coverage.add(source, {"d150"})
        self.assertEqual(coverage.report()["candidate_coverage@200_per_source"], 1.0)

    def test_choice_oracle_is_not_concatenated_union(self) -> None:
        rankings = {"a": {"q": ["a1", "a2", "a3", "a4", "a5", "gold"]}, "b": {"q": ["gold", "b"]}}
        chosen, provenance = exp111.best_source_oracle(rankings, {"q": {"gold"}}, ["q"])
        self.assertEqual(provenance["q"], "b")
        self.assertEqual(chosen["q"], ["gold", "b"])

    def test_family_weights_change_hierarchical_rrf(self) -> None:
        rankings = {"v1_surface_structural": ["a"], "v3_parent": ["b"]}
        self.assertEqual(exp111.hierarchical_rrf(rankings, family_weights={"structural": .1, "global": .9})[0], "b")

    def test_lambdamart_features_are_label_and_docid_free(self) -> None:
        views = ("v0_control", "v1_surface_structural", "v2_w384", "v2_w512", "v4_bigram", "v4_trigram", "v3_parent", "v5_citation")
        scores = {view: {"q": [{"doc_id": f"doc{rank}", "rank": rank + 1, "raw_score": float(10-rank), "robust_z": 0.0, "matching_units": 1, "score_gap": None} for rank in range(3)]} for view in views}
        docs, rows, names = exp111.sparse_feature_rows(scores, {"q": "Điều 33/2023/NĐ-CP"}, "q")
        self.assertEqual(docs, ["doc0", "doc1", "doc2"])
        self.assertEqual(len(rows[0]), len(names))
        self.assertFalse(any("doc_id" in name or "label" in name or "fold" in name for name in names))

    def test_compact_score_preserves_feature_inputs_and_order(self) -> None:
        source = {view: [{"doc_id": "d", "rank": 1, "raw_score": 2.0, "robust_z": .5, "matching_units": 1, "score_gap": None, "best_unit_id": "discarded"}] for view in exp111.SOURCE_VIEWS}
        compact = exp111.compact_source_record({"qid": "q", "sources": source})
        self.assertEqual(compact["sources"]["v0_control"][0], {"doc_id": "d", "rank": 1, "raw_score": 2.0, "robust_z": .5, "matching_units": 1, "score_gap": None})
        docs, values, names = exp111.sparse_feature_rows_from_sources(compact["sources"], "điều 33")
        self.assertEqual(docs, ["d"])
        self.assertEqual(len(values[0]), len(names))

    def test_metrics_ignore_non_evaluable(self) -> None:
        values = exp111.metric({"a": ["x"], "b": ["x"]}, {"a": {"x"}, "b": set()})
        self.assertEqual(values["evaluable_queries"], 1.0)
        self.assertEqual(values["recall@5"], 1.0)

    def test_sqlite_round_trip_preserves_raw_score(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "i.sqlite"; conn = exp111._db(path)
            conn.execute("CREATE VIRTUAL TABLE units USING fts5(text,unit_id UNINDEXED,doc_id UNINDEXED,node_id UNINDEXED)")
            conn.execute("INSERT INTO units(text,unit_id,doc_id,node_id) VALUES('alpha beta','u','d','n')"); conn.commit(); conn.close()
            rows = exp111.search_fts(path, "units", '"alpha"', 10, "unit_id,doc_id,node_id")
            self.assertEqual(rows[0]["doc_id"], "d")
            self.assertIsInstance(rows[0]["score"], float)

    def test_real_source_round_trip_first_parent(self) -> None:
        doc_id, text, _node = next(exp111.iter_parent_sources())
        record = next(exp111.windows(doc_id, text, 384, 96))
        self.assertEqual(record["raw_text_hash"], hashlib.sha256(text[record["char_start"]:record["char_end"]].encode("utf-8")).hexdigest())
        self.assertGreater(record["char_end"], record["char_start"])

    def test_fold0_is_not_in_inner_label_scope(self) -> None:
        folds, _ = exp111.load_folds()
        inner = {qid for fold in exp111.INNER_FOLDS for qid in folds[fold]}
        self.assertFalse(inner & set(folds["fold_0"]))

    def test_reading_audit_exists_before_code_execution(self) -> None:
        audit = json.loads((exp111.RESULTS / "READING_AUDIT.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["created_before_exp111_code"])
        self.assertEqual(audit["gate"], "PASS_READING_AUDIT")


if __name__ == "__main__":
    unittest.main()
