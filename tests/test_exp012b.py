from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp012b_bm25 import (
    BM25Searcher,
    aggregate_bm25_documents,
    build_fts5_index,
    safe_fts_query,
    search_bm25,
)
from exp012b_core import atomic_json, canonical_json, sha256_file, stage_run
from exp012b_dense import aggregate_dense_documents, top_indices
from exp012b_reranker import (
    _filtered_evidence_records,
    _score_pairs_with_stats,
    aggregate_ce_documents,
    mine_pairwise_examples,
    pairwise_softplus_loss,
    _score_input_ids_with_stats,
)
from exp012b_retrieval import (
    adaptive_evidence_limit,
    build_chunk_offset_index,
    candidate_union,
    evaluate_rankings,
    select_evidence_ids,
    weighted_rrf,
    load_chunk_records,
)
from exp012b_tuning import nested_tune_stage1, nested_tune_zero_shot
from exp012b_tuning import evaluate_zero_shot_artifacts
from exp012b_training_data import choose_channel_negatives, deterministic_internal_split
from exp012b_lora_eval import evaluate_oof, write_submission


class BM25Tests(unittest.TestCase):
    def test_query_is_escaped_and_uses_or(self):
        expression = safe_fts_query('"đăng_ký" OR xe*')
        self.assertEqual(expression, '"đăng_ký" OR "or" OR "xe"')

    def test_fts5_bm25_order_and_unique_parent_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = root / "fields"
            fields.mkdir()
            rows = [
                (1, "c1", "d1", "p1", "xe máy", "điều 1", "", "đăng ký xe máy"),
                (2, "c3", "d1", "p1", "xe máy", "điều 1", "", "xe máy"),
                (3, "c2", "d2", "p2", "ô tô", "điều 2", "", "đăng ký phương tiện"),
            ]
            path = fields / "bm25_fields.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        canonical_json(
                            {
                                "row_id": row[0], "chunk_id": row[1], "doc_id": row[2],
                                "parent_node_id": row[3], "document_label": row[4],
                                "hierarchy_text": row[5], "scope_text": row[6],
                                "passage_content": row[7],
                            }
                        ) + "\n"
                    )
            atomic_json(
                fields / "manifest.json",
                {"content_fingerprint": "fixture", "artifact_sha256": {path.name: sha256_file(path)}},
            )
            index = root / "index"
            result = build_fts5_index(fields, index, commit_every=1)
            self.assertEqual(result["counts"]["passages"], 3)
            hits = search_bm25(
                index / "bm25_v3.sqlite", "đăng ký xe máy", profile="balanced",
                segmenter=lambda text: text.casefold(), limit=3,
            )
            self.assertEqual(hits[0]["doc_id"], "d1")
            self.assertLessEqual(hits[0]["score"], hits[-1]["score"])
            documents = aggregate_bm25_documents(hits)
            d1 = next(row for row in documents if row["doc_id"] == "d1")
            self.assertEqual(d1["unique_parent_hits"], 1)

            database = index / "bm25_v3.sqlite"
            expression = safe_fts_query("đăng ký xe máy")
            connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
            legacy = connection.execute(
                """SELECT chunk_id, doc_id, parent_node_id,
                          bm25(passages, 1.0, 1.0, 1.0, 1.0) AS score
                   FROM passages WHERE passages MATCH ? AND doc_id = ?
                   ORDER BY score ASC, chunk_id ASC LIMIT 8""",
                (expression, "d1"),
            ).fetchall()
            connection.close()
            with BM25Searcher(
                database, profile="balanced", segmenter=lambda text: text.casefold()
            ) as searcher:
                bounded = searcher.search_document("đăng ký xe máy", "d1", limit=8)
            self.assertEqual([row[0] for row in legacy], [row["chunk_id"] for row in bounded])
            self.assertEqual([row[3] for row in legacy], [row["score"] for row in bounded])


class RetrievalTests(unittest.TestCase):
    def test_disk_chunk_lookup_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3 = root / "v3"
            v3.mkdir()
            chunks_path = v3 / "chunks.jsonl"
            rows = [
                {"chunk_id": "c1", "raw_text": "một"},
                {"chunk_id": "c2", "raw_text": "hai"},
                {"chunk_id": "c3", "raw_text": "ba"},
            ]
            with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(canonical_json(row) + "\n")
            atomic_json(v3 / "manifest.json", {
                "schema_version": "legalir.structural_chunks.v3",
                "content_fingerprint": "fixture",
                "counts": {"chunks": 3},
                "artifact_sha256": {"chunks.jsonl": sha256_file(chunks_path)},
            })
            lookup = root / "lookup"
            result = build_chunk_offset_index(v3, lookup)
            self.assertEqual(result["counts"]["chunks"], 3)
            found = load_chunk_records(
                v3, {"c3", "c1"}, lookup_database=lookup / "chunk_offsets.sqlite"
            )
            self.assertEqual(found["c1"]["raw_text"], "một")
            self.assertEqual(found["c3"]["raw_text"], "ba")

    def test_failed_rerun_removes_stale_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            atomic_json(run_dir / "_SUCCESS.json", {"stage": "old"})
            with self.assertRaises(RuntimeError):
                with stage_run(run_dir, "new"):
                    raise RuntimeError("fixture failure")

    def test_keyboard_interrupt_marks_stage_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                with stage_run(run_dir, "interruptible"):
                    raise KeyboardInterrupt()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "FAILED")
            self.assertIn("KeyboardInterrupt", status["error"])
            self.assertFalse((run_dir / "_SUCCESS.json").exists())
            self.assertEqual(
                json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"],
                "FAILED",
            )

    def test_union_and_weighted_rrf_are_deterministic(self):
        bm25 = [{"doc_id": "a", "rank": 1}, {"doc_id": "b", "rank": 2}]
        dense = [{"doc_id": "c", "rank": 1}, {"doc_id": "a", "rank": 2}]
        self.assertEqual(candidate_union(bm25, dense), {"a", "b", "c"})
        first = weighted_rrf({"bm25": bm25, "bge_leaf": dense}, limit=3)
        second = weighted_rrf({"bm25": bm25, "bge_leaf": dense}, limit=3)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["doc_id"], "a")

    def test_dense_length_penalty_and_top_indices(self):
        hits = [
            {"doc_id": "long", "parent_node_id": "p1", "score": 0.9},
            {"doc_id": "short", "parent_node_id": "p2", "score": 0.9},
        ]
        ranked = aggregate_dense_documents(
            hits, length_penalty=0.01, document_block_counts={"long": 100, "short": 1}
        )
        self.assertEqual(ranked[0]["doc_id"], "short")
        self.assertEqual(top_indices(np.array([0.1, 0.9, 0.3]), 2).tolist(), [1, 2])

    def test_evidence_diversity_and_adaptive_budget(self):
        dense = [
            {"chunk_id": "c1", "parent_node_id": "p1", "score": .9, "rank": 1},
            {"chunk_id": "c2", "parent_node_id": "p1", "score": .8, "rank": 2},
            {"chunk_id": "c3", "parent_node_id": "p2", "score": .7, "rank": 3},
        ]
        lexical = [
            {"chunk_id": "c1", "parent_node_id": "p1", "score": -1, "rank": 1},
            {"chunk_id": "c4", "parent_node_id": "p3", "score": -0.5, "rank": 2},
        ]
        selected = select_evidence_ids(dense, lexical)
        self.assertEqual([row["chunk_id"] for row in selected], ["c1", "c3", "c4"])
        self.assertEqual([adaptive_evidence_limit(rank) for rank in (1, 11, 31)], [3, 2, 1])

    def test_metrics_multi_answer(self):
        predictions = {"q": ["a", "x", "b", "z", "y"]}
        answers = {"q": {"a", "b"}}
        metrics = evaluate_rankings(predictions, answers, ks=(5,))
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["precision@5"], 0.4)


class RerankerTests(unittest.TestCase):
    def test_fold_evidence_offset_cache_preserves_source_order(self):
        class Logger:
            def log(self, *_args, **_kwargs):
                pass

            def status(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for qid in ("q2", "q1", "q3"):
                    handle.write(canonical_json({"qid": qid, "candidates": []}) + "\n")
            records = list(
                _filtered_evidence_records(path, {"q1", "q3"}, "fixture", Logger())
            )
            self.assertEqual([row["qid"] for row in records], ["q1", "q3"])
            self.assertTrue((path.parent / "evidence_qid_offsets.json").exists())

    def test_length_sorted_scoring_restores_input_order_and_caches_queries(self):
        import torch

        class FakeTokenizer:
            def __init__(self):
                self.query_calls = 0

            def __call__(self, first, second=None, **kwargs):
                if isinstance(first, str):
                    self.query_calls += 1
                    return {"input_ids": list(range(1, len(first.split()) + 1))}
                return {
                    "input_ids": [
                        [1] * (len(query.split()) + len(passage.split()) + 3)
                        for query, passage in zip(first, second)
                    ]
                }

            def decode(self, ids, **kwargs):
                return " ".join("q" for _ in ids)

            def pad(self, features, pad_to_multiple_of=8, **kwargs):
                maximum = max(len(row["input_ids"]) for row in features)
                maximum = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
                ids, masks = [], []
                for row in features:
                    length = len(row["input_ids"])
                    ids.append(row["input_ids"] + [0] * (maximum - length))
                    masks.append([1] * length + [0] * (maximum - length))
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

        class FakeModel:
            def __call__(self, attention_mask, **kwargs):
                return type("Output", (), {"logits": attention_mask.sum(1, keepdim=True).float()})

        tokenizer = FakeTokenizer()
        pairs = [
            ("same query", "one two three four"),
            ("same query", "one"),
            ("different", "one two"),
        ]
        scores, actual_batch, stats = _score_pairs_with_stats(
            FakeModel(), tokenizer, pairs, device="cpu", batch_size=2
        )
        self.assertEqual(scores.tolist(), [9.0, 6.0, 6.0])
        self.assertEqual(tokenizer.query_calls, 2)
        self.assertEqual(actual_batch, 2)
        self.assertEqual(stats["strategy"], "length_bucketed_adaptive_v1")

    def test_pretokenized_scoring_matches_reference_order(self):
        import torch

        class FakeTokenizer:
            def pad(self, features, pad_to_multiple_of=8, **kwargs):
                maximum = max(len(row["input_ids"]) for row in features)
                maximum = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
                masks = [
                    [1] * len(row["input_ids"]) + [0] * (maximum - len(row["input_ids"]))
                    for row in features
                ]
                ids = [row["input_ids"] + [0] * (maximum - len(row["input_ids"])) for row in features]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

        class FakeModel:
            def __call__(self, attention_mask, **kwargs):
                return type("Output", (), {"logits": attention_mask.sum(1, keepdim=True).float()})

        # TokenCacheReader exposes NumPy-backed rows; this specifically guards
        # the Transformers tokenizer.pad boundary that rejects numpy.int64.
        rows = [
            np.asarray([1] * 9, dtype=np.int64),
            np.asarray([1] * 3, dtype=np.int64),
            np.asarray([1] * 6, dtype=np.int64),
        ]
        scores, batch, stats = _score_input_ids_with_stats(
            FakeModel(), FakeTokenizer(), rows, device="cpu", batch_size=2
        )
        self.assertEqual(scores.tolist(), [9.0, 3.0, 6.0])
        self.assertEqual(batch, 2)
        self.assertIn("pretokenized", stats["strategy"])

    def test_ce_aggregation(self):
        rows = [
            {"qid": "q", "doc_id": "a", "score": 3.0},
            {"qid": "q", "doc_id": "a", "score": 1.0},
            {"qid": "q", "doc_id": "b", "score": 2.0},
        ]
        ranking = aggregate_ce_documents(rows, gamma=0.2)["q"]
        self.assertEqual(ranking[0]["doc_id"], "a")

    def test_negative_sources_and_no_answer_leakage(self):
        bundle = lambda doc: {"chunk_id": f"{doc}:c", "bundle_text": doc}
        rows = mine_pairwise_examples(
            qid="q", query="query", answers={"gold"},
            positive_bundles={"gold": bundle("gold")},
            channel_rankings={
                "bm25": [{"doc_id": "gold"}, {"doc_id": "bm"}],
                "bge": [{"doc_id": "dense"}],
                "hybrid": [{"doc_id": "hybrid"}],
            },
            best_bundle_by_doc={doc: bundle(doc) for doc in ("bm", "dense", "hybrid")},
        )
        self.assertEqual({row["negative_source"] for row in rows}, {"bm25", "bge", "hybrid"})
        self.assertFalse(any(row["negative_doc_id"] == "gold" for row in rows))
        self.assertTrue(all(row["query"] == "query" for row in rows))

    def test_pairwise_loss_prefers_positive_margin(self):
        import torch

        good = pairwise_softplus_loss(torch.tensor([2.0]), torch.tensor([0.0]))
        bad = pairwise_softplus_loss(torch.tensor([0.0]), torch.tensor([2.0]))
        self.assertLess(float(good), float(bad))


class FoldSafeTuningTests(unittest.TestCase):
    def test_zero_shot_report_requires_stage1_non_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            scores = root / "scores.jsonl"
            train = root / "train.json"
            folds = root / "folds.json"
            output = root / "metrics.json"
            candidates.write_text(
                canonical_json({
                    "qid": "q", "candidates": [
                        {"doc_id": "gold", "rank": 1},
                        {"doc_id": "other", "rank": 2},
                    ],
                }) + "\n", encoding="utf-8"
            )
            scores.write_text(
                canonical_json({"qid": "q", "doc_id": "gold", "score": 2.0}) + "\n" +
                canonical_json({"qid": "q", "doc_id": "other", "score": 1.0}) + "\n",
                encoding="utf-8",
            )
            atomic_json(train, {"q": {"question": "query", "answer": ["gold"]}})
            atomic_json(folds, {"fold_0": ["q"]})
            evaluate_zero_shot_artifacts(
                candidates, scores, train, folds, output,
                stage1_metrics={"recall@5": 1.1, "precision@5": 0.3},
            )
            gate = json.loads(output.read_text(encoding="utf-8"))["promotion_gate"]
            self.assertTrue(gate["recall_floor_pass"])
            self.assertFalse(gate["stage1_recall_non_regression_pass"])
            self.assertFalse(gate["overall_pass"])

    def test_stratified_internal_split_is_deterministic(self):
        answers = {
            **{f"q{i}": {"a"} for i in range(20)},
            **{f"m{i}": {"a", "b"} for i in range(10)},
        }
        first = deterministic_internal_split(sorted(answers), answers)
        second = deterministic_internal_split(sorted(answers), answers)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(first[0]) | set(first[1]), set(answers))

    def test_channel_negatives_prefer_complementarity(self):
        rankings = {
            "bm25": [{"doc_id": "gold"}, {"doc_id": "lex"}, {"doc_id": "shared"}],
            "bge_block": [{"doc_id": "dense"}],
            "bge_leaf": [{"doc_id": "gold"}, {"doc_id": "dense"}, {"doc_id": "shared"}],
        }
        negatives = choose_channel_negatives(rankings, {"gold"})
        self.assertIn(("bm25", "lex"), negatives)
        self.assertIn(("bge", "dense"), negatives)

    def test_stage1_nested_tuning_returns_oof_only(self):
        sources = {
            "q1": {
                "bm25": [{"doc_id": "a", "rank": 1}],
                "bge_block": [{"doc_id": "x", "rank": 1}],
                "bge_leaf": [{"doc_id": "a", "rank": 1}],
            },
            "q2": {
                "bm25": [{"doc_id": "y", "rank": 1}],
                "bge_block": [{"doc_id": "b", "rank": 1}],
                "bge_leaf": [{"doc_id": "b", "rank": 1}],
            },
        }
        result = nested_tune_stage1(
            sources,
            {"q1": {"a"}, "q2": {"b"}},
            {"fold_0": ["q1"], "fold_1": ["q2"]},
            weight_grid=[
                {"bm25": 1.0, "bge_block": 1.0, "bge_leaf": 1.0},
                {"bm25": 0.5, "bge_block": 1.0, "bge_leaf": 2.0},
            ],
        )
        self.assertEqual(set(result["predictions"]), {"q1", "q2"})
        self.assertEqual(result["metrics"]["recall@5"], 1.0)

    def test_submission_is_blocked_without_oof_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "oof.json"
            report.write_text(json.dumps({"promotion_gate": False}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                write_submission({"q": ["a"]}, root / "submission", promotion_report=report)

    def test_oof_gate_compares_against_zero_shot_not_legacy_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold = root / "fold.json"
            train = root / "train.json"
            baseline = root / "zero.json"
            output = root / "oof.json"
            atomic_json(
                fold,
                {
                    "fold": 0,
                    "metrics": {"recall@5": 1.0, "precision@5": 0.2},
                    "predictions": {"q": ["gold", "x", "y", "z", "w"]},
                },
            )
            atomic_json(train, {"q": {"question": "query", "answer": ["gold"]}})
            atomic_json(baseline, {"metrics": {"recall@5": 0.996, "precision@5": 0.2}})
            report = evaluate_oof(
                [fold], train, output, zero_shot_metrics_path=baseline
            )
            self.assertFalse(report["promotion_gate"]["recall_gain_pass"])
            self.assertFalse(report["promotion_gate"]["overall_pass"])

    def test_zero_shot_nested_tuning(self):
        stage1 = {
            "q1": [{"doc_id": "a", "rank": 1}, {"doc_id": "x", "rank": 2}],
            "q2": [{"doc_id": "b", "rank": 1}, {"doc_id": "y", "rank": 2}],
        }
        scores = [
            {"qid": "q1", "doc_id": "a", "score": 2.0},
            {"qid": "q1", "doc_id": "x", "score": 0.0},
            {"qid": "q2", "doc_id": "b", "score": 2.0},
            {"qid": "q2", "doc_id": "y", "score": 0.0},
        ]
        result = nested_tune_zero_shot(
            stage1,
            scores,
            {"q1": {"a"}, "q2": {"b"}},
            {"fold_0": ["q1"], "fold_1": ["q2"]},
        )
        self.assertEqual(result["metrics"]["recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
