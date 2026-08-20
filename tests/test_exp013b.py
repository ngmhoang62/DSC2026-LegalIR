import json
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp013b_candidates import build_candidates, build_query_memory
from exp013b_capsules import QWEN_PREFIX, QWEN_SUFFIX, _render, _select_ids
from exp013b_core import QWEN_INSTRUCTION, rank_ids
from exp013b_fusion import _ambiguity, _fuse
from exp013b_qwen import _score


class Exp013bTests(unittest.TestCase):
    class DummyTokenizer:
        def __call__(self, first, second=None, **_kwargs):
            text = str(first) if second is None else str(first) + " " + str(second)
            return {"input_ids": list(range(len(text.split())))}

        def decode(self, ids, **_kwargs):
            return " ".join(f"t{index}" for index in range(len(ids)))

    class DummyQwenTokenizer:
        def encode(self, text, **_kwargs):
            return [1] * max(1, len(str(text).split()))

        def __call__(self, values, **_kwargs):
            if isinstance(values, str): values = [values]
            return {"input_ids": [[1] * max(1, len(str(value).split())) for value in values]}

        def pad(self, rows, **_kwargs):
            import torch
            maximum = max(len(row["input_ids"]) for row in rows)
            return {
                "input_ids": torch.tensor([[0] * (maximum - len(row["input_ids"])) + row["input_ids"] for row in rows]),
                "attention_mask": torch.tensor([[0] * (maximum - len(row["input_ids"])) + [1] * len(row["input_ids"]) for row in rows]),
            }

    class DummyQwenModel:
        def __call__(self, input_ids, attention_mask):
            import torch
            logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            values = input_ids.sum(1).float()
            logits[:, -1, 2] = values
            logits[:, -1, 3] = -values
            return SimpleNamespace(logits=logits)

    def test_stable_rank_tie_break(self):
        self.assertEqual(rank_ids({"z": 1.0, "a": 1.0}), ["a", "z"])

    def test_query_memory_blocks_entire_heldout_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); train = root / "train.json"; folds = root / "folds.json"
            train.write_text(json.dumps({
                "q1": {"question": "thu tuc xe", "answer": ["d1"]},
                "q2": {"question": "thu tuc xe", "answer": ["d2"]},
                "q3": {"question": "thue", "answer": ["d3"]},
            }), encoding="utf-8")
            folds.write_text(json.dumps({"fold_0": ["q1", "q2"], "fold_1": ["q3"]}), encoding="utf-8")
            output = root / "memory"
            build_query_memory(split="train", train_path=train, query_path=train, folds_path=folds, output_dir=output, v3_fingerprint="v3")
            records = [json.loads(line) for line in (output / "train_memory.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in records:
                if row["qid"] in {"q1", "q2"}:
                    found = {qid for rank in row["rankings"] for qid in rank["neighbor_qids"]}
                    self.assertFalse(found & {"q1", "q2"})

    def test_candidate_union_uses_all_three_channels_and_stable_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); v3 = root / "v3"; v3.mkdir()
            (v3 / "documents.jsonl").write_text("\n".join(json.dumps({"doc_id": doc, "parse_mode": "structured"}) for doc in ("a", "b", "c")) + "\n", encoding="utf-8")
            (v3 / "doc_to_chunk_ids.json").write_text(json.dumps({"a": [], "b": [], "c": []}), encoding="utf-8")
            (v3 / "chunks.jsonl").write_text("", encoding="utf-8")
            ranking = root / "rankings.jsonl"; memory = root / "memory.jsonl"
            ranking.write_text(json.dumps({"qid": "q", "query": "x", "v3_fingerprint": "v3", "rankings": {"bge_leaf": [{"doc_id": "b", "rank": 1}], "bm25": [{"doc_id": "a", "rank": 1}]}}) + "\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"v3_fingerprint": "v3", "artifact_sha256": {"rankings.jsonl": hashlib.sha256(ranking.read_bytes()).hexdigest()}}), encoding="utf-8")
            memory.write_text(json.dumps({"qid": "q", "rankings": [{"doc_id": "c", "rank": 1, "score": 1.0, "neighbor_qids": []}]}) + "\n", encoding="utf-8")
            output = root / "out"; build_candidates(split="train", rankings_path=ranking, memory_path=memory, v3_dir=v3, output_dir=output, v3_fingerprint="v3")
            row = json.loads((output / "train_candidates.jsonl").read_text(encoding="utf-8")); self.assertEqual({item["doc_id"] for item in row["candidates"]}, {"a", "b", "c"})

    def test_rrf_fusion_is_deterministic(self):
        candidates = {"a": {"bge_rank": 1}, "b": {"bge_rank": 2}}
        lam = [{"doc_id": "a", "rank": 1, "lambda_score": 2.}, {"doc_id": "b", "rank": 2, "lambda_score": 1.}]
        qwen = [{"doc_id": "b", "qwen_score": 2.}, {"doc_id": "a", "qwen_score": 1.}]
        self.assertEqual(_fuse(lam, qwen, candidates), _fuse(lam, qwen, candidates))

    def test_capsule_prefers_bm25_from_different_parent(self):
        candidate = {"doc_id": "d", "sources": {}}
        source = {"evidence_by_doc": {"d": {"evidence": [
            {"chunk_id": "bge-1", "parent_node_id": "p1", "provenance": "bge_primary"},
            {"chunk_id": "bge-2", "parent_node_id": "p2", "provenance": "bge_structural_complement"},
            {"chunk_id": "bm25-1", "parent_node_id": "p3", "provenance": "bm25_lexical_complement"},
        ]}}}
        self.assertEqual(_select_ids(candidate, source, {}), [("bge-1", "p1"), ("bm25-1", "p3")])

    def test_ambiguity_is_deterministic(self):
        lam = [{"doc_id": str(index), "rank": index + 1, "lambda_score": 10. - index} for index in range(8)]
        candidates = {str(index): {"bge_rank": index + 1, "channel_count": 2} for index in range(8)}
        self.assertEqual(_ambiguity(lam, candidates), _ambiguity(lam, candidates))

    def test_capsule_pair_respects_qwen_budget(self):
        tokenizer = self.DummyTokenizer()
        chunk = {"retrieval_text": "[Chương] I\n[Điều] 1", "raw_text": " ".join(["noi-dung"] * 1200)}
        query, document = _render(tokenizer, " ".join(["query"] * 300), "van-ban", [chunk], max_length=768, scope_text="pham vi")
        rendered = QWEN_PREFIX + f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}" + QWEN_SUFFIX
        self.assertLessEqual(len(tokenizer(rendered)["input_ids"]), 768)
        self.assertLessEqual(len(tokenizer(query)["input_ids"]), 128)

    def test_qwen_batch_and_single_scores_match(self):
        tokenizer, model = self.DummyQwenTokenizer(), self.DummyQwenModel()
        pairs = [("q one", "doc one"), ("q two", "doc two longer")]
        batched = _score(model, tokenizer, 2, 3, pairs, max_length=128, device="cpu")
        singles = [_score(model, tokenizer, 2, 3, [pair], max_length=128, device="cpu")[0] for pair in pairs]
        self.assertEqual(batched, singles)


if __name__ == "__main__":
    unittest.main()
