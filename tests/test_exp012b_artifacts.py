from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp012b_benchmark import deterministic_sample
from exp012b_core import atomic_json, content_hash, load_v3_manifest, sha256_file
from exp012b_performance import execution_config
from exp012b_retrieval import PersistentChunkReader, build_chunk_offset_index
from exp012b_pipeline import completed_stage


class ArtifactOptimizationTests(unittest.TestCase):
    def test_completed_stage_uses_reference_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fast = root / "fast"
            reference = root / "reference"
            (reference / "chunk_lookup").mkdir(parents=True)
            (reference / "chunk_lookup" / "_SUCCESS.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                completed_stage(fast, reference, "chunk_lookup"),
                reference / "chunk_lookup",
            )

    def make_v3(self, root: Path) -> tuple[Path, Path]:
        v3 = root / "v3"
        v3.mkdir()
        artifacts = {
            "documents.jsonl": '{"doc_id":"d1"}\n',
            "nodes.jsonl": '{"node_id":"n1"}\n',
            "chunks.jsonl": '{"chunk_id":"c1","doc_id":"d1"}\n',
            "doc_to_chunk_ids.json": '{"d1":["c1"]}\n',
        }
        for name, value in artifacts.items():
            (v3 / name).write_text(value, encoding="utf-8")
        manifest = {
            "schema_version": "legalir.structural_chunks.v3",
            "content_fingerprint": "fixture-v3",
            "counts": {"chunks": 1},
            "artifact_sha256": {name: sha256_file(v3 / name) for name in artifacts},
        }
        atomic_json(v3 / "manifest.json", manifest)
        receipt = root / "receipt.json"
        verified = load_v3_manifest(v3, full_verify=True)
        atomic_json(
            receipt,
            {
                "v3_fingerprint": manifest["content_fingerprint"],
                "integrity_receipts": verified["integrity_receipts"],
            },
        )
        return v3, receipt

    def test_receipt_accepts_unchanged_and_detects_reformatted_file(self):
        with tempfile.TemporaryDirectory() as directory:
            v3, receipt = self.make_v3(Path(directory))
            loaded = load_v3_manifest(v3, full_verify=False, receipt_path=receipt)
            self.assertEqual(loaded["content_fingerprint"], "fixture-v3")
            target = v3 / "doc_to_chunk_ids.json"
            target.write_text('{\n  "d1": ["c1"]\n}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stale/corrupt"):
                load_v3_manifest(v3, full_verify=False, receipt_path=receipt)

    def test_profiles_keep_semantics_but_change_execution(self):
        reference = execution_config("reference")
        optimized = execution_config("optimized")
        self.assertEqual(reference.reranker_batch_size, 24)
        self.assertTrue(optimized.merge_lora)
        self.assertEqual(optimized.leaf_backend, "cuda_exact")

    def test_persistent_chunk_reader_reuses_handles_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3, _ = self.make_v3(root)
            lookup = root / "lookup"
            build_chunk_offset_index(v3, lookup)
            with PersistentChunkReader(v3, lookup / "chunk_offsets.sqlite") as reader:
                first = reader.load({"c1"})
                second = reader.load({"c1"})
            self.assertEqual(first, second)
            self.assertEqual(first["c1"]["doc_id"], "d1")


if __name__ == "__main__":
    unittest.main()
