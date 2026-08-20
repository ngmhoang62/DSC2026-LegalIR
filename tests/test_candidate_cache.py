from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export_candidate_cache import (
    CANDIDATE_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    RankingValidationError,
    export_candidates,
)
from structural_chunker_v3 import canonical_json


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


class CandidateCacheTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        cache = root / "chunks"
        cache.mkdir()
        documents = []
        chunks = []
        for index in range(60):
            doc_id = f"d{index:02d}"
            documents.append({"doc_id": doc_id})
            for part in range(3):
                chunks.append({"chunk_id": f"{doc_id}:c{part}", "doc_id": doc_id})
        write_jsonl(cache / "documents.jsonl", documents)
        write_jsonl(cache / "chunks.jsonl", chunks)
        (cache / "manifest.json").write_text(
            json.dumps({"content_fingerprint": "chunks-fp"}), encoding="utf-8"
        )
        audit = root / "audit.json"
        audit.write_text(
            json.dumps({"status": "PASS", "content_fingerprint": "chunks-fp"}),
            encoding="utf-8",
        )
        ranking_paths = {}
        for source_index, source in enumerate(("bge", "e5", "bm25")):
            path = root / f"{source}.jsonl"
            docs = [
                {"doc_id": f"d{index:02d}", "rank": index + 1, "score": 100.0 - index}
                for index in range(60)
            ]
            ranked_chunks = []
            rank = 1
            for index in range(60):
                doc_id = f"d{index:02d}"
                ranked_chunks.append(
                    {
                        "chunk_id": f"{doc_id}:c{source_index}",
                        "doc_id": doc_id,
                        "rank": rank,
                        "score": 1.0 / rank,
                    }
                )
                rank += 1
            write_jsonl(
                path,
                [
                    {
                        "schema_version": SOURCE_SCHEMA_VERSION,
                        "source": source,
                        "source_fingerprint": f"{source}-fp",
                        "qid": "q1",
                        "query": "Câu hỏi pháp luật",
                        "documents": docs,
                        "chunks": ranked_chunks,
                    }
                ],
            )
            ranking_paths[source] = path
        return cache, audit, ranking_paths

    def test_validate_and_export_top50_with_three_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, audit, ranking_paths = self.make_fixture(root)
            validation = export_candidates(
                chunk_cache_dir=cache,
                audit_report_path=audit,
                ranking_paths=ranking_paths,
                output_dir=root / "output",
                split="train",
                validate_only=True,
            )
            self.assertEqual(validation["counts"]["candidate_pool_min"], 50)
            self.assertEqual(validation["counts"]["evidence_max"], 3)
            self.assertFalse((root / "output").exists())

            manifest = export_candidates(
                chunk_cache_dir=cache,
                audit_report_path=audit,
                ranking_paths=ranking_paths,
                output_dir=root / "output",
                split="train",
            )
            self.assertEqual(manifest["schema_version"], CANDIDATE_SCHEMA_VERSION)
            records = [
                json.loads(line)
                for line in (root / "output" / "train_candidates.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records[0]["candidates"]), 50)
            self.assertEqual(len(records[0]["candidates"][0]["evidence"]), 3)
            self.assertNotIn("answer", records[0])

    def test_audit_and_fingerprint_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, audit, ranking_paths = self.make_fixture(root)
            audit.write_text(
                json.dumps({"status": "FAIL", "content_fingerprint": "chunks-fp"}),
                encoding="utf-8",
            )
            with self.assertRaises(RankingValidationError):
                export_candidates(
                    chunk_cache_dir=cache,
                    audit_report_path=audit,
                    ranking_paths=ranking_paths,
                    output_dir=root / "output",
                    split="train",
                    validate_only=True,
                )
            audit.write_text(
                json.dumps({"status": "PASS", "content_fingerprint": "stale"}),
                encoding="utf-8",
            )
            with self.assertRaises(RankingValidationError):
                export_candidates(
                    chunk_cache_dir=cache,
                    audit_report_path=audit,
                    ranking_paths=ranking_paths,
                    output_dir=root / "output",
                    split="train",
                    validate_only=True,
                )


if __name__ == "__main__":
    unittest.main()
