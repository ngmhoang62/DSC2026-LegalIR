"""Unit tests for submission integrity in Gemini namespace."""
from __future__ import annotations

import json
import zipfile
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/gemini/submission"


def test_submission_files_exist():
    assert (OUT_DIR / "submission.json").exists()
    assert (OUT_DIR / "submission_recall_first.json").exists()
    assert (OUT_DIR / "submission.zip").exists()
    assert (OUT_DIR / "SUBMISSION_MANIFEST.json").exists()


def test_submission_json_contract():
    data = json.loads((OUT_DIR / "submission.json").read_text(encoding="utf-8"))
    assert len(data) == 1000
    for qid, entry in data.items():
        assert "answer" in entry
        ans = entry["answer"]
        assert len(ans) == 5
        assert len(set(ans)) == 5
        for doc_id in ans:
            assert isinstance(doc_id, str) and doc_id.isdigit()


def test_submission_zip_matches_json():
    zip_path = OUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert zf.namelist() == ["submission.json"]
        zip_content = zf.read("submission.json")
        raw_content = (OUT_DIR / "submission.json").read_bytes()
        assert zip_content == raw_content


def test_submission_manifest_integrity():
    manifest = json.loads((OUT_DIR / "SUBMISSION_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE_SUBMISSION"
    assert manifest["queries"] == 1000
    assert manifest["uploaded"] is False
    assert len(manifest["files"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
