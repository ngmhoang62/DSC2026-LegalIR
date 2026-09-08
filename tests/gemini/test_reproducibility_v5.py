"""
Test Suite: Reproducibility and Contract Verification for Breakthrough Pipeline V5.
"""
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from gemini.labels import get_canonical_labels
from gemini.metrics import compute_metrics


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_oof_metrics():
    labels, _ = get_canonical_labels()
    oof_path = ROOT / "cache/gemini/nested_cv_v5_oof_preds.json"
    assert oof_path.exists(), f"Missing {oof_path}"

    preds = json.load(open(oof_path, "r", encoding="utf-8"))
    eval_qids = [q for q in preds if labels.get(q)]
    single_qids = [q for q in eval_qids if len(labels[q]) == 1]
    multi_qids = [q for q in eval_qids if len(labels[q]) > 1]

    assert len(eval_qids) == 6991, f"Expected 6991 evaluated queries, got {len(eval_qids)}"

    m = compute_metrics(preds, labels, eval_qids)
    m_s = compute_metrics(preds, labels, single_qids)
    m_m = compute_metrics(preds, labels, multi_qids)

    hits = int(round(m["recall_at_5"] * len(eval_qids)))

    assert hits == 6712, f"Expected 6712 hits, got {hits}"
    assert m["recall_at_5"] > 0.960000, f"Recall@5 {m['recall_at_5']} not > 0.960000"
    assert abs(m["recall_at_5"] - 0.960072) < 1e-5, f"Recall mismatch: {m['recall_at_5']}"
    assert m_m["recall_at_5"] >= 0.807, f"Multi-gold recall regressed: {m_m['recall_at_5']}"


def test_milestone_packages_registry():
    ms_dir = ROOT / "results/gemini/milestone_submissions"
    zips = sorted(list(ms_dir.glob("*.zip")))
    manifests = sorted(list(ms_dir.glob("MANIFEST_*.json")))

    assert len(zips) == 3, f"Expected exactly 3 milestone zips, found {len(zips)}: {[z.name for z in zips]}"
    assert len(manifests) == 3, f"Expected exactly 3 manifests, found {len(manifests)}: {[m.name for m in manifests]}"

    # Verify Milestone 1
    m1_zip = ms_dir / "submission_0.960072_consensus_refined_routing.zip"
    m1_manifest_path = ms_dir / "MANIFEST_0.960072_consensus_refined_routing.json"
    assert m1_zip.exists()
    assert m1_manifest_path.exists()

    m1_manifest = json.load(open(m1_manifest_path, "r", encoding="utf-8"))
    assert m1_manifest["governance"]["uploaded"] is False, "Uploaded flag must be false"
    assert m1_manifest["files"][m1_zip.name]["sha256"] == sha256_of_file(m1_zip)


def test_public_submission_contract():
    ms_dir = ROOT / "results/gemini/milestone_submissions"
    m1_zip = ms_dir / "submission_0.960072_consensus_refined_routing.zip"

    evidence_db = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
    conn = sqlite3.connect(f"file:{evidence_db.as_posix()}?mode=ro", uri=True)
    all_corpus_docs = set(str(row[0]) for row in conn.execute("SELECT doc FROM documents"))
    conn.close()

    zf = zipfile.ZipFile(m1_zip)
    sub_json = json.loads(zf.read("submission.json"))

    assert len(sub_json) == 1000, f"Expected 1000 public queries, got {len(sub_json)}"
    for qid, val in sub_json.items():
        ans = val["answer"]
        assert len(ans) == 5, f"Query {qid} has {len(ans)} answers instead of 5"
        assert len(set(ans)) == 5, f"Query {qid} has duplicate doc IDs: {ans}"
        assert set(ans) <= all_corpus_docs, f"Query {qid} has unknown doc IDs: {set(ans) - all_corpus_docs}"
