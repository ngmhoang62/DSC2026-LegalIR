"""Generate and verify public test submission for Gemini research namespace.

Conforms strictly to repository contract:
- 1,000 public test queries.
- Exactly 5 unique document IDs per query.
- Output directory: results/gemini/submission/
- Generated files: submission.json, submission_recall_first.json, submission.zip, SUBMISSION_MANIFEST.json.
- Zero external upload: offline local verification only.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gemini"))

from exp_statutory_kinship import apply_kinship_promotion, load_doc_labels

OUT_DIR = ROOT / "results/gemini/submission"
PUBLIC_PREDS_PATH = ROOT / "results/exp112_task_adaptive_retrieval/public/PREDICTIONS.json"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_submission():
    print("=" * 80)
    print("GENERATING GEMINI SOTA PUBLIC TEST SUBMISSION")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load document IDs and verify universe
    print("Loading document metadata...", flush=True)
    db = sqlite3.connect(f"file:{EVIDENCE_DB}?mode=ro", uri=True)
    doc_ids = set(str(row[0]) for row in db.execute("SELECT doc FROM documents"))
    db.close()
    print(f"Total corpus documents: {len(doc_ids)}", flush=True)

    # 2. Load public test queries
    db_sources = sqlite3.connect(f"file:{SOURCES_DB}?mode=ro", uri=True)
    from gemini.labels import get_canonical_labels
    labels, _ = get_canonical_labels()
    all_qids = set(row[0] for row in db_sources.execute("SELECT DISTINCT q FROM sources"))
    db_sources.close()
    public_qids = sorted(all_qids - set(labels.keys()))
    print(f"Total public test queries: {len(public_qids)}", flush=True)
    assert len(public_qids) == 1000, f"Expected 1000 public queries, got {len(public_qids)}"

    # 3. Load base rankings for public queries
    print("Loading base public predictions...", flush=True)
    raw_public = json.loads(PUBLIC_PREDS_PATH.read_text(encoding="utf-8"))
    assert set(raw_public.keys()) == set(public_qids), "Public query mismatch with PREDICTIONS.json"
    public_rankings = {q: raw_public[q]["order"] for q in public_qids}

    # 4. Apply Gemini Guarded Statutory Kinship Co-Retrieval (SOTA)
    print("Applying Guarded Statutory Kinship Co-Retrieval...", flush=True)
    doc_labels = load_doc_labels(EVIDENCE_DB)
    promoted_rankings, promo_count = apply_kinship_promotion(
        public_rankings, doc_labels, public_qids, top_k=2, cand_max=9
    )
    print(f"Statutory kinship promotions applied on public test: {promo_count} / 1000 queries", flush=True)

    # 5. Format submissions (Top-5 docs per query)
    submission_payload = {}
    recall_first_payload = {}
    for q in public_qids:
        top5 = promoted_rankings[q][:5]
        # Contract validation
        assert len(top5) == 5, f"Query {q} does not have exactly 5 predictions"
        assert len(set(top5)) == 5, f"Query {q} has duplicate predictions: {top5}"
        assert set(top5) <= doc_ids, f"Query {q} contains unknown doc IDs: {set(top5) - doc_ids}"
        submission_payload[q] = {"answer": top5}
        recall_first_payload[q] = {"answer": top5}

    # 6. Write JSON files
    sub_json_path = OUT_DIR / "submission.json"
    rf_json_path = OUT_DIR / "submission_recall_first.json"
    zip_path = OUT_DIR / "submission.zip"
    manifest_path = OUT_DIR / "SUBMISSION_MANIFEST.json"

    print("Writing submission files...", flush=True)
    sub_json_path.write_text(json.dumps(submission_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rf_json_path.write_text(json.dumps(recall_first_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. Create ZIP archive containing only submission.json
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_json_path, arcname="submission.json")

    # 8. Verify ZIP integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert names == ["submission.json"], f"Unexpected zip contents: {names}"
        zip_bytes = zf.read("submission.json")
        disk_bytes = sub_json_path.read_bytes()
        assert zip_bytes == disk_bytes, "Zip extracted bytes mismatch disk bytes"

    # 9. Create and write SUBMISSION_MANIFEST.json
    manifest = {
        "status": "COMPLETE_SUBMISSION",
        "queries": len(public_qids),
        "uploaded": False,
        "files": {
            "submission.json": sha256_file(sub_json_path),
            "submission_recall_first.json": sha256_file(rf_json_path),
            "submission.zip": sha256_file(zip_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("SUBMISSION VERIFICATION SUCCESSFUL")
    print("=" * 80)
    print(f"Output directory: {OUT_DIR}")
    print(f"Queries:          {manifest['queries']}")
    print(f"Status:           {manifest['status']}")
    print(f"Uploaded:         {manifest['uploaded']} (Offline verification strictly enforced)")
    print("File SHA256 Hashes:")
    for fn, h in manifest["files"].items():
        print(f"  {fn:30s} {h}")


if __name__ == "__main__":
    generate_submission()
