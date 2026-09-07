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

from gemini.kinship import (
    apply_kinship_promotion,
    apply_inverse_kinship_promotion,
    apply_multi_statute_promotion,
    apply_deep_statutory_kinship,
    apply_guarded_inverse_law,
    apply_topic_law_promotion,
    apply_preamble_citation_kinship,
    apply_hierarchical_midrank_inverse_kinship,
    apply_technical_standard_kinship,
    apply_superseded_statute_dedup,
    apply_operational_insurance_kinship,
    apply_corporate_entity_kinship,
    apply_targeted_statutory_kinship,
    load_doc_labels,
)
from gemini.labels import get_canonical_labels

OUT_DIR = ROOT / "results/gemini/submission"
PUBLIC_PREDS_PATH = ROOT / "results/exp112_task_adaptive_retrieval/public/PREDICTIONS.json"
PUBLIC_DATA_PATH = ROOT / "public_test_dataset/public-official.json"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"


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

    # 4. Apply Gemini Guarded Statutory Kinship & Multi-Statute Co-Retrieval (SOTA)
    print("Applying Guarded Statutory Kinship Co-Retrieval...", flush=True)
    doc_labels = load_doc_labels(EVIDENCE_DB)
    public_data = json.loads(PUBLIC_DATA_PATH.read_text(encoding="utf-8"))
    public_questions = {q: row["question"] for q, row in public_data.items()}

    kinship_rankings, promo_count = apply_kinship_promotion(
        public_rankings, doc_labels, public_qids, top_k=2, cand_max=9
    )
    print(f"Forward statutory kinship promotions applied on public test: {promo_count} / 1000 queries", flush=True)

    inv_rankings, inv_count = apply_inverse_kinship_promotion(
        kinship_rankings, doc_labels, public_qids, top_k=2, cand_max=9
    )
    print(f"Inverse statutory kinship promotions applied on public test: {inv_count} / 1000 queries", flush=True)

    final_rankings, ms_count = apply_multi_statute_promotion(
        inv_rankings, doc_labels, public_questions, public_qids
    )
    print(f"Multi-statute promotions applied on public test: {ms_count} / 1000 queries", flush=True)

    deep_rankings, deep_count = apply_deep_statutory_kinship(
        final_rankings, doc_labels, public_qids, top_k=2, cand_max=15
    )
    print(f"Deep statutory kinship promotions applied on public test: {deep_count} / 1000 queries", flush=True)

    inv_law_rankings, inv_law_count = apply_guarded_inverse_law(
        deep_rankings, doc_labels, public_qids, top_k=2, cand_max=12
    )
    print(f"Guarded inverse law promotions applied on public test: {inv_law_count} / 1000 queries", flush=True)

    topic_law_rankings, topic_count = apply_topic_law_promotion(
        inv_law_rankings, doc_labels, public_questions, public_qids, cand_max=6
    )
    print(f"Topic law promotions applied on public test: {topic_count} / 1000 queries", flush=True)

    doc_preambles = json.loads(DOC_PREAMBLES_PATH.read_text(encoding="utf-8")) if DOC_PREAMBLES_PATH.exists() else {}
    preamble_rankings, preamble_count = apply_preamble_citation_kinship(
        topic_law_rankings, doc_labels, doc_preambles, public_qids, top_k=2, cand_max=8
    )
    print(f"Preamble citation promotions applied on public test: {preamble_count} / 1000 queries", flush=True)

    mid_inv_rankings, mid_inv_count = apply_hierarchical_midrank_inverse_kinship(
        preamble_rankings, doc_labels, public_qids, cand_max=12
    )
    print(f"Mid-rank inverse kinship promotions applied on public test: {mid_inv_count} / 1000 queries", flush=True)

    tech_rankings, tech_count = apply_technical_standard_kinship(
        mid_inv_rankings, doc_labels, public_qids, cand_max=10
    )
    print(f"Technical standard promotions applied on public test: {tech_count} / 1000 queries", flush=True)

    dedup_rankings, dedup_count = apply_superseded_statute_dedup(
        tech_rankings, public_qids
    )
    print(f"Superseded statute de-duplications applied on public test: {dedup_count} / 1000 queries", flush=True)

    insurance_rankings, insurance_count = apply_operational_insurance_kinship(
        dedup_rankings, doc_labels, public_questions, public_qids
    )
    print(f"Operational insurance promotions applied on public test: {insurance_count} / 1000 queries", flush=True)

    corp_rankings, corp_count = apply_corporate_entity_kinship(
        insurance_rankings, doc_labels, public_questions, public_qids
    )
    print(f"Corporate entity promotions applied on public test: {corp_count} / 1000 queries", flush=True)

    final_rankings, targeted_count = apply_targeted_statutory_kinship(
        corp_rankings, doc_labels, public_questions, public_qids
    )
    print(f"Targeted statutory kinship promotions applied on public test: {targeted_count} / 1000 queries", flush=True)

    # 5. Format submissions (Top-5 docs per query)
    submission_payload = {}
    recall_first_payload = {}
    for q in public_qids:
        top5 = final_rankings[q][:5]
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
        "forward_kinship_promotions_applied": promo_count,
        "inverse_kinship_promotions_applied": inv_count,
        "multi_statute_promotions_applied": ms_count,
        "deep_kinship_promotions_applied": deep_count,
        "guarded_inverse_law_promotions_applied": inv_law_count,
        "topic_law_promotions_applied": topic_count,
        "preamble_citation_promotions_applied": preamble_count,
        "midrank_inverse_promotions_applied": mid_inv_count,
        "technical_standard_promotions_applied": tech_count,
        "contract": "canonical_duplicate_alias_drop_empty_passage_v1",
        "offline_verification": {
            "query_count_valid": len(public_qids) == 1000,
            "predictions_per_query": 5,
            "unique_per_query": True,
            "all_docs_in_corpus": True,
            "zip_valid": True,
        },
        "files": {
            "submission.json": {
                "size_bytes": sub_json_path.stat().st_size,
                "sha256": sha256_file(sub_json_path),
            },
            "submission_recall_first.json": {
                "size_bytes": rf_json_path.stat().st_size,
                "sha256": sha256_file(rf_json_path),
            },
            "submission.zip": {
                "size_bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSubmission generation and offline verification complete!")
    print(f"Manifest written to {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    generate_submission()
