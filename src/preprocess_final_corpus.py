"""Deterministic, non-destructive preprocessing for the final LegalIR corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from structural_chunker_v3 import canonical_json, corpus_fingerprint, sha256_file


SCHEMA_VERSION = "legalir.final_preprocessed.v1"
EMPTY_REASON = "empty_passage"
DUPLICATE_RETAINED = {"84226", "206810", "280171", "277743"}
DUPLICATE_EXCLUDED = {
    "121575": "84226",
    "158189": "206810",
    "184972": "206810",
    "254937": "280171",
    "35337": "277743",
}


def derive_name_from_link(link: str) -> str:
    """Derive a name from a URL final path segment without OS-path semantics."""
    path = urlparse(link).path.rstrip("/")
    final_segment = unquote(path.rsplit("/", 1)[-1])
    if "." in final_segment:
        final_segment = final_segment.rsplit(".", 1)[0]
    return final_segment


def normalize_retrieval_name(name: str) -> str:
    return " ".join(name.replace("-", " ").split())


def _read_train_answers(train_path: Path) -> dict[str, list[str]]:
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    return {str(query_id): [str(doc_id) for doc_id in row["answer"]] for query_id, row in payload.items()}


def build_preprocessed_corpus(
    raw_contexts: Path, train_path: Path, output_dir: Path
) -> dict[str, Any]:
    paths = sorted(raw_contexts.glob("context_*.json"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No context_*.json files found in {raw_contexts}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent)))
    try:
        contexts_dir = temp_dir / "contexts"
        contexts_dir.mkdir()
        exclusions: list[dict[str, Any]] = []
        raw_ids: set[str] = set()
        retained_ids: set[str] = set()
        raw_missing_name_count = 0
        retained_derived_name_count = 0
        for path in paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            doc_id = str(raw["id"])
            if doc_id in raw_ids:
                raise ValueError(f"Duplicate raw document id: {doc_id}")
            raw_ids.add(doc_id)
            supplied_name = str(raw.get("name") or "").strip()
            if not supplied_name:
                raw_missing_name_count += 1
            passage = str(raw.get("passage") or "")
            reasons: list[str] = []
            if not passage.strip():
                reasons.append(EMPTY_REASON)
            duplicate_retained_id = DUPLICATE_EXCLUDED.get(doc_id)
            if duplicate_retained_id:
                reasons.append("exact_duplicate_raw_passage")
            if reasons:
                exclusions.append({"doc_id": doc_id, "reasons": reasons, "duplicate_retained_id": duplicate_retained_id})
                continue
            link = str(raw.get("link") or "").strip()
            derived_name = "" if supplied_name else derive_name_from_link(link)
            if not supplied_name:
                retained_derived_name_count += 1
            base_name = supplied_name or derived_name
            retrieval_name = normalize_retrieval_name(base_name)
            processed = {
                "id": raw["id"], "passage": passage, "link": link,
                "name": supplied_name, "derived_name": derived_name,
                "retrieval_name": retrieval_name,
                "name_provenance": "supplied" if supplied_name else "derived_from_link_final_path_segment",
                "raw_source_file": path.name,
                "raw_source_sha256": sha256_file(path),
            }
            (contexts_dir / path.name).write_text(canonical_json(processed) + "\n", encoding="utf-8", newline="\n")
            retained_ids.add(doc_id)

        expected_retained = set(DUPLICATE_RETAINED) & raw_ids
        if expected_retained - retained_ids:
            raise ValueError("A manually retained duplicate document was not retained")
        answers = _read_train_answers(train_path)
        excluded_ids = {row["doc_id"] for row in exclusions}
        impact_rows = []
        impact_counts: Counter[str] = Counter()
        for query_id, answer_ids in sorted(answers.items()):
            removed = sorted(set(answer_ids) & excluded_ids)
            if removed:
                reasons = {row["doc_id"]: row["reasons"] for row in exclusions if row["doc_id"] in removed}
                for row_reasons in reasons.values():
                    impact_counts.update(row_reasons)
                impact_rows.append({"query_id": query_id, "gold_ids": answer_ids, "intentionally_excluded_gold_ids": removed, "reasons": reasons})
        exclusions_path = temp_dir / "exclusions.json"
        exclusions_path.write_text(json.dumps(exclusions, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        impacts_path = temp_dir / "train_label_impact.jsonl"
        impacts_path.write_text("".join(canonical_json(row) + "\n" for row in impact_rows), encoding="utf-8", newline="\n")
        retained_paths = sorted(contexts_dir.glob("context_*.json"), key=lambda path: path.name)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "raw_contexts_fingerprint": corpus_fingerprint(paths),
            "raw_context_count": len(paths), "retained_context_count": len(retained_paths),
            "excluded_context_count": len(exclusions),
            "raw_missing_name_count": raw_missing_name_count,
            "retained_derived_name_count": retained_derived_name_count,
            "duplicate_policy": {"retained_ids": sorted(DUPLICATE_RETAINED), "excluded_to_retained": DUPLICATE_EXCLUDED},
            "empty_passage_policy": "exclude_from_retrieval_corpus", "name_policy": "URL final path segment without extension; hyphens replaced and whitespace compacted",
            "exclusion_counts": dict(sorted(Counter(reason for row in exclusions for reason in row["reasons"]).items())),
            "train_label_impact": {"queries_affected": len(impact_rows), "gold_id_occurrences": sum(len(row["intentionally_excluded_gold_ids"]) for row in impact_rows), "reason_occurrences": dict(sorted(impact_counts.items()))},
            "artifacts": {"exclusions.json": sha256_file(exclusions_path), "train_label_impact.jsonl": sha256_file(impacts_path)},
            "processed_contexts_fingerprint": corpus_fingerprint(retained_paths),
        }
        manifest_payload = canonical_json(manifest).encode("utf-8")
        manifest["content_fingerprint"] = hashlib.sha256(manifest_payload).hexdigest()
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        (temp_dir / "_SUCCESS.json").write_text(json.dumps({"stage": "preprocessing", "content_fingerprint": manifest["content_fingerprint"]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing namespace: {output_dir}")
        temp_dir.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-contexts", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_preprocessed_corpus(args.raw_contexts, args.train_file, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
