"""EXP-024: strictly fold-isolated query memory and raw-char lexical backoff.

This experiment intentionally evaluates source rescue *beyond* the frozen
EXP-022 E5@100 + BM25@50 candidate list.  It does not tune a downstream
ranker or claim a public result.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, stage_run, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legalir.exp024_memory_lexical_backoff.v1"
MEMORY_CONFIGS = {"char_3_5": {"analyzer": "char_wb", "ngram_range": (3, 5)},
                  "word_1_2": {"analyzer": "word", "ngram_range": (1, 2)}}
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _read_train(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ({str(q): str(row["question"]) for q, row in raw.items()},
            {str(q): set(map(str, row["answer"])) for q, row in raw.items()})


def _folds(path: Path) -> dict[str, list[str]]:
    values = {name: sorted(map(str, rows)) for name, rows in json.loads(path.read_text(encoding="utf-8")).items()}
    if not values or len({qid for rows in values.values() for qid in rows}) != sum(map(len, values.values())):
        raise ValueError("fixed folds are missing or overlap")
    return values


def _rank_memory(*, train_ids: Sequence[str], target_ids: Sequence[str], queries: Mapping[str, str],
                 answers: Mapping[str, set[str]], config: Mapping[str, Any], neighbours: int, documents: int) -> dict[str, list[dict[str, Any]]]:
    """Fit only on train-side query text and transfer only train-side labels."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(**config, dtype=np.float32, sublinear_tf=True, norm="l2", max_features=160000)
    matrix = vectorizer.fit_transform([queries[qid] for qid in train_ids])
    target = vectorizer.transform([queries[qid] for qid in target_ids])
    output: dict[str, list[dict[str, Any]]] = {}
    for position, qid in enumerate(target_ids):
        scores = (target[position] @ matrix.T).toarray().ravel()
        order = sorted(range(len(train_ids)), key=lambda i: (-float(scores[i]), train_ids[i]))[:neighbours]
        docs: dict[str, float] = defaultdict(float); provenance: dict[str, list[str]] = defaultdict(list)
        for rank, index in enumerate(order, 1):
            if scores[index] <= 0: continue
            neighbour = train_ids[index]
            for doc_id in sorted(answers[neighbour]):
                docs[doc_id] += float(scores[index]) / rank
                provenance[doc_id].append(neighbour)
        ranked = sorted(docs, key=lambda doc: (-docs[doc], doc))[:documents]
        output[qid] = [{"doc_id": doc, "rank": rank, "score": float(docs[doc]), "neighbor_qids": provenance[doc]}
                       for rank, doc in enumerate(ranked, 1)]
    return output


def build_query_memory_oof(*, train_path: Path, folds_path: Path, output_dir: Path, neighbours: int = 10, documents: int = 10) -> dict[str, Any]:
    queries, answers = _read_train(train_path); folds = _folds(folds_path)
    if set(queries) != {qid for qids in folds.values() for qid in qids}: raise ValueError("train/folds mismatch")
    output_dir.mkdir(parents=True, exist_ok=True); all_rows: dict[str, list[dict[str, Any]]] = {}
    with stage_run(output_dir, "query-memory-oof", total=len(queries)) as logger:
        for name, config in MEMORY_CONFIGS.items():
            rows: list[dict[str, Any]] = []
            for fold, heldout in sorted(folds.items()):
                train_ids = sorted(set(queries) - set(heldout))
                ranked = _rank_memory(train_ids=train_ids, target_ids=heldout, queries=queries, answers=answers,
                                      config=config, neighbours=neighbours, documents=documents)
                for qid in heldout:
                    # An explicit provenance invariant makes leakage auditable from the artifact.
                    if any(set(row["neighbor_qids"]) & set(heldout) for row in ranked[qid]):
                        raise RuntimeError(f"heldout label leakage: {fold}/{qid}")
                    rows.append({"qid": qid, "fold": fold, "config": name, "rankings": ranked[qid]})
            rows.sort(key=lambda row: row["qid"]); path = output_dir / f"{name}.jsonl"; write_jsonl(path, rows); all_rows[name] = rows
            logger.log(f"config={name} queries={len(rows)}")
        manifest = {"schema_version": SCHEMA, "stage": "query-memory-oof", "configs": MEMORY_CONFIGS,
                    "neighbours": neighbours, "documents": documents, "fold_isolated_vectorizer": True,
                    "inputs": {"train_sha256": sha256_file(train_path), "folds_sha256": sha256_file(folds_path)},
                    "artifacts": {name: sha256_file(output_dir / f"{name}.jsonl") for name in MEMORY_CONFIGS}}
        atomic_json(output_dir / "manifest.json", manifest); atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": manifest["stage"]})
    return manifest


def build_char_index(*, chunks_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build a raw-text FTS5 trigram index; no Vietnamese word segmentation is used."""
    output_dir.mkdir(parents=True, exist_ok=True); db = output_dir / "raw_char_fts.sqlite"
    if db.exists(): raise RuntimeError("char index exists; use a new EXP namespace rather than overwriting it")
    with stage_run(output_dir, "build-char-fts", total=None) as logger:
        con = sqlite3.connect(db)
        try:
            con.execute("CREATE VIRTUAL TABLE chunks USING fts5(text, doc_id UNINDEXED, chunk_id UNINDEXED, tokenize='trigram')")
            for count, row in enumerate(read_jsonl(chunks_path), 1):
                text = (str(row.get("retrieval_text", "")) + " " + str(row.get("raw_text", ""))).casefold()
                con.execute("INSERT INTO chunks(text, doc_id, chunk_id) VALUES(?,?,?)", (text, str(row["doc_id"]), str(row["chunk_id"])))
                if count % 8192 == 0: con.commit(); logger.status(stage="build-char-fts", state="RUNNING", completed=count, total=None)
            con.commit()
        finally: con.close()
        manifest = {"schema_version": SCHEMA, "stage": "build-char-fts", "tokenizer": "fts5-trigram-raw-casefold",
                    "inputs": {"chunks_sha256": sha256_file(chunks_path)}, "artifacts": {db.name: sha256_file(db)}}
        atomic_json(output_dir / "manifest.json", manifest); atomic_json(output_dir / "_SUCCESS.json", {"schema_version": SCHEMA, "stage": manifest["stage"]})
    return manifest


def _char_expression(query: str) -> str:
    # A backoff must be selective.  Common short terms turn trigram FTS into
    # an exhaustive corpus scan; use the six most distinctive surface forms.
    tokens = sorted({token.casefold() for token in TOKEN.findall(query) if len(token) >= 5}, key=lambda token: (-len(token), token))[:6]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens) or '"__no_match__"'


def retrieve_char_backoff(*, train_path: Path, index_path: Path, output_path: Path, chunk_limit: int = 512, documents: int = 10,
                          only_qids: set[str] | None = None) -> None:
    queries, _ = _read_train(train_path); con = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    # This is a bounded backoff: its purpose is to surface a few lexical
    # alternatives, not to rerun an exhaustive sparse retrieval per query.
    rows=[]
    try:
        for qid in sorted(queries):
            if only_qids is not None and qid not in only_qids: continue
            # FTS5's bm25 auxiliary function cannot be nested in SQL aggregate
            # functions.  Chunks are score-sorted, so keep the first hit per
            # parent document as its deterministic document score.
            chunks = con.execute("SELECT doc_id, bm25(chunks) score FROM chunks WHERE chunks MATCH ? ORDER BY score, chunk_id LIMIT ?", (_char_expression(queries[qid]), chunk_limit)).fetchall()
            found: list[tuple[str, float]] = []; seen: set[str] = set()
            for doc, score in chunks:
                if str(doc) not in seen:
                    seen.add(str(doc)); found.append((str(doc), float(score)))
                if len(found) == documents: break
            rows.append({"qid": qid, "rankings": [{"doc_id": doc, "rank": i, "score": score} for i,(doc,score) in enumerate(found,1)]})
    finally: con.close()
    write_jsonl(output_path, rows)


def audit_rescue(*, train_path: Path, folds_path: Path, exclusions_path: Path, baseline_path: Path,
                 memory_dir: Path, char_path: Path | None, output_dir: Path) -> dict[str, Any]:
    queries, answers = _read_train(train_path); folds = _folds(folds_path); excluded={str(x["doc_id"]) for x in json.loads(exclusions_path.read_text(encoding="utf-8"))}
    baseline={str(row["qid"]): {str(x["doc_id"]) for x in row["candidates"]} for row in read_jsonl(baseline_path)}
    if set(baseline)!=set(queries): raise ValueError("baseline/train mismatch")
    sources={name:{str(row["qid"]):row["rankings"] for row in read_jsonl(memory_dir/f"{name}.jsonl")} for name in MEMORY_CONFIGS}
    if char_path: sources["char_backoff"]={str(row["qid"]):row["rankings"] for row in read_jsonl(char_path)}
    rows=[]; summary={}
    for name,ranking in sources.items():
        rescues=[]; added=[]
        for qid in sorted(queries):
            if qid not in ranking:
                continue
            gold=answers[qid]-excluded; docs=[str(x["doc_id"]) for x in ranking[qid]]; new=[doc for doc in docs if doc not in baseline[qid]]
            hit=sorted(gold & set(new)); added.append(len(new))
            for doc in hit: rescues.append({"qid":qid,"fold":next(f for f,v in folds.items() if qid in v),"gold_doc_id":doc,"rank":docs.index(doc)+1,"query":queries[qid]})
        by_fold={fold:sum(x["fold"]==fold for x in rescues) for fold in folds}
        summary[name]={"rescued_retained_gold_occurrences":len(rescues),"rescued_queries":len({x["qid"] for x in rescues}),"mean_novel_candidates":float(np.mean(added)),"rescues_by_fold":by_fold}
        rows.extend({"channel":name,**x} for x in rescues)
    output_dir.mkdir(parents=True,exist_ok=True); write_jsonl(output_dir/"rescues.jsonl",rows)
    report={"schema_version":SCHEMA,"status":"PASS","baseline":"EXP-022 fixed E5@100 + BM25@50 union (150 cap)","summary":summary,
            "limitations":"Source rescue is an unbounded diagnostic; a channel must next win a fixed-150 replacement ablation before downstream use."}
    atomic_json(output_dir/"report.json",report); return report


def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("memory","build-char-index","char-retrieve","audit")); p.add_argument("--cache",type=Path,default=ROOT/"cache"/"exp024_memory_lexical"); p.add_argument("--results",type=Path,default=ROOT/"results"/"exp024_memory_lexical"); p.add_argument("--train",type=Path,default=ROOT/"public_test_dataset"/"train.json"); p.add_argument("--folds",type=Path,default=ROOT/"cache"/"cv_folds.json"); p.add_argument("--only-qids-jsonl",type=Path); a=p.parse_args(argv)
    if a.stage=="memory": result=build_query_memory_oof(train_path=a.train,folds_path=a.folds,output_dir=a.cache/"memory")
    elif a.stage=="build-char-index": result=build_char_index(chunks_path=ROOT/"cache"/"structural_v3_e5_final_v1"/"chunks.jsonl",output_dir=a.cache/"char_fts")
    elif a.stage=="char-retrieve":
        only = {str(row["qid"]) for row in read_jsonl(a.only_qids_jsonl)} if a.only_qids_jsonl else None
        retrieve_char_backoff(train_path=a.train,index_path=a.cache/"char_fts"/"raw_char_fts.sqlite",output_path=a.cache/"char_backoff.jsonl",only_qids=only); result={"status":"PASS", "queries": len(only) if only else len(_read_train(a.train)[0])}
    else: result=audit_rescue(train_path=a.train,folds_path=a.folds,exclusions_path=ROOT/"cache"/"final_preprocessed_v2"/"exclusions.json",baseline_path=ROOT/"cache"/"exp022_e5_bm25_union"/"train_oof_candidates.jsonl",memory_dir=a.cache/"memory",char_path=(a.cache/"char_backoff.jsonl") if (a.cache/"char_backoff.jsonl").exists() else None,output_dir=a.results)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
