from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .contracts import CACHE, ROOT, RESULTS, V0, Progress, digest, read, records, sha, union, write
from .learning import ParentBank


class Data:
    def __init__(self):
        import exp109b_encoder_complementarity as old
        self.train = read(ROOT / "public_test_dataset/train.json")
        self.public = read(ROOT / "public_test_dataset/public-official.json")
        self.questions = {q: row["question"] for q, row in {**self.train, **self.public}.items()}
        self.original = {q: set(row["answer"]) for q, row in self.train.items()}
        self.gold, self.label_audit = old.canonical_labels()
        self.folds = read(ROOT / "cache/cv_folds.json")
        self.chunk_ids, self.chunk_parents = [], []
        for row in records(ROOT / "cache/e5_final_v1/chunk_ids.jsonl"):
            self.chunk_ids.append(row["chunk_id"]); self.chunk_parents.append(row["doc_id"])
        self.doc_ids = sorted(set(self.chunk_parents))
        self.doc_row = {d: i for i, d in enumerate(self.doc_ids)}
        self.parent = np.array([self.doc_row[d] for d in self.chunk_parents], dtype=np.int64)
        self.positions = [[] for _ in self.doc_ids]
        for i, p in enumerate(self.parent):
            self.positions[p].append(i)
        self.positions = [np.array(p, dtype=np.int64) for p in self.positions]
        self.fingerprint = digest([sha(ROOT / "public_test_dataset/train.json"), sha(ROOT / "cache/cv_folds.json"),
                                   read(ROOT / "cache/e5_final_v1/manifest.json")["cache_fingerprint"]])
        self._matrices, self._queries = {}, {}
        self.metadata = old.build_parent_text_metadata()

    def matrix(self, source):
        if source not in self._matrices:
            if source == "e5":
                path = ROOT / "cache/e5_final_v1/embeddings.f16.npy"
            else:
                path = CACHE / "lal.f16.npy"
                legacy = ROOT / "cache/exp112_task_adaptive_retrieval/lal.f16.npy"
                if not path.exists() and legacy.exists():
                    path = legacy
            self._matrices[source] = np.load(path, mmap_mode="r")
        return self._matrices[source]

    def query_vector(self, qid, source):
        if source not in self._queries:
            if source == "e5":
                folder = ROOT / "cache/exp021_e5_dense_candidates/query_embeddings"
                ids = read(folder / "train_query_ids.json")
                vectors = np.load(folder / "train_queries.f32.npy", mmap_mode="r")
            else:
                with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz", allow_pickle=False) as z:
                    ids, vectors = z["query_ids"].tolist(), np.array(z["vectors"], dtype=np.float32)
            self._queries[source] = ({str(q): i for i, q in enumerate(ids)}, vectors)
        indices, vectors = self._queries[source]
        if qid in indices:
            v = np.array(vectors[indices[qid]], dtype=np.float32)
        else:
            v = np.load(CACHE / "public_vectors" / source / f"{qid}.npy")
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def bank(self, source, device="cuda"):
        return ParentBank(self.matrix(source), self.parent, device=device)

    def exact(self, qid, doc_ids, source):
        q = self.query_vector(qid, source)
        matrix = self.matrix(source)
        values = []
        for d in doc_ids:
            v = np.asarray(matrix[self.positions[self.doc_row[d]]], dtype=np.float32)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            s = v @ q
            values.append(float(np.partition(s, -min(2, len(s)))[-2:].mean()))
        return values


def materialize_lal(data):
    folder = ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal"
    manifest = read(folder / "manifest.json")
    if not (folder / "_SUCCESS.json").exists():
        raise ValueError("LAL has no success marker")
    path = CACHE / "lal.f16.npy"
    marker = path.with_suffix(".json")
    signature = digest([manifest["content_fingerprint"], data.fingerprint])
    if marker.exists():
        info = read(marker)
        if info["signature"] != signature or sha(path) != info["sha256"]:
            raise ValueError("LAL materialized bank mismatch")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16, shape=(len(data.chunk_ids), 1024))
    offset = 0
    progress = Progress("materialize_lal", len(manifest["shards"]))
    for i, receipt in enumerate(manifest["shards"]):
        src = Path(receipt["path"])
        if sha(src) != receipt["sha256"]:
            raise ValueError(f"LAL shard hash mismatch: {src}")
        with np.load(src, allow_pickle=False) as z:
            n = len(z["vectors"])
            if list(map(str, z["ids"])) != data.chunk_ids[offset:offset+n]:
                raise ValueError("LAL chunk ordering mismatch")
            out[offset:offset+n] = z["vectors"]
            offset += n
        progress.update(i+1)
    if offset != len(data.chunk_ids):
        raise ValueError("Incomplete LAL bank")
    out.flush(); del out
    path.with_suffix(".tmp.npy").replace(path)
    write(marker, {"signature": signature, "sha256": sha(path)})


class SourceStore:
    """A bounded per-query reader; SQLite holds the full experiment, not Python."""
    def __init__(self, path=None):
        self.jina_enabled = (RESULTS / "RESOURCE_LOCK.json").exists() and read(RESULTS / "RESOURCE_LOCK.json").get("jina", False)
        path = Path(path or CACHE / "sources.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-32768")
        self.db.execute("CREATE TABLE IF NOT EXISTS sources(q TEXT, source TEXT, payload TEXT, signature TEXT, PRIMARY KEY(q,source))")

    def put(self, q, source, rows, signature):
        value = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        existing = self.db.execute("SELECT signature,payload FROM sources WHERE q=? AND source=?", (q, source)).fetchone()
        if existing and existing[0] != signature:
            raise ValueError(f"Source provenance changed {q}/{source}")
        if existing and existing[1] != value:
            raise ValueError(f'Source payload changed under the same fingerprint {q}/{source}')
        self.db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?)", (q, source, value, signature))

    def get(self, q, source):
        row = self.db.execute("SELECT payload FROM sources WHERE q=? AND source=?", (q, source)).fetchone()
        return json.loads(row[0]) if row else None

    def rankings(self, q):
        return {s: [r["doc_id"] for r in (self.get(q, s) or [])] if s != "jina" or self.jina_enabled else []
                for s in ("e5", "lal", "bm25", "trigram", "jina")}

    def candidates(self, q, base=False):
        sources=("e5", "lal", "bm25") if base else ("e5", "lal", "bm25", "trigram")
        return union([[r['doc_id'] for r in (self.get(q,s) or [])] for s in sources])

    def close(self):
        self.db.commit(); self.db.close()


def import_sources(store):
    progress = Progress("import_frozen_sources", 2)
    for i, (key, model) in enumerate((("e5", "vietlegal_e5"), ("lal", "vnlegal_lal"))):
        folder = ROOT / "cache/exp109b_encoder_complementarity/rankings" / model / "fold_0"
        manifest = read(folder / "manifest.json")
        if not (folder / "_SUCCESS.json").exists():
            raise ValueError(f"No source success marker: {folder}")
        for receipt in manifest["shards"]:
            path = folder / "shards" / receipt["name"]
            if sha(path) != receipt["sha256"]:
                raise ValueError(f"Source shard hash mismatch {path}")
            for row in records(path):
                store.put(str(row["qid"]), key, row["documents"], manifest["content_fingerprint"])
            store.db.commit()
        progress.update(i+1, force=True)
    for path in sorted((ROOT / "cache/exp111_multiview_sparse/source_scores_v2/inner").glob("scores-*.jsonl")):
        receipt = read(path.with_suffix(".json"))
        if sha(path) != receipt["sha256"]:
            raise ValueError("Trigram shard hash mismatch")
        for row in records(path):
            # V0 is intentionally not imported: those rows used fold-specific policies.
            rows = [dict(doc_id=r["doc_id"], rank=r["rank"], score=r["raw_score"]) for r in row["sources"]["v4_trigram"]]
            store.put(row["qid"], "trigram", rows, digest(["111-trigram", receipt["config_hash"]]))
        store.db.commit()


class TrigramReader:
    """Same EXP-111 SQL/scoring, bounded connection and tokenizer reuse."""
    def __init__(self):
        import exp111_multiview_sparse_retrieval as sparse
        self.sparse=sparse
        database=sparse.CACHE/'v2_v3/index.sqlite'
        self.conn=sqlite3.connect(f'file:{database.as_posix()}?mode=ro',uri=True)
        self.conn.execute('PRAGMA cache_size=-32768')
        self.total=int(self.conn.execute('SELECT count(*) FROM local384').fetchone()[0])
        self.tokenizer=sparse.ExactFtsSession(max_cache=2048)
        self.df={}
    def score(self,query):
        s=self.sparse
        if len(self.df)>4096:
            self.df.clear()
        terms=self.tokenizer.sequence(' '.join(s.surface(query)))
        expression=s.phrase_expression_terms(self.conn,'local384',terms,3,total_windows=self.total,df_cache=self.df)
        if expression=='"__exp111_no_token__"':
            return []
        columns='unit_id,doc_id,word_start,word_end,char_start,char_end,raw_text_hash'
        raw=self.conn.execute(f'SELECT {columns}, bm25(local384) AS score FROM local384 WHERE local384 MATCH ? ORDER BY score ASC, doc_id ASC, rowid ASC LIMIT ?', (expression,1500)).fetchall()
        rows=[dict(zip(columns.split(','),r[:-1]))|dict(score=float(r[-1]),unit_rank=i) for i,r in enumerate(raw,1)]
        return s.aggregate_units(rows,top=500,nonredundant=True,second_lambda=.6)
    def close(self):
        self.conn.close();self.tokenizer.close()


def prepare_sparse(data, store, qids):
    import exp111_multiview_sparse_retrieval as sparse
    progress = Progress("prepare_sparse", len(qids))
    signature = digest([V0, sha(Path(sparse.__file__))])
    trigram=None
    try:
        for i, q in enumerate(qids):
            for source, view in (("bm25", "v0_control"), ("trigram", "v4_trigram")):
                if store.get(q, source) is not None:
                    continue
                if source=='trigram':
                    if trigram is None:
                        trigram=TrigramReader()
                    rows=trigram.score(data.questions[q])
                else:
                    rows = sparse.source_rows(data.questions[q], view, limit=500, v0_config=V0)
                store.put(q, source, [dict(doc_id=r["doc_id"], rank=r["rank"], score=r["raw_score"]) for r in rows], signature)
            store.db.commit()
            progress.update(i+1)
    finally:
        if trigram:
            trigram.close()


def public_dense(data, store, source):
    import torch
    import exp109b_encoder_complementarity as old
    from .learning import QueryEncoder
    qids = [q for q in data.public if store.get(q, source) is None]
    if not qids:
        return
    vector_dir = CACHE / "public_vectors" / source; vector_dir.mkdir(parents=True, exist_ok=True)
    if source == "e5":
        model = QueryEncoder(); model.eval()
        encode = lambda texts: model(texts).detach().cpu().numpy()
    else:
        model = old.create_encoder("vnlegal_lal", device="cuda")
        encode = lambda texts: model.encode(texts, is_query=True).vectors
    try:
        for q in qids:
            with torch.no_grad():
                v = encode([data.questions[q]])[0]
            np.save(vector_dir / f"{q}.npy", v)
    finally:
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()
    bank = data.bank(source)
    progress = Progress("public_dense_"+source, len(qids))
    for i, q in enumerate(qids):
        scores, _ = bank.mine(torch.tensor(data.query_vector(q, source), device="cuda"))
        order = torch.argsort(scores[0], descending=True, stable=True)[:500].cpu().tolist()
        s = scores[0].cpu().tolist()
        store.put(q, source, [dict(doc_id=data.doc_ids[j], rank=r, score=s[j]) for r, j in enumerate(order, 1)], digest([data.fingerprint, source, "public-native-v1"]))
        store.db.commit(); progress.update(i+1)
    del bank; torch.cuda.empty_cache()
