"""Uniform approximate features: no learned-anchor refinement dependency."""
from __future__ import annotations

import gc
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .contracts import CACHE, ROOT, Progress, digest, read, records, sha, write


class BoundedMmapCache:
    def __init__(self, max_bytes=256*1024*1024):
        self.arrays=OrderedDict();self.bytes=0;self.max_bytes=max_bytes;self.opens=0
    def load(self,path):
        path=Path(path).resolve()
        if path in self.arrays:
            self.arrays.move_to_end(path);return self.arrays[path]
        value=np.load(path,mmap_mode='r');self.opens+=1
        while self.arrays and self.bytes+value.nbytes>self.max_bytes:
            _,old=self.arrays.popitem(last=False);self.bytes-=old.nbytes
        if value.nbytes<=self.max_bytes:
            self.arrays[path]=value;self.bytes+=value.nbytes
        return value
    @property
    def open_paths(self):
        return len(self.arrays)


def prepare_jina(data, store, qids, *, benchmark=False):
    import time
    import exp109c_latent_condition_late_interaction as old
    old.require_model_use_eligibility()
    root = old.CACHE_ROOT
    manifest = read(root / "queries/manifest.json")
    for name, expected in manifest["files"].items():
        if sha(root / "queries" / name) != expected:
            raise ValueError("Jina query hash mismatch")
    if manifest["config_e_contract"] != old.CONFIG_E_CONTRACT:
        raise ValueError("Jina model/index contract changed")
    index_header = read(root / 'index/manifest.json')
    idf_header = read(root/'idf/manifest.json')
    if sha(root/'idf/token_idf.json') != idf_header['files']['token_idf.json']:
        raise ValueError('Jina IDF hash mismatch')
    db = sqlite3.connect(CACHE / "jina_reuse.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS approximate(q TEXT,d TEXT,payload TEXT,PRIMARY KEY(q,d))")
    db.execute("CREATE TABLE IF NOT EXISTS imported(path TEXT PRIMARY KEY,sha TEXT)")
    for folder in sorted((root / "late_scores").rglob("manifest.json")):
        prior = read(folder)
        if (prior.get('index_fingerprint') != index_header['content_fingerprint']
            or prior.get('structural_fingerprint', index_header['structural_fingerprint']) != manifest['structural_fingerprint']
            or prior.get('idf_fingerprint', idf_header['content_fingerprint']) != idf_header['content_fingerprint']
            or prior.get('scorer_implementation_contract') != 'd100_batched_chunk_maxsim_masked_fp32_v1'):
            continue
        for receipt in prior.get("shards", []):
            path = folder.parent / "shards" / receipt["name"]
            if not path.exists():
                path = folder.parent / receipt["name"]
            previous = db.execute("SELECT sha FROM imported WHERE path=?", (str(path),)).fetchone()
            if previous and previous[0] == receipt["sha256"]:
                continue
            if sha(path) != receipt["sha256"]:
                raise ValueError("Jina score shard mismatch")
            for row in records(path):
                for item in row["scores"]:
                    if item.get("scoring_tier") != "approximate_medoids":
                        continue
                    f = {k: float(item["features"][k]) for k in old.LATE_FEATURES}
                    db.execute("INSERT OR REPLACE INTO approximate VALUES(?,?,?)", (row["qid"], item["doc_id"], json.dumps(f)))
            db.execute("INSERT OR REPLACE INTO imported VALUES(?,?)", (str(path), receipt["sha256"]))
            db.commit()
    query_ids = {q: i for i, q in enumerate(read(root / "queries/qids.json"))}
    vectors = np.load(root / "queries/query_vectors.f16.npy", mmap_mode="r")
    token_ids = np.load(root / "queries/query_token_ids.i64.npy", mmap_mode="r")
    offsets = np.load(root / "queries/query_offsets.i64.npy", mmap_mode="r")
    public_vectors = {}
    pending_public = [q for q in qids if q not in query_ids and store.get(q, "jina") is None]
    if pending_public:
        model, tokenizer, projection = old._load_model_for_encoding(device="cuda")
        for q in pending_public:
            v, ids, _, _ = old.encode_jina_texts(model, tokenizer, projection, [data.questions[q]], task="query", max_length=manifest["query_max_length"], device="cuda")
            public_vectors[q] = (v[0].astype(np.float16).astype(np.float32), ids[0])
        del model, tokenizer, projection
        gc.collect(); torch.cuda.empty_cache()
    catalog, index_manifest = old.load_index_catalog()
    idf = {int(k): v for k, v in read(root / "idf/token_idf.json").items()}
    mmap = BoundedMmapCache()
    signature = digest([data.fingerprint, index_manifest["content_fingerprint"], "uniform-approx-v1"])
    progress = Progress("jina_approximate", len(qids))
    started, repaired, actual, parity_errors = time.monotonic(), 0, 0, []
    for i, q in enumerate(qids):
        existing = store.get(q, 'jina')
        if existing is not None and not benchmark:
            continue
        if q in query_ids:
            row = query_ids[q]; a, b = int(offsets[row]), int(offsets[row+1])
            query, ids = np.array(vectors[a:b], dtype=np.float32), token_ids[a:b]
        else:
            query, ids = public_vectors[q]
        docs = store.candidates(q, base=True)
        local = {}
        for d in docs:
            saved = db.execute("SELECT payload FROM approximate WHERE q=? AND d=?", (q, d)).fetchone()
            if saved:
                local[d] = json.loads(saved[0])
        missing = [d for d in docs if d not in local]
        if benchmark:
            probe = sorted(local)[:2]
            if probe:
                checked, _ = old.score_candidate_pool_batched_cuda(query,
                    [dict(doc_id=d,candidate_rank=j+1) for j,d in enumerate(probe)],catalog,
                    idf_weights=[idf.get(int(t),1.) for t in ids],mmap_cache=mmap,full_tokens=False)
                for d in probe:
                    parity_errors.extend(abs(local[d][k]-checked[d]['late']['features'][k]) for k in old.LATE_FEATURES if k!='li_exact_rank_within_candidate_pool')
        if missing:
            new, timing = old.score_candidate_pool_batched_cuda(query,
                [dict(doc_id=d, candidate_rank=i+1) for i, d in enumerate(missing)], catalog,
                idf_weights=[idf.get(int(t), 1.) for t in ids], mmap_cache=mmap, full_tokens=False)
            for d in missing:
                local[d] = new[d]["late"]["features"]
            repaired += len(missing)
        order = sorted(docs, key=lambda d: (-local[d]["li_two_chunk_union_mean"], d))
        result = []
        for r, d in enumerate(order, 1):
            f = {k: float(local[d][k]) for k in old.LATE_FEATURES}
            f["li_exact_rank_within_candidate_pool"] = r
            result.append(dict(doc_id=d, rank=r, score=f["li_two_chunk_union_mean"], features=f))
        if existing is None:
            store.put(q, "jina", result, signature); store.db.commit()
        actual += 1
        progress.update(i+1)
    result = dict(seconds=time.monotonic()-started, queries=actual, repaired_pairs=repaired, signature=signature,
                  contract="uniform_approximate_no_learned_refinement", benchmark=benchmark)
    if benchmark:
        result['cache_parity_max_abs_error'] = max(parity_errors) if parity_errors else None
        result['passed'] = bool(parity_errors) and max(parity_errors)<=1e-5
        result['provenance'] = dict(query_manifest_sha256=sha(root/'queries/manifest.json'),index=index_header['content_fingerprint'],
                                    idf=idf_header['content_fingerprint'], legacy_missing_idf_header_checked_by_replay=True)
        write(CACHE/'JINA_REUSE_VALIDATION.json',result)
        if not result['passed']:
            raise ValueError('Cached Jina approximate features did not reproduce current frozen inputs')
    db.close(); del catalog, mmap
    gc.collect(); torch.cuda.empty_cache()
    return result
