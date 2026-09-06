from __future__ import annotations

import json
import sqlite3

import numpy as np

from .contracts import CACHE, ROOT, digest, records, sha, write


class Evidence:
    def __init__(self, data, tokenizer):
        self.data, self.tokenizer = data, tokenizer
        self.db = sqlite3.connect(CACHE / "evidence.sqlite")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks(doc TEXT, idx INTEGER PRIMARY KEY, payload TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc)")
        self.db.execute("CREATE TABLE IF NOT EXISTS documents(doc TEXT PRIMARY KEY,payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS packages(key TEXT PRIMARY KEY,payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS headings(node TEXT PRIMARY KEY,parent TEXT,heading TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS complete(kind TEXT PRIMARY KEY,signature TEXT)")
        signature = data.fingerprint
        prior = self.db.execute("SELECT signature FROM complete WHERE kind='inventory'").fetchone()
        if prior and prior[0] != signature:
            raise ValueError("Evidence inventory changed")
        if not prior:
            for i, row in enumerate(records(ROOT / "cache/structural_v3_e5_final_v1/chunks.jsonl")):
                if row["chunk_id"] != data.chunk_ids[i]:
                    raise ValueError("Evidence embedding order mismatch")
                if row["end"] - row["start"] != len(row["raw_text"]):
                    raise ValueError("Invalid chunk offsets")
                self.db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?)", (row["doc_id"], i, json.dumps(row, ensure_ascii=False)))
                if i % 1000 == 0:
                    self.db.commit()
            for row in records(ROOT / "cache/structural_v3_e5_final_v1/documents.jsonl"):
                self.db.execute("INSERT OR REPLACE INTO documents VALUES(?,?)", (row["doc_id"], json.dumps(row, ensure_ascii=False)))
            self.db.execute("INSERT INTO complete VALUES('inventory',?)", (signature,)); self.db.commit()
        if not self.db.execute("SELECT 1 FROM complete WHERE kind='headings'").fetchone():
            for row in records(ROOT / "cache/structural_v3_e5_final_v1/nodes.jsonl"):
                self.db.execute("INSERT OR REPLACE INTO headings VALUES(?,?,?)", (row["node_id"], row.get("parent_id"), row.get("heading_text", "")))
            self.db.execute("INSERT INTO complete VALUES('headings',?)", (signature,)); self.db.commit()
        if not self.db.execute("SELECT 1 FROM complete WHERE kind='source_verified'").fetchone():
            verified = 0
            for source in records(ROOT / 'cache/structural_v3_e5_final_v1/nodes.jsonl'):
                if source['kind'] != 'document':
                    continue
                for payload, in self.db.execute('SELECT payload FROM chunks WHERE doc=?', (source['doc_id'],)):
                    unit = json.loads(payload)
                    if source['raw_text'][unit['start']:unit['end']] != unit['raw_text']:
                        raise ValueError('Chunk is not an exact original document slice')
                    verified += 1
            if verified != len(data.chunk_ids):
                raise ValueError('Source ownership/coverage mismatch')
            self.db.execute("INSERT INTO complete VALUES('source_verified',?)", (signature,)); self.db.commit()

    def clip(self, text, budget):
        tokens = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(tokens["input_ids"]) <= budget:
            return text, len(text)
        end = tokens["offset_mapping"][budget-1][1] if budget > 0 else 0
        return text[:end], end

    def package(self, q, doc):
        from exp108_atomic_condition_reranker import lexical_score, parse_conditions, condition_coverage
        key = digest([self.data.fingerprint, q, doc, "evidence-v2-192-48-512-headings"])
        prior = self.db.execute("SELECT payload FROM packages WHERE key=?", (key,)).fetchone()
        if prior:
            return json.loads(prior[0])
        rows = [(i, json.loads(value)) for i, value in self.db.execute("SELECT idx,payload FROM chunks WHERE doc=? ORDER BY idx", (doc,))]
        if not rows:
            raise ValueError("Parent without evidence")
        question = self.data.questions[q]
        query_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        clipped = len(query_tokens) > 192
        if clipped:
            marker = self.tokenizer.encode(" [...] ", add_special_tokens=False)
            query_tokens = query_tokens[:128] + marker + query_tokens[-(64-len(marker)):]
            question = self.tokenizer.decode(query_tokens, skip_special_tokens=True)
        vectors = np.array(self.data.matrix("e5")[[i for i, _ in rows]], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        dense = vectors @ self.data.query_vector(q, "e5")
        lexical = [lexical_score(self.data.questions[q], r["raw_text"]) for _, r in rows]
        dr = {j: r for r, j in enumerate(sorted(range(len(rows)), key=lambda j: (-dense[j], rows[j][0])), 1)}
        lr = {j: r for r, j in enumerate(sorted(range(len(rows)), key=lambda j: (-lexical[j], rows[j][0])), 1)}
        ordered = sorted(range(len(rows)), key=lambda j: (-.5/(32+dr[j])-.5/(32+lr[j]), rows[j][0]))
        first = rows[ordered[0]][1]
        metadata = json.loads(self.db.execute("SELECT payload FROM documents WHERE doc=?", (doc,)).fetchone()[0])
        ancestry, node, seen = [], first.get("parent_node_id"), set()
        while node and node not in seen:
            seen.add(node)
            heading = self.db.execute("SELECT parent,heading FROM headings WHERE node=?", (node,)).fetchone()
            if not heading:
                break
            if heading[1]:
                ancestry.append(heading[1])
            node = heading[0]
        meta, _ = self.clip(" > ".join([metadata.get("name", ""), *reversed(ancestry)]).strip(" >"), 48)
        special = self.tokenizer.num_special_tokens_to_add(pair=True)
        body_budget = 512-special-len(self.tokenizer.encode(question, add_special_tokens=False))-len(self.tokenizer.encode(meta, add_special_tokens=False))-8
        body, end = self.clip(first["raw_text"], body_budget)
        selected = [dict(chunk_id=first["chunk_id"], start=first["start"], end=first["start"]+end, partial=end<len(first["raw_text"]))]
        conditions = parse_conditions(self.data.questions[q]); covered = condition_coverage(conditions, body)
        for j in ordered[1:]:
            second = rows[j][1]
            if max(first["start"], second["start"]) < min(first["end"], second["end"]):
                continue
            if not condition_coverage(conditions, second["raw_text"]) - covered:
                continue
            proposed = body + "\n\n" + second["raw_text"]
            if len(self.tokenizer.encode(proposed, add_special_tokens=False)) <= body_budget:
                body = proposed
                selected.append(dict(chunk_id=second["chunk_id"], start=second["start"], end=second["end"], partial=False))
                break
        document = body + ("\n"+meta if meta else "")
        encoded = self.tokenizer(question, document, truncation=False)
        while len(encoded["input_ids"]) > 512:
            # Explicit source-window shortening, never silent tokenizer truncation.
            if len(selected) > 1:
                selected = selected[:1]; body = first["raw_text"][:selected[0]["end"]-first["start"]]
            else:
                body = body[:-1]; selected[0]["end"] -= 1; selected[0]["partial"] = True
            document = body + ("\n"+meta if meta else "")
            encoded = self.tokenizer(question, document, truncation=False)
        if not body:
            raise ValueError("Empty evidence")
        value = dict(question=question, document=document, selected=selected, query_clipped=clipped,
                     pair_tokens=len(encoded["input_ids"]), source_exact=True, rendered_hash=digest([question, document]))
        self.db.execute("INSERT INTO packages VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False))); self.db.commit()
        return value
