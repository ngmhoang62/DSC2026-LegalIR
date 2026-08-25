import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from exp024_memory_lexical_backoff import _char_expression, build_query_memory_oof, retrieve_char_backoff

class Exp024Tests(unittest.TestCase):
    def test_memory_fits_only_train_side_and_blocks_heldout_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); train=root/"train.json"; folds=root/"folds.json"
            train.write_text(json.dumps({"a":{"question":"thu tuc xe","answer":["da"]},"b":{"question":"thu tuc xe","answer":["db"]},"c":{"question":"thue","answer":["dc"]}}),encoding="utf-8")
            folds.write_text(json.dumps({"fold_0":["a","b"],"fold_1":["c"]}),encoding="utf-8")
            build_query_memory_oof(train_path=train,folds_path=folds,output_dir=root/"out")
            for path in (root/"out").glob("*.jsonl"):
                for row in map(json.loads,path.read_text(encoding="utf-8").splitlines()):
                    forbidden={"a","b"} if row["fold"]=="fold_0" else {"c"}
                    self.assertFalse(forbidden & {q for r in row["rankings"] for q in r["neighbor_qids"]})
    def test_char_backoff_uses_raw_character_substrings(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); db=root/"x.sqlite"; con=sqlite3.connect(db)
            con.execute("CREATE VIRTUAL TABLE chunks USING fts5(text,doc_id UNINDEXED,chunk_id UNINDEXED,tokenize='trigram')")
            con.execute("INSERT INTO chunks VALUES('nguoi lao dong','d1','c1')"); con.commit(); con.close()
            train=root/"train.json"; train.write_text(json.dumps({"q":{"question":"nguoi lao dong","answer":["d1"]}},ensure_ascii=False),encoding="utf-8")
            out=root/"out.jsonl"; retrieve_char_backoff(train_path=train,index_path=db,output_path=out)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["rankings"][0]["doc_id"],"d1")
    def test_char_expression_is_safe_for_fts(self): self.assertNotIn("'",_char_expression("a' OR b"))

if __name__=="__main__": unittest.main()
