import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC=Path(__file__).resolve().parents[1]/"src"; sys.path.insert(0,str(SRC))
from exp029_nested_lora_reranker import MODELS, _job, _pair_scores, _qid_set, choose_screen, render_candidate

class Exp029Tests(unittest.TestCase):
    def test_model_registry_and_deterministic_selector(self):
        self.assertEqual(len(MODELS),6)
        choice=choose_screen([{ "model":"qwen3","recall@5":.5,"precision@5":.1,"seconds_per_query":1.}, {"model":"bge_m3","recall@5":.5,"precision@5":.1,"seconds_per_query":1.}])
        self.assertEqual(choice["model"],"bge_m3")

    def test_renderer_es_is_deterministic_and_e_contains_no_heading(self):
        row={"document_label":"Luat A","scope_node_ids":["n"],"evidence":[{"raw_text":"Noi dung phap ly "*100}]}; nodes={"n":{"heading_text":"Dieu 1"}}
        self.assertIn("Dieu 1",render_candidate("q",row,structural=True,nodes=nodes))
        self.assertNotIn("Dieu 1",render_candidate("q",row,structural=False,nodes=nodes))

    def test_resume_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); calls=[]
            first=_job(root,"ok","f",lambda:(calls.append(1) or {"ok":True}),resume=False)
            second=_job(root,"ok","f",lambda:(calls.append(2) or {"ok":False}),resume=True)
            failed=_job(root,"bad","f2",lambda:(_ for _ in ()).throw(ValueError("x")),isolate="FAILED_MODEL")
            self.assertEqual(calls,[1]); self.assertEqual(first["payload"],second["payload"]); self.assertEqual(failed["state"],"FAILED_MODEL")

    def test_gte_pair_adapter_removes_token_type_ids(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                import torch
                return {"input_ids":torch.ones((1,2),dtype=torch.long), "attention_mask":torch.ones((1,2),dtype=torch.long), "token_type_ids":torch.ones((1,2),dtype=torch.long)}
        class Model:
            def __call__(self, **batch):
                import torch
                self.batch=batch; return type("Output",(),{"logits":torch.tensor([[.5]])})()
        model=Model(); gte=[x for x in MODELS if x.key=="gte"][0]
        self.assertEqual(_pair_scores(gte,model,Tokenizer(),"q",["d"],"cpu"),[.5])
        self.assertNotIn("token_type_ids",model.batch)

    def test_worker_qids_accept_scalar_or_list_json(self):
        with tempfile.TemporaryDirectory() as d:
            one=Path(d)/"one.json"; many=Path(d)/"many.json"
            one.write_text(json.dumps("100004"),encoding="utf8"); many.write_text(json.dumps([100004,"x"]),encoding="utf8")
            self.assertEqual(_qid_set(one),{"100004"}); self.assertEqual(_qid_set(many),{"100004","x"})

if __name__=="__main__": unittest.main()
