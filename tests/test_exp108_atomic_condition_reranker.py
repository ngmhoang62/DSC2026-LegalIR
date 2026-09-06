from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp108_atomic_condition_reranker import (
    budget_package,
    choose_negatives,
    condition_coverage,
    parse_conditions,
    residual_loss,
    residual_rank_bounded,
    rrf_candidates,
    select_atomic_units,
    stage_complete,
    unit_context_text,
)


class FakeTokenizer:
    def __call__(self, left, right=None, add_special_tokens=True, return_offsets_mapping=False, **_):
        def encode(text):
            values=[]; offsets=[]; cursor=0
            for token in str(text).split():
                start=str(text).find(token,cursor); end=start+len(token); cursor=end
                values.append(abs(hash(token))%1000+1); offsets.append((start,end))
            return values,offsets
        ids,offsets=encode(left)
        if right is not None:
            more,_=encode(right); ids=ids+more
        if add_special_tokens: ids=[0]+ids+[2]
        result={"input_ids":ids}
        if return_offsets_mapping: result["offset_mapping"]=offsets
        return result


def candidate(doc, dense=None, sparse=None):
    sources={}
    if dense is not None: sources["e5"]={"rank":dense,"aggregate_score":1/dense,"evidence":[]}
    if sparse is not None: sources["bm25"]={"rank":sparse}
    return {"doc_id":doc,"rank":int(doc),"sources":sources}


def test_rrf_keeps_bm25_only_and_actual_score():
    rows=[candidate(str(i),i,None) for i in range(1,51)]+[candidate("51",None,1)]
    ranked=rrf_candidates(rows,.55,.45)
    assert "51" in {row["doc_id"] for row in ranked}
    row=next(row for row in ranked if row["doc_id"]=="51")
    assert np.isclose(row["rrf_score"],.45/33)
    assert row["rrf_score"] != 1/(32+row["rank"])


def test_parser_is_conservative_and_never_empty():
    values=parse_conditions("Theo khoản 2 Điều 10, khi doanh nghiệp có vốn 20 tỷ đồng thì thời hạn là 15 ngày?")
    kinds={row["kind"] for row in values}
    assert {"citation","number","condition"} <= kinds
    assert parse_conditions("quy định này áp dụng thế nào?")[0]["kind"]=="query"


def test_condition_selector_can_choose_two_disjoint_units():
    conditions=parse_conditions("khi vốn là 20 tỷ đồng thì thời hạn 15 ngày")
    units=[
        {"node_id":"a","doc_id":"1","start":0,"end":30,"raw_text":"vốn là 20 tỷ đồng","token_count":5,"dense_score":.8},
        {"node_id":"b","doc_id":"1","start":100,"end":140,"raw_text":"thời hạn giải quyết 15 ngày","token_count":6,"dense_score":.7},
        {"node_id":"c","doc_id":"1","start":10,"end":20,"raw_text":"nội dung khác","token_count":3,"dense_score":.1},
    ]
    selected=select_atomic_units(units,"vốn 20 tỷ đồng thời hạn 15 ngày",conditions,"condition")
    assert [row["node_id"] for row in selected]==["a","b"]


def test_breadcrumb_is_part_of_condition_coverage():
    conditions=parse_conditions("áp dụng khoản 2 Điều 10")
    unit={"node_id":"a","doc_id":"1","start":0,"end":20,"raw_text":"Nội dung áp dụng.","token_count":4,"dense_score":.5,"breadcrumb":["Điều 10","Khoản 2"]}
    selected=select_atomic_units([unit],"áp dụng khoản 2 Điều 10",conditions,"condition")
    assert {"c00","c01"} <= set(selected[0]["coverage"])


def test_budget_package_never_silently_exceeds_512():
    tokenizer=FakeTokenizer(); text=" ".join(f"t{i}" for i in range(900)); unit={"node_id":"a","doc_id":"1","start":10,"end":10+len(text),"raw_text":text,"token_count":900,"dense_score":1,"coverage":[],"breadcrumb":["Điều 1"],"document_label":"Luật thử"}
    package,selected,audit=budget_package(tokenizer,"hỏi điều kiện",[unit])
    assert audit["pair_tokens"]<=512
    assert selected[0]["atomic_window_fallback"] is True
    assert selected[0]["selected_end"]<=unit["end"]
    assert package


def test_curriculum_is_six_unique_non_gold_and_rotates():
    rows=[{"doc_id":str(i),"rank":i,"lexical_score":i/50,"condition_score":i/50} for i in range(1,51)]
    first=choose_negatives(rows,{"1"},0,"curriculum"); second=choose_negatives(rows,{"1"},1,"curriculum")
    assert len(first)==len({row["doc_id"] for row in first})==6
    assert not ({row["doc_id"] for row in first}&{"1"})
    assert [row["doc_id"] for row in first] != [row["doc_id"] for row in second]


def test_residual_train_and_inference_share_bounded_transform():
    rows=[{"doc_id":"a","stage1_score":.4},{"doc_id":"b","stage1_score":.3},{"doc_id":"c","stage1_score":.2}]
    ce={"a":-3.,"b":1.,"c":2.}
    ranking=residual_rank_bounded(rows,ce,.25)
    anchor=torch.tensor((np.asarray([.4,.3,.2])-.3)/np.std([.4,.3,.2]),dtype=torch.float32)
    raw=torch.tensor([-3.,1.,2.]); expected=(1-.25)*anchor+.25*torch.tanh(raw)
    assert ranking==[rows[i]["doc_id"] for i in torch.argsort(expected,descending=True).tolist()]
    loss=residual_loss(anchor,raw,torch.tensor([1.,0.,0.]),.25)
    assert torch.isfinite(loss)


def test_rejected_artifact_is_not_resumable_success(tmp_path):
    (tmp_path/"_SUCCESS.json").write_text("{}",encoding="utf8")
    (tmp_path/"REPORT.json").write_text('{"status":"REJECTED_EVIDENCE_GATE"}',encoding="utf8")
    assert stage_complete(tmp_path) is False
    (tmp_path/"REPORT.json").write_text('{"status":"PASS"}',encoding="utf8")
    assert stage_complete(tmp_path) is True
