"""EXP-038 multi-head query retrieval over frozen VietLegal-E5 embeddings.

Head zero is exact identity.  Two zero-output residual heads are trainable;
their head-normalized log-sum-exp is deliberately invariant to duplicating an
identical view.  This file contains the core model and a gate-safe CLI; it
does not alter any prior experiment or document embedding cache.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from exp012b_core import atomic_json, sha256_file
from exp034_shallow_retrieval import DIMENSION, RANK, evaluate_rankings

ROOT=Path(__file__).resolve().parents[1]; SCHEMA="legalir.exp038_multihead_e5_retrieval.v1"; SEED=2107
EPOCHS=4; BATCH=128; LR=1e-3; WEIGHT_DECAY=1e-4; GRAD_CLIP=1.; DIVERSITY=.02; KS=(5,16,24,32,40,50,64)

class MultiHeadProjection(nn.Module):
    """Identity plus two residual rank-32 projections; U=0 gives exact E5."""
    def __init__(self,dim:int=DIMENSION,rank:int=RANK)->None:
        super().__init__(); self.v=nn.ModuleList([nn.Linear(dim,rank,bias=False) for _ in range(2)]); self.u=nn.ModuleList([nn.Linear(rank,dim,bias=False) for _ in range(2)])
        for u in self.u: nn.init.zeros_(u.weight)
    def forward(self,q:torch.Tensor)->torch.Tensor:
        heads=[torch.nn.functional.normalize(q,dim=-1)]
        for u,v in zip(self.u,self.v): heads.append(torch.nn.functional.normalize(q+u(torch.nn.functional.gelu(v(q))),dim=-1))
        return torch.stack(heads,dim=1)
    def diversity(self)->torch.Tensor:
        a,b=(u.weight.flatten() for u in self.u); return (torch.nn.functional.cosine_similarity(a,b,dim=0)**2)

def aggregate_heads(head_scores:torch.Tensor)->torch.Tensor:
    """Normalized LSE: repeated equivalent heads leave the score unchanged."""
    return torch.logsumexp(head_scores,dim=-1)-math.log(head_scores.shape[-1])

def multi_positive_pairwise_loss(parent_scores:torch.Tensor,positive_mask:torch.Tensor,negative_mask:torch.Tensor)->torch.Tensor:
    """Mean LSE pairwise loss over every canonical positive parent."""
    pos=parent_scores[positive_mask]; neg=parent_scores[negative_mask]
    if not len(pos) or not len(neg): raise ValueError("both positives and clean negatives are required")
    return torch.nn.functional.softplus(neg[None,:]-pos[:,None]).logsumexp(dim=1).mean()-math.log(len(neg))

def identity_ranking(query:np.ndarray,docs:np.ndarray,doc_ids:Iterable[str])->list[str]:
    s=np.asarray(docs,dtype=np.float32)@np.asarray(query,dtype=np.float32); return [d for d,_ in sorted(zip(map(str,doc_ids),s),key=lambda x:(-float(x[1]),x[0]))]

def audit(out:Path)->dict:
    source=ROOT/"results"/"exp036_coverage_aware_fusion"/"REPORT.json"
    if not source.exists(): raise RuntimeError("EXP-036 report is required before EXP-038")
    status=json.loads(source.read_text(encoding="utf-8")).get("status")
    if status not in ("REJECTED","REJECTED_EARLY_FOLD0"):
        raise RuntimeError("EXP-038 is forbidden unless EXP-036 completed REJECTED")
    result={"schema_version":SCHEMA,"status":"READY_AFTER_EXP036_REJECTION","exp036_report_sha256":sha256_file(source),"architecture":{"heads":3,"identity_head":0,"trainable_heads":2,"rank":RANK,"diversity_penalty":DIVERSITY},"fold_isolation":"required for training, mining, calibration, and checkpoints"}
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/"AUDIT.json",result);atomic_json(out/"RUN_STATUS.json",{"status":result["status"],"stage":"audit"});return result

def run_fold(out:Path,fold:str)->dict:
    # Refuse accidental un-audited long training. The core model is above; a
    # subsequent implementation stage must bind frozen chunk/query manifests.
    a=json.loads((out/"AUDIT.json").read_text(encoding="utf-8"))
    result={"schema_version":SCHEMA,"status":"BLOCKED_MANIFEST_BINDING_REQUIRED","outer_fold":str(fold),"reason":"Exact chunk/query manifest and fold-isolated mining binding must be recorded before GPU training."}
    atomic_json(out/f"fold_{fold}"/"RUN_STATUS.json",result);return result

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("audit","fixture","run-fold","overnight","report"));p.add_argument("--outer",default="0");p.add_argument("--resume",action="store_true");a=p.parse_args();out=ROOT/"cache"/"exp038_multihead_e5_retrieval"
    r=audit(out) if a.stage in ("audit","fixture") else run_fold(out,a.outer)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":main()
