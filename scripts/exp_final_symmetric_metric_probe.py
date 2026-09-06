"""Strict held-fold probe for symmetric query/document E5 metric adaptation."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp_final.data import Data, SourceStore  # noqa: E402
from exp_final.symmetric_metric import (  # noqa: E402
    ProjectedParentBank,
    ResidualMetricTower,
    SymmetricTrainConfig,
    deterministic_negatives,
    symmetric_multi_loss,
    top5_boundary_loss,
)

OUT = ROOT / "results" / "exp_final_retrieval" / "symmetric_metric_probe"
CACHE = ROOT / "cache" / "exp_final_retrieval" / "symmetric_metric_probe"
SOURCE_DB = ROOT / "cache" / "exp112_task_adaptive_retrieval" / "sources.sqlite"
BASE_PATH = ROOT / "results" / "exp_final_retrieval" / "profile_ltr_probe" / "l15_t5" / "PREDICTIONS.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def metrics(rankings, labels, qids):
    recall, precision, single, multi, reciprocal = [], [], [], [], []
    for qid in qids:
        gold = labels.get(qid, set())
        if not gold:
            continue
        top = rankings[qid][:5]
        hits = len(set(top) & gold); value = hits / len(gold)
        recall.append(value); precision.append(hits / 5)
        (single if len(gold) == 1 else multi).append(value)
        reciprocal.append(next((1/r for r,d in enumerate(top,1) if d in gold),0.0))
    return {
        "recall_at_5": float(np.mean(recall)), "precision_at_5": float(np.mean(precision)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)), "mrr_at_5": float(np.mean(reciprocal)),
        "queries": len(recall),
    }


def metric_key(value):
    return (value["recall_at_5"], value["precision_at_5"], value["multi_gold_recall_at_5"], value["mrr_at_5"])


def fuse(base, expert, weight, constant=32):
    scores = {doc:(1-weight)/(constant+rank) for rank,doc in enumerate(base,1)}
    for rank,doc in enumerate(expert,1):
        scores[doc] = scores.get(doc,0.0)+weight/(constant+rank)
    return sorted(scores,key=lambda doc:(-scores[doc],doc))


def choice_oracle(left,right,labels,qids):
    pred={};choices={"left":0,"right":0,"tie":0}
    for q in qids:
        gold=labels[q];a=len(set(left[q][:5])&gold);b=len(set(right[q][:5])&gold)
        side="right" if b>a else "left" if a>b else "tie";choices[side]+=1
        pred[q]=right[q] if side=="right" else left[q]
    return {"metrics":metrics(pred,labels,qids),"choices":choices}


def query_matrix(data,qids):
    return normalize(np.stack([data.query_vector(q,"e5") for q in qids]))


def save_checkpoint(path,model,optimizer,epoch,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"config":config.__dict__},temporary)
    temporary.replace(path)


def train_one_epoch(data,store,model,bank,raw_bank,qids,qvectors,optimizer,config,epoch,device,max_updates=None):
    order=list(range(len(qids)));random.Random(config.seed+epoch).shuffle(order)
    model.train();started=time.monotonic();losses=[];updates=0
    for start in range(0,len(order),config.batch_size):
        rows=order[start:start+config.batch_size]
        raw_query=torch.as_tensor(qvectors[rows],device=device,dtype=torch.float32)
        adapted_query=model.query(raw_query)
        with torch.no_grad():
            parent_scores,top_chunks=bank.mine(adapted_query)
            ordering=torch.argsort(parent_scores,dim=1,descending=True,stable=True)
        local=[]
        for batch_row,row in enumerate(rows):
            qid=qids[row];current=[data.doc_ids[i] for i in ordering[batch_row].cpu().tolist()]
            sources=store.rankings(qid)
            negatives=deterministic_negatives(current,sources,data.gold[qid],data.doc_ids,qid,epoch)
            positives=sorted(data.gold[qid]);docs=positives+negatives
            parent_rows=torch.as_tensor([data.doc_row[d] for d in docs],device=device)
            chunk_rows=top_chunks[batch_row,parent_rows].reshape(-1).cpu().numpy()
            raw_document=torch.as_tensor(np.array(raw_bank[chunk_rows],dtype=np.float32,copy=True),device=device)
            adapted_document=model.document(raw_document).reshape(len(docs),2,-1)
            pair_scores=(adapted_document*adapted_query[batch_row][None,None,:]).sum(-1).mean(-1)
            positive=pair_scores[:len(positives)];negative=pair_scores[len(positives):]
            loss=symmetric_multi_loss(positive,negative)
            if epoch>0:
                loss=loss+config.boundary_weight*top5_boundary_loss(positive,negative,len(positives))
            qdrift=1-F.cosine_similarity(adapted_query[batch_row:batch_row+1],raw_query[batch_row:batch_row+1]).mean()
            ddrift=1-F.cosine_similarity(adapted_document.reshape(-1,adapted_document.shape[-1]),F.normalize(raw_document,dim=-1)).mean()
            local.append(loss+config.drift_weight*(qdrift+ddrift))
        total=torch.stack(local).mean();optimizer.zero_grad(set_to_none=True);total.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if float(gradient)==0:raise ValueError("Symmetric metric received zero gradient")
        optimizer.step();losses.append(float(total.detach()));updates+=1
        if updates%10==0 or start+len(rows)==len(order):
            elapsed=time.monotonic()-started
            eta=elapsed/max(1,start+len(rows))*(len(order)-start-len(rows))
            print(f"[symmetric-train] epoch={epoch+1} {start+len(rows)}/{len(order)} loss={np.mean(losses[-10:]):.4f} eta_s={eta:.0f}",flush=True)
        if max_updates and updates>=max_updates:break
    return {"updates":updates,"seconds":time.monotonic()-started,"mean_loss":float(np.mean(losses))}


@torch.no_grad()
def score(data,model,bank,qids,qvectors,device):
    rankings={};started=time.monotonic()
    for start in range(0,len(qids),16):
        ids=qids[start:start+16]
        query=model.query(torch.as_tensor(qvectors[start:start+len(ids)],device=device,dtype=torch.float32))
        parent_scores,_=bank.mine(query)
        order=torch.argsort(parent_scores,dim=1,descending=True,stable=True)[:,:500].cpu().numpy()
        for qid,indices in zip(ids,order):rankings[qid]=[data.doc_ids[int(i)] for i in indices]
        if (start//16+1)%20==0 or start+len(ids)==len(qids):
            print(f"[symmetric-score] {start+len(ids)}/{len(qids)}",flush=True)
    return rankings,time.monotonic()-started


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--outer",type=int,default=3);parser.add_argument("--epochs",type=int,default=2);parser.add_argument("--max-updates",type=int);args=parser.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu";config=SymmetricTrainConfig(epochs=args.epochs)
    data=Data();store=SourceStore(SOURCE_DB);raw_bank=data.matrix("e5")
    train_folds=list(range(args.outer));train_qids=[q for fold in train_folds for q in data.folds[f"fold_{fold}"] if data.gold.get(q)]
    test_qids=[q for q in data.folds[f"fold_{args.outer}"] if data.gold.get(q)]
    train_vectors=query_matrix(data,train_qids);test_vectors=query_matrix(data,test_qids)
    base={str(q):list(map(str,v)) for q,v in read(BASE_PATH).items()}
    native={q:[r["doc_id"] for r in store.get(q,"e5")] for q in test_qids}
    model=ResidualMetricTower(rank=config.rank).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    bank=ProjectedParentBank(raw_bank.shape,data.parent,device=device);history=[];systems={}
    print(f"[symmetric] device={device} outer=fold_{args.outer} train_folds={train_folds} train={len(train_qids)} test={len(test_qids)}",flush=True)
    try:
        for epoch in range(args.epochs):
            refresh_start=time.monotonic();bank.refresh(raw_bank,model);refresh_seconds=time.monotonic()-refresh_start
            training=train_one_epoch(data,store,model,bank,raw_bank,train_qids,train_vectors,optimizer,config,epoch,device,args.max_updates)
            history.append({"epoch":epoch+1,"refresh_seconds":refresh_seconds,**training})
            if args.max_updates:break
            bank.refresh(raw_bank,model)
            adapted,score_seconds=score(data,model,bank,test_qids,test_vectors,device)
            trials=[]
            for constant in (0,32):
                for weight in (.02,.05,.10,.15,.20,.30):
                    pred={q:fuse(base[q],adapted[q],weight,constant) for q in test_qids};value=metrics(pred,data.gold,test_qids)
                    trials.append({"constant":constant,"weight":weight,"metrics":value,"delta":value["recall_at_5"]-metrics(base,data.gold,test_qids)["recall_at_5"]})
            trials.sort(key=lambda row:metric_key(row["metrics"]),reverse=True)
            systems[f"epoch_{epoch+1}"]={
                "adapted":metrics(adapted,data.gold,test_qids),"native_e5":metrics(native,data.gold,test_qids),
                "base":metrics(base,data.gold,test_qids),"top_fusions":trials[:10],
                "choice_oracle":choice_oracle(base,adapted,data.gold,test_qids),"score_seconds":score_seconds,
            }
            write(OUT/f"fold_{args.outer}_epoch_{epoch+1}_top500.json",adapted)
            save_checkpoint(CACHE/f"fold_{args.outer}_epoch_{epoch+1}.pt",model,optimizer,epoch+1,config)
            print(f"[symmetric-eval] epoch={epoch+1} adapted_R5={systems[f'epoch_{epoch+1}']['adapted']['recall_at_5']:.6f} best_fused_R5={trials[0]['metrics']['recall_at_5']:.6f}",flush=True)
        report={
            "status":"BENCHMARK" if args.max_updates else "COMPLETE_SYMMETRIC_METRIC_HELD_FOLD_PROBE",
            "scope":{"outer":args.outer,"train_folds":train_folds,"fold4_untouched":args.outer!=4},
            "config":config.__dict__,"history":history,"systems":systems,
        }
        write(OUT/f"FOLD_{args.outer}_REPORT.json" if not args.max_updates else OUT/"BENCHMARK.json",report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    finally:
        store.close();del bank,model,optimizer
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=="__main__":main()
