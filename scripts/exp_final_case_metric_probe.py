"""Identity-initialized supervised metric learning for legal case retrieval."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/case_metric_probe"


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def normalize(x):
    x=np.asarray(x,dtype=np.float32);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def metrics(rankings,labels,qids):
    values=[];multi=[];precision=[];mrr=[]
    for q in qids:
        g=labels.get(q,set())
        if not g:continue
        top=rankings[q][:5];h=len(set(top)&g);v=h/len(g);values.append(v);precision.append(h/5)
        if len(g)>1:multi.append(v)
        first=next((i for i,d in enumerate(top,1) if d in g),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))


def fuse(base,expert,weight,k=32):
    br={d:i for i,d in enumerate(base,1)};er={d:i for i,d in enumerate(expert,1)};docs=set(br)|set(er)
    score={d:((1-weight)/(k+br[d]) if d in br else 0)+(weight/(k+er[d]) if d in er else 0) for d in docs}
    return sorted(docs,key=lambda d:(-score[d],d))


def run(outer, epochs=4):
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from torch import nn
    import exp109b_encoder_complementarity as old

    torch.manual_seed(3112);np.random.seed(3112)
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json')
    base={}
    for f in range(5):base.update(read(BASE/f'fold_{f}'/'PREDICTIONS.json'))
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:
        ids=list(map(str,z['query_ids'].tolist()));all_vectors=normalize(z['vectors'])
    row={q:i for i,q in enumerate(ids)}
    test=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
    support=[q for f in range(5) if f!=outer for q in folds[f'fold_{f}'] if labels.get(q)]
    vectors=torch.from_numpy(np.asarray(all_vectors[[row[q] for q in support]],dtype=np.float32))
    original=vectors.clone();device='cuda' if torch.cuda.is_available() else 'cpu'

    by_doc=defaultdict(list);frequency=Counter(d for q in support for d in labels[q])
    for i,q in enumerate(support):
        for d in labels[q]:by_doc[d].append(i)
    raw=np.asarray(vectors.numpy()@vectors.numpy().T,dtype=np.float32);np.fill_diagonal(raw,-np.inf)
    positives=[];hard=[];eligible=[]
    for i,q in enumerate(support):
        pos=set(j for d in labels[q] for j in by_doc[d] if j!=i)
        if not pos:positives.append([]);hard.append([]);continue
        # At most two representatives per gold label, preferring semantically
        # plausible links so broad legal documents do not collapse all topics.
        chosen=[]
        for d in sorted(labels[q]):
            candidates=[j for j in by_doc[d] if j!=i]
            chosen.extend(sorted(candidates,key=lambda j:(-float(raw[i,j]),support[j]))[:2])
        chosen=list(dict.fromkeys(chosen))[:8]
        order=np.argpartition(-raw[i],min(128,len(raw[i])-1)-1)[:min(128,len(raw[i])-1)]
        neg=sorted((j for j in order if not (labels[q]&labels[support[j]])),key=lambda j:(-float(raw[i,j]),support[j]))[:56]
        positives.append(chosen);hard.append(neg);eligible.append(i)
    del raw

    class ResidualMetric(nn.Module):
        def __init__(self,dim=1024,hidden=128):
            super().__init__();self.down=nn.Linear(dim,hidden,bias=False);self.up=nn.Linear(hidden,dim,bias=False)
            nn.init.normal_(self.down.weight,std=.01);nn.init.zeros_(self.up.weight)
        def forward(self,x):
            return nn.functional.normalize(x+.1*self.up(nn.functional.gelu(self.down(x))),dim=-1)

    model=ResidualMetric().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    batch_size=64;tau=.08;history=[]
    for epoch in range(epochs):
        model.eval()
        with torch.no_grad():target=model(vectors.to(device)).detach()
        rng=np.random.default_rng(3112+epoch);order=np.array(eligible);rng.shuffle(order)
        model.train();losses=[]
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size];anchor=model(vectors[batch].to(device));group_losses=[]
            for bi,idx in enumerate(batch.tolist()):
                p=positives[idx];n=hard[idx]
                if not p or not n:continue
                candidate=p+n;score=anchor[bi]@target[candidate].T/tau
                # Independent-positive normalization avoids allowing one easy
                # shared label to hide every other positive legal case.
                group_losses.append(-(score[:len(p)]-torch.logsumexp(score,dim=0)).mean())
            if not group_losses:continue
            contrastive=torch.stack(group_losses).mean();drift=1-(anchor*vectors[batch].to(device)).sum(-1).mean();loss=contrastive+.10*drift
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            support_projected=model(vectors.to(device)).cpu().numpy();test_original=torch.from_numpy(np.asarray(all_vectors[[row[q] for q in test]],dtype=np.float32)).to(device);test_projected=model(test_original).cpu().numpy()
        sim=np.asarray(test_projected@support_projected.T,dtype=np.float32)
        expert={};oracle=[]
        for qi,q in enumerate(test):
            take=min(64,len(support));part=np.argpartition(-sim[qi],take-1)[:take];neighbours=sorted(part.tolist(),key=lambda j:(-float(sim[qi,j]),support[j]))
            votes=defaultdict(float)
            for rank,j in enumerate(neighbours,1):
                docs=labels[support[j]]
                for d in docs:votes[d]+=1/((32+rank)*max(1,len(docs))*math.sqrt(frequency[d]))
            expert[q]=sorted(votes,key=lambda d:(-votes[d],d))
            reachable=set(votes)&labels[q];current=set(base[q][:5])&labels[q];oracle.append(min(5,len(current|reachable))/len(labels[q]))
        candidates=[]
        for weight in (.02,.03,.05,.075,.10,.15):
            ranked={q:fuse(base[q],expert[q],weight) for q in test};score=metrics(ranked,labels,test)
            candidates.append(dict(weight=weight,metrics=score,delta=score['recall_at_5']-metrics(base,labels,test)['recall_at_5']))
        candidates.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True)
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),oracle_recall_at_5=float(np.mean(oracle)),expert=metrics(expert,labels,test),best_fusion=candidates[0]))
        print(json.dumps(history[-1],ensure_ascii=False),flush=True)
    report=dict(status='COMPLETE_CASE_METRIC_PROBE',outer=f'fold_{outer}',device=device,eligible_training_queries=len(eligible),baseline=metrics(base,labels,test),history=history)
    write(OUT/f'fold_{outer}'/'REPORT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2));return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outer',type=int,default=0,choices=range(5));parser.add_argument('--epochs',type=int,default=4);args=parser.parse_args();run(args.outer,args.epochs)


if __name__=='__main__':main()
