"""Measure whether the completed EXP-111 sparse ranker complements memory LTR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MEMORY=ROOT/'results/exp_final_retrieval/memory_ltr_probe'
SPARSE=ROOT/'results/exp_final_retrieval/sparse_ltr_probe/lr_all_sparse/PREDICTIONS.json'
OUT=ROOT/'results/exp_final_retrieval/sparse_complement_probe'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def metrics(pred,labels,qids):
    values=[];multi=[];precision=[];mrr=[]
    for q in qids:
        g=labels.get(q,set())
        if not g:continue
        top=pred[q][:5];h=len(set(top)&g);v=h/len(g);values.append(v);precision.append(h/5)
        if len(g)>1:multi.append(v)
        first=next((i for i,d in enumerate(top,1) if d in g),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))
def fuse(a,b,w,k=32):
    ar={d:i for i,d in enumerate(a,1)};br={d:i for i,d in enumerate(b,1)};docs=set(ar)|set(br)
    score={d:(1-w)/(k+ar[d]) if d in ar else 0 for d in docs}
    for d in br:score[d]+=w/(k+br[d])
    return sorted(docs,key=lambda d:(-score[d],d))
def main():
    import sys;sys.path.insert(0,str(ROOT/'src'));import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');sparse=read(SPARSE);memory={}
    for f in range(1,5):memory.update(read(MEMORY/f'fold_{f}'/'PREDICTIONS.json'))
    qids=[q for f in range(1,5) for q in folds[f'fold_{f}'] if labels.get(q)]
    base=metrics(memory,labels,qids);sparse_metrics=metrics(sparse,labels,qids);trials=[]
    for w in (.01,.02,.03,.05,.075,.10,.15,.20,.30,.50):
        ranked={q:fuse(memory[q],sparse[q],w) for q in qids};score=metrics(ranked,labels,qids)
        per={f'fold_{f}':metrics(ranked,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)]) for f in range(1,5)}
        trials.append(dict(weight=w,metrics=score,delta=score['recall_at_5']-base['recall_at_5'],multi_delta=score['multi_gold_recall_at_5']-base['multi_gold_recall_at_5'],per_fold=per))
    trials.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True)
    wins=losses=0;oracle=[]
    for q in qids:
        g=labels[q];m=len(set(memory[q][:5])&g);s=len(set(sparse[q][:5])&g);wins+=s>m;losses+=s<m;oracle.append(max(m,s)/len(g))
    report=dict(status='COMPLETE_SPARSE_COMPLEMENT_PROBE',scope='F1-F4 only',memory=base,sparse=sparse_metrics,choice_oracle_recall_at_5=float(np.mean(oracle)),sparse_query_wins=wins,sparse_query_losses=losses,top_trials=trials[:10])
    write(OUT/'SPARSE_COMPLEMENT_REPORT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
