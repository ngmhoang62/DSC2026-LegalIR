"""Strict OOF dense label-centroid probe for seen-label LegalIR errors."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exp_final_retrieval/label_centroid_probe"


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def normalize(x):
    x=np.asarray(x,dtype=np.float32);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def metrics(rankings,labels,qids):
    values=[];precision=[];multi=[];mrr=[]
    for q in qids:
        gold=labels.get(q,set())
        if not gold:continue
        top=rankings[q][:5];hits=len(set(top)&gold);v=hits/len(gold);values.append(v);precision.append(hits/5)
        if len(gold)>1:multi.append(v)
        first=next((i for i,d in enumerate(top,1) if d in gold),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))
def prototypes(vectors,qids,labels):
    by_doc=defaultdict(list)
    for row,qid in enumerate(qids):
        for doc in labels.get(qid,()):by_doc[doc].append(row)
    docs=sorted(by_doc);plain=[]
    for doc in docs:plain.append(vectors[by_doc[doc]].mean(axis=0))
    plain=normalize(np.asarray(plain,dtype=np.float32));mean=vectors.mean(axis=0,keepdims=True)
    centered=normalize(np.asarray([vectors[by_doc[doc]].mean(axis=0)-mean[0] for doc in docs],dtype=np.float32))
    return docs,plain,centered,normalize(vectors-mean)
def load_vectors():
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:
        ids=list(map(str,z['query_ids'].tolist()));lal=normalize(z['vectors'])
    folder=ROOT/'cache/exp021_e5_dense_candidates/query_embeddings';eids=list(map(str,read(folder/'train_query_ids.json')));e5=normalize(np.load(folder/'train_queries.f32.npy',mmap_mode='r'))
    emap={q:i for i,q in enumerate(eids)};e5_aligned=np.asarray(e5[[emap[q] for q in ids]],dtype=np.float32)
    return ids,{'lal':lal,'e5':e5_aligned}
def rrf(base,specialist,weight,k=32):
    rb={d:i for i,d in enumerate(base,1)};rs={d:i for i,d in enumerate(specialist,1)};docs=set(rb)|set(rs)
    return sorted(docs,key=lambda d:(-((1-weight)/(k+rb[d]) if d in rb else 0)-(weight/(k+rs[d]) if d in rs else 0),d))
def main():
    sys.path.insert(0,str(ROOT/'src'));import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');ids,representations=load_vectors();qrow={q:i for i,q in enumerate(ids)}
    baseline={}
    for fold in range(5):baseline.update(read(ROOT/f'results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json'))
    systems={f'{rep}_{mode}':{} for rep in representations for mode in ('plain','centered')};fold_stats={}
    for outer in range(5):
        train=[q for f in range(5) if f!=outer for q in folds[f'fold_{f}'] if labels.get(q)];test=[q for q in folds[f'fold_{outer}'] if labels.get(q)];fold_stats[f'fold_{outer}']={}
        for rep,vectors in representations.items():
            support=vectors[[qrow[q] for q in train]];docs,plain,centered,centered_support=prototypes(support,train,labels);queries=vectors[[qrow[q] for q in test]];mean=support.mean(axis=0,keepdims=True);centered_queries=normalize(queries-mean)
            for mode,query_matrix,prototype in (('plain',queries,plain),('centered',centered_queries,centered)):
                score=query_matrix@prototype.T
                for row,qid in enumerate(test):
                    order=np.lexsort((np.asarray(docs),-score[row]));head=[docs[i] for i in order[:200]];seen=set(head);systems[f'{rep}_{mode}'][qid]=head+[d for d in baseline[qid] if d not in seen]
                fold_stats[f'fold_{outer}'][f'{rep}_{mode}']=metrics(systems[f'{rep}_{mode}'],labels,test)
                print(f'outer={outer} {rep}_{mode}={fold_stats[f"fold_{outer}"][f"{rep}_{mode}"]["recall_at_5"]:.9f}',flush=True)
    qids=[q for q in baseline if labels.get(q)];base=metrics(baseline,labels,qids);system_metrics={name:metrics(rows,labels,qids) for name,rows in systems.items()};oracle={}
    for q in qids:
        choices=[baseline[q]]+[systems[name][q] for name in systems];oracle[q]=max(choices,key=lambda order:len(set(order[:5])&labels[q]))
    trials=[]
    for name,rows in systems.items():
        for weight in (.01,.02,.03,.05,.075,.10,.15,.20):
            ranked={q:rrf(baseline[q],rows[q],weight) for q in qids};value=metrics(ranked,labels,qids);trials.append(dict(system=name,weight=weight,metrics=value,delta=value['recall_at_5']-base['recall_at_5']))
    trials.sort(key=lambda x:(x['metrics']['recall_at_5'],x['metrics']['precision_at_5'],x['metrics']['mrr_at_5']),reverse=True)
    report=dict(status='COMPLETE_LABEL_CENTROID_PROBE',scope_warning='Strict outer label/prototype isolation; bounded direct fusion inspected on development OOF.',baseline=base,standalone=system_metrics,folds=fold_stats,choice_oracle=metrics(oracle,labels,qids),top_fusions=trials[:20]);write(OUT/'LABEL_CENTROID_REPORT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
