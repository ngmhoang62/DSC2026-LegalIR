"""Fold-isolated extreme-classification probe using document query prototypes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "exp112_task_adaptive_retrieval"
OUT = ROOT / "results" / "exp_final_retrieval" / "xmc_probe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def stable_top(scores, ids, n=200):
    take = min(n, len(ids)); part = np.argpartition(-scores, take - 1)[:take]
    return [ids[i] for i in sorted(part.tolist(), key=lambda i: (-float(scores[i]), ids[i]))]


def fuse(base, expert, weight, k=32):
    br, er = {d:i for i,d in enumerate(base,1)}, {d:i for i,d in enumerate(expert,1)}
    docs = set(br) | set(er)
    score = {d:(1-weight)/(k+br[d]) if d in br else 0. for d in docs}
    for d in er: score[d] += weight/(k+er[d])
    return sorted(docs,key=lambda d:(-score[d],d))


def metrics(rankings, labels, qids):
    values=[];multi=[];precision=[]
    for q in qids:
        g=labels.get(q,set())
        if not g: continue
        h=len(set(rankings[q][:5])&g);v=h/len(g)
        values.append(v);precision.append(h/5)
        if len(g)>1:multi.append(v)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),queries=len(values))


def main():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json')
    base={}
    for f in range(5):
        rows=read(OLD/'outer'/f'fold_{f}'/'PREDICTIONS.json')
        base.update({q:list(map(str,r['order'])) for q,r in rows.items()})
    eval_ids=[q for q in base if labels.get(q)]; baseline=metrics(base,labels,eval_ids)
    e5_dir=ROOT/'cache/exp021_e5_dense_candidates/query_embeddings'
    ids=list(map(str,read(e5_dir/'train_query_ids.json')));e5=norm(np.load(e5_dir/'train_queries.f32.npy',mmap_mode='r'))
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:
        l_ids=list(map(str,z['query_ids'].tolist())); l=norm(z['vectors'])
    lm={q:i for i,q in enumerate(l_ids)};lal=l[[lm[q] for q in ids]];mean=norm(e5+lal); row={q:i for i,q in enumerate(ids)}
    representations={'e5':e5,'lal':lal,'mean_e5_lal':mean}
    experts={name:{} for name in representations}
    coverage={name:{} for name in representations}
    for f in range(5):
        held=[q for q in folds[f'fold_{f}'] if labels.get(q)]
        support=[q for j in range(5) if j!=f for q in folds[f'fold_{j}'] if labels.get(q)]
        docs=sorted(set().union(*(labels[q] for q in support)))
        by_doc=defaultdict(list)
        for q in support:
            for d in labels[q]:by_doc[d].append(row[q])
        for name,matrix in representations.items():
            centroids=norm(np.stack([matrix[by_doc[d]].mean(0) for d in docs]))
            sim=matrix[[row[q] for q in held]]@centroids.T
            recovered=0; oracle=[]
            for i,q in enumerate(held):
                ranked=stable_top(sim[i],docs,200);experts[name][q]=ranked
                current=set(base[q][:5])&labels[q];reachable=set(ranked)&labels[q];new=reachable-current
                recovered+=len(new);oracle.append(min(5,len(current)+len(new))/len(labels[q]))
            coverage[name][f'fold_{f}']=dict(oracle_recall_at_5=float(np.mean(oracle)),reachable_missing_gold_assignments=recovered,seen_documents=len(docs))
    trials=[]
    for name,expert in experts.items():
        standalone=metrics(expert,labels,eval_ids)
        for weight in (.02,.05,.10,.15,.20,.30,.50):
            ranked={q:fuse(base[q],expert[q],weight) for q in eval_ids}
            score=metrics(ranked,labels,eval_ids)
            per={f'fold_{f}':metrics(ranked,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)]) for f in range(5)}
            base_per={f'fold_{f}':metrics(base,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)]) for f in range(5)}
            trials.append(dict(representation=name,weight=weight,standalone=standalone,metrics=score,
                               delta=score['recall_at_5']-baseline['recall_at_5'],
                               multi_delta=score['multi_gold_recall_at_5']-baseline['multi_gold_recall_at_5'],
                               nonnegative_folds=sum(per[k]['recall_at_5']>=base_per[k]['recall_at_5'] for k in per),per_fold=per))
    trials.sort(key=lambda x:(x['metrics']['recall_at_5'],x['metrics']['precision_at_5']),reverse=True)
    report=dict(status='COMPLETE_DEVELOPMENT_XMC_PROBE',warning='Fold-isolated prototypes; recipes inspected on exposed development OOF.',baseline=baseline,coverage=coverage,trials=len(trials),top_trials=trials[:30])
    write(OUT/'XMC_PROBE.json',report);print(json.dumps({**{k:report[k] for k in ('status','baseline','coverage','trials')},'top5':trials[:5]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
