"""Bounded capacity screen on the new LAL case-memory feature contract.

Historical tree tuning predates these nonlinear memory features.  Reuse the
already materialized fold-isolated matrices and compare only a small set of
capacity/truncation hypotheses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CACHE = ROOT / "cache/exp112_task_adaptive_retrieval"
LEGACY_RESULTS = ROOT / "results/exp112_task_adaptive_retrieval"
MEMORY = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
OUT = ROOT / "results/exp_final_retrieval/memory_capacity_probe"


CONFIGS = {
    "l7_m50_t30": dict(num_leaves=7, min_child_samples=50, lambdarank_truncation_level=30),
    "l15_m50_t10": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=10),
    "l31_m50_t10": dict(num_leaves=31, min_child_samples=50, lambdarank_truncation_level=10),
    "l15_m20_t10": dict(num_leaves=15, min_child_samples=20, lambdarank_truncation_level=10),
    "l15_m50_t5": dict(num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5),
}


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


def main():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import lightgbm as lgb
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data,SourceStore
    from exp_final.fusion import features
    from exp_final_memory_ltr_probe import memory_features, support_index

    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');data=Data()
    store=SourceStore(LEGACY_CACHE/'sources.sqlite');store.jina_enabled=True
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:
        ids=list(map(str,z['query_ids'].tolist()));vectors=normalize(z['vectors'])
    qrow={q:i for i,q in enumerate(ids)}
    all_predictions={name:{} for name in CONFIGS};baseline={};fold_reports={}
    for outer in range(5):
        marker=read(LEGACY_CACHE/f'outer/fold_{outer}/outer-ml.json')
        train_qids=[q for q in map(str,marker['training_qids']) if labels.get(q)]
        test_qids=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        groups=[];y=[]
        for q in train_qids:
            docs=list(dict.fromkeys(store.candidates(q)+sorted(labels[q])));groups.append(len(docs));y.extend(d in labels[q] for d in docs)
        train_matrix=np.load(MEMORY/f'fold_{outer}'/'train_augmented.f32.npy',mmap_mode='r')
        if len(train_matrix)!=sum(groups):raise ValueError('training matrix group mismatch')
        by_doc,frequency=support_index(labels,train_qids)
        support=vectors[[qrow[q] for q in train_qids]];similarities=np.asarray(vectors[[qrow[q] for q in test_qids]]@support.T,dtype=np.float32)
        test_groups=[];test_docs=[];rows=[]
        for qi,q in enumerate(test_qids):
            docs=store.candidates(q);xb=np.asarray(features(data,store,q,docs,2));xm=memory_features(similarities[qi],docs,train_qids,labels,by_doc,frequency)
            rows.append(np.concatenate([xb,xm],axis=1));test_groups.append(len(docs));test_docs.append(docs)
        test_matrix=np.concatenate(rows,axis=0);del rows,similarities
        legacy=read(LEGACY_RESULTS/f'outer/fold_{outer}'/'PREDICTIONS.json');baseline.update({q:legacy[q]['order'] for q in test_qids})
        offset=np.cumsum([0]+test_groups)
        per_config={}
        for name,config in CONFIGS.items():
            model=lgb.LGBMRanker(objective='lambdarank',learning_rate=.05,n_estimators=300,feature_fraction=1.,bagging_fraction=1.,deterministic=True,force_col_wise=True,n_jobs=4,random_state=113,verbosity=-1,**config)
            model.fit(train_matrix,np.asarray(y,dtype=np.int8),group=groups,eval_at=[5])
            score=model.predict(test_matrix);ranked={}
            for qi,(q,docs) in enumerate(zip(test_qids,test_docs)):
                local=score[offset[qi]:offset[qi+1]];ranked[q]=[docs[i] for i in sorted(range(len(docs)),key=lambda i:(-float(local[i]),docs[i]))]
            all_predictions[name].update(ranked);per_config[name]=metrics(ranked,labels,test_qids)
            print(f'outer={outer} config={name} recall={per_config[name]["recall_at_5"]:.9f}',flush=True)
        fold_reports[f'fold_{outer}']=per_config
        del train_matrix,test_matrix
    eval_qids=[q for q in baseline if labels.get(q)];base_metrics=metrics(baseline,labels,eval_qids)
    aggregate=[]
    for name,pred in all_predictions.items():
        score=metrics(pred,labels,eval_qids)
        deltas=[fold_reports[f'fold_{f}'][name]['recall_at_5']-metrics(baseline,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)])['recall_at_5'] for f in range(5)]
        aggregate.append(dict(config=name,parameters=CONFIGS[name],metrics=score,delta=score['recall_at_5']-base_metrics['recall_at_5'],fold_deltas=deltas,nonnegative_folds=sum(d>=0 for d in deltas)))
        write(OUT/name/'PREDICTIONS.json',pred)
    aggregate.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True)
    report=dict(status='COMPLETE_MEMORY_CAPACITY_PROBE',scope_warning='Bounded capacity screen on previously exposed development OOF.',baseline=base_metrics,folds=fold_reports,aggregate=aggregate)
    write(OUT/'MEMORY_CAPACITY_REPORT.json',report);store.close();print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
