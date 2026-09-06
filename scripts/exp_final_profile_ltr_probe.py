"""Add fold-isolated supervised BM25 label-profile features to memory LTR."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LEGACY_CACHE=ROOT/'cache/exp112_task_adaptive_retrieval'
MEMORY=ROOT/'results/exp_final_retrieval/memory_ltr_probe'
OUT=ROOT/'results/exp_final_retrieval/profile_ltr_probe'
PROFILE_CONFIGS=((1,1.2,.75,0.0),(3,1.2,.75,.3))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def normalize(x):
    x=np.asarray(x,dtype=np.float32);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def profile_scores(text,model,config):
    from exp_final_supervised_profile_probe import feats
    max_ngram,k1,b,prior_power=config;postings,df,length,frequency=model;count=max(len(length),1);avg=sum(length.values())/count;scores={}
    from collections import defaultdict
    values=defaultdict(float)
    for feature in feats(text,max_ngram):
        posting=postings.get(feature)
        if not posting:continue
        idf=math.log1p((count-df[feature]+.5)/(df[feature]+.5));order=int(feature[0]);phrase=(1.,1.35,1.65)[order-1]
        for doc,tf in posting.items():
            norm=k1*(1-b+b*length[doc]/avg);values[doc]+=phrase*idf*tf*(k1+1)/(tf+norm)
    maximum=max(frequency.values(),default=1)
    for doc,value in values.items():values[doc]=value*((frequency[doc]+.5)/(maximum+.5))**prior_power
    return values
def profile_features(text,docs,model):
    columns=[];frequency=model[3]
    for config in PROFILE_CONFIGS:
        scores=profile_scores(text,model,config);ordered=sorted(scores,key=lambda d:(-scores[d],d));ranks={d:i for i,d in enumerate(ordered,1)}
        distribution=np.asarray(list(scores.values()),dtype=np.float64);mean=float(distribution.mean()) if len(distribution) else 0.;std=float(distribution.std()) if len(distribution) else 0.;rank5=scores[ordered[4]] if len(ordered)>=5 else (scores[ordered[-1]] if ordered else 0.)
        raw=np.asarray([scores.get(d,0.) for d in docs],dtype=np.float32);present=np.asarray([d in scores for d in docs],dtype=np.float32)
        columns.extend([present,raw,np.asarray([(scores.get(d,0.)-mean)/std if d in scores and std>1e-12 else 0. for d in docs]),np.asarray([1/(32+ranks[d]) if d in ranks else 0. for d in docs]),np.asarray([rank5-scores.get(d,0.) if d in scores else 0. for d in docs]),np.asarray([math.log1p(frequency.get(d,0)) for d in docs])])
    result=np.asarray(columns,dtype=np.float32).T
    if not np.isfinite(result).all():raise ValueError('nonfinite profile features')
    return result
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
    import sys;sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
    import lightgbm as lgb
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data,SourceStore
    from exp_final.fusion import features
    from exp_final_memory_ltr_probe import memory_features,support_index
    from exp_final_supervised_profile_probe import build_profiles
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');fold_of={q:f for f in range(5) for q in folds[f'fold_{f}']};data=Data();store=SourceStore(LEGACY_CACHE/'sources.sqlite');store.jina_enabled=True
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:ids=list(map(str,z['query_ids'].tolist()));vectors=normalize(z['vectors'])
    qrow={q:i for i,q in enumerate(ids)};questions=data.questions;all_predictions={};baseline={};fold_reports={}
    configs={'l7_t30':dict(num_leaves=7,min_child_samples=50,lambdarank_truncation_level=30),'l15_t5':dict(num_leaves=15,min_child_samples=50,lambdarank_truncation_level=5)};system_predictions={name:{} for name in configs}
    for outer in range(5):
        (OUT/f'fold_{outer}').mkdir(parents=True,exist_ok=True)
        marker=read(LEGACY_CACHE/f'outer/fold_{outer}/outer-ml.json');train_qids=[q for q in map(str,marker['training_qids']) if labels.get(q)];test_qids=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        base_matrix=np.load(MEMORY/f'fold_{outer}'/'train_augmented.f32.npy',mmap_mode='r');groups=[];train_docs=[];y=[]
        for q in train_qids:
            docs=list(dict.fromkeys(store.candidates(q)+sorted(labels[q])));groups.append(len(docs));train_docs.append(docs);y.extend(d in labels[q] for d in docs)
        if sum(groups)!=len(base_matrix):raise ValueError('base matrix mismatch')
        train_fold_models={f:build_profiles(questions,labels,[q for q in train_qids if fold_of[q]!=f]) for f in sorted(set(fold_of[q] for q in train_qids))}
        extra=np.lib.format.open_memmap(OUT/f'fold_{outer}'/'train_profile.f32.npy',mode='w+',dtype=np.float32,shape=(len(base_matrix),12));offset=0
        for qi,(q,docs) in enumerate(zip(train_qids,train_docs)):
            extra[offset:offset+len(docs)]=profile_features(questions[q],docs,train_fold_models[fold_of[q]]);offset+=len(docs)
            if (qi+1)%1000==0:print(f'outer={outer} train_profile={qi+1}/{len(train_qids)}',flush=True)
        extra.flush();x=np.lib.format.open_memmap(OUT/f'fold_{outer}'/'train_combined.f32.npy',mode='w+',dtype=np.float32,shape=(len(base_matrix),base_matrix.shape[1]+12));x[:,:base_matrix.shape[1]]=base_matrix;x[:,base_matrix.shape[1]:]=extra;x.flush();del extra
        by_doc,frequency=support_index(labels,train_qids);support=vectors[[qrow[q] for q in train_qids]];sim=np.asarray(vectors[[qrow[q] for q in test_qids]]@support.T,dtype=np.float32);test_model=build_profiles(questions,labels,train_qids);test_docs=[];test_groups=[];test_rows=[]
        for qi,q in enumerate(test_qids):
            docs=store.candidates(q);xb=np.asarray(features(data,store,q,docs,2));xm=memory_features(sim[qi],docs,train_qids,labels,by_doc,frequency);xp=profile_features(questions[q],docs,test_model);test_rows.append(np.concatenate([xb,xm,xp],axis=1));test_docs.append(docs);test_groups.append(len(docs))
        tx=np.concatenate(test_rows);ends=np.cumsum([0]+test_groups);legacy=read(MEMORY/f'fold_{outer}'/'PREDICTIONS.json');baseline.update(legacy);fold_reports[f'fold_{outer}']={}
        for name,config in configs.items():
            model=lgb.LGBMRanker(objective='lambdarank',learning_rate=.05,n_estimators=300,feature_fraction=1.,bagging_fraction=1.,deterministic=True,force_col_wise=True,n_jobs=4,random_state=4112,verbosity=-1,**config);model.fit(x,np.asarray(y,dtype=np.int8),group=groups,eval_at=[5]);score=model.predict(tx);pred={}
            for qi,(q,docs) in enumerate(zip(test_qids,test_docs)):
                local=score[ends[qi]:ends[qi+1]];pred[q]=[docs[i] for i in sorted(range(len(docs)),key=lambda i:(-float(local[i]),docs[i]))]
            system_predictions[name].update(pred);fold_reports[f'fold_{outer}'][name]=metrics(pred,labels,test_qids);print(f'outer={outer} profile_ltr={name} recall={fold_reports[f"fold_{outer}"][name]["recall_at_5"]:.9f}',flush=True)
        del x,tx,base_matrix
    qids=[q for q in baseline if labels.get(q)];base_metrics=metrics(baseline,labels,qids);aggregate=[]
    for name,pred in system_predictions.items():
        score=metrics(pred,labels,qids);deltas=[fold_reports[f'fold_{f}'][name]['recall_at_5']-metrics(baseline,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)])['recall_at_5'] for f in range(5)];aggregate.append(dict(system=name,config=configs[name],metrics=score,delta=score['recall_at_5']-base_metrics['recall_at_5'],fold_deltas=deltas,nonnegative_folds=sum(d>=0 for d in deltas)));write(OUT/name/'PREDICTIONS.json',pred)
    aggregate.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True);report=dict(status='COMPLETE_PROFILE_MEMORY_LTR_PROBE',scope_warning='Strict outer feature isolation; architecture inspected on exposed development OOF.',baseline=base_metrics,profile_configs=[list(c) for c in PROFILE_CONFIGS],folds=fold_reports,aggregate=aggregate);write(OUT/'PROFILE_LTR_REPORT.json',report);store.close();print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
