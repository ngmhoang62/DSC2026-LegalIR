"""Add fold-isolated semantic case-memory features to the EXP-112 frozen LTR.

The probe reuses hash-checked immutable 72D feature matrices. For every training
query its own labels are removed from the support memory before features are
computed, preventing the trivial self-match leak.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CACHE = ROOT / "cache/exp112_task_adaptive_retrieval"
LEGACY_RESULTS = ROOT / "results/exp112_task_adaptive_retrieval"
OUT = ROOT / "results/exp_final_retrieval/memory_ltr_probe"
MEMORY_NAMES = (
    "memory_seen", "memory_support_count", "memory_log_support_count",
    "memory_max_similarity", "memory_second_similarity", "memory_top2_mean",
    "memory_soft_vote", "memory_frequency_vote", "memory_vote_recip_rank",
    "memory_query_nearest", "memory_query_neighbor_margin", "memory_vote_entropy",
    "memory_vote_winner_margin", "memory_top_neighbor_agreement",
)


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


def support_index(labels,qids):
    by_doc=defaultdict(list)
    for i,q in enumerate(qids):
        for d in labels.get(q,()):by_doc[d].append(i)
    return by_doc,Counter(d for q in qids for d in labels.get(q,()))


def memory_features(similarities, docs, support_qids, labels, by_doc, frequency, self_qid=None, self_index=None):
    similarities=np.asarray(similarities,dtype=np.float32)
    if self_index is not None:
        similarities=similarities.copy();similarities[self_index]=-np.inf
    take=min(16,len(similarities));part=np.argpartition(-similarities,take-1)[:take]
    neighbours=sorted(part.tolist(),key=lambda i:(-float(similarities[i]),support_qids[i]))
    votes=defaultdict(float);nearest=float(similarities[neighbours[0]]);second_near=float(similarities[neighbours[1]]) if len(neighbours)>1 else nearest
    neighbor_labels=[]
    for i in neighbours:
        q=support_qids[i];gold=labels.get(q,());neighbor_labels.append(set(gold));aff=math.exp(20*(float(similarities[i])-1))/max(1,len(gold))
        for d in gold:votes[d]+=aff/max(1,frequency[d]-(1 if self_qid and d in labels.get(self_qid,()) else 0))
    vote_order=sorted(votes,key=lambda d:(-votes[d],d));vote_rank={d:i for i,d in enumerate(vote_order,1)}
    positive=np.asarray([v for v in votes.values() if v>0],dtype=np.float64);entropy=0.
    if len(positive)>1:
        p=positive/positive.sum();entropy=float(-(p*np.log(p)).sum()/math.log(len(p)))
    ordered=sorted(votes.values(),reverse=True);winner_margin=float(ordered[0]-(ordered[1] if len(ordered)>1 else 0)) if ordered else 0.
    counts=Counter(d for g in neighbor_labels for d in g);agreement=max(counts.values(),default=0)/max(1,len(neighbours))
    out=np.empty((len(docs),len(MEMORY_NAMES)),dtype=np.float32)
    self_gold=labels.get(self_qid,set()) if self_qid else set()
    for row,d in enumerate(docs):
        indices=[i for i in by_doc.get(d,()) if support_qids[i]!=self_qid]
        local=sorted((float(similarities[i]) for i in indices if np.isfinite(similarities[i])),reverse=True)
        count=max(0,frequency[d]-(1 if d in self_gold else 0));seen=float(bool(local))
        maximum=local[0] if local else -1.;second=local[1] if len(local)>1 else -1.;top2=float(np.mean(local[:2])) if local else -1.
        out[row]=[seen,count,math.log1p(count),maximum,second,top2,float(votes.get(d,0)),float(votes.get(d,0)),
                  1/vote_rank[d] if d in vote_rank else 0,nearest,nearest-second_near,entropy,winner_margin,agreement]
    return out


def run_outer(outer, variant='lal'):
    import sys,time
    sys.path.insert(0,str(ROOT/'src'))
    import lightgbm as lgb
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data,SourceStore
    from exp_final.fusion import features
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');data=Data()
    store=SourceStore(LEGACY_CACHE/'sources.sqlite');store.jina_enabled=True
    marker=read(LEGACY_CACHE/f'outer/fold_{outer}/outer-ml.json')
    marker_qids=list(map(str,marker['training_qids']))
    # EXP-112 records the complete outer-training scope in the marker but its
    # matrix builder skips canonically non-evaluable query groups.
    train_qids=[q for q in marker_qids if labels.get(q)]
    test_qids=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
    with np.load(ROOT/'cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz',allow_pickle=False) as z:
        ids=list(map(str,z['query_ids'].tolist()));lal_vectors=normalize(z['vectors'])
    qrow={q:i for i,q in enumerate(ids)}
    dense_representations={'lal':lal_vectors}
    if variant=='multi':
        e5_dir=ROOT/'cache/exp021_e5_dense_candidates/query_embeddings'
        e5_ids=list(map(str,read(e5_dir/'train_query_ids.json')));e5_values=normalize(np.load(e5_dir/'train_queries.f32.npy',mmap_mode='r'))
        e5_map={q:i for i,q in enumerate(e5_ids)}
        dense_representations['e5']=e5_values[[e5_map[q] for q in ids]]
    representation_names=list(dense_representations)+( ['char35'] if variant=='multi' else [] )
    by_doc,frequency=support_index(labels,train_qids)
    base_matrix=np.load(LEGACY_CACHE/f'outer/fold_{outer}/outer-ml.matrix.npy',mmap_mode='r')
    groups=[];all_docs=[];y=[]
    for q in train_qids:
        docs=list(dict.fromkeys(store.candidates(q)+sorted(labels[q])));groups.append(len(docs));all_docs.append(docs);y.extend(d in labels[q] for d in docs)
    if sum(groups)!=len(base_matrix):raise ValueError('Legacy matrix row contract mismatch')
    directory=OUT/(variant if variant!='lal' else 'lal')/f'fold_{outer}'
    target=directory/'train_augmented.f32.npy';target.parent.mkdir(parents=True,exist_ok=True)
    x=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(base_matrix),base_matrix.shape[1]+len(MEMORY_NAMES)*len(representation_names)))
    x[:,:base_matrix.shape[1]]=base_matrix
    test_similarities={};started=time.time()
    train_indices=[qrow[q] for q in train_qids];test_indices=[qrow[q] for q in test_qids]
    for ri,name in enumerate(representation_names):
        if name=='char35':
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=200000,sublinear_tf=True,norm='l2',dtype=np.float32)
            train_text=[data.questions[q] for q in train_qids];test_text=[data.questions[q] for q in test_qids]
            support=vectorizer.fit_transform(train_text);test_matrix=vectorizer.transform(test_text)
            train_sim=np.asarray((support@support.T).toarray(),dtype=np.float32)
            test_similarities[name]=np.asarray((test_matrix@support.T).toarray(),dtype=np.float32)
        else:
            vectors=dense_representations[name];support=vectors[train_indices]
            train_sim=np.asarray(support@support.T,dtype=np.float32)
            test_similarities[name]=np.asarray(vectors[test_indices]@support.T,dtype=np.float32)
        offset=0;column=base_matrix.shape[1]+ri*len(MEMORY_NAMES)
        for qi,(q,docs) in enumerate(zip(train_qids,all_docs)):
            n=len(docs);x[offset:offset+n,column:column+len(MEMORY_NAMES)]=memory_features(train_sim[qi],docs,train_qids,labels,by_doc,frequency,self_qid=q,self_index=qi);offset+=n
            if (qi+1)%800==0:print(f'outer={outer} rep={name} train_features={qi+1}/{len(train_qids)} elapsed={time.time()-started:.1f}s',flush=True)
        del train_sim
    x.flush()
    model=lgb.LGBMRanker(objective='lambdarank',num_leaves=7,min_child_samples=50,learning_rate=.05,n_estimators=300,
                        feature_fraction=1.,bagging_fraction=1.,deterministic=True,force_col_wise=True,n_jobs=4,random_state=113,verbosity=-1)
    model.fit(x,np.asarray(y,dtype=np.int8),group=groups,eval_at=[5]);del x
    ranked={};baseline={};prototype={}
    legacy_predictions=read(LEGACY_RESULTS/f'outer/fold_{outer}/PREDICTIONS.json')
    for qi,q in enumerate(test_qids):
        docs=store.candidates(q);xb=np.asarray(features(data,store,q,docs,2));memory_blocks=[memory_features(test_similarities[name][qi],docs,train_qids,labels,by_doc,frequency) for name in representation_names]
        xm=memory_blocks[0];score=model.predict(np.concatenate([xb,*memory_blocks],axis=1));order=[docs[i] for i in sorted(range(len(docs)),key=lambda i:(-float(score[i]),docs[i]))]
        ranked[q]=order;baseline[q]=legacy_predictions[q]['order']
        vote_order=sorted(range(len(docs)),key=lambda i:(-float(xm[i,7]),docs[i]));prototype[q]=[docs[i] for i in vote_order]
        if (qi+1)%300==0:print(f'outer={outer} score={qi+1}/{len(test_qids)}',flush=True)
    expanded_names=marker['feature_names']+[f'{rep}_{name}' for rep in representation_names for name in MEMORY_NAMES]
    result=dict(status='COMPLETE_MEMORY_LTR_PROBE',outer=f'fold_{outer}',variant=variant,representations=representation_names,feature_names=expanded_names,
                baseline=metrics(baseline,labels,test_qids),memory_ltr=metrics(ranked,labels,test_qids),prototype_in_pool=metrics(prototype,labels,test_qids))
    result['delta']=result['memory_ltr']['recall_at_5']-result['baseline']['recall_at_5']
    write(directory/'REPORT.json',result);write(directory/'PREDICTIONS.json',ranked);store.close();print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outer',default='0',choices=['0','1','2','3','4','all']);parser.add_argument('--variant',default='lal',choices=['lal','multi']);args=parser.parse_args()
    folds=range(5) if args.outer=='all' else [int(args.outer)];reports=[run_outer(f,args.variant) for f in folds]
    if len(reports)==5:
        write(OUT/args.variant/'SUMMARY.json',dict(status='COMPLETE_MEMORY_LTR_5FOLD',variant=args.variant,mean_delta=float(np.mean([r['delta'] for r in reports])),reports=reports))


if __name__=='__main__':main()
