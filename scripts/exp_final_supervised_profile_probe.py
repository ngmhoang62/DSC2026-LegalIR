"""Fold-isolated supervised BM25 label-profile retrieval.

Adapted from the teammate repro's supervised profile idea, but evaluated on the
project's canonical five-fold contract.  Each held-out fold is excluded from
profile construction and label priors.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MEMORY=ROOT/'results/exp_final_retrieval/memory_ltr_probe'
OUT=ROOT/'results/exp_final_retrieval/supervised_profile_probe'
TOKEN=re.compile(r'\w+',re.UNICODE)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def tokens(text):return TOKEN.findall((text or '').lower())
def feats(text,max_ngram=3):
    words=tokens(text);return {f'{n}:'+ ' '.join(words[i:i+n]) for n in range(1,max_ngram+1) for i in range(len(words)-n+1)}
def build_profiles(questions,labels,qids):
    postings=defaultdict(Counter);doc_length=Counter();frequency=Counter()
    for q in qids:
        values=feats(questions[q],3)
        for doc in labels[q]:
            frequency[doc]+=1
            for value in values:postings[value][doc]+=1;doc_length[doc]+=1
    return postings,{value:len(docs) for value,docs in postings.items()},doc_length,frequency
def rank_profile(text,model,max_ngram,k1,b,prior_power,depth=100):
    postings,df,length,frequency=model;count=max(len(length),1);avg=sum(length.values())/count;scores=defaultdict(float)
    for value in feats(text,max_ngram):
        posting=postings.get(value)
        if not posting:continue
        idf=math.log1p((count-df[value]+.5)/(df[value]+.5));order=int(value[0]);phrase=(1.,1.35,1.65)[order-1]
        for doc,tf in posting.items():
            norm=k1*(1-b+b*length[doc]/avg);scores[doc]+=phrase*idf*tf*(k1+1)/(tf+norm)
    maximum=max(frequency.values(),default=1)
    for doc in scores:scores[doc]*=((frequency[doc]+.5)/(maximum+.5))**prior_power
    return sorted(scores,key=lambda d:(-scores[d],d))[:depth]
def fuse(a,b,w,k):
    ar={d:i for i,d in enumerate(a,1)};br={d:i for i,d in enumerate(b,1)};docs=set(ar)|set(br)
    score={d:(1-w)/(k+ar[d]) if d in ar else 0 for d in docs}
    for d in br:score[d]+=w/(k+br[d])
    return sorted(docs,key=lambda d:(-score[d],d))
def metrics(pred,labels,qids):
    values=[];multi=[];precision=[];mrr=[]
    for q in qids:
        g=labels.get(q,set())
        if not g:continue
        top=pred[q][:5];h=len(set(top)&g);v=h/len(g);values.append(v);precision.append(h/5)
        if len(g)>1:multi.append(v)
        first=next((i for i,d in enumerate(top,1) if d in g),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))
def main():
    import sys;sys.path.insert(0,str(ROOT/'src'));import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');raw=read(ROOT/'public_test_dataset/train.json');questions={q:r['question'] for q,r in raw.items()}
    base={}
    for f in range(5):base.update(read(MEMORY/f'fold_{f}'/'PREDICTIONS.json'))
    configs=[(n,1.2,.75,p) for n in (1,2,3) for p in (-.3,0.,.3)]
    profiles={config:{} for config in configs};fold_profile_metrics={config:{} for config in configs}
    for f in range(5):
        support=[q for j in range(5) if j!=f for q in folds[f'fold_{j}'] if labels.get(q)];test=[q for q in folds[f'fold_{f}'] if labels.get(q)];model=build_profiles(questions,labels,support)
        for config in configs:
            pred={q:rank_profile(questions[q],model,*config) for q in test};profiles[config].update(pred);fold_profile_metrics[config][f'fold_{f}']=metrics(pred,labels,test)
        print(f'profile fold={f} complete support_docs={len(model[2])}',flush=True)
    qids=[q for f in range(5) for q in folds[f'fold_{f}'] if labels.get(q)];baseline=metrics(base,labels,qids);trials=[]
    profile_scores={config:metrics(profile,labels,qids) for config,profile in profiles.items()}
    # The original 270-way Python ranking loop was needlessly expensive.  The
    # profile itself is the hypothesis; keep only its three strongest OOF
    # configurations for a bounded fusion screen.
    selected=sorted(configs,key=lambda c:(profile_scores[c]['recall_at_5'],profile_scores[c]['precision_at_5'],profile_scores[c]['mrr_at_5']),reverse=True)[:3]
    base_per={f'fold_{f}':metrics(base,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)]) for f in range(5)}
    for config in selected:
        profile=profiles[config];standalone=profile_scores[config]
        for k in (0,10,32):
            for w in (.01,.02,.03,.05,.075,.10,.15,.20,.30,.50):
                pred={q:fuse(base[q],profile[q],w,k) for q in qids};score=metrics(pred,labels,qids);per={f'fold_{f}':metrics(pred,labels,[q for q in folds[f'fold_{f}'] if labels.get(q)]) for f in range(5)}
                trials.append(dict(config={'max_ngram':config[0],'k1':config[1],'b':config[2],'prior_power':config[3]},rrf_k=k,weight=w,standalone=standalone,metrics=score,delta=score['recall_at_5']-baseline['recall_at_5'],multi_delta=score['multi_gold_recall_at_5']-baseline['multi_gold_recall_at_5'],nonnegative_folds=sum(per[f'fold_{f}']['recall_at_5']>=base_per[f'fold_{f}']['recall_at_5'] for f in range(5)),per_fold=per))
    trials.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True)
    oracle=[]
    for q in qids:
        g=labels[q];oracle.append(max([len(set(base[q][:5])&g)/len(g)]+[len(set(profiles[c][q][:5])&g)/len(g) for c in configs]))
    report=dict(status='COMPLETE_SUPERVISED_PROFILE_PROBE',scope_warning='Fold-isolated labels, bounded configs inspected on exposed development OOF.',baseline=baseline,profile_metrics={str(c):profile_scores[c] for c in configs},selected_fusion_configs=[str(c) for c in selected],choice_oracle_recall_at_5=float(np.mean(oracle)),top_trials=trials[:40])
    write(OUT/'SUPERVISED_PROFILE_REPORT.json',report);best=trials[0];write(OUT/'BEST_PROFILE_PREDICTIONS.json',profiles[(best['config']['max_ngram'],best['config']['k1'],best['config']['b'],best['config']['prior_power'])]);print(json.dumps({'status':report['status'],'baseline':baseline,'profiles':report['profile_metrics'],'choice_oracle_recall_at_5':report['choice_oracle_recall_at_5'],'top5':trials[:5]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
