"""Bounded Fold-4 OOF probe for cleaner E5 query-adaptation objectives.

The target fold is Fold 4.  All training uses Folds 0--3, so every reported
prediction is label-isolated from the target.  This probe asks whether noisy
multi-gold supervision, rather than the adapter architecture itself, limits
the useful retrieval gain observed in EXP-112.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/exp_final_retrieval/adapter_objective_probe'
CACHE=ROOT/'cache/exp_final_retrieval/adapter_objective_probe'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def metrics(rankings,labels,qids):
    values=[];precision=[];single=[];multi=[];mrr=[]
    for q in qids:
        gold=labels.get(q,set())
        if not gold:continue
        top=rankings[q][:5];hits=len(set(top)&gold);v=hits/len(gold);values.append(v);precision.append(hits/5)
        if len(gold)>1:multi.append(v)
        else:single.append(v)
        first=next((i for i,d in enumerate(top,1) if d in gold),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),single_gold_recall_at_5=float(np.mean(single)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))
def compact_score(data,qids,checkpoint):
    from exp_final.learning import QueryEncoder
    model=QueryEncoder(checkpoint=checkpoint);model.eval();bank=data.bank('e5',device='cuda');rankings={};cosines=[];started=time.time()
    try:
        with torch.no_grad():
            for begin in range(0,len(qids),4):
                batch=qids[begin:begin+4];vectors=model([data.questions[q] for q in batch]);scores,_=bank.mine(vectors);values,indices=torch.topk(scores,k=200,dim=1,largest=True,sorted=False)
                for q,row_scores,row_indices,vector in zip(batch,values.cpu().numpy(),indices.cpu().numpy(),vectors.cpu().numpy()):
                    pairs=sorted(zip(row_scores.tolist(),row_indices.tolist()),key=lambda pair:(-float(pair[0]),data.doc_ids[pair[1]]));rankings[q]=[data.doc_ids[index] for _,index in pairs];cosines.append(float(vector@data.query_vector(q,'e5')))
                if begin%200==0:print(f'score {Path(checkpoint).parent.name}/{Path(checkpoint).stem} {min(begin+len(batch),len(qids))}/{len(qids)} elapsed={time.time()-started:.1f}s',flush=True)
    finally:
        del model,bank;import gc;gc.collect();torch.cuda.empty_cache()
    return rankings,dict(mean=float(np.mean(cosines)),p05=float(np.quantile(cosines,.05)),minimum=float(np.min(cosines)))
def rrf(base,adapted,weight):
    rb={d:i for i,d in enumerate(base,1)};ra={d:i for i,d in enumerate(adapted,1)};docs=set(rb)|set(ra);scores={d:(1-weight)/(32+rb[d]) if d in rb else 0. for d in docs}
    for d in docs:
        if d in ra:scores[d]+=weight/(32+ra[d])
    return sorted(docs,key=lambda d:(-scores[d],d))
def choice_oracle(left,right,labels,qids):
    selected={}
    wins={'left':0,'right':0,'tie':0}
    for q in qids:
        gold=labels.get(q,set())
        left_hits=len(set(left[q][:5])&gold)
        right_hits=len(set(right[q][:5])&gold)
        if right_hits>left_hits:selected[q]=right[q];wins['right']+=1
        elif left_hits>right_hits:selected[q]=left[q];wins['left']+=1
        else:selected[q]=left[q];wins['tie']+=1
    return dict(metrics=metrics(selected,labels,qids),query_choices=wins)
def main():
    sys.path.insert(0,str(ROOT/'src'));import exp109b_encoder_complementarity as old
    from exp_final.data import Data,SourceStore
    from exp_final.learning import train_query
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');data=Data();store=SourceStore(ROOT/'cache/exp112_task_adaptive_retrieval/sources.sqlite');store.jina_enabled=True
    test=[q for q in folds['fold_4'] if labels.get(q)];train=[q for fold in range(4) for q in folds[f'fold_{fold}'] if labels.get(q)]
    frozen={q:[row['doc_id'] for row in store.get(q,'e5')] for q in test};memory=read(ROOT/'results/exp_final_retrieval/memory_ltr_probe/fold_4/PREDICTIONS.json')
    adapted_cache=read(ROOT/'results/exp_final_retrieval/meta_ltr_probe/adapted_e5_top64.json');current={q:adapted_cache[q] for q in test}
    systems={'frozen_e5':metrics(frozen,labels,test),'memory':metrics(memory,labels,test),'exp112_all_positive_epoch2':metrics(current,labels,test)};trials=[];oracles={}
    oracles['memory_vs_exp112_all_positive_epoch2']=choice_oracle(memory,current,labels,test)
    for weight in (0.,.05,.10,.15,.20,.30,.50):
        fused={q:rrf(memory[q],current[q],weight) for q in test};value=metrics(fused,labels,test)
        trials.append(dict(system='exp112_all_positive_epoch2',weight=weight,metrics=value,delta_vs_memory=value['recall_at_5']-systems['memory']['recall_at_5']))
    for policy in ('single_only','content_primary'):
        folder=CACHE/policy
        train_query(data,store,train,folder,epochs=2,nominal_epochs=2,microbatch=4,positive_policy=policy,learning_rate=5e-5)
        for epoch in (1,2):
            prediction_path=OUT/policy/f'epoch-{epoch}-predictions.json'
            if prediction_path.exists():ranked=read(prediction_path);cosine=read(OUT/policy/f'epoch-{epoch}-cosine.json')
            else:
                ranked,cosine=compact_score(data,test,folder/f'epoch-{epoch}.pt');write(prediction_path,ranked);write(OUT/policy/f'epoch-{epoch}-cosine.json',cosine)
            name=f'{policy}_epoch{epoch}';systems[name]=metrics(ranked,labels,test)
            oracles[f'memory_vs_{name}']=choice_oracle(memory,ranked,labels,test)
            for weight in (0.,.05,.10,.15,.20,.30,.50):
                fused={q:rrf(memory[q],ranked[q],weight) for q in test};value=metrics(fused,labels,test);trials.append(dict(system=name,weight=weight,metrics=value,delta_vs_memory=value['recall_at_5']-systems['memory']['recall_at_5']))
            print(name,json.dumps(systems[name]),flush=True)
    trials.sort(key=lambda row:(row['metrics']['recall_at_5'],row['metrics']['precision_at_5'],row['metrics']['mrr_at_5']),reverse=True)
    report=dict(status='COMPLETE_ADAPTER_OBJECTIVE_FOLD4_PROBE',scope='Fold 4 OOF; trained on Folds 0-3; exposed development fold.',training_queries=len(train),test_queries=len(test),systems=systems,choice_oracles=oracles,top_fusions=trials[:30]);write(OUT/'ADAPTER_OBJECTIVE_REPORT.json',report);store.close();print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
