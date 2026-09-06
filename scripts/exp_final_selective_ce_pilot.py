"""Fold-isolated selective cross-encoder pilot at the top-five boundary."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

import exp_final_nested_slate_probe as common  # noqa:E402
from exp_final.contracts import digest,read,seed_all,sha,write  # noqa:E402
from exp_final.cross_encoder import CrossEncoder  # noqa:E402
from exp_final.data import Data  # noqa:E402
from exp_final.evidence import Evidence  # noqa:E402
from exp_final.learning import checkpoint,set_rng  # noqa:E402

OUT=ROOT/'results/exp_final_retrieval/selective_ce_pilot'
TRAIN_FOLDS=(0,1,2);CAL_FOLD=3;TEST_FOLD=4;ACCUMULATION_QUERIES=8;PAIR_MICROBATCH=4


def base_rankings():
    xgb=common.read(ROOT/'results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json')
    lgb=common.read(ROOT/'results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json')
    profile=common.read(ROOT/'results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json')
    return common.blend([xgb,lgb,profile])


def training_pairs(evidence,data,base,qid):
    positives=sorted(data.gold[qid]);negatives=[doc for doc in base[qid][4:10] if doc not in data.gold[qid]]
    if not negatives:negatives=[doc for doc in base[qid][:20] if doc not in data.gold[qid]][:6]
    pairs=[evidence.package(qid,doc) for doc in positives+negatives]
    weights=np.asarray([.5/max(1,len(positives))]*len(positives)+[.5/max(1,len(negatives))]*len(negatives),dtype=np.float32)
    targets=np.asarray([1]*len(positives)+[0]*len(negatives),dtype=np.float32)
    return pairs,weights,targets


def train(data,base,qids,directory,max_queries=None):
    from transformers import get_cosine_schedule_with_warmup
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);qids=[q for q in qids if data.gold[q]]
    order=list(qids);random.Random(112).shuffle(order)
    if max_queries:order=order[:max_queries]
    signature=digest(['selective-ce-balanced-boundary-v1',qids,max_queries,sha(Path(__file__))])
    success=directory/'_SUCCESS.json'
    if success.exists():
        value=read(success)
        if value['signature']!=signature or value['model_sha256']!=sha(directory/'model.pt'):raise ValueError('CE success artifact mismatch')
        return value
    seed_all();model=CrossEncoder();model.train();evidence=Evidence(data,model.tokenizer)
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.01)
    updates=math.ceil(len(order)/ACCUMULATION_QUERIES);scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(.1*updates)),updates)
    resume=directory/'resume.pt';position=0
    if resume.exists():
        receipt=resume.with_suffix('.sha.json')
        if not receipt.exists() or read(receipt)['sha256']!=sha(resume):raise ValueError('Unverified CE resume')
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise ValueError('CE resume scope mismatch')
        model.load_state_dict(state['adapter'],strict=False);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);set_rng(state['rng']);position=state['position']
    started=time.monotonic();pairs_seen=0;loss_sum=0.;optimizer.zero_grad(set_to_none=True)
    try:
        for index in range(position,len(order)):
            pairs,weights,targets=training_pairs(evidence,data,base,order[index]);query_loss=0.
            for start in range(0,len(pairs),PAIR_MICROBATCH):
                local=pairs[start:start+PAIR_MICROBATCH];logits=model(local);target=torch.as_tensor(targets[start:start+PAIR_MICROBATCH],device='cuda');weight=torch.as_tensor(weights[start:start+PAIR_MICROBATCH],device='cuda')
                loss=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/ACCUMULATION_QUERIES
                loss.backward();query_loss+=float(loss.detach())*ACCUMULATION_QUERIES;pairs_seen+=len(local)
            loss_sum+=query_loss
            boundary=(index+1)%ACCUMULATION_QUERIES==0 or index+1==len(order)
            if boundary:
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if boundary and ((index+1)%256==0 or index+1==len(order)):
                checkpoint(resume,model,optimizer,scheduler,signature=signature,position=index+1)
            if (index+1)%16==0 or index+1==len(order):
                elapsed=time.monotonic()-started;done=index+1-position;eta=elapsed/max(1,done)*(len(order)-index-1)
                print(f'train={index+1}/{len(order)} pairs={pairs_seen} loss={query_loss:.5f} qps={done/elapsed:.3f} eta_s={eta:.0f} vram={torch.cuda.max_memory_reserved()/2**30:.2f}GiB',flush=True)
        checkpoint(directory/'model.pt',model,optimizer,scheduler,signature=signature,position=len(order))
        result=dict(status='COMPLETE_CE_TRAIN',signature=signature,queries=len(order),pairs=pairs_seen,seconds=time.monotonic()-started,seconds_per_query=(time.monotonic()-started)/len(order),peak_vram=int(torch.cuda.max_memory_reserved()),model_sha256=sha(directory/'model.pt'),benchmark_only=bool(max_queries))
        write(success,result);return result
    finally:
        evidence.db.close();del evidence,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()


@torch.inference_mode()
def score(data,base,qids,model_path,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);model=CrossEncoder(model_path);model.eval();evidence=Evidence(data,model.tokenizer);result={};started=time.monotonic()
    try:
        for index,qid in enumerate(qids):
            path=directory/f'{qid}.json';docs=base[qid][:10];signature=digest([sha(model_path),qid,docs,'top10-v1'])
            if path.exists():
                row=read(path)
                if row['signature']!=signature:raise ValueError('CE score resume mismatch')
                result[qid]=row['scores'];continue
            values=[]
            for start in range(0,10,4):values.extend(model([evidence.package(qid,doc) for doc in docs[start:start+4]]).cpu().tolist())
            row={'signature':signature,'docs':docs,'scores':list(map(float,values))};write(path,row);result[qid]=row['scores']
            if (index+1)%32==0 or index+1==len(qids):print(f'score={index+1}/{len(qids)} qps={(index+1)/(time.monotonic()-started):.3f}',flush=True)
        return result
    finally:
        evidence.db.close();del evidence,model;gc.collect();torch.cuda.empty_cache()


def policy_ranking(order,ce_scores,kind,value):
    docs=order[:10];ce=np.asarray(ce_scores,dtype=np.float64)
    if kind=='fusion':
        base=-np.arange(1,11,dtype=np.float64);base=(base-base.mean())/base.std();normalized=(ce-ce.mean())/max(ce.std(),1e-12);score=(1-value)*base+value*normalized
        head=[docs[i] for i in sorted(range(10),key=lambda i:(-float(score[i]),docs[i]))];return head+order[10:]
    if kind=='replace':
        margin=(ce[5:].max()-ce[4])/max(ce.std(),1e-12)
        if margin<=value:return list(order)
        candidate=5+int(np.argmax(ce[5:]));head=list(order);doc=head.pop(candidate);head.insert(4,doc);return head
    if kind=='noop':return list(order)
    raise ValueError(kind)


def evaluate_policies(base,scores,labels,qids):
    policies=[('noop',0.)]+[('fusion',alpha) for alpha in (.05,.10,.20,.30)]+[('replace',threshold) for threshold in (0.,.25,.5,1.,2.)]
    rows=[]
    for policy in policies:
        pred={q:policy_ranking(base[q],scores[q],*policy) for q in qids};value=common.metrics(pred,labels,qids);actions=sum(pred[q][:5]!=base[q][:5] for q in qids)
        rows.append({'policy':{'kind':policy[0],'value':policy[1]},'metrics':value,'actions':actions,'predictions':pred})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true');args=parser.parse_args()
    import exp109b_encoder_complementarity as old
    data=Data();labels,_=old.canonical_labels();folds=common.read(ROOT/'cache/cv_folds.json');base=base_rankings();train_qids=[q for fold in TRAIN_FOLDS for q in folds[f'fold_{fold}'] if labels.get(q)]
    if args.benchmark_only:
        result=train(data,base,train_qids,OUT/'benchmark64_b4',max_queries=64);print(json.dumps(result,indent=2),flush=True);return
    training=train(data,base,train_qids,OUT/'model_b4')
    cal_qids=[q for q in folds[f'fold_{CAL_FOLD}'] if labels.get(q)];test_qids=[q for q in folds[f'fold_{TEST_FOLD}'] if labels.get(q)]
    cal_scores=score(data,base,cal_qids,OUT/'model_b4/model.pt',OUT/'scores_cal_b4');test_scores=score(data,base,test_qids,OUT/'model_b4/model.pt',OUT/'scores_test_b4')
    trials=evaluate_policies(base,cal_scores,labels,cal_qids);winner=max(trials,key=lambda row:common.selection_key(row['metrics'],row['actions']));lock={'train_folds':TRAIN_FOLDS,'calibration_fold':CAL_FOLD,'test_fold':TEST_FOLD,'pair_microbatch':PAIR_MICROBATCH,'policy':winner['policy'],'calibration_metrics':winner['metrics'],'calibration_actions':winner['actions'],'model_sha256':sha(OUT/'model_b4/model.pt')};common.write(OUT/'SELECTION_LOCK.json',lock)
    predictions={q:policy_ranking(base[q],test_scores[q],lock['policy']['kind'],lock['policy']['value']) for q in test_qids};common.write(OUT/'FOLD_4_PREDICTIONS_LOCK.json',predictions)
    baseline=common.metrics(base,labels,test_qids);value=common.metrics(predictions,labels,test_qids);events=[q for q in test_qids if predictions[q][:5]!=base[q][:5]]
    outcomes={'wins':sum(len(set(predictions[q][:5])&labels[q])>len(set(base[q][:5])&labels[q]) for q in events),'losses':sum(len(set(predictions[q][:5])&labels[q])<len(set(base[q][:5])&labels[q]) for q in events),'neutral':sum(len(set(predictions[q][:5])&labels[q])==len(set(base[q][:5])&labels[q]) for q in events)}
    report={'status':'COMPLETE_SELECTIVE_CE_FOLD4_PILOT','training':training,'selection_lock':lock,'base_fold4':baseline,'selective_ce_fold4':value,'delta':value['recall_at_5']-baseline['recall_at_5'],'actions':len(events),'outcomes':outcomes,'calibration_trials':[{k:v for k,v in row.items() if k!='predictions'} for row in trials]};common.write(OUT/'SELECTIVE_CE_FOLD4_REPORT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
