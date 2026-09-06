"""Strict nested residual Set-Transformer over the top-ten candidate slate."""
from __future__ import annotations

import copy
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

import exp_final_nested_slate_probe as common  # noqa:E402
from exp_final.data import Data  # noqa:E402
from exp_final.set_ranker import ResidualSetRanker,multi_positive_set_loss  # noqa:E402

OUT=ROOT/'results/exp_final_retrieval/nested_set_ranker_probe';CACHE=ROOT/'cache/exp_final_retrieval/nested_set_ranker_probe'
EPOCHS=(5,10,15);ALPHAS=(0.,.10,.25,.50,.75,1.)


def load_systems():
    memory={}
    for fold in range(5):memory.update(common.read(ROOT/f'results/exp_final_retrieval/memory_ltr_probe/fold_{fold}/PREDICTIONS.json'))
    return {
        'xgb':common.read(ROOT/'results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json'),
        'lgb':common.read(ROOT/'results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json'),
        'profile':common.read(ROOT/'results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json'),
        'memory':memory,
        'kernel':common.read(ROOT/'results/exp_final_retrieval/kernel_ltr_probe/l15_t5/PREDICTIONS.json'),
    }


def materialize(qids,folds,base,systems):
    path=CACHE/'top10_features_143d.f32.npy';ids_path=CACHE/'qids.json';manifest_path=CACHE/'manifest.json'
    inputs=[ROOT/f'cache/gemini/exp_authority_131d/fold_{fold}/test_131d.f32.npy' for fold in range(5)]
    contract={'version':'gemini-131d-readonly-plus-six-oof-ranks-v1','qids':qids,'input_hashes':[common.sha(p) for p in inputs]}
    if manifest_path.exists():
        manifest=common.read(manifest_path)
        if {key:manifest[key] for key in contract}!=contract or manifest['sha256']!=common.sha(path):raise ValueError('Set feature cache mismatch')
        return np.load(path,mmap_mode='r')
    CACHE.mkdir(parents=True,exist_ok=True);values=np.lib.format.open_memmap(path.with_suffix('.tmp.npy'),mode='w+',dtype=np.float32,shape=(len(qids),10,143));row={q:i for i,q in enumerate(qids)}
    for fold in range(5):
        fold_qids=[q for q in folds[f'fold_{fold}'] if q in row];docs=common.read(ROOT/f'cache/gemini/exp_unified_ltr/fold_{fold}/test_docs.json');groups=common.read(ROOT/f'cache/gemini/exp_unified_ltr/fold_{fold}/test_groups.json');matrix=np.load(inputs[fold],mmap_mode='r');ends=np.cumsum([0]+groups)
        if len(fold_qids)!=len(docs):raise ValueError('Gemini test-doc order mismatch')
        for index,(qid,candidates) in enumerate(zip(fold_qids,docs)):
            mapping={doc:position for position,doc in enumerate(candidates)};chosen=base[qid][:10]
            if any(doc not in mapping for doc in chosen):raise ValueError(f'Base top10 outside 131D pool: {qid}')
            base_rows=np.asarray(matrix[ends[index]:ends[index+1]])[[mapping[doc] for doc in chosen]]
            extra=[]
            all_orders={'base':base,**systems}
            for order in all_orders.values():
                ranks={doc:rank for rank,doc in enumerate(order[qid],1)};rank_values=np.asarray([ranks.get(doc,1000) for doc in chosen],dtype=np.float32);extra.extend((rank_values,np.where(rank_values<1000,1/(32+rank_values),0.)))
            values[row[qid]]=np.concatenate([base_rows,np.asarray(extra,dtype=np.float32).T],axis=1)
        print(f'materialize_fold={fold} queries={len(fold_qids)}',flush=True)
    values.flush();del values;path.with_suffix('.tmp.npy').replace(path);common.write(ids_path,qids);common.write(manifest_path,{**contract,'shape':[len(qids),10,143],'sha256':common.sha(path)})
    return np.load(path,mmap_mode='r')


def train_checkpoints(features,queries,targets,indices,outer):
    torch.manual_seed(112+outer);np.random.seed(112+outer);random.seed(112+outer)
    mean=np.asarray(features[indices].mean(axis=(0,1)),dtype=np.float32);std=np.asarray(features[indices].std(axis=(0,1)),dtype=np.float32);std=np.where(std>1e-6,std,1.)
    model=ResidualSetRanker(143,queries.shape[1]).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);states={};eligible=indices[targets[indices].sum(1)>0]
    for epoch in range(1,max(EPOCHS)+1):
        model.train();generator=np.random.default_rng(112+outer*100+epoch);order=generator.permutation(eligible);losses=[]
        for start in range(0,len(order),64):
            batch=order[start:start+64];x=torch.from_numpy((np.asarray(features[batch])-mean)/std).to('cuda');q=torch.from_numpy(queries[batch]).to('cuda');y=torch.from_numpy(targets[batch]).to('cuda')
            optimizer.zero_grad(set_to_none=True);loss=multi_positive_set_loss(model(x,q),y,multi_weight=2.);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        print(f'outer={outer} epoch={epoch} loss={np.mean(losses):.6f}',flush=True)
        if epoch in EPOCHS:states[epoch]=copy.deepcopy(model.state_dict())
    return model,states,mean,std


@torch.inference_mode()
def predict_logits(model,state,features,queries,indices,mean,std):
    model.load_state_dict(state);model.eval();result=[]
    for start in range(0,len(indices),256):
        batch=indices[start:start+256];x=torch.from_numpy((np.asarray(features[batch])-mean)/std).to('cuda');q=torch.from_numpy(queries[batch]).to('cuda');result.append(model(x,q).cpu().numpy())
    return np.concatenate(result)


def fused_predictions(base,qids,logits,alpha):
    result={}
    for qi,qid in enumerate(qids):
        docs=base[qid][:10];model_order=sorted(range(10),key=lambda index:(-float(logits[qi,index]),docs[index]));model_rank=np.empty(10,dtype=np.int16)
        for rank,index in enumerate(model_order,1):model_rank[index]=rank
        values=(1-alpha)/(32+np.arange(1,11))+alpha/(32+model_rank);head=[docs[index] for index in sorted(range(10),key=lambda index:(-float(values[index]),docs[index]))];result[qid]=head+base[qid][10:]
    return result


def main():
    import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=common.read(ROOT/'cache/cv_folds.json');systems=load_systems();base=common.blend([systems['xgb'],systems['lgb'],systems['profile']]);qids=[q for fold in range(5) for q in folds[f'fold_{fold}'] if labels.get(q)];row={q:i for i,q in enumerate(qids)}
    features=materialize(qids,folds,base,systems);data=Data();queries=np.asarray([data.query_vector(q,'e5') for q in qids],dtype=np.float32);targets=np.asarray([[doc in labels[q] for doc in base[q][:10]] for q in qids],dtype=np.float32)
    report={'status':'COMPLETE_STRICT_NESTED_SET_RANKER_PROBE','protocol':'three folds train; one calibration selects epoch and rank-fusion alpha; prediction lock before outer labels','gemini_namespace_written':False,'base':common.metrics(base,labels,qids),'folds':{}};oof={}
    for outer in range(5):
        calibration=common.CALIBRATION[outer];inner=[fold for fold in range(5) if fold not in (outer,calibration)];train_qids=[q for fold in inner for q in folds[f'fold_{fold}'] if labels.get(q)];cal_qids=[q for q in folds[f'fold_{calibration}'] if labels.get(q)];outer_qids=[q for q in folds[f'fold_{outer}'] if labels.get(q)];ti=np.asarray([row[q] for q in train_qids]);ci=np.asarray([row[q] for q in cal_qids]);oi=np.asarray([row[q] for q in outer_qids])
        model,states,mean,std=train_checkpoints(features,queries,targets,ti,outer);trials=[];cal_logits={}
        for epoch,state in states.items():
            logits=predict_logits(model,state,features,queries,ci,mean,std);cal_logits[epoch]=logits
            for alpha in ALPHAS:
                pred=fused_predictions(base,cal_qids,logits,alpha);value=common.metrics(pred,labels,cal_qids);actions=sum(pred[q][:5]!=base[q][:5] for q in cal_qids);trials.append({'epoch':epoch,'alpha':alpha,'metrics':value,'actions':actions})
        winner=max(trials,key=lambda item:common.selection_key(item['metrics'],item['actions']));outer_logits=predict_logits(model,states[winner['epoch']],features,queries,oi,mean,std);pred=fused_predictions(base,outer_qids,outer_logits,winner['alpha']);fold_dir=OUT/f'fold_{outer}';common.write(fold_dir/'PREDICTIONS_LOCK.json',pred);torch.save({'state':states[winner['epoch']],'mean':mean,'std':std,'winner':winner},fold_dir/'model.pt')
        lock={'outer_fold':outer,'calibration_fold':calibration,'inner_folds':inner,'winner':winner,'prediction_sha256':common.sha(fold_dir/'PREDICTIONS_LOCK.json'),'model_sha256':common.sha(fold_dir/'model.pt')};common.write(fold_dir/'SELECTION_LOCK.json',lock);value=common.metrics(pred,labels,outer_qids);baseline=common.metrics(base,labels,outer_qids);report['folds'][f'fold_{outer}']={'lock':lock,'base':baseline,'set_ranker':value,'delta':value['recall_at_5']-baseline['recall_at_5'],'top_calibration_trials':sorted(trials,key=lambda item:common.selection_key(item['metrics'],item['actions']),reverse=True)[:10]};oof.update(pred);print(f'outer={outer} winner=epoch{winner["epoch"]}/a{winner["alpha"]} delta={report["folds"][f"fold_{outer}"]["delta"]:+.9f}',flush=True);del model,states;gc.collect();torch.cuda.empty_cache()
    common.write(OUT/'OOF_PREDICTIONS.json',oof);report['oof']=common.metrics(oof,labels,qids);report['delta']=report['oof']['recall_at_5']-report['base']['recall_at_5'];report['nonnegative_folds']=sum(value['delta']>=0 for value in report['folds'].values());common.write(OUT/'NESTED_SET_RANKER_REPORT.json',report);print(json.dumps({'base':report['base'],'oof':report['oof'],'delta':report['delta'],'nonnegative_folds':report['nonnegative_folds']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
