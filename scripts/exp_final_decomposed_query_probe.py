"""Frozen E5 multi-view query retrieval inside the strong base top-100 pool."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

import exp_final_nested_slate_probe as common  # noqa:E402
from exp_final.contracts import digest,write  # noqa:E402
from exp_final.data import Data  # noqa:E402
from exp_final.decomposition import inverse_document_frequency,query_views  # noqa:E402
from exp_final.learning import ParentBank,QueryEncoder  # noqa:E402

OUT=ROOT/'results/exp_final_retrieval/decomposed_query_probe'
CACHE=ROOT/'cache/exp_final_retrieval/decomposed_query_probe'
WEIGHTS=(0.,.05,.10,.20,.30,.50)
DEPTH=64


def sources_and_base():
    xgb=common.read(ROOT/'results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json')
    lgb=common.read(ROOT/'results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json')
    profile=common.read(ROOT/'results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json')
    return (xgb,lgb,profile),common.blend([xgb,lgb,profile])


def encode_views(data,qids,questions):
    path=CACHE/'view_vectors.f16.npy';mask_path=CACHE/'view_mask.npy';views_path=CACHE/'views.json';marker_path=CACHE/'view_vectors_manifest.json'
    idf=inverse_document_frequency(questions[q] for q in qids)
    views={q:query_views(questions[q],idf) for q in qids}
    if any(not value or len(value)>3 for value in views.values()):raise ValueError('Invalid view count')
    contract=digest(['query-views-v1',qids,views,data.fingerprint])
    if marker_path.exists():
        marker=common.read(marker_path)
        if marker['contract']!=contract or marker['sha256']!=common.sha(path) or marker['mask_sha256']!=common.sha(mask_path):raise ValueError('View vector cache mismatch')
        return np.load(path,mmap_mode='r'),np.load(mask_path,mmap_mode='r'),views
    CACHE.mkdir(parents=True,exist_ok=True)
    values=np.lib.format.open_memmap(path.with_suffix('.tmp.npy'),mode='w+',dtype=np.float16,shape=(len(qids),3,1024));mask=np.zeros((len(qids),3),dtype=bool)
    model=QueryEncoder(device='cuda',source='e5');model.eval()
    with torch.inference_mode():
        observed=model([questions[qids[0]]])[0].float().cpu().numpy();expected=data.query_vector(qids[0],'e5')
    cosine=float(observed@expected/(max(np.linalg.norm(observed),1e-12)*max(np.linalg.norm(expected),1e-12)))
    if cosine<.9999:raise ValueError(f'Frozen E5 encoder parity failed: {cosine}')
    pending=[];locations=[]
    def flush():
        if not pending:return
        with torch.inference_mode():encoded=model(pending).float().cpu().numpy()
        for vector,(qi,vi) in zip(encoded,locations):values[qi,vi]=vector
        pending.clear();locations.clear()
    for qi,qid in enumerate(qids):
        for vi,text in enumerate(views[qid]):pending.append(text);locations.append((qi,vi));mask[qi,vi]=True
        if len(pending)>=24:flush()
        if (qi+1)%500==0:print(f'encode_views={qi+1}/{len(qids)}',flush=True)
    flush();values.flush();del values,model
    torch.cuda.empty_cache()
    path.with_suffix('.tmp.npy').replace(path);np.save(mask_path,mask);common.write(views_path,views)
    common.write(marker_path,dict(contract=contract,sha256=common.sha(path),mask_sha256=common.sha(mask_path),encoder_parity_cosine=cosine,shape=[len(qids),3,1024]))
    return np.load(path,mmap_mode='r'),np.load(mask_path,mmap_mode='r'),views


def local_top2(view_vectors,chunk_vectors,local_parent,parent_count):
    scores=view_vectors.float()@chunk_vectors.T;batch=len(scores);ids=local_parent.expand(batch,-1)
    first=torch.full((batch,parent_count),-torch.inf,device=scores.device);first.scatter_reduce_(1,ids,scores,reduce='amax',include_self=True)
    sentinel=scores.shape[1];chunks=torch.arange(sentinel,device=scores.device)
    arg1=torch.full((batch,parent_count),sentinel,device=scores.device,dtype=torch.long)
    eligible=torch.where(scores==first.gather(1,ids),chunks,sentinel);arg1.scatter_reduce_(1,ids,eligible,reduce='amin',include_self=True)
    rest=scores.masked_fill(chunks[None]==arg1.gather(1,ids),-torch.inf)
    second=torch.full_like(first,-torch.inf);second.scatter_reduce_(1,ids,rest,reduce='amax',include_self=True)
    counts=torch.bincount(local_parent[0],minlength=parent_count);second[:,counts==1]=first[:,counts==1]
    return .5*(first+second)


def score_views(data,qids,base,vectors,mask):
    scores_path=CACHE/'view_parent_scores_d64.f32.npy';ranks_path=CACHE/'view_parent_ranks_d64.i16.npy';marker_path=CACHE/'view_parent_d64_manifest.json'
    contract=digest(['local-top64-top2-v1',qids,data.fingerprint,common.sha(CACHE/'view_vectors.f16.npy')])
    if marker_path.exists():
        marker=common.read(marker_path)
        if marker['contract']!=contract or marker['scores_sha256']!=common.sha(scores_path) or marker['ranks_sha256']!=common.sha(ranks_path):raise ValueError('View parent cache mismatch')
        return np.load(scores_path,mmap_mode='r'),np.load(ranks_path,mmap_mode='r')
    score_out=np.lib.format.open_memmap(scores_path.with_suffix('.tmp.npy'),mode='w+',dtype=np.float32,shape=(len(qids),3,DEPTH));rank_out=np.lib.format.open_memmap(ranks_path.with_suffix('.tmp.npy'),mode='w+',dtype=np.int16,shape=(len(qids),3,DEPTH))
    bank=ParentBank(data.matrix('e5'),data.parent,device='cuda')
    with torch.inference_mode():
        for qi,qid in enumerate(qids):
            docs=base[qid][:DEPTH]
            if len(docs)!=DEPTH:raise ValueError('Base top64 unavailable')
            indices=np.concatenate([data.positions[data.doc_row[doc]] for doc in docs]);parents=np.concatenate([np.full(len(data.positions[data.doc_row[doc]]),index,dtype=np.int64) for index,doc in enumerate(docs)])
            view=torch.from_numpy(np.asarray(vectors[qi],dtype=np.float32)).to('cuda');chunk_indices=torch.from_numpy(indices).to('cuda');local_parent=torch.from_numpy(parents).to('cuda')[None]
            parent_scores=local_top2(view,bank.vectors[chunk_indices],local_parent,DEPTH).cpu().numpy();score_out[qi]=parent_scores
            for vi in range(3):
                order=sorted(range(DEPTH),key=lambda index:(-float(parent_scores[vi,index]),docs[index]));ranks=np.empty(DEPTH,dtype=np.int16)
                for rank,index in enumerate(order,1):ranks[index]=rank
                rank_out[qi,vi]=ranks
            if (qi+1)%100==0 or qi+1==len(qids):print(f'score_views={qi+1}/{len(qids)}',flush=True)
    score_out.flush();rank_out.flush();del score_out,rank_out,bank;torch.cuda.empty_cache()
    scores_path.with_suffix('.tmp.npy').replace(scores_path);ranks_path.with_suffix('.tmp.npy').replace(ranks_path)
    common.write(marker_path,dict(contract=contract,scores_sha256=common.sha(scores_path),ranks_sha256=common.sha(ranks_path),shape=[len(qids),3,DEPTH]))
    return np.load(scores_path,mmap_mode='r'),np.load(ranks_path,mmap_mode='r')


def ranking_for(base_order,ranks,active,aggregation,weight,k=32):
    docs=base_order[:DEPTH];base_recip=1/(k+np.arange(1,DEPTH+1,dtype=np.float32));view_recip=1/(k+ranks[active].astype(np.float32))
    specialist=view_recip.max(0) if aggregation=='max' else view_recip.mean(0)
    values=(1-weight)*base_recip+weight*specialist
    return [docs[index] for index in sorted(range(DEPTH),key=lambda index:(-float(values[index]),docs[index]))]


def main():
    import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=common.read(ROOT/'cache/cv_folds.json');data=Data();questions=data.questions
    systems,base=sources_and_base();qids=[q for fold in range(5) for q in folds[f'fold_{fold}'] if labels.get(q)]
    vectors,mask,views=encode_views(data,qids,questions);scores,ranks=score_views(data,qids,base,vectors,mask);row={q:i for i,q in enumerate(qids)}
    configurations=[('noop',0.)]+[(aggregation,weight) for aggregation in ('max','mean') for weight in WEIGHTS[1:]]
    predictions={config:{} for config in configurations}
    for qid in qids:
        qi=row[qid]
        for config in configurations:
            predictions[config][qid]=list(base[qid]) if config[0]=='noop' else ranking_for(base[qid],ranks[qi],mask[qi],*config)
    oof={};fold_reports={};locks={}
    for outer in range(5):
        calibration=common.CALIBRATION[outer];cal_qids=[q for q in folds[f'fold_{calibration}'] if labels.get(q)];outer_qids=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        trials=[]
        for config in configurations:
            value=common.metrics(predictions[config],labels,cal_qids);trials.append((common.selection_key(value,0),config,value))
        _,winner,cal_value=max(trials);selected={q:predictions[winner][q] for q in outer_qids};fold_dir=OUT/f'fold_{outer}';common.write(fold_dir/'PREDICTIONS_LOCK.json',selected)
        lock=dict(outer_fold=outer,calibration_fold=calibration,configuration={'aggregation':winner[0],'weight':winner[1]},calibration_metrics=cal_value,prediction_sha256=common.sha(fold_dir/'PREDICTIONS_LOCK.json'));common.write(fold_dir/'SELECTION_LOCK.json',lock)
        value=common.metrics(selected,labels,outer_qids);baseline=common.metrics(base,labels,outer_qids);fold_reports[f'fold_{outer}']=dict(lock=lock,base=baseline,decomposed=value,delta=value['recall_at_5']-baseline['recall_at_5']);locks[f'fold_{outer}']=lock['configuration'];oof.update(selected)
        print(f'outer={outer} config={winner} delta={fold_reports[f"fold_{outer}"]["delta"]:+.9f}',flush=True)
    base_metrics=common.metrics(base,labels,qids);oof_metrics=common.metrics(oof,labels,qids)
    oracle_values=[];retrieval_rescues=0
    for qid in qids:
        gold=labels[qid];best=max(len(set(predictions[config][qid][:5])&gold)/len(gold) for config in configurations);oracle_values.append(best)
        base_pool=set(base[qid][:DEPTH]);retrieval_rescues+=sum(doc in base_pool for doc in gold)
    report=dict(status='COMPLETE_DECOMPOSED_QUERY_PROBE',gemini_namespace_written=False,view_contract='head65-tail65-query-idf-rare',candidate_depth=DEPTH,view_count_distribution={str(n):sum(len(v)==n for v in views.values()) for n in (1,2,3)},base=base_metrics,folds=fold_reports,selected=locks,oof=oof_metrics,delta=oof_metrics['recall_at_5']-base_metrics['recall_at_5'],nonnegative_folds=sum(r['delta']>=0 for r in fold_reports.values()),choice_oracle_recall_at_5=float(np.mean(oracle_values)),base_depth_gold_assignment_coverage=retrieval_rescues/sum(len(labels[q]) for q in qids))
    common.write(OUT/'OOF_PREDICTIONS.json',oof);common.write(OUT/'DECOMPOSED_QUERY_REPORT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
