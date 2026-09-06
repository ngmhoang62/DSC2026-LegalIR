"""Read-only audit of Gemini OOF evidence, writing only EXP-final artifacts."""
from __future__ import annotations

import json
import sqlite3
import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from exp_final.relations import KinshipPolicy,apply_kinship  # noqa:E402

OUT=ROOT/'results/exp_final_retrieval/gemini_evidence_audit'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def metrics(pred,labels,qids,k=5):
    vals=[];prec=[];multi=[];single=[];mrr=[]
    for q in qids:
        g=labels.get(q,set())
        if not g:continue
        top=pred[q][:k];h=len(set(top)&g);v=h/len(g);vals.append(v);prec.append(h/k)
        (single if len(g)==1 else multi).append(v);mrr.append(next((1/i for i,d in enumerate(top,1) if d in g),0.))
    return dict(recall_at_5=float(np.mean(vals)),precision_at_5=float(np.mean(prec)),single_gold_recall_at_5=float(np.mean(single)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(vals))


def blend(systems,weights,k=10,depth=64):
    result={}
    for q in systems[0]:
        score={}
        for system,weight in zip(systems,weights):
            for rank,doc in enumerate(system[q][:depth],1):score[doc]=score.get(doc,0.)+weight/(k+rank)
        result[q]=sorted(score,key=lambda d:(-score[d],d))
    return result


def titles():
    db=sqlite3.connect(f"file:{(ROOT/'cache/exp112_task_adaptive_retrieval/evidence.sqlite').as_posix()}?mode=ro",uri=True)
    value={str(doc):json.loads(payload).get('retrieval_name','') for doc,payload in db.execute('select doc,payload from documents')};db.close();return value


def policy_predictions(base,title_map,qids,policy):
    pred={};events=[]
    for q in qids:
        pred[q],event=apply_kinship(base[q],title_map,policy)
        if event:events.append({'qid':q,**event})
    return pred,events


def load_gemini_reference():
    """Load the read-only reference implementation without executing its main."""
    path=ROOT/'scripts/gemini/exp_statutory_kinship.py'
    spec=importlib.util.spec_from_file_location('gemini_kinship_reference_readonly',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.apply_kinship_promotion


def main():
    import exp109b_encoder_complementarity as old
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');qids=[q for f in range(5) for q in folds[f'fold_{f}'] if labels.get(q)]
    xgb=read(ROOT/'results/gemini/exp_authority_131d/xgb_131d_OOF_PREDICTIONS.json')
    lgb=read(ROOT/'results/gemini/exp_authority_131d/lgbm_131d_OOF_PREDICTIONS.json')
    profile=read(ROOT/'results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json')
    gemini_final=read(ROOT/'results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json')
    base=blend([xgb,lgb,profile],[.3,.4,.3]);title_map=titles()
    base_metrics=metrics(base,labels,qids);gemini_metrics=metrics(gemini_final,labels,qids)
    policies=[KinshipPolicy(top,cand,subject) for subject in (False,True) for top in (1,2,3) for cand in (7,9,10)]
    policy_rows=[]
    for policy in policies:
        pred,events=policy_predictions(base,title_map,qids,policy);value=metrics(pred,labels,qids)
        policy_rows.append({'policy':policy.__dict__,'metrics':value,'delta':value['recall_at_5']-base_metrics['recall_at_5'],'events':len(events)})
    policy_rows.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5']),reverse=True)
    crossfit={};crossfit_pred={};crossfit_events=[]
    for outer in range(5):
        train=[q for f in range(5) if f!=outer for q in folds[f'fold_{f}'] if labels.get(q)]
        test=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        candidates=[]
        for policy in policies:
            pred,_=policy_predictions(base,title_map,train,policy);value=metrics(pred,labels,train)
            candidates.append((value['recall_at_5'],value['precision_at_5'],-policy.top_k,-policy.candidate_max,not policy.allow_subject,policy))
        selected=max(candidates)[-1];pred,events=policy_predictions(base,title_map,test,selected);crossfit_pred.update(pred);crossfit_events.extend([{'fold':outer,**e} for e in events])
        crossfit[f'fold_{outer}']={'selected':selected.__dict__,'train_metrics':max(candidates)[:2],'test':metrics(pred,labels,test),'base':metrics(base,labels,test),'events':len(events)}
    crossfit_metrics=metrics(crossfit_pred,labels,qids)
    # Exact-reference audit: isolate global OOF selection bias from differences
    # introduced by EXP-final's stricter relation parser.
    reference=load_gemini_reference();reference_policies=[(top,cand) for top in (1,2,3) for cand in (7,9,10)]
    reference_default,_=reference(base,title_map,qids,top_k=2,cand_max=9)
    reference_mismatches=sum(reference_default[q][:20]!=gemini_final[q][:20] for q in qids)
    reference_crossfit_pred={};reference_crossfit={}
    for outer in range(5):
        train=[q for f in range(5) if f!=outer for q in folds[f'fold_{f}'] if labels.get(q)]
        test=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        candidates=[]
        for top,cand in reference_policies:
            pred,_=reference(base,title_map,train,top_k=top,cand_max=cand);value=metrics(pred,labels,train)
            candidates.append((value['recall_at_5'],value['precision_at_5'],-top,-cand,top,cand))
        selected=max(candidates);pred,count=reference(base,title_map,test,top_k=selected[-2],cand_max=selected[-1]);reference_crossfit_pred.update(pred)
        reference_crossfit[f'fold_{outer}']={'selected':{'top_k':selected[-2],'candidate_max':selected[-1]},'test':metrics(pred,labels,test),'base':metrics(base,labels,test),'events':count}
    reference_crossfit_metrics=metrics(reference_crossfit_pred,labels,qids)
    # Exact deficit and top-k capacity in macro query-recall points.
    curves={};query_values={}
    for k in range(1,11):
        values={q:len(set(base[q][:k])&labels[q])/len(labels[q]) for q in qids};query_values[k]=values;curves[str(k)]=float(np.mean(list(values.values())))
    deficit=float(.96*len(qids)-sum(query_values[5].values()))
    recoverable_6_7=float(sum(query_values[7][q]-query_values[5][q] for q in qids))
    patterns={'candidate_gold_rank5_not':0,'rank5_gold_candidate_not':0,'both_gold':0,'neither_gold':0}
    for q in qids:
        gold=labels[q];rank5=base[q][4]
        for candidate in base[q][5:7]:
            a=rank5 in gold;b=candidate in gold
            key='both_gold' if a and b else 'rank5_gold_candidate_not' if a else 'candidate_gold_rank5_not' if b else 'neither_gold'
            patterns[key]+=1
    mismatches=sum(gemini_final[q][:20]!=policy_predictions(base,title_map,[q],KinshipPolicy())[0][q][:20] for q in qids)
    report={'status':'COMPLETE_READ_ONLY_GEMINI_EVIDENCE_AUDIT','gemini_namespace_written':False,'base_metrics':base_metrics,'gemini_final_recomputed':gemini_metrics,'default_policy_reproduction_mismatched_queries_top20':mismatches,'global_policy_trials':policy_rows,'crossfit':crossfit,'crossfit_metrics':crossfit_metrics,'crossfit_delta':crossfit_metrics['recall_at_5']-base_metrics['recall_at_5'],'gemini_exact_reference':{'default_mismatched_queries_top20':reference_mismatches,'default_metrics':metrics(reference_default,labels,qids),'crossfit':reference_crossfit,'crossfit_metrics':reference_crossfit_metrics,'crossfit_delta':reference_crossfit_metrics['recall_at_5']-base_metrics['recall_at_5']},'curves':curves,'target_096':{'remaining_query_recall_points':deficit,'recoverable_points_rank6_7':recoverable_6_7,'fraction_required':deficit/recoverable_6_7},'rank5_vs_rank6_7_patterns':patterns,'crossfit_events':crossfit_events}
    write(OUT/'GEMINI_EVIDENCE_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ('status','base_metrics','gemini_final_recomputed','default_policy_reproduction_mismatched_queries_top20','crossfit_metrics','crossfit_delta','gemini_exact_reference','curves','target_096','rank5_vs_rank6_7_patterns')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
