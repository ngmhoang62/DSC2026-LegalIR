"""Complete the EXP-111 disk-backed sparse ensemble from verified shards.

EXP-111 deliberately stopped before its LambdaMART evaluator existed.  This
probe consumes only the immutable, hash-verified F1--F4 feature shards and
performs strict held-fold cross-fitting without touching Fold 0.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cache/exp111_multiview_sparse/sparse_features_v2/inner"
OUT = ROOT / "results/exp_final_retrieval/sparse_ltr_probe"


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def load_records():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import exp111_multiview_sparse_retrieval as old
    records=defaultdict(list);feature_names=None;shards=0
    for shard,path,marker in old.iter_feature_v2_shards('inner'):
        del shard,marker;shards+=1
        with path.open('r',encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line);names=row['feature_names']
                if feature_names is None:feature_names=names
                if names!=feature_names:raise ValueError('feature order changed')
                records[row['fold']].append((str(row['qid']),list(map(str,row['docs'])),np.asarray(row['features'],dtype=np.float32),np.asarray(row['labels'],dtype=np.int8)))
    if shards!=88:raise ValueError(f'expected 88 verified shards, got {shards}')
    return records,feature_names


def metrics(predictions,answers,qids):
    values=[];multi=[];precision=[];mrr=[]
    for q in qids:
        gold=answers.get(q,set())
        if not gold:continue
        top=predictions[q][:5];hits=len(set(top)&gold);value=hits/len(gold);values.append(value);precision.append(hits/5)
        if len(gold)>1:multi.append(value)
        first=next((i for i,d in enumerate(top,1) if d in gold),None);mrr.append(0 if first is None else 1/first)
    return dict(recall_at_5=float(np.mean(values)),precision_at_5=float(np.mean(precision)),multi_gold_recall_at_5=float(np.mean(multi)),mrr_at_5=float(np.mean(mrr)),queries=len(values))


def blocks(names):
    globals_=[i for i,n in enumerate(names) if n.startswith('query_') or n=='source_agreement']
    def source(prefix):return [i for i,n in enumerate(names) if n.startswith(prefix+'_')]
    return {
        'v0':source('v0_control')+globals_,
        'v0_trigram':source('v0_control')+source('v4_trigram')+globals_,
        'structural':source('v0_control')+source('v1_surface_structural')+source('v2_w384')+source('v2_w512')+source('v4_bigram')+source('v4_trigram')+globals_,
        'all_sparse':list(range(len(names))),
    }


def main():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import exp111_multiview_sparse_retrieval as old

    questions,answers,folds,_fold_for,label_audit=old.canonical_inner();del questions
    records,names=load_records();feature_blocks=blocks(names);systems={}
    reports=[]
    for heldout in old.INNER_FOLDS:
        training=[row for fold in old.INNER_FOLDS if fold!=heldout for row in records[fold]];testing=records[heldout]
        groups=[len(row[1]) for row in training];x=np.concatenate([row[2] for row in training]);y=np.concatenate([row[3] for row in training])
        test_x=np.concatenate([row[2] for row in testing]);offset=np.cumsum([0]+[len(row[1]) for row in testing])
        fold_report={};
        for block,indices in feature_blocks.items():
            for family in ('lm','lr'):
                key=f'{family}_{block}'
                if family=='lm':
                    model=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',ndcg_at=[5],num_leaves=7,min_child_samples=50,learning_rate=.05,n_estimators=300,feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,deterministic=True,force_col_wise=True,n_jobs=4,random_state=111,verbosity=-1)
                    model.fit(x[:,indices],y,group=groups,eval_at=[5]);score=model.predict(test_x[:,indices])
                    importance=dict(sorted(zip((names[i] for i in indices),model.feature_importances_.tolist()),key=lambda z:-z[1])[:12])
                else:
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=111))
                    model.fit(x[:,indices],y);score=model.decision_function(test_x[:,indices]);importance={}
                prediction={}
                for qi,(qid,docs,_features,_labels) in enumerate(testing):
                    local=score[offset[qi]:offset[qi+1]];prediction[qid]=[docs[i] for i in sorted(range(len(docs)),key=lambda i:(-float(local[i]),docs[i]))]
                systems.setdefault(key,{}).update(prediction);fold_report[key]={'metrics':metrics(prediction,answers,[r[0] for r in testing]),'feature_count':len(indices),'top_importance':importance}
                print(f'{heldout} {key} recall={fold_report[key]["metrics"]["recall_at_5"]:.9f}',flush=True)
        reports.append({'heldout':heldout,'train_folds':[f for f in old.INNER_FOLDS if f!=heldout],'training_queries':len(training),'training_rows':len(x),'test_queries':len(testing),'systems':fold_report})
        del x,y,test_x
    qids=sorted(questions_q for fold in old.INNER_FOLDS for questions_q in folds[fold] if answers[questions_q])
    aggregate=[]
    for key,pred in systems.items():
        score=metrics(pred,answers,qids);per={r['heldout']:r['systems'][key]['metrics'] for r in reports}
        aggregate.append({'system':key,'metrics':score,'per_fold':per})
        write(OUT/key/'PREDICTIONS.json',pred)
    aggregate.sort(key=lambda r:(r['metrics']['recall_at_5'],r['metrics']['precision_at_5'],r['metrics']['mrr_at_5']),reverse=True)
    # Query-level choice oracle across all evaluated sparse systems is strictly
    # diagnostic; it quantifies ranker headroom, not an attainable submission.
    oracle=[]
    for q in qids:
        gold=answers[q];oracle.append(max(len(set(pred[q][:5])&gold)/len(gold) for pred in systems.values()))
    report={'status':'COMPLETE_STRICT_INNER_SPARSE_LTR','scope':'F1-F4 strict cross-fit; Fold0 not loaded','label_audit':label_audit,'feature_names':names,'blocks':{k:[names[i] for i in v] for k,v in feature_blocks.items()},'folds':reports,'aggregate':aggregate,'choice_oracle_recall@5':float(np.mean(oracle))}
    write(OUT/'SPARSE_LTR_REPORT.json',report);print(json.dumps({'status':report['status'],'aggregate':aggregate,'choice_oracle_recall@5':report['choice_oracle_recall@5']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
