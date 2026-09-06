"""Fold-safe prediction of whether a legal query has multiple gold parents."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))

import exp_final_nested_slate_probe as common  # noqa:E402
from exp_final.data import Data  # noqa:E402

OUT=ROOT/'results/exp_final_retrieval/multigold_propensity_probe'


def main():
    import exp109b_encoder_complementarity as old
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score,roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels,_=old.canonical_labels();folds=common.read(ROOT/'cache/cv_folds.json');data=Data()
    qids=[q for fold in range(5) for q in folds[f'fold_{fold}'] if labels.get(q)]
    vectors=np.asarray([data.query_vector(q,'e5') for q in qids],dtype=np.float32);row={q:i for i,q in enumerate(qids)}
    targets=np.asarray([len(labels[q])>1 for q in qids],dtype=np.int8);probability=np.empty(len(qids),dtype=np.float32);fold_reports={}
    for outer in range(5):
        train=[q for fold in range(5) if fold!=outer for q in folds[f'fold_{fold}'] if labels.get(q)];test=[q for q in folds[f'fold_{outer}'] if labels.get(q)]
        ti=np.asarray([row[q] for q in train]);vi=np.asarray([row[q] for q in test])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=112))
        model.fit(vectors[ti],targets[ti]);probability[vi]=model.predict_proba(vectors[vi])[:,1]
        fold_reports[f'fold_{outer}']={'roc_auc':float(roc_auc_score(targets[vi],probability[vi])),'average_precision':float(average_precision_score(targets[vi],probability[vi])),'queries':len(test),'positives':int(targets[vi].sum())}
        print(f'outer={outer} auc={fold_reports[f"fold_{outer}"]["roc_auc"]:.6f} ap={fold_reports[f"fold_{outer}"]["average_precision"]:.6f}',flush=True)
    enrichment={}
    order=np.argsort(-probability)
    prevalence=float(targets.mean())
    for fraction in (.05,.10,.20,.30,.50):
        count=int(np.ceil(fraction*len(qids)));rate=float(targets[order[:count]].mean());enrichment[str(fraction)]={'multi_rate':rate,'lift':rate/prevalence,'captured_fraction':float(targets[order[:count]].sum()/targets.sum())}
    report={'status':'COMPLETE_MULTIGOLD_PROPENSITY_PROBE','protocol':'fixed LR C=.1; each prediction trained on other four folds','prevalence':prevalence,'roc_auc':float(roc_auc_score(targets,probability)),'average_precision':float(average_precision_score(targets,probability)),'folds':fold_reports,'top_fraction_enrichment':enrichment,'qid_order':qids,'probabilities':probability.tolist()}
    common.write(OUT/'MULTIGOLD_PROPENSITY_REPORT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('qid_order','probabilities')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
