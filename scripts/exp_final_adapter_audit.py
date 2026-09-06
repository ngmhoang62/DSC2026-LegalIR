"""Read-only matched-scope audit of EXP-112 adapted E5 checkpoints."""
from __future__ import annotations
import json, re, sqlite3, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import exp109b_encoder_complementarity as old
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def metric(rows,labels,qids):return float(np.mean([len(set(rows[str(q)][:5])&labels[str(q)])/len(labels[str(q)]) for q in qids if labels.get(str(q))]))
def score_head(path):
    """Read only the scalar and first five IDs from the pretty-printed JSON."""
    cosine=None;order=[];inside=False
    with Path(path).open('r',encoding='utf-8') as handle:
        for line in handle:
            if '"frozen_query_cosine"' in line:
                cosine=float(line.split(':',1)[1].rstrip().rstrip(','))
            elif '"order"' in line:
                inside=True
            elif inside:
                match=re.search(r'"([^"]+)"',line)
                if match:order.append(match.group(1))
                if len(order)==5:break
    if cosine is None or len(order)!=5:raise ValueError(f'Cannot read score head: {path}')
    return order,cosine
def main():
    labels,_=old.canonical_labels();folds=read(ROOT/'cache/cv_folds.json');db=sqlite3.connect(ROOT/'cache/exp112_task_adaptive_retrieval/sources.sqlite');report={}
    for outer in range(5):
        cal=list(map(str,folds[f'fold_{(outer-1)%5}']));frozen={}
        for q in cal:frozen[q]=[row['doc_id'] for row in json.loads(db.execute('SELECT payload FROM sources WHERE q=? AND source=?',(q,'e5')).fetchone()[0])]
        item={'calibration_fold':f'fold_{(outer-1)%5}','frozen_e5_recall_at_5':metric(frozen,labels,cal),'epochs':{}}
        for epoch in (1,2):
            folder=ROOT/f'cache/exp112_task_adaptive_retrieval/outer/fold_{outer}/cal-query-{epoch}';rows={};cos=[]
            for q in cal:
                order,similarity=score_head(folder/f'{q}.json');rows[q]=order;cos.append(similarity)
            item['epochs'][str(epoch)]={'recall_at_5':metric(rows,labels,cal),'delta':metric(rows,labels,cal)-item['frozen_e5_recall_at_5'],'cosine_mean':float(np.mean(cos)),'cosine_min':float(np.min(cos)),'cosine_p05':float(np.quantile(cos,.05))}
        report[f'fold_{outer}']=item
    path=ROOT/'results/exp_final_retrieval/adapter_audit/ADAPTER_AUDIT.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
