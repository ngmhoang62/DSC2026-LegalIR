"""EXP-037 bounded cached-encoder complementarity screen; never a recall claim."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from exp012b_core import atomic_json, canonical_json, read_jsonl, sha256_file, write_jsonl
from exp030_legal_evidence_routing import LABEL_POLICY, canonical_answers
from exp015_model_screen import MODELS, create_encoder, encode_oom_safe
ROOT=Path(__file__).resolve().parents[1]; SCHEMA="legalir.exp037_cached_encoder_complementarity.v2"; POLICY="canonical_duplicate_alias_drop_empty_passage_v1"
KEYS=("vietlegal_e5","vietlegal_harrier_0_6b","vnlegal_lal","multilingual_e5_large_instruct","qwen3_embedding_0_6b","bge_m3","vietnamese_legal_embedding"); CAP=8; SEED=37037
def _j(p):return json.loads(p.read_text(encoding="utf-8"))
def _h(x):return hashlib.sha256(canonical_json(x).encode()).hexdigest()
def fixture(evidence,universe,train,preprocessing,chunks_path,out):
 if LABEL_POLICY!=POLICY:raise RuntimeError("canonical policy drift")
 labels,_=canonical_answers(train,preprocessing/"exclusions.json",preprocessing/"train_label_impact.jsonl"); u={str(x['qid']):x for x in read_jsonl(universe)}; rows=[]; wanted=set()
 for e in read_jsonl(evidence):
  q=str(e['qid']); ids=sorted(set(str(x['doc_id']) for x in u[q]['candidates'][:96])|labels[q]);wanted.update(ids);rows.append({'qid':q,'fold':e['fold'],'role':e['role'],'starter_tag':e['starter_tag'],'query':e['question'],'gold_doc_ids':sorted(labels[q]),'candidate_doc_ids':ids,'bounded_oracle_fixture_not_recall':True})
 n=defaultdict(int); chunks=[]
 for x in read_jsonl(chunks_path):
  d=str(x['doc_id'])
  if d in wanted and n[d]<CAP:chunks.append({'chunk_id':str(x['chunk_id']),'doc_id':d,'retrieval_text':str(x['retrieval_text'])});n[d]+=1
 if wanted-set(n):raise RuntimeError('fixture document lacks chunks')
 out.mkdir(parents=True,exist_ok=True);write_jsonl(out/'fixture.jsonl',rows);write_jsonl(out/'chunks.jsonl',chunks)
 r={'schema_version':SCHEMA,'status':'FIXTURE_READY_NOT_RECALL','queries':len(rows),'documents':len(wanted),'chunks':len(chunks),'max_chunks_per_document':CAP,'label_policy':POLICY,'bounded_oracle_fixture_not_recall':True,'promotion_gate':'FORBIDDEN','fixture_fingerprint':_h(rows),'inputs':{'evidence':sha256_file(evidence),'universe':sha256_file(universe),'chunks':sha256_file(chunks_path)}};atomic_json(out/'REPORT.json',r);atomic_json(out/'RUN_STATUS.json',{'status':r['status'],'stage':'fixture'});atomic_json(out/'_SUCCESS.json',r);return r
def rank(rows,chunks,qv,dv):
 by=defaultdict(list)
 for i,x in enumerate(chunks):by[x['doc_id']].append(i)
 result=[]
 for i,x in enumerate(rows):
  scores=[(d,float(np.max(qv[i]@dv[idx]))) for d,idx in by.items() if d in set(x['candidate_doc_ids'])];order=[d for d,_ in sorted(scores,key=lambda z:(-z[1],z[0]))];gold=set(x['gold_doc_ids']);first=next((j for j,d in enumerate(order,1) if d in gold),len(order)+1);result.append({'qid':x['qid'],'fold':x['fold'],'role':x['role'],'starter_tag':x['starter_tag'],'first_gold_rank':first,'candidate_count':len(order),'top5':order[:5],'top32':order[:32]})
 return result
def run_model(fixture_dir,out,key,device,batch_size,resume=False):
 if key not in KEYS:raise ValueError('unknown model')
 meta=_j(fixture_dir/'REPORT.json')
 if not meta.get('bounded_oracle_fixture_not_recall'):raise RuntimeError('malformed bounded policy')
 target=out/f'{key}.json';vec=out/f'{key}.embeddings.npz'
 if resume and target.exists() and vec.exists():return _j(target)
 rows=list(read_jsonl(fixture_dir/'fixture.jsonl'));chunks=list(read_jsonl(fixture_dir/'chunks.jsonl'));spec=MODELS[key];enc=create_encoder(spec,device,allow_download=False);t=time.perf_counter();dv,b=encode_oom_safe(enc,[spec.document_prefix+x['retrieval_text'] for x in chunks],batch_size,device);qv,b=encode_oom_safe(enc,[spec.query_prefix+x['query'] for x in rows],b,device);out.mkdir(parents=True,exist_ok=True);write_jsonl(out/f'{key}.rankings.jsonl',rank(rows,chunks,qv,dv));np.savez_compressed(vec,qids=np.asarray([x['qid'] for x in rows]),chunk_ids=np.asarray([x['chunk_id'] for x in chunks]),query_vectors=qv,chunk_vectors=dv);r={'schema_version':SCHEMA,'model':key,'runtime':{'device':device,'effective_batch':b,'elapsed_seconds':time.perf_counter()-t},'bounded_oracle_fixture_not_recall':True,'promotion_gate':'FORBIDDEN','fixture_fingerprint':meta['fixture_fingerprint']};atomic_json(target,r);return r
def report(fixture_dir,out):
 base={x['qid']:x for x in read_jsonl(out/'vietlegal_e5.rankings.jsonl')};models=[];rng=np.random.default_rng(SEED)
 for k in KEYS:
  p=out/f'{k}.rankings.jsonl'
  if not p.exists():continue
  cur={x['qid']:x for x in read_jsonl(p)}; qs=sorted(cur);d=np.asarray([(base[q]['first_gold_rank']-cur[q]['first_gold_rank'])/base[q]['candidate_count'] for q in qs]);boot=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(10000)];blind=[q for q in qs if cur[q]['starter_tag']=='source_generation_miss'];added=sum(cur[q]['first_gold_rank']<=32 and base[q]['first_gold_rank']>32 for q in blind);ctrl=[q for q in qs if cur[q]['role']=='control'];lost=np.mean([base[q]['first_gold_rank']<=5 and cur[q]['first_gold_rank']>5 for q in ctrl]) if ctrl else 0;fold={f:float(np.mean([d[i] for i,q in enumerate(qs) if cur[q]['fold']==f])) for f in sorted({cur[q]['fold'] for q in qs})};passed=np.quantile(boot,.025)>0 and added>=15 and lost<=.02 and all(v>=0 for v in fold.values());models.append({'model':k,'bounded_oracle_fixture_not_recall':True,'promotion_gate':'FORBIDDEN','paired_normalized_rank_delta':{'mean':float(d.mean()),'bootstrap_ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]},'blind_gold_added_to_bounded_top32':added,'matched_control_top5_exit_rate':float(lost),'per_fold_mean_delta':fold,'complementarity_gate':'PASS' if passed else 'FAIL'})
 r={'schema_version':SCHEMA,'status':'COMPLETE_BOUNDED_NOT_RECALL','bounded_oracle_fixture_not_recall':True,'promotion_gate':'FORBIDDEN','models':models,'passed_models':[x['model'] for x in models if x['complementarity_gate']=='PASS']};atomic_json(out/'REPORT.json',r);atomic_json(out/'RUN_STATUS.json',{'status':r['status'],'stage':'report'});atomic_json(out/'_SUCCESS.json',r);return r
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','fixture','run-fold','overnight','report'));p.add_argument('--model',choices=KEYS);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--resume',action='store_true');a=p.parse_args();cache=ROOT/'cache'/'exp037_cached_encoder_complementarity';out=ROOT/'results'/'exp037_cached_encoder_complementarity'
 if a.stage in ('audit','fixture'):r=fixture(ROOT/'results'/'exp035_retrieval_error_adjudication'/'evidence_pack.jsonl',ROOT/'cache'/'exp036_coverage_aware_fusion'/'universe.jsonl',ROOT/'public_test_dataset'/'train.json',ROOT/'cache'/'final_preprocessed_v2',ROOT/'cache'/'structural_v3_e5_final_v1'/'chunks.jsonl',cache)
 elif a.stage=='run-fold':
  if not a.model:raise SystemExit('--model required')
  r=run_model(cache,out,a.model,a.device,a.batch_size,a.resume)
 elif a.stage=='report':r=report(cache,out)
 else:raise SystemExit('overnight intentionally disabled; launch cached models one at a time')
 print(json.dumps(r,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
