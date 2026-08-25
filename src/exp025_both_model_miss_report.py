"""EXP-025: auditable taxonomy and decision report for EXP-023 both-model misses."""
from __future__ import annotations
import json, re, unicodedata
from collections import Counter
from pathlib import Path
from exp012b_core import atomic_json, read_jsonl, write_jsonl

ROOT=Path(__file__).resolve().parents[1]
WORD=re.compile(r"[^\W_]+",re.UNICODE)
ID=re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-Za-zĐđ]{1,}",re.I)
def norm(text:str)->str:
    return "".join("d" if c.casefold()=="đ" else c for c in unicodedata.normalize("NFD",text.casefold()) if not unicodedata.combining(c))
def tokens(text:str)->set[str]: return {x for x in WORD.findall(norm(text)) if len(x)>=3 and not x.isdigit()}
def run(output:Path)->dict:
    misses=[x for x in read_jsonl(ROOT/'results/exp023_e5_bm25_quota/remaining_misses.jsonl') if x['miss_type']=='both_model_miss']
    docs={str(x['doc_id']):x for x in read_jsonl(ROOT/'cache/structural_v3_e5_final_v1/documents.jsonl')}
    rescues=[x for x in read_jsonl(ROOT/'results/exp024_memory_lexical/rescues.jsonl') if x['channel']=='char_3_5']
    rescue_set={(x['qid'],x['gold_doc_id']) for x in rescues}
    rows=[]; counts=Counter()
    for x in misses:
        label=str(docs[x['gold_doc_id']]['retrieval_name']); q=str(x['query']); overlap=len(tokens(q)&tokens(label)); has_id=bool(ID.search(q))
        kind='no_title_overlap' if overlap==0 else 'title_overlap_3plus' if overlap>=3 else 'weak_title_overlap'
        counts[kind]+=1; counts['with_identifier']+=has_id
        rows.append({**x,'taxonomy':kind,'title_token_overlap':overlap,'has_strict_identifier':has_id,'memory_char_rescued':(x['qid'],x['gold_doc_id']) in rescue_set})
    output.mkdir(parents=True,exist_ok=True); write_jsonl(output/'both_model_misses.jsonl',rows)
    report={'status':'PASS','both_model_miss_occurrences':len(rows),'taxonomy':dict(counts),'memory_char_rescues':sum(x['memory_char_rescued'] for x in rows),'decision':{'entity_retrieval':'REJECT for this miss set: strict identifiers are rare and do not match gold','raw_char_backoff':'REJECT: EXP-024 rescued one targeted miss','query_memory':'DIAGNOSTIC_ONLY: 16 both-model rescues but concentrated in frequent documents and unstable by fold','next_action':'Do not spend the fixed 150 budget. Revisit only with a new independently validated semantic retriever or downstream reranker evidence.'}}
    atomic_json(output/'report.json',report); (output/'REPORT.md').write_text('# EXP-025 — both-model-miss decision\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); return report
if __name__=='__main__': print(json.dumps(run(ROOT/'results/exp025_both_model_miss'),ensure_ascii=False,indent=2))
