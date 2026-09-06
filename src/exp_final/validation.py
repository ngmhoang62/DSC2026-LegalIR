"""Real bounded CPU integration screen, separate from scientific calibration."""
from pathlib import Path
import time
import numpy as np
import psutil
from .contracts import CACHE, RESULTS, ROOT, read, write, sha, splits, Progress
from .data import Data, SourceStore, prepare_sparse
from .fusion import features, fit_ranker, upstream


def validate():
    data, store = Data(), SourceStore()
    train, cal, outer = splits(data.folds, 'fold_0')
    # Use already prepared preflight queries, all from the inner training side.
    available = [q for q in train if data.gold[q] and store.get(q,'bm25') is not None]
    selected = available[:256]
    if len(selected) < 128:
        raise ValueError('Need prepared real inner-training fixtures')
    rss, times = [], []
    progress = Progress('real_feature_validation', len(selected))
    start = time.monotonic()
    for i,q in enumerate(selected):
        tick = time.monotonic()
        docs = list(dict.fromkeys(store.candidates(q)+sorted(data.gold[q])))
        for block in (0,1):
            x = features(data, store, q, docs, block)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite real features')
        times.append(time.monotonic()-tick)
        rss.append(psutil.Process().memory_info().rss)
        progress.update(i+1)
    feature_seconds = time.monotonic()-start
    training = selected[:96]; testing = selected[96:128]
    fit_seconds, predictions = {}, {}
    for family in ('lm','lr'):
        tick = time.monotonic()
        model = fit_ranker(data,store,training,family,1,CACHE/'validation'/f'{family}.pkl')
        recipe = dict(family=family,block=1,pool='frozen',epoch=0,beta=0)
        predictions[family] = {q:upstream(data,store,q,model,recipe)['order'][:5] for q in testing}
        fit_seconds[family] = time.monotonic()-tick
    b2_timing={}
    if read(RESULTS/'RESOURCE_LOCK.json').get('jina'):
        jq=read(RESULTS/'REPRODUCTION_REPORT.json')['qids']
        for family in ('lm','lr'):
            tick=time.monotonic()
            model=fit_ranker(data,store,jq[:6],family,2,CACHE/'validation'/f'{family}-b2.pkl')
            for q in jq[6:]:
                r=upstream(data,store,q,model,dict(family=family,block=2,pool='frozen'))
                if len(r['order'])<5:
                    raise ValueError('B2 integration produced too few predictions')
            b2_timing[family]=time.monotonic()-tick
    from transformers import AutoTokenizer
    from .learning import local_snapshot
    from .evidence import Evidence
    tokenizer = AutoTokenizer.from_pretrained(str(local_snapshot('BAAI/bge-reranker-v2-m3')),local_files_only=True)
    evidence = Evidence(data,tokenizer)
    packages = []
    for q in selected[:16]:
        for d in store.candidates(q)[:2]:
            p = evidence.package(q,d)
            assert p['pair_tokens']<=512 and p['source_exact']
            packages.append(dict(qid=q,doc_id=d,**p))
    evidence.db.close(); store.close()
    result = dict(passed=True,scope='engineering_inner_train_only_not_calibration',outer_labels_used=False,
                  qids=selected,source_exact_packages=packages,feature_seconds=feature_seconds,
                  b2_fit_predict_seconds=b2_timing,
                  feature_seconds_per_query=feature_seconds/len(selected),feature_p50=float(np.median(times)),
                  feature_p90=float(np.quantile(times,.9)),ml_seconds_96_training_queries=fit_seconds,
                  rss_samples=rss,ram_growth_last_half_bytes=max(rss[len(rss)//2:])-min(rss[len(rss)//2:]),
                  available_ram_bytes=psutil.virtual_memory().available,
                  synthetic=False,production_training=False,model_predictions=predictions,
                  code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/exp_final').glob('*.py')})
    write(RESULTS/'IMPLEMENTATION_VALIDATION.json', result)
    return result


def public_smoke():
    from .data import public_dense
    data,store=Data(),SourceStore()
    qids=list(data.public)[:2]
    data.public={q:data.public[q] for q in qids}
    started=time.monotonic()
    prepare_sparse(data,store,qids)
    timings={}
    for source in ('e5','lal'):
        tick=time.monotonic()
        public_dense(data,store,source)
        timings[source]=time.monotonic()-tick
        for q in qids:
            rows=store.get(q,source)
            if len(rows)!=500 or not np.isfinite(data.query_vector(q,source)).all():
                raise ValueError('Public native retrieval smoke failed')
    result=dict(passed=True,public_qids=qids,labels_used=False,timings=timings,seconds=time.monotonic()-started,
                source_pool_sizes={q:len(store.candidates(q)) for q in qids})
    write(RESULTS/'PUBLIC_PATH_SMOKE.json',result)
    store.close();return result


def sparse_check():
    from .data import TrigramReader
    data,store=Data(),SourceStore()
    qids=list(data.folds['fold_0'])[:12]+list(data.public)[:4]
    reader=TrigramReader(); times=[]; parity=[]
    progress=Progress('sparse_missing_source_benchmark',len(qids))
    for i,q in enumerate(qids):
        tick=time.monotonic();rows=reader.score(data.questions[q]);times.append(time.monotonic()-tick)
        expected=store.get(q,'trigram')
        if expected is not None:
            observed=[dict(doc_id=r['doc_id'],rank=r['rank'],score=r['raw_score']) for r in rows]
            if expected!=observed:
                raise ValueError('Persistent trigram scorer differs from original cached scorer')
            parity.append(q)
        progress.update(i+1,seconds_per_query=times[-1])
    reader.close();store.close()
    result=dict(passed=bool(parity),qids=qids,labels_used=False,parity_qids=parity,seconds_per_query=float(np.mean(times)),
                p50=float(np.median(times)),p90=float(np.quantile(times,.9)),times=times)
    write(RESULTS/'SPARSE_PATH_BENCHMARK.json',result);return result
