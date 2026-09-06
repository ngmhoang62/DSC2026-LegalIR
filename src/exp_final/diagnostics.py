"""Read-only diagnostics; never used to retune an outer selection."""
import numpy as np
from .contracts import report_metrics


def ranking_diagnostics(data, store, rows, anchor, adapted=None):
    qids = list(rows)
    gold = {q: data.gold[q] for q in qids}
    original = {q: data.original[q] for q in qids}
    def measured(p):
        # A frozen specialist may legitimately retrieve no document for a query
        # (the trigram source currently does so for a small number of queries).
        # Production predictions still require 1--5 IDs; only this read-only
        # source diagnostic scores an empty retrieval as zero.
        return report_metrics(p, original, gold, allow_empty_predictions=True)
    pools, coverage, oracle, movement = [], [], [], []
    buckets = dict.fromkeys(('1-5','6-10','11-50','51-100','outside100','outside_candidate_union'), 0)
    sources = {s: {} for s in ('e5','lal','bm25','trigram')}
    exclusive = dict.fromkeys(sources, 0)
    for q in qids:
        ranked = rows[q]['order']; universe = set(ranked)
        pools.append(len(ranked))
        g = gold[q]
        if g:
            coverage.append(len(g & universe)/len(g))
            oracle.append(min(5, len(g & universe))/len(g))
        before = set(anchor[q]['order'][:5]) & g
        after = set(ranked[:5]) & g
        movement.append(dict(qid=q, recovered=sorted(after-before), lost=sorted(before-after)))
        positions = {d:i for i,d in enumerate(ranked,1)}
        for d in g:
            r = positions.get(d)
            key = 'outside_candidate_union' if r is None else '1-5' if r<=5 else '6-10' if r<=10 else '11-50' if r<=50 else '51-100' if r<=100 else 'outside100'
            buckets[key] += 1
        ranks = store.rankings(q)
        for s in sources:
            sources[s][q] = ranks[s]
            others = set().union(*(set(ranks[t][:5]) for t in sources if t != s))
            exclusive[s] += len(g & (set(ranks[s][:5])-others))
    result = dict(frozen_sources={s: measured(v) for s,v in sources.items()},
                  frozen_source_metric_contract="diagnostic_empty_retrieval_scores_zero",
                  frozen_source_empty_queries={s: sum(not v[q] for q in qids) for s,v in sources.items()},
                  candidate_pool_size=dict(min=min(pools),mean=float(np.mean(pools)),max=max(pools)),
                  canonical_candidate_coverage=float(np.mean(coverage)) if coverage else None,
                  canonical_top5_oracle=float(np.mean(oracle)) if oracle else None,
                  canonical_gold_rank_buckets=buckets, gold_movement=movement,
                  canonical_source_exclusive_top5_hits=exclusive)
    if adapted is not None:
        result['adapted_alone'] = measured({q:r['order'][:200] for q,r in adapted.items()})
        values = [r['frozen_query_cosine'] for r in adapted.values() if 'frozen_query_cosine' in r]
        result['query_drift_mean'] = float(1-np.mean(values)) if values else None
    return result
