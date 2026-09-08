"""
Vietnamese LegalIR Pipeline V5: Hierarchical Consensus Reranking and Slate Routing.

Provides pure functional implementations of all 4 stages:
- Stage 1: Continuous Bayesian Reranking
- Stage 2: Corroborated Surplus Consensus Refinement
- Stage 3: Contrastive Learned Slate Routing
- Stage 4: Multi-Model Consensus Refinement
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def generate_bayes_slate(
    q: str,
    preds_base: dict[str, list[str]],
    evidence_top50: dict[str, list[str]],
    hf_scores: dict[str, dict[str, float]],
    k_base: float = 7.0,
    beta: float = 0.09,
    fl: float = -2.6,
    tau: float = 0.6,
    surplus_preds: dict[str, list[str]] | None = None,
    surplus_k: int = 7,
    min_ce: float = -1.0,
    min_margin: float = 1.4,
) -> list[str]:
    """Generate Stage 1 Bayes slate + Stage 2 Corroborated Surplus Refinement."""
    cands_base = list(preds_base[q][:20])
    b20_set = set(cands_base)
    ev_c = evidence_top50.get(q, [])
    new_docs = [d for d in ev_c if d not in b20_set][:2]
    pool = cands_base + new_docs

    sc = np.array([hf_scores.get(q, {}).get(d, -10.0) for d in pool], dtype=np.float32)
    eff_sc = np.where(sc >= fl, sc, -10.0)
    max_sc = np.max(eff_sc)
    exp_s = np.exp((eff_sc - max_sc) / tau)
    p_ce = exp_s / (np.sum(exp_s) + 1e-9)

    ranks = np.zeros(len(pool), dtype=np.float32)
    for r in range(len(cands_base)):
        ranks[r] = r + 1
    for r, d in enumerate(new_docs):
        ev_r = ev_c.index(d) + 1 if d in ev_c else 25
        ranks[len(cands_base) + r] = 20 + ev_r

    tot = 1.0 / (k_base + ranks) + beta * p_ce
    top_idx = np.argsort(-tot)
    cur_p = [pool[i] for i in top_idx]

    if surplus_preds is None:
        return cur_p

    # Stage 2: Corroborated Surplus Refinement
    top5 = cur_p[:5]
    rest = cur_p[5:22]
    d5 = top5[4]
    sc5 = hf_scores.get(q, {}).get(d5, -10.0)
    surplus_window = set(surplus_preds.get(q, [])[:surplus_k])

    best_cand = None
    best_cand_sc = -10.0
    for d in rest:
        sc_d = hf_scores.get(q, {}).get(d, -10.0)
        if sc_d >= min_ce and (sc_d - sc5) >= min_margin:
            if d in surplus_window:
                if sc_d > best_cand_sc:
                    best_cand_sc = sc_d
                    best_cand = d

    if best_cand is not None:
        return top5[:4] + [best_cand] + [d5] + [d for d in cur_p[5:] if d != best_cand]
    return cur_p


def extract_contrastive_features(
    q: str,
    preds_base: dict[str, list[str]],
    preds_surplus: dict[str, list[str]],
    preds_v3: dict[str, list[str]],
    hf_scores: dict[str, dict[str, float]],
    questions: dict[str, str],
) -> list[float]:
    """Extract 21 contrastive features for routing between V3 and Surplus."""
    txt = questions.get(q, "")
    q_len = len(txt.split())

    b5 = preds_base[q][:5]
    s5 = preds_surplus[q][:5]
    v5 = preds_v3[q][:5]

    sc_b5 = [hf_scores.get(q, {}).get(d, -10.0) for d in b5]
    sc_s5 = [hf_scores.get(q, {}).get(d, -10.0) for d in s5]
    sc_v5 = [hf_scores.get(q, {}).get(d, -10.0) for d in v5]

    rest = preds_base[q][5:15]
    sc_rest = [hf_scores.get(q, {}).get(d, -10.0) for d in rest]

    v_not_s = [d for d in v5 if d not in set(s5)]
    s_not_v = [d for d in s5 if d not in set(v5)]

    sc_v_diff = [hf_scores.get(q, {}).get(d, -10.0) for d in v_not_s] if v_not_s else [-10.0]
    sc_s_diff = [hf_scores.get(q, {}).get(d, -10.0) for d in s_not_v] if s_not_v else [-10.0]

    base_ranks_v_diff = [preds_base[q].index(d) + 1 if d in preds_base[q] else 30 for d in v_not_s] if v_not_s else [0]
    base_ranks_s_diff = [preds_base[q].index(d) + 1 if d in preds_base[q] else 30 for d in s_not_v] if s_not_v else [0]

    s5_min = float(np.min(sc_s5))
    v5_min = float(np.min(sc_v5))
    s5_mean = float(np.mean(sc_s5))
    v5_mean = float(np.mean(sc_v5))

    s_diff_m = float(np.mean(sc_s_diff))
    v_diff_m = float(np.mean(sc_v_diff))

    return [
        q_len,
        float(np.mean(sc_b5)),
        float(np.min(sc_b5)),
        s5_mean,
        s5_min,
        v5_mean,
        v5_min,
        s5_min - v5_min,
        s5_mean - v5_mean,
        float(np.mean(sc_rest)) if sc_rest else -10.0,
        float(np.max(sc_rest)) if sc_rest else -10.0,
        s_diff_m,
        v_diff_m,
        s_diff_m - v_diff_m,
        float(s_diff_m > v_diff_m),
        float(np.mean(base_ranks_s_diff)),
        float(np.mean(base_ranks_v_diff)),
        float(np.mean(base_ranks_s_diff) - np.mean(base_ranks_v_diff)),
        len(set(b5) & set(s5)),
        len(set(b5) & set(v5)),
        len(set(v5) & set(s5)),
    ]


def run_contrastive_router(
    eval_qids: list[str],
    qid_to_fold: dict[str, int],
    preds_v3: dict[str, list[str]],
    preds_surplus: dict[str, list[str]],
    X_norm: np.ndarray,
    labels: dict[str, set[str]],
    C: float = 0.20,
    thresh: float = 0.48,
) -> dict[str, list[str]]:
    """Stage 3: Run Cost-Sensitive Contrastive Router under strict nested 5-fold CV."""
    y_all = np.zeros(len(eval_qids), dtype=np.int32)
    w_all = np.zeros(len(eval_qids), dtype=np.float32)

    for i, q in enumerate(eval_qids):
        g = set(labels[q])
        hb = len(g & set(preds_v3[q][:5])) / len(g)
        hs = len(g & set(preds_surplus[q][:5])) / len(g)
        if hb >= hs:
            y_all[i] = 1
        else:
            y_all[i] = 0
        w_all[i] = abs(hb - hs)

    oof_preds_v4 = {}
    for test_f in range(5):
        inner_idx = [i for i, q in enumerate(eval_qids) if qid_to_fold[q] != test_f]
        test_idx = [i for i, q in enumerate(eval_qids) if qid_to_fold[q] == test_f]

        clf = LogisticRegression(C=C, random_state=42 + test_f, max_iter=1000)
        clf.fit(X_norm[inner_idx], y_all[inner_idx], sample_weight=w_all[inner_idx] + 0.001)

        probs = clf.predict_proba(X_norm[test_idx])[:, 0]

        for idx, p_surplus in zip(test_idx, probs):
            q = eval_qids[idx]
            if set(preds_v3[q][:5]) != set(preds_surplus[q][:5]) and p_surplus > thresh:
                oof_preds_v4[q] = preds_surplus[q][:5]
            else:
                oof_preds_v4[q] = preds_v3[q][:5]

    return oof_preds_v4


def apply_consensus_refinement(
    p5: list[str],
    q: str,
    preds_base: dict[str, list[str]],
    preds_xgb: dict[str, list[str]],
    hf_scores: dict[str, dict[str, float]],
    k_xgb: int = 4,
    k_base: int = 6,
    min_ce: float = -2.5,
    min_margin: float = 0.0,
) -> list[str]:
    """Stage 4: Apply multi-model consensus refinement to slate."""
    d5 = p5[-1]
    sc5 = hf_scores.get(q, {}).get(d5, -10.0)

    xgb_set = set(preds_xgb.get(q, [])[:k_xgb])
    cands = [d for d in preds_base[q][:k_base] if d in xgb_set and d not in set(p5)]

    best_c = None
    best_sc = -999.0
    for c in cands:
        sc = hf_scores.get(q, {}).get(c, -10.0)
        if sc > best_sc:
            best_sc = sc
            best_c = c

    if best_c is not None and best_sc >= min_ce and (best_sc - sc5) >= min_margin:
        return p5[:4] + [best_c]
    return p5
