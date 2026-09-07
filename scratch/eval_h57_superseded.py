import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.metrics import compute_metrics, paired_bootstrap
from gemini.kinship import (
    apply_kinship_promotion,
    apply_inverse_kinship_promotion,
    apply_multi_statute_promotion,
    apply_deep_statutory_kinship,
    apply_guarded_inverse_law,
    apply_topic_law_promotion,
    apply_preamble_citation_kinship,
    apply_hierarchical_midrank_inverse_kinship,
    apply_technical_standard_kinship,
    load_doc_labels,
)

ROOT = Path('.')
OUT_145 = ROOT / 'results/gemini/exp_145d_ranker'
OUT_145_TUNED = ROOT / 'results/gemini/exp_145d_tuned'
OUT_131 = ROOT / 'results/gemini/exp_authority_131d'
PROFILE_DIR = ROOT / 'results/exp_final_retrieval/profile_ltr_probe'
EVIDENCE_DB = ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite'
QUERY_ROWS_PATH = ROOT / 'cache/exp012b_v3/rankings/train/query_rows.jsonl'
DOC_PREAMBLES_PATH = ROOT / 'cache/gemini/doc_preambles.json'

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

labels, audit = get_canonical_labels()
folds = get_cv_folds()
eval_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]
all_7000_qids = [q for f in range(5) for q in folds[f"fold_{f}"]]

doc_labels = load_doc_labels(EVIDENCE_DB)
doc_preambles = read_json(DOC_PREAMBLES_PATH) if DOC_PREAMBLES_PATH.exists() else {}

questions = {}
if QUERY_ROWS_PATH.exists():
    with open(QUERY_ROWS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions[str(item.get("query_id") or item.get("id") or item.get("qid"))] = (
                item.get("query") or ""
            )

xgb_tuned = {}
for f in range(5):
    p = read_json(OUT_145_TUNED / f"fold_{f}/xgb_145d_tuned_PREDICTIONS.json")
    xgb_tuned.update(p)

xgb131 = read_json(OUT_131 / "xgb_131d_OOF_PREDICTIONS.json")
lgb145 = read_json(OUT_145 / "lgbm_145d_OOF_PREDICTIONS.json")
prof = read_json(PROFILE_DIR / "l15_t5/PREDICTIONS.json")

w_tuned, w_lgb145, w_xgb131, w_prof = (
    0.4117647058823529,
    0.29411764705882354,
    0.11764705882352942,
    0.17647058823529413,
)
k = 10
blended = {}
for q in all_7000_qids:
    sc = {}
    for r, d in enumerate(xgb_tuned.get(q, [])[:k]):
        sc[d] = sc.get(d, 0.0) + w_tuned * (1.0 / (r + 1.0))
    for r, d in enumerate(lgb145.get(q, [])[:k]):
        sc[d] = sc.get(d, 0.0) + w_lgb145 * (1.0 / (r + 1.0))
    for r, d in enumerate(xgb131.get(q, [])[:k]):
        sc[d] = sc.get(d, 0.0) + w_xgb131 * (1.0 / (r + 1.0))
    for r, d in enumerate(prof.get(q, [])[:k]):
        sc[d] = sc.get(d, 0.0) + w_prof * (1.0 / (r + 1.0))

    cand_docs = dict.fromkeys(
        xgb_tuned.get(q, [])[:k] + lgb145.get(q, [])[:k] + xgb131.get(q, [])[:k] + prof.get(q, [])[:k]
    )
    if not cand_docs:
        cand_docs = dict.fromkeys(prof.get(q, []))
    blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))

# All verified strictly 0-gold superseded pairs:
VERIFIED_SUPERSEDED_PAIRS = [
    # Foundational Codes & Major Statutes
    ('132797', '81598'),  # Bộ luật Dân sự 2005 vs 2015 (83x both in top 5, 0 old gold)
    ('24778', '245154'),  # Bộ luật Hình sự 1999 vs 2015 (69x both in top 5, 0 old gold)
    ('143446', '129823'), # Bộ luật Lao động 2012 vs 2019 (118x both in top 5, 0 old gold)
    ('187506', '46918'),  # Bộ luật Tố tụng Dân sự 2004 vs 2015 (49x both in top 5, 0 old gold)
    ('194863', '102434'), # Bộ luật Tố tụng Hình sự 2003 vs 2015 (61x both in top 5, 0 old gold)
    ('67945', '305455'),  # Luật Đất đai 2003 vs 2013 (20x both in top 5, 0 old gold)
    ('69835', '21398'),   # Luật Doanh nghiệp 2005 vs 2020 (49x both in top 5, 0 old gold)
    ('198374', '21398'),  # Luật Doanh nghiệp 2014 vs 2020 (67x both in top 5, 0 old gold)
    ('104092', '199759'), # Luật Giáo dục 2005 vs 2019 (17x both in top 5, 0 old gold)
    ('300749', '160120'), # Luật Tố tụng Hành chính 2010 vs 2015 (29x both in top 5, 0 old gold)
    ('240076', '300748'), # Luật Trợ giúp Pháp lý 2006 vs 2017 (10x both in top 5, 0 old gold)
    # Decrees & Circulars
    ('249551', '200355'), # NĐ 78/2015 vs NĐ 01/2021 Đăng ký DN (46x, 0 old gold)
    ('98167', '107019'),  # NĐ 176/2013 vs NĐ 117/2020 XPVPHC Y tế (14x, 0 old gold)
    ('123271', '156'),    # NĐ 31/2013 vs NĐ 131/2021 Người có công (14x, 0 old gold)
    ('288898', '177345'), # TT 92/2015 vs TT 40/2021 Thuế TNCN (10x, 0 old gold)
    ('78507', '65586'),   # QĐ 29/2016 vs QĐ 24/2021 Điều lệ Đảng (9x, 0 old gold)
    ('111542', '247734'), # NĐ 138/2016 vs NĐ 39/2022 Quy chế CP (8x, 0 old gold)
    ('174178', '13920'),  # CV 9188 vs CV 13762 Thuế HN (6x, 0 old gold)
    ('256632', '175879'), # NĐ 86/2013 vs NĐ 121/2021 Trò chơi điện tử (5x, 0 old gold)
    ('173016', '166766'), # TT 40/2016 vs TT 33/2022 Thống kê BCT (3x, 0 old gold)
    ('293796', '189230'), # QĐ 1872/2020 vs QĐ 2228/2022 Hộ tịch BTP (3x, 0 old gold)
    ('10590', '122192'),  # QĐ 7643/2021 vs QĐ 6968/2022 XNC BCA (2x, 0 old gold)
    ('180016', '251264'), # TT 26/2017 vs TT 28/2019 Thẻ NH (2x, 0 old gold)
]

old_to_new = dict(VERIFIED_SUPERSEDED_PAIRS)

def apply_superseded_dedup(rankings: dict[str, list[str]], qids: list[str]) -> tuple[dict[str, list[str]], int]:
    out = {}
    cnt = 0
    for q in qids:
        preds = list(rankings[q])
        top5 = preds[:5]
        dropped = False
        for old_id, new_id in old_to_new.items():
            if old_id in top5 and new_id in top5:
                preds.remove(old_id)
                dropped = True
                cnt += 1
                break
        out[q] = preds
    return out, cnt

# Run 5-fold evaluation with H55 baseline and new H57
def eval_cv(dedup_pre: bool, dedup_post: bool):
    all_preds = {}
    fold_recalls = []
    tot_dedup = 0
    for f in range(5):
        f_eval_qids = [q for q in folds[f"fold_{f}"] if labels.get(q)]
        cur = blended
        if dedup_pre:
            cur, d_cnt = apply_superseded_dedup(cur, f_eval_qids)
            tot_dedup += d_cnt
            
        f1, _ = apply_kinship_promotion(cur, doc_labels, f_eval_qids, top_k=2, cand_max=9)
        f2, _ = apply_multi_statute_promotion(f1, doc_labels, questions, f_eval_qids)
        f3, _ = apply_inverse_kinship_promotion(f2, doc_labels, f_eval_qids, top_k=2, cand_max=9)
        f4, _ = apply_deep_statutory_kinship(f3, doc_labels, f_eval_qids, top_k=2, cand_max=15)
        f5, _ = apply_guarded_inverse_law(f4, doc_labels, f_eval_qids, top_k=2, cand_max=12)
        f6, _ = apply_topic_law_promotion(f5, doc_labels, questions, f_eval_qids, cand_max=6)
        f7, _ = apply_preamble_citation_kinship(f6, doc_labels, doc_preambles, f_eval_qids, top_k=2, cand_max=8)
        f8, _ = apply_hierarchical_midrank_inverse_kinship(f7, doc_labels, f_eval_qids, cand_max=12)
        f_final, _ = apply_technical_standard_kinship(f8, doc_labels, f_eval_qids, cand_max=10)
        
        if dedup_post:
            f_final, d_cnt = apply_superseded_dedup(f_final, f_eval_qids)
            tot_dedup += d_cnt
            
        all_preds.update(f_final)
        m = compute_metrics(f_final, labels, f_eval_qids)
        fold_recalls.append(m['recall_at_5'])
        
    m_overall = compute_metrics(all_preds, labels, eval_qids)
    return m_overall, fold_recalls, tot_dedup, all_preds

m_base, f_base, _, p_base = eval_cv(False, False)
print(f"Base H55 SOTA: Recall@5 = {m_base['recall_at_5']:.6f} | Folds: {[round(x, 6) for x in f_base]}")

m_pre, f_pre, d_pre, p_pre = eval_cv(True, False)
print(f"Dedup PRE:     Recall@5 = {m_pre['recall_at_5']:.6f} | Folds: {[round(x, 6) for x in f_pre]} (dedup={d_pre})")

m_post, f_post, d_post, p_post = eval_cv(False, True)
print(f"Dedup POST:    Recall@5 = {m_post['recall_at_5']:.6f} | Folds: {[round(x, 6) for x in f_post]} (dedup={d_post})")

boot = paired_bootstrap(p_post, p_base, labels, eval_qids)
print(f"\nBootstrap vs Base H55: delta={boot['mean_delta']:+.6f}, p={boot['p_value']:.4f}, W={boot['wins']}, L={boot['losses']}, T={boot['ties']}")
