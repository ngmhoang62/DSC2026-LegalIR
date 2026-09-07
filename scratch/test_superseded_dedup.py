import sys
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from gemini.labels import get_canonical_labels
from gemini.kinship import load_doc_labels

ROOT = Path('.')
doc_labels = load_doc_labels(ROOT / 'cache/exp112_task_adaptive_retrieval/evidence.sqlite')
preds = json.load(open('results/gemini/best_ensemble/BEST_ENSEMBLE_PREDICTIONS.json'))
labels, meta = get_canonical_labels()

eval_qids = [qid for qid, gold in labels.items() if gold]

# Pure zero-loss superseded pairs identified:
# (old_id, new_id)
ZERO_LOSS_PAIRS = [
    ('143446', '129823'), # Bộ luật Lao động 2012 vs 2019 (118 queries, 0 old gold)
    ('249551', '200355'), # NĐ 78/2015 vs NĐ 01/2021 Đăng ký DN (46 queries, 0 old gold)
    ('300749', '160120'), # Luật Tố tụng Hành chính 2010 vs 2015 (29 queries, 0 old gold)
    ('98167', '107019'),  # NĐ 176/2013 vs NĐ 117/2020 XPVPHC Y tế (14 queries, 0 old gold)
    ('123271', '156'),    # NĐ 31/2013 vs NĐ 131/2021 Người có công (14 queries, 0 old gold)
    ('288898', '177345'), # TT 92/2015 vs TT 40/2021 Thuế TNCN (10 queries, 0 old gold)
    ('78507', '65586'),   # QĐ 29/2016 vs QĐ 24/2021 Điều lệ Đảng (9 queries, 0 old gold)
    ('111542', '247734'), # NĐ 138/2016 vs NĐ 39/2022 Quy chế CP (8 queries, 0 old gold)
    ('174178', '13920'),  # CV 9188 vs CV 13762 Thuế HN (6 queries, 0 old gold)
    ('256632', '175879'), # NĐ 86/2013 vs NĐ 121/2021 Trò chơi điện tử (5 queries, 0 old gold)
    ('173016', '166766'), # TT 40/2016 vs TT 33/2022 Thống kê BCT (3 queries, 0 old gold)
    ('293796', '189230'), # QĐ 1872/2020 vs QĐ 2228/2022 Hộ tịch BTP (3 queries, 0 old gold)
    ('10590', '122192'),  # QĐ 7643/2021 vs QĐ 6968/2022 XNC BCA (2 queries, 0 old gold)
    ('180016', '251264'), # TT 26/2017 vs TT 28/2019 Thẻ NH (2 queries, 0 old gold)
]

old_to_new = {old: new for old, new in ZERO_LOSS_PAIRS}

# Let's test simply shifting: remove old_id from Top 5 if new_id is also in Top 5
wins = 0
losses = 0
win_qids = []

for qid in eval_qids:
    top = list(preds[qid])
    top5 = top[:5]
    golds = labels[qid]
    
    # Check if any old_id is in top5 while its new_id is also in top5
    replaced = False
    for old_id, new_id in old_to_new.items():
        if old_id in top5 and new_id in top5:
            # Drop old_id from ranking, everything after shifts up!
            top.remove(old_id)
            replaced = True
            break
            
    if replaced:
        new_top5 = top[:5]
        old_hits = len(set(top5) & golds)
        new_hits = len(set(new_top5) & golds)
        
        # Recall delta
        old_rec = old_hits / len(golds)
        new_rec = new_hits / len(golds)
        
        if new_rec > old_rec:
            wins += 1
            win_qids.append((qid, old_rec, new_rec, top5, new_top5))
        elif new_rec < old_rec:
            losses += 1
            print(f"LOSS on QID {qid}: {old_rec} -> {new_rec}")

print(f"\n--- RESULTS OF SUPERSEDED DOCUMENT DE-DUPLICATION ---")
print(f"Total Wins: {wins}")
print(f"Total Losses: {losses}")
print(f"Net Gain: +{wins - losses}")

for qid, old_rec, new_rec, old_top5, new_top5 in win_qids[:10]:
    golds = labels[qid]
    print(f"\nWin on QID {qid}: Recall {old_rec:.3f} -> {new_rec:.3f}")
    promoted_doc = new_top5[4]
    print(f"  Promoted to Rank 5: {promoted_doc} ({doc_labels.get(promoted_doc, '')}) [IS_GOLD={promoted_doc in golds}]")
