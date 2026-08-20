import json
import lightgbm as lgb
from pathlib import Path
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

from exp012b_core import load_answers, read_jsonl
from exp012b_retrieval import evaluate_rankings
from exp012b_tuning import load_folds
from exp013b_core import rank_ids
from exp013b_ranker import _features, FEATURE_COLUMNS

folds = load_folds(Path('cache/cv_folds.json'))
answers = load_answers(Path('public_test_dataset/train.json'))
candidates_records = list(read_jsonl(Path('cache/exp013b_cascade/candidates/train/train_candidates.jsonl')))
qids_fold0 = {str(v) for v in folds['fold_0']}

# Load Qwen/BGE scores for Fold 0
qwen_fold0 = {str(r['qid']): {str(x['doc_id']): float(x['qwen_score']) for x in r['scores']} for r in read_jsonl(Path('cache/exp013b_cascade/qwen/fold_0/scores.jsonl')) if str(r['qid']) in qids_fold0}

# Extract base features
feature_records, type_codes = _features(candidates_records)

# Build training set for fold 0 validation
# Train on folds 1..4, evaluate on fold 0
train_qids = set()
for f in range(1, 5):
    train_qids.update(str(v) for v in folds[f'fold_{f}'])

X_train, y_train, q_train = [], [], []
for qid in sorted(train_qids):
    gold = answers[qid]
    rows = feature_records[qid]
    for r in rows:
        feat = [r[col] for col in FEATURE_COLUMNS]
        label = 1 if r['doc_id'] in gold else 0
        X_train.append(feat)
        y_train.append(label)
    q_train.append(len(rows))

X_val, y_val, q_val, val_doc_ids = [], [], [], []
for qid in sorted(qids_fold0):
    gold = answers[qid]
    rows = feature_records[qid]
    doc_list = []
    for r in rows:
        feat = [r[col] for col in FEATURE_COLUMNS]
        label = 1 if r['doc_id'] in gold else 0
        X_val.append(feat)
        y_val.append(label)
        doc_list.append(r['doc_id'])
    q_val.append(len(rows))
    val_doc_ids.append((qid, doc_list))

# Train base LambdaMART
model = lgb.LGBMRanker(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=15,
    random_state=42,
    importance_type="gain"
)
model.fit(
    np.array(X_train), np.array(y_train), group=q_train,
    eval_set=[(np.array(X_val), np.array(y_val))], eval_group=[q_val],
    eval_at=[5], eval_metric="ndcg",
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_preds = model.predict(np.array(X_val))
predictions = {}
offset = 0
for qid, doc_list in val_doc_ids:
    cnt = len(doc_list)
    scores = dict(zip(doc_list, val_preds[offset:offset+cnt]))
    predictions[qid] = rank_ids(scores)
    offset += cnt

gold_val = {qid: answers[qid] for qid in qids_fold0}
metrics_base = evaluate_rankings(predictions, gold_val, ks=(5, 10, 24))
print("Tuned LambdaMART (26 features) Fold 0:", metrics_base)
