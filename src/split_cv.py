import json
import os
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(train_path: str, output_path: str, n_splits: int = 5, seed: int = 42):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    q_ids = list(train_data.keys())
    # Stratify by ground truth answer length
    labels = [len(train_data[qid]['answer']) for qid in q_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(q_ids, labels)):
        val_qids = [q_ids[i] for i in val_idx]
        folds[f"fold_{fold}"] = val_qids

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(folds, f, ensure_ascii=False, indent=4)

    print(f"Created {n_splits}-fold Stratified CV splits at {output_path}")
    for fold, val_qids in folds.items():
        print(f"  {fold}: {len(val_qids)} validation queries")

if __name__ == "__main__":
    train_file = "d:/Study/DSC2026/LegalIR/public_test_dataset/train.json"
    out_file = "d:/Study/DSC2026/LegalIR/cache/cv_folds.json"
    create_stratified_folds(train_file, out_file)
