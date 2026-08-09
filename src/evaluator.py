import numpy as np
from typing import Dict, List

def eval_retrieval(y_pred: Dict[str, Dict[str, List[str]]], y_true: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Replicates official scoring.py retrieval evaluation.
    y_pred: dict mapping question_id -> {"answer": [doc_id1, doc_id2, ...]}
    y_true: dict mapping question_id -> [doc_id1, doc_id2, ...]
    """
    ids_truth = list(y_true.keys())

    recall_list = []
    precision_list = []

    for k in ids_truth:
        pred_answers = y_pred.get(k, {}).get('answer', [])
        true_answers = y_true[k]

        # Official condition check: 1 <= len(pred) <= 5
        if 0 < len(pred_answers) <= 5:
            intersection = set(true_answers) & set(pred_answers)
            recall = len(intersection) / len(true_answers)
            precision = len(intersection) / len(pred_answers)
        else:
            recall = 0.0
            precision = 0.0

        recall_list.append(recall)
        precision_list.append(precision)

    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    mean_precision = float(np.mean(precision_list)) if precision_list else 0.0

    return {
        "recall": mean_recall,
        "precision": mean_precision
    }
