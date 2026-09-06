"""Rescreen dual-case LAL sources against the strongest existing Fold-4 base."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from exp_final_dual_case_adapter_probe import (
    bounded_screen_rrf, choice_oracle, metrics, normalize, prototype_orders, read, write,
)

CACHE = ROOT / "cache/exp_final_retrieval/dual_case_adapter_probe/fold_4"
OUT = ROOT / "results/exp_final_retrieval/dual_case_adapter_probe/STRONG_BASE_RESCREEN.json"


def load_rankings(folder, qids):
    return {qid: list(map(str, read(folder / f"{qid}.json")["order"])) for qid in qids}


def main():
    import exp109b_encoder_complementarity as old
    from exp_final.data import Data

    labels, _ = old.canonical_labels(); folds = read(ROOT / "cache/cv_folds.json")
    train_qids = [q for fold in range(4) for q in folds[f"fold_{fold}"] if labels.get(q)]
    test_qids = [q for q in folds["fold_4"] if labels.get(q)]
    data = Data()
    bases = {
        "incumbent_memory": read(ROOT / "results/exp_final_retrieval/memory_ltr_probe/fold_4/PREDICTIONS.json"),
        "profile_l15": read(ROOT / "results/exp_final_retrieval/profile_ltr_probe/l15_t5/PREDICTIONS.json"),
    }
    report = {
        "status": "COMPLETE_DUAL_CASE_STRONG_BASE_RESCREEN",
        "scope": "Exposed Fold 4 OOF diagnostic; no new training or outer claim.",
        "bases": {name: metrics(rows, labels, test_qids) for name, rows in bases.items()},
        "epochs": {},
    }
    for epoch in (1, 2):
        support = normalize(np.load(CACHE / f"vectors-epoch-{epoch}-train/vectors.f32.npy", mmap_mode="r"))
        target = normalize(np.load(CACHE / f"vectors-epoch-{epoch}-test/vectors.f32.npy", mmap_mode="r"))
        modes = prototype_orders(target, support, train_qids, labels, data.doc_ids)
        content = load_rankings(CACHE / f"content-epoch-{epoch}", test_qids)
        epoch_report = {"sources": {"content": metrics(content, labels, test_qids)}, "bases": {}}
        for base_name, base in bases.items():
            trials = []
            source_oracles = {"content": choice_oracle(base, content, labels, test_qids)}
            for mode, orders in modes.items():
                prototype = {qid: orders[i] for i, qid in enumerate(test_qids)}
                source_oracles[f"prototype_{mode}"] = choice_oracle(base, prototype, labels, test_qids)
                for content_weight in (0.0, .05, .10, .15):
                    for prototype_weight in (.025, .05, .10, .15):
                        if content_weight + prototype_weight > .30:
                            continue
                        ranking = {
                            qid: bounded_screen_rrf(
                                {"base": base[qid], "content": content[qid], "prototype": prototype[qid]},
                                {"base": 1 - content_weight - prototype_weight,
                                 "content": content_weight, "prototype": prototype_weight},
                            ) for qid in test_qids
                        }
                        measured = metrics(ranking, labels, test_qids)
                        trials.append({
                            "prototype_mode": mode, "content_weight": content_weight,
                            "prototype_weight": prototype_weight, "metrics": measured,
                            "delta_vs_base": measured["recall_at_5"] - report["bases"][base_name]["recall_at_5"],
                        })
            trials.sort(key=lambda row: (
                row["metrics"]["recall_at_5"], row["metrics"]["precision_at_5"],
                row["metrics"]["multi_gold_recall_at_5"], row["metrics"]["mrr_at_5"],
            ), reverse=True)
            epoch_report["bases"][base_name] = {"source_oracles": source_oracles, "top_trials": trials[:15]}
        report["epochs"][str(epoch)] = epoch_report
    write(OUT, report); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
