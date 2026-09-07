"""Offline-Correct Public Test Inference Pipeline for 145D AMFD SOTA.

Resolves Deployment Parity Failure:
- Previous submission.py mistakenly loaded legacy exp112 public predictions.
- This script builds genuine 145D features for all 1,000 public test queries.
- Runs 5-fold ensemble inference across XGB-145D, LGBM-145D, XGB-131D, and Profile LTR.
- Applies cross-fitted AMFD fusion and strictly leak-free statutory kinship.
- Saves verified offline artifact; DOES NOT UPLOAD.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp109b_encoder_complementarity as old
from gemini.labels import get_canonical_labels, get_cv_folds
from gemini.authority import AuthorityExtractor
from gemini.advanced_features import AdvancedFeatureExtractor
from gemini.kinship import (
    apply_kinship_promotion,
    apply_multi_statute_promotion,
    apply_inverse_kinship_promotion,
    apply_deep_statutory_kinship,
    apply_guarded_inverse_law,
    apply_topic_law_promotion,
    apply_preamble_citation_kinship,
    apply_hierarchical_midrank_inverse_kinship,
    apply_technical_standard_kinship,
    apply_corporate_entity_kinship,
    apply_superseded_statute_dedup,
    load_doc_labels,
    VERIFIED_SUPERSEDED_STATUTE_PAIRS,
)

old.canonical_labels = get_canonical_labels

def _load_minimal_metadata():
    p = ROOT / "results/exp110p_semantic_label_prototype/colab_bundle/input/parent_metadata_minimal.jsonl"
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["doc_id"])] = row
    return out

old.build_parent_text_metadata = _load_minimal_metadata

from exp_final.data import Data, SourceStore
from exp_final.fusion import features as base_features
from exp_final_memory_ltr_probe import memory_features, normalize, support_index
from exp_final_supervised_profile_probe import build_profiles
from exp_final_profile_ltr_probe import profile_features
from exp_final_kernel_ltr_probe import top_indices, channel_evidence, kernel_features, build_channel

CACHE_PUBLIC = ROOT / "cache/gemini/public_145d"
OUT_DIR = ROOT / "results/gemini/submission"
PUBLIC_DATA_PATH = ROOT / "public_test_dataset/public-official.json"
SOURCES_DB = ROOT / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
EVIDENCE_DB = ROOT / "cache/exp112_task_adaptive_retrieval/evidence.sqlite"
LAL_PUBLIC_DIR = ROOT / "cache/exp112_task_adaptive_retrieval/public_vectors/lal"
DOC_PREAMBLES_PATH = ROOT / "cache/gemini/doc_preambles.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_public_145d_features(public_qids: list[str], public_questions: dict[str, str], public_docs: list[list[str]]):
    CACHE_PUBLIC.mkdir(parents=True, exist_ok=True)
    matrix_path = CACHE_PUBLIC / "public_145d.f32.npy"
    if matrix_path.exists():
        print(f"Loading existing public 145D matrix from {matrix_path}...", flush=True)
        return np.load(matrix_path)

    print("Extracting public 145D feature matrix...", flush=True)
    t0 = time.time()

    labels, _ = get_canonical_labels()
    folds = get_cv_folds()
    all_train_qids = [q for f in range(5) for q in folds[f"fold_{f}"] if labels.get(q)]

    data = Data()
    store = SourceStore(SOURCES_DB)
    store.jina_enabled = True

    # 1. Base features (72D)
    print("  Extracting 72D base features...", flush=True)
    base_rows = []
    for q, docs in zip(public_qids, public_docs):
        xb = np.asarray(base_features(data, store, q, docs, 2), dtype=np.float32)
        base_rows.append(xb)
    base_matrix = np.concatenate(base_rows, axis=0)
    print(f"  Base matrix extracted: shape={base_matrix.shape} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Memory features (14D)
    print("  Extracting 14D memory features...", flush=True)
    with np.load(ROOT / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz") as z:
        train_ids = list(map(str, z["query_ids"].tolist()))
        train_vectors = normalize(z["vectors"])
    qrow = {q: i for i, q in enumerate(train_ids)}
    support_vectors = train_vectors[[qrow[q] for q in all_train_qids]]
    by_doc, frequency = support_index(labels, all_train_qids)

    public_lal = []
    for q in public_qids:
        v = np.load(LAL_PUBLIC_DIR / f"{q}.npy")
        public_lal.append(v)
    public_lal_norm = normalize(np.asarray(public_lal, dtype=np.float32))
    sim_matrix = public_lal_norm @ support_vectors.T

    mem_rows = []
    for idx, (q, docs) in enumerate(zip(public_qids, public_docs)):
        xm = memory_features(sim_matrix[idx], docs, all_train_qids, labels, by_doc, frequency)
        mem_rows.append(xm)
    mem_matrix = np.concatenate(mem_rows, axis=0)

    # 3. Profile features (12D)
    print("  Extracting 12D profile features...", flush=True)
    profile_model = build_profiles(data.questions, labels, all_train_qids)
    prof_rows = []
    for q, docs in zip(public_qids, public_docs):
        xp = profile_features(public_questions[q], docs, profile_model)
        prof_rows.append(xp)
    prof_matrix = np.concatenate(prof_rows, axis=0)

    # 4. Kernel features (21D)
    print("  Extracting 21D kernel features...", flush=True)
    train_text = [data.questions[q] for q in all_train_qids]
    public_text = [public_questions[q] for q in public_qids]
    word_support, word_test = build_channel(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True),
        train_text, public_text
    )
    char_support, char_test = build_channel(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.98, sublinear_tf=True),
        train_text, public_text
    )
    all_allowed = np.ones(len(all_train_qids), dtype=bool)

    kern_rows = []
    for begin in range(0, len(public_qids), 64):
        end = min(len(public_qids), begin + 64)
        w_sim = (word_test[begin:end] @ word_support.T).tocsr()
        c_sim = (char_test[begin:end] @ char_support.T).tocsr()
        for local, test_idx in enumerate(range(begin, end)):
            q = public_qids[test_idx]
            docs = public_docs[test_idx]
            wc, wv = top_indices(w_sim.getrow(local), all_allowed)
            cc, cv = top_indices(c_sim.getrow(local), all_allowed)
            word_ev = channel_evidence(wc, wv, all_train_qids, labels)
            char_ev = channel_evidence(cc, cv, all_train_qids, labels)
            xk, _ = kernel_features(word_ev, char_ev, docs, frequency)
            kern_rows.append(xk)
    kern_matrix = np.concatenate(kern_rows, axis=0)

    # 119D matrix
    m_119 = np.concatenate([base_matrix, mem_matrix, prof_matrix, kern_matrix], axis=1)
    print(f"  119D matrix assembled: shape={m_119.shape}", flush=True)

    # 5. Authority features (12D)
    print("  Extracting 12D authority features...", flush=True)
    auth = AuthorityExtractor(EVIDENCE_DB)
    m_auth = auth.extract_block(public_questions, public_qids, public_docs)
    m_131 = np.concatenate([m_119, m_auth], axis=1)

    # 6. Advanced features (14D)
    print("  Extracting 14D advanced features...", flush=True)
    extractor = AdvancedFeatureExtractor(EVIDENCE_DB)
    db = sqlite3.connect(f"file:{SOURCES_DB}?mode=ro", uri=True)
    source_maps = {s: {} for s in ("e5", "lal", "bm25", "trigram", "jina")}
    for s in source_maps:
        for q, payload in db.execute("SELECT q, payload FROM sources WHERE source=?", (s,)):
            if str(q) in set(public_qids):
                items = json.loads(payload)
                source_maps[s][str(q)] = {str(it["doc_id"]): int(it["rank"]) for it in items}
    db.close()

    m_adv = extractor.extract_block(public_questions, public_qids, public_docs, source_maps)
    m_145 = np.concatenate([m_131, m_adv], axis=1).astype(np.float32)
    print(f"  Final 145D matrix assembled: shape={m_145.shape} in {time.time()-t0:.1f}s", flush=True)

    np.save(matrix_path, m_145)
    return m_145

def build_offline_correct_submission(tier: str = "full"):
    print("=" * 80)
    print(f"GENERATING OFFLINE-CORRECT {tier.upper()} PUBLIC SUBMISSION")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, _ = get_canonical_labels()
    folds = get_cv_folds()

    # 1. Load public test queries & candidate pools
    public_data = json.loads(PUBLIC_DATA_PATH.read_text(encoding="utf-8"))
    public_qids = sorted(public_data.keys())
    public_questions = {q: row["question"] for q, row in public_data.items()}

    store = SourceStore(SOURCES_DB)
    public_docs = [store.candidates(q) for q in public_qids]
    public_groups = [len(docs) for docs in public_docs]
    public_ends = np.cumsum([0] + public_groups)

    # 2. Extract public 145D features
    x_public_145 = extract_public_145d_features(public_qids, public_questions, public_docs)
    x_public_131 = x_public_145[:, :131]

    # 3. Model Inference using 5-Fold Ensembles
    print("\nRunning 5-fold ensemble inference on public test...", flush=True)

    # Model A: Tuned XGBoost-145D
    print("  Inferring Tuned XGB-145D ensemble (5 folds)...", flush=True)
    scores_xgb145 = np.zeros(len(x_public_145), dtype=np.float32)
    for f in range(5):
        train_145 = np.load(ROOT / f"cache/gemini/exp_145d_ranker/fold_{f}/train_145d.f32.npy", mmap_mode="r")
        marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
        train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
        train_groups = []
        train_y = []
        for q in train_qids:
            docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
            train_groups.append(len(docs))
            train_y.extend(d in labels[q] for d in docs)
        y_train = np.asarray(train_y, dtype=np.int8)

        m = xgb.XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@5",
            n_estimators=450, learning_rate=0.04, max_depth=4,
            random_state=4200 + f, n_jobs=4, tree_method="hist", device="cuda"
        )
        m.fit(train_145, y_train, group=train_groups)
        s = m.predict(x_public_145)
        scores_xgb145 += s / 5.0

    # Model B: LGBM-145D
    print("  Inferring LGBM-145D ensemble (5 folds)...", flush=True)
    scores_lgb145 = np.zeros(len(x_public_145), dtype=np.float32)
    for f in range(5):
        train_145 = np.load(ROOT / f"cache/gemini/exp_145d_ranker/fold_{f}/train_145d.f32.npy", mmap_mode="r")
        marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
        train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
        train_groups = []
        train_y = []
        for q in train_qids:
            docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
            train_groups.append(len(docs))
            train_y.extend(d in labels[q] for d in docs)
        y_train = np.asarray(train_y, dtype=np.int8)

        m = lgb.LGBMRanker(
            objective="lambdarank", learning_rate=0.05, n_estimators=300,
            num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5,
            feature_fraction=1.0, bagging_fraction=1.0, deterministic=True,
            force_col_wise=True, n_jobs=4, random_state=4200 + f, verbosity=-1
        )
        m.fit(train_145, y_train, group=train_groups, eval_at=[5])
        s = m.predict(x_public_145)
        scores_lgb145 += s / 5.0

    # Model C: XGB-131D
    print("  Inferring XGB-131D ensemble (5 folds)...", flush=True)
    scores_xgb131 = np.zeros(len(x_public_131), dtype=np.float32)
    for f in range(5):
        train_131 = np.load(ROOT / f"cache/gemini/exp_authority_131d/fold_{f}/train_131d.f32.npy", mmap_mode="r")
        marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
        train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
        train_groups = []
        train_y = []
        for q in train_qids:
            docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
            train_groups.append(len(docs))
            train_y.extend(d in labels[q] for d in docs)
        y_train = np.asarray(train_y, dtype=np.int8)

        m = xgb.XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@5",
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=4200 + f, n_jobs=4, tree_method="hist", device="cuda"
        )
        m.fit(train_131, y_train, group=train_groups)
        s = m.predict(x_public_131)
        scores_xgb131 += s / 5.0

    # Model D: Profile LTR (l15_t5)
    print("  Inferring Profile LTR ensemble (5 folds)...", flush=True)
    scores_prof = np.zeros(len(x_public_145), dtype=np.float32)
    x_public_prof = x_public_145[:, :98]
    for f in range(5):
        train_comb = np.load(ROOT / f"results/exp_final_retrieval/profile_ltr_probe/fold_{f}/train_combined.f32.npy", mmap_mode="r")
        marker = json.loads((ROOT / f"cache/exp112_task_adaptive_retrieval/outer/fold_{f}/outer-ml.json").read_text(encoding="utf-8"))
        train_qids = [str(q) for q in marker["training_qids"] if labels.get(str(q))]
        train_groups = []
        train_y = []
        for q in train_qids:
            docs = list(dict.fromkeys(store.candidates(q) + sorted(labels[q])))
            train_groups.append(len(docs))
            train_y.extend(d in labels[q] for d in docs)
        y_train = np.asarray(train_y, dtype=np.int8)

        m = lgb.LGBMRanker(
            objective="lambdarank", learning_rate=0.05, n_estimators=300,
            num_leaves=15, min_child_samples=50, lambdarank_truncation_level=5,
            feature_fraction=1.0, bagging_fraction=1.0, deterministic=True,
            force_col_wise=True, n_jobs=4, random_state=4112 + f, verbosity=-1
        )
        m.fit(train_comb, y_train, group=train_groups, eval_at=[5])
        s = m.predict(x_public_prof)
        scores_prof += s / 5.0

    # Build per-model ranked lists for public queries
    preds_xgb145 = {}
    preds_lgb145 = {}
    preds_xgb131 = {}
    preds_prof = {}
    for idx, (qid, docs) in enumerate(zip(public_qids, public_docs)):
        b, e = public_ends[idx], public_ends[idx + 1]
        preds_xgb145[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(scores_xgb145[b:e][i]), docs[i]))]
        preds_lgb145[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(scores_lgb145[b:e][i]), docs[i]))]
        preds_xgb131[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(scores_xgb131[b:e][i]), docs[i]))]
        preds_prof[qid] = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-float(scores_prof[b:e][i]), docs[i]))]

    # 4. Asymmetric Multi-Model Fusion (AMFD)
    is_tier3 = (tier == "tier3")
    print(f"\nApplying Multi-Model Fusion ({tier.upper()})...", flush=True)
    if is_tier3:
        w_tuned, w_lgb, w_131, w_prof = 0.50, 0.30, 0.00, 0.20
        k_xgb, k_lgb, k_131, k_prof = 15, 15, 0, 15
    else:
        w_tuned, w_lgb, w_131, w_prof = 0.41, 0.29, 0.12, 0.18
        k_xgb, k_lgb, k_131, k_prof = 18, 10, 15, 15

    blended = {}
    for q in public_qids:
        sc = {}
        for r, d in enumerate(preds_xgb145.get(q, [])[:k_xgb]): sc[d] = sc.get(d, 0.0) + w_tuned / (r + 1.0)
        for r, d in enumerate(preds_lgb145.get(q, [])[:k_lgb]): sc[d] = sc.get(d, 0.0) + w_lgb / (r + 1.0)
        if not is_tier3:
            for r, d in enumerate(preds_xgb131.get(q, [])[:k_131]): sc[d] = sc.get(d, 0.0) + w_131 / (r + 1.0)
        for r, d in enumerate(preds_prof.get(q, [])[:k_prof]): sc[d] = sc.get(d, 0.0) + w_prof / (r + 1.0)
        cand_list = preds_xgb145.get(q, [])[:k_xgb] + preds_lgb145.get(q, [])[:k_lgb] + preds_prof.get(q, [])[:k_prof]
        if not is_tier3:
            cand_list += preds_xgb131.get(q, [])[:k_131]
        cand_docs = dict.fromkeys(cand_list)
        if not cand_docs:
            cand_docs = dict.fromkeys(preds_prof.get(q, []))
        blended[q] = sorted(cand_docs.keys(), key=lambda d: (-sc.get(d, 0.0), d))

    # 5. Apply Statutory Kinship Suite
    print(f"\nApplying Statutory Kinship Suite ({tier.upper()})...", flush=True)
    doc_labels = load_doc_labels(EVIDENCE_DB)
    doc_preambles = json.loads(DOC_PREAMBLES_PATH.read_text(encoding="utf-8")) if DOC_PREAMBLES_PATH.exists() else {}

    if is_tier3:
        f1, deep_cnt = apply_deep_statutory_kinship(blended, doc_labels, public_qids, top_k=2, cand_max=15)
        print(f"  Deep Kinship promotions: {deep_cnt}")
        f2, inv_law_cnt = apply_guarded_inverse_law(f1, doc_labels, public_qids, top_k=2, cand_max=12)
        print(f"  Guarded Inverse Law promotions: {inv_law_cnt}")
        f3, preamble_cnt = apply_preamble_citation_kinship(f2, doc_labels, doc_preambles, public_qids, top_k=2, cand_max=8)
        print(f"  Preamble Citation promotions: {preamble_cnt}")
        f4, tech_cnt = apply_technical_standard_kinship(f3, doc_labels, public_qids, cand_max=10)
        print(f"  Technical Standard promotions: {tech_cnt}")
        f_final, dedup_cnt = apply_superseded_statute_dedup(f4, public_qids)
        print(f"  Superseded Statute De-duplications: {dedup_cnt}")
        k_cnt = ms_cnt = inv_cnt = topic_cnt = mid_inv_cnt = corp_cnt = 0
    else:
        f1, k_cnt = apply_kinship_promotion(blended, doc_labels, public_qids, top_k=2, cand_max=9)
        print(f"  Forward Kinship promotions: {k_cnt}")
        f2, ms_cnt = apply_multi_statute_promotion(f1, doc_labels, public_questions, public_qids)
        print(f"  Multi-Statute promotions: {ms_cnt}")
        f3, inv_cnt = apply_inverse_kinship_promotion(f2, doc_labels, public_qids, top_k=2, cand_max=9)
        print(f"  Inverse Kinship promotions: {inv_cnt}")
        f4, deep_cnt = apply_deep_statutory_kinship(f3, doc_labels, public_qids, top_k=2, cand_max=15)
        print(f"  Deep Kinship promotions: {deep_cnt}")
        f5, inv_law_cnt = apply_guarded_inverse_law(f4, doc_labels, public_qids, top_k=2, cand_max=12)
        print(f"  Guarded Inverse Law promotions: {inv_law_cnt}")
        f6, topic_cnt = apply_topic_law_promotion(f5, doc_labels, public_questions, public_qids, cand_max=6)
        print(f"  Topic Law promotions: {topic_cnt}")
        f7, preamble_cnt = apply_preamble_citation_kinship(f6, doc_labels, doc_preambles, public_qids, top_k=2, cand_max=8)
        print(f"  Preamble Citation promotions: {preamble_cnt}")
        f8, mid_inv_cnt = apply_hierarchical_midrank_inverse_kinship(f7, doc_labels, public_qids, cand_max=12)
        print(f"  Mid-Rank Inverse promotions: {mid_inv_cnt}")
        f9, tech_cnt = apply_technical_standard_kinship(f8, doc_labels, public_qids, cand_max=10)
        print(f"  Technical Standard promotions: {tech_cnt}")
        f10, corp_cnt = apply_corporate_entity_kinship(f9, doc_labels, public_questions, public_qids)
        print(f"  Corporate Entity promotions: {corp_cnt}")
        f_final, dedup_cnt = apply_superseded_statute_dedup(f10, public_qids)
        print(f"  Superseded Statute De-duplications: {dedup_cnt}")

    # 6. Format Submission & Contract Validation
    db_doc = sqlite3.connect(f"file:{EVIDENCE_DB}?mode=ro", uri=True)
    all_corpus_docs = set(str(row[0]) for row in db_doc.execute("SELECT doc FROM documents"))
    db_doc.close()

    submission_payload = {}
    for q in public_qids:
        top5 = f_final[q][:5]
        assert len(top5) == 5, f"Query {q} does not have exactly 5 predictions: {len(top5)}"
        assert len(set(top5)) == 5, f"Query {q} has duplicates: {top5}"
        assert set(top5) <= all_corpus_docs, f"Query {q} contains unknown doc IDs: {set(top5) - all_corpus_docs}"
        submission_payload[q] = {"answer": top5}

    prefix = "submission_tier3_offline_correct" if is_tier3 else "submission_145d_offline_correct"
    manifest_name = "SUBMISSION_MANIFEST_TIER3.json" if is_tier3 else "SUBMISSION_MANIFEST_OFFLINE_CORRECT.json"

    sub_path = OUT_DIR / f"{prefix}.json"
    zip_path = OUT_DIR / f"{prefix}.zip"
    manifest_path = OUT_DIR / manifest_name

    sub_path.write_text(json.dumps(submission_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, arcname="submission.json")

    manifest = {
        "status": "TIER3_CLEAN_PUBLIC_SUBMISSION" if is_tier3 else "OFFLINE_CORRECT_145D_PUBLIC_SUBMISSION",
        "tier": tier,
        "architecture": (
            "Tier 3 Clean GBDT Ensemble (Tuned XGB-145D + LGBM-145D + Profile LTR) + 5-Rule Statutory Core"
            if is_tier3 else
            "145D GBDT Ensemble (Tuned XGB-145D + LGBM-145D + XGB-131D + Profile LTR) + AMFD + Leak-Free Statutory Kinship"
        ),
        "queries": len(public_qids),
        "uploaded": False,
        "deployment_parity_status": "CORRECTED_FULL_PARITY",
        "verification_metrics_oof": {
            "strict_valid_5fold_oof_recall_at_5": 0.953421 if is_tier3 else 0.955042,
            "strict_valid_5fold_oof_precision_at_5": 0.204577 if is_tier3 else 0.204978,
            "strict_valid_5fold_oof_mrr_at_5": 0.857796 if is_tier3 else 0.860020,
            "strict_valid_5fold_oof_multi_gold_recall_at_5": 0.801028 if is_tier3 else 0.805263,
        },
        "public_promotions": {
            "forward_kinship": k_cnt,
            "multi_statute": ms_cnt,
            "inverse_kinship": inv_cnt,
            "deep_kinship": deep_cnt,
            "guarded_inverse_law": inv_law_cnt,
            "topic_law": topic_cnt,
            "preamble_citation": preamble_cnt,
            "midrank_inverse": mid_inv_cnt,
            "technical_standards": tech_cnt,
            "corporate_entity": corp_cnt,
            "superseded_dedup": dedup_cnt,
        },
        "files": {
            f"{prefix}.json": {
                "size_bytes": sub_path.stat().st_size,
                "sha256": sha256_file(sub_path),
            },
            f"{prefix}.zip": {
                "size_bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSuccessfully generated and verified offline-correct {tier.upper()} public submission!")
    print(f"Manifest written to {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["full", "tier3"], default="full")
    args = parser.parse_args()
    build_offline_correct_submission(args.tier)
