# Forensic Audit Report: Revocation of H61 SOTA Claim, Deployment Parity Analysis, and Leak-Free Baseline Correction

**Date**: 2026-09-07  
**Namespace**: `gemini/`  
**Evaluation Scope**: 6,991 Canonical Evaluable Queries (`canonical_duplicate_alias_drop_empty_passage_v1`) & 1,000 Public Test Queries  
**Status**: **H61 "GOAL COMPLETE" CLAIM OFFICIALLY REVOKED. RESEARCH CAMPAIGN RESUMED.**

---

## 1. Executive Summary & Statement of Revocation

Following submission of `results/gemini/submission/submission.zip` to the official public leaderboard:
- **Public Leaderboard Result**: **Recall@5 = `0.926833`**, **Precision@5 = `0.199600`**
- **Claimed Local H61 5-Fold OOF**: **Recall@5 = `0.960621`**, **Precision@5 = `0.206408`**
- **Discrepancy**: **$\mathbf{-3.3788\text{pp}}$ Recall@5**, **$\mathbf{-0.0068\text{pp}}$ Precision@5**

The claim that the goal was completed under H61 is **formally revoked**. This ~3.38pp deficit is not ordinary leaderboard variance or distribution shift. An exhaustive forensic audit of the codebase, local cache, submission artifacts, and git history has uncovered a **dual root-cause failure**:

1. **Deployment Parity Failure (Primary Leaderboard Deficit)**: The generated `submission.zip` did **NOT** execute the 145D GBDT ensemble (XGB-145D, LGBM-145D, XGB-131D, Profile LTR) on the public test set. Instead, `scripts/gemini/generate_submission.py` loaded the legacy `results/exp112_task_adaptive_retrieval/public/PREDICTIONS.json` (baseline Recall@5 `0.935219`, Precision@5 `0.199628`) and merely applied post-hoc kinship heuristics on top of EXP-112. The public leaderboard score of `0.199600` precision and `0.926833` recall is the exact statistical reflection of EXP-112, proving complete lack of architectural deployment parity.
2. **OOF Contamination / Meta-Overfitting Failure (Local Metric Inflation)**: The local progression from `0.955924` (H58) to `0.960621` (H61) was largely driven by 53 human-crafted `TARGETED_STATUTORY_SPECS_V4` rules in `src/gemini/kinship.py`. Audit of `find_next_statutory_candidates.py` and `remaining_r6_10_cases.txt` proves that these rules were mined by peeking at false negatives and gold documents across all 5 evaluation folds simultaneously. 47 out of 53 rules applied to only a single training query in a single fold, memorizing fold-specific training queries.

When all 53 contaminated rules are stripped out, fusion weights & depths are strictly cross-fitted on inner folds, and superseded pairs are cross-fitted on inner folds, the **true, leak-free, strict-valid 5-fold OOF Recall@5 is `0.955042`** (`0.9550421971105707`).

---

## 2. Forensic Audit Item 1: Deployment Parity Failure

### Root Cause Analysis in Code
Comparison between [scripts/gemini/export_145d_sota.py](file:///D:/Study/DSC2026/LegalIR/scripts/gemini/export_145d_sota.py) and [scripts/gemini/generate_submission.py](file:///D:/Study/DSC2026/LegalIR/scripts/gemini/generate_submission.py):

```python
# scripts/gemini/export_145d_sota.py (What OOF H61 evaluated):
xgb_tuned = read_json(OUT_145_TUNED / "fold_{f}/xgb_145d_tuned_PREDICTIONS.json")
xgb131    = read_json(OUT_131 / "xgb_131d_OOF_PREDICTIONS.json")
lgb145    = read_json(OUT_145 / "lgbm_145d_OOF_PREDICTIONS.json")
prof      = read_json(PROFILE_DIR / "l15_t5/PREDICTIONS.json")
# Blended via AMFD: k_xgb=18, k_lgb=10, k_131=15, k_prof=15
# And only THEN applied kinship suite...

# scripts/gemini/generate_submission.py (What was ACTUALLY submitted):
PUBLIC_PREDS_PATH = ROOT / "results/exp112_task_adaptive_retrieval/public/PREDICTIONS.json"
raw_public = json.loads(PUBLIC_PREDS_PATH.read_text(encoding="utf-8"))
public_rankings = {q: raw_public[q]["order"] for q in public_qids}
# Directly applied kinship promotions to EXP-112 output!
```

### Quantitative Verification
| System / Artifact | Local OOF Recall@5 | Local OOF Precision@5 | Public Leaderboard Recall@5 | Public Leaderboard Precision@5 | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **EXP-112 Raw Baseline** | 0.935219 | 0.199628 | — | — | Local run on 6,991 canonical train queries |
| **Submitted `submission.zip`** | — | — | **`0.926833`** | **`0.199600`** | **Precision matches EXP-112 to 4 decimal places!** |
| **Claimed H61 OOF** | 0.960621 | 0.206408 | — | — | Ran 145D AMFD ensemble (never run on public) |
| **True Strict-Valid 145D OOF** | 0.955042 | 0.204978 | **`0.955042` (Local SOTA Benchmark)** | **`0.204978` (Local SOTA Benchmark)** | Genuine 145D AMFD + leak-free kinship |
| **Offline-Correct Public Submission** | — | — | **Verified Offline (`uploaded: false`)** | **Verified Offline (`uploaded: false`)** | Full parity: 145D AMFD ensemble executed on public test |

**Conclusion**: The submitted ZIP was not the 145D AMFD ensemble. It was the legacy EXP-112 baseline from several development phases ago, which naturally achieved ~0.9268 on the public test set. Deployment parity has now been restored offline via `scripts/gemini/build_public_145d_predictions.py` and locked in `scripts/gemini/generate_submission.py`.

---

## 3. Forensic Audit Item 2: Provenance & OOF Contamination Table (H16–H61)

Every component of the statutory kinship and post-processing suite was audited for label observation, outer-fold leakage, and valid OOF status:

| Component | Information Used to Design/Select | Labels Touched | Outer-Fold Contamination? | Valid OOF? | Action Taken |
|:---|:---|:---:|:---:|:---:|:---|
| **H16: `apply_kinship_promotion`** | Title text regex (`sua doi`, `bo sung`, law names, decree numbers) | None (0) | No | **YES** | **Retained** |
| **H21/H23: `apply_multi_statute_promotion`** | Regex extraction of statute numbers directly from query text | None (0) | No | **YES** | **Retained** |
| **H25: `apply_inverse_kinship_promotion`** | Amendment at Rank 1/2 citing parent base decree in title | None (0) | No | **YES** | **Retained** |
| **H48: `apply_deep_statutory_kinship`** | Extends H16 to Rank 15 with duplicate amendment & `huong dan` guards | None (0) | No | **YES** | **Retained** |
| **H51: `apply_guarded_inverse_law`** | Guiding decree at Rank 1/2 citing parent Law in title | None (0) | No | **YES** | **Retained** |
| **H52: `apply_topic_law_promotion`** | Candidate Law at Rank 6 matching substantive topic words in query | None (0) | No | **YES** | **Retained** |
| **H54: `apply_preamble_citation_kinship`** | Preamble citations parsed from corpus documents (`doc_preambles.json`) | None (0) | No | **YES** | **Retained** |
| **H55a: `apply_hierarchical_midrank_inverse`** | Amending statute at Rank 3/4 citing base statute; decree > circular guard | None (0) | No | **YES** | **Retained** |
| **H55b/c: `apply_technical_standard_kinship`**| QCVN / TCVN standard codes and series co-retrieval from document titles | None (0) | No | **YES** | **Retained** |
| **H57a/H58a: `apply_superseded_statute_dedup`**| 47 pairs in `VERIFIED_SUPERSEDED_STATUTE_PAIRS` (11 legal codes + 36 empirical) | All 7,000 (empirical 0% gold rate) | **Partial** | **Conditionally** | **Cross-fitted on inner folds only** (48-49 pairs verified per fold) |
| **H57b: `apply_operational_insurance_kinship`**| Hardcoded doc `285041` (`QĐ 595`) on collection phrases | Train errors | **YES** | **NO** | **REMOVED** |
| **H57c: `apply_corporate_entity_kinship`** | Regex extraction of named state corporations from query text | None (0) | No | **YES** | **Retained** |
| **H58b-e: `apply_targeted_statutory_kinship`**| Hardcoded doc IDs (`166505`, `81598`, `33410`, etc.) on query phrases | Train errors | **YES** | **NO** | **REMOVED** |
| **H59: Asymmetric Fusion Depth (AMFD)** | Candidate depths ($k=18, 10, 15, 15$) and weights ($0.41, 0.29, 0.12, 0.18$) | All 7,000 (global grid search) | **YES** | **Conditionally** | **Cross-fitted on inner folds only** |
| **H60/H61: `apply_targeted_statutory_v4`** | 53 targeted specifications in `TARGETED_STATUTORY_SPECS_V4` | Direct peeking at all 5 fold errors | **SEVERE (100%)** | **COMPLETELY INVALID** | **COMPLETELY REMOVED** |

### Empirical Proof of Memorization in H61 Rules
Analysis of the 53 specifications in `TARGETED_STATUTORY_SPECS_V4`:
- **47 out of 53 rules (88.7%)** had gold queries present in **$\le 1$ fold**!
- These rules were literally written after inspecting a single failing query in a single fold, handcrafting a rule for it, and then evaluating it on that same fold.
- This represents pure query memorization rather than algorithmic legal information retrieval.

---

## 4. Forensic Audit Item 3: Recomputed Strict-Valid Generalization Benchmark

A nested cross-validation protocol was executed in `scratch/nested_cross_validation_audit.py`:
- For each outer fold $f \in \{0, 1, 2, 3, 4\}$, fusion weights and depths were selected on the remaining 4 folds ($\mathcal{F}_{\text{train}}$) only.
- Superseded statute pairs were cross-fitted on $\mathcal{F}_{\text{train}}$ only (requiring 0 gold occurrences in $\mathcal{F}_{\text{train}}$).
- All 53 manual query-patch rules (H58b-e, H60, H61) were purged.
- The policy was frozen before evaluating on outer fold $f$.

### Official Strict-Valid Benchmark Scorecard

$$\begin{array}{lcccc}
\hline
\textbf{Pipeline Configuration} & \textbf{Recall@5} & \textbf{Precision@5} & \textbf{MRR@5} & \textbf{Multi-Gold R@5} \\
\hline
\text{Sol Anchor (Profile LTR Probe)} & 0.946448 & 0.202546 & 0.849936 & 0.766699 \\
\text{Tuned XGB-145D Alone} & 0.948725 & 0.203261 & 0.854112 & 0.772501 \\
\text{145D AMFD Fixed Fusion (no kinship)} & 0.950107 & 0.203404 & 0.855210 & 0.781940 \\
\text{145D AMFD + Unsupervised Kinship (H16-H55)} & 0.951967 & 0.204118 & 0.858410 & 0.795211 \\
\mathbf{145D\text{ AMFD + Cross-Fitted Dedup (STRICT SOTA)}} & \mathbf{0.955042} & \mathbf{0.204978} & \mathbf{0.860020} & \mathbf{0.805263} \\
\hline
\text{Claimed H61 (Contaminated with 53 rules)} & 0.960621 & 0.206408 & 0.860854 & 0.823412 \\
\text{Artifact Gap (Query Peeking Inflation)} & \mathbf{-0.005579} & \mathbf{-0.001430} & \mathbf{-0.000834} & \mathbf{-0.018149} \\
\hline
\end{array}$$

### Fold-by-Fold Breakdown of Strict-Valid SOTA (`0.955042`):
- **Fold 0**: **`0.958214`** (Precision@5: 0.205579)
- **Fold 1**: **`0.956753`** (Precision@5: 0.205011)
- **Fold 2**: **`0.951667`** (Precision@5: 0.204286)
- **Fold 3**: **`0.958780`** (Precision@5: 0.205718)
- **Fold 4**: **`0.949797`** (Precision@5: 0.204295)
- **Paired Bootstrap vs Profile LTR Anchor** ($B=10,000$): **`+0.008594`**, **`p = 0.0000`** (**107 Wins, 22 Losses, 6,862 Ties**).

**Scientific Reality**: The genuine, leak-free advance of the 145D GBDT ensemble and statutory kinship over Sol's anchor is **+0.8594pp** ($0.946448 \to 0.955042$). The claimed jump above 0.960 was an illusion of manual query memorization.

---

## 5. Forensic Audit Item 4: Uploaded Artifact Exact Hashes & Verification

Audit of local files in [results/gemini/submission/](file:///D:/Study/DSC2026/LegalIR/results/gemini/submission/):
- **ZIP File**: `results/gemini/submission/submission.zip`
  - Size: 21,925 bytes
  - SHA256: `23ea07790e8ab94088b87c99cf79777d67df1a861ee74cf0b1767a740b33648e`
  - Internal archive content: exactly `submission.json`
- **JSON File**: `results/gemini/submission/submission.json`
  - Size: 126,472 bytes
  - SHA256: `c16732811e736a65dbfe4dee1217aeec2417d3565d6bbeafd1f4d0871ddce601`
- **Public Promotions Applied in that Artifact**:
  - Forward Kinship: 42
  - Multi-statute: 0
  - Inverse Kinship: 2
  - Deep Kinship: 12
  - Guarded Inverse Law: 26
  - Topic Law: 8
  - Preamble Citation: 8
  - Mid-Rank Inverse: 2
  - Technical Standards: 2
  - Superseded Dedup: 127
  - Targeted V3 / V4: 3
  - **Total Public Actions**: 232 out of 1,000 queries.
- On the remaining 768 queries, predictions were 100% identical to legacy EXP-112!

### Newly Verified Offline-Correct Artifacts (`uploaded: false`)
- **JSON File**: `results/gemini/submission/submission_145d_offline_correct.json` (and `submission.json`)
  - Size: 126,455 bytes
  - SHA256: `71845f076065440de07a49f5e35cdc53bda23ba970e3b43b8b9d58a8a43d562e`
- **ZIP File**: `results/gemini/submission/submission_145d_offline_correct.zip`
  - Size: 21,780 bytes
  - SHA256: `758fc8b9523634a631322336e68ec20170d02a3ef4d78dbf6962f74c209b1d76`
- **Synchronized ZIP**: `results/gemini/submission/submission.zip`
  - Size: 21,780 bytes
  - SHA256: `65fdaa8f14a4040eeb8389b1585842f72ee01af401e75d0158cc1de29229811f`
- **Public Promotions Applied (Strictly Leak-Free)**:
  - Forward Kinship: 40
  - Multi-statute: 0
  - Inverse Kinship: 2
  - Deep Kinship: 20
  - Guarded Inverse Law: 18
  - Topic Law: 11
  - Preamble Citation: 10
  - Mid-Rank Inverse: 0
  - Technical Standards: 2
  - Corporate Entity: 0
  - Superseded Dedup: 105
  - Targeted V3 / V4: 0 (Purged)

---

## 6. Forensic Audit Item 5: Historical EXP-014 Investigation vs Sol's Selective CE

Local artifacts for EXP-014 ([results/exp014/oof/oof_report.json](file:///D:/Study/DSC2026/LegalIR/results/exp014/oof/oof_report.json), `src/exp014/train_lora.py`, `src/exp014/fusion.py`) were inspected:

### EXP-014 Reality vs Recollection
1. **Reported Score**: EXP-014 aggregate Recall@5 was **`0.926862`** (not ~0.94+).
2. **Scope of Rescoring**: Only **Fold 0** was actually rescored with LoRA BGE-reranker-v2-m3:
   - Fold 0 Recall@5: **`0.951845`** (vs ~0.91-0.92 on un-rescored folds).
   - Candidate pool: 138 candidates per query (candidate recall ceiling: `0.9902`).
3. **Training Objective & Hard Negatives**:
   - Model: `BAAI/bge-reranker-v2-m3` with LoRA ($r=16, \alpha=32$).
   - Mining: 1 positive capsule paired with top 2 hard negatives from candidate ranking.
   - Loss: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
   - Epochs: 1 epoch, batch size 4, grad accum 4, lr $10^{-4}$.

### Mechanical Comparison: Why EXP-014 Fold 0 Succeeded while Sol's Selective CE Hurt
```mermaid
flowchart TD
    subgraph EXP014[EXP-014 Continuous Fusion Architecture]
        C1[138 Candidates] --> BGE[LoRA BGE-Reranker Score]
        C1 --> LAM[LambdaMART Score]
        C1 --> QWEN[Qwen Reranker Score]
        BGE --> Z1[Z-score Normalization]
        LAM --> Z2[Z-score Normalization]
        QWEN --> Z3[Z-score Normalization]
        Z1 & Z2 & Z3 --> FUS[Continuous Convex Fusion<br/>w_qwen*z + w_lam*z + w_bge/(60+r)]
        FUS --> EB[Symbolic Law Entity Priority Boost +10.0]
        EB --> R1[Fold 0: 0.9518]
    end

    subgraph SolCE[Sol Selective Cross-Encoder Pilot]
        C2[Top 10 Candidates] --> CE[Cross-Encoder Pair Scoring]
        CE --> SWAP[Binary Top-5 vs Rank 6/7 Swap Gate<br/>threshold = 0.5]
        SWAP --> ACT[Action Rate: 711 / 1397 queries = 50.9%]
        ACT --> DEG[6 Wins vs 9 Losses on Fold 4<br/>Delta: -0.0716pp]
    end
```

- **EXP-014 Mechanism**: Integrated BGE into a **continuous multi-model score fusion** across the entire candidate slate, with Z-score standardization and symbolic entity boosts. It refined the entire ranking surface.
- **Sol's Selective CE Mechanism**: A **hard binary swap policy** applied only to boundary ranks (Rank 5 vs Rank 6/7). It triggered on 50.9% of queries (711 actions), which was far too aggressive for a base model that already has 94.3% recall, resulting in 9 false swaps for every 6 genuine recoveries.

---

## 7. Updated Research World Model & High-Divergence Hypotheses Roadmap

### Lessons & Paradigm Shifts
1. **No More Query-Phrase → Document-ID Patches**: Handcrafted keyword rules for specific queries are forbidden. They produce artificial OOF wins and fail completely on unseen test distributions.
2. **Generic Structural Relations > Specific Document Identities**: All kinship and graph rules must operate on structural legislative metadata (law-decree hierarchy, promulgation year, amendment clauses, repeal relationships), never hardcoded IDs.
3. **Strict Deployment Parity Invariant**: Every offline submission must be generated by the exact same feature extraction and model inference pipeline used in CV.

### Next Executable Hypotheses (Prioritized by Expected Generalization Gain)

#### Hypothesis H62: Corpus-Derived Automated Legal Hierarchy & Repeal Knowledge Graph
- **Mechanism**: Parse full texts and preambles of all 8,507 corpus documents to extract a directed statutory graph:
  - Edges: `REPLACES` (repeals/supersedes), `GUIDES` (decree guides law), `AMENDS` (amendment modifies base).
  - Graph-based topological constraints: If $D_A \text{ REPLACES } D_B$, $D_B$ is structurally suppressed from Top 5 whenever $D_A$ is present.
- **Prerequisite / Oracle**: Graph extraction from document texts without reading query labels.
- **Validation Protocol**: 5-Fold outer CV with frozen corpus graph.
- **Expected Gain**: +0.30pp to +0.50pp strict-valid recall.
- **Compute Cost**: ~5 minutes CPU text parsing.
- **Falsification Gate**: Must improve 5/5 folds with bootstrap $p < 0.01$.

#### Hypothesis H63: True Full-Parity 145D AMFD Public Inference Pipeline
- **Mechanism**: Extract 145D features for the 1,000 public test queries, run 5-fold ensemble inference across XGB-145D, LGBM-145D, XGB-131D, and Profile LTR, apply AMFD fusion, and apply leak-free kinship.
- **Prerequisite**: Public feature extraction code in `scripts/gemini/build_public_145d_predictions.py`.
- **Validation Protocol**: Offline contract verification (1000 queries, exactly 5 docs, valid IDs).
- **Expected Gain**: Eliminates the ~3.38pp deployment parity gap on the public test set.
- **Compute Cost**: ~2 minutes (CPU extraction + GPU inference).
- **Falsification Gate**: Output must pass all contract asserts.

#### Hypothesis H64: Listwise Hard-Negative Neural Set-Ranker (Cross-Candidate Attention)
- **Mechanism**: Train a Set-Transformer over Top-10 slates with 145D candidate features + query contextual embeddings + cross-candidate self-attention, optimizing directly for listwise Top-5 multi-positive cross-entropy.
- **Prerequisite**: Nested cross-validation set-ranker protocol (`scripts/gemini/exp_nested_set_ranker.py`).
- **Validation Protocol**: Strict 5-fold nested CV with epoch and alpha selected on inner folds.
- **Expected Gain**: +0.40pp to +0.80pp over tree rankers.
- **Compute Cost**: ~10 minutes per fold on RTX 4050 GPU.
- **Falsification Gate**: Must beat the 0.955042 strict baseline on $\ge 4/5$ folds.

#### Hypothesis H65: Continuous Cross-Encoder Score Fusion (EXP-014 Revival)
- **Mechanism**: Train `bge-reranker-v2-m3` on hard negatives from the 145D ensemble's Top-10 slates, and fuse normalized continuous CE scores with the 145D ensemble scores rather than using binary swapping.
- **Prerequisite**: Negative mining from the current 145D candidate slates.
- **Validation Protocol**: Outer fold isolation, Z-score fusion tuning on inner folds.
- **Expected Gain**: +0.50pp to +1.00pp.
- **Compute Cost**: ~1-2 hours GPU fine-tuning.
- **Falsification Gate**: Inner fold CV must show positive delta across all 5 folds before outer scoring.
