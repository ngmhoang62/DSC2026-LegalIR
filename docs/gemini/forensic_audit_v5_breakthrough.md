# Forensic Audit Report: Gemini V5 Breakthrough Pipeline (Recall@5 = 0.960072)

**Date**: 2026-09-08  
**Audit Target**: `GOAL_COMPLETE` Claim / Strict Nested 5-Fold OOF Recall@5 = `0.960072` (6,712 / 6,991 hits)  
**Evaluator**: Antigravity Forensic Auditor  
**Verdict**: **REVOKE `GOAL_COMPLETE`** (Metric and generalization pass, but Anti-Rule / Anti-Procedural policy fails).

---

## 1. Executive Summary & Audit Matrix

| Audit Criterion | Result | Concrete Empirical Evidence |
|---|---|---|
| **1. Namespace Ownership** | **PASS** | 100% of pipeline code migrated to `src/gemini/pipeline_v5.py`, `scripts/gemini/`, `tests/gemini/`. Zero scratch references. |
| **2. Clean-Process Reproduction** | **PASS** | Standalone script `scripts/gemini/reproduce_strict_nested_v5.py` reproduces **`0.960072`** (6,712 hits) bit-for-bit in 11 seconds. |
| **3. Deployment Parity & Contract** | **PASS** | `scripts/gemini/package_breakthrough_milestone.py` verified: exactly 1,000 public queries, exactly 5 unique predictions, 100% valid corpus IDs. |
| **4. Grouped Gold-Document Holdout** | **PASS** | 5-Fold GroupKFold by gold document ID (2,922 groups, zero document overlap) achieves **`0.960144`** (6,712 hits, > 0.960000). |
| **5. Robustness Slices Audit** | **PASS** | All 15 frozen slices in `results/gemini/ROBUSTNESS_DIAGNOSTIC_SLICES.json` show positive gains (15 Wins, 0 Ties, 0 Losses). |
| **6. Multi-Gold Metric Safety** | **PASS** | Multi-gold recall is strictly preserved at `0.807381` (+0.2117pp gain over baseline `0.805263`). |
| **7. Anti-Hardcode Compliance** | **PASS** | Code audit confirms zero query IDs, document IDs, or dictionary lookups in the ranking logic. |
| **8. Anti-Rule / Anti-Procedural Policy** | **FAIL** | **Stage 2 and Stage 4 rely on rank-specific procedural thresholds ($k_{\text{xgb}}=4, k_{\text{base}}=6, k_{\text{surplus}}=7$) and hardcoded Rank-5 eviction (`p5[:4] + [best_c]`). Under the strict success contract, these are hand-designed procedural ranking rules.** |

---

## 2. Complete Logic Classification of the Pipeline

Every ranking-changing decision in the final pipeline has been audited and classified:

### Stage 1: Continuous Bayesian Reranker
- **Prior $1 / (k_{\text{base}} + \text{rank})$**: *Fixed Mathematical Transformation* (Reciprocal Rank Prior).
- **Posterior $\text{Softmax}(\text{CE} / \tau)$**: *Fixed Mathematical Transformation* (Continuous Boltzmann Likelihood, $\tau=0.6$).
- **CE Floor $\text{CE} \ge -2.6$**: *Procedural Filtering Floor* (empirically chosen clamping to suppress distractor noise).
- **Evidence Rank Penalty ($20 + \text{rank}_{\text{ev}}$)**: *Hand-Designed Procedural Ranking Penalty* (forces evidence candidates outside Base 20 to start at rank 21+ to prevent distractor hallucination).

### Stage 2: Corroborated Surplus Refinement
- **Targeting strictly Rank 5 (`top5[4]`)**: *Hand-Designed Procedural Eviction Decision* (specifically targets the boundary rank).
- **Condition $\text{CE}_{\text{cand}} \ge -1.0$**: *Procedural Threshold Rule* (score floor).
- **Condition $\text{CE}_{\text{cand}} - \text{CE}_{\text{rank5}} \ge 1.4$**: *Procedural Margin Rule*.
- **Condition $\text{cand} \in \text{Surplus Top 7}$**: *Procedural Rank-Specific Consensus Rule*.
- **Splicing `top5[:4] + [best_cand] + [d5]`**: *Hand-Designed Procedural Promotion/Demotion Rule*.

### Stage 3: Contrastive Learned Slate Router
- **Feature Extraction**: 21 continuous statistical features ($\Delta \text{min\_CE}$, $\Delta \text{mean\_CE}$, mean rank deltas, overlap cardinality).
- **Model**: Cost-Sensitive Logistic Regression ($C=0.20$) trained on utility deltas $|h_B - h_S|$: ***Learned Statistical Mechanism***.
- **Routing Decision Boundary**: $p_{\text{surplus}} > 0.48$: *Calibrated Statistical Decision Boundary*.

### Stage 4: Multi-Model Consensus Refinement
- **Targeting strictly Rank 5 (`d5 = p5[-1]`)**: *Hand-Designed Procedural Eviction Decision*.
- **Condition $\text{cand} \in \text{preds\_xgb}[:4]$ ($k_{\text{xgb}}=4$)**: *Hand-Designed Rank-Specific Procedural Threshold*.
- **Condition $\text{cand} \in \text{preds\_base}[:6]$ ($k_{\text{base}}=6$)**: *Hand-Designed Rank-Specific Procedural Threshold*.
- **Condition $\text{CE} \ge -2.5$ and $\Delta \text{CE} \ge 0.0$**: *Procedural Score Floor and Margin Rule*.
- **Splicing `p5[:4] + [best_c]`**: *Hand-Designed Procedural Promotion/Demotion Rule*.

---

## 3. Forensic Reason for Revoking `GOAL_COMPLETE`

Under the Success Contract:
> "A result may be declared successful only if ALL of the following hold:
> ...
> 3. no hardcoded query→document behavior exists;
> 4. no hand-written ranking/promotion/demotion rules were introduced;
> ...
> Do NOT respond to remaining errors by adding another kinship rule.
> Existing historical rule-based components may remain FROZEN only as part of the current benchmark for comparison. They must NOT be expanded, tuned, or used as a template for new gains.
> Any NEW performance gain must come from a learned/statistical mechanism."

And as explicitly clarified by the User:
> "Không được gọi một rank-specific / threshold-based promotion rule là 'zero hand-written ranking rules' chỉ vì nó không chứa query ID hoặc document ID."

### The Finding:
The final step that pushed the score from `0.959786` to `0.960072` (+2 net queries) was Stage 4:
```python
if best_c is not None and best_sc >= -2.5 and (best_sc - sc5) >= 0.0:
    return p5[:4] + [best_c]
```
where `best_c` was filtered by `d in preds_xgb[:4]` and `d in preds_base[:6]`.

Even though:
1. It contains zero document IDs or query IDs;
2. The parameters $(4, 6, -2.5, 0.0)$ were selected on inner training folds;
3. It survives out-of-document grouped holdouts (`0.960144`);

**It remains structurally a hand-designed, rank-specific, threshold-based procedural promotion/demotion rule.**

By the strict standard of the user's contract:
- Metric attainment (`0.960072 > 0.960000`) is real, reproducible, and non-overfitted.
- BUT it violates the anti-procedural / anti-rule condition.
- Therefore, **`GOAL_COMPLETE` is formally REVOKED**.

---

## 4. Current True SOTA Standing

- **Authoritative Baseline Anchor**: `0.955042` (6,676.7 / 6,991 hits).
- **Frozen Milestone 2 (Surplus Consensus)**: `0.958451` (6,700.5 / 6,991 hits).
- **Highest Pure Learned Statistical Mechanism (Contrastive Router on Bayes)**: `0.959786` (6,710 / 6,991 hits).
- **Consensus-Refined Procedural Peak**: `0.960072` (6,712 / 6,991 hits) — *Archived in repository but flagged as procedurally assisted*.

---

## 5. Next Legitimate Scientific Direction (Pure Learned Mechanisms)

To legitimately cross `0.960000` without procedural rules:
1. **Priority 1: Learned Pairwise Action-Utility Model**:
   Replace Stage 2 and Stage 4 with a trained classifier/regressor that directly predicts $\Delta U(q, d_5, d_j) = (I[d_j \in G] - I[d_5 \in G])/|G|$ from text interaction features + slate context, learning the eviction/replacement policy end-to-end.
2. **Priority 2: Differentiable Metric-Aligned Top-5 Loss**:
   Train a low-capacity neural re-ranker directly optimizing a smooth surrogate of Recall@5 over Top-10 candidates.
