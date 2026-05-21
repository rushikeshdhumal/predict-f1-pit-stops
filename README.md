# F1 Pit Stop Prediction — ML Rigor in a Kaggle Competition

Binary classification: predict whether a Formula 1 driver will pit on the next lap. This project showcases **rigorous ML engineering** — validated on 439k rows across 13 notebooks covering validation design, feature engineering, model diversity, hyperparameter tuning, and ensemble methods.

**Best LB AUC: 0.92845** (rank avg LGBM + CatBoost)  
**Best CV AUC: 0.9032** (rank avg LGBM NB12 + CatBoost NB11)  
**Baseline (Logistic Regression): 0.7717 AUC**  
**Leaderboard top: 0.95488** — gap: 0.026  
**Kaggle Competition:** Playground Series S6E5

---

## Why This Matters: Validation Design

The core insight is **preventing data leakage in temporally-structured data.** Naive k-fold CV places laps from the same race in both train and validation — the model memorizes race-specific context (circuit characteristics, tyre degradation) rather than generalizing.

**Solution:** GroupKFold by `Race + Year` ensures no race appears in both train and validation across all 13 notebooks. This simulates the true deployment scenario: predict pit stops in races the model has never seen.

**CV→LB calibration:** Two actual submissions confirmed a consistent +0.025–0.026 CV→LB boost. This is not a model quality signal — it reflects that the test races happen to be slightly easier to rank than CV validation races. All projections use +0.025.

---

## Notebook Pipeline

| # | Title | Key output | Best AUC |
|---|-------|-----------|----------|
| 01 | EDA | Domain understanding — tyre cliffs, class imbalance, compound pit rates | — |
| 02 | Feature Engineering | `train_features.parquet` — 29-feature baseline | — |
| 03 | Validation Strategy | `fold_assignments.parquet` — GroupKFold by Race×Year | — |
| 04 | Baseline Models | Logistic Regression, Decision Tree | DT: 0.8678 |
| 05 | Gradient Boosting | LGBM + XGBoost Optuna-tuned | LGBM: 0.8558 |
| 06 | Interpretability | SHAP rankings, PDP plots | — |
| 07 | Calibration | Platt scaling, Brier score analysis | — |
| 08 | Ensemble Diagnostics | Spearman ρ, stacking V1/V2 | 0.8569 |
| 09 | Submission v001 | `submission_v001_stacking_v2.csv` | **LB 0.90610** |
| 10 | Advanced Features | `train_features_v2.parquet` — 38 features | 0.8987 |
| 11 | Model Diversity | CatBoost-Plain, RF, ET, DART | **CB: 0.9016** |
| 12 | Re-Tuning | LGBM Optuna 200-trial multi-fold | **LGBM: 0.9024** |
| 13 | Advanced Ensemble | `submission_v002_advanced_ensemble.csv` | **LB 0.92845** |

---

## Key Technical Decisions

### Feature Engineering: 38 Features, Temporal Integrity

Built in two phases. All lag/rolling features grouped by `(Race, Year, Driver, Stint)` — never across stint boundaries. Target encodings computed inside each CV fold.

**Phase 1 — 29 features (Notebook 02):**

| Category | Examples | Domain Insight |
|----------|----------|----------------|
| **Tyre Age** | `TyreLife_normalized_by_compound`, `TyreLife_sq` | Cliff thresholds from P(pit) S-curves: SOFT=13, MED=49, HARD=61. Not median stint length. |
| **Degradation** | `Cumul_Deg_winsorized`, `Degradation_rate` | Winsorized at [−205, +122] (1st/99th pct). Rate-of-change signals the cliff. |
| **Trajectory** | `LapTime_lag{1-3}`, `LapTime_Delta_lag{1-3}` | Momentum signal. Grouped within `(Race, Year, Driver, Stint)`. |
| **Interactions** | `TyreLife × laps_remaining`, `Degradation × RaceProgress` | Joint effects; TyreLife×laps_rem is #2 by SHAP. |
| **Context** | `Stint`, `laps_remaining`, `Position` | `Stint` is strongest predictor (SHAP 0.86, 2× any other feature). |
| **Encodings** | `Race_target_encoded`, `Driver_target_encoded` | Fold-aware only — prevents AUC leakage of 0.01–0.02. |

**Phase 2 — 9 additional features (Notebook 10, +0.044 CV AUC gain):**

The dominant gain came from a single tier — **Position Volatility (+0.0391 CV AUC):**

```python
df['abs_position_change'] = df['Position_Change'].abs()
df['pos_change_rolling_std_3'] = (
    df.groupby(['Race','Year','Driver'])['Position_Change']
    .rolling(3, min_periods=1).std()
    .droplevel([0,1,2]).sort_index().fillna(0)
)
```

Grouped by `(Race, Year, Driver)` only — **intentionally spans stint boundaries** — because a driver erratically losing positions at stint-end is on degraded tyres regardless of when they pitted. Fold std dropped from ±0.020 to ±0.012 after adding these features.

Other kept features: `prime_pit_window` (RaceProgress 40–70% has 2.6× pit rate), `PitStop_lag1`, `laps_to_driver_end`, `TyreLife_x_compound_ordinal`.

**Incremental tier evaluation:** Each tier was tested cumulatively (not all at once) to isolate marginal contribution and apply a +0.001 keep/drop threshold. Tiers 5 and 6 were dropped after confirming negative marginal contribution.

---

### Model Evaluation: Understanding Failure

**Phase 1 (5 architectures, Notebooks 04–05):**

| Model | CV AUC | Interpretation |
|-------|--------|----------------|
| Logistic Regression | 0.7717 | Baseline; coefficients sign-correct |
| **Decision Tree** | **0.8678** | Best performer — adjusts depth where LGBM over-regularizes |
| LightGBM (Optuna, 100 trials) | 0.8558 | `reg_alpha=9.79` dominant; folds 1–2 stop at ~35 trees |
| XGBoost | 0.8492 | Validates LGBM rankings; 0.0066 gap |
| CatBoost (ordered TE) | ~0.78 | **Fails:** Ordered target encoding within GroupKFold produces near-empty folds |

**Why DT beats LGBM?** Fold composition, not features. Folds 1–2 are structurally difficult — LGBM's gradient descent stops at 35 trees on early stopping; DT adjusts depth naturally. This was confirmed via SHAP analysis: the GBM gap is not a tuning problem.

**Phase 2 (model diversity, Notebook 11):**

| Model | CV AUC | ρ vs LGBM | Verdict |
|-------|--------|-----------|---------|
| LGBM reference | 0.8978 | — | baseline |
| Random Forest | 0.8780 | 0.946 | Excluded — too correlated |
| Extra-Trees | 0.8788 | 0.931 | Excluded — too correlated |
| LGBM-DART | 0.8953 | 0.973 | Excluded — identical error pattern |
| **CatBoost-Plain** | **0.9016** | 0.952 | **New best — first model to beat LGBM** |

CatBoost fix: `boosting_type='Plain'` disables ordered target encoding, making it compatible with GroupKFold. Pass Driver/Race as pre-computed numeric encodings, not `cat_features`.

**Phase 3 (re-tuning, Notebook 12):**

Multi-fold Optuna (200 trials, folds 0+3 as objective) on the 38-feature set:

| Param | NB05 | NB12 | Effect |
|-------|------|------|--------|
| `num_leaves` | 62 | 31 | More regularized; reduces fold-overfitting |
| `reg_alpha` | 9.79 | 17.28 | Stronger L1; zeroes noisy features |
| `max_bin` | — | 499 | Finer thresholds for compound-weighted features |
| `min_gain_to_split` | — | 0.369 | Additional split regularization |
| `learning_rate` | 0.049 | 0.022 | Slower; uses 219–1498 trees vs 35–462 |

LGBM CV AUC improved 0.8978 → **0.9024** (+0.0046).

---

### Ensemble: Correlation Bounds the Ceiling

**Spearman ρ matrix (NB13, 38-feature set):**

| Pair | ρ |
|------|---|
| LGBM NB12 × CatBoost NB11 | 0.969 |
| LGBM NB12 × Extra-Trees | 0.927 |
| CatBoost NB11 × Extra-Trees | 0.909 |

All architectures ρ ≥ 0.91 — all fail on the same hard laps (early-race and late-race edge cases). Two key findings:

1. **LogisticRegression metalearner hurts at ρ=0.969** (−0.0017 vs LGBM solo). The logit features carry nearly identical information at this correlation level; the metalearner overfits to noise.

2. **Rank average gain (+0.0008 CV) is noise.** LB confirmation: rank avg LB 0.92845 vs LGBM solo LB 0.92840 (+0.00005). CV gains below ±0.001 cannot be trusted.

**Practical rule:** Ensembles require ρ < 0.90 for measurable gain. Breaking this ceiling requires a genuinely different architecture (e.g., MLP with entity embeddings) — not more GBM variants.

---

### Calibration: Platt Scaling for Blending

- **LGBM:** Nearly perfectly calibrated raw (Platt a=0.969, b≈0). Minimal correction.
- **XGBoost:** Base-rate shift; Platt b=0.136 corrects overconfidence on negatives.
- **Key constraint:** Platt input must be `logit(p)`, not raw `p`. Feeding raw probabilities produces a double-sigmoid that worsens Brier score.
- **Isotonic ECE=0 is a training artifact** — 272 knots interpolate OOF data exactly. Use Brier score for held-out calibration quality.

---

## What This Demonstrates

| Skill | Evidence |
|-------|----------|
| **Validation Design** | GroupKFold by Race prevents temporal leakage. Confirmed with empirical leakage experiment. Two actual LB submissions calibrated the CV→LB boost to +0.025. |
| **Feature Engineering** | 38 features in two phases. Position volatility (+0.0391 CV) — the dominant gain — required cross-stint grouping as a deliberate domain decision. Incremental tier evaluation with keep/drop threshold. |
| **Model Debugging** | Diagnosed why DT beats LGBM (fold structure), why CatBoost fails (ordered encoding), why DART adds no diversity (same features → same errors). Understood failure modes, not just results. |
| **Hyperparameter Optimization** | Optuna 200-trial multi-fold objective (folds 0+3). Identified reg_alpha and max_bin as the key parameters on the new feature set. |
| **Ensemble Reasoning** | Quantified why ensembling doesn't help (ρ=0.969). Confirmed noise floor via LB submission. Knew when NOT to blend — rarer skill than knowing when to blend. |
| **Calibration** | Platt scaling with correct logit transform. Understood isotonic ECE=0 as training artifact. Separated calibration from rank-order AUC. |
| **Interpretability** | SHAP confirms domain alignment (Stint #1, TyreLife×laps_rem #2). Used to validate model, not justify it. |
| **Experiment Tracking** | All CV results, LB scores, parameter changes, and failure modes documented with causality (not just outcomes). |

---

## Dataset & Setup

**Data:** 439,140 training rows × 14 columns; 188,165 test rows.  
**Class imbalance:** ~20% pit events.  
**Kaggle:** [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)

To reproduce:
1. Download `train.csv`, `test.csv` from Kaggle into `data/raw/`
2. `pip install -r requirements.txt`
3. Run notebooks `01` through `13` in order — each is fully self-contained (no `src/` imports; uploadable to Kaggle as-is)

---

## Open Questions / Next Steps

**Gap to close: LB 0.026** (0.928 → 0.955). Requires CV AUC ~0.930. CV gains below +0.005 are noise.

1. **MLP with entity embeddings** — only known path to ρ < 0.90 vs GBMs. Architecture: `Driver` (50-dim), `Race` (13-dim), `Compound` (3-dim) embeddings → [256, 128, 64, 1] with BatchNorm + Dropout(0.3). Would be the first model to fail differently from the GBM cluster.

2. **Pseudo-labeling** — high-confidence test rows (p < 0.02 or p > 0.85) added to training to expose the model to unseen race distributions. Expected +0.002–0.006 CV.

3. **Additional feature engineering** — Tier 1 compound-weighted features (corr=0.26–0.30) are zeroed by reg_alpha=17.28. Re-tune with lower L1; add tyre cliff proximity and Race×Compound target encoding.

---

## References

- **Competition:** Kaggle Playground Series S6E5
- **Code:** Fully reproducible Jupyter notebooks in `notebooks/` — self-contained for Kaggle upload
- **Key papers:** Optuna (Akiba et al., 2019) · SHAP (Lundberg & Lee, 2017) · Platt scaling (Platt, 1999)
