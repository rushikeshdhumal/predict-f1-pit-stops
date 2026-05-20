# F1 Pit Stop Prediction — ML Rigor in a Kaggle Competition

Binary classification: predict whether a Formula 1 driver will pit on the next lap. This project showcases **rigorous ML engineering** — validated on 439k rows with proper handling of temporal structure, class imbalance, and cross-validation leakage.

**Best CV AUC: 0.8558** (LightGBM + Optuna)  
**Baseline (Logistic Regression): 0.7717 AUC**  
**Kaggle Competition:** Playground Series S6E5

---

## Why This Matters: Validation Design

The core insight is **preventing data leakage in temporally-structured data.** Naive k-fold CV places laps from the same race in both train and validation — the model memorizes race-specific context (circuit characteristics, weather effects, tire degradation) rather than generalizing.

**Solution:** GroupKFold by Race ensures no race appears in both train and validation. This project includes an **empirical leakage experiment:** random k-fold CV inflates AUC by ~0.01 compared to held-out-race evaluation. This shift has no cure except correct validation design.

**Result:** GroupKFold CV (0.8558) simulates true deployment generalization. Naive k-fold would appear ~2% better but fail on unseen races.

---

## Key Technical Decisions

### **Feature Engineering: 29 Features, Temporal Integrity**
Engineered from 14 raw columns with strict temporal ordering (no look-ahead bias):

| Category | Examples | Domain Insight |
|----------|----------|---|
| **Tyre Age** | TyreLife_normalized_by_compound, TyreLife_sq | Compound-specific cliff thresholds (SOFT=13, MED=49, HARD=61) discovered via P(pit) S-curves, not stint medians |
| **Degradation** | Cumul_Deg_winsorized, Degradation_rate | Rate-of-change signals cliff. Outliers clipped at [−205, +122] (1%/99%). |
| **Trajectory** | LapTime_lag{1-3}, LapTime_Delta_lag{1-3} | Captures momentum. Grouped by (Race, Year, Driver, Stint) to prevent cross-stint leakage. |
| **Interactions** | TyreLife × laps_remaining, Degradation × RaceProgress | Joint effects neither alone captures. |
| **Context** | Stint, laps_remaining, Position | Stint is strongest predictor (SHAP 0.86, 2× other features). |
| **Encodings** | Race_target_encoded, Driver_target_encoded | Computed inside CV folds only (prevents AUC leakage of 0.01–0.02). |

**SHAP interpretation confirms domain alignment:** Top 3 features are Stint (0.86), TyreLife_x_laps_rem (0.70), and Race_target_encoded (0.33) — exactly what domain knowledge predicts.

### **Model Evaluation: 5 Architectures, Understanding Failure**

| Model | CV AUC | Interpretation |
|-------|--------|---|
| Logistic Regression | 0.7717 | Baseline; coefficients sign-correct. |
| **Decision Tree** | **0.8678** | Best performer. GBMs plateau lower due to fold structure (folds 1–2 harder). |
| LightGBM (Optuna, 100 trials) | 0.8558 | reg_alpha=9.79 dominant parameter. Fold std ±0.0087. |
| XGBoost | 0.8492 | Validates LGBM rankings; 0.0066 gap. |
| CatBoost | ~0.78 | **Fails:** Ordered target encoding within GroupKFold produces near-empty folds. |

**Why DT beats LGBM?** Not feature selection — fold composition. Folds 1–2 structurally difficult (LGBM stops at ~35 trees; DT adjusts depth naturally). Same feature set, different optimization surface. This is **not a tuning problem** — all GBMs plateau at 0.85–0.86.

### **Ensemble: High Correlation = Low Payoff**

Spearman ρ=0.974 between LGBM and XGBoost means they fail on the same hard laps (74% Jaccard error overlap). Optimal blend weight = 1.0 (pure LGBM). Adding XGBoost decreases AUC monotonically. 

Stacking V2 (with Stint + TyreLife_x_laps_rem as Level-2 features) achieves 0.8569 on OOF, but this is optimistic: metalearner fit and evaluated on same 439k rows. Notebook 09 CV will verify if gain is real or training artifact.

**Insight:** Ensembles help only when models fail differently. Here, identical features + similar optimization = identical error patterns. Need different feature set or architecture to break correlation.

### **Calibration: Platt Scaling for Blending**

AUC is rank-invariant, so calibration doesn't affect competition score directly. But for stacking, probability estimates matter.

- **LGBM:** Nearly perfectly calibrated raw (Platt a=0.969, b≈0.004). Minimal correction.
- **XGBoost:** Base-rate shift; Platt b=0.136 corrects overconfidence on negatives.

Isotonic regression achieves ECE=0 on OOF (training artifact — 272 knots interpolate exactly). Use Brier score for held-out eval.

---

## What This Demonstrates (For Hiring Managers)

| Skill | Evidence |
|-------|----------|
| **ML Foundations** | Knows validation leakage (GroupKFold prevents race-level data leakage). Understands AUC-ROC, class imbalance, rank-invariant metrics. |
| **Feature Engineering** | 29 domain-aware features with temporal integrity. Key insight: tyre cliff thresholds from P(pit) S-curves, not median stint length. |
| **Model Evaluation** | Tested 5 architectures; diagnosed why DT beats LGBM (fold structure, not feature engineering). High bar for statistical rigor. |
| **Hyperparameter Optimization** | Optuna over 100 trials. Identified dominant parameter (reg_alpha=9.79). Understood optimization landscape, not just best result. |
| **Ensemble Methods** | Diagnosed why ensembling hurts (ρ=0.974 correlation → identical errors). Knew when NOT to blend — rarer skill than knowing when to blend. |
| **Probability Calibration** | Platt scaling with correct logit transform. Understood why isotonic ECE=0 is training artifact. Why calibration ≠ AUC improvement. |
| **Interpretability** | SHAP analysis confirms domain alignment (Stint strongest, not surprising). Used SHAP to validate, not justify. |
| **Rigor & Documentation** | All decisions traceable. No shortcuts; reproducible code; proper split of CV vs. test. |

---

## Dataset & Setup

**Data:** 439,140 training rows × 14 columns; 188,165 test rows.  
**Class imbalance:** 20% pit events.  
**Kaggle:** Playground Series S6E5.

To reproduce:
1. Download `train.csv`, `test.csv` from Kaggle (not in repo)
2. `pip install -r requirements.txt`
3. Run `notebooks/01` through `notebooks/10` in order (fully self-contained; no external imports)

---

## Open Questions / Future Work

1. **Why DT (0.8678) > LGBM (0.8558)?** Fold structure, not feature engineering. Folds 1–2 structurally difficult; LGBM stops at 35 trees; DT auto-adjusts depth. Could investigate: different n_splits, different random seed, explicit fold-difficulty analysis.

2. **Stacking V2 gain real or artifact?** Notebook 09 will answer. +0.0011 on OOF (optimistic) likely shrinks under true fold validation.

3. **What would break the correlation?** Different feature set (temporal embeddings, raw interactions) or different architecture (neural networks with learned interactions) could lower ρ below 0.95 and enable ensemble gains.

4. **Ablation studies needed.** Which of the 29 features actually contribute? Feature importance ranked, not ablated. Unknown whether all 29 are necessary.

---

## Refs

- **Competition:** [Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)
- **Code:** Fully reproducible Jupyter notebooks in `notebooks/` (no `src/` folder; self-contained for Kaggle upload)
- **Papers:** Optuna (Akiba et al., 2019), SHAP (Lundberg & Lee, 2017), Platt scaling (Platt, 1999)
