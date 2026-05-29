# Predicting F1 Pit Stops — A Complete ML Engineering Case Study

Binary classification on 439k lap-level rows · AUC-ROC metric · 26 notebooks · 8 leaderboard submissions

**AUC: 0.7717 (logistic regression) → 0.9366 (Hybrid GRU+FC, 3-seed avg)**  
**Kaggle Playground Series S6E5** — [competition link](https://www.kaggle.com/competitions/playground-series-s6e5)

---

## What makes this project worth reading

Most Kaggle notebooks show what worked. This one documents *why* things failed — and uses that diagnosis to decide what to try next. Four model families, three non-obvious architectural experiments, and an empirically calibrated noise floor that prevented wasted submissions.

Three moments of deep ML reasoning:

**1. A Decision Tree beat a tuned LightGBM — and SHAP explained why.**  
Not a fluke. LGBM's gradient descent was stopping at 35 trees on two structurally difficult folds due to early stopping dynamics, not because the model was wrong. Decision Trees adjust depth per-fold naturally. The diagnostic was SHAP analysis + fold-level AUC breakdown. The fix was a different feature set and a 200-trial multi-fold Optuna objective — which increased tree count from 35 to 1,498 on those folds.

**2. An AUCMLoss optimizer produced AUC = 0.27 (inverted predictions) — traced to a saddle-point inversion.**  
AUCMLoss is a min-max optimization: model weights descend, Lagrange multiplier α must *ascend*. Wrapping `AdamW` around both `model.parameters()` and `criterion.parameters()` applied gradient descent to α — flipping the optimization direction entirely. Predictions became anti-correlated with the target. Fix: switch to the PESG optimizer, which handles primal descent + dual ascent correctly, with no learning rate scheduler (PESG has internal epoch-decay; stacking CosineAnnealingLR on top destabilizes the saddle-point dynamics).

**3. Every neural architecture above AUC 0.905 converged to Spearman ρ ≈ 0.91 vs MLP — regardless of design.**  
LSTM: ρ = 0.913. Hybrid GRU+FC with fully separated branches: ρ = 0.919. This isn't a tuning coincidence — it's a structural property of the dataset. Any capable model on F1 telemetry ultimately solves the same two sub-problems (tyre cliff detection + strategic timing window), producing correlated error patterns even with architecturally distinct approaches. The only model to break the ρ gate was a custom TFT Encoder (ρ = 0.549 vs LGBM) — which failed the AUC gate (0.848) due to a different mismatch. This discovery redefined the ensemble strategy entirely.

---

## Skills at a glance

| Area | Demonstrated by |
|------|----------------|
| **Temporal validation design** | GroupKFold by Race+Year; CV→LB boost calibrated per model class from actual submissions |
| **Feature engineering** | 38-feature set; domain-driven cliff normalization; cross-stint grouping as deliberate domain decision; incremental tier evaluation with empirical threshold |
| **Failure diagnosis** | DT > LGBM traced to fold dynamics; CatBoost ordered encoding incompatibility; AUCMLoss dual variable inversion; FastF1 pretraining distribution mismatch read from epoch convergence pattern |
| **Gradient boosting** | LGBM, XGBoost, CatBoost, RF, ET, DART; 200-trial multi-fold Optuna; Spearman ρ for diversity measurement; confirmed noise floor via leaderboard |
| **Custom neural architectures** | MLP + entity embeddings; LSTM; TFT Encoder from scratch (VSN, GRN, static covariate init, multi-head attention, ~400 lines PyTorch); Hybrid GRU+FC dual-branch |
| **Loss function research** | LibAUC compositional AUC maximization; PESG saddle-point optimizer; diagnosed optimizer-criterion interaction causing prediction inversion |
| **Transfer learning** | FastF1 API pipeline; selective weight transfer across embedding cardinality mismatch; output bias correction for domain shift (3% → 20% pit rate) |
| **Ensemble theory** | Neural ρ convergence as discovered structural property; neural-on-neural failure mode identified; noise floor confirmed empirically (CV +0.0008 = LB +0.00005) |
| **Kaggle engineering** | T4×2 DataParallel; subprocess + SQLite Optuna (avoids CUDA fork + pickle failures in Jupyter); self-contained notebooks without local imports |

---

## The progression

| Stage | CV AUC | LB AUC | What changed |
|-------|--------|--------|--------------|
| Logistic Regression | 0.772 | — | Baseline; confirms coefficient signs match domain |
| LightGBM, Optuna-tuned | 0.856 | — | Folds 1–2 stop at 35 trees — a training dynamics problem |
| 38-feature v2 set + re-tuning | 0.902 | 0.928 | Position volatility (+0.039 AUC alone); LGBM now uses 219–1498 trees |
| MLP + entity embeddings | 0.915 | 0.932 | First model below ρ = 0.90 vs GBMs; broke the correlation ceiling |
| Hybrid GRU+FC | 0.919 | 0.936 | Best solo; lowest fold std (±0.007); neural ρ convergence pattern confirmed |
| 3-seed Hybrid rank avg | **0.922** | **0.937** | Final best; +0.00033 LB vs solo |

---

## Phase 1 — Validation and feature engineering

### The leakage that random k-fold misses

The dataset is 439k rows — one per lap. Each race spans 50–70 laps. Features like `TyreLife`, `Cumulative_Degradation`, and `RaceProgress` carry race-specific trajectory context. With random k-fold, laps from the same race appear in both train and validation — the model memorizes race-specific patterns instead of generalizing. CV looks good; out-of-sample performance collapses.

```python
groups = df['Race'].astype(str) + '_' + df['Year'].astype(str)
gkf = GroupKFold(n_splits=5)  # 42 Race+Year groups, 5 folds
```

GroupKFold assigns entire races to a single fold. This simulates the real deployment scenario — predicting pit stops in races the model has never seen.

**Calibrating the CV→LB signal:** Two GBM submissions established a consistent +0.025–0.026 CV→LB boost. MLP ensembles showed +0.017. Hybrid solo: +0.018. These are model-class-specific constants, not universal. Every projection in this project uses the correct class-specific boost rather than a single assumed gap.

### Feature engineering: domain knowledge as inductive bias

Before writing model code, I plotted P(PitNextLap) vs TyreLife per compound. The curves are S-shaped with compound-specific inflection points — those become the normalization anchors, not the median stint lengths (which are strategically cut short and therefore lower):

| Compound | Cliff at TyreLife | Pit rate | Counterintuitive note |
|----------|------------------|----------|----------------------|
| SOFT | 13 | 19.3% | Short stints, but low pit rate — changed frequently by design |
| MEDIUM | 49 | 10.1% | Lowest pit rate; used when teams want one-stop |
| **HARD** | **61** | **32.7%** | **Highest pit rate despite most durable** — teams run to the cliff |
| INTERMEDIATE / WET | — | 15.2% / 2.5% | Not on the dry durability axis; separate binary flag |

The HARD compound result is counterintuitive: the most durable tyre has the highest pit rate. The reason is strategic — teams choose HARD specifically to run long stints, which means they're always running close to the cliff threshold. The pit rate measures strategic choice, not failure.

**The strongest feature by SHAP (mean |SHAP| = 0.86, nearly 2× the next):** `Stint` — the count of pit stops already made. It acts as a structural prior for whether another stop is "due." Not derived, not engineered — it's a raw column that captures cumulative race strategy better than any interaction term.

All lag and rolling features are grouped within `(Race, Year, Driver, Stint)`. Position volatility is the one deliberate exception — grouped only by `(Race, Year, Driver)` to span stint boundaries, because a driver losing positions erratically near stint-end is on degraded tyres regardless of which stint they're in. This distinction produced +0.039 CV AUC — the largest single gain in the feature engineering phase.

Target encodings (Race pit rate, Driver pit rate) are computed inside each CV fold using only that fold's training rows. Computing on the full dataset before splitting inflates validation AUC by 0.01–0.02.

---

## Phase 2 — Scaling GBMs and learning the ensemble ceiling

### Model diversity analysis

Five architectures tested on the 38-feature set, with Spearman ρ measured between OOF predictions:

| Model | CV AUC | ρ vs LGBM | Decision |
|-------|--------|-----------|---------|
| LGBM re-tuned (200 Optuna trials) | 0.902 | — | Anchor |
| CatBoost-Plain | 0.902 | 0.952 | Include — symmetric-tree structure differs from leaf-wise |
| LGBM-DART | 0.895 | **0.973** | Exclude — dropout changes path, not destination |
| Extra-Trees | 0.879 | 0.931 | Exclude — too weak; dilutes rank ordering of stronger models |
| Random Forest | 0.878 | 0.946 | Exclude — same reason |

> **Key insight:** A weak model in rank average doesn't just fail to help — it corrupts the rank ordering of the stronger models' correct predictions. The inclusion threshold isn't just AUC; it's AUC relative to the ensemble average.

**CatBoost needed a fix:** `boosting_type='Plain'` disables ordered target encoding, which is incompatible with GroupKFold-by-Race. With ordered encoding, the model sees 2–14 trees across 4 of 5 folds because held-out races have no encoding support. Plain mode treats it as standard GBDT.

### Confirming the noise floor with real submissions

LGBM+CatBoost rank average: CV +0.0008 vs LGBM solo. LB result: +0.00005. The leaderboard confirmed what the math implied — Spearman ρ = 0.969 means these models fail on exactly the same laps. No blending strategy recovers from that level of correlation.

> **Rule derived from data:** Ensemble benefit requires ρ < 0.90. Breaking that ceiling requires a genuinely different learning paradigm — not more GBM variants.

---

## Phase 3 — Neural architectures

### MLP with entity embeddings: breaking the correlation ceiling

Target encodings compress 887 driver identities to a single number per fold. Entity embeddings preserve a 32-dimensional learned representation — the network can find interaction patterns between driver identity and other features that scalars flatten away.

**Architecture:** Driver (887 → 32-dim), Race (26 → 8), Compound (5 → 3), Year (4 → 2) embeddings concatenated with 38 scalar features → [256, 128, 64, 1] with BatchNorm and Dropout. ~92k parameters.

**Result:** CV AUC 0.910. Spearman ρ = **0.791 vs LGBM** — first model in the pipeline to break the 0.90 correlation ceiling. LB 0.932 (+0.017 boost). On the hardest fold (where every GBM scored 0.882), the MLP scored 0.907 — a +0.025 gain attributed to race-identity embeddings learning fold-specific pit patterns directly.

### Custom TFT Encoder (~400 lines PyTorch)

F1 lap data maps exactly to the Temporal Fusion Transformer's input taxonomy:
- **Static covariates:** Driver/Race identity
- **Time-varying known future:** RaceProgress, LapNumber
- **Time-varying observed past:** LapTime, TyreLife, degradation

Implemented encoder-only (decoder and quantile heads are unnecessary for binary classification): Variable Selection Networks → per-feature Gated Residual Networks → static covariate init for LSTM cell/hidden → multi-head temporal self-attention.

**Result:** AUC 0.848 — below the inclusion gate. But Spearman ρ vs LGBM = **0.549** — the most architecturally diverse model in the pipeline by a wide margin. A second training run with corrected hyperparameters (lower LR, longer patience, warmup schedule, gradient clipping) scored 0.839 — *worse*. Fold 3 peaked at epoch 7 and degraded through the full extended patience window. The conclusion: VSN gating on static feature frames is an architectural mismatch with this perturbed synthetic dataset — not a training stability problem. Extraordinary diversity cannot compensate for weak solo AUC in rank averaging.

### Hybrid GRU+FC: dual-branch inductive bias

**Design hypothesis:** An MLP mixes tyre dynamics and race strategy in the same weight space. Explicitly separating them should change which laps the model fails on — reducing ρ vs MLP.

```
GRU branch: [LapTime, TyreLife, Degradation, LapTime_Delta, Position] × 10-lap window
            → GRU(128 units, 2 layers) → last hidden state

FC branch:  [Stint, RaceProgress, laps_remaining, compound_ordinal, ...] × 13 strategic scalars
            → Linear → BatchNorm → ReLU × 2 layers (64 units)

Merge: concat(GRU out, FC out, Driver/Race/Compound/Year embeddings) → head(237→128→64→1)
```

**Result:** CV AUC 0.919 (best solo). LB 0.936. Fold std ±0.007 — the most stable model in the pipeline. But Spearman ρ vs MLP = **0.919** — the explicit branch separation made no difference to correlation.

> **The discovered pattern:** Every architecture that achieves AUC ≥ 0.905 on F1 telemetry clusters at ρ ≈ 0.91–0.92 vs MLP, regardless of design. The models solve the same two underlying problems — tyre cliff detection and strategic timing — from the same data. The ensemble gate was revised accordingly: the relevant diversity measure is ρ vs the GBM anchor (~0.80 for all neural models), not ρ vs MLP.

---

## Phase 4 — Exhausting every remaining lever

With the best LB at 0.936 and the top at 0.955, the remaining gap required approaches that change what the model fundamentally learns — not more tuning.

### What was tried and what each result taught

**Multi-seed averaging (3 seeds):** CV +0.004. LB +0.0003. Seeds 0/1/2 have ρ = 0.938–0.949 — same architecture, slightly different random error. This was the only lever with a positive LB return, even if marginal.

**Optuna architecture search (50 trials, T4×2 GPUs):** Best window size: 20 (vs 10). Attention mechanism rejected by 9 of 10 top trials — pooling the final GRU hidden state outperforms attention-weighted pooling, because tyre trajectory information is already concentrated in the final state; attention redistributes it toward less-informative early laps. Net LB: worse than current best.

**AUCMLoss + PESG optimizer:** BCE optimizes log-likelihood, not AUC. LibAUC's compositional AUC maximization directly optimizes the AUC surrogate using a min-max formulation.

First run: AUC = 0.2715 — model predicting the inverse of the target.

Root cause: AUCMLoss introduces a Lagrange multiplier α that must be *maximized* (dual ascent). `AdamW` applied to `model.parameters() + criterion.parameters()` descends on α — inverting the dual update. Fix:

```python
# Wrong — AdamW descends on α (the dual variable), inverting the saddle point
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(criterion.parameters()), lr=1e-3
)

# Correct — PESG handles primal descent + dual ascent
optimizer = PESG(
    model.parameters(), loss_fn=criterion,
    lr=0.1, epoch_decay=2e-3, momentum=0.9
)
scheduler = None  # PESG has internal epoch-decay; LR schedulers destabilize it
```

Corrected result: AUC 0.917 — δ = −0.002 vs BCE. AUCM provides no benefit for this architecture on this data.

**FastF1 pretraining:** Downloaded 98,774 real F1 laps (2022–2025) via the FastF1 API. Engineered the same 38-feature schema. Pretrained the Hybrid GRU+FC for 30 epochs (real F1 val AUC: 0.836). Selectively transferred GRU, FC branch, and head weights — embedding tables re-initialized due to cardinality mismatch (31 real drivers vs 887 in competition). Reset output bias from real F1 log-odds (−3.4, pit rate 3%) to competition log-odds (−1.4, pit rate 20%).

Fine-tuned result: AUC 0.907 — Δ = −0.011 vs Hybrid from scratch.

> **The diagnostic:** All 5 folds ran to near-maximum epochs (32–50) without early stopping. A model with a good initialization converges quickly; a model fighting a bad prior grinds slowly toward a mediocre minimum. Real F1 tyre degradation curves are qualitatively different from the synthetically perturbed competition distributions — 50 fine-tuning epochs at 1e-4 LR cannot overcome 30 epochs of real-data pretraining when the distributions diverge at this magnitude.

---

## Architecture reference

```
data/raw/           ← immutable — train.csv, test.csv
data/processed/     ← parquet outputs: features, fold assignments, OOF predictions
notebooks/          ← 26 numbered notebooks, each self-contained for Kaggle upload
models/             ← per-fold .pt weights (local only)
submissions/        ← versioned CSVs + leaderboard_log.md
scripts/            ← FastF1 data pipeline, pretraining script
```

Each notebook defines all helper functions inline — no `src/` imports, no relative path assumptions. A path-detection block walks up from cwd to find the project root, making notebooks runnable identically in VSCode and Kaggle.

---

## Setup

```powershell
git clone <repo> && cd predict-f1-pit-stops
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e5) into `data/raw/`. Run notebooks 01–26 in order. Notebooks 18, 21, 23, 25, and 26 require a GPU — designed for Kaggle T4×2 (30 min/fold on T4 vs ~4 hr on CPU).

---
