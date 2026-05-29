"""
Pretrain the HybridGRUFC model on real F1 data (from FastF1) before fine-tuning
on the competition dataset in NB26.

Prerequisites:
    1. Run scripts/fastf1_pretraining_data.py first to generate data/fastf1/f1_pretrain_features.parquet
    2. Ensure .venv is active with torch installed

Usage:
    cd c:/Repos/predict-f1-pit-stops
    .venv\\Scripts\\Activate.ps1
    python scripts/pretrain_hybrid.py

Outputs (saved to models/):
    - hybrid_pretrained.pt        — pretrained model weights (upload to Kaggle as dataset)
    - pretrain_metrics.pkl        — loss/AUC per epoch

The pretrained weights capture real tyre physics before fine-tuning on synthetic data.
Fine-tuning is done in NB26 on Kaggle.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import pickle, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Path detection ─────────────────────────────────────────────────────────────
cwd = Path(__file__).resolve()
while cwd.name != 'predict-f1-pit-stops' and cwd.parent != cwd:
    cwd = cwd.parent
PROJECT_ROOT  = cwd
DATA_DIR      = PROJECT_ROOT / 'data' / 'fastf1'
MODELS_DIR    = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

print(f'Project root : {PROJECT_ROOT}')
print(f'Data dir     : {DATA_DIR}')

# ── Load preprocessed FastF1 data ─────────────────────────────────────────────
feat_path = DATA_DIR / 'f1_pretrain_features.parquet'
if not feat_path.exists():
    print(f'ERROR: {feat_path} not found. Run fastf1_pretraining_data.py first.')
    raise SystemExit(1)

df = pd.read_parquet(feat_path)
print(f'Loaded: {df.shape}  Pit rate: {df["PitNextLap"].mean():.2%}')

# ── Feature definitions (must match NB21 / NB26) ─────────────────────────────
SEQ_COLS = [
    'LapTime (s)', 'TyreLife', 'Cumulative_Degradation_winsorized',
    'LapTime_Delta', 'Position',
]
STRAT_COLS = [
    'Stint', 'RaceProgress', 'laps_remaining', 'compound_ordinal', 'is_wet_tyre',
    'prime_pit_window', 'laps_to_driver_end', 'abs_position_change',
    'pos_change_rolling_std_3', 'PitStop_lag1',
    'TyreLife_x_laps_remaining', 'Degradation_x_RaceProgress', 'Position_x_RaceProgress',
]
CAT_COLS = ['Driver', 'Race', 'Compound', 'Year']

# Validate that required columns exist
missing = [c for c in SEQ_COLS + STRAT_COLS + CAT_COLS + ['PitNextLap']
           if c not in df.columns]
if missing:
    print(f'WARNING: Missing columns: {missing}. Filling with 0.')
    for c in missing:
        df[c] = 0

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW       = 10
BATCH_SIZE   = 4096
PRETRAIN_EPOCHS = 30
LR           = 5e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.10    # 10% of pretrain data for validation
POS_WEIGHT   = (1 - df['PitNextLap'].mean()) / df['PitNextLap'].mean()
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}, pos_weight: {POS_WEIGHT:.2f}')

# ── Label encoders: fit on real F1 only (test-time uses competition encoders) ─
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    label_encoders[col] = le
    df[col + '_idx'] = le.transform(df[col].astype(str))
    print(f'  {col}: {len(le.classes_)} classes')

EMB_DIMS = {
    'Driver':   (len(label_encoders['Driver'].classes_)   + 1, 32),
    'Race':     (len(label_encoders['Race'].classes_)     + 1,  8),
    'Compound': (len(label_encoders['Compound'].classes_) + 1,  3),
    'Year':     (len(label_encoders['Year'].classes_)     + 1,  2),
}

# ── Build 10-lap windows ──────────────────────────────────────────────────────
df = df.sort_values(['Race', 'Year', 'Driver', 'LapNumber']).reset_index(drop=True)
seq_scaler = StandardScaler()
seq_scaler.fit(df[SEQ_COLS].values)
df[SEQ_COLS] = seq_scaler.transform(df[SEQ_COLS].values)

N         = len(df)
N_SF      = len(SEQ_COLS)
seq_vals  = df[SEQ_COLS].values.astype(np.float32)
windows   = np.zeros((N, WINDOW, N_SF), dtype=np.float32)
masks     = np.zeros((N, WINDOW), dtype=bool)

print(f'Building {WINDOW}-lap windows for {N:,} rows...')
t0 = time.time()
for _, grp_idx in df.groupby(['Race', 'Year', 'Driver'], sort=False).groups.items():
    idxs  = grp_idx.values
    n_grp = len(idxs)
    for i in range(n_grp):
        hl = min(i, WINDOW)
        if hl > 0:
            windows[idxs[i], WINDOW - hl:] = seq_vals[idxs[i - hl : i]]
            masks[idxs[i],   WINDOW - hl:] = True
print(f'Done in {time.time()-t0:.1f}s')

strat_raw  = df[STRAT_COLS].values.astype(np.float32)
strat_scal = StandardScaler()
strat_raw  = strat_scal.fit_transform(strat_raw)

targets    = df['PitNextLap'].values.astype(np.float32)
cat_idxs   = {c: df[c + '_idx'].values for c in CAT_COLS}

# ── Dataset ───────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self):  return len(self.idxs)
    def __getitem__(self, i):
        j = self.idxs[i]
        return {
            'window':   torch.tensor(windows[j],  dtype=torch.float32),
            'mask':     torch.tensor(masks[j],    dtype=torch.bool),
            'strat':    torch.tensor(strat_raw[j], dtype=torch.float32),
            'driver':   torch.tensor(cat_idxs['Driver'][j],   dtype=torch.long),
            'race':     torch.tensor(cat_idxs['Race'][j],     dtype=torch.long),
            'compound': torch.tensor(cat_idxs['Compound'][j], dtype=torch.long),
            'year':     torch.tensor(cat_idxs['Year'][j],     dtype=torch.long),
            'target':   torch.tensor(targets[j],  dtype=torch.float32),
        }

n_val  = int(N * VAL_FRAC)
n_tr   = N - n_val
all_idx = np.arange(N)
np.random.shuffle(all_idx)
tr_idx, va_idx = all_idx[n_val:], all_idx[:n_val]

tr_loader = DataLoader(PretrainDataset(tr_idx), batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
va_loader = DataLoader(PretrainDataset(va_idx), batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

print(f'Train: {len(tr_idx):,} rows, Val: {len(va_idx):,} rows')

# ── Model (identical to NB21 HybridGRUFC) ─────────────────────────────────────
class HybridGRUFC(nn.Module):
    def __init__(self, n_strat=13, n_seq=5, gru_hidden=128, n_layers=2,
                 gru_drop=0.15, emb_dims=None):
        super().__init__()
        if emb_dims is None:
            emb_dims = EMB_DIMS
        self.driver_emb   = nn.Embedding(*emb_dims['Driver'])
        self.race_emb     = nn.Embedding(*emb_dims['Race'])
        self.compound_emb = nn.Embedding(*emb_dims['Compound'])
        self.year_emb     = nn.Embedding(*emb_dims['Year'])
        emb_total = sum(v[1] for v in emb_dims.values())

        self.gru = nn.GRU(n_seq, gru_hidden, n_layers, batch_first=True,
                          dropout=gru_drop if n_layers > 1 else 0.0)
        self.strat_fc = nn.Sequential(
            nn.Linear(n_strat, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU())
        merge_dim = gru_hidden + 64 + emb_total
        self.head = nn.Sequential(
            nn.Linear(merge_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, window, mask, strat, driver, race, compound, year):
        emb = torch.cat([self.driver_emb(driver), self.race_emb(race),
                         self.compound_emb(compound), self.year_emb(year)], dim=1)
        seq_len  = mask.sum(dim=1).long().clamp(min=1)
        packed   = nn.utils.rnn.pack_padded_sequence(
            window, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n   = self.gru(packed)
        gru_feat = h_n[-1]
        strat_f  = self.strat_fc(strat)
        return self.head(torch.cat([gru_feat, strat_f, emb], dim=1)).squeeze(1)


model     = HybridGRUFC().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

# ── Pretraining loop ──────────────────────────────────────────────────────────
metrics = {'train_loss': [], 'val_auc': []}

for epoch in range(PRETRAIN_EPOCHS):
    # Train
    model.train()
    tr_loss = 0.0
    for batch in tr_loader:
        win, mask_b, st = batch['window'].to(DEVICE), batch['mask'].to(DEVICE), batch['strat'].to(DEVICE)
        drv, rc, cmp, yr = (batch[k].to(DEVICE) for k in ['driver', 'race', 'compound', 'year'])
        tgt = batch['target'].to(DEVICE)
        logits = model(win, mask_b, st, drv, rc, cmp, yr)
        loss   = criterion(logits, tgt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * len(win)
    scheduler.step()
    tr_loss /= len(tr_loader.dataset)

    # Validate
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in va_loader:
            win, mask_b, st = batch['window'].to(DEVICE), batch['mask'].to(DEVICE), batch['strat'].to(DEVICE)
            drv, rc, cmp, yr = (batch[k].to(DEVICE) for k in ['driver', 'race', 'compound', 'year'])
            all_logits.append(model(win, mask_b, st, drv, rc, cmp, yr).cpu().numpy())
    val_probs = torch.sigmoid(torch.tensor(np.concatenate(all_logits))).numpy()
    val_auc   = roc_auc_score(targets[va_idx], val_probs)

    metrics['train_loss'].append(tr_loss)
    metrics['val_auc'].append(val_auc)
    print(f'Epoch {epoch+1:3d}/{PRETRAIN_EPOCHS}  loss={tr_loss:.4f}  val_auc={val_auc:.4f}')

# ── Save pretrained weights ───────────────────────────────────────────────────
weights_path = MODELS_DIR / 'hybrid_pretrained.pt'
torch.save(model.state_dict(), weights_path)
print(f'\nSaved pretrained weights: {weights_path}')

# Save scalers and label encoders alongside weights (needed for NB26 fine-tuning)
pretrain_meta = {
    'seq_scaler':      seq_scaler,
    'strat_scaler':    strat_scal,
    'label_encoders':  label_encoders,
    'emb_dims':        EMB_DIMS,
    'seq_cols':        SEQ_COLS,
    'strat_cols':      STRAT_COLS,
    'cat_cols':        CAT_COLS,
    'window':          WINDOW,
    'final_val_auc':   metrics['val_auc'][-1],
    'metrics':         metrics,
}
meta_path = MODELS_DIR / 'hybrid_pretrain_meta.pkl'
with open(meta_path, 'wb') as f:
    pickle.dump(pretrain_meta, f)
print(f'Saved pretrain meta: {meta_path}')
print(f'\nFinal val AUC (real F1 data): {metrics["val_auc"][-1]:.4f}')
print('\nNext: upload hybrid_pretrained.pt + hybrid_pretrain_meta.pkl to Kaggle as a dataset,')
print('then run notebooks/26_pretrained_hybrid.ipynb for fine-tuning.')
