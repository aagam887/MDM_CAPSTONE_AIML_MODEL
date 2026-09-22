"""
LSTM / GRU Deep Learning Baseline — Option C (Detection vs Prediction)
Author: Ayush Bankar (Capstone Project)

Runs TWO labeling strategies side-by-side:
  1. DETECTION  — naive labeling (septic period included in input)  -> overestimates
  2. PREDICTION — proper 6h lookahead (strictly past input)         -> realistic

Both use identical:
  - 3-fold GroupKFold by Patient_ID
  - LSTM and GRU architectures
  - Class-weighted loss, early stopping

Usage: python -X utf8 -u baseline_lstm_gru_option_c.py
"""

# ---------- UTF-8 I/O FIX (must be first) ----------
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------- IMPORTS ----------
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

import torch
torch.set_num_threads(4)   # prevents Windows CPU thread deadlock
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, accuracy_score
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'csv_path': 'Dataset.csv',
    'seq_len': 48,
    'lookahead': 6,
    'n_splits': 3,
    'batch_size': 128,
    'epochs': 15,
    'patience': 3,
    'lr': 1e-3,
    'hidden_size': 32,
    'num_layers': 1,
    'dropout': 0.3,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")
print(f"Torch threads: {torch.get_num_threads()}")
print(f"Config: {CONFIG}")
print("", flush=True)


# ============================================================
# 1. LOAD DATA
# ============================================================
print("[1] Loading data...", flush=True)
df = pd.read_csv(CONFIG['csv_path'], low_memory=False)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

before = len(df)
df = df.dropna(subset=['Patient_ID', 'SepsisLabel']).reset_index(drop=True)
df['Patient_ID'] = df['Patient_ID'].astype(int)
df['SepsisLabel'] = df['SepsisLabel'].astype(int)
print(f"  Dropped {before - len(df)} corrupt row(s)")
print(f"  Rows: {len(df):,} | Patients: {df['Patient_ID'].nunique():,}", flush=True)

ignore_cols = ['SepsisLabel', 'Patient_ID', 'Unit1', 'Unit2']
feature_cols = [c for c in df.columns if c not in ignore_cols]
print(f"  Features: {len(feature_cols)}", flush=True)


# ============================================================
# 2A. BUILD SEQUENCES — DETECTION (naive, last window)
# ============================================================
def build_sequences_detection(df, feature_cols, seq_len, lookahead):
    """
    DETECTION strategy (includes septic period in input).
    """
    X_list, y_list, pid_list = [], [], []

    for pid, g in df.groupby('Patient_ID', sort=False):
        g = g.sort_values('ICULOS')
        feats = g[feature_cols].values.astype(np.float32)
        labels = g['SepsisLabel'].values.astype(np.int8)
        T = len(g)

        if T < 2:
            continue

        if T >= seq_len:
            feats_win = feats[-seq_len:]
            future = labels[-lookahead:] if T >= lookahead else labels
            label = int(future.max())
        else:
            pad_len = seq_len - T
            feats_win = np.vstack([np.zeros((pad_len, feats.shape[1]), dtype=np.float32), feats])
            future = labels[-min(lookahead, T):]
            label = int(future.max())

        X_list.append(feats_win)
        y_list.append(label)
        pid_list.append(pid)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    pids = np.array(pid_list, dtype=np.int64)
    return X, y, pids


# ============================================================
# 2B. BUILD SEQUENCES — PREDICTION (proper sliding window)
# ============================================================
def build_sequences_prediction(df, feature_cols, seq_len, lookahead):
    """
    PREDICTION strategy (proper early warning, no leakage).
    """
    X_list, y_list, pid_list = [], [], []

    for pid, g in df.groupby('Patient_ID', sort=False):
        g = g.sort_values('ICULOS')
        feats = g[feature_cols].values.astype(np.float32)
        labels = g['SepsisLabel'].values.astype(np.int8)
        T = len(g)

        if T < seq_len + lookahead:
            continue

        for t in range(seq_len, T - lookahead + 1):
            window = feats[t - seq_len : t]
            past_labels = labels[t - seq_len : t]
            future = labels[t : t + lookahead]

            if past_labels.max() == 1:
                continue

            X_list.append(window)
            y_list.append(int(future.max()))
            pid_list.append(pid)

    if not X_list:
        raise ValueError("No prediction sequences built. Check seq_len vs data.")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    pids = np.array(pid_list, dtype=np.int64)
    return X, y, pids


# ============================================================
# 3. FILL NaNs
# ============================================================
def fill_nans(X):
    X_flat = X.reshape(-1, X.shape[-1])
    medians = np.nanmedian(X_flat, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    nan_mask = np.isnan(X)
    return np.where(nan_mask, medians[None, None, :], X).astype(np.float32)


# ============================================================
# 4. DATASET + MODEL
# ============================================================
class SepsisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SepsisRNN(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1,
                 dropout=0.3, rnn_type='lstm'):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


# ============================================================
# 5. TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


# ============================================================
# 6. GROUPKFOLD CV
# ============================================================
def run_cv(rnn_type, X, y, pids, config, tag=""):
    print(f"\n{'='*60}")
    print(f"  {tag} | {rnn_type.upper()} | {config['n_splits']}-Fold GroupKFold")
    print(f"{'='*60}", flush=True)

    gkf = GroupKFold(n_splits=config['n_splits'])
    fold_results = []
    oof_true, oof_prob = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, pids), 1):
        print(f"\n  Fold {fold}/{config['n_splits']}", flush=True)
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        print(f"    Train: {len(y_tr):,} | Pos: {y_tr.sum():,} ({y_tr.mean():.2%})")
        print(f"    Val:   {len(y_va):,} | Pos: {y_va.sum():,} ({y_va.mean():.2%})", flush=True)

        n_feat = X_tr.shape[-1]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape).astype(np.float32)
        X_va = scaler.transform(X_va.reshape(-1, n_feat)).reshape(X_va.shape).astype(np.float32)

        pos_weight = torch.tensor(
            [(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)],
            dtype=torch.float32
        ).to(config['device'])

        train_loader = DataLoader(SepsisDataset(X_tr, y_tr),
                                  batch_size=config['batch_size'],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(SepsisDataset(X_va, y_va),
                                batch_size=config['batch_size'],
                                shuffle=False, num_workers=0)

        model = SepsisRNN(n_feat, config['hidden_size'], config['num_layers'],
                          config['dropout'], rnn_type).to(config['device'])
        optimizer = Adam(model.parameters(), lr=config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auroc, best_state, patience_ctr = 0.0, None, 0
        t0 = time.time()

        for epoch in range(1, config['epochs'] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config['device'])
            y_t, y_p = evaluate(model, val_loader, config['device'])
            try:
                auroc = roc_auc_score(y_t, y_p)
                auprc = average_precision_score(y_t, y_p)
            except ValueError:
                auroc, auprc = 0.0, 0.0
            scheduler.step(auroc)

            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:2d} | loss={train_loss:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}", flush=True)

            if patience_ctr >= config['patience']:
                print(f"    Early stop at epoch {epoch}", flush=True)
                break

        elapsed = time.time() - t0
        model.load_state_dict(best_state)
        y_t, y_p = evaluate(model, val_loader, config['device'])
        y_pred = (y_p >= 0.5).astype(int)

        metrics = {
            'fold': fold,
            'AUROC': roc_auc_score(y_t, y_p),
            'AUPRC': average_precision_score(y_t, y_p),
            'Recall': recall_score(y_t, y_pred, zero_division=0),
            'Precision': precision_score(y_t, y_pred, zero_division=0),
            'F1-Score': f1_score(y_t, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_t, y_pred),
            'time_sec': elapsed,
        }
        fold_results.append(metrics)
        oof_true.extend(y_t.tolist())
        oof_prob.extend(y_p.tolist())

        print(f"    >> Fold {fold}: AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | "
              f"Recall={metrics['Recall']:.4f} ({elapsed:.1f}s)", flush=True)

    df_res = pd.DataFrame(fold_results)
    mean_r = df_res.drop(columns=['fold', 'time_sec']).mean().to_dict()
    std_r = df_res.drop(columns=['fold', 'time_sec']).std().to_dict()
    print(f"\n  == {tag} {rnn_type.upper()} MEAN AUROC: {mean_r['AUROC']:.4f} +/- {std_r['AUROC']:.4f}")
    print(f"  == {tag} {rnn_type.upper()} MEAN AUPRC: {mean_r['AUPRC']:.4f} +/- {std_r['AUPRC']:.4f}")
    print(f"  == {tag} {rnn_type.upper()} MEAN Recall: {mean_r['Recall']:.4f} +/- {std_r['Recall']:.4f}", flush=True)

    return {
        'metrics_df': df_res,
        'mean': mean_r,
        'std': std_r,
        'oof_true': np.array(oof_true),
        'oof_proba': np.array(oof_prob),
    }


# ============================================================
# 7. RUN — DETECTION
# ============================================================
print("\n\n" + "#"*60)
print("# PHASE 1: DETECTION (naive labeling - includes leakage)")
print("#"*60, flush=True)

X_det, y_det, pids_det = build_sequences_detection(df, feature_cols, CONFIG['seq_len'], CONFIG['lookahead'])
X_det = fill_nans(X_det)
print(f"  Detection sequences: {X_det.shape} | Positives: {y_det.sum():,} ({y_det.mean():.2%})", flush=True)

det_lstm = run_cv('lstm', X_det, y_det, pids_det, CONFIG, tag="DETECTION")
det_gru  = run_cv('gru',  X_det, y_det, pids_det, CONFIG, tag="DETECTION")


# ============================================================
# 8. RUN — PREDICTION
# ============================================================
print("\n\n" + "#"*60)
print("# PHASE 2: PREDICTION (proper 6h lookahead - no leakage)")
print("#"*60, flush=True)

X_pred, y_pred_seq, pids_pred = build_sequences_prediction(df, feature_cols, CONFIG['seq_len'], CONFIG['lookahead'])
X_pred = fill_nans(X_pred)
print(f"  Prediction sequences: {X_pred.shape} | Positives: {y_pred_seq.sum():,} ({y_pred_seq.mean():.2%})", flush=True)

pred_lstm = run_cv('lstm', X_pred, y_pred_seq, pids_pred, CONFIG, tag="PREDICTION")
pred_gru  = run_cv('gru',  X_pred, y_pred_seq, pids_pred, CONFIG, tag="PREDICTION")


# ============================================================
# 9. SAVE EVERYTHING
# ============================================================
print("\n\n[9] Saving results...", flush=True)

det_lstm['metrics_df'].to_csv('detection_lstm_per_fold.csv', index=False)
det_gru['metrics_df'].to_csv('detection_gru_per_fold.csv', index=False)
pred_lstm['metrics_df'].to_csv('prediction_lstm_per_fold.csv', index=False)
pred_gru['metrics_df'].to_csv('prediction_gru_per_fold.csv', index=False)

np.save('detection_lstm_oof_true.npy', det_lstm['oof_true'])
np.save('detection_lstm_oof_prob.npy', det_lstm['oof_proba'])
np.save('detection_gru_oof_true.npy',  det_gru['oof_true'])
np.save('detection_gru_oof_prob.npy',  det_gru['oof_proba'])
np.save('prediction_lstm_oof_true.npy', pred_lstm['oof_true'])
np.save('prediction_lstm_oof_prob.npy', pred_lstm['oof_proba'])
np.save('prediction_gru_oof_true.npy',  pred_gru['oof_true'])
np.save('prediction_gru_oof_prob.npy',  pred_gru['oof_proba'])

summary = {
    'detection': {
        'lstm': {**det_lstm['mean'], **{f'{k}_std': v for k, v in det_lstm['std'].items()}},
        'gru':  {**det_gru['mean'],  **{f'{k}_std': v for k, v in det_gru['std'].items()}},
    },
    'prediction': {
        'lstm': {**pred_lstm['mean'], **{f'{k}_std': v for k, v in pred_lstm['std'].items()}},
        'gru':  {**pred_gru['mean'],  **{f'{k}_std': v for k, v in pred_gru['std'].items()}},
    }
}
with open('dl_metrics_option_c.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("  [OK] 4 per-fold CSVs")
print("  [OK] 8 OOF prediction .npy files")
print("  [OK] dl_metrics_option_c.json", flush=True)


# ============================================================
# 10. FINAL COMPARISON TABLE
# ============================================================
print("\n\n" + "="*70)
print("  FINAL COMPARISON: DETECTION vs PREDICTION")
print("="*70)
print(f"{'Strategy':<14} {'Model':<6} {'AUROC':<22} {'AUPRC':<22} {'Recall':<10}")
print("-"*70)
for strategy, res_dict in [('DETECTION', {'LSTM': det_lstm, 'GRU': det_gru}),
                           ('PREDICTION', {'LSTM': pred_lstm, 'GRU': pred_gru})]:
    for model_name, res in res_dict.items():
        m, s = res['mean'], res['std']
        print(f"{strategy:<14} {model_name:<6} "
              f"{m['AUROC']:.4f} +/- {s['AUROC']:.4f}    "
              f"{m['AUPRC']:.4f} +/- {s['AUPRC']:.4f}    "
              f"{m['Recall']:.4f}")

print("\n" + "="*70)
print("  KEY INSIGHT")
print("="*70)
drop_auroc = det_gru['mean']['AUROC'] - pred_gru['mean']['AUROC']
print(f"  AUROC drop (Detection -> Prediction): {drop_auroc:+.4f}")
print(f"  This gap quantifies the target-leakage effect of naive labeling.")
print("\n[OK] Option C complete.", flush=True)