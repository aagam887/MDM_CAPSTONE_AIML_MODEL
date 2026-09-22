"""
LSTM / GRU Deep Learning Baseline for Early Sepsis Prediction
Author: Ayush Bankar (Capstone Project)

- Per-patient sequence modeling (48-hour windows)
- GroupKFold cross-validation (grouped by Patient_ID)
- 6-hour lookahead label
- Compares LSTM vs GRU against classical baselines

Usage: python baseline_lstm_gru.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

import torch
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
    'seq_len': 48,                 # 48-hour windows
    'lookahead': 6,                # 6h prediction horizon
    'n_splits': 3,                 # GroupKFold folds
    'batch_size': 64,
    'epochs': 30,
    'patience': 5,                 # early stopping
    'lr': 1e-3,
    'hidden_size': 64,
    'num_layers': 1,
    'dropout': 0.3,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")
print(f"Config: {CONFIG}")


# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("\n[1] Loading data...")
df = pd.read_csv(CONFIG['csv_path'], low_memory=False)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Clean corrupt rows
before = len(df)
df = df.dropna(subset=['Patient_ID', 'SepsisLabel']).reset_index(drop=True)
df['Patient_ID'] = df['Patient_ID'].astype(int)
df['SepsisLabel'] = df['SepsisLabel'].astype(int)
print(f"  Dropped {before - len(df)} corrupt row(s)")
print(f"  Rows: {len(df):,} | Patients: {df['Patient_ID'].nunique():,}")

# Feature columns
ignore_cols = ['SepsisLabel', 'Patient_ID', 'Unit1', 'Unit2']
feature_cols = [c for c in df.columns if c not in ignore_cols]
print(f"  Features: {len(feature_cols)}")


# ============================================================
# 2. BUILD PER-PATIENT SEQUENCES WITH 6H LOOKAHEAD LABELS
# ============================================================
print("\n[2] Building sequences...")

def build_sequences(df, feature_cols, seq_len, lookahead):
    """
    For each patient, create sliding windows of length seq_len.
    Label for each window = 1 if sepsis occurs within next `lookahead` hours.
    """
    X_list, y_list, pid_list = [], [], []

    for pid, g in df.groupby('Patient_ID', sort=False):
        g = g.sort_values('ICULOS')  # ensure time order
        feats = g[feature_cols].values.astype(np.float32)
        labels = g['SepsisLabel'].values.astype(np.int8)
        T = len(g)

        if T < 2:
            continue

        # Pad/truncate to seq_len
        if T >= seq_len:
            # Use the LAST seq_len hours (most recent context)
            feats_win = feats[-seq_len:]
            labels_win = labels[-seq_len:]
            # Label = sepsis within next `lookahead` hours from end of window
            future = labels[-(lookahead):] if T >= lookahead else labels
            label = int(future.max())
        else:
            pad_len = seq_len - T
            feats_win = np.vstack([np.zeros((pad_len, feats.shape[1]), dtype=np.float32), feats])
            labels_win = np.concatenate([np.zeros(pad_len, dtype=np.int8), labels])
            future = labels[-min(lookahead, T):]
            label = int(future.max())

        X_list.append(feats_win)
        y_list.append(label)
        pid_list.append(pid)

    X = np.stack(X_list)                 # (N, seq_len, F)
    y = np.array(y_list, dtype=np.int64)
    pids = np.array(pid_list, dtype=np.int64)
    return X, y, pids


X, y, pids = build_sequences(df, feature_cols, CONFIG['seq_len'], CONFIG['lookahead'])
print(f"  Sequences: {X.shape}")
print(f"  Positives: {y.sum():,} ({y.mean():.2%})")


# ============================================================
# 3. HANDLE MISSING VALUES (per-sequence median fill)
# ============================================================
print("\n[3] Handling NaNs...")

def fill_nans_per_patient(X):
    """Fill NaNs with the feature median computed across ALL sequences."""
    X_flat = X.reshape(-1, X.shape[-1])
    medians = np.nanmedian(X_flat, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    nan_mask = np.isnan(X)
    X_filled = np.where(nan_mask, medians[None, None, :], X)
    return X_filled.astype(np.float32)

X = fill_nans_per_patient(X)
print(f"  Remaining NaNs: {np.isnan(X).sum()}")


# ============================================================
# 4. PYTORCH DATASET + MODELS
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
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)
        # Take last timestep
        last = out[:, -1, :]
        return self.fc(self.dropout(last)).squeeze(-1)


# ============================================================
# 5. TRAINING / EVAL LOOPS
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
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(yb.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ============================================================
# 6. GROUPKFOLD TRAINING
# ============================================================
def run_cv(rnn_type, X, y, pids, config):
    print(f"\n{'='*60}")
    print(f"  {rnn_type.upper()} | {config['n_splits']}-Fold GroupKFold")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=config['n_splits'])
    fold_results = []
    oof_true, oof_prob = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, pids), 1):
        print(f"\n  Fold {fold}/{config['n_splits']}")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        print(f"    Train: {len(y_tr):,} seqs | Pos: {y_tr.sum():,} ({y_tr.mean():.2%})")
        print(f"    Val:   {len(y_va):,} seqs | Pos: {y_va.sum():,} ({y_va.mean():.2%})")

        # Scale features (fit on train only)
        n_feat = X_tr.shape[-1]
        scaler = StandardScaler()
        X_tr_flat = X_tr.reshape(-1, n_feat)
        X_va_flat = X_va.reshape(-1, n_feat)
        scaler.fit(X_tr_flat)
        X_tr = scaler.transform(X_tr_flat).reshape(X_tr.shape).astype(np.float32)
        X_va = scaler.transform(X_va_flat).reshape(X_va.shape).astype(np.float32)

        # Class weight for imbalance
        pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)],
                                  dtype=torch.float32).to(config['device'])

        train_ds = SepsisDataset(X_tr, y_tr)
        val_ds = SepsisDataset(X_va, y_va)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                                shuffle=False, num_workers=0)

        model = SepsisRNN(
            n_features=n_feat,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            rnn_type=rnn_type,
        ).to(config['device'])

        optimizer = Adam(model.parameters(), lr=config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auroc = 0.0
        best_state = None
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, config['epochs'] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config['device'])
            y_true, y_prob = evaluate(model, val_loader, config['device'])
            try:
                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
            except ValueError:
                auroc, auprc = 0.0, 0.0

            scheduler.step(auroc)

            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:2d} | loss={train_loss:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

            if patience_counter >= config['patience']:
                print(f"    Early stop at epoch {epoch}")
                break

        elapsed = time.time() - t0
        # Load best model and get final metrics
        model.load_state_dict(best_state)
        y_true, y_prob = evaluate(model, val_loader, config['device'])
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            'fold': fold,
            'AUROC': roc_auc_score(y_true, y_prob),
            'AUPRC': average_precision_score(y_true, y_prob),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_true, y_pred),
            'time_sec': elapsed,
        }
        fold_results.append(metrics)
        oof_true.extend(y_true.tolist())
        oof_prob.extend(y_prob.tolist())

        print(f"    ► Fold {fold}: AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | Recall={metrics['Recall']:.4f} | ({elapsed:.1f}s)")

    df_res = pd.DataFrame(fold_results)
    mean_r = df_res.drop(columns=['fold', 'time_sec']).mean().to_dict()
    std_r = df_res.drop(columns=['fold', 'time_sec']).std().to_dict()
    print(f"\n  ►► {rnn_type.upper()} MEAN AUROC: {mean_r['AUROC']:.4f} ± {std_r['AUROC']:.4f}")
    print(f"  ►► {rnn_type.upper()} MEAN AUPRC: {mean_r['AUPRC']:.4f} ± {std_r['AUPRC']:.4f}")
    print(f"  ►► {rnn_type.upper()} MEAN Recall: {mean_r['Recall']:.4f} ± {std_r['Recall']:.4f}")

    return {
        'metrics_df': df_res,
        'mean': mean_r,
        'std': std_r,
        'oof_true': np.array(oof_true),
        'oof_proba': np.array(oof_prob),
    }


# ============================================================
# 7. RUN LSTM AND GRU
# ============================================================
lstm_results = run_cv('lstm', X, y, pids, CONFIG)
gru_results = run_cv('gru', X, y, pids, CONFIG)


# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n[8] Saving results...")

lstm_results['metrics_df'].to_csv('lstm_per_fold.csv', index=False)
gru_results['metrics_df'].to_csv('gru_per_fold.csv', index=False)

summary = {
    'lstm': {**lstm_results['mean'], **{f'{k}_std': v for k, v in lstm_results['std'].items()}},
    'gru': {**gru_results['mean'], **{f'{k}_std': v for k, v in gru_results['std'].items()}},
}
with open('dl_metrics.json', 'w') as f:
    json.dump(summary, f, indent=2)

np.save('lstm_oof_true.npy', lstm_results['oof_true'])
np.save('lstm_oof_prob.npy', lstm_results['oof_proba'])
np.save('gru_oof_true.npy', gru_results['oof_true'])
np.save('gru_oof_prob.npy', gru_results['oof_proba'])

print("  ✓ lstm_per_fold.csv")
print("  ✓ gru_per_fold.csv")
print("  ✓ dl_metrics.json")
print("  ✓ lstm_oof_*.npy, gru_oof_*.npy")

print("\n" + "="*60)
print("  FINAL DEEP LEARNING RESULTS")
print("="*60)
print(f"  LSTM: AUROC={lstm_results['mean']['AUROC']:.4f} ± {lstm_results['std']['AUROC']:.4f} | "
      f"AUPRC={lstm_results['mean']['AUPRC']:.4f} | Recall={lstm_results['mean']['Recall']:.4f}")
print(f"  GRU:  AUROC={gru_results['mean']['AUROC']:.4f} ± {gru_results['std']['AUROC']:.4f} | "
      f"AUPRC={gru_results['mean']['AUPRC']:.4f} | Recall={gru_results['mean']['Recall']:.4f}")
print("\n✓ Deep learning baselines complete.")