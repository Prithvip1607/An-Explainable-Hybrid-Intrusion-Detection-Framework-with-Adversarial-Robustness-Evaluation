"""Five-seed reproducibility study for the headline clean-data models.

Run run_project.py first. The script reuses the sampled CICIDS2017 data,
performs a fresh stratified train/validation/test split for each seed, tunes
thresholds and RF/CNN fusion on validation only, and reports mean +/- std plus
an approximate 95% confidence interval across seeds.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import _clean_frame
from src.models import seed_everything, train_rf, train_cnn, cnn_proba
from src.evaluation import metrics, select_security_threshold

ROOT = Path(__file__).resolve().parent


def main(a):
    data_path = ROOT / 'data' / 'processed' / 'sampled_binary.csv'
    if not data_path.exists():
        raise FileNotFoundError('Run python run_project.py first so data/processed/sampled_binary.csv exists.')
    df = pd.read_csv(data_path)
    X, y, _ = _clean_frame(df)
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    rows = []
    for seed in a.seeds:
        seed_everything(seed)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=a.test_size, stratify=y, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=a.validation_size,
            stratify=y_train_full, random_state=seed
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train).astype('float32')
        Xv = scaler.transform(X_val).astype('float32')
        Xte = scaler.transform(X_test).astype('float32')
        ytr = y_train.to_numpy(); yv = y_val.to_numpy(); yte = y_test.to_numpy()

        rf = train_rf(Xtr, ytr, a.rf_estimators, seed)
        rp_v = rf.predict_proba(Xv)[:, 1]; rp = rf.predict_proba(Xte)[:, 1]
        rt, _ = select_security_threshold(yv, rp_v, a.max_fpr)
        rm = metrics(yte, (rp >= rt).astype(int), rp)
        rows.append({'seed': seed, 'model': 'Random Forest', 'threshold': rt, **rm})

        cnn, _, dev = train_cnn(
            Xtr, ytr, Xtr.shape[1], a.epochs, a.batch_size, a.lr,
            a.weight_decay, a.cnn_hidden, seed=seed
        )
        cp_v = cnn_proba(cnn, Xv, dev); cp = cnn_proba(cnn, Xte, dev)
        ct, _ = select_security_threshold(yv, cp_v, a.max_fpr)
        cm = metrics(yte, (cp >= ct).astype(int), cp)
        rows.append({'seed': seed, 'model': '1D CNN', 'threshold': ct, **cm})

        best = None
        for w in np.arange(0, 1.01, 0.1):
            hpv = w * rp_v + (1-w) * cp_v
            ht, _ = select_security_threshold(yv, hpv, a.max_fpr)
            m = metrics(yv, (hpv >= ht).astype(int), hpv)
            key = (m['fnr'], m['fpr'], -m['f1'])
            if best is None or key < best[0]:
                best = (key, float(w), float(ht))
        w, ht = best[1], best[2]
        hp = w * rp + (1-w) * cp
        hm = metrics(yte, (hp >= ht).astype(int), hp)
        rows.append({
            'seed': seed, 'model': 'RF + CNN Hybrid',
            'threshold': ht, 'rf_weight': w, 'cnn_weight': 1-w, **hm
        })
        print(f'seed={seed}: RF F1={rm["f1"]:.4f}, CNN F1={cm["f1"]:.4f}, Hybrid F1={hm["f1"]:.4f}, hybrid_w={w:.1f}')

    raw = pd.DataFrame(rows)
    raw.to_csv(out / 'multiseed_raw.csv', index=False)
    numeric = ['accuracy','precision','recall','f1','fpr','fnr','roc_auc','pr_auc']
    summary = raw.groupby('model')[numeric].agg(['mean','std']).reset_index()
    summary.to_csv(out / 'multiseed_summary.csv', index=False)

    # Paper-friendly long-format summary with 95% CI of the mean.
    paper_rows = []
    n = len(a.seeds)
    for model, g in raw.groupby('model'):
        for metric_name in numeric:
            vals = g[metric_name].astype(float)
            mean = vals.mean(); std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ci = 1.96 * std / np.sqrt(max(n, 1))
            paper_rows.append({
                'model': model, 'metric': metric_name, 'n_seeds': n,
                'mean': mean, 'std': std,
                'ci95_low': max(0.0, mean-ci), 'ci95_high': min(1.0, mean+ci),
                'mean_plus_minus_std': f'{mean:.4f} +/- {std:.4f}'
            })
    paper = pd.DataFrame(paper_rows)
    paper.to_csv(out / 'multiseed_summary_paper.csv', index=False)

    # Also report the selected threshold/fusion-weight variability.
    tuning = raw.groupby('model')[['threshold'] + ([ 'rf_weight','cnn_weight' ] if 'rf_weight' in raw.columns else [])].agg(['mean','std']).reset_index()
    tuning.to_csv(out / 'multiseed_tuning_summary.csv', index=False)

    print('\n=== MULTI-SEED SUMMARY ===')
    print(summary.round(4).to_string(index=False))
    print('\nSaved: multiseed_raw.csv, multiseed_summary.csv, multiseed_summary_paper.csv, multiseed_tuning_summary.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[42,21,7,52,100])
    p.add_argument('--test-size', type=float, default=.25)
    p.add_argument('--validation-size', type=float, default=.15)
    p.add_argument('--max-fpr', type=float, default=.01)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--cnn-hidden', type=int, default=128)
    p.add_argument('--rf-estimators', type=int, default=250)
    main(p.parse_args())
