from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)


def metrics(y, pred, proba=None):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    d = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'fpr': fp / (fp + tn) if fp + tn else 0.0,
        'fnr': fn / (fn + tp) if fn + tp else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    if proba is not None and len(np.unique(y)) == 2:
        d['roc_auc'] = roc_auc_score(y, proba)
        d['pr_auc'] = average_precision_score(y, proba)
    return d


def threshold_metrics(y, proba, threshold):
    return metrics(y, (np.asarray(proba) >= threshold).astype(int), proba)


def threshold_sweep(y, proba, thresholds=None):
    thresholds = np.asarray(thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19))
    rows = []
    for t in thresholds:
        m = threshold_metrics(y, proba, float(t))
        rows.append({'threshold': float(t), **m})
    return pd.DataFrame(rows)


def select_security_threshold(y, proba, max_fpr=0.01, objective='fnr'):
    """Select the threshold with lowest FNR under an FPR constraint.

    If no threshold satisfies the constraint, select the threshold with the
    lowest FPR and then lowest FNR. This rule is determined before test use by
    applying it to validation data in the main pipeline.
    """
    sweep = threshold_sweep(y, proba, np.linspace(0.05, 0.95, 37))
    feasible = sweep[sweep['fpr'] <= max_fpr]
    if len(feasible):
        best = feasible.sort_values([objective, 'fpr', 'threshold'], ascending=[True, True, True]).iloc[0]
    else:
        best = sweep.sort_values(['fpr', 'fnr', 'threshold'], ascending=[True, True, True]).iloc[0]
    return float(best['threshold']), sweep


def save_confusion(y, pred, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y, pred, labels=[0, 1]),
        display_labels=['Benign', 'Attack']
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_curves(y, probas, pathdir, prefix=''):
    pathdir = Path(pathdir); pathdir.mkdir(exist_ok=True, parents=True)
    suffix = f'_{prefix}' if prefix else ''
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y,p):.3f})')
    ax.plot([0, 1], [0, 1], '--'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve'); ax.legend(); fig.tight_layout(); fig.savefig(pathdir/f'roc_curve{suffix}.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f'{name} (AP={average_precision_score(y,p):.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curve')
    ax.legend(); fig.tight_layout(); fig.savefig(pathdir/f'pr_curve{suffix}.png', dpi=180); plt.close(fig)


def save_robustness_plot(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in df.groupby('model'):
        g = g.set_index('condition').reindex(['Clean', 'FGSM', 'PGD']).dropna()
        ax.plot(g.index, g['f1'], marker='o', label=model)
    ax.set_ylabel('F1-score'); ax.set_xlabel('Condition'); ax.set_ylim(0, 1.02)
    ax.set_title('Adversarial Robustness — F1'); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_threshold_plot(sweep, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep['threshold'], sweep['fpr'], marker='o', label='FPR')
    ax.plot(sweep['threshold'], sweep['fnr'], marker='o', label='FNR')
    ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Rate'); ax.set_title('FPR/FNR vs Decision Threshold')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_attack_strength_plot(df, path):
    """Plot accuracy degradation as attack strength increases."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (model, attack), g in df.groupby(['model', 'attack']):
        g = g.sort_values('epsilon')
        ax.plot(g['epsilon'], g['accuracy'], marker='o', label=f'{model} — {attack}')
    ax.set_xlabel('FGSM/PGD epsilon')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title('Attack-Strength Sensitivity')
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
