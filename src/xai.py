from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def shap_rf(model, X_background, X_sample, feature_names, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    import shap
    # TreeExplainer does not require the background matrix for RF, but keeping
    # the argument preserves the project API and documents the intended scope.
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        vals = sv[1]
    elif getattr(sv, 'ndim', 0) == 3:
        vals = sv[:, :, 1]
    else:
        vals = sv
    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=feature_names).sort_values(ascending=False)
    mean_abs.to_csv(outdir / 'top_features.csv', header=['mean_abs_shap'])
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs.head(15).sort_values().plot.barh(ax=ax)
    ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Top SHAP Features — Random Forest')
    fig.tight_layout(); fig.savefig(outdir / 'shap_summary.png', dpi=180); plt.close(fig)
    return mean_abs


def shap_values_rf(model, X_sample):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X_sample)
    if isinstance(sv, list): return np.asarray(sv[1])
    if getattr(sv, 'ndim', 0) == 3: return np.asarray(sv[:, :, 1])
    return np.asarray(sv)


def lime_rf(model, X_train, row, feature_names, outdir, seed=42):
    from lime.lime_tabular import LimeTabularExplainer
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=['BENIGN', 'ATTACK'],
        mode='classification', discretize_continuous=True, random_state=seed
    )
    exp = explainer.explain_instance(row, model.predict_proba, num_features=10)
    exp.save_to_file(str(outdir / 'lime_explanation.html'))
    pd.DataFrame(exp.as_list(), columns=['feature_condition', 'lime_weight']).to_csv(
        outdir / 'lime_local_features.csv', index=False
    )
    return exp


def integrated_gradients(model, X, device, baseline=None, steps=24):
    """Feature attribution for the CNN branch using Integrated Gradients."""
    import torch
    model.eval()
    x = torch.from_numpy(np.asarray(X, dtype='float32')).to(device).unsqueeze(1)
    base_np = np.zeros_like(X, dtype='float32') if baseline is None else np.asarray(baseline, dtype='float32')
    base = torch.from_numpy(base_np).to(device).unsqueeze(1)
    total_grad = torch.zeros_like(x)
    for alpha in np.linspace(0.0, 1.0, steps):
        xi = (base + alpha * (x - base)).detach().requires_grad_(True)
        logits = model(xi)[:, 1].sum()
        model.zero_grad(); logits.backward()
        total_grad += xi.grad.detach()
    avg_grad = total_grad / steps
    attr = ((x - base) * avg_grad).squeeze(1).detach().cpu().numpy()
    return attr


def hybrid_attributions(rf, cnn, X, feature_names, device, rf_weight=0.5):
    rf_attr = shap_values_rf(rf, X)
    cnn_attr = integrated_gradients(cnn, X, device)
    # The hybrid probability is a weighted fusion; scaling branch attribution
    # by the same fusion weights provides a transparent branch-level diagnostic.
    hybrid_attr = rf_weight * rf_attr + (1 - rf_weight) * cnn_attr
    return rf_attr, cnn_attr, hybrid_attr


def topk_overlap(a, b, k=10):
    ia = set(np.argsort(np.abs(a))[-k:]); ib = set(np.argsort(np.abs(b))[-k:])
    return len(ia & ib) / max(1, len(ia | ib))


def explanation_stability(clean_attr, adv_attr, feature_names, outdir, k=10):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(clean_attr)):
        rows.append({
            'sample': i,
            'top_k': k,
            'jaccard_top_k': topk_overlap(clean_attr[i], adv_attr[i], k),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'xai_stability.csv', index=False)
    summary = pd.DataFrame([{
        'top_k': k,
        'mean_jaccard': df['jaccard_top_k'].mean() if len(df) else np.nan,
        'std_jaccard': df['jaccard_top_k'].std(ddof=1) if len(df) > 1 else 0.0,
    }])
    summary.to_csv(outdir / 'xai_stability_summary.csv', index=False)
    return summary


def save_branch_importance(rf_attr, cnn_attr, hybrid_attr, feature_names, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({
        'feature': feature_names,
        'rf_mean_abs_shap': np.abs(rf_attr).mean(axis=0),
        'cnn_mean_abs_ig': np.abs(cnn_attr).mean(axis=0),
        'hybrid_mean_abs_attribution': np.abs(hybrid_attr).mean(axis=0),
    }).sort_values('hybrid_mean_abs_attribution', ascending=False)
    data.to_csv(outdir / 'hybrid_feature_attributions.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    data.head(15).sort_values('hybrid_mean_abs_attribution').plot.barh(
        x='feature', y='hybrid_mean_abs_attribution', legend=False, ax=ax
    )
    ax.set_xlabel('Mean absolute attribution'); ax.set_title('Hybrid Feature Attribution')
    fig.tight_layout(); fig.savefig(outdir / 'hybrid_feature_attribution.png', dpi=180); plt.close(fig)
    return data


def natural_language_explanation(feature_importance, prediction):
    label = 'ATTACK' if int(prediction) == 1 else 'BENIGN'
    top = feature_importance.head(5).index.tolist()
    return f'The model predicts {label}. The most influential traffic features for this decision are: ' + ', '.join(top) + '.'
