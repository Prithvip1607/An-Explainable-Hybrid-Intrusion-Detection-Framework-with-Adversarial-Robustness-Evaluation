import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data import ensure_dataset, sample_csvs, sample_csvs_per_source, clean_with_validation, daywise_with_validation, class_distribution
from src.models import seed_everything, train_rf, train_cnn, cnn_proba, CNN1D
from src.attacks import fgsm, pgd, hybrid_proba
from src.evaluation import metrics, save_confusion, save_curves, save_robustness_plot, save_attack_strength_plot, threshold_sweep, select_security_threshold, save_threshold_plot
from src.xai import shap_rf, lime_rf, hybrid_attributions, explanation_stability, save_branch_importance, natural_language_explanation

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def evaluate_clean_models(Xtr, Xval, Xte, ytr, yval, yte, features, args, out, models_dir, prefix=''):
    """Train RF/CNN and evaluate threshold selection using validation only."""
    tag = f'_{prefix}' if prefix else ''
    seed_everything(args.seed)
    rf = train_rf(Xtr, ytr, args.rf_estimators, args.seed)
    rf_p_val = rf.predict_proba(Xval)[:, 1]
    rf_p = rf.predict_proba(Xte)[:, 1]
    rf_threshold, rf_sweep = select_security_threshold(yval, rf_p_val, args.max_fpr)
    rf_sweep.to_csv(out / f'threshold_sweep_rf{tag}.csv', index=False)
    if not prefix:
        save_threshold_plot(rf_sweep, out / 'threshold_tradeoff_rf.png')
    rf_m = metrics(yte, (rf_p >= rf_threshold).astype(int), rf_p)
    rf_m['threshold'] = rf_threshold
    if not prefix:
        joblib.dump(rf, models_dir / 'random_forest.joblib')
        save_confusion(yte, (rf_p >= rf_threshold).astype(int), out / 'confusion_matrix_rf.png', 'Random Forest — Clean Test')

    cnn, history, device = train_cnn(Xtr, ytr, len(features), args.epochs, args.batch_size, args.lr, args.weight_decay, args.cnn_hidden, seed=args.seed)
    cnn_p_val = cnn_proba(cnn, Xval, device); cnn_p = cnn_proba(cnn, Xte, device)
    cnn_threshold, cnn_sweep = select_security_threshold(yval, cnn_p_val, args.max_fpr)
    cnn_sweep.to_csv(out / f'threshold_sweep_cnn{tag}.csv', index=False)
    if not prefix:
        save_threshold_plot(cnn_sweep, out / 'threshold_tradeoff_cnn.png')
    cnn_m = metrics(yte, (cnn_p >= cnn_threshold).astype(int), cnn_p); cnn_m['threshold'] = cnn_threshold
    if not prefix:
        torch.save({'state_dict': cnn.state_dict(), 'n_features': len(features), 'hidden': args.cnn_hidden}, models_dir / 'cnn.pt')
        save_confusion(yte, (cnn_p >= cnn_threshold).astype(int), out / 'confusion_matrix_cnn.png', '1D CNN — Clean Test')

    # Tune hybrid fusion weight + threshold on validation data, never on test.
    candidates = np.arange(0.0, 1.01, 0.1)
    fusion_rows = []
    best = None
    for w in candidates:
        hpv = w * rf_p_val + (1 - w) * cnn_p_val
        threshold, _ = select_security_threshold(yval, hpv, args.max_fpr)
        m = metrics(yval, (hpv >= threshold).astype(int), hpv)
        fusion_rows.append({'rf_weight': float(w), 'cnn_weight': float(1-w), 'threshold': threshold, **m})
        key = (m['fnr'], m['fpr'], -m['f1'])
        if best is None or key < best[0]: best = (key, float(w), threshold)
    pd.DataFrame(fusion_rows).to_csv(out / f'fusion_weight_validation{tag}.csv', index=False)
    rf_weight, hybrid_threshold = best[1], best[2]
    hy_p = rf_weight * rf_p + (1 - rf_weight) * cnn_p
    hy_m = metrics(yte, (hy_p >= hybrid_threshold).astype(int), hy_p); hy_m['rf_weight'] = rf_weight; hy_m['cnn_weight'] = 1-rf_weight; hy_m['threshold'] = hybrid_threshold
    if not prefix:
        save_confusion(yte, (hy_p >= hybrid_threshold).astype(int), out / 'confusion_matrix_hybrid.png', 'RF + CNN Hybrid — Clean Test')
        save_curves(yte, {'Random Forest': rf_p, '1D CNN': cnn_p, 'Hybrid': hy_p}, out)
        joblib.dump({'rf_weight': rf_weight, 'threshold': hybrid_threshold}, models_dir / 'hybrid_config.joblib')
    return rf, cnn, device, rf_m, cnn_m, hy_m, rf_p, cnn_p, hy_p, rf_threshold, cnn_threshold, hybrid_threshold, rf_weight


def multiclass_experiment(df, args, out):
    """Secondary multiclass RF experiment using original CICIDS labels."""
    from src.data import _clean_frame
    X, y_binary, meta = _clean_frame(df)
    labels = df.loc[X.index, '__original_label'].astype(str).str.strip().to_numpy()
    # Keep classes with enough examples for a meaningful held-out evaluation.
    vc = pd.Series(labels).value_counts()
    keep_classes = vc[vc >= args.min_multiclass_samples].index.tolist()
    mask = np.isin(labels, keep_classes)
    X = X.iloc[np.where(mask)[0]]; labels = labels[mask]
    if len(np.unique(labels)) < 3:
        return {'status': 'skipped', 'reason': 'Fewer than 3 classes met the minimum sample threshold.'}
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=args.test_size, stratify=labels, random_state=args.seed)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    model = RandomForestClassifier(n_estimators=args.rf_estimators, random_state=args.seed, n_jobs=-1, class_weight='balanced_subsample')
    model.fit(Xtr_s, ytr); pred = model.predict(Xte_s)
    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / 'multiclass_classification_report.csv')
    pd.DataFrame({'true': yte, 'pred': pred}).to_csv(out / 'multiclass_predictions.csv', index=False)
    return {'status': 'completed', 'classes': keep_classes, 'samples': int(len(labels)), 'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score']}


def main(a):
    seed_everything(a.seed)
    out = ROOT / 'results'; models_dir = ROOT / 'models'; out.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True)
    processed = ROOT / 'data' / 'processed'; processed.mkdir(exist_ok=True, parents=True)
    t0 = time.time()

    csvs = ensure_dataset(ROOT / 'data' / 'raw')
    print('CSV files:', [p.name for p in csvs])
    df = sample_csvs(csvs, a.max_per_class, a.seed)
    df.to_csv(processed / 'sampled_binary.csv', index=False)
    class_distribution(df).to_csv(out / 'original_class_distribution.csv', header=['count'])

    Xtr, Xval, Xte, ytr, yval, yte, features, scaler, meta_tr, meta_val, meta_te = clean_with_validation(
        df, a.seed, a.test_size, a.validation_size
    )
    joblib.dump({'scaler': scaler, 'features': features}, models_dir / 'preprocessor.joblib')
    np.save(processed / 'test_labels.npy', yte)
    np.save(processed / 'test_features_scaled.npy', Xte)
    print(f'Prepared train={len(ytr):,}, validation={len(yval):,}, test={len(yte):,}, features={len(features)}')
    print('Train classes:', dict(pd.Series(ytr).value_counts().sort_index()))

    # 1) Clean baseline + adaptive threshold + validation-tuned fusion.
    rf, cnn, device, rf_m, cnn_m, hy_m, rf_p, cnn_p, hy_p, rf_thr, cnn_thr, hy_thr, rf_weight = evaluate_clean_models(
        Xtr, Xval, Xte, ytr, yval, yte, features, a, out, models_dir
    )

    comparison = pd.DataFrame([
        {'model': 'Random Forest', **rf_m},
        {'model': '1D CNN', **cnn_m},
        {'model': 'RF + CNN Hybrid', **hy_m},
    ])
    comparison.to_csv(out / 'model_comparison.csv', index=False)
    comparison.to_json(out / 'model_comparison.json', orient='records', indent=2)

    # 1b) Temporal/day-wise generalization. The final source CSV is held out;
    # preprocessing is fit only on the training files. This is a separate
    # experiment and does not replace the conventional random split.
    if not a.skip_daywise:
        try:
            daywise_df = sample_csvs_per_source(csvs, max_per_class=a.daywise_max_per_class, seed=a.seed)
            daywise_df.to_csv(processed / 'daywise_sampled_binary.csv', index=False)
            (Xd_tr, Xd_val, Xd_te, yd_tr, yd_val, yd_te, dfeatures, dscaler, dmeta_tr, dmeta_val, dmeta_te, test_sources) = daywise_with_validation(
                daywise_df, test_source_pattern=a.daywise_test_pattern, test_day=a.daywise_test_day,
                seed=a.seed, validation_size=a.validation_size
            )
            drf, dcnn, ddev, drm, dcm, dhm, drp, dcp, dhp, drthr, dcthr, dhthr, drw = evaluate_clean_models(
                Xd_tr, Xd_val, Xd_te, yd_tr, yd_val, yd_te, dfeatures, a, out, models_dir, prefix='daywise'
            )
            day_comp = pd.DataFrame([
                {'split': 'day-wise', 'test_sources': ';'.join(test_sources), 'model': 'Random Forest', **drm},
                {'split': 'day-wise', 'test_sources': ';'.join(test_sources), 'model': '1D CNN', **dcm},
                {'split': 'day-wise', 'test_sources': ';'.join(test_sources), 'model': 'RF + CNN Hybrid', **dhm},
            ])
            day_comp.to_csv(out / 'daywise_model_comparison.csv', index=False)
            day_info = {
                'test_sources': test_sources,
                'test_rows': int(len(yd_te)),
                'test_benign': int((yd_te == 0).sum()),
                'test_attack': int((yd_te == 1).sum()),
                'train_rows': int(len(yd_tr)),
                'validation_rows': int(len(yd_val)),
                'note': 'Whole CICIDS2017 weekday/capture group held out; scaler and model selection use training/validation data only.'
            }
            save_json(out / 'daywise_split_info.json', day_info)
            # Do not overwrite the main trained models: those remain the random-split models.
            del drf, dcnn
        except Exception as e:
            (out / 'daywise_error.txt').write_text(str(e), encoding='utf-8')
            print(f'[WARNING] Day-wise evaluation skipped: {e}')

    # 2) Adversarial robustness of CNN and hybrid. Attacks target the CNN branch;
    # this is explicitly labeled as a transfer/branch-targeted threat model.
    n_adv = min(a.max_adv_samples, len(Xte))
    idx = np.random.default_rng(a.seed).choice(len(Xte), n_adv, replace=False)
    Xa, ya = Xte[idx], yte[idx]
    adv_f = fgsm(cnn, Xa, ya, a.fgsm_epsilon, device)
    adv_p = pgd(cnn, Xa, ya, a.pgd_epsilon, a.pgd_alpha, a.pgd_steps, device)
    rows = []
    for condition, Xattack in [('FGSM', adv_f), ('PGD', adv_p)]:
        cp = cnn_proba(cnn, Xattack, device); hp = hybrid_proba(rf, cnn, Xattack, device, rf_weight)
        rows += [
            {'model': 'CNN', 'condition': condition, 'attack_target': 'CNN branch', **metrics(ya, (cp >= cnn_thr).astype(int), cp)},
            {'model': 'Hybrid', 'condition': condition, 'attack_target': 'CNN branch / transfer to hybrid', **metrics(ya, (hp >= hy_thr).astype(int), hp)},
        ]
    cp_clean = cnn_p[idx]; hp_clean = hy_p[idx]
    rows += [
        {'model': 'CNN', 'condition': 'Clean', 'attack_target': 'None', **metrics(ya, (cp_clean >= cnn_thr).astype(int), cp_clean)},
        {'model': 'Hybrid', 'condition': 'Clean', 'attack_target': 'None', **metrics(ya, (hp_clean >= hy_thr).astype(int), hp_clean)},
    ]
    robust = pd.DataFrame(rows)

    # 3) Adversarial training variants: FGSM and PGD. Each defended CNN gets
    # its own validation-tuned hybrid threshold, and attacks are regenerated
    # against that defended model. This makes the robustness comparison fair.
    adv_train_rows = []
    robust_models = {}
    robust_configs = {}
    n_robust = min(a.max_adv_train, len(Xtr)); ridx = np.random.default_rng(a.seed + 1).choice(len(Xtr), n_robust, replace=False)
    Xbase, ybase = Xtr[ridx], ytr[ridx]
    for defence in ['FGSM', 'PGD']:
        if defence == 'FGSM':
            Xadv_train = fgsm(cnn, Xbase, ybase, a.fgsm_epsilon, device)
            train_seed = a.seed + 11
        else:
            Xadv_train = pgd(cnn, Xbase, ybase, a.pgd_epsilon, a.pgd_alpha, a.pgd_steps, device)
            train_seed = a.seed + 17
        Xaug = np.concatenate([Xbase, Xadv_train]); yaug = np.concatenate([ybase, ybase])
        robust_cnn, _, rdev = train_cnn(
            Xaug, yaug, len(features), max(3, a.adv_epochs), a.batch_size,
            a.lr, a.weight_decay, a.cnn_hidden, device, seed=train_seed
        )
        robust_models[defence] = robust_cnn
        # Tune the defended hybrid threshold on validation data only.
        rp_val = rf.predict_proba(Xval)[:, 1]
        rcp_val = cnn_proba(robust_cnn, Xval, rdev)
        best_def = None
        for w in np.arange(0.0, 1.01, 0.1):
            hpv_def = w * rp_val + (1-w) * rcp_val
            t_def, _ = select_security_threshold(yval, hpv_def, a.max_fpr)
            m_def = metrics(yval, (hpv_def >= t_def).astype(int), hpv_def)
            key = (m_def['fnr'], m_def['fpr'], -m_def['f1'])
            if best_def is None or key < best_def[0]:
                best_def = (key, float(w), float(t_def))
        def_weight, def_threshold = best_def[1], best_def[2]
        robust_configs[defence] = {'rf_weight': def_weight, 'cnn_weight': 1-def_weight, 'threshold': def_threshold}
        if defence == 'FGSM':
            torch.save({'state_dict': robust_cnn.state_dict(), 'n_features': len(features), 'hidden': a.cnn_hidden}, models_dir / 'cnn_adversarial_fgsm.pt')
        else:
            torch.save({'state_dict': robust_cnn.state_dict(), 'n_features': len(features), 'hidden': a.cnn_hidden}, models_dir / 'cnn_adversarial_pgd.pt')

        # Generate a fresh attack set against the defended model itself.
        def_adv_f = fgsm(robust_cnn, Xa, ya, a.fgsm_epsilon, rdev)
        def_adv_p = pgd(robust_cnn, Xa, ya, a.pgd_epsilon, a.pgd_alpha, a.pgd_steps, rdev)
        def_clean_p = cnn_proba(robust_cnn, Xte, rdev)
        for condition, Xe, ye in [('Clean', Xte, yte), ('FGSM', def_adv_f, ya), ('PGD', def_adv_p, ya)]:
            cp_def = cnn_proba(robust_cnn, Xe, rdev)
            hp_def = def_weight * rf.predict_proba(Xe)[:, 1] + (1-def_weight) * cp_def
            adv_train_rows.append({
                'model': f'Hybrid + {defence} adversarial training',
                'condition': condition,
                'attack_target': f'{defence}-trained CNN / hybrid transfer',
                'threshold': def_threshold,
                'rf_weight': def_weight,
                'cnn_weight': 1-def_weight,
                **metrics(ye, (hp_def >= def_threshold).astype(int), hp_def)
            })

    robust = pd.concat([robust, pd.DataFrame(adv_train_rows)], ignore_index=True)
    robust.to_csv(out / 'robustness_comparison.csv', index=False)
    save_robustness_plot(robust, out / 'robustness_comparison.png')

    # 3b) Paper-ready ablation: compare the base system with each defense on
    # clean, FGSM and PGD conditions. Each defense was tuned on validation only.
    ablation_rows = []
    ablation_rows.extend([
        {'configuration': 'Random Forest', 'condition': 'Clean', **rf_m},
        {'configuration': '1D CNN', 'condition': 'Clean', **cnn_m},
        {'configuration': 'RF + CNN Hybrid', 'condition': 'Clean', **hy_m},
    ])
    # Base hybrid under attack.
    for condition in ['FGSM', 'PGD']:
        r = robust[(robust['model'] == 'Hybrid') & (robust['condition'] == condition)].iloc[0].to_dict()
        r.pop('model', None); r.pop('attack_target', None)
        ablation_rows.append({'configuration': 'RF + CNN Hybrid', 'condition': condition, **r})
    # Defended hybrids under clean and attack conditions.
    for defence in ['FGSM', 'PGD']:
        for condition in ['Clean', 'FGSM', 'PGD']:
            r = robust[(robust['model'] == f'Hybrid + {defence} adversarial training') & (robust['condition'] == condition)].iloc[0].to_dict()
            r.pop('model', None); r.pop('attack_target', None)
            ablation_rows.append({'configuration': f'Hybrid + {defence} adversarial training', 'condition': condition, **r})
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(out / 'ablation_study.csv', index=False)
    # Compact table for direct insertion into a paper.
    ablation[['configuration','condition','accuracy','precision','recall','f1','fpr','fnr','roc_auc','pr_auc']].to_csv(out / 'ablation_study_paper.csv', index=False)

    # Binary classification report for the main hybrid.
    report = classification_report(yte, (hy_p >= hy_thr).astype(int), target_names=['BENIGN','ATTACK'], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / 'binary_classification_report.csv')

    # Fixed 0.5 results are retained as a transparent baseline against the
    # validation-tuned threshold results in model_comparison.csv.
    fixed = pd.DataFrame([
        {'model': 'Random Forest', **metrics(yte, (rf_p>=0.5).astype(int), rf_p)},
        {'model': '1D CNN', **metrics(yte, (cnn_p>=0.5).astype(int), cnn_p)},
        {'model': 'RF + CNN Hybrid', **metrics(yte, (hy_p>=0.5).astype(int), hy_p)},
    ])
    fixed.to_csv(out / 'model_comparison_fixed_0.5.csv', index=False)

    # 4) Attack-strength sweep for the undefended CNN/hybrid. Include epsilon=0
    # as an explicit clean anchor and report absolute/relative degradation.
    strength_rows = []
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    clean_acc = {}
    for eps in epsilons:
        if eps == 0.0:
            attack_sets = [('Clean', Xte)]
        else:
            af = fgsm(cnn, Xa, ya, eps, device)
            ap = pgd(cnn, Xa, ya, eps, eps/4, max(5, a.pgd_steps), device)
            attack_sets = [('FGSM', af), ('PGD', ap)]
        for attack_name, Xattack in attack_sets:
            if attack_name == 'Clean':
                y_attack = yte
            else:
                y_attack = ya
            cp = cnn_proba(cnn, Xattack, device)
            hp = hybrid_proba(rf, cnn, Xattack, device, rf_weight)
            for model_name, prob, thr in [('CNN', cp, cnn_thr), ('Hybrid', hp, hy_thr)]:
                m = metrics(y_attack, (prob >= thr).astype(int), prob)
                if attack_name == 'Clean':
                    clean_acc[model_name] = m['accuracy']
                row = {'epsilon': eps, 'attack': attack_name, 'model': model_name, **m}
                if model_name in clean_acc:
                    row['accuracy_drop_vs_clean'] = clean_acc[model_name] - m['accuracy']
                    row['relative_accuracy_drop_pct'] = 100 * (clean_acc[model_name] - m['accuracy']) / max(clean_acc[model_name], 1e-12)
                strength_rows.append(row)
    strength = pd.DataFrame(strength_rows)
    strength.to_csv(out / 'attack_strength_sweep.csv', index=False)
    save_attack_strength_plot(strength, out / 'attack_strength_sweep.png')

    # 5) Threshold analysis on validation set; the test set is not used for tuning.
    hpv = rf_weight * rf.predict_proba(Xval)[:,1] + (1-rf_weight) * cnn_proba(cnn, Xval, device)
    threshold_df = threshold_sweep(yval, hpv, np.linspace(0.05, 0.95, 37))
    threshold_df.to_csv(out / 'hybrid_threshold_sweep_validation.csv', index=False)
    save_threshold_plot(threshold_df, out / 'hybrid_threshold_tradeoff.png')

    # 6) Explainability: RF SHAP + LIME + CNN IG + weighted hybrid attribution.
    n_xai = min(a.max_xai_samples, len(Xte)); xi = np.random.default_rng(a.seed+2).choice(len(Xte), n_xai, replace=False)
    Xx = Xte[xi]
    try:
        fi = shap_rf(rf, Xtr, Xx, features, out)
        lime_rf(rf, Xtr, Xx[0], features, out, seed=a.seed)
        rf_attr, cnn_attr, hybrid_attr = hybrid_attributions(rf, cnn, Xx, features, device, rf_weight)
        save_branch_importance(rf_attr, cnn_attr, hybrid_attr, features, out)
        # Explain clean vs FGSM and PGD on the same subset.
        clean_attr = hybrid_attr
        fg_attr = hybrid_attributions(rf, cnn, adv_f[:n_xai], features, device, rf_weight)[2]
        pg_attr = hybrid_attributions(rf, cnn, adv_p[:n_xai], features, device, rf_weight)[2]
        explanation_stability(clean_attr, fg_attr, features, out / 'xai_fgsm', k=10)
        explanation_stability(clean_attr, pg_attr, features, out / 'xai_pgd', k=10)
        explanation = natural_language_explanation(fi, int((hy_p[xi[0]] >= hy_thr)))
        (out / 'example_explanation.txt').write_text(explanation, encoding='utf-8')
    except Exception as e:
        fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        fi.to_csv(out / 'top_features.csv', header=['rf_importance'])
        (out / 'xai_error.txt').write_text(f'XAI failed: {e}\nRF feature importance was saved instead.', encoding='utf-8')

    # 7) Multiclass secondary study.
    if not a.skip_multiclass:
        multi = multiclass_experiment(df, a, out)
    else:
        multi = {'status': 'skipped', 'reason': 'disabled by user'}

    # 8) Efficiency and reproducibility metadata.
    efficiency = {
        'rf_n_estimators': a.rf_estimators,
        'cnn_epochs': a.epochs,
        'adv_epochs': a.adv_epochs,
        'adv_eval_samples': n_adv,
        'xai_samples': n_xai,
        'device': str(device),
    }
    save_json(out / 'efficiency_config.json', efficiency)

    summary = {
        'dataset_files': [p.name for p in csvs],
        'sampled_rows_before_cleaning': int(len(df)),
        'clean_rows': int(len(ytr)+len(yval)+len(yte)),
        'train_rows': int(len(ytr)), 'validation_rows': int(len(yval)), 'test_rows': int(len(yte)),
        'features': int(len(features)), 'max_per_original_class': a.max_per_class,
        'device': str(device), 'seed': a.seed, 'epochs': a.epochs,
        'validation_size': a.validation_size, 'adaptive_threshold_max_fpr': a.max_fpr,
        'hybrid_rf_weight': rf_weight, 'hybrid_cnn_weight': 1-rf_weight, 'hybrid_threshold': hy_thr,
        'fgsm_epsilon': a.fgsm_epsilon, 'pgd_epsilon': a.pgd_epsilon, 'pgd_alpha': a.pgd_alpha, 'pgd_steps': a.pgd_steps,
        'multiclass': multi,
        'runtime_minutes': round((time.time()-t0)/60, 2),
        'note': 'All reported metrics are generated from the local CICIDS2017 data and the sampling/evaluation settings recorded here. No synthetic results are produced by the pipeline.'
    }
    save_json(out / 'run_summary.json', summary)
    (out / 'run_summary.txt').write_text('\n'.join(f'{k}: {v}' for k,v in summary.items()), encoding='utf-8')

    print('\n=== CLEAN MODEL COMPARISON ===')
    print(comparison.round(4).to_string(index=False))
    print('\n=== ROBUSTNESS ===')
    print(robust.round(4).to_string(index=False))
    print(f'\nDone. Results saved to {out}.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='STAAR-IDS research-grade reproducible pipeline')
    p.add_argument('--max-per-class', type=int, default=30000)
    p.add_argument('--max-adv-samples', type=int, default=4000)
    p.add_argument('--max-adv-train', type=int, default=8000)
    p.add_argument('--max-xai-samples', type=int, default=100)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--adv-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--cnn-hidden', type=int, default=128)
    p.add_argument('--rf-estimators', type=int, default=250)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=.25)
    p.add_argument('--validation-size', type=float, default=.15)
    p.add_argument('--max-fpr', type=float, default=.01)
    p.add_argument('--fgsm-epsilon', type=float, default=.03)
    p.add_argument('--pgd-epsilon', type=float, default=.03)
    p.add_argument('--pgd-alpha', type=float, default=.0075)
    p.add_argument('--pgd-steps', type=int, default=5)
    p.add_argument('--min-multiclass-samples', type=int, default=1000)
    p.add_argument('--skip-multiclass', action='store_true')
    p.add_argument('--skip-daywise', action='store_true')
    p.add_argument('--daywise-test-pattern', type=str, default=None, help='Regex for source CSVs to hold out; overrides automatic weekday selection.')
    p.add_argument('--daywise-test-day', type=str, default=None, help='CICIDS weekday to hold out, e.g. Friday; overrides automatic selection.')
    p.add_argument('--daywise-max-per-class', type=int, default=5000, help='Per-source class cap used only for day-wise evaluation.')
    main(p.parse_args())
