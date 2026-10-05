# Research-paper results guide

## Recommended research questions

**RQ1.** Does RF–CNN fusion improve detection compared with individual RF and 1D-CNN models?

**RQ2.** How much do FGSM and PGD degrade detection performance?

**RQ3.** Does adversarial training recover robustness under the tested threat model?

**RQ4.** Does adversarial perturbation change the model's feature explanations?

**RQ5.** Does the model generalize from randomly split traffic to a temporally separated capture day/source file?

**RQ6.** What is the trade-off between false-positive and false-negative rates at different decision thresholds?

## Tables to use in the paper

1. Dataset composition and preprocessing.
2. Clean RF/CNN/Hybrid performance.
3. Ablation: baseline → hybrid → adversarial training → adaptive threshold.
4. FGSM/PGD robustness.
5. Attack-strength sweep.
6. Random split vs day-wise split.
7. XAI feature attribution and clean-vs-adversarial stability.
8. Secondary multiclass performance.
9. Five-seed mean ± standard deviation for headline models.
10. Computational configuration/cost.

## Figures to use

- `roc_curve.png`
- `pr_curve.png`
- three clean confusion matrices
- `robustness_comparison.png`
- `hybrid_threshold_tradeoff.png`
- `hybrid_feature_attribution.png`
- SHAP summary
- clean-vs-adversarial explanation stability summary/plot (create a paper plot from the CSVs if desired)

## Claims to avoid

- Do not claim the hybrid is universally best if RF has higher F1 on the final test set.
- Do not claim adversarial training improves robustness unless the final measured results support it.
- Do not call CNN-targeted FGSM/PGD a full white-box attack on the RF–CNN hybrid.
- Do not call the Streamlit dashboard a live production IDS.
- Do not report a single random seed as evidence of reproducibility.
- Do not claim rare multiclass categories are reliable without sufficient test support.


## Final experiment additions (September 2026)

The final pipeline now includes three strengthened experiments:

1. **Day-wise generalization:** CICIDS2017 CSVs are grouped by weekday. The default selector holds out the latest available weekday containing both BENIGN and ATTACK traffic, avoiding the previous attack-only CSV failure. Use `--daywise-test-day Friday` to force a weekday or `--daywise-test-pattern REGEX` to force a source pattern.
2. **Attack-strength sweep:** evaluates epsilon = 0, 0.005, 0.01, 0.02, 0.03, 0.05 for FGSM and PGD, with accuracy drop and relative degradation saved in `results/attack_strength_sweep.csv` plus `attack_strength_sweep.png`.
3. **Defense ablation:** compares the base hybrid with FGSM-trained and PGD-trained hybrid variants under clean, FGSM and PGD conditions. Each defended variant tunes its fusion weight and threshold on validation data only. Paper-ready output is `results/ablation_study_paper.csv`.

The five-seed script also produces `multiseed_summary_paper.csv` with mean, standard deviation and approximate 95% confidence intervals.
