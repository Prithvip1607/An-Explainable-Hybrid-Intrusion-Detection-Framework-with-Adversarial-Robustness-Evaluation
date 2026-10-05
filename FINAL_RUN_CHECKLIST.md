# Final run checklist

## Before the final run

- [ ] Put the real CICIDS2017 `MachineLearningCSV.zip` in `data/raw/`.
- [ ] Confirm the project is using the intended Python environment.
- [ ] Record CPU/GPU, RAM, OS and Python/library versions.
- [ ] Decide and freeze the final sampling cap, epochs and RF estimators.
- [ ] Do not tune thresholds or model settings using the final test set.

## Main run

```bash
python run_project.py
```

## Reproducibility run

After the main run:

```bash
python run_multiseed.py --seeds 42,21,7,52,100
```

This produces `results/multiseed_raw.csv` and `results/multiseed_summary.csv`.

## Review the outputs

- [ ] `model_comparison.csv`
- [ ] `robustness_comparison.csv`
- [ ] `attack_strength_sweep.csv`
- [ ] `daywise_model_comparison.csv`
- [ ] `hybrid_threshold_sweep_validation.csv`
- [ ] `hybrid_feature_attributions.csv`
- [ ] `xai_fgsm/xai_stability_summary.csv`
- [ ] `xai_pgd/xai_stability_summary.csv`
- [ ] `multiclass_classification_report.csv`
- [ ] `multiseed_summary.csv`
- [ ] `run_summary.json`

## Before writing the paper

Freeze the exact CSVs/figures used in the manuscript and do not replace them with better-looking runs without updating the experimental description.


## Final reproducibility run

After installing requirements and placing `MachineLearningCSV.zip` in `data/raw/`:

```bash
python run_project.py
python run_multiseed.py --seeds 42,21,7,52,100
```

Confirm these files exist before writing the paper:
- `results/model_comparison.csv`
- `results/robustness_comparison.csv`
- `results/ablation_study_paper.csv`
- `results/attack_strength_sweep.csv`
- `results/daywise_model_comparison.csv`
- `results/daywise_split_info.json`
- `results/multiseed_summary_paper.csv`
- `results/binary_classification_report.csv`
- `results/model_comparison_fixed_0.5.csv`

Do not mix these final results with the older Review-2 numbers.
