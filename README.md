# IDS — Explainable & Adversarially Robust Intrusion Detection System

**Publication-oriented final project implementation**

Team: Prithvi Kumar Pandey(Team Lead), Abhipsa Satapathy, Laxmishree Prusty, Ashmit Tirkey  
Supervisor: Dr. Binita Kumari

## What this version contains

The project keeps the original Review-2 architecture and adds the experimental controls needed for a research paper:

- CICIDS2017 CSV/ZIP discovery and automatic ZIP extraction
- Memory-conscious per-original-class sampling
- NaN/Infinity handling, numeric feature selection and duplicate removal
- Train/validation/test split with the scaler fitted on **training data only**
- Random Forest baseline
- PyTorch 1D-CNN baseline
- Validation-tuned RF–CNN probability fusion
- Validation-tuned security threshold under an FPR constraint
- Accuracy, precision, recall, F1, FPR, FNR, ROC-AUC and PR-AUC
- Confusion matrices and ROC/PR curves
- FGSM and PGD adversarial evaluation
- FGSM-trained and PGD-trained CNN variants
- Attack-strength sweep
- Day-wise/source-file generalization experiment
- SHAP for the RF branch
- LIME local explanation
- Integrated Gradients for the CNN branch
- Weighted branch-level hybrid attribution
- Clean-vs-adversarial explanation stability using top-k Jaccard overlap
- Secondary multiclass Random Forest study for sufficiently represented classes
- Reproducibility metadata and saved configuration
- Streamlit analyst dashboard

## Dataset setup

The real CICIDS2017 MachineLearningCSV ZIP is **not included** because it is large.

Place it here:

```text
data/raw/MachineLearningCSV.zip
```

or extract the CSV files under `data/raw/`.

The pipeline automatically extracts the ZIP if no CSV files are already present.

The placeholder `data/raw/PUT_MACHINELEARNINGCSV_ZIP_HERE.txt` documents this requirement.

## Recommended environment

Python 3.10 or 3.11 is recommended.

### macOS/Linux

```bash
cd STAAR_IDS_Review2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp /path/to/MachineLearningCSV.zip data/raw/MachineLearningCSV.zip
python run_project.py
```

### Windows PowerShell

```powershell
cd STAAR_IDS_Review2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item C:\path\to\MachineLearningCSV.zip data\raw\MachineLearningCSV.zip
python run_project.py
```

## Practical laptop run

The default settings are designed to be feasible on a reasonably capable laptop while still producing the research outputs.

For a faster development/smoke run:

```bash
python run_project.py --max-per-class 5000 --max-adv-samples 1000 --max-adv-train 2000 --max-xai-samples 50 --epochs 3 --adv-epochs 3 --rf-estimators 100 --skip-multiclass --skip-daywise
```

For a stronger final run, increase `--max-per-class`, `--epochs`, and `--rf-estimators` according to available RAM/CPU/GPU. Do not change settings after final results are selected without recording the new configuration.

## Main outputs

After a successful run, inspect `results/`:

### Clean performance

- `model_comparison.csv`
- `model_comparison_fixed_0.5.csv`
- `binary_classification_report.csv`
- `ablation_study.csv`
- `confusion_matrix_rf.png`
- `confusion_matrix_cnn.png`
- `confusion_matrix_hybrid.png`
- `roc_curve.png`
- `pr_curve.png`

### Robustness

- `robustness_comparison.csv`
- `robustness_comparison.png`
- `attack_strength_sweep.csv`

### Generalization

- `daywise_model_comparison.csv`
- `daywise_error.txt` if the selected source-file holdout cannot be evaluated

### Threshold/fusion analysis

- `fusion_weight_validation.csv`
- `hybrid_threshold_sweep_validation.csv`
- `hybrid_threshold_tradeoff.png`

### Explainability

- `shap_summary.png`
- `top_features.csv`
- `lime_explanation.html`
- `lime_local_features.csv`
- `hybrid_feature_attributions.csv`
- `hybrid_feature_attribution.png`
- `xai_fgsm/xai_stability_summary.csv`
- `xai_pgd/xai_stability_summary.csv`

### Secondary multiclass study

- `multiclass_classification_report.csv`
- `multiclass_predictions.csv`

### Reproducibility

- `run_summary.json`
- `run_summary.txt`
- `original_class_distribution.csv`
- `efficiency_config.json`

## Important research interpretation

The project does **not** hard-code or fabricate the results from the Review-2 presentation. Every metric in `results/` is generated from the local CICIDS2017 files and the recorded settings.

The primary binary task is BENIGN vs ATTACK. The multiclass study is secondary and only retains classes meeting `--min-multiclass-samples`.

The adversarial attacks are gradient-based attacks against the differentiable CNN branch. For the hybrid model, these attacks are therefore reported as **CNN-branch-targeted / transfer attacks**, not as a full white-box attack against the non-differentiable Random Forest.

The adaptive threshold and RF/CNN fusion weight are selected on the validation set. The final test set is reserved for reporting.

The day-wise experiment holds out the latest-looking source CSV according to filename ordering. The selected file(s) are recorded in `daywise_model_comparison.csv`. For a paper, inspect the selected source file and state the exact train/test capture days.

## Dashboard

After training:

```bash
streamlit run dashboard/app.py
```

The dashboard loads the trained RF/CNN models, validation-tuned hybrid configuration, feature list, SHAP/global attribution results and robustness tables. It is a proof-of-concept analyst interface, **not** a live packet sensor or automatic firewall.

## Suggested paper experiment set

1. RF vs 1D-CNN vs RF–CNN clean performance.
2. Ablation of the hybrid/defence/threshold components.
3. Clean vs FGSM vs PGD.
4. Attack-strength sweep across multiple epsilon values.
5. FGSM-trained vs PGD-trained CNN robustness.
6. Random split vs day-wise/source-file generalization.
7. SHAP + LIME + CNN Integrated Gradients.
8. Clean-vs-adversarial explanation stability.
9. Secondary multiclass evaluation.
10. Computational/reproducibility reporting.

## What must still be done before paper submission

The code is prepared for these experiments, but **the actual paper results must be generated from your CICIDS2017 dataset**. Do not copy the Review-2 numbers into the paper unless the final run reproduces them under the documented settings.

For publication, run a final locked configuration, save all CSVs/figures, record the hardware/software versions, and perform an additional multi-seed study (for example seeds 42, 21, 7, 52, 100) for the final headline models.
