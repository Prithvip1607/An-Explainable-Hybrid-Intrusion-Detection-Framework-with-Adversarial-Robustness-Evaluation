# Interim Review-2 — What to demonstrate

## 1. Design Document Preparation — 5 marks
- Proposed architecture: CICIDS2017 → preprocessing → RF + 1D CNN → hybrid → robustness → XAI → dashboard.
- Explain every module and why it exists.
- Show data-flow diagram and technology stack.

## 2. Core Modules / Testbed / Platform — 10 marks
- Dataset: CICIDS2017 MachineLearningCSV.
- Preprocessing: NaN/Infinity handling, numeric features, duplicate removal, scaling.
- RF baseline.
- 1D CNN.
- RF + CNN soft voting.
- FGSM and PGD evaluation.
- Adversarial training.
- SHAP/LIME explainability.
- Streamlit analyst dashboard.
- Testbed: Python, scikit-learn, PyTorch, SHAP, LIME, Streamlit.

## 3. Compliance with Review-1 Suggestions — 5 marks

| Review-1 point | Review-2 evidence |
|---|---|
| Suitable dataset | CICIDS2017 selected and acquired |
| ML/DL investigation | RF + 1D CNN implementation |
| Explainability | SHAP + LIME module |
| Adversarial robustness | FGSM + PGD + adversarial training |
| Large tabular data | Memory-conscious per-class sampling |
| Better metrics | F1, FPR, FNR, ROC-AUC, PR-AUC in addition to accuracy |

## 4. Task Completion — 5 marks

Use `results/run_summary.txt`, `results/model_comparison.csv`, `results/robustness_comparison.csv`, plots, and the dashboard as evidence.

### Do not claim
- Do not claim the old synthetic baseline numbers as CICIDS2017 results.
- Do not claim production/live IDS deployment.
- Do not claim rare CICIDS2017 attack classes are reliably classified unless you have sufficient test samples and evidence.
