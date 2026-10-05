from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import CNN1D, cnn_proba
from src.xai import natural_language_explanation

st.set_page_config(page_title='STAAR-IDS', layout='wide')
st.title('STAAR-IDS — Explainable & Adversarially Robust IDS')
st.caption('CICIDS2017 | RF + 1D-CNN hybrid | adversarial evaluation | SHAP/LIME')

models = ROOT / 'models'; results = ROOT / 'results'; processed = ROOT / 'data' / 'processed'
prep_path = models / 'preprocessor.joblib'
rf_path = models / 'random_forest.joblib'; cnn_path = models / 'cnn.pt'
if not (prep_path.exists() and rf_path.exists() and cnn_path.exists()):
    st.warning('Train the project first with: python run_pipeline.py')
    st.stop()

prep = joblib.load(prep_path); scaler = prep['scaler']; features = prep['features']
rf = joblib.load(rf_path)
config = joblib.load(models / 'hybrid_config.joblib') if (models / 'hybrid_config.joblib').exists() else {'rf_weight': .5, 'threshold': .5}
ckpt = torch.load(cnn_path, map_location='cpu', weights_only=False)
cnn = CNN1D(ckpt['n_features'], ckpt['hidden']); cnn.load_state_dict(ckpt['state_dict']); cnn.eval()

st.sidebar.header('Input')
mode = st.sidebar.selectbox('Input mode', ['Use first test row', 'Manual feature vector'])
if mode == 'Use first test row' and (processed / 'sampled_binary.csv').exists():
    raw = pd.read_csv(processed / 'sampled_binary.csv', nrows=1)
    label = [c for c in raw.columns if c.lower() == 'label'][0]
    row = raw.drop(columns=[label], errors='ignore').select_dtypes('number').iloc[0].reindex(features).fillna(0).to_numpy(float)
else:
    vals = [st.sidebar.number_input(f, value=0.0, format='%.6g') for f in features]
    row = np.array(vals, dtype=float)

X = scaler.transform(row.reshape(1, -1)).astype('float32')
rf_p = float(rf.predict_proba(X)[:, 1][0]); cnn_p = float(cnn_proba(cnn, X)[0])
rf_w = float(config.get('rf_weight', .5)); threshold = float(config.get('threshold', .5))
hybrid_p = rf_w * rf_p + (1-rf_w) * cnn_p
pred = int(hybrid_p >= threshold)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Prediction', 'ATTACK' if pred else 'BENIGN')
c2.metric('Hybrid attack probability', f'{hybrid_p*100:.2f}%')
c3.metric('Decision threshold', f'{threshold:.3f}')
c4.metric('RF / CNN probability', f'{rf_p:.3f} / {cnn_p:.3f}')

st.info(f'Validation-tuned hybrid fusion: RF weight={rf_w:.2f}, CNN weight={1-rf_w:.2f}. The threshold is selected using validation data, not the final test set.')
st.divider()

st.subheader('Explainability')
fi_path = results / 'hybrid_feature_attributions.csv'
if fi_path.exists():
    fi = pd.read_csv(fi_path)
    st.write('Top features in the weighted hybrid attribution analysis:')
    st.dataframe(fi.head(15), use_container_width=True)
else:
    fi = pd.DataFrame({'feature': features, 'rf_mean_abs_shap': rf.feature_importances_}).sort_values('rf_mean_abs_shap', ascending=False)
    st.dataframe(fi.head(15), use_container_width=True)

if (results / 'shap_summary.png').exists():
    st.image(str(results / 'shap_summary.png'), caption='Global SHAP importance — Random Forest branch')
if (results / 'lime_explanation.html').exists():
    st.caption('A local LIME explanation is generated as results/lime_explanation.html and can be opened in a browser.')

st.subheader('Robustness evidence')
rob_path = results / 'robustness_comparison.csv'
if rob_path.exists():
    rob = pd.read_csv(rob_path)
    st.dataframe(rob[['model','condition','accuracy','precision','recall','f1','fpr','fnr']].round(4), use_container_width=True)
if (results / 'robustness_comparison.png').exists():
    st.image(str(results / 'robustness_comparison.png'))

st.subheader('Research outputs')
for name in ['model_comparison.csv','daywise_model_comparison.csv','attack_strength_sweep.csv','hybrid_threshold_sweep_validation.csv','xai_stability_summary.csv','multiclass_classification_report.csv']:
    path = results / name
    if path.exists(): st.write(f'✓ {name}')

st.caption('Proof-of-concept analyst interface. This application does not capture live packets, block traffic, or provide production network enforcement.')
