import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import time
import torch
import tensorflow as tf
import os
import plotly.graph_objects as go
from huggingface_hub import snapshot_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Depression Severity Benchmarking Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Severity Color Utility Functions
def get_severity_styles(label):
    styles = {
        'severe': {"color": "#FF4B4B", "bg": "#FFEBEB", "emoji": "🔴"},
        'moderate': {"color": "#FFA500", "bg": "#FFF5E6", "emoji": "🔸"},
        'mild': {"color": "#F1C40F", "bg": "#FEF9E7", "emoji": "🟡"},
        'minimum': {"color": "#2ECC71", "bg": "#EAFAF1", "emoji": "🟢"}
    }
    return styles.get(label.lower(), {"color": "#31333F", "bg": "#F0F2F6", "emoji": "📄"})

# ==========================================
# 2. SMART HYBRID REPOSITORY DETECT PIPELINE
# ==========================================
REPO_ID = "atomdev-ibktommy/depression-severity-weights"
DOWNLOAD_DIR = "downloaded_models"

# Check if weights exist directly in your root project folder (Local Development)
if os.path.exists("tfidf_vectorizer.pkl"):
    PATH_PREFIX = ""
    is_local_mode = True
# Check if weights exist in the download directory (Fallback / Post-Cloud Download)
elif os.path.exists(f"{DOWNLOAD_DIR}/tfidf_vectorizer.pkl"):
    PATH_PREFIX = f"{DOWNLOAD_DIR}/"
    is_local_mode = True
else:
    PATH_PREFIX = f"{DOWNLOAD_DIR}/"
    is_local_mode = False

@st.cache_resource
def verify_and_load_assets():
    """Triggers Hugging Face download ONLY if weights are completely missing from the workspace."""
    if not is_local_mode:
        with st.spinner("📦 First-time server boot: Fetching deep learning weights from your Hugging Face Hub..."):
            try:
                snapshot_download(
                    repo_id=REPO_ID,
                    local_dir=DOWNLOAD_DIR,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                st.error(f"Failed to pull files from Hugging Face model hub. Error details: {e}")
                st.stop()

# Run checking loop. Locally, this runs instantly with zero download lag!
verify_and_load_assets()

# ==========================================
# 3. CACHED RESOURCE LOADING (PREVENTS CRASHES)
# ==========================================
@st.cache_resource
def load_all_models():
    # Model 1: Hybrid Components
    tfidf = joblib.load(f'{PATH_PREFIX}tfidf_vectorizer.pkl')
    lr_comp = joblib.load(f'{PATH_PREFIX}hybrid_lr_component.pkl')
    svm_comp = joblib.load(f'{PATH_PREFIX}hybrid_svm_model.pkl')

    # Model 2: GRU Components
    gru_model = load_model(f'{PATH_PREFIX}gru_optimized_model.h5')
    with open(f'{PATH_PREFIX}tokenizer_gru.pkl', 'rb') as f:
        gru_tokenizer = pickle.load(f)
    with open(f'{PATH_PREFIX}label_encoder_gru.pkl', 'rb') as f:
        gru_le = pickle.load(f)

    # Model 3: DistilBERT Components
    # Check if the tokenizer files are in the root download directory or the subfolder
    if os.path.exists(os.path.join(PATH_PREFIX, "tokenizer_config.json")):
        tokenizer_path = PATH_PREFIX
        model_path = PATH_PREFIX
    else:
        tokenizer_path = f"{PATH_PREFIX}distilbert_saved_model"
        model_path = f"{PATH_PREFIX}distilbert_saved_model"

    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model.to(device)

    return tfidf, lr_comp, svm_comp, gru_model, gru_tokenizer, gru_le, bert_tokenizer, bert_model, device


try:
    tfidf, lr_comp, svm_comp, gru_model, gru_tokenizer, gru_le, bert_tokenizer, bert_model, device = load_all_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error initializing cached model components: {e}")


# Test Dataset Explorer loader hook
@st.cache_data
def load_mock_test_set():
    return pd.read_csv('app_test_samples.csv')


test_df = load_mock_test_set()

# ==========================================
# 4. SIDEBAR EXPERIMENTAL LOGS
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 Academic Performance Matrix")
    st.markdown("Background metrics locked in from the empirical holdout evaluations.")
    st.markdown("---")

    # Model 1 Metrics
    st.markdown("#### 🛠️ Model 1: True Hybrid Baseline")
    st.caption("Architecture: TF-IDF + Logistic Regression Probabilities + Linear SVM")
    st.metric(label="Overall Accuracy", value="65.57%", delta="Baseline")
    st.metric(label="Macro F1-Score", value="65.84%")
    st.markdown("---")

    # Model 2 Metrics
    st.markdown("#### 🧬 Model 2: Optimized GRU")
    st.caption("Architecture: Pre-padded Token Sequences + Embedding + GRU Layer")
    st.metric(label="Overall Accuracy", value="69.65%", delta="+4.08% vs M1")
    st.metric(label="Macro F1-Score", value="69.43%")
    st.markdown("---")

    # Model 3 Metrics
    st.markdown("#### 🚀 Model 3: DistilBERT Transformer")
    st.caption("Architecture: Multi-Head Bidirectional Self-Attention Fine-Tuning")
    st.metric(label="Overall Accuracy", value="73.46%", delta="+7.89% vs M1")
    st.metric(label="Macro F1-Score", value="74.38%")

# ==========================================
# 5. APP BRANDING HEADER
# ==========================================
st.title("🧠 Multi-Model Depression Severity Benchmarking Dashboard")
st.markdown("""
This interactive platform runs real-time inference across three distinct natural language processing architectures 
developed for classification tasks. It evaluates static token associations, sequential pipelines, and transformer context arrays.
""")
st.markdown("---")

# ==========================================
# 6. INPUT SELECTION TABS
# ==========================================
tab1, tab2 = st.tabs(["📂 Test Set Explorer Mode", "✍️ Manual Text Inference Mode"])

input_text = ""
selected_true_label = None

with tab1:
    st.subheader("Select Sample from Empirical Test Split")
    sample_index = st.selectbox("Choose a test sample row index:", options=range(len(test_df)))

    selected_row = test_df.iloc[sample_index]
    input_text = selected_row['text']
    selected_true_label = selected_row['true_label']

    styles = get_severity_styles(selected_true_label)

    st.markdown("#### Ground Truth Classification")
    st.markdown(
        f"<div style='padding:10px 20px; border-radius:5px; background-color:{styles['bg']}; "
        f"color:{styles['color']}; font-weight:bold; display:inline-block; font-size:18px; "
        f"border: 2px solid {styles['color']};'>"
        f"True Status: {selected_true_label.upper()}</div>",
        unsafe_allow_html=True
    )
    st.info(f"**Sample Text String:**\n\n\"{input_text}\"")

with tab2:
    st.subheader("Interactive Text Entry")
    manual_text = st.text_area(
        "Paste custom evaluation text below:",
        placeholder="Type clinical expressions or narrative transcripts here to assess model predictions..."
    )
    if manual_text:
        input_text = manual_text
        selected_true_label = None

# ==========================================
# 7. FINAL PRODUCTION PREDICTION ENGINE (Corrected Layout Level)
# ==========================================
if models_loaded:
    if st.button("🔥 Run Parallel Comparative Inference", type="primary"):
        if not input_text.strip():
            st.warning("⚠️ Please select a test sample or type some text first before running inference!")
        else:
            # --- MODEL 1: HYBRID BASELINE INFERENCE ---
            X_tfidf = tfidf.transform([input_text])
            m1_probs = lr_comp.predict_proba(X_tfidf)[0]

            import scipy.sparse as sp
            X_hybrid = sp.hstack((X_tfidf, lr_comp.predict_proba(X_tfidf)), format='csr')
            m1_pred = svm_comp.predict(X_hybrid)[0]  # Outputs string directly ('mild', etc.)

            # --- MODEL 2: OPTIMIZED GRU INFERENCE ---
            gru_seq = gru_tokenizer.texts_to_sequences([input_text])
            gru_padded = pad_sequences(gru_seq, maxlen=256, padding='pre', truncating='pre')
            m2_probs = gru_model.predict(gru_padded)[0]
            m2_pred_idx = np.argmax(m2_probs)
            m2_pred = gru_le.inverse_transform([m2_pred_idx])[0]

            # --- MODEL 3: DISTILBERT INFERENCE ---
            inputs = bert_tokenizer(input_text, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            m3_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            m3_pred_idx = np.argmax(m3_probs)

            # 🎯 PERFECT REALIGNED TRANSFORMER KEYMAP
            transformer_classes = ['minimum', 'mild', 'moderate', 'severe']
            m3_pred = transformer_classes[m3_pred_idx]

            # ==========================================
            # 8. RENDERING INFERENCE OUTPUT MATRIX
            # ==========================================
            st.markdown("### 🎯 Parallel Inference Decision Matrix")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Model 1")
                st.markdown("**True Hybrid Baseline**")
                styles_m1 = get_severity_styles(m1_pred)
                st.markdown(
                    f"<div style='padding:15px; border-radius:5px; background-color:{styles_m1['bg']}; "
                    f"color:{styles_m1['color']}; font-weight:bold; font-size:22px; text-align:center;"
                    f"border: 1px solid {styles_m1['color']};'>{m1_pred.upper()}</div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("### Model 2")
                st.markdown("**Optimized GRU**")
                styles_m2 = get_severity_styles(m2_pred)
                st.markdown(
                    f"<div style='padding:15px; border-radius:5px; background-color:{styles_m2['bg']}; "
                    f"color:{styles_m2['color']}; font-weight:bold; font-size:22px; text-align:center;"
                    f"border: 1px solid {styles_m2['color']};'>{m2_pred.upper()}</div>",
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown("### Model 3")
                st.markdown("**Fine-Tuned DistilBERT**")
                styles_m3 = get_severity_styles(m3_pred)
                st.markdown(
                    f"<div style='padding:15px; border-radius:5px; background-color:{styles_m3['bg']}; "
                    f"color:{styles_m3['color']}; font-weight:bold; font-size:22px; text-align:center;"
                    f"border: 1px solid {styles_m3['color']};'>{m3_pred.upper()}</div>",
                    unsafe_allow_html=True
                )

            # ==========================================
            # 9. VISUAL DISTRIBUTION COMPARISON CHART
            # ==========================================
            st.markdown("### 📊 Distribution Probability Across Architectural Models")

            categories = ['Mild', 'Minimum', 'Moderate', 'Severe']
            fig = go.Figure()

            # Model 1 Map (Alphabetical order)
            fig.add_trace(go.Bar(x=categories, y=[m1_probs[0], m1_probs[1], m1_probs[2], m1_probs[3]],
                                 name='Model 1: Hybrid Base', marker_color='#94A3B8'))

            # Model 2 Map (Alphabetical order)
            fig.add_trace(go.Bar(x=categories, y=[m2_probs[0], m2_probs[1], m2_probs[2], m2_probs[3]],
                                 name='Model 2: Optimized GRU', marker_color='#2ECC71'))

            # Model 3 Transformer Map Realignment for chart consistency
            fig.add_trace(go.Bar(x=categories, y=[m3_probs[1], m3_probs[0], m3_probs[2], m3_probs[3]],
                                 name='Model 3: DistilBERT', marker_color='#8B5CF6'))

            fig.update_layout(
                barmode='group',
                xaxis_title="Predicted Severity Class Layers",
                yaxis_title="Confidence Probability Matrix",
                legend_title="Architectures",
                template="plotly_white",
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Awaiting deployment export arrays to begin processing.")