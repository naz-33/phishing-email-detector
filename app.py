"""
Phishing Detection App - Streamlit
Compare Replication (Logistic Regression) vs Hybrid (LR + LightGBM) Models
"""

import os
import sys
import json
import joblib
import pickle
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import preprocessing function
from utils.preprocess import (
    preprocess_email_baseline as preprocess_email_replication,
    preprocess_email_hybrid,
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Phishing Email Detection System",
    layout="wide"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0.5rem;
    }
    .header-copy {
        color: #5f6368;
        font-size: 1rem;
        margin: 0.8rem 0 1.8rem;
    }
    .model-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .model-tag {
        background: #eef2ff;
        border: 1px solid #c5cae9;
        border-radius: 999px;
        color: #283593;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.35rem 0.8rem;
    }
    .section-label {
        color: #37474f;
        font-size: 1.1rem;
        font-weight: 650;
        margin: 0 0 0.65rem;
    }
    .model-comparison-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 148px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
    }
    div[data-testid="stHorizontalBlock"]:has(.model-comparison-card) {
        align-items: stretch;
        margin-bottom: 1.5rem;
    }
    .model-comparison-card.proposed {
        background: #f8fbff;
        border: 1px solid #4a90c2;
        box-shadow: 0 3px 8px rgba(31, 119, 180, 0.18);
    }
    .model-card-header {
        align-items: flex-start;
        display: flex;
        justify-content: space-between;
        min-height: 5rem;
    }
    .model-card-name {
        color: #1f2937;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .model-card-type {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }
    .model-card-stats {
        border-top: 1px solid #e2e8f0;
        display: grid;
        gap: 1rem;
        grid-template-columns: 1fr 1fr;
        margin-top: 1rem;
        padding-top: 0.85rem;
    }
    .model-card-stat-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .model-card-stat-value {
        color: #1f2937;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .results-placeholder {
        background: #f8fafc;
        border: 1px dashed #b0bec5;
        border-radius: 10px;
        color: #607d8b;
        padding: 2rem;
        text-align: center;
    }
    .badge-phishing {
        background-color: #c62828;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-legitimate {
        background-color: #2e7d32;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTextArea textarea {
        font-size: 14px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING (with caching)
# ==============================================================================
@st.cache_resource
def load_models():
    """Load both replication and hybrid models with caching."""
    
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to models (using your folder structure: Zander Etal)
    REPLICATION_PATH = os.path.join(BASE_DIR, "models", "baseline")
    HYBRID_PATH = os.path.join(BASE_DIR, "models", "hybrid")
    
    models = {
        "replication": {"loaded": False},
        "hybrid": {"loaded": False}
    }
    
    # ---- Load Replication Model ----
    try:
        with st.spinner("Loading Replication Model..."):
            # Load model and vectorizer
            models["replication"]["model"] = joblib.load(
                os.path.join(REPLICATION_PATH, "logistic_regression_model.joblib")
            )
            models["replication"]["tfidf"] = joblib.load(
                os.path.join(REPLICATION_PATH, "tfidf_vectorizer.joblib")
            )
            # Load config from model_config.json
            config_path = os.path.join(REPLICATION_PATH, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    models["replication"]["config"] = json.load(f)
            else:
                # Fallback config
                models["replication"]["config"] = {
                    "performance_metrics": {
                        "accuracy": 0.9579,
                        "precision": 0.9528,
                        "recall": 0.9437,
                        "f1_score": 0.9482
                    },
                    "dataset_info": {
                        "training_samples": "N/A",
                        "vocabulary_size": "N/A"
                    }
                }
            models["replication"]["loaded"] = True
    except Exception as e:
        models["replication"]["error"] = str(e)
        st.error(f"Failed to load Replication Model: {e}")
    
    # ---- Load Hybrid Model ----
    try:
        with st.spinner("Loading Hybrid Model..."):
            # Load models
            models["hybrid"]["lr_model"] = joblib.load(
                os.path.join(HYBRID_PATH, "logistic_regression_model.joblib")
            )
            models["hybrid"]["lgbm_model"] = joblib.load(
                os.path.join(HYBRID_PATH, "lightgbm_model.joblib")
            )
            # Load vectorizers
            models["hybrid"]["word_vec"] = joblib.load(
                os.path.join(HYBRID_PATH, "word_vectorizer.joblib")
            )
            models["hybrid"]["char_vec"] = joblib.load(
                os.path.join(HYBRID_PATH, "char_vectorizer.joblib")
            )
            models["hybrid"]["scaler"] = joblib.load(
                os.path.join(HYBRID_PATH, "scaler.joblib")
            )
            
            # Load config from config.pkl
            config_path = os.path.join(HYBRID_PATH, "config.pkl")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
                models["hybrid"]["config"] = config
            else:
                # Fallback config
                models["hybrid"]["config"] = {
                    "ensemble_weights": {
                        "logistic_regression": 0.60,
                        "lightgbm": 0.40
                    },
                    "threshold": 0.50,
                    "performance_metrics": {
                        "accuracy": 0.9612,
                        "precision": 0.9584,
                        "recall": 0.9512,
                        "f1_score": 0.9548
                    }
                }
            models["hybrid"]["loaded"] = True
    except Exception as e:
        models["hybrid"]["error"] = str(e)
        st.error(f"Failed to load Hybrid Model: {e}")
    
    return models

def compute_prediction_confidence(phishing_probability, prediction):
    """Return confidence in the predicted class for binary classification."""
    if prediction == 1:
        return float(phishing_probability)
    return float(1.0 - phishing_probability)


def predict_replication(text, model, tfidf):
    """Make prediction using replication model."""
    clean_text = preprocess_email_replication(text)
    X_tfidf = tfidf.transform([clean_text])
    proba = model.predict_proba(X_tfidf)[0][1]
    pred = 1 if proba >= 0.5 else 0
    confidence = compute_prediction_confidence(proba, pred)
    
    return {
        "prediction": pred,
        "label": "Phishing" if pred == 1 else "Legitimate",
        "phishing_probability": proba,
        "confidence": confidence,
        "clean_text": clean_text
    }

def predict_hybrid(text, models):
    """Make prediction using hybrid model ensemble."""
    clean_text = preprocess_email_hybrid(text)
    
    # Stylometric features
    stylo_features = extract_stylometrics(text)
    if stylo_features.size == 0:
        raise ValueError("Stylometric features are empty")

    feature_labels = [
        "char_count",
        "word_count",
        "upper_ratio",
        "exclamation_count",
        "question_count",
        "digit_ratio",
        "url_count",
        "urgency_ratio",
        "entropy",
    ]
    print("[DEBUG] Raw stylometric features:")
    for label, value in zip(feature_labels, stylo_features):
        print(f"[DEBUG]   {label}: {value}")

    urgency_words = {
        "urgent", "immediately", "verify", "suspended", "click",
        "password", "security", "alert", "expire", "action",
        "required", "login", "bank", "confirm", "final", "notice"
    }
    words = str(text).split()
    urgency_words_found = [
        word for word in words if word.lower() in urgency_words
    ]
    print(f"[DEBUG] Urgency words found: {urgency_words_found}")
    print(
        f"[DEBUG] Urgency words count: {len(urgency_words_found)} "
        f"/ {len(words)} total words"
    )

    raw_urgency_ratio = stylo_features[7]
    if raw_urgency_ratio > 1.0 or raw_urgency_ratio < 0.0:
        print(
            f"[WARNING] Raw urgency_ratio is outside [0, 1]: "
            f"{raw_urgency_ratio}"
        )
    stylo_scaled = models["scaler"].transform([stylo_features])
    scaled_values = (
        stylo_scaled.toarray().ravel().tolist()
        if hasattr(stylo_scaled, "toarray")
        else np.asarray(stylo_scaled).ravel().tolist()
    )
    print(f"[DEBUG] Scaled stylometric features: {scaled_values}")
    print(f"[DEBUG] Scaled stylometric feature shape: {stylo_scaled.shape}")
    print(f"[DEBUG] Scaled urgency_ratio: {scaled_values[7]}")
    
    # TF-IDF features
    word_features = models["word_vec"].transform([clean_text])
    char_features = models["char_vec"].transform([clean_text])
    if word_features.shape[1] == 0:
        raise ValueError("Word features are empty")
    if char_features.shape[1] == 0:
        raise ValueError("Character features are empty")
    
    # Fuse features
    from scipy.sparse import hstack
    X_fused = hstack([word_features, char_features, stylo_scaled])
    print(f"[DEBUG] Fused matrix shape (word + char + stylometric): {X_fused.shape}")
    print(f"[DEBUG] Total fused features: {X_fused.shape[1]}")
    
    # Get predictions
    lr_proba = models["lr_model"].predict_proba(X_fused)[0][1]
    lgbm_proba = models["lgbm_model"].predict_proba(X_fused)[0][1]

    expected_feature_count = (
        word_features.shape[1] + char_features.shape[1] + stylo_features.size
    )
    model_feature_counts = {
        "Logistic Regression": getattr(models["lr_model"], "n_features_in_", None),
        "LightGBM": getattr(models["lgbm_model"], "n_features_in_", None),
    }
    print(f"[DEBUG] Runtime feature count: {expected_feature_count}")
    print(f"[DEBUG] Saved model feature counts: {model_feature_counts}")
    config_feature_count = models["config"].get("feature_count")
    print(f"[DEBUG] Config feature count: {config_feature_count}")
    for model_name, feature_count in model_feature_counts.items():
        if feature_count is not None and feature_count != expected_feature_count:
            print(
                f"[WARNING] {model_name} expects {feature_count} features, "
                f"but inference produced {expected_feature_count}"
            )

    feature_names = (
        list(models["word_vec"].get_feature_names_out())
        + list(models["char_vec"].get_feature_names_out())
        + [
            "char_count",
            "word_count",
            "upper_ratio",
            "exclamation_count",
            "question_count",
            "digit_ratio",
            "url_count",
            "urgency_ratio",
            "entropy",
        ]
    )
    feature_importances = np.asarray(models["lgbm_model"].feature_importances_)
    if feature_importances.size != expected_feature_count:
        print(
            f"[WARNING] LightGBM returned {feature_importances.size} importances, "
            f"but inference produced {expected_feature_count} features"
        )
    else:
        top_indices = np.argsort(feature_importances)[::-1][:20]
        print("[DEBUG] LightGBM top 20 feature importances:")
        for index in top_indices:
            print(
                f"[DEBUG]   {index}: {feature_names[index]} = "
                f"{feature_importances[index]}"
            )
    
    # Ensemble weights from config
    config = models["config"]
    ensemble_weights = config.get("ensemble_weights", {})
    lr_weight = config.get(
        "ensemble_weight_lr",
        ensemble_weights.get("logistic_regression", 0.60)
    )
    lgbm_weight = config.get(
        "ensemble_weight_lgbm",
        ensemble_weights.get("lightgbm", 0.40)
    )
    threshold = config.get("threshold", 0.50)
    print(
        f"[DEBUG] Ensemble weights: logistic_regression={lr_weight}, "
        f"lightgbm={lgbm_weight}, threshold={threshold}"
    )
    
    final_proba = (lr_weight * lr_proba) + (lgbm_weight * lgbm_proba)
    pred = 1 if final_proba >= threshold else 0
    confidence = compute_prediction_confidence(final_proba, pred)
    
    return {
        "prediction": pred,
        "label": "Phishing" if pred == 1 else "Legitimate",
        "confidence": confidence,
        "final_proba": final_proba,
        "lr_proba": lr_proba,
        "lgbm_proba": lgbm_proba,
        "lr_weight": lr_weight,
        "lgbm_weight": lgbm_weight,
        "clean_text": clean_text
    }

def extract_stylometrics(text):
    """Extract 9 stylometric features from raw text."""
    import math
    import re
    
    text_str = str(text)
    words = text_str.split()
    word_count = len(words) + 1e-5
    char_count = len(text_str) + 1e-5
    
    # Entropy
    prob = [float(text_str.count(c)) / char_count for c in set(text_str)]
    entropy = -sum([p * math.log(p, 2) for p in prob])
    
    # Urgency words
    urgency_words = {
        'urgent', 'immediately', 'verify', 'suspended', 'click',
        'password', 'security', 'alert', 'expire', 'action',
        'required', 'login', 'bank', 'confirm', 'final', 'notice'
    }
    
    features = [
        char_count,
        word_count,
        sum(1 for c in text_str if c.isupper()) / char_count,
        text_str.count('!'),
        text_str.count('?'),
        sum(1 for c in text_str if c.isdigit()) / char_count,
        len(re.findall(r'https?://\S+|www\.\S+', text_str)),
        sum(1 for w in words if w.lower() in urgency_words) / word_count,
        entropy,
    ]
    feature_names = [
        "Character count",
        "Word count",
        "Uppercase character ratio",
        "Exclamation mark count",
        "Question mark count",
        "Digit ratio",
        "URL count",
        "Urgency word ratio",
        "Character entropy",
    ]
    feature_array = np.array(features)
    print(f"[DEBUG] Raw text (first 100 characters): {text_str[:100]!r}")
    print(f"[DEBUG] Number of stylometric features: {len(feature_array)}")
    print(f"[DEBUG] Stylometric feature values: {feature_array.tolist()}")
    print("[DEBUG] Stylometric feature names and values:")
    for name, value in zip(feature_names, feature_array):
        print(f"[DEBUG]   {name}: {value}")
    return feature_array

def create_confidence_gauge(confidence, title, color):
    """Create a confidence gauge using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': "%", 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#e8f5e9"},
                {'range': [30, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
    return fig

def display_prediction_card(result, model_name, color):
    """Display a prediction result card."""
    if result is None:
        return
    
    is_phishing = result['prediction'] == 1
    label_text = "PHISHING" if is_phishing else "LEGITIMATE"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <div style="font-size: 1.2rem; font-weight: 600; color: {color};">
                {model_name}
            </div>
            <div class="{'badge-phishing' if is_phishing else 'badge-legitimate'}" 
                 style="display: inline-block; margin-top: 0.5rem;">
                {label_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_confidence_gauge(
            result['confidence'], 
            "Confidence", 
            color
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 0.5rem;">
            <div style="font-size: 0.9rem; color: #666;">
                <b>Probability:</b> {result['confidence']*100:.2f}%
            </div>
        """, unsafe_allow_html=True)
        
        if 'lr_proba' in result:
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #666;">
                <b>Logistic Regression:</b> {result['lr_proba']*100:.2f}%<br>
                <b>LightGBM:</b> {result['lgbm_proba']*100:.2f}%
            </div>
            """, unsafe_allow_html=True)

        if 'lr_weight' in result:
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                <b>Ensemble weights:</b> Logistic Regression {result['lr_weight']:.2f},
                LightGBM {result['lgbm_weight']:.2f}<br>
                <b>Final probability:</b> {result['final_proba']*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


BATCH_TEXT_COLUMNS = ["email_text", "body", "message", "Email Text", "Message", "text"]
BATCH_LABEL_COLUMNS = ["label", "Category", "Email Type", "class", "target"]
STYLOMETRIC_LABELS = [
    "char_count",
    "word_count",
    "upper_ratio",
    "exclamation_count",
    "question_count",
    "digit_ratio",
    "url_count",
    "urgency_ratio",
    "entropy",
]


def read_batch_csv(uploaded_file):
    """Read an uploaded CSV with UTF-8 first and latin-1 as a fallback."""
    csv_bytes = uploaded_file.getvalue()
    try:
        return pd.read_csv(BytesIO(csv_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(csv_bytes), encoding="latin-1")


def find_column(columns, candidates):
    """Find a candidate column, accepting case differences and whitespace."""
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_batch_label(value):
    """Map common dataset labels to the app's standard class names."""
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "phishing email", "spam", "phishing"}:
        return "Phishing"
    if normalized in {"0", "safe email", "ham", "legitimate", "safe"}:
        return "Legitimate"
    return None


def predict_batch(df, text_column, models, progress_callback=None):
    """Run both available model pipelines for each row in a batch."""
    results = []
    total = len(df)
    for index, raw_text in enumerate(df[text_column].fillna("")):
        email_text = str(raw_text)
        row_result = {
            "email_text": email_text[:200],
            "baseline_pred": "Unavailable",
            "proposed_pred": "Unavailable",
            "baseline_probability": np.nan,
            "hybrid_probability": np.nan,
            "hybrid_confidence": np.nan,
            "discrepancy_flag": False,
        }
        try:
            if models["replication"]["loaded"]:
                baseline = predict_replication(
                    email_text,
                    models["replication"]["model"],
                    models["replication"]["tfidf"],
                )
                row_result["baseline_pred"] = baseline["label"]
                row_result["baseline_probability"] = baseline["phishing_probability"]
            if models["hybrid"]["loaded"]:
                proposed = predict_hybrid(email_text, models["hybrid"])
                row_result["proposed_pred"] = proposed["label"]
                row_result["hybrid_probability"] = proposed["final_proba"]
                row_result["hybrid_confidence"] = proposed["confidence"]
            if (
                row_result["baseline_pred"] != "Unavailable"
                and row_result["proposed_pred"] != "Unavailable"
            ):
                row_result["discrepancy_flag"] = (
                    row_result["baseline_pred"] != row_result["proposed_pred"]
                )
        except Exception as error:
            row_result["error"] = str(error)

        features = extract_stylometrics(email_text)
        row_result.update(dict(zip(STYLOMETRIC_LABELS, features)))
        results.append(row_result)
        if progress_callback:
            progress_callback((index + 1) / max(total, 1))
    return pd.DataFrame(results)


def render_batch_results(source_df, batch_results, label_column):
    """Render batch summary, optional benchmark metrics, table, and export."""
    hybrid_predictions = batch_results["proposed_pred"]
    phishing_count = (hybrid_predictions == "Phishing").sum()
    legitimate_count = (hybrid_predictions == "Legitimate").sum()
    mean_confidence = batch_results["hybrid_confidence"].mean()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Emails Processed", len(batch_results))
    metric_columns[1].metric("Phishing Count", int(phishing_count))
    metric_columns[2].metric("Legitimate Count", int(legitimate_count))
    metric_columns[3].metric(
        "Mean Hybrid Confidence",
        f"{mean_confidence * 100:.1f}%" if not pd.isna(mean_confidence) else "-",
    )

    if label_column:
        evaluation_df = source_df[[label_column]].copy()
        evaluation_df["actual"] = evaluation_df[label_column].map(normalize_batch_label)
        valid_labels = evaluation_df["actual"].notna()
        if valid_labels.any():
            actual = evaluation_df.loc[valid_labels, "actual"]
            metrics = []
            for model_name, prediction_column in (
                ("Baseline", "baseline_pred"),
                ("Proposed", "proposed_pred"),
            ):
                valid_predictions = valid_labels & batch_results[prediction_column].isin(
                    ["Phishing", "Legitimate"]
                )
                if not valid_predictions.any():
                    continue
                actual_predictions = evaluation_df.loc[valid_predictions, "actual"]
                predictions = batch_results.loc[valid_predictions, prediction_column]
                metrics.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(actual_predictions, predictions),
                    "Precision": precision_score(actual_predictions, predictions, pos_label="Phishing", zero_division=0),
                    "Recall": recall_score(actual_predictions, predictions, pos_label="Phishing", zero_division=0),
                    "F1-Score": f1_score(actual_predictions, predictions, pos_label="Phishing", zero_division=0),
                })
            st.subheader("Benchmark / Evaluation")
            if metrics:
                st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"), use_container_width=True)
            else:
                st.warning("No valid model predictions were available for evaluation.")
        else:
            st.warning("The detected label column has no recognized values, so evaluation metrics were skipped.")

    st.subheader("Batch Results")
    display_columns = [
        "email_text", "baseline_pred", "proposed_pred", "hybrid_confidence", "discrepancy_flag"
    ]
    st.dataframe(
        batch_results[display_columns].style.apply(
            lambda row: [
                "background-color: #fff3cd" if row["discrepancy_flag"] else ""
                for _ in row
            ],
            axis=1,
        ).format({"hybrid_confidence": "{:.3f}"}),
        use_container_width=True,
    )

    export_df = source_df.copy()
    for column in batch_results.columns:
        if column == "email_text":
            continue
        export_df[column] = batch_results[column].values
    st.download_button(
        "Download Enriched Results",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="phishing_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ==============================================================================
# SAMPLE EMAILS
# ==============================================================================
SAMPLE_EMAILS = {
    "Phishing - Account Suspended": """
URGENT: Your account has been suspended! 
To reactivate your account, please click the link below and verify your identity immediately.
https://suspicious-link.com/verify
Failure to do so will result in permanent closure of your account.
    """,
    "Phishing - Password Reset": """
Dear user,
Your password will expire in 24 hours. 
Please click here to reset your password now: https://fake-bank.com/reset
If you do not reset your password, you will lose access to your account.
    """,
    "Phishing - Bank Alert": """
ALERT: Unusual activity detected on your account.
Your account has been compromised. Please confirm your credentials to secure your account.
Click here: https://secure-verify.com
    """,
    "Legitimate - Meeting Invite": """
Hi team,
Let's meet tomorrow at 10 AM to discuss the project progress.
We'll cover the Q3 results and Q4 planning.
Best regards,
John
    """,
    "Legitimate - Invoice": """
Dear Customer,
Your invoice #INV-2024-001 is attached.
Payment is due in 30 days.
Thank you for your business.
    """,
    "Legitimate - Newsletter": """
Hello from AI Insights!
Check out our latest blog post about machine learning trends in 2024.
Read more: https://ai-insights.com/blog
Subscribe for more updates.
    """
}

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.markdown('<div class="main-title">Phishing Email Detection System</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="model-tags">
        <span class="model-tag">Baseline: Logistic Regression</span>
        <span class="model-tag">Proposed: LR + LightGBM</span>
    </div>
    <div class="header-copy">Paste an email or choose a sample to compare both phishing detection models.</div>
    ''', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    if not models["replication"]["loaded"] and not models["hybrid"]["loaded"]:
        st.error("Both models failed to load. Please check model files.")
        return
    elif not models["replication"]["loaded"]:
        st.warning("Replication model failed to load. Only hybrid predictions available.")
    elif not models["hybrid"]["loaded"]:
        st.warning("Hybrid model failed to load. Only replication predictions available.")

    render_analysis_page(models)

# ==============================================================================
# PAGE: ANALYSIS
# ==============================================================================
def render_analysis_page(models):
    """Render the main analysis page."""

    def select_sample():
        selected_sample = st.session_state.sample_choice
        st.session_state.email_input = SAMPLE_EMAILS.get(selected_sample, "")

    def clear_email_input():
        st.session_state.email_input = ""
        st.session_state.sample_choice = "Custom Text"

    if "email_input" not in st.session_state:
        st.session_state.email_input = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "batch_source_df" not in st.session_state:
        st.session_state.batch_source_df = None
    if "batch_label_column" not in st.session_state:
        st.session_state.batch_label_column = None

    input_mode = st.radio(
        "Input mode",
        ["Single Email", "Batch CSV Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    input_col, results_col = st.columns([55, 45], gap="large")

    with input_col:
        st.markdown('<div class="section-label">Email to analyze</div>', unsafe_allow_html=True)
        if input_mode == "Single Email":
            st.selectbox(
                "Test sample",
                ["Custom Text"] + list(SAMPLE_EMAILS.keys()),
                key="sample_choice",
                on_change=select_sample,
                label_visibility="collapsed",
            )
            email_text = st.text_area(
                "Email content",
                height=280,
                placeholder="Paste email content here...",
                key="email_input",
                label_visibility="collapsed",
            )
            action_col1, action_col2 = st.columns([1, 1])
            with action_col1:
                analyze_btn = st.button(":material/search: Analyze Email", use_container_width=True, type="primary")
            with action_col2:
                st.button(":material/delete: Clear Input", use_container_width=True, on_click=clear_email_input)
        else:
            sample_csv = pd.DataFrame({
                "email_text": [
                    "URGENT: Verify your account immediately at https://example.com",
                    "Hi team, our meeting is confirmed for tomorrow at 10 AM.",
                    "Your password expires today. Click here to reset it.",
                ],
                "label": ["Phishing Email", "Safe Email", "spam"],
            })
            st.download_button(
                "Download Sample CSV Template",
                data=sample_csv.to_csv(index=False).encode("utf-8"),
                file_name="phishing_email_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV files should contain an email text column and may include a label column.",
            )
            batch_process_btn = st.button(
                ":material/upload: Process CSV",
                use_container_width=True,
                type="primary",
                disabled=uploaded_file is None,
            )

            if uploaded_file is not None:
                try:
                    uploaded_df = read_batch_csv(uploaded_file)
                except Exception as error:
                    st.error(f"Unable to read CSV: {error}")
                    uploaded_df = None

                if uploaded_df is not None:
                    if uploaded_df.empty:
                        st.warning("The uploaded CSV contains no rows to process.")
                    else:
                        detected_text_column = find_column(uploaded_df.columns, BATCH_TEXT_COLUMNS)
                        if detected_text_column is None:
                            st.warning("No standard email text column was detected. Select one manually.")
                            text_column = st.selectbox("Email text column", list(uploaded_df.columns))
                        else:
                            text_column = detected_text_column
                            st.caption(f"Detected email text column: {text_column}")

                        detected_label_column = find_column(uploaded_df.columns, BATCH_LABEL_COLUMNS)
                        if detected_label_column:
                            st.info(f"Benchmark / Evaluation Mode: using '{detected_label_column}' as the label column.")
                        else:
                            st.info("Prediction Only Mode: no recognized label column was found.")

                        if batch_process_btn and text_column:
                            progress_bar = st.progress(0, text="Generating batch predictions...")
                            st.session_state.batch_results = predict_batch(
                                uploaded_df,
                                text_column,
                                models,
                                progress_callback=lambda value: progress_bar.progress(
                                    value, text="Generating batch predictions..."
                                ),
                            )
                            st.session_state.batch_source_df = uploaded_df
                            st.session_state.batch_label_column = detected_label_column
                            progress_bar.empty()

    with results_col:
        st.markdown('<div class="section-label">Live analysis</div>', unsafe_allow_html=True)

        if input_mode == "Batch CSV Upload":
            if st.session_state.batch_results is None:
                st.markdown(
                    '<div class="results-placeholder">Upload a CSV and process it to see batch results.</div>',
                    unsafe_allow_html=True,
                )
            else:
                render_batch_results(
                    st.session_state.batch_source_df,
                    st.session_state.batch_results,
                    st.session_state.batch_label_column,
                )
            return

    if analyze_btn and email_text.strip():
        with st.spinner("Analyzing email..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preprocessing text...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            # Replication prediction
            status_text.text("Running Replication Model...")
            progress_bar.progress(50)
            replication_result = None
            if models["replication"]["loaded"]:
                try:
                    replication_result = predict_replication(
                        email_text,
                        models["replication"]["model"],
                        models["replication"]["tfidf"]
                    )
                except Exception as e:
                    st.error(f"Replication error: {e}")
            
            # Hybrid prediction
            status_text.text("Running Hybrid Model...")
            progress_bar.progress(75)
            hybrid_result = None
            if models["hybrid"]["loaded"]:
                try:
                    hybrid_result = predict_hybrid(email_text, models["hybrid"])
                except Exception as e:
                    st.error(f"Hybrid error: {e}")
            
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.analysis_results = (replication_result, hybrid_result)

    with results_col:
        results = st.session_state.analysis_results
        if not results:
            st.markdown(
                '<div class="results-placeholder">Your verdict and model comparison will appear here after analysis.</div>',
                unsafe_allow_html=True,
            )
            return

        replication_result, hybrid_result = results
        primary_result = hybrid_result or replication_result
        if primary_result:
            banner = st.error if primary_result["prediction"] else st.success
            banner(
                f"**Final Verdict (Proposed): {primary_result['label']}**  "
                f"| Confidence: {primary_result['confidence'] * 100:.1f}%"
            )
            st.progress(
                float(primary_result["confidence"]),
                text="Proposed model detection confidence"
            )

        comparison_col1, comparison_col2 = st.columns(2)
        with comparison_col1:
            replication_label = replication_result["label"] if replication_result else "Unavailable"
            replication_confidence = (
                f"{replication_result['confidence'] * 100:.1f}%"
                if replication_result else "-"
            )
            st.markdown(f"""
            <div class="model-comparison-card">
                <div class="model-card-header">
                    <div>
                        <div class="model-card-name">Baseline</div>
                        <div class="model-card-type">Logistic Regression</div>
                    </div>
                </div>
                <div class="model-card-stats">
                    <div>
                        <div class="model-card-stat-label">Prediction</div>
                        <div class="model-card-stat-value">{replication_label}</div>
                    </div>
                    <div>
                        <div class="model-card-stat-label">Confidence</div>
                        <div class="model-card-stat-value">{replication_confidence}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with comparison_col2:
            hybrid_label = hybrid_result["label"] if hybrid_result else "Unavailable"
            hybrid_confidence = (
                f"{hybrid_result['confidence'] * 100:.1f}%"
                if hybrid_result else "-"
            )
            st.markdown(f"""
            <div class="model-comparison-card proposed">
                <div class="model-card-header">
                    <div>
                        <div class="model-card-name">Proposed</div>
                        <div class="model-card-type">LR + LightGBM Ensemble</div>
                    </div>
                </div>
                <div class="model-card-stats">
                    <div>
                        <div class="model-card-stat-label">Prediction</div>
                        <div class="model-card-stat-value">{hybrid_label}</div>
                    </div>
                    <div>
                        <div class="model-card-stat-label">Confidence</div>
                        <div class="model-card-stat-value">{hybrid_confidence}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if primary_result.get("clean_text"):
            with st.expander("View processed text"):
                st.write(primary_result["clean_text"])

# ==============================================================================
# RUN APP
# ==============================================================================
if __name__ == "__main__":
    main()