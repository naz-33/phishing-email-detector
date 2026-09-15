# Phishing Detection App

## Overview

Streamlit web application comparing Baseline (Logistic Regression) vs Hybrid (LR + LightGBM) phishing detection models.

## Folder Structure

Zander Etal/
├── models/
│ ├── hybrid/
│ │ ├── char_vectorizer.joblib
│ │ ├── config.pkl
│ │ ├── lightgbm_model.joblib
│ │ ├── logistic_regression_model.joblib
│ │ ├── scaler.joblib
│ │ └── word_vectorizer.joblib
│ └── baseline/
│ ├── logistic_regression_model.joblib
│ ├── model_config.json
│ └── tfidf_vectorizer.joblib
├── utils/
│ └── preprocess.py
├── app.py
├── requirements.txt
└── README.md

text

## Installation

Install the dependencies and start the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```
