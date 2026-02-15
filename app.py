
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ---------- Page & menu: ----------
st.set_page_config(
    page_title="Adult Income Classification",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

# (button & element toolbars)
st.markdown("""
<style>
  .stAppDeployButton { display: none; }  /* newer versions */
  .stDeployButton    { display: none; }  /* older builds  */
  [data-testid="stElementToolbar"] { display: none; }  /* per-element toolbars */
</style>
""", unsafe_allow_html=True)

st.title("Adult Income Classification")
st.write("Upload test CSV → choose a model → view evaluation metrics and confusion matrix / classification report.")

# ---------- Paths & schema ----------
ARTIFACTS_DIR = "model/artifacts"
REQUIRED_NUMERIC = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
REQUIRED_CATEG  = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
TARGET = "income"
EXPECTED_FEATURES = REQUIRED_NUMERIC + REQUIRED_CATEG

# ---------- Load shared artifacts ----------
try:
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
except Exception:
    st.error("Artifacts missing. Please run the training script to create files under model/artifacts/.")
    st.stop()

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "kNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "Random Forest":       "random_forest.joblib",
    "XGBoost":             "xgboost.joblib",
}

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Select model", list(MODELS.keys()))
try:
    clf = joblib.load(os.path.join(ARTIFACTS_DIR, MODELS[model_name]))
except Exception:
    st.error(f"Model file for '{model_name}' not found in {ARTIFACTS_DIR}. Train first, then retry.")
    st.stop()

# ---------- Helpers ----------
RENAME_MAP = {
    "education.num": "education-num",
    "marital.status": "marital-status",
    "native.country": "native-country",
    "capital.gain": "capital-gain",
    "capital.loss": "capital-loss",
    "hours.per.week": "hours-per-week",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Rename dotted headers to dashed equivalents
    to_rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
    # Strip whitespace on string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

def validate_required(df: pd.DataFrame):
    return [c for c in EXPECTED_FEATURES if c not in df.columns]

# ---------- Required Feature #1: CSV Upload ----------
st.subheader("1) Upload Test CSV")
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded:
    st.info("Upload test CSV to proceed.")
    st.stop()

# ---------- Load & validate ----------
try:
    df_in = pd.read_csv(uploaded)
    df_in = normalize_columns(df_in)

    # quick preview without toolbar
    st.write("Preview (first 5 rows):")
    st.table(df_in.head(5))

    missing = validate_required(df_in)
    if missing:
        st.error(
            "Your CSV is missing required columns: " + ", ".join(missing) +
            ". If your file uses dotted names (e.g., marital.status), rename to dashed (marital-status)."
        )
        st.stop()

    has_target = TARGET in df_in.columns
    X = df_in.drop(columns=[TARGET]) if has_target else df_in

except Exception as e:
    st.error(
        "Could not process the uploaded CSV. Verify headers & data types match the Adult dataset schema. "
        f"(Detail: {type(e).__name__}: {e})"
    )
    st.stop()

# ---------- Inference ----------
y_pred = clf.predict(X)
y_prob = None
if hasattr(clf, "predict_proba"):
    try:
        y_prob = clf.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None

# ---------- Required Feature #3: Evaluation Metrics----------
if has_target:
    st.subheader("2) Evaluation Metrics")
    y_true = label_encoder.transform(df_in[TARGET])

    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       (roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) == 2) else None),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1 Score":  f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }
    st.table(pd.DataFrame(metrics, index=["Value"]).T)

    # ---------- Required Feature #4: Confusion Matrix / Classification Report ----------
    st.subheader("3) Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.table(pd.DataFrame(
        cm,
        index=["True <=50K", "True >50K"],
        columns=["Pred <=50K", "Pred >50K"]
    ))

    st.subheader("Classification Report")
    st.code(classification_report(y_true, y_pred, target_names=label_encoder.classes_), language="text")
else:
    # If target not present, we do not show metrics/CM (not required by brief)
    st.info("Ground truth column 'income' not found in uploaded file. Metrics and confusion matrix require ground truth.")
