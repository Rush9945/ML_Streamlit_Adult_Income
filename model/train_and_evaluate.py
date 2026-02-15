
# model/train_and_evaluate.py
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib

# Adult (Census Income) columns & target
TARGET_COL = "income"
NUMERIC_COLS = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
CATEGORICAL_COLS = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

def _ohe_dense():
    """
    Return an OneHotEncoder instance that produces DENSE output,
    compatible with different scikit-learn versions.
    """
    try:
        
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
 
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_preprocessor(df: pd.DataFrame, dense: bool = False):
    """
    Build a ColumnTransformer. If dense=True, OHE will return dense arrays
    (needed for GaussianNB). Otherwise, OHE returns sparse (default).
    """
    numeric = [c for c in NUMERIC_COLS if c in df.columns]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    if dense:
        ohe = _ohe_dense()
    else:
        # Default sparse OHE (memory efficient for tree/linear models)
        try:
            ohe = OneHotEncoder(handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", ohe, categorical),
        ]
    )

def load_adult_local(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize common name variants from different mirrors
    rename_map = {
        "education.num": "education-num",
        "marital.status": "marital-status",
        "native.country": "native-country",
        "capital.gain": "capital-gain",
        "capital.loss": "capital-loss",
        "hours.per.week": "hours-per-week",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Strip whitespace ONLY on object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # Convert '?' to NaN on object columns and drop rows with missing values
    for col in obj_cols:
        df[col] = df[col].replace({"?": None})
    df.dropna(inplace=True)

    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' not found."
    return df

def compute_metrics(y_true, y_prob, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

def main(args):
    os.makedirs(args.artifacts, exist_ok=True)
    df = load_adult_local(args.csv)

    # Encode target ('<=50K', '>50K')
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=42
        ),
    }

    metrics_table, confusions, fitted = {}, {}, {}

    for name, clf in models.items():
        # Use dense preprocessor for Naive Bayes only
        dense = (name == "naive_bayes")
        pre = make_preprocessor(df, dense=dense)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics_table[name] = compute_metrics(y_test, y_prob, y_pred)
        confusions[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        fitted[name] = pipe

    # Save artifacts
    joblib.dump(le, os.path.join(args.artifacts, "label_encoder.joblib"))
    default_pre = make_preprocessor(df, dense=False)
    joblib.dump(default_pre, os.path.join(args.artifacts, "encoders.joblib"))

    for name, pipe in fitted.items():
        joblib.dump(pipe, os.path.join(args.artifacts, f"{name}.joblib"))

    # Save metrics & confusions
    pd.DataFrame(metrics_table).T.to_csv(os.path.join(args.artifacts, "metrics.csv"))
    with open(os.path.join(args.artifacts, "confusions.json"), "w") as f:
        json.dump(confusions, f, indent=2)

    # Print compact table
    out = pd.DataFrame(metrics_table).T[["accuracy","auc","precision","recall","f1","mcc"]].round(4)
    print("\n=== Comparison Table ===")
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Adult dataset CSV (with 'income' target)")
    ap.add_argument("--artifacts", default="model/artifacts", help="Artifacts output dir")
    args = ap.parse_args()
    main(args)
