import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import shap

def train_or_load(model_path: str, data_csv: str, feats: List[str]) -> XGBClassifier:
    if os.path.exists(model_path):
        return joblib.load(model_path)
    df = pd.read_csv(data_csv)
    X = df[feats].copy()
    y = df["dropout_flag"].astype(int)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, subsample=0.9, colsample_bytree=0.9,
        learning_rate=0.08, eval_metric="logloss", random_state=42, n_jobs=2
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

def train_from_csv(model_path: str, csv_path: str, feats: List[str]) -> Tuple[XGBClassifier, Dict]:
    df = pd.read_csv(csv_path)
    X = df[feats].copy()
    y = df["dropout_flag"].astype(int)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, subsample=0.9, colsample_bytree=0.9,
        learning_rate=0.06, eval_metric="logloss", random_state=42, n_jobs=2
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    metrics = {
        "n_samples": int(len(df)),
        "class_balance": {"dropout_0": int((y==0).sum()), "dropout_1": int((y==1).sum())},
    }
    return model, metrics

def predict_with_bands(model: XGBClassifier, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    prob = model.predict_proba(X)[:,1]
    # quintile bands (stable even for small batches)
    qs = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
    labels = []
    for p in prob:
        if p <= qs[0]: labels.append("Very Low")
        elif p <= qs[1]: labels.append("Low")
        elif p <= qs[2]: labels.append("Medium")
        elif p <= qs[3]: labels.append("High")
        else: labels.append("Very High")
    return prob, labels

def explain_rows(model: XGBClassifier, X: pd.DataFrame, top_k: int) -> List[Dict[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # shap_values is (n, d). positive => increases predicted probability (risk)
    out = []
    feature_names = list(X.columns)
    for row_vals in shap_values:
        pairs = list(zip(feature_names, row_vals))
        # sort by absolute contribution, keep signed values
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        contrib = {k: float(v) for k, v in pairs[:top_k]}
        out.append(contrib)
    return out
