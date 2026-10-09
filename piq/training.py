"""Trains several models, keeps the best one and saves everything the app needs."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from piq import config, explain
from piq.data import load_dataset


def _pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(config.NUM_FEATURES)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(config.CAT_FEATURES)),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_save(verbose: bool = False) -> dict:
    df = load_dataset()
    X = df[config.FEATURES]
    y = df[config.TARGET].map({l: i for i, l in enumerate(config.CLASS_LABELS)})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=config.RANDOM_STATE)
    rs = config.RANDOM_STATE
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.9, eval_metric="mlogloss", random_state=rs, n_jobs=1),
    }
    rows, fitted = [], {}
    for name, clf in candidates.items():
        pipe = _pipeline(clf)
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="accuracy").mean()
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append({"Model": name,
                     "Test accuracy": round(accuracy_score(yte, pred), 3),
                     "Test F1": round(f1_score(yte, pred, average="weighted"), 3),
                     "5-fold accuracy": round(float(cv), 3)})
        fitted[name] = pipe
        if verbose:
            print(rows[-1])

    best = max(rows, key=lambda r: r["5-fold accuracy"])["Model"]
    pipe = fitted[best]
    proba, pred = pipe.predict_proba(Xte), pipe.predict(Xte)
    metrics = {
        "best_model": best,
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, average="weighted")),
        "roc_auc": float(roc_auc_score(yte, proba, multi_class="ovr", average="macro")),
        "cv_mean": float(next(r["5-fold accuracy"] for r in rows if r["Model"] == best)),
        "confusion": confusion_matrix(yte, pred).tolist(),
        "labels": config.CLASS_LABELS,
        "comparison": sorted(rows, key=lambda r: -r["5-fold accuracy"]),
    }

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    explain.global_importance(pipe, Xte).to_csv(config.IMPORTANCE_PATH, index=False)
    return metrics
