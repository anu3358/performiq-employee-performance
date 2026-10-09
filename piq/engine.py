"""The ONLY file the app pages talk to. It hides all the machine-learning details."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from piq import config, explain
from piq.data import load_dataset
from piq.training import train_and_save

ADVICE = {
    "training_hours": "Training time ({v} hrs/year) is holding the result back. Offer a focused learning plan for the skills this role needs.",
    "goal_completion": "Goal completion ({v}%) is holding the result back. Hold a short check-in to clarify priorities and remove blockers.",
    "peer_rating": "Peer feedback is holding the result back. Encourage a team feedback session and pair with a supportive mentor.",
    "satisfaction": "Job satisfaction is holding the result back. A career-growth conversation could help.",
    "work_life_balance": "Work-life balance is a concern. Look at flexible hours or workload sharing.",
    "overtime_hours": "{v} overtime hours a month is affecting the result. Review workload before burnout sets in.",
    "absences": "{v} absence days a year is affecting the result. Check in kindly to see if support is needed.",
    "projects_completed": "Projects delivered ({v}) is holding the result back. Break work into smaller, clearer milestones.",
    "years_at_company": "Newer team members benefit from onboarding support and a buddy.",
}


@lru_cache(maxsize=1)
def _artifacts():
    paths = (config.MODEL_PATH, config.METRICS_PATH, config.IMPORTANCE_PATH)
    if not all(p.exists() for p in paths):
        train_and_save()
    try:
        pipe = joblib.load(config.MODEL_PATH)
    except Exception:  # e.g. saved with a different library version
        train_and_save()
        pipe = joblib.load(config.MODEL_PATH)
    metrics = json.loads(config.METRICS_PATH.read_text())
    importance = pd.read_csv(config.IMPORTANCE_PATH)
    return pipe, metrics, importance


def get_dataset() -> pd.DataFrame:
    return load_dataset().copy()


def get_metrics() -> dict:
    return _artifacts()[1]


def global_importance() -> pd.DataFrame:
    return _artifacts()[2].copy()


def predict_one(profile: dict) -> dict:
    pipe = _artifacts()[0]
    proba = pipe.predict_proba(pd.DataFrame([profile])[config.FEATURES])[0]
    idx = int(np.argmax(proba))
    return {"label": config.CLASS_LABELS[idx],
            "confidence": float(proba[idx]),
            "probabilities": {l: float(p) for l, p in zip(config.CLASS_LABELS, proba)}}


def explain_one(profile: dict) -> pd.DataFrame:
    """Positive impact = the factor lifts performance, negative = it pulls performance down."""
    pipe = _artifacts()[0]
    X = pd.DataFrame([profile])[config.FEATURES]
    c = explain.contributions(pipe, X)[0]          # (features, classes)
    vals = c[:, -1] - c[:, 0]                      # 'Exceeds' push minus 'Needs Improvement' push
    df = pd.DataFrame({"key": config.FEATURES,
                       "feature": [config.LABELS[f] for f in config.FEATURES],
                       "impact": vals})
    return df.reindex(df["impact"].abs().sort_values(ascending=False).index).head(8).reset_index(drop=True)


def recommendations(profile: dict, drivers: pd.DataFrame) -> list:
    """Plain-language next steps based on the factors that hurt the result the most."""
    cutoff = 0.1 * drivers["impact"].abs().max() if len(drivers) else 0
    negatives = drivers[(drivers["impact"] < -cutoff) & drivers["key"].isin(ADVICE)]
    tips = [ADVICE[k].format(v=profile.get(k, "")) for k in negatives.sort_values("impact")["key"].head(3)]
    if not tips:
        tips = ["No major risk factors found. Keep recognising good work and plan growth opportunities."]
    return tips


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    pipe = _artifacts()[0]
    X = df.copy()
    for k, v in config.NUM_FEATURES.items():
        X[k] = pd.to_numeric(X[k], errors="coerce").fillna(v[3])
    proba = pipe.predict_proba(X[config.FEATURES])
    out = df.copy().reset_index(drop=True)
    out["predicted_performance"] = [config.CLASS_LABELS[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    out["exceeds_probability"] = proba[:, -1].round(3)
    return out


def fairness_report() -> pd.DataFrame:
    pipe = _artifacts()[0]
    df = load_dataset()
    pred = pipe.predict(df[config.FEATURES])
    df = df.assign(_top=(pred == len(config.CLASS_LABELS) - 1))
    attrs = [a for a in config.FAIRNESS_ATTRIBUTES if a in df.columns] or ["department"]
    rows = []
    for a in attrs:
        for g, sub in df.groupby(a):
            rows.append({"attribute": a, "group": str(g), "employees": len(sub),
                         "exceeds_rate": float(sub["_top"].mean())})
    return pd.DataFrame(rows)
