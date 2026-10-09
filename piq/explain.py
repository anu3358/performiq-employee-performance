"""Explanations: which factors pushed a prediction up or down.

Uses SHAP (tree or linear explainer, depending on the winning model). Falls back to a simple 'swap in a typical value and see
what changes' method so the app never breaks.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from piq import config

_EXPLAINERS: dict = {}


def _original_name(col: str) -> str:
    if col.startswith("num__"):
        return col[5:]
    rest = col[5:]
    for key in config.CAT_FEATURES:
        if rest.startswith(key + "_"):
            return key
    return rest


def _typical_row() -> dict:
    """A 'typical employee' built from the data: median numbers, most common categories."""
    from piq.data import load_dataset
    df = load_dataset()
    row = {k: float(df[k].median()) for k in config.NUM_FEATURES}
    row.update({k: df[k].mode().iloc[0] for k in config.CAT_FEATURES})
    return row


def _shap_contributions(pipe, X: pd.DataFrame):
    import shap
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    key = id(pipe)
    if key not in _EXPLAINERS:
        if isinstance(clf, LogisticRegression):
            from piq.data import load_dataset
            background = pre.transform(load_dataset()[config.FEATURES].sample(100, random_state=config.RANDOM_STATE))
            _EXPLAINERS[key] = shap.LinearExplainer(clf, background)
        else:
            _EXPLAINERS[key] = shap.TreeExplainer(clf)
    Xt = pre.transform(X[config.FEATURES])
    sv = _EXPLAINERS[key].shap_values(Xt)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 2:
        sv = sv[..., None]
    names = pre.get_feature_names_out()
    out = np.zeros((len(X), len(config.FEATURES), sv.shape[-1]))
    for j, name in enumerate(names):
        out[:, config.FEATURES.index(_original_name(name)), :] += sv[:, j, :]
    return out


def _occlusion_contributions(pipe, X: pd.DataFrame):
    base = _typical_row()
    p0 = pipe.predict_proba(X[config.FEATURES])
    out = np.zeros((len(X), len(config.FEATURES), p0.shape[1]))
    for k, feat in enumerate(config.FEATURES):
        Xb = X[config.FEATURES].copy()
        Xb[feat] = base[feat]
        out[:, k, :] = p0 - pipe.predict_proba(Xb)
    return out


def contributions(pipe, X: pd.DataFrame) -> np.ndarray:
    """Returns array (rows, features, classes)."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, (RandomForestClassifier, XGBClassifier, LogisticRegression)):
        try:
            return _shap_contributions(pipe, X)
        except Exception:
            pass
    return _occlusion_contributions(pipe, X)


def global_importance(pipe, X: pd.DataFrame, sample: int = 300) -> pd.DataFrame:
    X = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
    c = np.abs(contributions(pipe, X)).mean(axis=(0, 2))
    df = pd.DataFrame({"feature": [config.LABELS[f] for f in config.FEATURES], "importance": c})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
