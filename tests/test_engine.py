"""Quick safety checks:  python -m pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piq import config, engine


def _profile():
    p = {k: v[3] for k, v in config.NUM_FEATURES.items()}
    p.update({k: v[1][0] for k, v in config.CAT_FEATURES.items()})
    return p


def test_prediction_is_valid():
    r = engine.predict_one(_profile())
    assert r["label"] in config.CLASS_LABELS
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_explanation_and_tips():
    d = engine.explain_one(_profile())
    assert {"feature", "impact", "key"} <= set(d.columns)
    assert len(engine.recommendations(_profile(), d)) >= 1


def test_batch_and_fairness():
    out = engine.predict_batch(pd.DataFrame([_profile(), _profile()]))
    assert "predicted_performance" in out and len(out) == 2
    assert not engine.fairness_report().empty


def test_model_quality():
    assert engine.get_metrics()["accuracy"] > 0.6
