"""Single-employee prediction with explanations and a live what-if."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from piq import config, engine
from piq.ui import COLORS, hero, page_setup, section, style_fig

page_setup("Predict", "🎯")
hero("Performance Predictor",
     "Move the sliders and watch the prediction change live — a built-in what-if simulator.",
     tag="Live · Explainable")

left, right = st.columns([1, 1.25], gap="large")

profile = {}
with left:
    section("Employee profile", "Adjust any value to test a scenario")
    c1, c2 = st.columns(2)
    for i, (key, (label, lo, hi, default, step)) in enumerate(config.NUM_FEATURES.items()):
        with (c1 if i % 2 == 0 else c2):
            profile[key] = st.slider(label, lo, hi, default, step, key=f"num_{key}")
    for key, (label, options) in config.CAT_FEATURES.items():
        profile[key] = st.selectbox(label, options, key=f"cat_{key}")

result = engine.predict_one(profile)
label = result["label"]
probs = result["probabilities"]
top = config.CLASS_LABELS[-1]  # "Exceeds Expectations"

with right:
    section("Prediction")
    st.markdown(
        f'<div class="result"><div class="label">Predicted performance</div>'
        f'<div class="value" style="color:{COLORS[label]}">{label}</div>'
        f'<div class="conf">Model confidence: {result["confidence"]:.0%}</div></div>',
        unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    if b1.button("📌 Save as baseline"):
        st.session_state["baseline"] = dict(probs)
    base = st.session_state.get("baseline")
    if base:
        delta = (probs[top] - base[top]) * 100
        b2.metric(f"Chance of '{top}'", f"{probs[top]:.0%}", f"{delta:+.1f} pts vs baseline")
    else:
        b2.metric(f"Chance of '{top}'", f"{probs[top]:.0%}")

    fig = go.Figure(go.Bar(
        x=[probs[k] for k in config.CLASS_LABELS], y=config.CLASS_LABELS, orientation="h",
        marker_color=[COLORS[k] for k in config.CLASS_LABELS],
        text=[f"{probs[k]:.0%}" for k in config.CLASS_LABELS], textposition="outside"))
    fig.update_layout(title="Probability of each outcome", xaxis=dict(range=[0, 1.1], tickformat=".0%"))
    st.plotly_chart(style_fig(fig, 240), width="stretch")

st.divider()
d1, d2 = st.columns([1.2, 1], gap="large")
drivers = engine.explain_one(profile)
with d1:
    section("Why this prediction?", "Green lifts performance, red pulls it down (compared with a typical employee)")
    drivers = drivers.sort_values("impact")
    fig = go.Figure(go.Bar(
        x=drivers["impact"], y=drivers["feature"], orientation="h",
        marker_color=["#34D399" if v >= 0 else "#F43F5E" for v in drivers["impact"]]))
    fig.update_layout(xaxis_title="Effect on performance")
    st.plotly_chart(style_fig(fig, 380), width="stretch")
with d2:
    section("Suggested next steps", "Practical, fair and based on the biggest drivers")
    for tip in engine.recommendations(profile, drivers):
        st.markdown(f'<div class="tip">💡 {tip}</div>', unsafe_allow_html=True)
    st.caption("Predictions support — never replace — a manager's judgement.")
