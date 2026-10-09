"""PerformIQ - Home page. Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

from piq import config, engine
from piq.ui import COLORS, card, hero, kpi, page_setup, section, style_fig

page_setup("Home", "📈")

hero("PerformIQ",
     "Predict employee performance early, understand exactly why, and act fairly — "
     "an explainable, bias-aware AI assistant for modern HR teams.",
     tag="AI for People Analytics · Explainable · Fairness-checked")

df = engine.get_dataset()
m = engine.get_metrics()

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Employees analysed", f"{len(df):,}", "in the training data")
kpi(c2, "Top performers", f"{(df['performance'] == 'Exceeds Expectations').mean():.0%}", "exceed expectations")
kpi(c3, "Average satisfaction", f"{df['satisfaction'].mean():.1f} / 5", "across all teams")
kpi(c4, "Model accuracy", f"{m['accuracy']:.0%}", f"best model: {m['best_model']}")

st.write("")
left, right = st.columns(2, gap="large")
with left:
    section("Performance mix by department", "Where are our strongest and most at-risk teams?")
    fig = px.histogram(df, x="department", color="performance", barmode="stack",
                       category_orders={"performance": config.CLASS_LABELS},
                       color_discrete_map=COLORS)
    fig.update_layout(xaxis_title=None, yaxis_title="Employees", legend_title=None)
    st.plotly_chart(style_fig(fig), width="stretch")
with right:
    section("Does training help?", "Training hours for each performance group")
    fig = px.box(df, x="performance", y="training_hours", color="performance",
                 category_orders={"performance": config.CLASS_LABELS},
                 color_discrete_map=COLORS)
    fig.update_layout(xaxis_title=None, yaxis_title="Training hours", showlegend=False)
    st.plotly_chart(style_fig(fig), width="stretch")

st.write("")
section("What you can do here")
a, b, c = st.columns(3, gap="medium")
card(a, "🎯 Predict", "Fill in an employee profile and instantly see the expected performance level, "
                     "with the reasons behind it.")
card(b, "📂 Analyse a team", "Upload a spreadsheet of employees and get predictions for the whole "
                            "team, ready to download.")
card(c, "🔍 Trust the model", "See which factors matter most and check that the model treats "
                             "different groups fairly.")
st.info("👈 Use the menu on the left to explore the other pages.")
