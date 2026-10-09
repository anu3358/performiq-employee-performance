"""Trust page: accuracy, important factors and a fairness check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from piq import config, engine
from piq.ui import hero, kpi, page_setup, section, style_fig

page_setup("Model Insights", "🔍")
hero("Model Insights", "How good is the model, what does it look at, and is it fair?",
     tag="Transparency")

m = engine.get_metrics()
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Accuracy", f"{m['accuracy']:.1%}", "on unseen employees")
kpi(c2, "F1 score", f"{m['f1']:.1%}", "balance of right calls")
kpi(c3, "ROC-AUC", f"{m['roc_auc']:.2f}", "ranking quality")
kpi(c4, "Cross-check", f"{m['cv_mean']:.1%}", "5-round average")

st.write("")
a, b = st.columns(2, gap="large")
with a:
    section("What matters most?", "Average influence of each factor on predictions")
    imp = engine.global_importance().sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Purples")
    fig.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Importance")
    st.plotly_chart(style_fig(fig, 420), width="stretch")
with b:
    section("Where does the model get it right?", "Rows = real result, columns = predicted")
    fig = px.imshow(m["confusion"], x=m["labels"], y=m["labels"], text_auto=True,
                    color_continuous_scale="Purples")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 420), width="stretch")

section("Models we compared", "We tried several and kept the best")
st.dataframe(pd.DataFrame(m["comparison"]), width="stretch", hide_index=True)

st.divider()
section("Fairness check",
        "Share of employees predicted to 'Exceed Expectations' in each group. "
        "Big gaps between groups are a warning sign worth investigating.")
fair = engine.fairness_report()
attrs = list(fair["attribute"].unique())
cols = st.columns(len(attrs))
for col, attr in zip(cols, attrs):
    sub = fair[fair["attribute"] == attr]
    fig = px.bar(sub, x="group", y="exceeds_rate", text=sub["exceeds_rate"].map("{:.0%}".format),
                 title=attr.replace("_", " ").title(), color_discrete_sequence=["#7C5CFF"])
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    col.plotly_chart(style_fig(fig, 300), width="stretch")
