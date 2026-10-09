"""Upload a spreadsheet of employees and score them all at once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from piq import config, engine
from piq.ui import COLORS, hero, kpi, page_setup, section, style_fig

page_setup("Team Analysis", "📂")
hero("Team Analysis", "Upload a CSV of employees and get predictions for the whole team in seconds.",
     tag="Batch scoring")

template = pd.DataFrame([{
    "employee_id": "E001",
    **{k: v[3] for k, v in config.NUM_FEATURES.items()},
    **{k: v[1][0] for k, v in config.CAT_FEATURES.items()},
}])

section("Step 1 · Get the template")
st.download_button("⬇️ Download CSV template", template.to_csv(index=False).encode(),
                   "performiq_template.csv", "text/csv")

section("Step 2 · Upload your file")
file = st.file_uploader("Drop a CSV here", type="csv")
use_sample = st.checkbox("No file handy? Use sample data instead")

data = None
if file is not None:
    data = pd.read_csv(file)
elif use_sample:
    data = engine.get_dataset().drop(columns=["performance"]).head(150)

if data is not None:
    missing = [c for c in list(config.NUM_FEATURES) + list(config.CAT_FEATURES) if c not in data.columns]
    if missing:
        st.error(f"Your file is missing these columns: {', '.join(missing)}")
        st.stop()

    out = engine.predict_batch(data)
    st.success(f"Scored {len(out):,} employees.")

    k1, k2, k3 = st.columns(3)
    kpi(k1, "Likely top performers", f"{(out['predicted_performance'] == config.CLASS_LABELS[-1]).sum()}", "exceed expectations")
    kpi(k2, "Need support", f"{(out['predicted_performance'] == config.CLASS_LABELS[0]).sum()}", "needs improvement")
    kpi(k3, "Average confidence", f"{out['confidence'].mean():.0%}", "of the model")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Outcome split")
        counts = out["predicted_performance"].value_counts().reindex(config.CLASS_LABELS).fillna(0).reset_index()
        counts.columns = ["performance", "employees"]
        fig = px.pie(counts, names="performance", values="employees", hole=.55,
                     color="performance", color_discrete_map=COLORS)
        st.plotly_chart(style_fig(fig), width="stretch")
    with c2:
        section("By department")
        fig = px.histogram(out, x="department", color="predicted_performance", barmode="group",
                           category_orders={"predicted_performance": config.CLASS_LABELS},
                           color_discrete_map=COLORS)
        fig.update_layout(legend_title=None, xaxis_title=None)
        st.plotly_chart(style_fig(fig), width="stretch")

    section("Results table")
    pick = st.multiselect("Filter by outcome", config.CLASS_LABELS, default=config.CLASS_LABELS)
    shown = out[out["predicted_performance"].isin(pick)]
    st.dataframe(shown, width="stretch", height=360)
    st.download_button("⬇️ Download results", shown.to_csv(index=False).encode(),
                       "performiq_results.csv", "text/csv")
