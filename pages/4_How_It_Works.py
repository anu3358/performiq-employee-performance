"""Project story page - also your interview cheat-sheet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from piq.ui import card, hero, page_setup, section

page_setup("How It Works", "🧠")
hero("How PerformIQ Works", "From raw HR data to a fair, explainable prediction.", tag="Under the hood")

section("The pipeline")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1B2238", fontcolor="white",
        color="#7C5CFF", fontname="Helvetica", margin="0.25,0.15"];
  edge [color="#22D3EE"];
  Data [label="1. HR data"]; Clean [label="2. Clean &\\nprepare"];
  Train [label="3. Train & compare\\nmodels"]; Explain [label="4. Explain\\n(SHAP)"];
  Fair [label="5. Fairness\\ncheck"]; App [label="6. Streamlit\\napp"];
  Data -> Clean -> Train -> Explain -> Fair -> App;
}
""")

st.write("")
a, b, c = st.columns(3, gap="medium")
card(a, "🧹 Prepare", "Missing values are filled, categories are turned into numbers and everything is put on a common scale.")
card(b, "🤖 Learn", "Several models compete. The one that predicts best on employees it has never seen is kept.")
card(c, "💬 Explain", "Every prediction comes with the exact reasons, so managers can trust and challenge it.")

st.write("")
section("Interview questions you may get")
qa = {
    "Why did you build this?": "Performance reviews are slow and subjective. This tool spots patterns early so HR can offer support sooner — not to replace people's judgement.",
    "Why is explainability important in HR?": "Decisions about people must be justified. Showing the reasons behind each prediction makes it fair, challengeable and trustworthy.",
    "How did you check for bias?": "I compared prediction rates across groups such as gender and age band. Large gaps trigger a review of the data and features.",
    "What are the limits?": "It learns from past data, so past unfairness can leak in. It should support decisions, never make them alone.",
    "What would you improve next?": "Use real company data, track results over time, and add attrition (people leaving) prediction.",
}
for q, ans in qa.items():
    with st.expander(q):
        st.write(ans)
