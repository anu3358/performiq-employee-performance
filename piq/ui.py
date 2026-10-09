"""Shared look-and-feel for every page of PerformIQ."""
import streamlit as st

COLORS = {
    "Needs Improvement": "#F43F5E",
    "Meets Expectations": "#FBBF24",
    "Exceeds Expectations": "#34D399",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(124,92,255,.20), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(34,211,238,.13), transparent 55%),
    #0B0F1A;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #0E1424; border-right: 1px solid rgba(255,255,255,.06); }
.block-container { padding-top: 2rem; max-width: 1280px; }

.hero {
  padding: 30px 34px; border-radius: 22px; margin-bottom: 22px;
  background: linear-gradient(135deg, rgba(124,92,255,.22), rgba(34,211,238,.10));
  border: 1px solid rgba(255,255,255,.09);
  box-shadow: 0 10px 40px rgba(124,92,255,.15);
}
.hero h1 { margin: 6px 0 6px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #fff, #B8A8FF 60%, #67E8F9); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { margin: 0; color: #A9B3CF; font-size: 1.05rem; max-width: 780px; }
.tag { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600;
  color: #C9BEFF; background: rgba(124,92,255,.18); border: 1px solid rgba(124,92,255,.4); }

.kpi { padding: 18px 20px; border-radius: 18px; background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(8px); transition: transform .2s, border-color .2s; }
.kpi:hover { transform: translateY(-3px); border-color: rgba(124,92,255,.55); }
.kpi-label { color: #8E9ABB; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #fff; margin-top: 4px; }
.kpi-hint { color: #6E7BA0; font-size: .8rem; margin-top: 2px; }

.card { padding: 20px 22px; border-radius: 18px; background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08); height: 100%; }
.card h4 { margin: 0 0 6px 0; color: #fff; }
.card p { margin: 0; color: #A9B3CF; font-size: .93rem; }

.section-title { font-size: 1.15rem; font-weight: 700; color: #fff; margin: 10px 0 2px 0; }
.section-cap { color: #8E9ABB; font-size: .88rem; margin-bottom: 10px; }

.result { padding: 22px; border-radius: 20px; text-align: center; border: 1px solid rgba(255,255,255,.1);
  background: rgba(255,255,255,.04); }
.result .label { font-size: .8rem; color: #8E9ABB; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.result .value { font-size: 2rem; font-weight: 800; margin-top: 4px; }
.result .conf { color: #A9B3CF; font-size: .9rem; margin-top: 2px; }

.tip { padding: 12px 16px; border-radius: 14px; margin-bottom: 8px; background: rgba(34,211,238,.07);
  border-left: 3px solid #22D3EE; color: #D5DCF0; font-size: .93rem; }

.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 600; border: 1px solid rgba(124,92,255,.5);
  background: linear-gradient(135deg, #7C5CFF, #5B8CFF); color: white; }
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.1); border-color: #B8A8FF; }
</style>
"""


def page_setup(title: str, icon: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"PerformIQ · {title}", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 📈 PerformIQ")
        st.caption("Explainable workforce intelligence")
        st.divider()
        st.caption("Built with Streamlit · scikit-learn · SHAP")


def hero(title: str, subtitle: str, tag: str | None = None) -> None:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="hero">{tag_html}<h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-cap">{caption}</div>', unsafe_allow_html=True)


def kpi(container, label: str, value, hint: str = "") -> None:
    container.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-hint">{hint}</div></div>',
        unsafe_allow_html=True)


def card(container, title: str, text: str) -> None:
    container.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>',
                       unsafe_allow_html=True)


def style_fig(fig, height: int = 360):
    fig.update_layout(template="plotly_dark", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, sans-serif", color="#D5DCF0"),
                      legend=dict(orientation="h", y=-0.2))
    return fig
