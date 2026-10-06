import streamlit as st

# ธีมร่วมของทุกหน้า: ฟอนต์ไทย IBM Plex Sans Thai + โค้ดด้วย IBM Plex Mono
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');

:root {
    --ink: #16202a;
    --muted: #55606b;
    --accent: #1f6f5c;
    --accent-soft: #d8ebe4;
    --line: #dfe3e6;
}
html, body, [class*="css"], .stMarkdown, button, input, label, [data-testid="stSidebarNav"] {
    font-family: 'IBM Plex Sans Thai', system-ui, sans-serif;
}
.stMarkdown p, .stMarkdown li { line-height: 1.75; max-width: 72ch; }
h1, h2, h3 {
    font-family: 'IBM Plex Sans Thai', system-ui, sans-serif !important;
    color: var(--ink);
    letter-spacing: -0.01em;
    text-wrap: balance;
}
h1 { font-weight: 700 !important; font-size: clamp(2rem, 4vw, 2.8rem) !important; }
h2 { font-weight: 600 !important; margin-top: 2.5rem !important; padding-bottom: .4rem; border-bottom: 1px solid var(--line); }
h3 { font-weight: 600 !important; font-size: 1.15rem !important; margin-top: 1.5rem !important; }
code, pre, .stCode { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }
.block-container { padding-top: 2.5rem; max-width: 1080px; }
::selection { background: var(--accent-soft); color: var(--ink); }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stImage"] img { border-radius: 8px; }
.lede { color: var(--muted); font-size: 1.1rem; margin-top: -.5rem; }
.verdict { font-size: 1.6rem; font-weight: 700; color: var(--accent); margin: .25rem 0 1rem; }
</style>
"""


def apply_style(page_title: str, page_icon: str):
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def lede(text: str):
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)
