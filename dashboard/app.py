"""
Streamlit dashboard entry point.
Run: streamlit run dashboard/app.py

Free hosting: deploy to https://streamlit.io/cloud
  → Connect GitHub repo → Set main file: dashboard/app.py
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Algo Swing Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("📈 Algo Swing Trader")
st.sidebar.caption("NSE Swing Trading — Paper Mode")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Open Positions", "Today's Signals", "Trade History", "Backtest"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption(
    "Strategy: EMA20/50 + RSI + MACD + Volume\n\n"
    "Broker: Upstox (₹0 delivery brokerage)\n\n"
    "Target: 15–20% CAGR\n\n"
    "⚠️ Paper trading only — not financial advice"
)

# Route to page
if page == "Overview":
    from dashboard.pages.overview import render
    render()

elif page == "Open Positions":
    from dashboard.pages.positions import render
    render()

elif page == "Today's Signals":
    from dashboard.pages.signals import render
    render()

elif page == "Trade History":
    from dashboard.pages.history import render
    render()

elif page == "Backtest":
    from dashboard.pages.backtest import render
    render()
