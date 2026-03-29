"""Dashboard — Overview page: portfolio value, equity curve, daily P&L."""

import json
import os
import streamlit as st
import pandas as pd

from dashboard.charts import equity_curve_chart, pnl_bar_chart
from config.settings import INITIAL_CAPITAL, OUTPUTS_DIR


def render():
    st.title("Portfolio Overview")

    # Load portfolio state
    state_path = os.path.join(OUTPUTS_DIR, "portfolio_state.json")
    if not os.path.exists(state_path):
        st.warning("No portfolio data yet. Run `python main.py run` first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    # ── KPI Cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total = state.get("total_value", INITIAL_CAPITAL)
    cash  = state.get("cash", 0)
    cum   = state.get("cumulative_pnl", 0)
    day   = state.get("daily_pnl", 0)
    pct   = (total - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    col1.metric("Total Value", f"₹{total:,.0f}", f"₹{cum:+,.0f} ({pct:+.1f}%)")
    col2.metric("Cash",        f"₹{cash:,.0f}")
    col3.metric("Day P&L",     f"₹{day:+,.0f}")
    col4.metric("Positions",   state.get("open_positions", 0))

    st.divider()

    # ── Equity Curve ───────────────────────────────────────────────────
    try:
        from db.repository import load_snapshots
        snaps = load_snapshots()
        if snaps:
            dates  = [s.date for s in snaps]
            values = [s.total_value for s in snaps]
            daily_pnl = [s.daily_pnl for s in snaps]
            st.plotly_chart(equity_curve_chart(dates, values, INITIAL_CAPITAL), use_container_width=True)
            st.plotly_chart(pnl_bar_chart(dates, daily_pnl), use_container_width=True)
        else:
            st.info("Equity curve will appear after the first trading day.")
    except Exception as e:
        st.error(f"Could not load equity curve: {e}")
