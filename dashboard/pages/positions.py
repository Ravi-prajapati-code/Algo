"""Dashboard — Open Positions page."""

import json
import os
import streamlit as st
import pandas as pd

from dashboard.charts import sector_allocation_pie
from config.settings import OUTPUTS_DIR


def render():
    st.title("Open Positions")

    state_path = os.path.join(OUTPUTS_DIR, "portfolio_state.json")
    if not os.path.exists(state_path):
        st.warning("No portfolio data yet.")
        return

    with open(state_path) as f:
        state = json.load(f)

    positions = state.get("positions", [])
    if not positions:
        st.info("No open positions currently.")
        return

    df = pd.DataFrame(positions)
    df["unreal_pnl_str"] = df["unrealized_pnl"].apply(lambda x: f"₹{x:+,.2f}")
    df["unreal_pct_str"] = df["unrealized_pct"].apply(lambda x: f"{x:+.2f}%")

    # Colour rows by P&L
    def highlight(row):
        color = "#1a3d2b" if row["unrealized_pnl"] >= 0 else "#3d1a1a"
        return [f"background-color: {color}"] * len(row)

    display_cols = [
        "symbol", "sector", "entry_date", "entry_price", "current_price",
        "shares", "stop_loss", "take_profit", "unreal_pnl_str", "unreal_pct_str"
    ]
    st.dataframe(
        df[display_cols].rename(columns={
            "unreal_pnl_str": "Unrealized P&L",
            "unreal_pct_str": "Change %",
        }).style.apply(highlight, axis=1),
        use_container_width=True, height=400
    )

    st.plotly_chart(sector_allocation_pie(positions), use_container_width=True)
