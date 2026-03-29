"""Dashboard — Trade History page."""

import streamlit as st
import pandas as pd

from dashboard.charts import trade_history_chart
from db.repository import load_trades


def render():
    st.title("Trade History")

    trades = load_trades()
    if not trades:
        st.info("No completed trades yet.")
        return

    rows = []
    for t in trades:
        rows.append({
            "Symbol":      t.symbol,
            "Sector":      t.sector,
            "Entry Date":  str(t.entry_date),
            "Exit Date":   str(t.exit_date),
            "Hold Days":   t.hold_days,
            "Entry ₹":     t.entry_price,
            "Exit ₹":      t.exit_price,
            "Shares":      t.shares,
            "Gross P&L":   t.gross_pnl,
            "Charges":     t.charges,
            "Net P&L":     t.net_pnl,
            "Exit Reason": t.exit_reason,
        })

    df = pd.DataFrame(rows)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    winners = df[df["Net P&L"] > 0]
    win_rate = len(winners) / len(df) * 100 if len(df) > 0 else 0
    col1.metric("Total Trades",  len(df))
    col2.metric("Win Rate",      f"{win_rate:.1f}%")
    col3.metric("Total Net P&L", f"₹{df['Net P&L'].sum():+,.0f}")
    col4.metric("Total Charges", f"₹{df['Charges'].sum():,.0f}")

    st.divider()
    st.plotly_chart(trade_history_chart(rows), use_container_width=True)

    # Full trade log
    st.subheader("All Trades")

    def color_pnl(val):
        color = "green" if val > 0 else "red"
        return f"color: {color}"

    st.dataframe(
        df.style.applymap(color_pnl, subset=["Net P&L", "Gross P&L"]),
        use_container_width=True, height=450
    )
