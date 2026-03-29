"""Dashboard — Today's Signals page."""

import json
import os
import streamlit as st
import pandas as pd

from config.settings import OUTPUTS_DIR


def render():
    st.title("Today's Signals")

    sig_path = os.path.join(OUTPUTS_DIR, "signals.json")
    if not os.path.exists(sig_path):
        st.warning("No signals yet. Run `python main.py run` first.")
        return

    with open(sig_path) as f:
        data = json.load(f)

    st.caption(f"Generated: {data.get('generated_at', 'unknown')}")

    signals = data.get("signals", [])
    if not signals:
        st.info("No signals generated today.")
        return

    buys  = [s for s in signals if s["action"] == "BUY"]
    sells = [s for s in signals if s["action"] == "SELL"]
    holds = [s for s in signals if s["action"] == "HOLD"]

    col1, col2, col3 = st.columns(3)
    col1.metric("BUY",  len(buys),  delta_color="normal")
    col2.metric("SELL", len(sells), delta_color="inverse")
    col3.metric("HOLD", len(holds))

    if buys:
        st.subheader("🟢 BUY Signals")
        buy_rows = []
        for s in buys:
            ind = s.get("indicators", {})
            buy_rows.append({
                "Symbol":  s["symbol"],
                "Price":   f"₹{s['price']:.2f}",
                "Score":   s["score"],
                "RSI":     ind.get("rsi", "-"),
                "MACD H":  ind.get("macd_hist", "-"),
                "Vol Ratio": ind.get("vol_ratio", "-"),
                "Reason":  s["reason"],
            })
        st.dataframe(pd.DataFrame(buy_rows), use_container_width=True)

    if sells:
        st.subheader("🔴 SELL Signals")
        sell_rows = [
            {"Symbol": s["symbol"], "Price": f"₹{s['price']:.2f}", "Reason": s["reason"]}
            for s in sells
        ]
        st.dataframe(pd.DataFrame(sell_rows), use_container_width=True)

    if holds:
        st.subheader("⏸ Holding")
        hold_rows = [
            {"Symbol": s["symbol"], "Price": f"₹{s['price']:.2f}"}
            for s in holds
        ]
        st.dataframe(pd.DataFrame(hold_rows), use_container_width=True)
