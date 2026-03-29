"""Dashboard — Backtest page: run backtest from UI."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta


def render():
    st.title("Backtesting")
    st.caption("Run the strategy on historical data to validate 15–20% CAGR target")

    col1, col2, col3 = st.columns(3)
    start_date = col1.date_input("Start Date", value=date.today() - timedelta(days=3*365))
    end_date   = col2.date_input("End Date",   value=date.today() - timedelta(days=1))
    capital    = col3.number_input("Starting Capital (₹)", value=75000, step=5000)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Fetching data & running backtest… this may take 2–5 minutes"):
            try:
                from data.fetcher import fetch_all
                from data.universe import get_all_symbols
                from backtest.engine import BacktestEngine
                from backtest.metrics import calculate_metrics
                from db.repository import init_db

                init_db()
                symbols = get_all_symbols()
                lookback = (end_date - start_date).days + 60
                data = fetch_all(symbols, lookback_days=lookback)

                if not data:
                    st.error("No data fetched — check internet connection")
                    return

                engine = BacktestEngine(data, start_date, end_date, float(capital))
                result = engine.run()
                metrics = calculate_metrics(result, float(capital))

                st.session_state["backtest_result"] = result
                st.session_state["backtest_metrics"] = metrics
                st.success("Backtest complete!")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

    if "backtest_metrics" not in st.session_state:
        st.info("Configure dates above and click Run Backtest.")
        return

    metrics = st.session_state["backtest_metrics"]
    result  = st.session_state["backtest_result"]

    # ── Metrics ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    cagr_pass = "✅" if metrics["passes_15pct_target"] else "❌"
    col1.metric("CAGR",          f"{metrics['cagr_pct']:+.2f}%  {cagr_pass}")
    col2.metric("Sharpe Ratio",  metrics["sharpe_ratio"])
    col3.metric("Max Drawdown",  f"{metrics['max_drawdown_pct']:.2f}%")
    col4.metric("Win Rate",      f"{metrics['win_rate_pct']:.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades",  metrics["total_trades"])
    col2.metric("Profit Factor", metrics["profit_factor"])
    col3.metric("Avg Hold Days", metrics["avg_hold_days"])
    col4.metric("Charges Drag",  f"{metrics['charges_drag_pct']:.2f}%")

    verdict = "✅ PASS — Ready for paper trading" if metrics["all_criteria_met"] else "❌ FAIL — Needs parameter tuning"
    st.info(f"**Overall: {verdict}**")

    # ── Equity Curve ───────────────────────────────────────────────────
    if result.equity_curve:
        dates  = sorted(result.equity_curve.keys())
        values = [result.equity_curve[d] for d in dates]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines",
                                 name="Portfolio", line=dict(color="#00cc88")))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray",
                      annotation_text="Starting Capital")
        fig.update_layout(title="Backtest Equity Curve", template="plotly_dark",
                          xaxis_title="Date", yaxis_title="Value (₹)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Trade Log ──────────────────────────────────────────────────────
    if result.trades:
        st.subheader("Trade Log")
        rows = [
            {"Symbol": t.symbol, "Entry": str(t.entry_date), "Exit": str(t.exit_date),
             "Days": t.hold_days, "Entry ₹": t.entry_price, "Exit ₹": t.exit_price,
             "Net P&L": t.net_pnl, "Charges": t.charges, "Reason": t.exit_reason}
            for t in result.trades
        ]
        df = pd.DataFrame(rows)

        def color_pnl(val):
            return f"color: {'green' if val > 0 else 'red'}"

        st.dataframe(
            df.style.applymap(color_pnl, subset=["Net P&L"]),
            use_container_width=True, height=400
        )
