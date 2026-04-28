from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from portfolio_intel.analytics import _last_two_closes, compute_portfolio_metrics
from portfolio_intel.bootstrap import ensure_default_portfolios
from portfolio_intel.broker import OrderIntent, PaperBroker
from portfolio_intel.importer import parse_holdings_csv
from portfolio_intel.marketdata import CachedOHLCVProvider, DiskOHLCVCache, default_cache_dir, provider_from_env
from portfolio_intel.risk_modes import cfg_for
from portfolio_intel.strategy import HybridStrategy
from portfolio_intel.store import PortfolioStore
from portfolio_intel.eval import compute_trade_metrics


st.set_page_config(page_title="Paper Portfolio Manager", layout="wide")


@st.cache_resource
def _store() -> PortfolioStore:
    s = PortfolioStore()
    ensure_default_portfolios(s)
    return s


@st.cache_resource
def _ohlcv():
    inner = provider_from_env()
    return CachedOHLCVProvider(inner, DiskOHLCVCache(default_cache_dir()))


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _metrics_panel(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Value", _fmt_money(m.total_value))
    c2.metric("Cash", _fmt_money(m.cash))
    c3.metric("Total Return", f"{_fmt_money(m.total_return)} ({m.total_return_pct*100:.2f}%)")
    c4.metric("Daily Change", f"{_fmt_money(m.daily_change)} ({m.daily_change_pct*100:.2f}%)")
    c5.metric("Diversification", f"{m.diversification_score:.0f}/100")


def _holdings_table(m):
    df = pd.DataFrame(
        [
            {
                "Symbol": h.symbol,
                "Name": h.name or "",
                "Qty": h.quantity,
                "Avg Buy": h.avg_buy_price,
                "Price": h.current_price,
                "Chg $": h.price_change,
                "Chg %": h.price_change_pct,
                "Mkt Value": h.market_value,
                "Cost Basis": h.cost_basis,
                "G/L $": h.gain_loss,
                "G/L %": h.gain_loss_pct,
                "Alloc %": h.allocation_pct,
                "Reinvest": h.reinvest,
            }
            for h in m.holdings
        ]
    )
    if df.empty:
        st.info("No holdings yet.")
        return
    st.dataframe(
        df.style.format(
            {
                "Qty": "{:,.4f}",
                "Avg Buy": "${:,.2f}",
                "Price": "${:,.2f}",
                "Chg $": "${:,.2f}",
                "Chg %": "{:.2%}",
                "Mkt Value": "${:,.2f}",
                "Cost Basis": "${:,.2f}",
                "G/L $": "${:,.2f}",
                "G/L %": "{:.2%}",
                "Alloc %": "{:.2%}",
            }
        ),
        use_container_width=True,
        height=420,
    )


def _trades_panel(trades):
    if not trades:
        st.info("No trades logged yet.")
        return
    df = pd.DataFrame(
        [
            {
                "Time (UTC)": t.timestamp,
                "Symbol": t.symbol,
                "Action": t.action,
                "Qty Δ": t.quantity_delta,
                "Price": t.price,
                "Notional": t.notional,
                "Realized PnL": t.pnl_realized,
                "Reason": t.reason,
            }
            for t in trades
        ]
    )
    st.dataframe(
        df.style.format(
            {
                "Qty Δ": "{:,.4f}",
                "Price": "${:,.2f}",
                "Notional": "${:,.2f}",
                "Realized PnL": "${:,.2f}",
            }
        ),
        use_container_width=True,
        height=340,
    )


def main():
    store = _store()
    ohlcv = _ohlcv()
    broker = PaperBroker(store)
    strat = HybridStrategy(ohlcv)

    portfolios = store.list_portfolios()
    if not portfolios:
        st.error("No portfolios available.")
        return

    with st.sidebar:
        st.header("Portfolios")
        selected = st.selectbox("Select portfolio", portfolios, format_func=lambda p: f"{p.name} ({p.portfolio_id})")
        st.divider()
        st.header("Risk mode")
        mode = st.selectbox("Mode", ["medium", "low", "aggressive"], index=["medium", "low", "aggressive"].index(selected.risk_mode))
        if mode != selected.risk_mode:
            selected.risk_mode = mode  # type: ignore[assignment]
            store.save_portfolio(selected)
            st.success(f"Risk mode set to {mode}")
        cfg = cfg_for(selected.risk_mode)
        st.caption(f"Max position weight: {cfg.max_position_weight:.0%}")
        st.caption(f"Target turnover/day: {cfg.target_turnover_per_day}")
        st.divider()
        st.header("Refresh")
        if st.button("Refresh prices"):
            st.rerun()
        st.divider()
        st.header("Bot")
        universe_txt = st.text_area(
            "Universe (comma-separated tickers)",
            value="QQQ, SMH, NVDA, AVGO, MU, GOOGL, ORCL, CRWD, PLTR, CVX, SLB",
            height=80,
        )
        if st.button("Run one bot step (paper)"):
            universe = [t.strip().upper() for t in universe_txt.split(",") if t.strip()]
            intents = strat.propose_intents(selected, universe)
            # Execute intents sequentially; failures are shown but do not stop the loop
            executed = 0
            for it in intents:
                try:
                    broker.apply_intent(selected, it)
                    executed += 1
                except Exception as e:
                    st.warning(f"{it.action} {it.symbol} skipped: {e}")
            st.success(f"Bot step complete. Executed {executed}/{len(intents)} intents.")
            st.rerun()

    cfg = cfg_for(selected.risk_mode)
    metrics = compute_portfolio_metrics(selected, ohlcv, max_position_weight=cfg.max_position_weight)

    st.title("Paper Trading Portfolio Manager (Simulation Only)")
    _metrics_panel(metrics)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Holdings")
        _holdings_table(metrics)

    with c2:
        st.subheader("Portfolio analytics")
        st.markdown("**Top gainers (daily %)**")
        st.write([f"{h.symbol}: {h.price_change_pct*100:.2f}%" for h in metrics.top_gainers] or ["(none)"])
        st.markdown("**Top losers (daily %)**")
        st.write([f"{h.symbol}: {h.price_change_pct*100:.2f}%" for h in metrics.top_losers] or ["(none)"])
        st.markdown("**Overexposed**")
        st.write([f"{h.symbol}: {h.allocation_pct*100:.1f}%" for h in metrics.overexposed] or ["(none)"])

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Manual portfolio input")
        txt = st.text_area("Paste holdings (SYMBOL, QTY, BUY_PRICE per line)", height=160)
        cash = st.number_input("Starting cash to set (overwrites portfolio cash)", min_value=0.0, value=float(selected.cash), step=100.0)
        if st.button("Import into this portfolio"):
            res = parse_holdings_csv(txt)
            if res.errors:
                st.error("Import errors:\n" + "\n".join(res.errors))
            else:
                selected.holdings = res.holdings  # type: ignore[assignment]
                selected.cash = float(cash)  # type: ignore[assignment]
                store.save_portfolio(selected)
                st.success(f"Imported {len(res.holdings)} holdings.")
                st.rerun()

    with right:
        st.subheader("Paper trade (manual)")
        sym = st.text_input("Symbol", value="")
        action = st.selectbox("Action", ["BUY", "SELL", "TRIM", "EXIT", "HOLD"])
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        reason = st.text_area("Reason (required)", height=90)
        if st.button("Execute simulated action"):
            if not sym.strip():
                st.error("Symbol required.")
            elif not reason.strip():
                st.error("Reason required.")
            else:
                # price from market data
                try:
                    px, _ = _last_two_closes(ohlcv, sym.upper())
                except Exception as e:
                    st.error(f"Could not fetch price: {e}")
                    return
                try:
                    trade = broker.apply_intent(
                        selected,
                        OrderIntent(symbol=sym, action=action, quantity=float(qty), price=float(px), reason=reason),
                    )
                    st.success(f"Logged {trade.action} for {trade.symbol} @ ${trade.price:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader("Trade activity")
    trades = store.load_trades(selected.portfolio_id, limit=500)
    pm = compute_trade_metrics(trades)
    st.caption(
        f"Trades: {pm.trades} | Win rate: {pm.win_rate*100:.1f}% | Total realized PnL: ${pm.total_pnl:,.2f} | "
        f"Max drawdown (realized): ${pm.max_drawdown:,.2f}"
    )
    _trades_panel(trades)


if __name__ == "__main__":
    main()

