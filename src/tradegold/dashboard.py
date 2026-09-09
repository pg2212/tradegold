"""Streamlit dashboard for the GOLD STRATEGY.

Two charts, because the strategy lives on two timeframes:

  * **Entry timeframe** -- candles with the context EMAs stepped across them, the armed
    setup's three levels drawn only while that setup is live, and the trigger
    where price came back into the level.
  * **Context (60m)** -- the candles the setups are actually made of, with the
    volume bar that had to clear its own average for a setup to exist at all.

    streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .evaluation import evaluate_fixed, expanding_folds
from .strategy import LONG, SHORT

RUNS_DIR = Path("runs")

UP, DOWN = "#0ca30c", "#d03b3b"
# EMAs and setup levels overlay the same panel, so no hue may be reused
# between the two groups. Validated for CVD separation in both modes.
EMA_FAST, EMA_SLOW, EMA_TREND = "#3987e5", "#d95926", "#9085e9"
LEVEL_HIGH, LEVEL_ENTRY, LEVEL_STOP = "#d55181", "#c98500", "#e66767"
TARGET_LINE, EQUITY = "#1baf7a", "#eb6834"


def money(value: float | None, dp: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{'-' if value < 0 else '+'}${abs(value):,.{dp}f}"


@dataclass
class Run:
    path: Path
    name: str
    fingerprint: str
    stamp: str
    n_trades: int | None = None
    net_pnl: float | None = None

    @property
    def when(self) -> str:
        try:
            return datetime.strptime(self.stamp, "%Y%m%dT%H%M%SZ").strftime("%b %d %H:%M")
        except ValueError:
            return self.stamp[:8]

    @property
    def label(self) -> str:
        """Self-describing, so a run can be chosen without selecting it first."""
        if self.n_trades is None:
            return f"{self.name}  ·  {self.when}  ·  {self.fingerprint}"
        pnl = "n/a" if self.net_pnl is None else money(self.net_pnl, 0) + "/oz"
        return f"{self.name}  ·  {self.n_trades} trades  ·  {pnl}  ·  {self.when}"


def discover_runs(root: Path = RUNS_DIR) -> list[Run]:
    """Every readable run, newest first. Directories without signal artifacts
    (older pipelines) are skipped rather than shown as broken entries."""
    runs: list[Run] = []
    for path in sorted(root.glob("*__*__*"), reverse=True):
        if not (path / "signals.parquet").exists():
            continue
        stamp, name, fingerprint = path.name.split("__", 2)
        n_trades = net_pnl = None
        try:
            metrics = json.loads((path / "metrics.json").read_text())
            n_trades = metrics["trades"]["n_trades"]
            net_pnl = metrics["equity"]["net_pnl"]
        except (OSError, ValueError, KeyError):
            pass  # label falls back to the fingerprint form
        runs.append(Run(path=path, name=name, fingerprint=fingerprint, stamp=stamp,
                        n_trades=n_trades, net_pnl=net_pnl))
    return runs


@st.cache_data(show_spinner=False)
def load_run(path_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    path = Path(path_str)
    signals = pd.read_parquet(path / "signals.parquet")
    signals.index = pd.to_datetime(signals.index, utc=True)

    context = pd.read_parquet(path / "context.parquet")
    context.index = pd.to_datetime(context.index, utc=True)

    trades = pd.read_csv(path / "trades.csv")
    trades = trades.drop(columns=[c for c in trades.columns if c.startswith("Unnamed")])
    for col in ("signal_time", "entry_time", "exit_time", "setup_time"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], utc=True)

    equity = pd.read_csv(path / "equity.csv", index_col=0)
    equity.index = pd.to_datetime(equity.index, utc=True)

    metrics = json.loads((path / "metrics.json").read_text())
    return signals, context, trades, equity, metrics


def setup_id(signals: pd.DataFrame) -> pd.Series:
    """Number each armed setup so its levels can be drawn as separate segments."""
    armed = signals["active_dir"] != 0
    new = armed & (
        (signals["active_dir"] != signals["active_dir"].shift())
        | (signals["active_trigger"] != signals["active_trigger"].shift())
    )
    return new.cumsum().where(armed)


def _segments(index: pd.Index, values: pd.Series, groups: pd.Series) -> tuple[list, list]:
    """Concatenate per-setup runs with NaN separators so lines never join up."""
    xs: list = []
    ys: list = []
    for _, idx in groups.dropna().groupby(groups.dropna()).groups.items():
        xs.extend(list(idx) + [None])
        ys.extend(list(values.loc[idx]) + [None])
    return xs, ys


def entry_chart(
    view: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, *,
    show_emas: bool, show_levels: bool, show_volume: bool, dark: bool,
    bars: pd.DataFrame | None = None,
    entry_interval: str = "", context_interval: str = "60m",
) -> go.Figure:
    rows = 2 + int(show_volume)
    heights = [0.66, 0.16, 0.18] if show_volume else [0.78, 0.22]
    vol_title = f"Volume ({entry_interval})" if entry_interval else "Volume"
    titles = ["", vol_title, "Cumulative net PnL ($/oz)"] if show_volume \
        else ["", "Cumulative net PnL ($/oz)"]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=heights, subplot_titles=titles)

    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"],
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            name="Price", showlegend=False,
        ),
        row=1, col=1,
    )

    if show_emas:
        for col, color, label in (
            ("ema_fast", EMA_FAST, f"EMA 10 ({context_interval})"),
            ("ema_slow", EMA_SLOW, f"EMA 20 ({context_interval})"),
            ("ema_trend", EMA_TREND, f"EMA 50 ({context_interval})"),
        ):
            if col not in view.columns:
                continue
            fig.add_trace(
                go.Scatter(x=view.index, y=view[col], mode="lines", name=label,
                           line=dict(color=color, width=1.5, shape="hv"),
                           hovertemplate=label + " $%{y:,.2f}<extra></extra>"),
                row=1, col=1,
            )

    if show_levels:
        groups = setup_id(view)
        for col, color, label, dash in (
            ("active_high", LEVEL_HIGH, "Setup high", "solid"),
            ("active_trigger", LEVEL_ENTRY, "Entry level (retracement)", "solid"),
            ("active_low", LEVEL_STOP, "Stop side", "dot"),  # dotted: not a price you trade
        ):
            xs, ys = _segments(view.index, view[col], groups)
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name=label, connectgaps=False,
                           line=dict(color=color, width=2, dash=dash),
                           hovertemplate=label + " $%{y:,.2f}<extra></extra>"),
                row=1, col=1,
            )

    # Triggers that the backtest declined (no retracement) still matter -- they
    # show where the rules fired but the fill was not real.
    skipped = view[(view["signal"] != 0) & view["trigger_immediate"]]
    if not skipped.empty:
        fig.add_trace(
            go.Scatter(
                x=skipped.index, y=skipped["active_trigger"], mode="markers",
                marker=dict(symbol="circle-open", size=11, color="#9aa0a6",
                            line=dict(width=1.6)),
                name=f"Trigger, no retracement ({len(skipped)})",
                hovertemplate="fired with price already through the level<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    for direction, marker, name in ((LONG, "triangle-up", "BUY"), (SHORT, "triangle-down", "SELL")):
        subset = trades[trades["direction"] == direction]
        if subset.empty:
            continue
        colors = [UP if p > 0 else DOWN for p in subset["net_pnl"]]
        fig.add_trace(
            go.Scatter(
                x=subset["entry_time"], y=subset["entry_price"], mode="markers",
                marker=dict(symbol=marker, size=14, color=colors,
                            line=dict(color="rgba(255,255,255,.9)", width=1.3)),
                name=f"{name} ({len(subset)})",
                customdata=subset[["net_pnl", "exit_reason", "bars_held", "r_multiple"]].to_numpy(),
                hovertemplate=(
                    f"<b>{name}</b> %{{x}}<br>fill $%{{y:,.2f}}"
                    "<br>net %{customdata[0]:+,.2f} $/oz (%{customdata[3]:+.2f} R)"
                    "<br>%{customdata[1]} after %{customdata[2]} bars<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Each trade's realised path: entry to exit, plus its stop and target.
    # Shape coordinates are ISO strings, not Timestamps: the browser renderer
    # tolerates Timestamps but static export (kaleido) cannot serialise them.
    if bars is None:
        bars = view
    for t in trades.itertuples():
        color = UP if t.net_pnl > 0 else DOWN
        x0, x1 = t.entry_time.isoformat(), t.exit_time.isoformat()
        fig.add_shape(type="line", x0=x0, x1=x1,
                      y0=t.entry_price, y1=t.exit_price,
                      line=dict(color=color, width=1.6), row=1, col=1)
        fig.add_shape(type="line", x0=x0, x1=x1,
                      y0=t.target_price, y1=t.target_price,
                      line=dict(color=TARGET_LINE, width=1, dash="dot"), row=1, col=1)
        if getattr(t, "trailing_stop", False) and bars is not None:
            # Reconstruct the ratchet: the running best of the last completed
            # candle's low (long) or high (short) across the trade's life.
            span = bars.loc[t.entry_time:t.exit_time]
            if not span.empty:
                ref = span["lowC"] if t.direction == LONG else span["highC"]
                path = ref.cummax() if t.direction == LONG else ref.cummin()
                path = (path.clip(lower=t.initial_stop) if t.direction == LONG
                        else path.clip(upper=t.initial_stop))
                fig.add_trace(
                    go.Scatter(
                        x=path.index, y=path, mode="lines", showlegend=False,
                        line=dict(color=LEVEL_STOP, width=1.4, dash="dot", shape="hv"),
                        hovertemplate="trailing stop $%{y:,.2f}<extra></extra>",
                    ),
                    row=1, col=1,
                )
        else:
            fig.add_shape(type="line", x0=x0, x1=x1,
                          y0=t.stop_price, y1=t.stop_price,
                          line=dict(color=LEVEL_STOP, width=1, dash="dot"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=[t.exit_time], y=[t.exit_price], mode="markers",
                       marker=dict(symbol="x", size=9, color=color),
                       showlegend=False,
                       hovertemplate=f"exit {t.exit_reason}<br>$%{{y:,.2f}}<extra></extra>"),
            row=1, col=1,
        )

    if show_volume:
        colors = np.where(view["Close"] >= view["Open"], UP, DOWN)
        fig.add_trace(
            go.Bar(x=view.index, y=view["Volume"], marker_color=colors, name="Volume",
                   showlegend=False, hovertemplate="vol %{y:,.0f}<extra></extra>"),
            row=2, col=1,
        )

    eq_row = rows
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity["equity"], mode="lines",
                   line=dict(color=EQUITY, width=1.8), name="Equity", showlegend=False,
                   hovertemplate="%{y:+,.2f} $/oz<extra></extra>"),
        row=eq_row, col=1,
    )
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,.5)", width=1), row=eq_row, col=1)

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=880, margin=dict(l=10, r=10, t=28, b=10),
        hovermode="x unified", xaxis_rangeslider_visible=False, dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="$/oz", row=1, col=1)
    return fig


def context_chart(view: pd.DataFrame, dark: bool) -> go.Figure:
    """The 60m candles setups are made of, with the volume gate that defines them."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.68, 0.32],
                        subplot_titles=("", "Volume vs setup threshold"))

    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["openC"], high=view["highC"],
            low=view["lowC"], close=view["closeC"],
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            name="60m", showlegend=False,
        ),
        row=1, col=1,
    )
    for col, color, label in (("ema_fast", EMA_FAST, "EMA 10"),
                              ("ema_slow", EMA_SLOW, "EMA 20"),
                              ("ema_trend", EMA_TREND, "EMA 50")):
        fig.add_trace(
            go.Scatter(x=view.index, y=view[col], mode="lines", name=label,
                       line=dict(color=color, width=1.5)),
            row=1, col=1,
        )

    for flag, marker, color, label in (
        ("bull_setup", "triangle-up", UP, "Bullish setup"),
        ("bear_setup", "triangle-down", DOWN, "Bearish setup"),
    ):
        subset = view[view[flag]]
        if subset.empty:
            continue
        y = subset["lowC"] * 0.999 if flag == "bull_setup" else subset["highC"] * 1.001
        fig.add_trace(
            go.Scatter(x=subset.index, y=y, mode="markers",
                       marker=dict(symbol=marker, size=13, color=color,
                                   line=dict(color="rgba(255,255,255,.9)", width=1.2)),
                       name=f"{label} ({len(subset)})",
                       customdata=subset[["setup_trigger", "setup_high", "setup_low"]].to_numpy(),
                       hovertemplate=(
                           "<b>" + label + "</b> %{x}"
                           "<br>entry $%{customdata[0]:,.2f}"
                           "<br>high $%{customdata[1]:,.2f} · low $%{customdata[2]:,.2f}"
                           "<extra></extra>"
                       )),
            row=1, col=1,
        )

    is_setup = view["bull_setup"] | view["bear_setup"]
    fig.add_trace(
        go.Bar(x=view.index, y=view["volumeC"],
               marker_color=np.where(is_setup, EMA_SLOW, "rgba(130,130,130,.55)"),
               name="Volume", showlegend=False,
               hovertemplate="vol %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view.index, y=view["vol_threshold"], mode="lines",
                   line=dict(color=LEVEL_STOP, width=1.6, dash="dash"),
                   name="Setup threshold",
                   hovertemplate="threshold %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view.index, y=view["vol_ma"], mode="lines",
                   line=dict(color="rgba(150,150,150,.85)", width=1.2),
                   name="Volume average",
                   hovertemplate="average %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=720, margin=dict(l=10, r=10, t=28, b=10),
        hovermode="x unified", xaxis_rangeslider_visible=False, dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="$/oz", row=1, col=1)
    return fig



def export_figure(fig: go.Figure, path: str | Path, **kwargs) -> Path:
    """Write a figure to a static image file.

    Plotly keeps a tz-aware DatetimeIndex as an object array of Timestamps.
    The browser renderer copes; the static renderer serialises through orjson
    and cannot. Normalising to ISO strings on a copy keeps the interactive path
    untouched while making export work.
    """
    import copy

    out = copy.deepcopy(fig)
    for trace in out.data:
        x = getattr(trace, "x", None)
        if x is None:
            continue
        trace.x = [
            None if v is None else (v.isoformat() if isinstance(v, pd.Timestamp) else v)
            for v in x
        ]
    for shape in out.layout.shapes:
        for attr in ("x0", "x1"):
            value = getattr(shape, attr)
            if isinstance(value, pd.Timestamp):
                setattr(shape, attr, value.isoformat())

    # The charts put their legend just above the plot, where Streamlit supplies
    # the heading separately. A figure-level title lands in the same band, so
    # make room for both rather than letting them overlap.
    title = out.layout.title
    if title is not None and title.text:
        top = max(out.layout.margin.t or 0, 118)
        out.update_layout(
            margin=dict(t=top),
            title=dict(x=0.01, xanchor="left", y=0.985, yanchor="top"),
            legend=dict(y=1.012, yanchor="bottom"),
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_image(path, **kwargs)
    return path


HOW_TO_READ = """
### What this strategy does

It waits for a **60-minute candle** that looks like real buying or selling
pressure, then tries to enter on the **pullback** rather than chasing the move.

A 60m candle becomes a **setup** only if all five hold:

| # | Condition | Why |
|---|---|---|
| 1 | volume > its own 30-period average × 1.5 | real participation, not drift |
| 2 | candle closed in the trade's direction | the move is confirmed |
| 3 | the 3 candles before it did *not* all go the same way | don't chase a stale move |
| 4 | close on the right side of the 50 EMA | trade with the trend |
| 5 | the 10 EMA is leading the 20 or 50 | momentum is present |

That candle's high and low then define **three levels**, and the strategy waits —
sometimes for hours — for price to pull back to the entry level on the lower timeframe.
Touching it is the trade. Each setup fires once, then is discarded.

### Reading the Entry chart (5m)

| What you see | What it is |
|---|---|
| **Blue / orange / violet steps** | the 60m EMAs (10 / 20 / 50). Stairs,
  because they only move when a 60m candle closes |
| **Pink line** | the setup candle's high |
| **Amber line** | the entry level — price must pull back to here |
| **Dotted red line** | the stop side (the far end of the setup range) |
| **▲ / ▼ markers** | actual fills, green if the trade won, red if it lost |
| **✕** | the exit, with a line from entry to exit |
| **Hollow grey ○** | a trigger the backtest **refused**: price was already
  past the entry level when the setup formed, so no pullback happened |

Levels appear **only while their setup is live**, which is why the chart is empty
between trades. That is the strategy waiting, not missing data.

### Reading the Context (60m) tab

The candles the setups are built from. In the volume panel, the **red dashed
line is the threshold** a candle must clear. Bars turn **orange** when that
candle actually became a setup — a tall grey bar cleared the volume test but
failed one of the other four.

### The numbers at the top

- **Win rate** — the share of trades that made money. Compare it to the
  break-even rate in the run report: this setup geometry pays about 0.67:1, so
  it needs to win **more than ~60%** of the time just to break even.
- **Profit factor** — gross wins ÷ gross losses. Above 1.0 is profitable.
- **Setups armed** — how many setups formed. Far more than trades, because many
  never get their pullback.

**These results cover about two months** — the data provider caps 5-minute
history at 60 days. With ~34 trades, treat everything here as a directional
read, not a verdict.
"""



def walk_forward_chart(rows: pd.DataFrame, dark: bool) -> go.Figure:
    """Per-fold net PnL. A strategy with an edge should not need every bar green,
    but it should not be one colour either."""
    colors = [UP if v > 0 else DOWN for v in rows["test_net"]]
    labels = [f"{r.fold}<br>{r.test_start:%b %d} – {r.test_end:%b %d}"
              for r in rows.itertuples()]

    fig = go.Figure(
        go.Bar(
            x=labels, y=rows["test_net"], marker_color=colors,
            customdata=rows[["test_n", "test_win"]].to_numpy(),
            hovertemplate=("%{x}<br>net %{y:+,.2f} $/oz"
                           "<br>%{customdata[0]} trades"
                           "<br>win rate %{customdata[1]:.1%}<extra></extra>"),
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(140,140,140,.7)", width=1.2))
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Net PnL ($/oz)", showlegend=False, bargap=0.35,
    )
    return fig


def header_metrics(metrics: dict) -> None:
    t, e, s = metrics["trades"], metrics["equity"], metrics["setups"]
    cols = st.columns(6)
    cols[0].metric("Net PnL ($/oz)", money(e["net_pnl"], 0))
    cols[1].metric("Trades", f"{t['n_trades']}")
    cols[2].metric("Win rate", f"{(t['win_rate'] or 0) * 100:.1f}%" if t["n_trades"] else "n/a")
    pf = t.get("profit_factor")
    cols[3].metric("Profit factor", f"{pf:.2f}" if pf else "n/a")
    cols[4].metric("Max drawdown ($/oz)", money(e["max_drawdown"], 0))
    cols[5].metric("Setups armed", f"{s['setups_armed']}")


def main() -> None:
    st.set_page_config(page_title="GOLD STRATEGY", page_icon="📊", layout="wide")
    dark = st.get_option("theme.base") != "light"

    runs = discover_runs()
    if not runs:
        st.error(
            "No runs found in `runs/`. Generate one first:\n\n"
            "```bash\ntradegold run -c configs/default.yaml\n```"
        )
        return

    st.sidebar.title("GOLD STRATEGY")
    st.sidebar.caption("Volume setups on the higher timeframe, "
                       "retracement entries on the lower")

    run = st.sidebar.selectbox("Run", runs, format_func=lambda r: r.label)
    signals, context, trades, equity, metrics = load_run(str(run.path))

    st.sidebar.divider()
    st.sidebar.subheader("View")
    modes = ["Around a trade", "Around a setup", "Recent bars", "Custom dates"]
    mode = st.sidebar.radio("Navigate by", modes, label_visibility="collapsed")

    if mode == "Around a trade" and not trades.empty:
        labels = [
            f"#{i}  {t.side:5s}  {t.entry_time:%b %d %H:%M}"
            f"  {t.net_pnl:+.2f}  {t.exit_reason}"
            for i, t in enumerate(trades.itertuples())
        ]
        pick = st.sidebar.selectbox("Trade", range(len(labels)), format_func=lambda i: labels[i])
        pad = st.sidebar.slider("Bars either side", 20, 500, 120, step=20)
        t = trades.iloc[pick]
        lo = signals.index.searchsorted(t["entry_time"]) - pad
        hi = signals.index.searchsorted(t["exit_time"]) + pad
        view = signals.iloc[max(0, lo): min(len(signals), hi)]
    elif mode == "Around a setup":
        armed = signals[signals["active_dir"] != 0]
        times = armed.index[armed["active_dir"].ne(armed["active_dir"].shift())
                            | armed["active_trigger"].ne(armed["active_trigger"].shift())]
        if len(times) == 0:
            st.sidebar.warning("No setups armed in this run.")
            view = signals.iloc[-400:]
        else:
            pick = st.sidebar.selectbox(
                "Setup", range(len(times)),
                format_func=lambda i: (
                    f"#{i}  {times[i]:%b %d %H:%M}  "
                    + ("LONG" if signals.loc[times[i], "active_dir"] == LONG else "SHORT")
                ),
            )
            pad = st.sidebar.slider("Bars either side", 20, 500, 120, step=20)
            centre = signals.index.searchsorted(times[pick])
            view = signals.iloc[max(0, centre - pad): min(len(signals), centre + pad * 2)]
    elif mode == "Recent bars":
        n = st.sidebar.slider("Bars", 100, 3000, 500, step=100)
        view = signals.iloc[-n:]
    else:
        lo_d, hi_d = signals.index[0].date(), signals.index[-1].date()
        picked = st.sidebar.date_input("Range", (lo_d, hi_d), lo_d, hi_d)
        if isinstance(picked, tuple) and len(picked) == 2:
            start = pd.Timestamp(picked[0], tz="UTC")
            end = pd.Timestamp(picked[1], tz="UTC") + pd.Timedelta(1, unit="D")
            view = signals.loc[start:end]
        else:
            view = signals

    if view.empty:
        st.warning("No bars in the selected window.")
        return

    st.sidebar.divider()
    show_emas = st.sidebar.checkbox("60m EMAs (10 / 20 / 50)", value=True)
    show_levels = st.sidebar.checkbox("Setup levels", value=True)
    show_volume = st.sidebar.checkbox("Volume panel", value=True)

    lo_t, hi_t = view.index[0], view.index[-1]
    in_view = trades[(trades["entry_time"] >= lo_t) & (trades["entry_time"] <= hi_t)]

    st.title("GOLD STRATEGY")
    st.caption(
        f"`{run.name}` · {metrics['data']['entry_interval']} entries on "
        f"{metrics['data']['context_interval']} setups · config `{run.fingerprint}` · "
        f"showing {len(view):,} of {len(signals):,} bars · {len(in_view)} trades in view"
    )
    header_metrics(metrics)
    with st.expander("How to read this  —  what the strategy does, and what every line means"):
        st.markdown(HOW_TO_READ)
    st.divider()

    entry_tf = metrics["data"]["entry_interval"]
    context_tf = metrics["data"]["context_interval"]
    tab_entry, tab_context, tab_wf, tab_trades = st.tabs(
        [f"Entry chart ({entry_tf})", f"Context ({context_tf})",
         "Walk-forward", "Trade ledger"]
    )

    with tab_entry:
        st.plotly_chart(
            entry_chart(view, in_view, equity.loc[lo_t:hi_t],
                        show_emas=show_emas, show_levels=show_levels,
                        show_volume=show_volume, dark=dark, bars=signals,
                        entry_interval=entry_tf, context_interval=context_tf),
            width="stretch", config={"scrollZoom": True},
        )
        st.caption(
            "Pink = setup high · amber = entry (retracement) level · dotted red = stop side. "
            "Levels are drawn only while that setup is armed. Hollow grey circles are triggers "
            "that fired with price already through the level, which the backtest skips."
        )

    with tab_context:
        pad = pd.Timedelta(2, unit="D")
        cview = context.loc[lo_t - pad: hi_t + pad]
        if cview.empty:
            st.info("No context candles in this window.")
        else:
            st.plotly_chart(context_chart(cview, dark), width="stretch",
                            config={"scrollZoom": True})
            st.caption(
                "Orange volume bars are candles that cleared the threshold *and* passed every "
                "other filter, becoming setups. A tall grey bar cleared volume but failed the "
                "trend, direction or pullback test."
            )

    with tab_wf:
        st.markdown(
            "**Expanding-window folds.** Each fold covers a later slice of the sample, "
            "having 'seen' everything before it. There is no model to train here, so "
            "this is a consistency check: a real edge should not confine itself to one "
            "stretch of the year."
        )
        n_folds = st.slider("Folds", 2, 8, 4, key="wf_folds")
        warm = st.slider("History reserved before the first fold", 0.10, 0.60, 0.30,
                         step=0.05, key="wf_warm")
        try:
            folds = expanding_folds(signals.index, n_folds, warm)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            wf = evaluate_fixed(trades, folds)
            rows = wf.frame
            if rows.empty or rows["test_n"].sum() == 0:
                st.info("No trades fall inside the fold windows.")
            else:
                s_ = wf.summary()
                c = st.columns(4)
                c[0].metric("Folds in profit", f"{s_['positive_folds']}/{s_['folds']}")
                c[1].metric("Total across folds ($/oz)", money(s_["total_net"], 0))
                c[2].metric("Best fold ($/oz)", money(s_["best_fold"], 0))
                c[3].metric("Worst fold ($/oz)", money(s_["worst_fold"], 0))

                st.plotly_chart(walk_forward_chart(rows, dark), width="stretch")

                show = rows.copy()
                show["window"] = [f"{r.test_start:%b %d} – {r.test_end:%b %d}"
                                  for r in rows.itertuples()]
                st.dataframe(
                    show[["fold", "window", "test_n", "test_net", "test_win", "test_pf"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "fold": st.column_config.TextColumn("Fold"),
                        "window": st.column_config.TextColumn("Test window"),
                        "test_n": st.column_config.NumberColumn("Trades"),
                        "test_net": st.column_config.NumberColumn("Net $/oz", format="%+.2f"),
                        "test_win": st.column_config.NumberColumn("Win rate", format="percent"),
                        "test_pf": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                    },
                )
                st.caption(
                    "To re-select a parameter on each fold's history instead of holding it "
                    "fixed, run `tradegold walkforward --select depth`. On this data that "
                    "made results *worse* than not tuning at all."
                )

    with tab_trades:
        if in_view.empty:
            st.info("No trades in this window.")
        else:
            cols = ["side", "entry_time", "exit_time", "entry_price", "trigger_level",
                    "exit_price", "stop_price", "target_price", "bars_held",
                    "net_pnl", "r_multiple", "exit_reason"]
            st.dataframe(
                in_view[[c for c in cols if c in in_view.columns]],
                width="stretch", hide_index=True,
                column_config={
                    "side": st.column_config.TextColumn("Side"),
                    "entry_time": st.column_config.DatetimeColumn("Entry", format="MMM D, HH:mm"),
                    "exit_time": st.column_config.DatetimeColumn("Exit", format="MMM D, HH:mm"),
                    "entry_price": st.column_config.NumberColumn("Fill $", format="%.2f"),
                    "trigger_level": st.column_config.NumberColumn("Level $", format="%.2f"),
                    "exit_price": st.column_config.NumberColumn("Exit $", format="%.2f"),
                    "stop_price": st.column_config.NumberColumn("Stop $", format="%.2f"),
                    "target_price": st.column_config.NumberColumn("Target $", format="%.2f"),
                    "bars_held": st.column_config.NumberColumn("Bars"),
                    "net_pnl": st.column_config.NumberColumn("Net $/oz", format="%+.2f"),
                    "r_multiple": st.column_config.NumberColumn("R", format="%+.2f"),
                    "exit_reason": st.column_config.TextColumn("Exit"),
                },
            )

    with st.expander("Run configuration"):
        st.code((run.path / "config.yaml").read_text(), language="yaml")
    with st.expander("Full metrics"):
        st.json(metrics, expanded=False)


if __name__ == "__main__":
    main()
