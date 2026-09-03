# tradegold — GOLD STRATEGY

A Python port of the TradingView **GOLD STRATEGY** indicator, with a backtester,
a candlestick dashboard, and tests that pin the original's semantics.

```bash
uv venv --python 3.12
uv pip install -r requirements-dev.txt -e .

tradegold run -c configs/default.yaml   # backtest, writes a run directory
streamlit run app.py                    # inspect it on the charts
```

## The strategy

Setups are defined on **completed 60-minute candles**; entries trigger on
**5-minute bars**.

A candle becomes a setup when all of these hold:

| Condition | Meaning |
|---|---|
| `volume > SMA(volume, 30) × 1.5` | volume expansion |
| `close > open` (long) / `close < open` (short) | the candle moved in the trade's direction |
| not all three preceding candles same direction | don't chase a move already three legs old |
| `close ≥ EMA50` (long) / `close ≤ EMA50` (short) | on the right side of the trend |
| `EMA50 < EMA10` or `EMA20 < EMA10` (long) | fast EMA leading |

That candle's range then defines three levels:

```
long :  high = highC              low = min(lowC, prev close)
        entry = high − 0.40 × (high − low)

short:  low  = lowC               high = max(highC, prev close)
        entry = high − 0.60 × (high − low)     (= low + 0.40 × range)
```

The strategy then waits — possibly for hours — for price to pull back into that
entry level on the 5m chart. Touching it is the trade. A setup is consumed by
its trigger or replaced by an opposite setup; it never re-arms.

## Charts

`streamlit run app.py` gives three tabs:

- **Entry chart (5m)** — candles with the 60m EMAs stepped across them, the armed
  setup's three levels drawn *only while that setup is live*, BUY/SELL markers,
  and each trade's stop and target spanning its life.
- **Context (60m)** — the candles setups are made of, with the volume bar that had
  to clear its threshold. Orange bars became setups; a tall grey bar cleared
  volume but failed the trend, direction or pullback test.
- **Trade ledger** — every trade in view, with the level, the fill, and how far
  apart they were.

Navigate by trade, by setup, by recent bars, or by date. Each run also writes a
static `strategy_chart.png` centred on its last trade.

## Faithfulness notes

Three places where the source is ambiguous or self-contradicting. All are
preserved as configuration rather than silently "fixed":

1. **Volume gate.** The comment says `volMA90` (3× the average); the code tests
   `volMA45` (1.5×). The code is what ran, so 1.5 is the default —
   `configs/strict_volume.yaml` uses 3.0.
2. **The two retracement settings are both measured from the high**, which makes
   the pair look asymmetric when it is not. `long_retracement: 0.40` and
   `short_retracement: 0.60` both put the entry **40% back from the extreme the
   move made**, and both risk 60% of the range — R:R 0.667 on each side.
   They are exposed separately so the symmetry can be broken deliberately.
3. **No exit rules.** The source is an *indicator*: it emits entries and plots
   the setup extremes, but names no exit. The default reads its own plots —
   stop at the far side of the setup candle, target at the near side. That
   geometry is **0.67 reward:risk**, so it needs a >60% hit rate to break even.

## Execution realism

The single largest determinant of results is not the strategy — it is how a
trigger becomes a fill. Two controls exist because the naive choices produce
large illusory profits:

- **`execution.entry_fill`** defaults to `next_open`: the alert fires on the
  trigger bar's close, the fill is the next bar's open. Assuming instead that a
  limit order fills at the level added **$165 of the original $230/oz** — 72% of
  all profit came from filling *through* the level rather than at it.
- **`execution.skip_immediate_triggers`** defaults to true. 45 of 80 triggers
  fired on the very bar their setup armed, meaning price was *already* past the
  entry level and no retracement ever happened. Those are not trades you could
  take.
- **`execution.min_risk_fraction`** rejects fills landing on top of their stop.
  One short filled $1.60 from its stop, producing an 18.9 R multiple that alone
  dragged the mean planned reward:risk from 0.65 to 1.74.

## Results

### Full year 2025 — XAUUSD spot, 15m entries (the real test)

23,182 bars of 15-minute XAUUSD, Jan–Dec 2025, with the 60m setup context
resampled from the same file. **114 trades over twelve months.**

| Config | Trades | Win rate | Break-even | Net $/oz | Profit factor |
|---|---|---|---|---|---|
| `xauusd_2025` (rules as written) | 114 | 54.4% | 60.8% | **−90.44** | 0.84 |
| `xauusd_2025_expiry` (setups expire after 4h) | 93 | 54.8% | 61.5% | −35.94 | 0.91 |
| `xauusd_2025_rtarget` (2R exits) | 108 | 36.1% | 33.3% | −137.78 | 0.84 |
| `xauusd_2025_strict` (3× volume gate) | 9 | 88.9% | 57.3% | +119.32 | 13.67 |

**Over a full year, the strategy loses money.** The reason is structural, not
bad luck: the setup-extreme exits pay about **0.65:1**, so the strategy needs to
win **60.8%** of the time to break even, and it wins **54.4%**.

It is a consistent shortfall rather than one bad patch:

| Quarter | Trades | Win rate | vs break-even | Net $/oz |
|---|---|---|---|---|
| 2025 Q1 | 28 | 60.7% | −0.0 pts | +31.80 |
| 2025 Q2 | 30 | 56.7% | −4.1 pts | −44.20 |
| 2025 Q3 | 37 | 48.6% | −12.1 pts | −30.50 |
| 2025 Q4 | 19 | 52.6% | −8.1 pts | −47.50 |

Only **3 of 12 months** cleared the break-even win rate; only 4 were profitable.
Longs and shorts lost at the same rate (54.5% / 54.2%), so the short skew seen in
the short sample was noise.

Statistically the case is strong but not airtight: 62/114 wins gives a 95%
confidence interval of **[44.8%, 63.7%]**, which still contains the 60.8% needed
(one-sided p = 0.098). The honest reading is *no demonstrated edge*, not *proven
loser* — but the burden of proof sits with the strategy, and a full year of data
does not discharge it.

Two results worth not misreading:

- **`xauusd_2025_rtarget` shows the problem is not the exit geometry.** Switching
  to 2R targets moves the break-even win rate from 60.8% down to 33.3%, and the
  win rate falls almost exactly in step (54.4% → 36.1%). Profit factor is
  unchanged at 0.84. Changing where you take profit redistributes the same
  non-edge.
- **`xauusd_2025_strict` is not a discovery.** A profit factor of 13.67 on
  **9 trades** in a year is noise, and its 0.7% market exposure means it is
  barely a strategy. Do not tune toward it.

### Two-month sample, GC=F futures, 5m entries (superseded)

The original 60-day yfinance window produced 34 trades at a 76.5% win rate and
+$156/oz. The full-year test above shows that for what it was: a small sample.
Kept here as a caution, not a result.

### Retracement depth sweep, and why it changes nothing

`long_retracement` is the strategy's one genuinely free parameter, and it sets
the geometry directly: entering `d` of the way back from the move's extreme gives
a reward:risk of `d / (1 - d)` and a break-even win rate of `1 - d`. Deeper entry,
better geometry, but fewer fills and a lower hit rate. Sweeping the full year:

| depth | R:R | break-even | trades | win rate | net $/oz | PF |
|---|---|---|---|---|---|---|
| 0.20 | 0.25 | 80% | 43 | 65.1% | +37.48 | 1.43 |
| 0.30 | 0.43 | 70% | 83 | 66.3% | +4.58 | 1.01 |
| **0.40** (as written) | 0.67 | 60% | 114 | 54.4% | **−90.44** | 0.84 |
| 0.50 | 1.00 | 50% | 141 | 53.2% | +41.78 | 1.07 |
| **0.55** | 1.22 | 45% | 151 | 51.7% | **+61.20** | 1.09 |
| 0.60 | 1.50 | 40% | 151 | 47.0% | +53.25 | 1.08 |
| 0.70 | 2.33 | 30% | 147 | 34.7% | −72.85 | 0.89 |

The shipped 0.40 lands in the worst trough of the whole sweep, and 0.55 looks like
a find at +$61.20. **It is not.** Splitting the year in half kills it:

| depth | H1 Jan–Jun | H2 Jul–Dec | works in both? |
|---|---|---|---|
| 0.20 | +34.55 | +2.93 | yes, but H2 is ~zero on 21 trades |
| 0.40 (as written) | −12.45 | −77.99 | no |
| 0.50 | +62.95 | −21.16 | no |
| **0.55 (sweep optimum)** | **+83.39** | **−22.19** | **no** |
| 0.60 | +78.11 | −24.86 | no |
| 0.70 | +31.89 | −104.73 | no |

**Every depth except 0.20 loses money in the second half of the year**, and 0.20's
second half is +$2.93 across 21 trades — indistinguishable from zero. The sweep
optimum was fitted to January–June and does not survive July–December.

That is the sweep's real finding: the depth parameter is not underspecified, it is
inert. What changed between halves was the market, not the setting.

## Walk-forward evaluation (the reference harness)

```bash
tradegold walkforward -b configs/default.yaml -c configs/xauusd_2025.yaml --folds 4
tradegold walkforward -c configs/xauusd_2025.yaml --folds 4 --select depth
```

This strategy fits no model, so expanding folds are not about training leakage.
They answer a harder question: **would a choice made with only the data available
at the time have worked afterwards?** Each fold trains on everything before it and
is scored only on itself.

`--select` re-picks a parameter on each fold's history. That converts the earlier
depth sweep from an in-sample optimisation into an out-of-sample record.

### Result: tuning makes it worse

| Setup | Total $/oz | Folds in profit |
|---|---|---|
| Trailing stop, parameters fixed | **−52.50** | 1/4 |
| Volume multiplier re-selected per fold | −109.13 | 1/4 |
| Parameters fixed (rules as written) | −141.95 | 1/4 |
| **Entry depth re-selected per fold** | **−170.99** | 1/4 |

Re-selecting entry depth is the **worst** outcome — worse than never tuning at
all. The mechanism is visible fold by fold:

| Fold | Test window | Depth chosen | In-sample | Out-of-sample |
|---|---|---|---|---|
| 1 | Apr 23 – Jun 26 | 0.55 | +136.54 | −53.15 |
| 2 | Jun 26 – Aug 28 | 0.55 | +83.39 | +7.16 |
| 3 | Aug 28 – Oct 29 | 0.60 | +105.21 | −49.64 |
| 4 | Oct 29 – Dec 31 | 0.35 | +58.07 | −75.37 |

Every fold picks a value that made three-figure profits on its own history, and
every one of them loses forward. That is overfitting caught in the act, and it is
why the +$61/oz from the whole-year depth sweep should be ignored: the harness
shows what that number is worth when you cannot see the future.

Volume selection "helped" only in the sense of losing less — and it chose 1.25 in
all four folds, so it is not adapting, just a different fixed value.

**Every configuration lands on 1 of 4 folds profitable.** That consistency, across
fixed and tuned settings alike, is the strongest evidence in this repository that
the edge is absent rather than mis-specified.

## Configuration

Nothing needs editing. Every value is validated, and overridable:

```bash
tradegold run --set strategy.volume_multiplier=3.0
tradegold run --set execution.entry_fill=level        # see the illusory version
tradegold run -b configs/default.yaml -c configs/r_target.yaml
```

| Command | Purpose |
|---|---|
| `tradegold run` | Backtest; writes a versioned run directory |
| `tradegold fetch --refresh` | Download and cache both timeframes |
| `tradegold show-config` | Print the resolved, validated config |
| `tradegold registry` | List pluggable components |
| `tradegold walkforward` | Expanding-fold evaluation; `--select` tunes per fold |
| `tradegold prune` | Remove stale and duplicate run directories |

### Using your own data

The `csv` provider reads a local file, matching column names case-insensitively
and ignoring extras like `symbol`. When only one timeframe is available, set
`context_source: resample` and the setup timeframe is aggregated from the entry
bars — which also guarantees the two series agree about every candle.

```yaml
data:
  provider: csv
  ticker: "data/raw/XAUUSD_spot_bid_15m_2025.csv"
  entry_interval: "15m"
  context_interval: "60m"
  context_source: resample
```


Runs accumulate fast — every chart tweak writes another directory with identical
settings, and a stale run showing superseded numbers is worse than clutter. Prune
reports by default and only deletes with `--yes`:

```bash
tradegold prune --keep-per-name 1          # dry run: newest run per config
tradegold prune --keep-per-name 1 --yes    # apply
```

Three rules: directories without signal artifacts belong to a retired pipeline;
runs sharing a config fingerprint beyond `--keep-per-config` are identical
re-runs; runs sharing a *name* beyond `--keep-per-name` are superseded revisions
— a changed config gets a new fingerprint, which must not protect an old result.

## Testing

```bash
pytest            # 87 tests
ruff check src tests
```

The tests that matter most are in `tests/test_alignment.py`. Reproducing Pine's
`request.security(..., close[1], lookahead_on)` is the one place this project
could silently acquire lookahead bias, so context bars are stamped with their
**close** time and merged backward — a 5m bar at 10:05 sees the 60m candle that
closed at 10:00 and nothing newer. One test mutates a future context bar and
asserts no earlier row changes.

`tests/test_strategy.py` pins the setup conditions, both retracement formulas,
and the state machine (arm → wait → consume, replacement by an opposite setup,
expiry, and the immediate-trigger flag).

## Layout

```
app.py              Streamlit entry point
configs/            default, strict_volume, r_target, long_only
src/tradegold/
  config.py         typed, validated, fingerprinted settings
  data.py           dual-timeframe loading + as-of context alignment
  strategy.py       context indicators, setup detection, trigger state machine
  backtest.py       fill models, stop/target, long and short
  metrics.py        trade, setup and equity statistics
  dashboard.py      Streamlit charts
  reporting.py      static run chart
tests/
legacy/             the original five ML scripts, and the ML pipeline that
                    replaced them, archived rather than deleted
runs/               generated, gitignored
data/               cached market data, gitignored
```
