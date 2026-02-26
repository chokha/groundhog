# Groundhog — NQ Futures Pre-Market Intelligence System

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Daily Operations Playbook](#daily-operations-playbook)
5. [Module Reference](#module-reference)
6. [Data Sources & Pipelines](#data-sources--pipelines)
7. [Playbook System](#playbook-system)
8. [Journal & Learning Loop](#journal--learning-loop)
9. [Output Reference](#output-reference)
10. [Trading Concepts & Glossary](#trading-concepts--glossary)
11. [Backtesting](#backtesting)
12. [Troubleshooting](#troubleshooting)

---

## Overview

Groundhog is an NQ futures pre-market intelligence system that combines **GEX (Gamma Exposure)** analysis, **ICT liquidity levels**, and **Opening Range Breakout (ORB)** mechanics into a structured morning brief. It generates an actionable pre-market plan before RTH open and an ORB execution plan after the opening range closes.

The system has three pillars:

1. **Morning Brief** — Pre-market plan with GEX regime, ICT levels, playbook pattern matches, prior session context, and if/then trade plans.
2. **Post-Game Journal** — Outcome recording with ORB break direction, MFE/MAE, T1/runner hits, and checkpoints at 9:35/9:45/10:00/10:30/11:00/EOD.
3. **Auto-Learning** — Offline pattern discovery that scores existing playbook patterns, finds condition correlations, and suggests new patterns from journal data.

### System Phases

```
Phase 1: Infrastructure
  playbook.yaml    — Human-editable pattern rules
  playbook.py      — YAML loader + condition evaluator
  journal.py       — Post-game outcome recorder
  learn.py         — Offline analysis + pattern discovery

Phase 2: Live Intelligence
  Scorecard in brief    — T1 hit rate, MFE, MAE, confidence per playbook match
  Pattern decay         — Warns when a pattern's recent performance drops
  Prior session context — Yesterday's outcome printed in today's brief
  Auto-journal flag     — `--journal` shortcut from dashboard CLI
```

---

## Architecture

### Module Dependency Graph

```
                      ┌──────────────────────┐
                      │    dashboard.py       │  Main Brief Generator
                      │   (Two-Phase Output)  │
                      └──────────┬───────────┘
                                 │
          ┌──────────────────────┼───────────────────────┐
          │                      │                       │
   ┌──────▼──────┐        ┌─────▼─────┐          ┌──────▼──────┐
   │   gex.py    │        │  ict.py   │          │   orb.py    │
   │  GEX Maps   │        │ICT Levels │          │ORB Simulation│
   └──────┬──────┘        └─────┬─────┘          └──────┬──────┘
          │                     │                       │
   ┌──────▼──────┐              │                       │
   │  orats.py   │              │                       │
   │ ORATS API   │              │                       │
   └─────────────┘              │                       │
                                │                       │
        ┌───────────────────────┼───────────────────────┤
        │                       │                       │
 ┌──────▼──────┐         ┌──────▼──────┐         ┌─────▼──────┐
 │playbook.py  │         │ journal.py  │         │  learn.py  │
 │  Matching   │         │  Outcomes   │         │  Learning  │
 └──────┬──────┘         └─────────────┘         └────────────┘
        │
 ┌──────▼──────┐
 │playbook.yaml│
 │  7 Patterns │
 └─────────────┘

Shared: config.py (API keys, constants, session times)
Utils:  earnings_calendar.py, archive_bars.py
```

### Directory Structure

```
futures/
├── config.py                  # Constants, API keys, session times, regime labels
├── dashboard.py               # Main brief generator (live, sim, backtest modes)
├── journal.py                 # Post-game journal entry generator
├── learn.py                   # Pattern learning + auto-discovery
├── playbook.py                # YAML pattern matcher + condition evaluator
├── playbook.yaml              # Human-editable trading pattern rules (7 patterns)
├── gex.py                     # GEX computation engine
├── ict.py                     # ICT level computation
├── orb.py                     # ORB simulation engine
├── orats.py                   # ORATS API client + caching
├── backtest.py                # Historical backtest runner
├── earnings_calendar.py       # Major NDX earnings checker
├── archive_bars.py            # Bar data archiver
│
├── daily_briefs/              # Generated pre-market briefs (auto-created)
│   ├── brief_YYYY-MM-DD.json  #   Full brief JSON per date
│   └── daily_log.csv          #   Running CSV log of all briefs
│
├── journal/                   # Post-game journal entries (auto-created)
│   └── YYYY-MM-DD.json        #   One entry per trading day
│
├── orats_cache/               # Cached ORATS API responses (auto-created)
│   └── {NDX,QQQ}_YYYY-MM-DD.json
│
├── backtest_output/           # Backtest results (auto-created)
│   ├── backtest_all.csv
│   ├── backtest_5m.csv
│   ├── backtest_15m.csv
│   └── skipped_days.csv
│
├── flow_snapshots/            # Live option flow snapshots (auto-created)
├── archive/                   # Archived 1m bar data (optional)
├── nq_1m.parquet              # Primary NQ 1-min bars (Nov 2024 - Nov 2025)
└── nq_1m_databento.parquet    # Databento NQ 1-min bars (alternative source)
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- ORATS Data API subscription (https://orats.com)

### 1. Install dependencies

```bash
pip install pandas numpy requests pyarrow yfinance pyyaml pandas_market_calendars pytz
```

### 2. Configure ORATS API key

Set your token as an environment variable:
```bash
export ORATS_TOKEN=your_token_here
```

Or edit directly in `config.py`:
```python
ORATS_TOKEN = "your_token_here"
```

### 3. Place NQ parquet data (optional)

For backtest/sim modes, place your `nq_1m.parquet` file in the project root. The file should contain 1-minute OHLCV bars with columns: `ts` (datetime), `open`, `high`, `low`, `close`, `volume`.

Live mode fetches data from yfinance and does not require a parquet file.

---

## Daily Operations Playbook

### Morning Routine (09:15 - 09:25 ET)

**Run the pre-market brief:**

```bash
# Live mode — fetches all data from APIs
python dashboard.py

# Sim mode — replay a historical date
python dashboard.py --date 2026-02-24

# Sim mode with specific time
python dashboard.py --date 2026-02-24 --time 09:35
```

**What you get:**

1. **Prior Session Context** — Yesterday's ORB break, T1/runner outcome, MFE/MAE, lesson
2. **Premarket Plan** — Spots, VIX regime, GEX map, ICT levels, trade state, day mode, if/then plan
3. **Playbook Matches** — Pattern matches with live scorecard stats (T1 rate, MFE, MAE, confidence) and decay warnings
4. **ORB Execution Plan** (if ORB closed) — Break/hold/retest/fade guidance with entry mode

**Example premarket plan output:**

```
  ══════════════════════════════════════════════════════════════
    NQ PRE-MARKET PLAN — 2026-02-24
  ══════════════════════════════════════════════════════════════

  -- PRIOR SESSION (2026-02-20) ──────────────────────────────
  ORB Break: LONG @ 09:40  |  Day Range: 384 pts
  T1: HIT  |  Runner: HIT  |  MFE: +240  MAE: -38
  Patterns: runner_wide_range, relief_bounce_after_selloff
  Lesson: Relief bounce played out

  SPOTS:   NQ 24826  |  NDX 24777  |  QQQ 602.88  |  Basis 49
  VIX:     21.45 (ELEVATED) — Case 1 setups appearing
  TARGET:  100 pts  |  RUNNER: 212 pts

  KEY LEVELS:
    PDH   25040    PDL   24668    PDC   24770
    ONH   24902    ONL   24761
    GEX   RES 24949  |  SUP 24649

  TRADE STATE: NEG_GAMMA_TREND  |  MODE: RUNNER
  BIAS: NEUTRAL — let price action decide

  ── PLAYBOOK MATCHES ──────────────────────────────────────
  [HIGH  ] Runner Mode + Wide Overnight Range       2/2 (FULL)
           T1: 9/10 (90%)  |  MFE: 271  |  MAE: 117  |  DEVELOPING
           ** Runner + wide range — T1 hits 90% of the time
           -> Trust the ORB break. Take T1 with confidence, trail for runner.
  [MEDIUM] Relief Bounce After Selloff               3/3 (FULL)
           T1: 5/7 (71%)  |  MFE: 198  |  MAE: 85  |  DEVELOPING
           ** Cautious long — T1 only, no runner
  ──────────────────────────────────────────────────────────
```

### After Market Close (16:00+ ET)

**Generate journal entry:**

```bash
# Quick — auto-detects today's date
python dashboard.py --journal

# Specific date with context
python journal.py --date 2026-02-24 --macro "tariff fears" --lesson "Relief bounce played out"

# Specific date via dashboard shortcut
python dashboard.py --journal --date 2026-02-24
```

**What gets saved** (`journal/2026-02-24.json`):
- Full brief snapshot from that morning
- Computed outcomes: ORB break direction/time, price checkpoints, MFE/MAE, T1/runner hit
- Matched playbook patterns
- Your macro context and lesson notes

### Weekly Review

**Run the learning report:**

```bash
# Terminal report only
python learn.py

# Also write suggestion file
python learn.py --yaml

# Require minimum 3 observations per pattern
python learn.py --min-n 3
```

**What you get:**

1. **Pattern Scorecard** — Grade each playbook pattern (A-F) with T1 rate, MFE, MAE, confidence level
2. **Condition Correlations** — Which brief fields best predict good outcomes
3. **Pattern Suggestions** — Auto-discovered 2-3 condition combos with ready-to-paste YAML

**Adding discovered patterns:**

1. Review `playbook_suggestions.yaml` (or terminal output)
2. If a pattern looks meaningful, copy the YAML block into `playbook.yaml`
3. Adjust `priority`, `bias`, `warning`, `action`, and `notes` as needed
4. The pattern will be matched in the next morning brief

---

## Module Reference

### dashboard.py — Main Brief Generator

The central orchestrator. Fetches all data, computes GEX/ICT/ORB, composes the brief, prints it, and saves JSON + CSV output.

**Three execution modes:**

| Mode | Command | Data Source |
|------|---------|-------------|
| **Live** | `python dashboard.py` | yfinance + ORATS live APIs |
| **Sim** | `python dashboard.py --date 2026-02-24` | yfinance historical bars + ORATS cached chains |
| **Backtest** | `python dashboard.py --date 2025-11-14 --parquet nq_1m.parquet` | Parquet bars + ORATS cached chains |

**CLI arguments:**

| Flag | Description |
|------|-------------|
| `--date YYYY-MM-DD` | Trade date (triggers sim mode) |
| `--time HH:MM` | Sim time in ET (default: 09:25) |
| `--parquet PATH` | Parquet file (triggers legacy backtest mode) |
| `--journal` | Generate post-game journal for date, then exit |

**Key functions:**

| Function | Purpose |
|----------|---------|
| `run_live_brief()` | Live mode: all data from APIs |
| `run_sim_brief(date, time)` | Sim mode: yfinance bars clamped to simulated time |
| `run_backtest_brief(date, parquet)` | Legacy backtest mode with parquet data |
| `generate_and_print_brief(...)` | Core generator: GEX + ICT + playbook + output |
| `classify_vix(vix_price)` | VIX regime classification (LOW/NORMAL/ELEVATED/HIGH/PANIC) |
| `classify_trade_state(gex, spot)` | Pre-market state: VACUUM_UP/DOWN, INSIDE_INV, etc. |
| `classify_day_mode(...)` | Day mode: SCALP, RUNNER, or MIXED |
| `detect_trade_date()` | Auto-detect trade date from current ET time |
| `_load_prior_journal(trade_date)` | Load yesterday's journal for prior session context |
| `_print_prior_session(entry)` | Print prior session outcome block |

### gex.py — GEX Computation Engine

Computes Gamma Exposure maps from option chain data. Produces gamma flip, call/put walls, shelves, air (distance to next node), regime, and inventory zones.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `compute_gex_map(df, spot, ...)` | Core GEX engine: net gamma by strike, flip, walls, shelves, regime |
| `combine_qqq_ndx_gex(qqq_gex, ndx_gex, spot_nq, basis)` | Merge QQQ + NDX GEX maps to unified NQ scale |
| `filter_today_expiry(df, today)` | Filter chain to 0DTE options |
| `compute_flow_walls(df_wide, spot, ...)` | Real-time flow walls from live option volume |
| `combine_flow_walls(qqq_flow, ndx_flow, spot_nq, basis)` | Merge flow walls from both tickers |
| `classify_case(gex, sweep, direction)` | Return CASE_1/2/3 based on sweep + regime + air |

**GEX map output fields:**

| Field | Description |
|-------|-------------|
| `gamma_flip` | Strike where cumulative GEX crosses zero |
| `call_wall` / `put_wall` | Largest positive/negative GEX strike |
| `shelves` | Secondary GEX concentration zones |
| `nearest_above` / `nearest_below` | Next GEX node above/below spot |
| `air_up` / `air_dn` | Distance (pts) from spot to next node |
| `regime` | negative / positive / neutral |
| `inside_inventory` | Boolean: spot within call/put wall range |
| `dist_to_inv` | Distance to nearest inventory boundary |

### ict.py — ICT Level Computation

Computes liquidity levels from NQ 1-minute bars: prior day reference, overnight range, session ranges, and equal highs/lows (clustered liquidity pools).

**Key functions:**

| Function | Purpose |
|----------|---------|
| `compute_ict_levels(df, trade_date)` | Compute all ICT levels for a trading day |
| `load_nq_data(parquet_path)` | Load + normalize NQ 1-min bars |
| `get_all_ict_levels_as_list(ict)` | Flatten to sorted (price, label) list for sweep detection |

**Level types:**

| Level | Session | Description |
|-------|---------|-------------|
| PDH / PDL / PDC | Prior RTH | Prior day high, low, close |
| ONH / ONL | Overnight | Overnight high/low (18:00 prior day - 09:29 today) |
| Asia Hi/Lo | Asia | 18:00 - 00:00 ET |
| London Hi/Lo | London | 03:00 - 08:00 ET |
| Equal Highs/Lows | 3-day scan | Clustered price levels (liquidity pools) |
| ORB 5/15 Hi/Lo | RTH open | Opening range (first 5 or 15 minutes) |

### orb.py — ORB Simulation Engine

Detects sweeps of ICT levels, ORB breaks, entries, and tracks MFE/MAE/targets through the session.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `compute_orb(bars_1m, trade_date, ...)` | Compute ORB high/low/mid from 1-min bars |
| `simulate_orb_day(rth_bars, ict, gex, ...)` | Full day simulation: sweep -> entry -> trade tracking |

**ORB definition:** First 5 bars of RTH (09:30-09:34). Break = first 1-min candle closing beyond the range.

### orats.py — ORATS API Client

Fetches QQQ and NDX option chains from the ORATS Data API with automatic disk caching.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `fetch_chain(ticker, trade_date)` | Pull full option chain (cached) |
| `fetch_qqq_and_ndx(trade_date)` | Pull both QQQ + NDX chains |
| `fetch_live_chain(ticker, expiry)` | Fetch LIVE 0DTE snapshot (real-time) |
| `get_stock_price(ticker, trade_date)` | Extract stock price from chain data |
| `get_prior_trading_date(d)` | Prior trading session (NYSE calendar) |
| `get_next_trading_date(d)` | Next trading session (NYSE calendar) |
| `save_flow_snapshot(...)` | Save live chain snapshot for replay |

**Caching:** All chain pulls are cached to `orats_cache/{TICKER}_{date}.json`. Subsequent requests for the same date hit disk, not the API.

### playbook.py — Pattern Matcher

Loads `playbook.yaml`, evaluates conditions against brief JSON fields, and returns sorted matches. Designed with zero imports from `dashboard.py` to avoid circular dependencies.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `load_playbook(path)` | Load YAML with mtime cache |
| `match_patterns(brief_data)` | Match brief against all patterns (>=50% threshold) |
| `format_playbook_matches(matches, scorecard, decay_warnings)` | Format for terminal with live stats |

**Condition operators:** `eq`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `bool`, `range`

### journal.py — Post-Game Journal

Computes trading outcomes from 1-min bars and saves structured journal entries.

**CLI arguments:**

| Flag | Description |
|------|-------------|
| `--date YYYY-MM-DD` | Trade date (required) |
| `--macro TEXT` | Macro/news context (optional) |
| `--lesson TEXT` | Lesson learned (optional) |

**Computed outcomes per entry:**

| Field | Description |
|-------|-------------|
| `orb_break_dir` | LONG or SHORT (first close beyond ORB range) |
| `orb_break_time` | Time of ORB break (e.g., "09:40") |
| `mfe_from_orb_break` | Maximum Favorable Excursion from break price |
| `mae_from_orb_break` | Maximum Adverse Excursion from break price |
| `t1_hit` | Boolean: did price reach T1 (suggested_target) from break? |
| `runner_hit` | Boolean: did price reach 2x T1? |
| `price_0935` ... `price_eod` | Price checkpoints through the session |
| `day_range` | RTH high - RTH low |

### learn.py — Pattern Learning

Scans journal entries, scores playbook patterns, finds condition correlations, and suggests new patterns.

**CLI arguments:**

| Flag | Description |
|------|-------------|
| `--yaml` | Write `playbook_suggestions.yaml` |
| `--min-n INT` | Minimum observations per condition (default: 1) |

**Key functions:**

| Function | Purpose |
|----------|---------|
| `load_live_intelligence()` | Single-pass loader: returns scorecard + decay warnings for dashboard |
| `load_scorecard_cache()` | Returns scorecard dict keyed by pattern_id |
| `compute_pattern_decay(entries, playbook, recent_n)` | Detect patterns with declining T1 rate |
| `compute_pattern_scorecard(entries, playbook)` | Grade each pattern: T1 rate, MFE, MAE, confidence |
| `compute_condition_correlations(entries)` | Find best-predicting brief conditions |
| `discover_pattern_candidates(entries, ...)` | Auto-discover 2-3 condition pattern combos |

**Outcome scoring formula (0.0 - 1.0):**
- T1 hit: +0.4
- MFE ratio (vs target): +0.3
- MAE containment (vs target): +0.2
- Runner hit: +0.1

**Confidence labels:**

| N (observations) | Label |
|-------------------|-------|
| 0 | NO DATA |
| 1-2 | ANECDOTAL |
| 3-5 | EARLY |
| 6-10 | DEVELOPING |
| 11+ | SOLID |

### earnings_calendar.py — Earnings Checker

Checks if any of the top 10 NDX components (NVDA, MSFT, AAPL, AMZN, META, GOOGL, GOOG, AVGO, TSLA, COST) report earnings on the trade date. These 10 stocks represent ~55% of NDX weighting. When present, GEX maps may be distorted.

### config.py — Configuration

All constants, API keys, session times, and tunable parameters in one place.

**Session windows (Eastern Time):**

| Session | Start | End |
|---------|-------|-----|
| Globex/Overnight | 18:00 (prior day) | 09:29 |
| Asia | 18:00 (prior day) | 00:00 |
| London | 03:00 | 08:00 |
| RTH | 09:30 | 16:00 |
| Pre-market snapshot | 09:25 | — |

**Tunable parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_POINTS` | 100 | ORB trade profit target (NQ points) |
| `STOP_POINTS` | 30 | Max stop loss (NQ points) |
| `SWEEP_BUFFER` | 3 | Points through a level = valid sweep |
| `EQUAL_LEVEL_TOL` | 4 | Tolerance for equal hi/lo clustering |
| `MIN_AIR_POINTS` | 60 | Min air to next GEX node for Case 1 |
| `NEUTRAL_BAND` | 20 | Points around flip = neutral regime |

**VIX regime thresholds and suggested targets:**

| VIX Range | Regime | Suggested Target |
|-----------|--------|------------------|
| < 15 | LOW | 60 pts |
| 15 - 20 | NORMAL | 80 pts |
| 20 - 25 | ELEVATED | 100 pts |
| 25 - 35 | HIGH | 150 pts |
| > 35 | PANIC | 200 pts |

---

## Data Sources & Pipelines

### Data Flow

```
ORATS API ──► orats.py ──► orats_cache/*.json ──► gex.py (GEX maps)
                                                 ├── compute_gex_map()
                                                 └── compute_flow_walls()

yfinance  ──► dashboard.py (live spots, VIX, NQ 1m bars)
           └► ict.py (ICT levels from 1m bars)

Parquet   ──► ict.py (historical ICT levels)
           └► journal.py (outcome computation from 1m bars)
```

### Source Details

| Data | Source | Cost | Cache |
|------|--------|------|-------|
| QQQ + NDX option chains | ORATS Data API | ~$99-249/mo | `orats_cache/` (per date) |
| NQ 1-min OHLCV (live) | yfinance (`NQ=F`) | Free | None (fetched live) |
| NQ 1-min OHLCV (historical) | Parquet files | One-time | `nq_1m.parquet` |
| VIX price | yfinance (`^VIX`) | Free | None |
| QQQ/NDX spots (live) | yfinance (`QQQ`, `^NDX`) | Free | None |
| NYSE trading calendar | `pandas_market_calendars` | Free | In-memory |
| Earnings dates | yfinance ticker info | Free | None |

### Bar Data Cascade

When loading 1-min NQ bars, the system tries sources in order:

1. `archive/` directory (archived parquet files)
2. `nq_1m.parquet` (covers Nov 2024 - Nov 2025)
3. `nq_1m_databento.parquet` (alternative source)
4. yfinance `NQ=F` (live/recent data, ~30 days of 1-min history)

### Data Quality Guards

- `_is_sane_outcome()` in learn.py filters entries with:
  - `day_range > 2000` (bad ticks)
  - `mae > 1000` (invalid data)
  - `orb_high - orb_low > 2000` or `orb_low < 1000` (bad ORB data)
- Known issue: 2025-11-14 parquet data has bad ticks (low=233, high=25777) — filtered automatically

---

## Playbook System

### How It Works

The playbook is a set of human-editable pattern rules in `playbook.yaml`. Each pattern defines conditions that are evaluated against the morning brief's JSON fields. When >=50% of conditions match, the pattern fires.

### Pattern Structure

```yaml
pattern_id:
  name: "Human-readable name"
  priority: HIGH | MEDIUM | LOW          # Sort order in output
  conditions:                            # List of field checks
    - field: regime                      # Brief JSON field name
      op: eq                             # Operator
      value: negative                    # Expected value
    - field: premarket_range
      op: gt
      value: 140
  bias: LONG_BIAS | SHORT_BIAS | null    # Directional bias
  bias_override: false                   # Override brief's computed bias?
  warning: "Short text for alert"        # Printed in brief
  action: "What to do"                   # Actionable instruction
  notes: "Why this pattern matters"      # Context for the trader
```

### Available Fields for Conditions

These fields from the brief JSON can be used in conditions:

| Field | Type | Example Values |
|-------|------|----------------|
| `vix_regime` | enum | LOW, NORMAL, ELEVATED, HIGH, PANIC |
| `regime` | enum | negative, positive, neutral |
| `premarket_bias` | enum | LONG_BIAS, SHORT_BIAS, NEUTRAL |
| `trade_state` | enum | VACUUM_UP, VACUUM_DOWN, INSIDE_INV, OUTSIDE_UP, OUTSIDE_DOWN |
| `mode` | enum | SCALP, RUNNER, MIXED |
| `inside_inventory` | bool | true, false |
| `has_major_earnings` | bool | true, false |
| `runner_permission` | bool | true, false |
| `gap_from_pdc` | numeric | -120.5, 0, 56.8 |
| `vix_price` | numeric | 14.2, 21.45, 32.0 |
| `premarket_range` | numeric | 65.0, 141.2, 220.0 |
| `dist_to_inv` | numeric | 0.0, 45.3, 120.0 |
| `gap_direction` | derived | up, down, flat (derived from gap_from_pdc) |
| `prior_day_range` | derived | PDH - PDL |

### Current Patterns (7)

| ID | Priority | Description |
|----|----------|-------------|
| `gap_up_into_catalyst` | HIGH | Gap up + earnings + wide range = trap risk |
| `day2_macro_selloff` | HIGH | Neg gamma + gap down + elevated VIX = continuation |
| `relief_bounce_after_selloff` | MEDIUM | Neg gamma + gap up + elevated VIX = cautious long |
| `vix_mispriced` | MEDIUM | Low VIX but wide overnight range = vol spike incoming |
| `inside_inventory_neg_gamma` | MEDIUM | Inside GEX walls + neg gamma = overshoot |
| `outside_inventory_neg_gamma` | HIGH | Outside GEX walls + neg gamma = no cushion |
| `runner_wide_range` | HIGH | RUNNER mode + ON range > 140 = 90% T1 hit |

### Matching Logic

1. Each condition is evaluated independently against the enriched brief
2. Match ratio = conditions_met / conditions_total
3. Patterns with ratio >= 0.5 are returned
4. Sorted by: full_match (100%) > priority rank > match ratio descending

### Live Scorecard (Phase 2)

When journal data exists, each matched pattern shows:

```
[HIGH  ] Runner Mode + Wide Overnight Range       2/2 (FULL)
         T1: 9/10 (90%)  |  MFE: 271  |  MAE: 117  |  DEVELOPING
         ** Runner + wide range — T1 hits 90% of the time
```

- **T1: X/N (pct%)** — How often T1 was hit when this pattern matched
- **MFE** — Average maximum favorable excursion
- **MAE** — Average maximum adverse excursion
- **Confidence** — NO DATA / ANECDOTAL / EARLY / DEVELOPING / SOLID

### Pattern Decay Warnings

When a pattern's recent T1 rate (last 5 matches) drops below 60% of its all-time rate:

```
         !! DECAY: T1 dropped from 90% (all-time) to 40% (last 5)
```

This warns you to re-evaluate the pattern before trusting it blindly.

---

## Journal & Learning Loop

### The Feedback Loop

```
Morning Brief  ──►  Trade Session  ──►  Journal Entry  ──►  Learn Report
     ▲                                       │                    │
     │                                       ▼                    ▼
     └──────────── Playbook Updates ◄─── Pattern Discovery ◄─────┘
```

### Journal Entry Structure

```json
{
  "date": "2026-02-24",
  "brief": {
    "spot_nq": 24826.25,
    "vix_regime": "ELEVATED",
    "regime": "negative",
    "mode": "RUNNER",
    "premarket_range": 141.2,
    "suggested_target": 100,
    "..."
  },
  "outcome": {
    "orb_break_dir": "long",
    "orb_break_time": "09:40",
    "mfe_from_orb_break": 240.0,
    "mae_from_orb_break": 38.0,
    "t1_hit": true,
    "runner_hit": true,
    "day_range": 384.25,
    "price_0935": 24736.5,
    "price_0945": 24922.0,
    "price_1000": 24831.25,
    "price_1030": 25016.25,
    "price_1100": 24984.75,
    "price_eod": 25031.5
  },
  "matched_patterns": ["runner_wide_range", "relief_bounce_after_selloff", "..."],
  "macro_context": "tariff fears",
  "lesson": "Relief bounce played out"
}
```

### Learning Report Sections

**1. Pattern Scorecard** — Grades each playbook pattern A through F:

```
  Pattern                             N    T1    MFE      MAE    Grade  Confidence
  Runner Mode + Wide Overnight Range  10  9/10  271.0    117.0      A   DEVELOPING
  Inside Inventory + Neg Gamma         8  6/8   198.0     85.0      B   DEVELOPING
  Relief Bounce After Selloff          7  5/7   198.0     85.0      B   DEVELOPING
```

**2. Condition Correlations** — Which brief fields predict good outcomes:

```
  Condition                        When TRUE      When FALSE    Delta
  gap_direction=up                 0.857 (N=7)    0.584 (N=5)  +0.273
  vix_regime=NORMAL                0.832 (N=4)    0.608 (N=8)  +0.224
```

**3. Pattern Suggestions** — Auto-discovered multi-condition patterns with ready-to-paste YAML.

### Discovered Insights

Notable patterns found through the learning loop:
- `runner_wide_range`: RUNNER mode + premarket_range > 140 has a 90% T1 hit rate (9/10)
- Gap up days outperform gap down (score 0.857 vs 0.584)
- VIX NORMAL outperforms ELEVATED (0.832 vs 0.608)

---

## Output Reference

### Brief JSON (`daily_briefs/brief_YYYY-MM-DD.json`)

The full brief contains ~80+ fields. Key sections:

| Section | Fields | Description |
|---------|--------|-------------|
| Metadata | `status`, `date`, `time_et`, `sim_mode` | Brief type and timing |
| Spots | `spot_nq`, `spot_ndx`, `spot_qqq`, `basis` | Price levels at 09:25 ET |
| VIX | `vix_price`, `vix_regime`, `vix_note`, `suggested_target` | Volatility regime |
| ICT Levels | `PDH`, `PDL`, `PDC`, `ONH`, `ONL` | Prior day + overnight reference |
| GEX (0DTE) | `gex_0dte_regime`, `gex_0dte_flip`, `gex_0dte_call_wall`, `gex_0dte_put_wall` | Same-day expiry GEX |
| GEX (Multi) | `gex_multi_regime`, `gex_multi_flip`, `gex_multi_call_wall`, `gex_multi_put_wall` | Multi-day expiry GEX |
| Combined GEX | `regime`, `inside_inventory`, `dist_to_inv`, `inv_low`, `inv_high` | Merged GEX regime |
| Trade Plan | `trade_state`, `mode`, `premarket_bias`, `confidence`, `if_then_plan` | Actionable guidance |
| Key Levels | `key_levels` (array of `{label, price}`) | Sorted reference levels |
| Playbook | `playbook_matches` (array) | Matched pattern summaries |
| Flow Walls | `flow_call_wall`, `flow_put_wall`, `flow_source` | Real-time option flow |
| Prior Session | `prior_session` (object or null) | Yesterday's outcome summary |
| ORB Plan | `orb_plan` (object, if ORB closed) | Post-ORB execution guidance |

### Combined JSON in Brief (`prior_session` field)

When a prior journal exists:

```json
{
  "prior_session": {
    "date": "2026-02-20",
    "orb_break_dir": "long",
    "day_range": 384.25,
    "t1_hit": true,
    "lesson": "Relief bounce played out"
  }
}
```

### Daily Log CSV (`daily_briefs/daily_log.csv`)

A flattened running CSV of all briefs, with nested dicts expanded using underscore-separated keys. Useful for spreadsheet analysis across multiple trading days.

---

## Trading Concepts & Glossary

### Core Concepts

| Term | Definition |
|------|-----------|
| **ORB** | Opening Range Breakout. The high and low of the first 5 minutes of RTH (09:30-09:34). A "break" occurs when a 1-min candle closes beyond the range. |
| **T1** | First profit target = `suggested_target` points from ORB break price. Varies by VIX regime (60-200 pts). |
| **Runner** | Second profit target = 2x T1. Trail the second half of position after T1 is hit. |
| **MFE** | Maximum Favorable Excursion. The best unrealized profit during a trade (from entry). |
| **MAE** | Maximum Adverse Excursion. The worst unrealized loss during a trade (from entry). |
| **GEX** | Gamma Exposure. Net gamma exposure of market makers at each strike price. |
| **Gamma Flip** | The strike where cumulative GEX crosses zero. Below = negative gamma, above = positive. |
| **Call Wall / Put Wall** | Strikes with the largest positive/negative GEX. Act as magnets/resistance. |
| **Negative Gamma** | Dealers are short gamma. They must buy highs and sell lows, amplifying moves. |
| **Positive Gamma** | Dealers are long gamma. They sell highs and buy lows, dampening moves. |
| **Inventory Zone** | The range between call wall and put wall. Inside = dealer cushion exists. |
| **Air** | The distance from spot to the next GEX node. More air = more room to run. |
| **Sweep** | Price trades through an ICT level by a few points, then closes back. An engineered stop run. |
| **Vacuum** | Large air gap in one direction. Price tends to fill vacuums rapidly. |

### The 3 Case Types

| Case | Condition | Edge | Action |
|------|-----------|------|--------|
| **Case 1** | Sweep + negative gamma + air to next node | HIGH | Press — full target, trail for runner |
| **Case 2** | Sweep + positive gamma + wall nearby | LOW | Fade or skip — wall absorbs move |
| **Case 3** | No sweep, drift between walls | NONE | Skip or scalp extremes only |

### Trade States

| State | Meaning |
|-------|---------|
| `VACUUM_UP` | Large air gap above, price likely to fill upward |
| `VACUUM_DOWN` | Large air gap below, price likely to fill downward |
| `INSIDE_INV` | Spot within call/put wall range, dealer cushion exists |
| `OUTSIDE_UP` | Spot above call wall, no upside cushion |
| `OUTSIDE_DOWN` | Spot below put wall, no downside cushion |
| `WALL_PINNED` | Spot near a major wall, expect pinning |
| `NEG_GAMMA_TREND` | Negative gamma with momentum, moves overshoot |

### Day Modes

| Mode | Condition | Implication |
|------|-----------|-------------|
| `RUNNER` | Wide range + air + neg gamma | Full T1 + trail for runner |
| `SCALP` | Tight range or pos gamma or pinned | T1 only, no runner attempt |
| `MIXED` | Moderate conditions | Read price action at T1 to decide |

---

## Backtesting

### Running a Full Backtest

```bash
python backtest.py
```

This runs the ORB simulation over every trading day in the parquet file, computing GEX + ICT + sweep detection + entry + MFE/MAE for each day.

**Output files in `backtest_output/`:**

| File | Description |
|------|-------------|
| `backtest_all.csv` | All results (5m + 15m ORB combined) |
| `backtest_5m.csv` | 5-minute ORB results only |
| `backtest_15m.csv` | 15-minute ORB results only |
| `skipped_days.csv` | Days with missing data or errors |

**Key backtest columns:**

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `orb_minutes` | ORB window (5 or 15) |
| `sweep_occurred` | Boolean: ICT level sweep detected? |
| `sweep_direction` | LONG or SHORT |
| `day_type` | CASE_1, CASE_2, or CASE_3 |
| `trade_signal` | Whether entry criteria were met |
| `entry_price` / `entry_dir` | Entry details |
| `mfe` / `mae` / `pnl` | Trade metrics |
| `hit_target` / `hit_stop` | Boolean outcomes |
| `session_range` | Full RTH range |

### FXReplay Workflow

1. Run `python dashboard.py --date YYYY-MM-DD` to get the brief
2. Open FXReplay and load the trade date
3. Draw levels from the brief (gamma flip, call/put walls, ONH/ONL, PDH/PDL)
4. Replay the ORB session bar-by-bar
5. Execute based on sweep + reclaim rule
6. Compare your execution to the backtest's theoretical outcome

---

## Troubleshooting

### Common Issues

**"No brief found" when running journal:**
```
[ERROR] No brief found at daily_briefs/brief_2026-02-24.json
```
Run the brief first: `python dashboard.py --date 2026-02-24`

**ORATS API errors:**
- Verify `ORATS_TOKEN` is set: `echo $ORATS_TOKEN`
- Check if your ORATS subscription is active
- Cached chains work offline — only new dates need API access

**yfinance returning empty data:**
- yfinance 1-min data is limited to ~30 days of history
- For older dates, use parquet files or Databento
- Weekend/holiday dates will return no data

**Playbook matches showing "NO DATA":**
- No journal entries exist yet for that pattern
- Run `python journal.py --date YYYY-MM-DD` for past dates to build history
- Scorecard requires at least 1 journal entry where the pattern matched

**Pattern decay not triggering:**
- Needs at least `recent_n + 2` (7 by default) matches for a pattern
- Only triggers when recent T1 rate drops below 60% of all-time rate

**Bad data in learn.py output:**
- `_is_sane_outcome()` filters extreme values automatically
- Known issue: 2025-11-14 parquet has bad ticks — filtered out
- If you see suspicious data, check the journal JSON manually

### Files Excluded from Git

The `.gitignore` excludes generated/cached data:
```
__pycache__/
*.pyc
.DS_Store
orats_cache/          # Cached API responses
nq_1m.parquet         # Bar data files
nq_1m_databento.parquet
backtest_output/      # Backtest results
daily_briefs/         # Generated briefs
archive/              # Archived bar data
journal/              # Journal entries
```

These directories are auto-created as needed. The journal and brief directories contain your trading data — back them up separately if needed.
