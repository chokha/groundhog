# NQ ORB System — GEX + ICT Backtest & Daily Brief

## What This Is

A two-phase system for trading NQ ORB setups:
1. **Backtest** — validates the edge statistically over historical data
2. **Daily Brief** — pre-market level map for live trading and FXReplay practice

---

## Directory Structure

```
nq_orb_system/
  config.py       # All constants, API keys, parameters
  orats.py        # ORATS API client (QQQ + NDX chains)
  gex.py          # GEX computation (flip, walls, air, regime)
  ict.py          # ICT level computation (PDH/PDL, ONH/ONL, Asia, London, equal hi/lo)
  orb.py          # ORB simulation (sweep detection, entry, MFE/MAE)
  backtest.py     # Main backtest loop
  dashboard.py    # Daily pre-market brief generator
  orats_cache/    # Auto-created — cached ORATS chain pulls
  backtest_output/ # Auto-created — backtest CSVs
  daily_briefs/   # Auto-created — daily JSON + log CSV
```

---

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy requests pyarrow
```

### 2. Get ORATS API key
- Sign up at https://orats.com → Data API → Research plan
- Set your token:
  ```bash
  export ORATS_TOKEN=your_token_here
  ```
  Or paste directly into `config.py`

### 3. Place NQ parquet file
- Copy your `nq_1m.parquet` to this directory
- Or update `NQ_PARQUET_PATH` in `config.py` with full path

---

## Running the Backtest

```bash
cd nq_orb_system
python backtest.py nq_1m.parquet
```

**Output files:**
- `backtest_output/backtest_all.csv` — all results (both ORB timeframes)
- `backtest_output/backtest_5m.csv` — 5-min ORB only
- `backtest_output/backtest_15m.csv` — 15-min ORB only
- `backtest_output/skipped_days.csv` — days that couldn't run

---

## Running the Daily Brief

```bash
# At 9:15–9:25 ET, get NQ/NDX/QQQ spot prices from your broker, then:
python dashboard.py --spot-nq 25120.25 --spot-ndx 25082.30 --spot-qqq 498.12 nq_1m.parquet
```

**Output:**
- Terminal printout of full level map and trade signal
- `daily_briefs/brief_YYYY-MM-DD.json` — all levels for FXReplay import
- `daily_briefs/daily_log.csv` — running log of all briefs

---

## FXReplay Workflow

1. Run `dashboard.py` at 9:15 ET → get brief JSON
2. Open FXReplay → load trade_date
3. Manually draw levels from brief (2 min):
   - Gamma Flip, Call Wall, Put Wall
   - ONH, ONL, PDH, PDL
   - London Hi/Lo, Equal Hi/Lo if relevant
4. Replay ORB session
5. Execute based on sweep + reclaim rule
6. Compare your execution to backtest theoretical

---

## The 3 Case Types

| Case | Condition | Action |
|------|-----------|--------|
| Case 1 | Sweep + negative gamma + air to next node | PRESS — 100 pt target |
| Case 2 | Sweep + positive gamma + wall nearby | FADE or skip |
| Case 3 | No sweep, drift between walls | SKIP or scalp extremes |

---

## Tunable Parameters (config.py)

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `TARGET_POINTS` | 100 | ORB trade target in NQ points |
| `STOP_POINTS` | 30 | Max stop in NQ points |
| `SWEEP_BUFFER` | 3 | Points through level = valid sweep |
| `MIN_AIR_POINTS` | 60 | Min air to next node for Case 1 |
| `EQUAL_LEVEL_TOL` | 4 | Tolerance for equal hi/lo clustering |
| `NEUTRAL_BAND` | 20 | Points around flip = neutral regime |

---

## Data Sources

| Data | Source | Cost |
|------|--------|------|
| QQQ + NDX option chains | ORATS API | ~$99-249/mo |
| NQ 1-min OHLCV | Your existing parquet | Free |
| Live pre-market spot | IBKR / Tradovate API | (you have this) |

---

## Verifying ORATS Field Names

ORATS field names may differ slightly by subscription tier.
Run this to check what your plan returns:

```python
from orats import fetch_chain
df = fetch_chain("QQQ", "2025-11-14")
print(df.columns.tolist())
print(df.head())
```

If column names differ from expected, update `_normalize_chain()` in `orats.py`.
