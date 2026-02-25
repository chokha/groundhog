"""
config.py — API keys, constants, session times
Set ORATS_TOKEN in your environment or paste directly here for testing.
"""

import os
from datetime import time

# ─── API Keys ────────────────────────────────────────────────────────────────
ORATS_TOKEN = "56733fcd-8798-4026-8762-2da18b3967c0"
ORATS_BASE  = "https://api.orats.io/datav2"

# ─── Instruments ─────────────────────────────────────────────────────────────
PRIMARY_TICKER   = "NDX"   # NDX is primary — same price scale as NQ futures
SECONDARY_TICKER = "QQQ"   # QQQ for confluence only
QQQ_TICKER  = "QQQ"
NDX_TICKER  = "NDX"
CONTRACT_MULT = 100        # standard options contract multiplier

# ─── Session Windows (all times Eastern) ─────────────────────────────────────
# Overnight / globex session
GLOBEX_START    = time(18, 0)   # prior day 6:00 PM ET
GLOBEX_END      = time(9, 29)   # morning before open

# Asia range (used for liquidity level detection)
ASIA_START      = time(18, 0)
ASIA_END        = time(0, 0)    # midnight ET

# London range
LONDON_START    = time(3, 0)
LONDON_END      = time(8, 0)

# RTH
RTH_OPEN        = time(9, 30)
RTH_CLOSE       = time(16, 0)

# Pre-market GEX snapshot time
PREMARKET_SNAP  = time(9, 25)

# ─── ORB Definitions ─────────────────────────────────────────────────────────
ORB_5_MINUTES   = 5     # bars after open
ORB_15_MINUTES  = 15    # bars after open

# ─── Backtest Parameters ─────────────────────────────────────────────────────
TARGET_POINTS   = 100   # profit target in NQ points
STOP_POINTS     = 30    # max stop in NQ points
SWEEP_BUFFER    = 3     # points — how far through a level = valid sweep
EQUAL_LEVEL_TOL = 4     # points — how close = "equal" high or low
MIN_AIR_POINTS  = 60    # minimum air to next node (NDX points ≈ NQ points)

# ─── Regime Labels ───────────────────────────────────────────────────────────
REGIME_NEGATIVE = "negative"   # open below gamma flip → expansion prone
REGIME_POSITIVE = "positive"   # open above gamma flip → dampened
REGIME_NEUTRAL  = "neutral"    # open within 20 pts of flip

NEUTRAL_BAND    = 20    # points around flip = neutral zone

# ─── Day Type Labels ─────────────────────────────────────────────────────────
CASE_1 = "case_1"   # sweep + neg gamma + air = expansion
CASE_2 = "case_2"   # sweep + pos gamma + wall = trap/reversal
CASE_3 = "case_3"   # no sweep, drift between walls

# ─── NQ Parquet Path ─────────────────────────────────────────────────────────
NQ_PARQUET_PATH = "nq_1m.parquet"   # place file in same dir, or set full path
