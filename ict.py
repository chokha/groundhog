"""
ict.py — Compute ICT liquidity levels from NQ 1-min OHLCV data

Levels computed:
    PDH / PDL / PDC     prior day high, low, close
    ONH / ONL           overnight (globex) high, low
    Asia Hi / Lo        18:00–00:00 ET
    London Hi / Lo      03:00–08:00 ET
    Equal Highs/Lows    clustered liquidity pools near current price
    ORB Hi / Lo         5-min and 15-min opening range
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, time, timedelta
from typing import Optional

from config import (
    GLOBEX_START, GLOBEX_END,
    ASIA_START, ASIA_END,
    LONDON_START, LONDON_END,
    RTH_OPEN, RTH_CLOSE,
    ORB_5_MINUTES, ORB_15_MINUTES,
    EQUAL_LEVEL_TOL,
)


def get_spots(df: pd.DataFrame, trade_date: date) -> dict:
    """
    Get spot prices at 9:25 ET for GEX computation.
    Returns NQ spot (from parquet), QQQ and NDX derived.
    """
    snap = df[
        (df["date"] == trade_date) &
        (df["ts"].dt.time >= time(9, 20)) &
        (df["ts"].dt.time <= time(9, 29))
    ]
    spot_nq = float(snap["close"].iloc[-1]) if not snap.empty else None

    # QQQ ≈ NQ / 40
    spot_qqq = round(spot_nq / 40.0, 2) if spot_nq else None

    # NDX ≈ NQ / 1.0004 (NQ trades at slight premium to NDX cash)
    spot_ndx = round(spot_nq / 1.0004, 2) if spot_nq else None

    return {
        "spot_nq":  spot_nq,
        "spot_qqq": spot_qqq,
        "spot_ndx": spot_ndx,
        "basis":    round(spot_nq - spot_ndx, 2) if spot_nq else None,
    }


def load_nq_data(parquet_path: str) -> pd.DataFrame:
    """
    Load NQ 1-min parquet and normalize.
    Expected columns: timestamp/datetime, open, high, low, close, volume
    Returns DataFrame indexed by timezone-naive ET datetime.
    """
    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]

    # Normalize timestamp column — prefer 'ts' first
    ts_col = None
    for name in ["ts", "timestamp", "datetime", "time"]:
        if name in df.columns:
            ts_col = name
            break
    if ts_col is None and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")

    if ts_col and ts_col != "ts":
        df["ts"] = pd.to_datetime(df[ts_col])
    elif ts_col == "ts":
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df["ts"] = df.index

    # Explicit timezone handling — never assume
    if df["ts"].dt.tz is not None:
        # tz-aware: convert to ET then strip
        df["ts"] = df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    # tz-naive: assume already in ET (e.g. Databento data converted during parquet creation)

    df = df.sort_values("ts").reset_index(drop=True)

    # Use existing 'date' column if present, otherwise compute it
    if "date" not in df.columns:
        df["date"] = df["ts"].dt.date

    # Ensure OHLCV columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df


def compute_ict_levels(df: pd.DataFrame, trade_date: date) -> dict:
    """
    Compute all ICT liquidity levels for trade_date.
    
    Args:
        df:           Full NQ 1-min DataFrame (multiple days)
        trade_date:   The date you're computing levels FOR (the trading day)
    
    Returns dict of all ICT levels.
    """
    td     = trade_date
    td_str = str(td)

    # ── Prior RTH session (the day before trade_date's RTH) ──────────────────
    prior_rth = _get_prior_rth(df, td)

    # ── Overnight session: prior day 18:00 → trade_date 09:29 ────────────────
    overnight = _get_overnight(df, td)

    # ── Asia range: prior 18:00 → 00:00 ──────────────────────────────────────
    asia = _get_session(df, td, ASIA_START, ASIA_END, prior_day=True)

    # ── London range: trade_date 03:00 → 08:00 ────────────────────────────────
    london = _get_session(df, td, LONDON_START, LONDON_END, prior_day=False)

    # ── ORB ranges (computed from RTH data live, not pre-market) ─────────────
    # These are passed in later after open; return None here as placeholder
    rth_today = df[
        (df["date"] == td) &
        (df["ts"].dt.time >= RTH_OPEN) &
        (df["ts"].dt.time <= RTH_CLOSE)
    ].copy()

    orb_5  = _compute_orb(rth_today, ORB_5_MINUTES)
    orb_15 = _compute_orb(rth_today, ORB_15_MINUTES)

    # ── Pre-open spot (9:25 ET candle for GEX computation) ───────────────────
    premarket_candles = df[
        (df["date"] == td) &
        (df["ts"].dt.time >= time(9, 20)) &
        (df["ts"].dt.time <= time(9, 29))
    ]
    spot_premarket = float(premarket_candles["close"].iloc[-1]) if not premarket_candles.empty else None

    # ── Equal highs/lows near current area ───────────────────────────────────
    # Use ONH/ONL range as the "near area" to scan
    eq_highs, eq_lows = _find_equal_levels(df, td)

    return {
        # Prior day levels
        "PDH":          _safe_val(prior_rth, "high", "max"),
        "PDL":          _safe_val(prior_rth, "low",  "min"),
        "PDC":          _safe_val(prior_rth, "close", "last"),

        # Overnight levels
        "ONH":          _safe_val(overnight, "high",  "max"),
        "ONL":          _safe_val(overnight, "low",   "min"),

        # Session ranges
        "asia_hi":      _safe_val(asia,   "high",  "max"),
        "asia_lo":      _safe_val(asia,   "low",   "min"),
        "london_hi":    _safe_val(london, "high",  "max"),
        "london_lo":    _safe_val(london, "low",   "min"),

        # ORB levels (5-min and 15-min)
        "orb5_hi":      orb_5["hi"],
        "orb5_lo":      orb_5["lo"],
        "orb5_mid":     orb_5["mid"],
        "orb15_hi":     orb_15["hi"],
        "orb15_lo":     orb_15["lo"],
        "orb15_mid":    orb_15["mid"],

        # Liquidity pools
        "equal_highs":  eq_highs,    # list of price levels
        "equal_lows":   eq_lows,

        # Pre-market spot for GEX compute
        "spot_925":     spot_premarket,

        # RTH open price
        "rth_open":     float(rth_today["open"].iloc[0]) if not rth_today.empty else None,
    }


def get_all_ict_levels_as_list(ict: dict) -> list:
    """
    Flatten ICT dict into a sorted list of (price, label) tuples.
    Useful for sweep detection.
    """
    levels = []
    scalar_keys = [
        ("PDH", "PDH"), ("PDL", "PDL"), ("PDC", "PDC"),
        ("ONH", "ONH"), ("ONL", "ONL"),
        ("asia_hi", "AsiaHi"), ("asia_lo", "AsiaLo"),
        ("london_hi", "LonHi"), ("london_lo", "LonLo"),
        ("orb5_hi", "ORB5Hi"), ("orb5_lo", "ORB5Lo"),
        ("orb15_hi", "ORB15Hi"), ("orb15_lo", "ORB15Lo"),
    ]
    for key, label in scalar_keys:
        val = ict.get(key)
        if val is not None:
            levels.append((float(val), label))

    for price in ict.get("equal_highs", []):
        levels.append((float(price), "EqHi"))
    for price in ict.get("equal_lows", []):
        levels.append((float(price), "EqLo"))

    return sorted(levels, key=lambda x: x[0])


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_prior_rth(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Get prior regular trading hours session bars."""
    # Find the most recent RTH session before trade_date
    prior_dates = sorted([d for d in df["date"].unique() if d < trade_date], reverse=True)
    if not prior_dates:
        return pd.DataFrame()
    prior = prior_dates[0]
    return df[
        (df["date"] == prior) &
        (df["ts"].dt.time >= RTH_OPEN) &
        (df["ts"].dt.time <= RTH_CLOSE)
    ]


def _get_overnight(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """
    Overnight = prior day 18:00 → trade_date 09:29.
    """
    prior_day = trade_date - timedelta(days=1)
    # Include weekends: go back to last weekday
    while prior_day.weekday() > 4:
        prior_day -= timedelta(days=1)

    mask = (
        (
            (df["date"] == prior_day) &
            (df["ts"].dt.time >= GLOBEX_START)
        ) |
        (
            (df["date"] == trade_date) &
            (df["ts"].dt.time < RTH_OPEN)
        )
    )
    return df[mask]


def _get_session(
    df: pd.DataFrame,
    trade_date: date,
    start_time: time,
    end_time: time,
    prior_day: bool = False
) -> pd.DataFrame:
    """Get bars for a specific time window."""
    d = trade_date - timedelta(days=1) if prior_day else trade_date
    if end_time == time(0, 0):
        # midnight boundary
        mask = (df["date"] == d) & (df["ts"].dt.time >= start_time)
    else:
        mask = (
            (df["date"] == d) &
            (df["ts"].dt.time >= start_time) &
            (df["ts"].dt.time < end_time)
        )
    return df[mask]


def _compute_orb(rth_bars: pd.DataFrame, minutes: int) -> dict:
    """Compute opening range high/low for first N minutes of RTH."""
    if rth_bars.empty:
        return {"hi": None, "lo": None, "mid": None}
    orb = rth_bars.head(minutes)
    hi  = float(orb["high"].max())
    lo  = float(orb["low"].min())
    return {
        "hi":  hi,
        "lo":  lo,
        "mid": round((hi + lo) / 2, 2),
    }


def _find_equal_levels(df: pd.DataFrame, trade_date: date) -> tuple:
    """
    Scan last 3 days of highs/lows for clusters within EQUAL_LEVEL_TOL.
    Returns (equal_highs list, equal_lows list) as price levels.
    """
    recent_dates = sorted([d for d in df["date"].unique() if d <= trade_date])[-4:-1]
    if not recent_dates:
        return [], []

    recent = df[df["date"].isin(recent_dates)]
    rth    = recent[
        (recent["ts"].dt.time >= RTH_OPEN) &
        (recent["ts"].dt.time <= RTH_CLOSE)
    ]

    # Get all swing highs and lows (simple local extrema on 5-min bars)
    rth_5m = rth.resample("5min", on="ts").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()

    highs = rth_5m["high"].values
    lows  = rth_5m["low"].values

    eq_highs = _cluster_levels(highs)
    eq_lows  = _cluster_levels(lows)

    return eq_highs, eq_lows


def _cluster_levels(prices: np.ndarray, tol: float = EQUAL_LEVEL_TOL) -> list:
    """
    Find price clusters within tolerance — these are liquidity pools.
    Returns list of cluster centroid prices where 2+ prices cluster.
    """
    if len(prices) == 0:
        return []

    sorted_p = np.sort(prices)
    clusters = []
    current  = [sorted_p[0]]

    for p in sorted_p[1:]:
        if p - current[-1] <= tol:
            current.append(p)
        else:
            if len(current) >= 2:
                clusters.append(round(float(np.mean(current)), 2))
            current = [p]

    if len(current) >= 2:
        clusters.append(round(float(np.mean(current)), 2))

    return clusters


def _safe_val(df: pd.DataFrame, col: str, method: str) -> Optional[float]:
    """Safely extract value from DataFrame."""
    if df.empty or col not in df.columns:
        return None
    try:
        if method == "max":
            return float(df[col].max())
        elif method == "min":
            return float(df[col].min())
        elif method == "last":
            return float(df[col].iloc[-1])
    except Exception:
        return None
