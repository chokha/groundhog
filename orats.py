"""
orats.py — Fetch QQQ and NDX option chains from ORATS Data API
Handles chain pull, field normalization, and basic caching to disk.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

from config import ORATS_TOKEN, ORATS_BASE, QQQ_TICKER, NDX_TICKER

# ─── Local cache dir (avoids re-hitting API for same dates) ──────────────────
CACHE_DIR = Path("orats_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, trade_date: str) -> Path:
    return CACHE_DIR / f"{ticker}_{trade_date}.json"


def _orats_get(endpoint: str, params: dict) -> dict:
    """Raw ORATS GET with auth injected."""
    if ORATS_TOKEN == "YOUR_ORATS_TOKEN_HERE":
        raise RuntimeError("Set ORATS_TOKEN in config.py or as environment variable.")
    p = dict(params)
    p["token"] = ORATS_TOKEN
    r = requests.get(f"{ORATS_BASE}/{endpoint}", params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_chain(ticker: str, trade_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Pull full option chain for ticker on trade_date (YYYY-MM-DD).
    Returns DataFrame with normalized columns:
        strike, expiry, putCall, openInterest, gamma, delta, iv, dte

    ORATS endpoint used: strikes/options
    Verify field names match your subscription tier at:
        https://docs.orats.io/datav2-api-guide/data.html
    """
    cache_file = _cache_path(ticker, trade_date)

    # ── Return cached if available ────────────────────────────────────────────
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            raw = json.load(f)
        return _normalize_chain(pd.DataFrame(raw))

    # ── Hit API ───────────────────────────────────────────────────────────────
    try:
        raw = _orats_get("hist/strikes", {
            "ticker":    ticker,
            "tradeDate": trade_date,
        })
    except requests.HTTPError as e:
        print(f"[ORATS] HTTP error for {ticker} {trade_date}: {e}")
        return pd.DataFrame()

    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not data:
        print(f"[ORATS] No data returned for {ticker} {trade_date}")
        return pd.DataFrame()

    # ── Cache to disk ─────────────────────────────────────────────────────────
    if use_cache:
        with open(cache_file, "w") as f:
            json.dump(data, f)

    return _normalize_chain(pd.DataFrame(data))


def _normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpivot ORATS wide format → long format.

    ORATS returns one row per strike with separate call/put columns:
        strike, expirDate, dte, stockPrice,
        callOpenInterest, putOpenInterest,
        gamma (call gamma — same abs value for puts),
        delta (call delta — put delta = call delta - 1)

    We produce long-format rows with:
        strike, expiry, dte, putCall, openInterest, gamma, delta
    """
    if df.empty:
        return df

    # Rename expirDate → expiry if present
    if "expirDate" in df.columns:
        df = df.rename(columns={"expirDate": "expiry"})

    # Coerce numerics
    for col in ["strike", "dte", "gamma", "delta", "callOpenInterest", "putOpenInterest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── CALL rows ──────────────────────────────────────────────────────────────
    calls = df[["strike", "expiry", "dte", "callOpenInterest", "gamma", "delta"]].copy()
    calls = calls.rename(columns={"callOpenInterest": "openInterest"})
    calls["putCall"] = "C"

    # ── PUT rows ───────────────────────────────────────────────────────────────
    puts = df[["strike", "expiry", "dte", "putOpenInterest", "gamma", "delta"]].copy()
    puts = puts.rename(columns={"putOpenInterest": "openInterest"})
    puts["putCall"] = "P"
    puts["delta"] = puts["delta"] - 1.0

    # ── Concatenate and filter ─────────────────────────────────────────────────
    long = pd.concat([calls, puts], ignore_index=True)
    long = long[long["openInterest"] != 0]

    # Drop effectively-zero-gamma rows (dte=0 kept for 0DTE maps)
    long = long[long["gamma"] >= 1e-10]

    return long[["strike", "expiry", "dte", "putCall", "openInterest", "gamma", "delta"]]


def get_stock_price(ticker: str, trade_date: str, use_cache: bool = True) -> float:
    """
    Extract stockPrice from ORATS chain data for a ticker/date.
    Returns the closing stock price ORATS recorded for that trade date.
    """
    cache_file = _cache_path(ticker, trade_date)

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            raw = json.load(f)
    else:
        try:
            result = _orats_get("hist/strikes", {
                "ticker": ticker,
                "tradeDate": trade_date,
            })
            raw = result.get("data", result) if isinstance(result, dict) else result
        except Exception:
            return None

    if raw and isinstance(raw, list) and len(raw) > 0:
        return float(raw[0].get("stockPrice", 0))
    return None


def fetch_qqq_and_ndx(trade_date: str, use_cache: bool = True) -> dict:
    """
    Pull both QQQ and NDX chains for a given date.
    Returns: {"qqq": DataFrame, "ndx": DataFrame}
    """
    return {
        "qqq": fetch_chain(QQQ_TICKER, trade_date, use_cache=use_cache),
        "ndx": fetch_chain(NDX_TICKER, trade_date, use_cache=use_cache),
    }


def get_prior_trading_date(d: date) -> date:
    """Return prior trading session using NYSE calendar (handles holidays)."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    start = d - timedelta(days=14)
    schedule = nyse.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=d.strftime("%Y-%m-%d"),
    )
    sessions = [s for s in schedule.index.date.tolist() if s < d]
    return sessions[-1] if sessions else d - timedelta(days=1)


def get_next_trading_date(d: date) -> date:
    """Return next trading session using NYSE calendar (handles holidays/weekends)."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    end = d + timedelta(days=14)
    schedule = nyse.schedule(
        start_date=d.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    sessions = [s for s in schedule.index.date.tolist() if s > d]
    return sessions[0] if sessions else d + timedelta(days=1)


if __name__ == "__main__":
    # Quick test — replace with a real recent date
    test_date = "2025-11-14"
    print(f"Testing ORATS fetch for QQQ on {test_date}...")
    df = fetch_chain("QQQ", test_date)
    if not df.empty:
        n_calls = (df["putCall"] == "C").sum()
        n_puts  = (df["putCall"] == "P").sum()
        print(f"  Call rows: {n_calls}   Put rows: {n_puts}   Total: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Top 10 by openInterest where OI > 100
        big_oi = df[df["openInterest"] > 100].nlargest(10, "openInterest")
        print(f"\n  Top 10 rows by OI (where OI > 100):")
        print(big_oi.to_string(index=False))

        # Total OI by side
        call_oi = df.loc[df["putCall"] == "C", "openInterest"].sum()
        put_oi  = df.loc[df["putCall"] == "P", "openInterest"].sum()
        print(f"\n  Total Call OI: {call_oi:,.0f}   Total Put OI: {put_oi:,.0f}")
    else:
        print("  No data — check your ORATS_TOKEN and subscription plan.")
