"""
dashboard.py — NQ Pre-Market Brief Generator (Two-Phase ORB Guidance)

Two output phases:
  PREMARKET PLAN:      Always printed (~09:25). Premarket context + if/then plan.
  ORB EXECUTION PLAN:  Printed after ORB close (~09:35). Break/hold/retest/fade guidance.

Two modes:
  LIVE:     python dashboard.py
            All data from live APIs (yfinance + ORATS). Zero file dependencies.

  BACKTEST: python dashboard.py --date 2025-11-14 --parquet nq_1m.parquet
            Spots from parquet at 9:25 ET, ICT from parquet, VIX from yfinance
            history, ORATS chains for GEX.
"""

import argparse
import json
import sys
import pandas as pd
import yfinance as yf
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytz

from config import (
    TARGET_POINTS, STOP_POINTS,
    REGIME_NEGATIVE, REGIME_POSITIVE, REGIME_NEUTRAL,
    MIN_AIR_POINTS,
)
from orats import (
    fetch_chain, fetch_qqq_and_ndx, get_stock_price,
    get_prior_trading_date, get_next_trading_date,
    fetch_live_qqq_and_ndx, save_flow_snapshot,
)
from gex import (
    compute_gex_map, combine_qqq_ndx_gex, filter_today_expiry,
    compute_flow_walls, combine_flow_walls,
)
from orb import compute_orb
from ict import load_nq_data, compute_ict_levels, get_all_ict_levels_as_list
from earnings_calendar import check_earnings_today
from playbook import match_patterns, format_playbook_matches
from learn import load_live_intelligence


# ─── VIX classification ─────────────────────────────────────────────────────

def classify_vix(vix_price) -> dict:
    """Classify VIX into regime and compute suggested target."""
    if vix_price is None:
        return {
            "vix_price": None, "vix_regime": "N/A",
            "vix_note": "VIX unavailable", "suggested_target": TARGET_POINTS,
        }

    vix_price = round(float(vix_price), 2)
    if vix_price < 15:
        regime, note, target = "LOW", "Tight ranges, fade breakouts", 60
    elif vix_price < 20:
        regime, note, target = "NORMAL", "Mixed — wait for confluence", 80
    elif vix_price < 25:
        regime, note, target = "ELEVATED", "Case 1 setups appearing", 100
    elif vix_price < 35:
        regime, note, target = "HIGH", "Best Case 1 conditions", 150
    else:
        regime, note, target = "PANIC", "Explosive moves — reduce size", 200

    return {
        "vix_price": vix_price,
        "vix_regime": regime,
        "vix_note": note,
        "suggested_target": target,
    }


# ─── Detect trade date ──────────────────────────────────────────────────────

def detect_trade_date() -> tuple:
    """
    Determine trade_date based on current ET time.
    Returns (trade_date, now_et).

    Before 6:00 AM ET  -> today (pre-market for today's session)
    6:00 AM - 4:00 PM  -> today (active session)
    After 4:00 PM ET   -> next trading day (prep for tomorrow)
    """
    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)
    today = now_et.date()

    if now_et.time() > time(16, 0):
        trade_date = get_next_trading_date(today)
    else:
        trade_date = today

    return trade_date, now_et


# ─── LIVE MODE ───────────────────────────────────────────────────────────────

def get_live_spots() -> dict:
    """
    Fetch live premarket spots via yfinance 1-min bars (prepost=True).
    NDX derived from QQQ_live x median(NDX/QQQ) 20-day ratio.
    """
    print("  Fetching live spots via yfinance...")

    # ── NQ futures — live premarket ──────────────────────
    nq_hist = yf.Ticker("NQ=F").history(period="1d", interval="1m",
                                         prepost=True)
    spot_nq = round(float(nq_hist["Close"].iloc[-1]), 2)

    # ── QQQ — live premarket via 1m bars with prepost ────
    qqq_hist = yf.Ticker("QQQ").history(period="1d", interval="1m",
                                         prepost=True)
    spot_qqq = round(float(qqq_hist["Close"].iloc[-1]), 2)

    # ── NDX proxy — derived from QQQ using 20-day ratio ──
    # NDX/QQQ ratio is stable ~40.8-41.2x
    qqq_20d = yf.Ticker("QQQ").history(period="30d", interval="1d")
    ndx_20d = yf.Ticker("^NDX").history(period="30d", interval="1d")

    if not qqq_20d.empty and not ndx_20d.empty:
        common = qqq_20d.index.intersection(ndx_20d.index)
        if len(common) >= 5:
            ratio = float((ndx_20d.loc[common, "Close"] /
                           qqq_20d.loc[common, "Close"]).median())
        else:
            ratio = 41.0
    else:
        ratio = 41.0

    spot_ndx = round(spot_qqq * ratio, 2)
    basis    = round(spot_nq - spot_ndx, 2)

    # ── VIX ──────────────────────────────────────────────
    vix_hist = yf.Ticker("^VIX").history(period="1d", interval="1m",
                                          prepost=True)
    vix_price = round(float(vix_hist["Close"].iloc[-1]), 2)

    print(f"  NQ:  {spot_nq}  QQQ: {spot_qqq}  "
          f"NDX(est): {spot_ndx}  Ratio: {ratio:.3f}  "
          f"Basis: {basis:+.2f}  VIX: {vix_price}")

    return {
        "spot_nq":  spot_nq,
        "spot_ndx": spot_ndx,
        "spot_qqq": spot_qqq,
        "basis":    basis,
        "vix":      vix_price,
        "ndx_qqq_ratio": ratio,
    }


def get_ict_levels_live(trade_date: date) -> tuple:
    """Fetch NQ=F 1-min history from yfinance and compute ICT levels.
    Returns (ict_dict, nq_bars_df) so bars can be reused for ORB/trap checks."""
    print("  Fetching NQ=F 1-min history for ICT levels...")

    df = yf.Ticker("NQ=F").history(period="5d", interval="1m")
    if df.empty:
        print("  [WARN] No NQ=F 1-min data from yfinance")
        return {}, pd.DataFrame()

    # Convert yfinance format -> ict.py expected format
    df = df.reset_index()
    col_map = {"Datetime": "ts", "Date": "ts",
               "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["ts"].dt.date

    return compute_ict_levels(df, trade_date), df


def run_live_brief():
    """LIVE mode: all data from APIs, zero file dependencies."""
    trade_date, now_et = detect_trade_date()
    prior_date = get_prior_trading_date(trade_date)
    prior_str = str(prior_date)

    print(f"\n  Trade date:  {trade_date}")
    print(f"  Prior date:  {prior_str}")
    print(f"  Current ET:  {now_et.strftime('%H:%M:%S')}")

    # 1. Live spots (prints its own detail line)
    spots = get_live_spots()

    # 2. Earnings check
    print("  Checking earnings calendar...")
    earnings = check_earnings_today(trade_date)
    if earnings["has_major_earnings"]:
        print(f"\n{'=' * 58}")
        print(f"  ⚠️  MAJOR EARNINGS TODAY: {', '.join(earnings['earnings_today'])}")
        print(f"  GEX map is distorted by earnings premium")
        print(f"  RECOMMENDED: SKIP today — trade tomorrow")
        print(f"{'=' * 58}")

    # 3. ICT levels + raw NQ bars from yfinance 1-min history
    ict, nq_bars = get_ict_levels_live(trade_date)

    # 4. ORATS chains (prior-close for inventory context)
    print(f"\n  Fetching ORATS chains for {prior_str}...")
    chains = fetch_qqq_and_ndx(prior_str, use_cache=True)

    # 5. LIVE chain snapshots for flow walls (today's 0DTE expiry)
    td_str = str(trade_date)
    print(f"  Fetching LIVE 0DTE chains for flow walls (expiry {td_str})...")
    live_chains = fetch_live_qqq_and_ndx(td_str)

    # 6. Generate and print
    generate_and_print_brief(spots, ict, chains, trade_date, now_et,
                             earnings=earnings, nq_bars=nq_bars,
                             live_chains=live_chains)


# ─── BACKTEST MODE ───────────────────────────────────────────────────────────

def run_backtest_brief(date_str: str, parquet_path: str):
    """BACKTEST mode: spots from parquet, ICT from parquet, chains from ORATS."""
    trade_date = date.fromisoformat(date_str)
    prior_date = get_prior_trading_date(trade_date)
    prior_str = str(prior_date)

    et = pytz.timezone("America/New_York")

    print(f"\n  Trade date:  {date_str}")
    print(f"  Prior date:  {prior_str}")
    print(f"  Parquet:     {parquet_path}")

    # 1. Load parquet
    df = load_nq_data(parquet_path)

    # 2. Get spots from parquet at 9:25 ET
    snap = df[
        (df["date"] == trade_date) &
        (df["ts"].dt.time >= time(9, 20)) &
        (df["ts"].dt.time <= time(9, 29))
    ]
    if snap.empty:
        print(f"  ERROR: No data for {date_str} in parquet")
        sys.exit(1)

    spot_nq  = round(float(snap["close"].iloc[-1]), 2)
    spot_ndx = round(spot_nq / 1.0004, 2)
    spot_qqq = round(spot_nq / 40.0, 2)
    basis    = round(spot_nq - spot_ndx, 2)

    # 3. VIX from yfinance history
    vix_hist = yf.Ticker("^VIX").history(
        start=date_str,
        end=str(trade_date + timedelta(days=1))
    )
    vix_price = round(float(vix_hist["Close"].iloc[0]), 2) if not vix_hist.empty else None

    spots = {
        "spot_nq":  spot_nq,
        "spot_ndx": spot_ndx,
        "spot_qqq": spot_qqq,
        "basis":    basis,
        "vix":      vix_price,
    }
    vix_str = f"{vix_price:.2f}" if vix_price else "N/A"
    print(f"  NQ:  {spot_nq:.2f}  NDX: {spot_ndx:.2f}  "
          f"QQQ: {spot_qqq:.2f}  VIX: {vix_str}")

    # 4. Earnings check
    print("  Checking earnings calendar...")
    earnings = check_earnings_today(trade_date)
    if earnings["has_major_earnings"]:
        print(f"\n{'=' * 58}")
        print(f"  ⚠️  MAJOR EARNINGS TODAY: {', '.join(earnings['earnings_today'])}")
        print(f"  GEX map is distorted by earnings premium")
        print(f"  RECOMMENDED: SKIP today — trade tomorrow")
        print(f"{'=' * 58}")

    # 5. ICT from parquet
    ict = compute_ict_levels(df, trade_date)

    # 6. ORATS chains (historical)
    print(f"\n  Fetching ORATS chains for {prior_str}...")
    chains = fetch_qqq_and_ndx(prior_str, use_cache=True)

    # 7. Generate and print (use 9:25 ET as brief time)
    bt_time = et.localize(datetime.combine(trade_date, time(9, 25)))
    generate_and_print_brief(spots, ict, chains, trade_date, bt_time,
                             earnings=earnings, nq_bars=df)


# ─── SIM MODE (backtest simulation) ──────────────────────────────────────────

def _prepare_yf_bars(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance 1m DataFrame to ICT-compatible format (tz-naive ET)."""
    df = df_raw.reset_index()
    col_map = {"Datetime": "ts", "Date": "ts",
               "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    return df


_ARCHIVE_DIR = Path(__file__).parent / "archive"
_ARCHIVE_FILES = {
    "NQ=F":  "nq_1m_archive.parquet",
    "QQQ":   "qqq_1m_archive.parquet",
    "^VIX":  "vix_1m_archive.parquet",
}

def _load_archive(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load archived 1m bars for a ticker, filtered to [start, end)."""
    fname = _ARCHIVE_FILES.get(ticker)
    if not fname:
        return pd.DataFrame()
    path = _ARCHIVE_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"])
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    df = df[(df["ts"] >= start_dt) & (df["ts"] < end_dt)]
    if "date" not in df.columns:
        df["date"] = df["ts"].dt.date
    return df


def _clamp_bars(df: pd.DataFrame, cutoff_et) -> pd.DataFrame:
    """Filter bars to ts <= cutoff_et.  Handles tz-aware cutoff with tz-naive ts column."""
    if hasattr(cutoff_et, "tzinfo") and cutoff_et.tzinfo is not None:
        cutoff = cutoff_et.replace(tzinfo=None)
    else:
        cutoff = cutoff_et
    return df[df["ts"] <= cutoff].copy()


def run_sim_brief(date_str: str = None, time_str: str = None):
    """SIM mode: yfinance bars clamped to simulated timestamp, no flow walls."""
    et = pytz.timezone("America/New_York")

    # ── Parse trade_date + sim time ───────────────────────────────────────────
    if date_str:
        trade_date = date.fromisoformat(date_str)
    else:
        trade_date = datetime.now(et).date()

    if time_str:
        h, m = time_str.split(":")
        sim_time = time(int(h), int(m))
    else:
        sim_time = time(9, 25)

    sim_et = et.localize(datetime.combine(trade_date, sim_time))
    prior_date = get_prior_trading_date(trade_date)
    prior_str = str(prior_date)

    # Date range for yfinance 1m fetch (need prior day for ICT + trade day)
    fetch_start = str(prior_date - timedelta(days=3))
    fetch_end = str(trade_date + timedelta(days=1))

    print(f"\n  Trade date:    {trade_date}")
    print(f"  Sim time (ET): {sim_et.strftime('%H:%M')}")
    print(f"  Prior date:    {prior_str}")
    print(f"  yfinance 1m range: {fetch_start} → {fetch_end}")

    # ── NQ=F 1m bars, clamped ─────────────────────────────────────────────────
    print("  Fetching NQ=F 1-min bars (clamping to sim time)...")
    nq_raw = yf.Ticker("NQ=F").history(start=fetch_start, end=fetch_end,
                                        interval="1m")
    if not nq_raw.empty:
        nq_bars = _prepare_yf_bars(nq_raw)
    else:
        print(f"  yfinance empty — trying archive...")
        nq_bars = _load_archive("NQ=F", fetch_start, fetch_end)
        if nq_bars.empty:
            print(f"  [ERROR] No NQ=F 1-min data from yfinance or archive for {fetch_start}..{fetch_end}")
            print(f"  Run: python archive_bars.py  (daily, before data expires)")
            sys.exit(1)
        print(f"  Loaded {len(nq_bars)} bars from archive")
    nq_bars = _clamp_bars(nq_bars, sim_et)
    if nq_bars.empty:
        print(f"  [ERROR] No NQ bars at or before {sim_et.strftime('%H:%M')} ET on {trade_date}")
        print(f"  Bars fetched: {len(_prepare_yf_bars(nq_raw))} rows, "
              f"date range: {_prepare_yf_bars(nq_raw)['date'].min()} – {_prepare_yf_bars(nq_raw)['date'].max()}")
        sys.exit(1)

    spot_nq = round(float(nq_bars["close"].iloc[-1]), 2)

    # ── QQQ 1m bars, clamped ─────────────────────────────────────────────────
    print("  Fetching QQQ 1-min bars (clamping to sim time)...")
    qqq_raw = yf.Ticker("QQQ").history(start=fetch_start, end=fetch_end,
                                        interval="1m", prepost=True)
    if not qqq_raw.empty:
        qqq_bars = _prepare_yf_bars(qqq_raw)
    else:
        print(f"  yfinance empty — trying archive...")
        qqq_bars = _load_archive("QQQ", fetch_start, fetch_end)
        if not qqq_bars.empty:
            print(f"  Loaded {len(qqq_bars)} bars from archive")

    if not qqq_raw.empty or not qqq_bars.empty:
        qqq_bars = _clamp_bars(qqq_bars, sim_et)
        if not qqq_bars.empty:
            spot_qqq = round(float(qqq_bars["close"].iloc[-1]), 2)
        else:
            spot_qqq = round(spot_nq / 41.0, 2)
    else:
        spot_qqq = round(spot_nq / 41.0, 2)

    # ── NDX from QQQ ratio ────────────────────────────────────────────────────
    qqq_20d = yf.Ticker("QQQ").history(period="30d", interval="1d")
    ndx_20d = yf.Ticker("^NDX").history(period="30d", interval="1d")
    if not qqq_20d.empty and not ndx_20d.empty:
        common = qqq_20d.index.intersection(ndx_20d.index)
        ratio = float((ndx_20d.loc[common, "Close"] /
                        qqq_20d.loc[common, "Close"]).median()) if len(common) >= 5 else 41.0
    else:
        ratio = 41.0

    spot_ndx = round(spot_qqq * ratio, 2)
    basis = round(spot_nq - spot_ndx, 2)

    # ── VIX — try 1m clamped, fall back to archive, then daily, then N/A ─────
    vix_price = None
    try:
        vix_raw = yf.Ticker("^VIX").history(start=fetch_start, end=fetch_end,
                                             interval="1m", prepost=True)
        if not vix_raw.empty:
            vix_bars = _prepare_yf_bars(vix_raw)
        else:
            print(f"  VIX yfinance empty — trying archive...")
            vix_bars = _load_archive("^VIX", fetch_start, fetch_end)
            if not vix_bars.empty:
                print(f"  Loaded {len(vix_bars)} VIX bars from archive")
        if not vix_bars.empty:
            vix_bars = _clamp_bars(vix_bars, sim_et)
            if not vix_bars.empty:
                vix_price = round(float(vix_bars["close"].iloc[-1]), 2)
        # Fall back to daily close if 1m unavailable
        if vix_price is None:
            vix_daily = yf.Ticker("^VIX").history(
                start=str(trade_date), end=str(trade_date + timedelta(days=1)))
            if not vix_daily.empty:
                vix_price = round(float(vix_daily["Close"].iloc[0]), 2)
    except Exception:
        pass

    spots = {
        "spot_nq": spot_nq, "spot_ndx": spot_ndx,
        "spot_qqq": spot_qqq, "basis": basis,
        "vix": vix_price,
    }
    vix_str = f"{vix_price:.2f}" if vix_price else "N/A (sim)"
    print(f"  NQ: {spot_nq:.2f}  QQQ: {spot_qqq:.2f}  NDX(est): {spot_ndx:.2f}  "
          f"Ratio: {ratio:.3f}  Basis: {basis:+.2f}  VIX: {vix_str}")

    # ── Earnings ──────────────────────────────────────────────────────────────
    print("  Checking earnings calendar...")
    earnings = check_earnings_today(trade_date)
    if earnings["has_major_earnings"]:
        print(f"\n{'=' * 58}")
        print(f"  MAJOR EARNINGS TODAY: {', '.join(earnings['earnings_today'])}")
        print(f"  GEX map distorted by earnings premium")
        print(f"{'=' * 58}")

    # ── ICT from clamped bars ─────────────────────────────────────────────────
    ict = compute_ict_levels(nq_bars, trade_date)

    # ── ORATS chains (prior-close inventory — always available) ───────────────
    print(f"\n  Fetching ORATS chains for {prior_str}...")
    chains = fetch_qqq_and_ndx(prior_str, use_cache=True)

    # ── Flow walls: disabled in sim mode ──────────────────────────────────────
    print("  FLOW WALLS: unavailable in backtest (no intraday ORATS history)")

    # ── Generate ──────────────────────────────────────────────────────────────
    generate_and_print_brief(spots, ict, chains, trade_date, sim_et,
                             earnings=earnings, nq_bars=nq_bars,
                             live_chains=None, sim_mode=True)


# ─── Trade state classification ──────────────────────────────────────────────

def classify_trade_state(gex: dict, spot: float) -> str:
    """
    Splits positive gamma into PIN_FADE vs VACUUM_TREND.

    PIN_FADE:        price inside inventory -> mean revert / scalp
    VACUUM_TREND:    price outside inventory + vacuum in any direction -> runner
                     No dist_to_inv gate — if vacuum exists, the room to run
                     exists regardless of how close to inventory edge.
    NEG_GAMMA_TREND: negative gamma -> trend (dealers chase moves)
    NEUTRAL:         outside inventory but no vacuum (nodes packed tight)
    """
    regime = (gex.get("regime") or "").upper()
    inside = bool(gex.get("inside_inventory"))
    vacuum_up = bool(gex.get("vacuum_up"))
    vacuum_dn = bool(gex.get("vacuum_dn"))

    if regime == "NEGATIVE":
        return "NEG_GAMMA_TREND"

    if regime == "POSITIVE":
        if inside:
            return "PIN_FADE"
        if vacuum_up or vacuum_dn:
            return "VACUUM_TREND"
        return "NEUTRAL"

    return "NEUTRAL"


# ─── Day mode / helper functions ─────────────────────────────────────────────

def classify_day_mode(trade_state: str, dist_to_inv: float,
                      premarket_range: float,
                      regime: str, is_trap: bool) -> dict:
    """
    Classify session as RUNNER / SCALP / TRAP.

    NEG_GAMMA_TREND:  runner permission needs only premarket_range >= 60.
                      dist_to_inv irrelevant — dealers are short gamma,
                      being inside inventory means they CHASE moves (amplify).
    VACUUM_TREND:     runner permission needs dist_to_inv >= 80 + premarket_range >= 60.
                      must be far from positive-gamma pinning zone.

    Returns {"mode", "runner_permission", "confidence"}.
    """
    if trade_state in ("NEG_GAMMA_TREND", "VACUUM_TREND"):
        runner_permission = premarket_range >= 60
    else:
        runner_permission = False

    if is_trap:
        mode = "TRAP"
    elif runner_permission:
        mode = "RUNNER"
    elif trade_state == "PIN_FADE" or (dist_to_inv < 40 and regime == REGIME_POSITIVE):
        mode = "SCALP"
    else:
        mode = "SCALP"

    # Confidence 0-3
    score = 0
    if trade_state in ("VACUUM_TREND", "NEG_GAMMA_TREND"):
        score += 1
    if trade_state == "NEG_GAMMA_TREND" or dist_to_inv >= 80:
        score += 1
    if premarket_range >= 60:
        score += 1

    return {"mode": mode, "runner_permission": runner_permission, "confidence": score}


def check_trap_condition(nq_bars, trade_date, inv_low: float, inv_high: float) -> bool:
    """Check if last 3 1-min closes are inside inventory zone (trap day)."""
    if nq_bars is None or nq_bars.empty:
        return False
    td = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
    rth = nq_bars[(nq_bars["date"] == td) & (nq_bars["ts"].dt.time >= time(9, 30))]
    if len(rth) < 3:
        return False
    last3 = rth.tail(3)["close"].tolist()
    return all(inv_low <= c <= inv_high for c in last3)


def compute_premarket_bias(spot_nq, pdh, pdl, onh, onl) -> str:
    """Directional bias from spot vs ICT levels.
    Gap beyond prior-day extreme is sufficient for bias —
    don't require both PDx AND ONx."""
    # Strong: beyond both prior day and overnight
    if pdh and onh and spot_nq > pdh and spot_nq > onh:
        return "LONG_BIAS"
    if pdl and onl and spot_nq < pdl and spot_nq < onl:
        return "SHORT_BIAS"
    # Moderate: gapped beyond prior-day extreme
    if pdh and spot_nq > pdh:
        return "LONG_BIAS"
    if pdl and spot_nq < pdl:
        return "SHORT_BIAS"
    return "NEUTRAL"


def detect_ict_sweeps(nq_bars, trade_date, onh, onl, pdh, pdl) -> dict:
    """
    Scan 09:30-09:40 bars for sweep + hold + failure of ONH/ONL/PDH/PDL.
    Returns dict with swept_X, hold_above_X/hold_below_X, failed_sweep_X for each level.
    """
    td = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
    result = {}

    if nq_bars is None or nq_bars.empty:
        for label in ("ONH", "ONL", "PDH", "PDL"):
            result[f"swept_{label}"] = False
            side_key = "hold_above" if label in ("ONH", "PDH") else "hold_below"
            result[f"{side_key}_{label}"] = False
            result[f"failed_sweep_{label}"] = False
        return result

    rth = nq_bars[
        (nq_bars["date"] == td) &
        (nq_bars["ts"].dt.time >= time(9, 30)) &
        (nq_bars["ts"].dt.time <= time(9, 40))
    ]

    # High-side levels (ONH, PDH): swept if max_high > level
    for label, level in [("ONH", onh), ("PDH", pdh)]:
        if level is None or rth.empty:
            result[f"swept_{label}"] = False
            result[f"hold_above_{label}"] = False
            result[f"failed_sweep_{label}"] = False
            continue

        swept = bool(rth["high"].max() > level)
        hold = False
        failed = False
        if swept and len(rth) >= 2:
            last2 = rth.tail(2)["close"].tolist()
            hold = all(c > level for c in last2)
            failed = all(c < level for c in last2)

        result[f"swept_{label}"] = swept
        result[f"hold_above_{label}"] = hold
        result[f"failed_sweep_{label}"] = failed

    # Low-side levels (ONL, PDL): swept if min_low < level
    for label, level in [("ONL", onl), ("PDL", pdl)]:
        if level is None or rth.empty:
            result[f"swept_{label}"] = False
            result[f"hold_below_{label}"] = False
            result[f"failed_sweep_{label}"] = False
            continue

        swept = bool(rth["low"].min() < level)
        hold = False
        failed = False
        if swept and len(rth) >= 2:
            last2 = rth.tail(2)["close"].tolist()
            hold = all(c < level for c in last2)
            failed = all(c > level for c in last2)

        result[f"swept_{label}"] = swept
        result[f"hold_below_{label}"] = hold
        result[f"failed_sweep_{label}"] = failed

    return result


def compute_orb_acceptance(nq_bars, trade_date, orb_high, orb_low, orb_minutes=5) -> dict:
    """
    Check if ORB boundary has been broken AND accepted (2 consecutive closes beyond).
    Returns dict with break_up, accept_up, break_dn, accept_dn.
    """
    null = {"break_up": False, "accept_up": False, "break_dn": False, "accept_dn": False}

    if nq_bars is None or nq_bars.empty or orb_high is None or orb_low is None:
        return null

    td = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
    rth = nq_bars[(nq_bars["date"] == td) & (nq_bars["ts"].dt.time >= time(9, 30))]

    if len(rth) <= orb_minutes:
        return null

    post_orb = rth.iloc[orb_minutes:]

    break_up = bool(post_orb["high"].max() > orb_high)
    break_dn = bool(post_orb["low"].min() < orb_low)

    # Accept = 2 consecutive closes beyond the level (no close inside during those 2)
    accept_up = False
    accept_dn = False
    closes = post_orb["close"].tolist()

    for i in range(1, len(closes)):
        if closes[i - 1] > orb_high and closes[i] > orb_high:
            accept_up = True
            break

    for i in range(1, len(closes)):
        if closes[i - 1] < orb_low and closes[i] < orb_low:
            accept_dn = True
            break

    return {"break_up": break_up, "accept_up": accept_up,
            "break_dn": break_dn, "accept_dn": accept_dn}


def classify_entry_mode(runner_permission: bool, orb_accept: dict,
                        ict_sweeps: dict, mode: str) -> dict:
    """
    Classify entry mode and direction from ORB acceptance + ICT sweeps.
    Returns {"entry_mode", "direction"}.
    """
    # Determine direction from acceptance/break
    direction = "NEUTRAL"
    if orb_accept["accept_up"]:
        direction = "LONG"
    elif orb_accept["accept_dn"]:
        direction = "SHORT"
    elif orb_accept["break_up"]:
        direction = "LONG"
    elif orb_accept["break_dn"]:
        direction = "SHORT"

    # Check for any failed sweep
    any_failed = any(v for k, v in ict_sweeps.items()
                     if k.startswith("failed_sweep_") and v)

    # Determine entry mode
    if runner_permission and (orb_accept["accept_up"] or orb_accept["accept_dn"]):
        entry_mode = "BREAK_AND_HOLD"
    elif runner_permission and (orb_accept["break_up"] or orb_accept["break_dn"]):
        entry_mode = "BREAK_AND_RETEST"
    elif mode == "SCALP" or any_failed:
        entry_mode = "FADE_SWEEP"
        # For fade, determine direction from failed sweep if still neutral
        if direction == "NEUTRAL":
            if ict_sweeps.get("failed_sweep_ONH") or ict_sweeps.get("failed_sweep_PDH"):
                direction = "SHORT"
            elif ict_sweeps.get("failed_sweep_ONL") or ict_sweeps.get("failed_sweep_PDL"):
                direction = "LONG"
    else:
        entry_mode = "WAIT"

    return {"entry_mode": entry_mode, "direction": direction}


# ─── Premarket plan composition ──────────────────────────────────────────────

def _compose_premarket_plan(spots, ict, active_gex, vix_data, trade_state,
                            day_mode, premarket_bias, earnings,
                            trade_date, now_et) -> dict:
    """Compose the PREMARKET PLAN dict with all context needed before ORB."""
    spot_nq = spots["spot_nq"]
    onh = ict.get("ONH")
    onl = ict.get("ONL")
    pdh = ict.get("PDH")
    pdl = ict.get("PDL")
    pdc = ict.get("PDC")

    if onh is not None and onl is not None:
        premarket_range = round(onh - onl, 1)
        range_source = "ONH-ONL"
    elif pdh is not None and pdl is not None:
        premarket_range = round(pdh - pdl, 1)
        range_source = "PDH-PDL"
    else:
        premarket_range = 0.0
        range_source = "N/A"
    gap_from_pdc = round(spot_nq - pdc, 1) if pdc else 0

    inv_low = active_gex.get("inv_low") or spot_nq
    inv_high = active_gex.get("inv_high") or spot_nq
    dist = active_gex.get("dist_to_inv") or 0.0
    regime = active_gex.get("regime", REGIME_NEUTRAL)

    # Nearest node strings
    na = active_gex.get("nearest_above")
    nb = active_gex.get("nearest_below")
    air_up = active_gex.get("air_up")
    air_dn = active_gex.get("air_dn")
    nearest_above_str = f"{_fmt(na)} ({air_up} pts)" if na and air_up else "VACUUM"
    nearest_below_str = f"{_fmt(nb)} ({air_dn} pts)" if nb and air_dn else "VACUUM"

    # Key levels — sorted by distance from spot
    key_levels = []
    for label, val in [("ONH", onh), ("PDH", pdh), ("ONL", onl), ("PDC", pdc), ("PDL", pdl)]:
        if val is not None:
            key_levels.append({"label": label, "price": val})
    # Add GEX inventory walls as tradeable levels
    if inv_low and inv_low != spot_nq:
        key_levels.append({"label": "GEX_SUP", "price": round(inv_low, 2)})
    if inv_high and inv_high != spot_nq:
        key_levels.append({"label": "GEX_RES", "price": round(inv_high, 2)})
    key_levels.sort(key=lambda x: abs(spot_nq - x["price"]))
    key_levels = key_levels[:7]

    # Time-gated status
    t = now_et.time()
    if t < time(9, 30):
        phase_status = "PREMARKET"
    elif t < time(9, 35):
        phase_status = "OPENING_PHASE"
    else:
        phase_status = "INTRADAY_CONTEXT"

    # If/then plan text
    mode = day_mode["mode"]
    suggested_t1 = vix_data["suggested_target"]
    raw_runner = max(80, round(1.5 * premarket_range)) if premarket_range > 0 else 80
    runner_target = max(80, min(raw_runner, round(suggested_t1 * 2.5)))

    # Nearest ICT levels above and below current price (for actionable entries)
    all_ict = [(l, v) for l, v in
               [("ONH", onh), ("PDH", pdh), ("PDC", pdc), ("ONL", onl), ("PDL", pdl)]
               if v is not None]
    ict_above = sorted([(l, v) for l, v in all_ict if v > spot_nq], key=lambda x: x[1])
    ict_below = sorted([(l, v) for l, v in all_ict if v < spot_nq], key=lambda x: -x[1])
    near_above = ict_above[0] if ict_above else None
    near_below = ict_below[0] if ict_below else None

    # GEX wall proximity — bounce/break setups
    dist_to_gex_sup = round(spot_nq - inv_low, 1) if inv_low and inv_low != spot_nq else None
    dist_to_gex_res = round(inv_high - spot_nq, 1) if inv_high and inv_high != spot_nq else None
    near_gex_support = dist_to_gex_sup is not None and 0 < dist_to_gex_sup <= 120
    near_gex_resistance = dist_to_gex_res is not None and 0 < dist_to_gex_res <= 120

    if mode == "RUNNER":
        # ── Context line ─────────────────────────────────────────────────────
        if trade_state == "NEG_GAMMA_TREND":
            context = "NEGATIVE GAMMA — dealers short gamma, moves OVERSHOOT."
        else:
            vac_parts = []
            if active_gex.get("vacuum_up"):
                vac_parts.append(f"above ({active_gex.get('air_up', '?')} pts)")
            if active_gex.get("vacuum_dn"):
                vac_parts.append(f"below ({active_gex.get('air_dn', '?')} pts)")
            vac_str = " and ".join(vac_parts) if vac_parts else "detected"
            context = f"VACUUM {vac_str} — room to run."

        # ── Bias line ─────────────────────────────────────────────────────────
        if premarket_bias == "LONG_BIAS":
            bias_line = "Bias: LONG — favor upside."
        elif premarket_bias == "SHORT_BIAS":
            bias_line = "Bias: SHORT — favor downside."
        else:
            bias_line = "Bias: NEUTRAL — let price action decide."

        # ── Build GEX shelf ladder (nearest nodes to spot) ────────────────────
        shelves = active_gex.get("shelves", [])
        s_above = sorted([s for s in shelves if s > spot_nq])[:4]
        s_below = sorted([s for s in shelves if s <= spot_nq], reverse=True)[:4]

        shelf_lines = []
        for s in reversed(s_above):
            d = round(s - spot_nq)
            tag = " ** call wall" if inv_high and abs(s - inv_high) < 5 else ""
            shelf_lines.append(f"    {_fmt(s)}   +{d} pts{tag}")
        shelf_lines.append(f"    -- NQ {_fmt(spot_nq)} --")
        for s in s_below:
            d = round(spot_nq - s)
            tag = " ** put wall" if inv_low and abs(s - inv_low) < 5 else ""
            shelf_lines.append(f"    {_fmt(s)}   -{d} pts{tag}")
        # Note gap below last shelf
        if s_below:
            last_below = s_below[-1]
            even_lower = sorted([s for s in shelves if s < last_below])
            if even_lower:
                gap = round(last_below - even_lower[-1])
                if gap > 100:
                    shelf_lines.append(f"    ... {gap} pt gap to next node {_fmt(even_lower[-1])}")
        shelf_ladder = "\n".join(shelf_lines)

        # ── Choose plan type based on regime + wall proximity ─────────────────
        if trade_state == "NEG_GAMMA_TREND" and (near_gex_support or near_gex_resistance):
            # ── NEG GAMMA NEAR WALL — 3 patterns (hold / overshoot / breakdown)
            if near_gex_support and (not near_gex_resistance
                                     or dist_to_gex_sup <= dist_to_gex_res):
                w = _fmt(inv_low)
                # Find next shelf below the wall for overshoot zone context
                below_wall = sorted([s for s in shelves if s < inv_low], reverse=True)
                if below_wall:
                    next_dn = _fmt(below_wall[0])
                    gap_dn = round(inv_low - below_wall[0])
                    overshoot_note = f"Next node below: {next_dn} ({gap_dn} pts below wall)"
                else:
                    overshoot_note = "No mapped nodes below wall — overshoot zone is open"

                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  GEX SHELVES:\n"
                    f"{shelf_ladder}\n"
                    f"\n"
                    f"  KEY WALL: Put wall {w} ({dist_to_gex_sup:.0f} pts below)\n"
                    f"  {overshoot_note}\n"
                    f"\n"
                    f"  PATTERN 1 — Wall Hold (bounce):\n"
                    f"    Price tests {w} and holds. Higher low on 1m.\n"
                    f"    -> LONG: T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: below {w}\n"
                    f"\n"
                    f"  PATTERN 2 — Overshoot + V-Shape:\n"
                    f"    Price BREAKS below {w} — neg gamma amplifies the drop.\n"
                    f"    Wait for reversal: higher low on 1m BELOW the wall.\n"
                    f"    -> LONG: confirmed when price closes back above {w}\n"
                    f"    -> T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: below V-bottom\n"
                    f"\n"
                    f"  PATTERN 3 — Clean Breakdown:\n"
                    f"    Price breaks below {w}, NO reversal. Lower highs continue.\n"
                    f"    -> SHORT: continuation + lower highs\n"
                    f"    -> T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: above ORB HIGH\n"
                    f"\n"
                    f"  Rule: In neg gamma, first move through a wall OVERSHOOTS.\n"
                    f"        Do NOT chase the break. Wait for pattern 2 or 3."
                )
            else:
                w = _fmt(inv_high)
                above_wall = sorted([s for s in shelves if s > inv_high])
                if above_wall:
                    next_up = _fmt(above_wall[0])
                    gap_up = round(above_wall[0] - inv_high)
                    overshoot_note = f"Next node above: {next_up} ({gap_up} pts above wall)"
                else:
                    overshoot_note = "No mapped nodes above wall — overshoot zone is open"

                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  GEX SHELVES:\n"
                    f"{shelf_ladder}\n"
                    f"\n"
                    f"  KEY WALL: Call wall {w} ({dist_to_gex_res:.0f} pts above)\n"
                    f"  {overshoot_note}\n"
                    f"\n"
                    f"  PATTERN 1 — Wall Rejection:\n"
                    f"    Price tests {w} and rejects. Lower high on 1m.\n"
                    f"    -> SHORT: T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: above {w}\n"
                    f"\n"
                    f"  PATTERN 2 — Overshoot + V-Shape:\n"
                    f"    Price BREAKS above {w} — neg gamma amplifies the spike.\n"
                    f"    Wait for reversal: lower high on 1m ABOVE the wall.\n"
                    f"    -> SHORT: confirmed when price closes back below {w}\n"
                    f"    -> T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: above overshoot high\n"
                    f"\n"
                    f"  PATTERN 3 — Clean Breakout:\n"
                    f"    Price breaks above {w}, NO reversal. Higher lows continue.\n"
                    f"    -> LONG: continuation + higher lows\n"
                    f"    -> T1 +{suggested_t1} pts | Runner +{runner_target} pts\n"
                    f"    -> Stop: below ORB LOW\n"
                    f"\n"
                    f"  Rule: In neg gamma, first move through a wall OVERSHOOTS.\n"
                    f"        Do NOT chase the break. Wait for pattern 2 or 3."
                )

        elif near_gex_support or near_gex_resistance:
            # ── VACUUM NEAR WALL — clean hold/break (positive gamma dampens) ──
            if near_gex_support and (not near_gex_resistance
                                     or dist_to_gex_sup <= dist_to_gex_res):
                w = _fmt(inv_low)
                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  GEX SHELVES:\n"
                    f"{shelf_ladder}\n"
                    f"\n"
                    f"  KEY WALL: Put wall {w} ({dist_to_gex_sup:.0f} pts below)\n"
                    f"\n"
                    f"  ABOVE WALL → LONG:\n"
                    f"    Entry: ORB breakout up, or bounce off {w} + higher low\n"
                    f"    T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    Stop: below {w}  |  Trail: under higher lows after T1\n"
                    f"\n"
                    f"  BELOW WALL → SHORT:\n"
                    f"    Entry: 2 consecutive 1m closes below {w}\n"
                    f"    T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    Stop: above ORB HIGH  |  Trail: above lower highs after T1\n"
                    f"\n"
                    f"  Rule: Wall decides. LONG only above it. SHORT only below it."
                )
            else:
                w = _fmt(inv_high)
                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  GEX SHELVES:\n"
                    f"{shelf_ladder}\n"
                    f"\n"
                    f"  KEY WALL: Call wall {w} ({dist_to_gex_res:.0f} pts above)\n"
                    f"\n"
                    f"  BELOW WALL → SHORT:\n"
                    f"    Entry: rejection at {w} + 2 closes below, or ORB break down\n"
                    f"    T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    Stop: above {w}  |  Trail: above lower highs after T1\n"
                    f"\n"
                    f"  ABOVE WALL → LONG:\n"
                    f"    Entry: 2 consecutive 1m closes above {w}\n"
                    f"    T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    Stop: below ORB LOW  |  Trail: under higher lows after T1\n"
                    f"\n"
                    f"  Rule: Wall decides. SHORT only below it. LONG only above it."
                )
        else:
            # ── ORB-CENTRIC — no nearby GEX wall, use ICT levels ─────────────
            if near_above:
                la, pa = near_above
                da = round(pa - spot_nq)
                long_trigger = f"ORB break above {la} {_fmt(pa)} ({da} pts above)"
            else:
                long_trigger = "ORB break above ORB HIGH (vacuum above)"

            if near_below:
                lb, pb = near_below
                db = round(spot_nq - pb)
                short_trigger = f"ORB break below {lb} {_fmt(pb)} ({db} pts below)"
            else:
                short_trigger = "ORB break below ORB LOW (vacuum below)"

            fade_rule = ("Do NOT fade. Follow the ORB break direction."
                         if trade_state == "NEG_GAMMA_TREND"
                         else "Wait for ORB break + acceptance (2 closes beyond).")

            if premarket_bias == "LONG_BIAS":
                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  LONG [PRIMARY]: {long_trigger}\n"
                    f"    -> T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    -> Stop: below ORB LOW  |  Trail: under higher lows after T1\n"
                    f"\n"
                    f"  SHORT [counter-trend]: {short_trigger}\n"
                    f"    -> Scalp only: +{min(30, suggested_t1)} pts. Do NOT hold.\n"
                    f"\n"
                    f"  Rule: {fade_rule}"
                )
            elif premarket_bias == "SHORT_BIAS":
                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  SHORT [PRIMARY]: {short_trigger}\n"
                    f"    -> T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    -> Stop: above ORB HIGH  |  Trail: above lower highs after T1\n"
                    f"\n"
                    f"  LONG [counter-trend]: {long_trigger}\n"
                    f"    -> Scalp only: +{min(30, suggested_t1)} pts. Do NOT hold.\n"
                    f"\n"
                    f"  Rule: {fade_rule}"
                )
            else:
                if_then = (
                    f"{context}\n"
                    f"  {bias_line}\n"
                    f"\n"
                    f"  LONG: {long_trigger}\n"
                    f"    -> T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    -> Stop: below ORB LOW  |  Trail: under higher lows after T1\n"
                    f"\n"
                    f"  SHORT: {short_trigger}\n"
                    f"    -> T1: +{suggested_t1} pts  |  Runner: +{runner_target} pts\n"
                    f"    -> Stop: above ORB HIGH  |  Trail: above lower highs after T1\n"
                    f"\n"
                    f"  Rule: {fade_rule}"
                )
    elif mode == "SCALP":
        if_then = (
            f"Fade failed breaks at nearest nodes\n"
            f"  -> Target: 10-30 pts or nearest GEX node\n"
            f"  Rule: Do NOT hold runners. Take profits into nodes."
        )
    elif mode == "TRAP":
        if_then = (
            f"Price stuck inside inventory. No breakout chasing.\n"
            f"  -> Fade back toward nearest support/resistance\n"
            f"  Rule: Mean reversion only. Prioritize ICT levels."
        )
    else:
        if_then = "Wait for ORB close for direction confirmation."

    return {
        "status": phase_status,
        "date": str(trade_date),
        "time_et": now_et.strftime("%H:%M"),
        "spot_nq": spots["spot_nq"],
        "spot_ndx": spots["spot_ndx"],
        "spot_qqq": spots["spot_qqq"],
        "basis": spots["basis"],
        "vix_price": vix_data["vix_price"],
        "vix_regime": vix_data["vix_regime"],
        "vix_note": vix_data["vix_note"],
        "suggested_target": vix_data["suggested_target"],
        "PDH": pdh,
        "PDL": pdl,
        "PDC": pdc,
        "ONH": onh,
        "ONL": onl,
        "premarket_range": premarket_range,
        "range_source": range_source,
        "gap_from_pdc": gap_from_pdc,
        "trade_state": trade_state,
        "regime": regime,
        "mode": day_mode["mode"],
        "runner_permission": day_mode["runner_permission"],
        "confidence": day_mode["confidence"],
        "premarket_bias": premarket_bias,
        "inv_low": inv_low,
        "inv_high": inv_high,
        "dist_to_inv": dist,
        "nearest_above": na,
        "nearest_above_str": nearest_above_str,
        "nearest_below": nb,
        "nearest_below_str": nearest_below_str,
        "vacuum_up": active_gex.get("vacuum_up", False),
        "vacuum_dn": active_gex.get("vacuum_dn", False),
        "inside_inventory": active_gex.get("inside_inventory", False),
        "wall_magnetism": active_gex.get("wall_magnetism", "neutral"),
        "regime_confluence": active_gex.get("regime_confluence", False),
        "key_levels": key_levels,
        "if_then_plan": if_then,
        # Earnings
        "earnings_today": earnings.get("earnings_today", []),
        "has_major_earnings": earnings.get("has_major_earnings", False),
        "earnings_warning": earnings.get("warning"),
    }


# ─── ORB execution plan composition ─────────────────────────────────────────

def _compose_orb_plan(premarket_plan, orb_data, ict_sweeps, orb_accept,
                      entry_info, spots, active_gex, ict, now_et,
                      flow_walls=None) -> dict:
    """Compose the ORB EXECUTION PLAN dict with break/hold/retest/fade guidance."""
    if flow_walls is None:
        from gex import _empty_flow_combined
        flow_walls = _empty_flow_combined()

    orb_high = orb_data.get("orb_high")
    orb_low = orb_data.get("orb_low")
    orb_mid = orb_data.get("orb_mid")
    orb_range = orb_data.get("orb_range")

    entry_mode = entry_info["entry_mode"]
    direction = entry_info["direction"]
    mode = premarket_plan["mode"]
    spot_nq = spots["spot_nq"]

    # Status
    if mode == "TRAP":
        status = "TRAP_DAY"
    elif entry_mode == "BREAK_AND_HOLD":
        status = "TRADE_READY"
    elif entry_mode == "BREAK_AND_RETEST":
        status = "WAIT_FOR_RETEST"
    elif entry_mode == "FADE_SWEEP":
        status = "SCALP_ONLY"
    else:
        status = "WAIT_FOR_RETEST"

    # Earnings risk flag (not an override — keep real status)
    earnings_risk = "HIGH" if premarket_plan.get("has_major_earnings") else None

    inv_low = premarket_plan.get("inv_low")
    inv_high = premarket_plan.get("inv_high")

    # Targets
    if orb_range and orb_range > 0:
        t1_raw = orb_range * 2
        t1_target = max(20, min(40, t1_raw))
    else:
        t1_target = 30

    premarket_range = premarket_plan.get("premarket_range", 0)
    runner_target = max(80, round(1.5 * premarket_range)) if premarket_range > 0 else 80

    # Hold line + invalidation + trail
    if direction == "LONG":
        hold_line = orb_high
        invalidation = (
            f"1-min close back inside ORB range ({_fmt(orb_low)}-{_fmt(orb_high)})\n"
            f"                OR re-enter inventory ({_fmt(inv_low)}-{_fmt(inv_high)}) and hold 3 min"
        )
        trail_rule = "After T1, move stop to ORB HIGH; trail under higher lows"
    elif direction == "SHORT":
        hold_line = orb_low
        invalidation = (
            f"1-min close back inside ORB range ({_fmt(orb_low)}-{_fmt(orb_high)})\n"
            f"                OR re-enter inventory ({_fmt(inv_low)}-{_fmt(inv_high)}) and hold 3 min"
        )
        trail_rule = "After T1, move stop to ORB LOW; trail above lower highs"
    else:
        hold_line = None
        invalidation = "Wait for direction confirmation"
        trail_rule = ""

    # Nearest node context for runner target description
    na = active_gex.get("nearest_above")
    nb = active_gex.get("nearest_below")
    na_str = f"VACUUM" if not na else f"{_fmt(na)}"
    nb_str = f"VACUUM" if not nb else f"{_fmt(nb)}"

    if direction == "LONG":
        runner_context = f"trail -- {na_str} above" if not na else f"nearest node {na_str}"
    elif direction == "SHORT":
        runner_context = f"trail -- {nb_str} below" if not nb else f"nearest node {nb_str}"
    else:
        runner_context = ""

    # ── Flow wall context for execution guidance ─────────────────────────────
    fcw = flow_walls.get("flow_call_wall")
    fpw = flow_walls.get("flow_put_wall")

    # Flow wall proximity management
    flow_partials_note = None
    if fcw and direction == "LONG":
        dist_to_fcw = fcw - spot_nq
        if 0 < dist_to_fcw <= 15:
            flow_partials_note = f"AT flow call resistance {_fmt(fcw)} ({dist_to_fcw:.0f} pts) -- do NOT chase; take profits NOW"
            t1_target = min(t1_target, max(10, int(dist_to_fcw - 3)))
        elif 15 < dist_to_fcw <= 40:
            flow_partials_note = f"Resistance within {dist_to_fcw:.0f} pts -- take partials into {_fmt(fcw)}; do NOT add size; tighten trail"
            t1_target = min(t1_target, max(20, int(dist_to_fcw - 5)))
        elif 40 < dist_to_fcw <= 60:
            flow_partials_note = f"Resistance nearby -- consider partials at {_fmt(fcw)} ({dist_to_fcw:.0f} pts)"
    if fpw and direction == "SHORT":
        dist_to_fpw = spot_nq - fpw
        if 0 < dist_to_fpw <= 15:
            flow_partials_note = f"AT flow put support {_fmt(fpw)} ({dist_to_fpw:.0f} pts) -- do NOT chase; take profits NOW"
            t1_target = min(t1_target, max(10, int(dist_to_fpw - 3)))
        elif 15 < dist_to_fpw <= 40:
            flow_partials_note = f"Support within {dist_to_fpw:.0f} pts -- take partials into {_fmt(fpw)}; do NOT add size; tighten trail"
            t1_target = min(t1_target, max(20, int(dist_to_fpw - 5)))
        elif 40 < dist_to_fpw <= 60:
            flow_partials_note = f"Support nearby -- consider partials at {_fmt(fpw)} ({dist_to_fpw:.0f} pts)"

    # Scalp plan: use flow walls as main zones
    scalp_plan = None
    if mode == "SCALP" or entry_mode == "FADE_SWEEP":
        if fcw and spot_nq < fcw and (fcw - spot_nq) <= 60:
            scalp_plan = (
                f"SHORT SCALP: tag flow call wall {_fmt(fcw)} + rejection confirmation\n"
                f"  Confirmation: 1m close back below wall OR lower-high after 2nd test\n"
                f"  Stop: above wick high\n"
                f"  Target: 10-30 pts or to {_fmt(fpw) if fpw else 'nearest support'}"
            )
        elif fpw and spot_nq > fpw and (spot_nq - fpw) <= 60:
            scalp_plan = (
                f"LONG SCALP: tag flow put wall {_fmt(fpw)} + rejection confirmation\n"
                f"  Confirmation: 1m close back above wall OR higher-low after 2nd test\n"
                f"  Stop: below wick low\n"
                f"  Target: 10-30 pts or to {_fmt(fcw) if fcw else 'nearest resistance'}"
            )

    return {
        "status": status,
        "earnings_risk": earnings_risk,
        "earnings_tickers": premarket_plan.get("earnings_today", []),
        "date": premarket_plan["date"],
        "time_et": now_et.strftime("%H:%M"),
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_mid": orb_mid,
        "orb_range": orb_range,
        "sweeps": ict_sweeps,
        "orb_accept": orb_accept,
        "entry_mode": entry_mode,
        "direction": direction,
        "hold_line": hold_line,
        "invalidation": invalidation,
        "t1_target": t1_target,
        "runner_target": runner_target,
        "runner_context": runner_context,
        "trail_rule": trail_rule,
        "mode": mode,
        # Flow walls
        "flow_walls": flow_walls,
        "flow_partials_note": flow_partials_note,
        "scalp_plan": scalp_plan,
    }


# ─── Terminal output — Premarket plan ────────────────────────────────────────

def _print_premarket_plan(p: dict):
    """Print formatted PREMARKET PLAN to terminal."""
    spot_nq = p["spot_nq"]
    spot_ndx = p["spot_ndx"]
    spot_qqq = p["spot_qqq"]
    basis = p["basis"]

    vix_line = (f"  VIX: {p['vix_price']:.2f}  [{p['vix_regime']}]"
                if p["vix_price"] else "  VIX: N/A")

    # Earnings warning banner
    earnings_banner = ""
    if p.get("has_major_earnings"):
        tickers = ", ".join(p["earnings_today"])
        earnings_banner = f"""
{'!' * 60}
  EARNINGS TODAY: {tickers}
  GEX map distorted by earnings premium -- SKIP day
{'!' * 60}"""

    runner_perm = "YES" if p.get("runner_permission") else "NO"

    # Time-gated header
    status = p["status"]
    sim_tag = " (SIM)" if p.get("sim_mode") else ""
    if status == "INTRADAY_CONTEXT":
        header_label = "NQ INTRADAY CONTEXT"
    elif status == "OPENING_PHASE":
        header_label = "NQ OPENING PHASE"
    else:
        header_label = "NQ PREMARKET PLAN"

    print(f"""{earnings_banner}
{'=' * 60}
  {header_label} -- {p['date']}  {p['time_et']} ET{sim_tag}
  STATUS: {p['status']}
{'=' * 60}
  NQ {spot_nq:.2f}  |  NDX {spot_ndx:.2f}  |  QQQ {spot_qqq:.2f}  |  Basis {basis:+.2f}
{vix_line}

-- CONTEXT -------------------------------------------------
  PDH:  {_fmt(p['PDH'])}   PDL: {_fmt(p['PDL'])}   PDC: {_fmt(p['PDC'])}
  ONH:  {_fmt(p['ONH'])}   ONL: {_fmt(p['ONL'])}
  Premarket Range:   {p['premarket_range']:.0f} pts ({p.get('range_source', 'ONH-ONL')})
  Gap from PDC:      {p['gap_from_pdc']:+.0f} pts

-- ORATS INVENTORY (prior-close OI -- context only) --------
  Trade State:       {p['trade_state']}
  Regime:            {p['regime'].upper()}
  Inv:               {_fmt(p['inv_low'])} -- {_fmt(p['inv_high'])}  (dist: {p['dist_to_inv']:.0f} pts)
  Nearest Above:     {p['nearest_above_str']}
  Nearest Below:     {p['nearest_below_str']}

-- DAY MODE ------------------------------------------------
  MODE:              {p['mode']}
  RUNNER PERMISSION: {runner_perm}
  BIAS:              {p['premarket_bias']}

-- KEY LEVELS TO WATCH ({len(p['key_levels'])}) ---------------------------------""")

    for i, kl in enumerate(p["key_levels"], 1):
        print(f"  {i}. {kl['label']:<8}{_fmt(kl['price'])}")

    print(f"""
-- PLAN ----------------------------------------------------""")
    for line in p["if_then_plan"].split("\n"):
        print(f"  {line}")

    print(f"{'=' * 60}")


# ─── Terminal output — ORB execution plan ────────────────────────────────────

def _print_orb_plan(p: dict):
    """Print formatted ORB EXECUTION PLAN to terminal."""
    sweeps = p.get("sweeps", {})
    orb_accept = p.get("orb_accept", {})
    flow = p.get("flow_walls", {})

    # Earnings risk suffix for status line
    earnings_risk = p.get("earnings_risk")
    sim_tag = " (SIM)" if p.get("sim_mode") else ""
    status_str = p["status"]
    if earnings_risk:
        status_str += f"  (EARNINGS RISK {earnings_risk})"

    print(f"""
{'=' * 60}
  NQ ORB EXECUTION PLAN -- {p['date']}  {p['time_et']} ET{sim_tag}
  STATUS: {status_str}
{'=' * 60}""")

    if earnings_risk:
        tickers = ", ".join(p.get("earnings_tickers", []))
        if not tickers:
            tickers = "major name"
        print(f"  ** EARNINGS WARNING: {tickers} reporting -- GEX map distorted, widen stops **")

    print(f"""
-- ORB LEVELS ----------------------------------------------
  ORB HIGH:    {_fmt(p['orb_high'])}
  ORB LOW:     {_fmt(p['orb_low'])}
  ORB MID:     {_fmt(p['orb_mid'])}
  ORB RANGE:   {_fmt(p['orb_range'], 1) if p.get('orb_range') else 'N/A'} pts

-- ICT SWEEPS (09:30-09:40) --------------------------------""")

    # Print sweep status for each level
    for label in ["ONH", "ONL", "PDH", "PDL"]:
        swept = sweeps.get(f"swept_{label}", False)
        if not swept:
            print(f"  {label}:  not swept")
        else:
            is_high = label in ("ONH", "PDH")
            if is_high:
                hold = sweeps.get(f"hold_above_{label}", False)
            else:
                hold = sweeps.get(f"hold_below_{label}", False)
            failed = sweeps.get(f"failed_sweep_{label}", False)

            if hold:
                side = "ABOVE" if is_high else "BELOW"
                print(f"  {label}:  SWEPT + HOLD {side}")
            elif failed:
                print(f"  {label}:  SWEPT + FAILED")
            else:
                print(f"  {label}:  SWEPT")

    bu = "YES" if orb_accept.get("break_up") else "no"
    au = "YES" if orb_accept.get("accept_up") else "no"
    bd = "YES" if orb_accept.get("break_dn") else "no"
    ad = "YES" if orb_accept.get("accept_dn") else "no"
    if orb_accept.get("break_up"):
        bu += " (bar high > ORB HIGH)"
    if orb_accept.get("accept_up"):
        au += " (2 consecutive closes above ORB HIGH)"
    if orb_accept.get("break_dn"):
        bd += " (bar low < ORB LOW)"
    if orb_accept.get("accept_dn"):
        ad += " (2 consecutive closes below ORB LOW)"

    print(f"""
-- ORB ACCEPTANCE ------------------------------------------
  Break UP:    {bu}
  Accept UP:   {au}
  Break DN:    {bd}
  Accept DN:   {ad}""")

    # ── FLOW WALLS section ───────────────────────────────────────────────────
    if p.get("sim_mode"):
        print(f"""
-- FLOW WALLS ----------------------------------------------
  FLOW WALLS: unavailable in backtest unless snapshots logged""")
    else:
        fcw = flow.get("flow_call_wall")
        fpw = flow.get("flow_put_wall")
        fcs = flow.get("flow_call_struct")
        fps = flow.get("flow_put_struct")
        nfa = flow.get("nearest_flow_above")
        nfb = flow.get("nearest_flow_below")
        fau = flow.get("flow_air_up")
        fad = flow.get("flow_air_dn")
        top_nodes = flow.get("top_nodes_flow", [])
        src = flow.get("source", "N/A")

        nfa_str = f"{_fmt(nfa)} ({fau} pts)" if nfa and fau else "VACUUM"
        nfb_str = f"{_fmt(nfb)} ({fad} pts)" if nfb and fad else "VACUUM"

        print(f"""
-- FLOW WALLS (intraday -- for scalps) ---------------------
  Source:                {src}
  Flow Call Resistance:  {_fmt(fcw)}
  Flow Put Support:      {_fmt(fpw)}
  Nearest Flow Above:    {nfa_str}
  Nearest Flow Below:    {nfb_str}
  Flow Top Nodes:        {', '.join(_fmt(n) for n in top_nodes[:5]) if top_nodes else 'N/A'}""")

    # ── ENTRY section ────────────────────────────────────────────────────────
    print(f"""
-- ENTRY ---------------------------------------------------
  ENTRY MODE:  {p['entry_mode']}
  DIRECTION:   {p['direction']}
""")

    entry_mode = p["entry_mode"]
    direction = p["direction"]

    if entry_mode == "BREAK_AND_HOLD":
        if direction == "LONG":
            side_label = "ORB HIGH"
            level = p["orb_high"]
            t1_approx = round(level + p["t1_target"], 2) if level else None
        else:
            side_label = "ORB LOW"
            level = p["orb_low"]
            t1_approx = round(level - p["t1_target"], 2) if level else None

        t1_str = f" (take partial at ~{_fmt(t1_approx)})" if t1_approx else ""
        print(f"  Entry:        Already accepted {'above' if direction == 'LONG' else 'below'} {side_label} {_fmt(level)}")
        print(f"  HOLD LINE:    {side_label} {_fmt(p['hold_line'])}")
        print(f"  INVALIDATION: {p['invalidation']}")
        print(f"  T1:           +{p['t1_target']} pts{t1_str}")
        print(f"  Runner:       +{p['runner_target']} pts ({p['runner_context']})")
        print(f"  Trail:        {p['trail_rule']}")
        # DO NOW line
        print(f"")
        print(f"  DO NOW:       Hold {direction} above {side_label} {_fmt(level)}; partial at T1 ~{_fmt(t1_approx)}; trail remainder")

    elif entry_mode == "BREAK_AND_RETEST":
        if direction == "LONG":
            side_label = "ORB HIGH"
            level = p["orb_high"]
        else:
            side_label = "ORB LOW"
            level = p["orb_low"]

        print(f"  Entry:        Wait for retest of {side_label} {_fmt(level)}")
        print(f"  HOLD LINE:    {side_label} {_fmt(p['hold_line'])}")
        print(f"  INVALIDATION: {p['invalidation']}")
        print(f"  T1:           +{p['t1_target']} pts")
        print(f"  Runner:       +{p['runner_target']} pts ({p['runner_context']})")
        print(f"  Trail:        {p['trail_rule']}")
        # DO NOW line
        print(f"")
        print(f"  DO NOW:       Wait for pullback to {side_label} {_fmt(level)}; enter {direction} on hold; stop below")

    elif entry_mode == "FADE_SWEEP":
        failed_labels = [k.replace("failed_sweep_", "")
                         for k, v in p.get("sweeps", {}).items()
                         if k.startswith("failed_sweep_") and v]
        failed_str = ", ".join(failed_labels) if failed_labels else "ORB level"
        dir_str = direction.lower() if direction != "NEUTRAL" else "neutral"
        print(f"  Entry:        Fade failed sweep of {failed_str} -- {dir_str}")
        print(f"  Target:       10-30 pts or nearest node")
        print(f"  Rule:         Do NOT hold runners. Take profits into nodes.")
        # DO NOW line
        print(f"")
        print(f"  DO NOW:       Fade {failed_str}; scalp 10-30 pts; no runners")

    else:  # WAIT
        print(f"  Entry:        WAIT for direction confirmation")
        print(f"  Rule:         No trade until ORB break + acceptance")
        print(f"")
        print(f"  DO NOW:       Sit on hands; wait for ORB break + 2 closes")

    # ── Flow wall guidance (mode-specific) ───────────────────────────────────
    flow_note = p.get("flow_partials_note")
    scalp_plan = p.get("scalp_plan")

    if flow_note or scalp_plan:
        print(f"\n-- FLOW GUIDANCE -------------------------------------------")
        if flow_note:
            print(f"  ** {flow_note}")
        if scalp_plan:
            for line in scalp_plan.split("\n"):
                print(f"  {line}")

    print(f"{'=' * 60}")


# ─── Shared brief generation (two-phase) ─────────────────────────────────────

def generate_and_print_brief(
    spots: dict, ict: dict, chains: dict,
    trade_date: date, now_et,
    output_dir: str = "daily_briefs",
    earnings: dict = None,
    nq_bars: pd.DataFrame = None,
    live_chains: dict = None,
    sim_mode: bool = False,
) -> dict:
    """Compute GEX, compose + print PREMARKET PLAN (always), then ORB EXECUTION PLAN (if ORB closed)."""
    if earnings is None:
        earnings = {"earnings_today": [], "has_major_earnings": False, "warning": None}

    spot_nq  = spots["spot_nq"]
    spot_ndx = spots["spot_ndx"]
    spot_qqq = spots["spot_qqq"]
    basis    = spots["basis"]

    # ── 1. Compute GEX maps (always) — prior-close OI for inventory ──────────

    # 0DTE map (critical for ORB first hour)
    qqq_today = filter_today_expiry(chains["qqq"], trade_date)
    ndx_today = filter_today_expiry(chains["ndx"], trade_date)

    # Diagnostic: unique expiries in chain (top 3 by row count)
    for lbl, chain_df in [("QQQ", chains["qqq"]), ("NDX", chains["ndx"])]:
        if not chain_df.empty and "expiry" in chain_df.columns:
            expiry_counts = chain_df.groupby("expiry").size().sort_values(ascending=False)
            top3 = expiry_counts.head(3)
            top3_str = ", ".join(f"{exp} ({cnt})" for exp, cnt in top3.items())
            td = pd.Timestamp(trade_date).date() if not isinstance(trade_date, date) else trade_date
            today_count = chain_df[pd.to_datetime(chain_df["expiry"]).dt.date == td].shape[0]
            print(f"[CHAIN] {lbl}: {len(chain_df)} rows | top expiries: {top3_str} | expiry=={trade_date}: {today_count} rows")
        else:
            print(f"[CHAIN] {lbl}: empty")
    print(f"[0DTE] QQQ rows after today-expiry filter: {len(qqq_today)}")
    print(f"[0DTE] NDX rows after today-expiry filter: {len(ndx_today)}")

    # Compute 0DTE maps
    qqq_gex_0dte = compute_gex_map(qqq_today, spot_qqq,
                                    label="qqq_0dte",
                                    dte_min=0, dte_max=999)
    ndx_gex_0dte = compute_gex_map(ndx_today, spot_ndx,
                                    label="ndx_0dte",
                                    dte_min=0, dte_max=999)
    gex_0dte = combine_qqq_ndx_gex(qqq_gex_0dte, ndx_gex_0dte,
                                    spot_nq, basis=basis)

    # Multi-day map (structural positioning)
    qqq_gex_multi = compute_gex_map(chains["qqq"], spot_qqq,
                                     label="qqq_multi", dte_min=1, dte_max=60)
    ndx_gex_multi = compute_gex_map(chains["ndx"], spot_ndx,
                                     label="ndx_multi", dte_min=1, dte_max=60)
    gex_multi = combine_qqq_ndx_gex(qqq_gex_multi, ndx_gex_multi,
                                     spot_nq, basis=basis)

    # VIX classification
    vix_data = classify_vix(spots.get("vix"))

    # ── Active GEX selection ─────────────────────────────────────────────────
    active_gex = gex_0dte if gex_0dte.get("regime") != "neutral" else gex_multi
    active_label = "0DTE" if gex_0dte.get("regime") != "neutral" else "MULTI"

    # If 0DTE is truly empty, fall back
    if (active_gex.get("regime") == REGIME_NEUTRAL
            and active_gex.get("gamma_flip") is None
            and active_gex.get("call_wall") is None):
        active_gex = gex_multi
        active_label = "MULTI (0DTE empty)"

    # ── 1b. Compute FLOW walls from LIVE chain (if available) ────────────────

    from gex import _empty_flow_combined
    flow_walls = _empty_flow_combined()

    if live_chains is not None:
        ndx_live_wide = live_chains.get("ndx", {}).get("wide", pd.DataFrame())
        qqq_live_wide = live_chains.get("qqq", {}).get("wide", pd.DataFrame())

        ndx_flow = compute_flow_walls(ndx_live_wide, spot_ndx, label="ndx_live")
        qqq_flow = compute_flow_walls(qqq_live_wide, spot_qqq, label="qqq_live")
        flow_walls = combine_flow_walls(qqq_flow, ndx_flow, spot_nq, basis=basis)

        # ── Debug: flow wall sanity check ────────────────────────────────────
        _print_flow_debug(ndx_flow, qqq_flow, flow_walls, spot_ndx, spot_nq, basis)

        # ── Save flow snapshots for future replay ────────────────────────────
        time_et_str = now_et.strftime("%H:%M")
        td_str_snap = str(trade_date)
        if not ndx_live_wide.empty:
            save_flow_snapshot(ndx_live_wide, "NDX", spot_ndx, td_str_snap, time_et_str)
        if not qqq_live_wide.empty:
            save_flow_snapshot(qqq_live_wide, "QQQ", spot_qqq, td_str_snap, time_et_str)

    # ── 2. Compute common context ────────────────────────────────────────────

    trade_state = classify_trade_state(active_gex, spot_nq)

    inv_low  = active_gex.get("inv_low") or spot_nq
    inv_high = active_gex.get("inv_high") or spot_nq
    dist     = active_gex.get("dist_to_inv") or 0.0
    regime   = active_gex.get("regime", REGIME_NEUTRAL)

    onh_val = ict.get("ONH")
    onl_val = ict.get("ONL")
    pdh_val = ict.get("PDH")
    pdl_val = ict.get("PDL")
    if onh_val is not None and onl_val is not None:
        premarket_range = round(onh_val - onl_val, 1)
    elif pdh_val is not None and pdl_val is not None:
        premarket_range = round(pdh_val - pdl_val, 1)
    else:
        premarket_range = 0.0
    is_trap = check_trap_condition(nq_bars, trade_date, inv_low, inv_high)
    day_mode = classify_day_mode(trade_state, dist, premarket_range, regime, is_trap)
    premarket_bias = compute_premarket_bias(
        spot_nq, ict.get("PDH"), ict.get("PDL"), ict.get("ONH"), ict.get("ONL")
    )

    # ── 3. Compose + print PREMARKET PLAN (always) ───────────────────────────

    premarket_plan = _compose_premarket_plan(
        spots, ict, active_gex, vix_data, trade_state,
        day_mode, premarket_bias, earnings, trade_date, now_et
    )
    if sim_mode:
        premarket_plan["sim_mode"] = True
    _print_premarket_plan(premarket_plan)

    # ── 3a. Prior session context ─────────────────────────────────────────
    prior_journal = _load_prior_journal(trade_date)
    if prior_journal:
        _print_prior_session(prior_journal)

    # ── 3b. Playbook pattern matching ─────────────────────────────────────
    try:
        playbook_matches = match_patterns(premarket_plan)
    except Exception as e:
        print(f"  [WARN] Playbook matching failed: {e}")
        playbook_matches = []

    # Load scorecard + decay warnings for live intelligence
    scorecard = {}
    decay_warnings = []
    try:
        intel = load_live_intelligence()
        scorecard = intel["scorecard"]
        decay_warnings = intel["decay_warnings"]
    except Exception:
        pass

    if playbook_matches:
        print(format_playbook_matches(playbook_matches, scorecard=scorecard,
                                      decay_warnings=decay_warnings))

    # ── 4. If ORB closed: compute ORB + compose + print ORB EXECUTION PLAN ──

    if nq_bars is not None and not nq_bars.empty:
        orb_data = compute_orb(nq_bars, trade_date)
    else:
        orb_data = {"orb_high": None, "orb_low": None, "orb_mid": None,
                    "orb_range": None, "orb_closed": False}

    orb_plan = None
    if orb_data.get("orb_closed"):
        ict_sweeps = detect_ict_sweeps(
            nq_bars, trade_date,
            ict.get("ONH"), ict.get("ONL"), ict.get("PDH"), ict.get("PDL")
        )
        orb_accept = compute_orb_acceptance(
            nq_bars, trade_date, orb_data["orb_high"], orb_data["orb_low"]
        )
        entry_info = classify_entry_mode(
            day_mode["runner_permission"], orb_accept, ict_sweeps, day_mode["mode"]
        )
        orb_plan = _compose_orb_plan(
            premarket_plan, orb_data, ict_sweeps, orb_accept,
            entry_info, spots, active_gex, ict, now_et,
            flow_walls=flow_walls,
        )
        if sim_mode:
            orb_plan["sim_mode"] = True
        _print_orb_plan(orb_plan)

    # ── 5. Save combined JSON ────────────────────────────────────────────────

    combined = dict(premarket_plan)
    # Preserve GEX detail fields for backward compatibility
    combined.update({
        "active_gex_label": active_label,
        "gex_0dte_regime":         gex_0dte.get("regime", REGIME_NEUTRAL),
        "gex_0dte_flip":           gex_0dte.get("gamma_flip"),
        "gex_0dte_call_wall":      gex_0dte.get("call_wall"),
        "gex_0dte_put_wall":       gex_0dte.get("put_wall"),
        "gex_0dte_nearest_above":  gex_0dte.get("nearest_above"),
        "gex_0dte_nearest_below":  gex_0dte.get("nearest_below"),
        "gex_0dte_air_up":         gex_0dte.get("air_up"),
        "gex_0dte_air_dn":         gex_0dte.get("air_dn"),
        "gex_multi_regime":        gex_multi.get("regime", REGIME_NEUTRAL),
        "gex_multi_flip":          gex_multi.get("gamma_flip"),
        "gex_multi_call_wall":     gex_multi.get("call_wall"),
        "gex_multi_put_wall":      gex_multi.get("put_wall"),
        "gex_multi_call_wall_struct": gex_multi.get("call_wall_struct"),
        "gex_multi_put_wall_struct":  gex_multi.get("put_wall_struct"),
        "gex_multi_call_wall_global": gex_multi.get("call_wall_global"),
        "gex_multi_put_wall_global":  gex_multi.get("put_wall_global"),
        "gex_multi_nearest_above": gex_multi.get("nearest_above"),
        "gex_multi_nearest_below": gex_multi.get("nearest_below"),
        "gex_multi_air_up":        gex_multi.get("air_up"),
        "gex_multi_air_dn":        gex_multi.get("air_dn"),
        "shelves":                 gex_multi.get("shelves", []),
        # ICT levels not already in premarket_plan
        "asia_hi":          ict.get("asia_hi"),
        "asia_lo":          ict.get("asia_lo"),
        "london_hi":        ict.get("london_hi"),
        "london_lo":        ict.get("london_lo"),
        "equal_highs":      ict.get("equal_highs", []),
        "equal_lows":       ict.get("equal_lows", []),
        # Flow walls
        "flow_call_wall":    flow_walls.get("flow_call_wall"),
        "flow_put_wall":     flow_walls.get("flow_put_wall"),
        "flow_source":       flow_walls.get("source"),
    })
    combined["playbook_matches"] = [
        {"pattern_id": m["pattern_id"], "name": m["name"],
         "priority": m["priority"], "match_ratio": m["match_ratio"],
         "bias": m["bias"], "bias_override": m["bias_override"],
         "warning": m["warning"]}
        for m in playbook_matches
    ] if playbook_matches else []
    if orb_plan:
        combined["orb_plan"] = orb_plan

    combined["prior_session"] = {
        "date": prior_journal["date"],
        "orb_break_dir": prior_journal.get("outcome", {}).get("orb_break_dir"),
        "day_range": prior_journal.get("outcome", {}).get("day_range"),
        "t1_hit": prior_journal.get("outcome", {}).get("t1_hit"),
        "lesson": prior_journal.get("lesson"),
    } if prior_journal else None

    td_str = str(trade_date)
    Path(output_dir).mkdir(exist_ok=True)
    json_path = f"{output_dir}/brief_{td_str}.json"
    with open(json_path, "w") as f:
        json.dump(_serialize(combined), f, indent=2)
    print(f"\n  Saved: {json_path}")

    csv_path = f"{output_dir}/daily_log.csv"
    _append_csv(combined, csv_path)

    return combined


def _print_flow_debug(ndx_flow, qqq_flow, flow_walls, spot_ndx, spot_nq, basis):
    """Print once-per-run flow wall debug/sanity block."""
    print(f"\n[FLOW DEBUG] ────────────────────────────────────────────")

    # Confirm live chain includes required columns
    ndx_bs = ndx_flow.get("by_strike_flow", pd.DataFrame())
    qqq_bs = qqq_flow.get("by_strike_flow", pd.DataFrame())
    ndx_ok = not ndx_bs.empty and "callVolume" in ndx_bs.columns
    qqq_ok = not qqq_bs.empty and "callVolume" in qqq_bs.columns
    print(f"  NDX live chain: {'OK' if ndx_ok else 'MISSING'} ({len(ndx_bs)} flow strikes)")
    print(f"  QQQ live chain: {'OK' if qqq_ok else 'MISSING'} ({len(qqq_bs)} flow strikes)")

    # Top 3 NDX strikes by |flow| near spot
    if ndx_ok:
        near = ndx_bs[(ndx_bs["strike"] >= spot_ndx - 200) &
                       (ndx_bs["strike"] <= spot_ndx + 200)]
        top3 = near.reindex(near["flow"].abs().sort_values(ascending=False).index).head(3)
        print(f"  Top 3 NDX flow strikes (near spot {spot_ndx:.0f}):")
        for _, r in top3.iterrows():
            nq_strike = r["strike"] + basis
            sign = "+" if r["flow"] > 0 else ""
            tag = "CALL RES" if r["flow"] > 0 else "PUT SUP"
            print(f"    NDX {r['strike']:.0f} (NQ ~{nq_strike:.0f})  "
                  f"cVol={r['callVolume']:.0f} pVol={r['putVolume']:.0f}  "
                  f"flow={sign}{r['flow']:,.0f}  [{tag}]")

    # Combined NQ walls
    fcw = flow_walls.get("flow_call_wall")
    fpw = flow_walls.get("flow_put_wall")
    print(f"  NQ Flow Call Wall: {_fmt(fcw)}  |  NQ Flow Put Wall: {_fmt(fpw)}")
    print(f"[/FLOW DEBUG] ───────────────────────────────────────────")


# ─── Utilities ───────────────────────────────────────────────────────────────

def _load_prior_journal(trade_date):
    """Load yesterday's journal entry. Returns dict or None."""
    prior = get_prior_trading_date(trade_date)
    path = Path("journal") / f"{prior}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def _print_prior_session(entry):
    """Print prior session context block."""
    o = entry.get("outcome", {})
    d = entry.get("date", "?")

    brk = o.get("orb_break_dir", "none")
    brk_time = o.get("orb_break_time", "?")
    day_range = o.get("day_range", "?")
    t1 = "HIT" if o.get("t1_hit") else "MISS"
    runner = "HIT" if o.get("runner_hit") else "MISS"
    mfe = o.get("mfe_from_orb_break")
    mae = o.get("mae_from_orb_break")
    patterns = entry.get("matched_patterns", [])
    lesson = entry.get("lesson")

    print(f"\n  -- PRIOR SESSION ({d}) ──────────────────────────────")
    print(f"  ORB Break: {brk.upper() if brk else 'NONE'} @ {brk_time}  |  "
          f"Day Range: {day_range} pts")

    mfe_str = f"MFE: +{mfe:.0f}" if mfe is not None else "MFE: ---"
    mae_str = f"MAE: -{mae:.0f}" if mae is not None else "MAE: ---"
    print(f"  T1: {t1}  |  Runner: {runner}  |  {mfe_str}  {mae_str}")

    if patterns:
        print(f"  Patterns: {', '.join(patterns[:3])}")
    if lesson:
        print(f"  Lesson: {lesson}")


def _append_csv(b: dict, path: str):
    """Append brief as a row to the running daily log CSV."""
    flat = {}
    for k, v in b.items():
        if isinstance(v, dict):
            # Flatten nested dicts (orb_plan, sweeps, orb_accept) with prefix
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        flat[f"{k}_{k2}_{k3}"] = v3
                elif isinstance(v2, list):
                    flat[f"{k}_{k2}"] = str(v2)
                else:
                    flat[f"{k}_{k2}"] = v2
        elif isinstance(v, list):
            flat[k] = str(v)
        else:
            flat[k] = v

    row_df = pd.DataFrame([flat])
    if Path(path).exists():
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, mode="w", header=True, index=False)


def _fmt(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _serialize(obj):
    """Make dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NQ Pre-Market Brief")
    parser.add_argument("--date", type=str, default=None,
                        help="Trade date YYYY-MM-DD (triggers sim mode)")
    parser.add_argument("--time", type=str, default=None,
                        help="Sim time HH:MM ET (triggers sim mode)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Parquet file for legacy backtest mode")
    parser.add_argument("--journal", action="store_true",
                        help="Generate post-game journal entry")
    args = parser.parse_args()

    if args.journal:
        from journal import generate_journal
        d = args.date or str(detect_trade_date()[0])
        generate_journal(d)
        return

    if args.parquet and args.date:
        # Legacy parquet backtest
        print(f"[MODE] Backtest — {args.date}")
        run_backtest_brief(args.date, args.parquet)
    elif args.date or args.time:
        # Sim mode (yfinance bars clamped to --time, no flow walls)
        sim_date_label = args.date or "today"
        sim_time_label = args.time or "09:25"
        print(f"[MODE] Backtest Simulation — {sim_date_label}  {sim_time_label} ET")
        run_sim_brief(args.date, args.time)
    else:
        print(f"[MODE] Live")
        run_live_brief()


if __name__ == "__main__":
    main()
