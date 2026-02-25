"""
dashboard.py — NQ Pre-Market Brief Generator

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
)
from gex import compute_gex_map, combine_qqq_ndx_gex, filter_today_expiry
from ict import load_nq_data, compute_ict_levels, get_all_ict_levels_as_list
from earnings_calendar import check_earnings_today


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
    NDX derived from QQQ_live × median(NDX/QQQ) 20-day ratio.
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


def get_ict_levels_live(trade_date: date) -> dict:
    """Fetch NQ=F 1-min history from yfinance and compute ICT levels."""
    print("  Fetching NQ=F 1-min history for ICT levels...")

    df = yf.Ticker("NQ=F").history(period="5d", interval="1m")
    if df.empty:
        print("  [WARN] No NQ=F 1-min data from yfinance")
        return {}

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

    return compute_ict_levels(df, trade_date)


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

    # 3. ICT levels from yfinance 1-min history
    ict = get_ict_levels_live(trade_date)

    # 4. ORATS chains
    print(f"\n  Fetching ORATS chains for {prior_str}...")
    chains = fetch_qqq_and_ndx(prior_str, use_cache=True)

    # 5. Generate and print
    generate_and_print_brief(spots, ict, chains, trade_date, now_et, earnings=earnings)


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
    generate_and_print_brief(spots, ict, chains, trade_date, bt_time, earnings=earnings)


# ─── Shared brief generation ────────────────────────────────────────────────

def generate_and_print_brief(
    spots: dict, ict: dict, chains: dict,
    trade_date: date, now_et,
    output_dir: str = "daily_briefs",
    earnings: dict = None,
) -> dict:
    """Compute GEX, compose brief, print, and save."""
    if earnings is None:
        earnings = {"earnings_today": [], "has_major_earnings": False, "warning": None}

    spot_nq  = spots["spot_nq"]
    spot_ndx = spots["spot_ndx"]
    spot_qqq = spots["spot_qqq"]
    basis    = spots["basis"]

    # 0DTE map (critical for ORB first hour)
    # Filter to today's expiry by DATE not DTE
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

    # Compute 0DTE maps — no DTE filter needed, already filtered by expiry
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

    # Compose and print
    brief = _compose_brief(gex_0dte, gex_multi, ict, spots, trade_date, now_et, vix_data, earnings)
    _print_brief(brief)

    # Save
    td_str = str(trade_date)
    Path(output_dir).mkdir(exist_ok=True)
    json_path = f"{output_dir}/brief_{td_str}.json"
    with open(json_path, "w") as f:
        json.dump(_serialize(brief), f, indent=2)
    print(f"\n  Saved: {json_path}")

    csv_path = f"{output_dir}/daily_log.csv"
    _append_csv(brief, csv_path)

    return brief


# ─── Trade state classification ──────────────────────────────────────────────

def classify_trade_state(gex: dict, spot: float) -> str:
    """
    Splits positive gamma into PIN_FADE vs VACUUM_TREND.

    PIN_FADE:       price inside inventory -> mean revert
    VACUUM_TREND:   price outside inventory + vacuum -> continuation
    NEG_GAMMA_TREND: negative gamma -> trend
    NEUTRAL:        unclear
    """
    regime = (gex.get("regime") or "").upper()
    inside = bool(gex.get("inside_inventory"))
    vacuum_up = bool(gex.get("vacuum_up"))
    vacuum_dn = bool(gex.get("vacuum_dn"))
    dist_to_inv = gex.get("dist_to_inv") or 0.0

    if regime == "NEGATIVE":
        return "NEG_GAMMA_TREND"

    if regime == "POSITIVE":
        if inside:
            return "PIN_FADE"
        if dist_to_inv >= 40 and (vacuum_up or vacuum_dn):
            return "VACUUM_TREND"
        return "NEUTRAL"

    return "NEUTRAL"


# ─── Brief composition ──────────────────────────────────────────────────────

def _compose_brief(gex_0dte: dict, gex_multi: dict, ict: dict, spots: dict, trade_date: date, now_et, vix_data: dict, earnings: dict = None) -> dict:
    """Assemble all level data into brief structure."""
    spot_nq = spots["spot_nq"]

    # Time-based GEX selection: 0DTE before 10:30, multi-day after
    et_time = now_et.time() if hasattr(now_et.time(), 'hour') else now_et.time()
    use_0dte = et_time < time(10, 30)
    active_gex = gex_0dte if use_0dte else gex_multi
    active_label = "0DTE" if use_0dte else "MULTI"

    # If 0DTE map is empty (no data), fall back to multi-day
    if use_0dte and active_gex.get("regime") == REGIME_NEUTRAL and active_gex.get("gamma_flip") is None and active_gex.get("call_wall") is None:
        active_gex = gex_multi
        active_label = "MULTI (0DTE empty)"

    regime = active_gex.get("regime", REGIME_NEUTRAL)
    confluence = active_gex.get("regime_confluence", False)
    wall_mag = active_gex.get("wall_magnetism", "neutral")

    # Use 0DTE map if available, fall back to multi
    active_gex = gex_0dte if gex_0dte.get("regime") != "neutral" else gex_multi
    if gex_0dte.get("regime") != "neutral":
        active_label = "0DTE"

    # Read inventory zone from gex engine (already computed vs spot)
    inv_low  = active_gex.get("inv_low") or spot_nq
    inv_high = active_gex.get("inv_high") or spot_nq
    dist     = active_gex.get("dist_to_inv") or 0.0

    trade_state = classify_trade_state(active_gex, spot_nq)

    if trade_state in ("VACUUM_TREND", "NEG_GAMMA_TREND"):
        bias = "TREND_CONTINUATION"
    elif trade_state == "PIN_FADE":
        bias = "FADE_EXTREMES"
    else:
        bias = "NEUTRAL"

    # Setup text — use active bias but multi-day walls (structural)
    suggested_target = vix_data["suggested_target"]
    active_gex["bias"] = bias
    setups = _identify_setups(active_gex, gex_multi, ict, spot_nq, suggested_target)

    # Case scoring — uses active regime + multi-day air (structural)
    multi_air_up = gex_multi.get("air_up") or 0
    multi_air_dn = gex_multi.get("air_dn") or 0
    regime_ok = (regime == REGIME_NEGATIVE)
    air_ok = multi_air_up >= MIN_AIR_POINTS or multi_air_dn >= MIN_AIR_POINTS
    case_score = 0
    if regime_ok:    case_score += 1
    if air_ok:       case_score += 1
    if ict.get("ONL") or ict.get("ONH"):  case_score += 1
    if confluence:   case_score += 1

    trade_or_skip = "TRADE" if case_score >= 2 else "WATCH" if case_score == 1 else "SKIP"

    # Earnings override — force SKIP
    if earnings is None:
        earnings = {"earnings_today": [], "has_major_earnings": False, "warning": None}
    if earnings["has_major_earnings"]:
        trade_or_skip = "SKIP"

    return {
        "date":             str(trade_date),
        "time_et":          now_et.strftime("%H:%M"),
        "spot_nq":          spots["spot_nq"],
        "spot_ndx":         spots["spot_ndx"],
        "spot_qqq":         spots["spot_qqq"],
        "basis":            spots["basis"],
        # Active GEX selection
        "active_gex_label": active_label,
        "regime":           regime,
        "bias":             bias,
        "trade_state":      trade_state,
        "inv_low":          inv_low,
        "inv_high":         inv_high,
        "dist_to_inv":      dist,
        "vacuum_up":        active_gex.get("vacuum_up", False),
        "vacuum_dn":        active_gex.get("vacuum_dn", False),
        "inside_inventory": active_gex.get("inside_inventory", False),
        "wall_magnetism":   wall_mag,
        "regime_confluence": confluence,
        "qqq_regime":       active_gex.get("qqq_regime"),
        # 0DTE map
        "gex_0dte_regime":         gex_0dte.get("regime", REGIME_NEUTRAL),
        "gex_0dte_flip":           gex_0dte.get("gamma_flip"),
        "gex_0dte_call_wall":      gex_0dte.get("call_wall"),
        "gex_0dte_put_wall":       gex_0dte.get("put_wall"),
        "gex_0dte_nearest_above":  gex_0dte.get("nearest_above"),
        "gex_0dte_nearest_below":  gex_0dte.get("nearest_below"),
        "gex_0dte_air_up":         gex_0dte.get("air_up"),
        "gex_0dte_air_dn":         gex_0dte.get("air_dn"),
        # Multi-day map
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
        # ICT levels
        "PDH":              ict.get("PDH"),
        "PDL":              ict.get("PDL"),
        "PDC":              ict.get("PDC"),
        "ONH":              ict.get("ONH"),
        "ONL":              ict.get("ONL"),
        "asia_hi":          ict.get("asia_hi"),
        "asia_lo":          ict.get("asia_lo"),
        "london_hi":        ict.get("london_hi"),
        "london_lo":        ict.get("london_lo"),
        "equal_highs":      ict.get("equal_highs", []),
        "equal_lows":       ict.get("equal_lows", []),
        # VIX data
        "vix_price":        vix_data["vix_price"],
        "vix_regime":       vix_data["vix_regime"],
        "vix_note":         vix_data["vix_note"],
        "suggested_target": vix_data["suggested_target"],
        # Earnings
        "earnings_today":   earnings["earnings_today"],
        "has_major_earnings": earnings["has_major_earnings"],
        "earnings_warning": earnings["warning"],
        # Trade guidance
        "setups":           setups,
        "case_score":       case_score,
        "trade_signal":     trade_or_skip,
    }


def _identify_setups(active_gex: dict, multi_gex: dict, ict: dict, spot_nq: float, suggested_target: int = 100) -> list:
    """Generate setup watch text based on active bias + multi-day walls."""
    setups = []
    bias = active_gex.get("bias", "neutral")
    call_wall = multi_gex.get("call_wall")
    put_wall = multi_gex.get("put_wall")

    # Collect ICT levels with labels for sweep target identification
    ict_levels = []
    for key, label in [("ONL", "ONL"), ("PDL", "PDL"), ("asia_lo", "Asia Lo"),
                        ("london_lo", "London Lo")]:
        val = ict.get(key)
        if val is not None:
            ict_levels.append((float(val), label))
    for key, label in [("ONH", "ONH"), ("PDH", "PDH"), ("asia_hi", "Asia Hi"),
                        ("london_hi", "London Hi")]:
        val = ict.get(key)
        if val is not None:
            ict_levels.append((float(val), label))

    if bias == "TREND_CONTINUATION":
        # Find nearest ICT level BELOW spot for pullback entry
        below_levels = [(v, l) for v, l in ict_levels if v < spot_nq]
        if below_levels:
            below_levels.sort(key=lambda x: spot_nq - x[0])
            pull_val, pull_label = below_levels[0]
            setups.append(
                f"Watch:   Pullback to {pull_label} {pull_val:.2f} -> hold -> LONG continuation"
            )
        setups.append("Rule:    NO SHORTS unless price re-enters inventory and holds 3+ minutes")
        # Target: use nearest node above spot, NOT a wall that may be below spot
        na = active_gex.get("nearest_above")
        cw_struct = active_gex.get("call_wall_struct")
        if na is not None and na > spot_nq:
            setups.append(
                f"Target:  Nearest node above {na:.2f} (+{na - spot_nq:.0f} pts)"
            )
        elif cw_struct is not None and cw_struct > spot_nq:
            setups.append(
                f"Target:  Structural wall {cw_struct:.2f} (+{cw_struct - spot_nq:.0f} pts)"
            )
        else:
            setups.append(
                "Target:  Trail stop / measured move (open air above inventory)"
            )

    elif bias == "long":
        # Find closest ICT level BELOW spot
        below = [(v, l) for v, l in ict_levels if v < spot_nq]
        if below:
            below.sort(key=lambda x: spot_nq - x[0])
            sweep_val, sweep_label = below[0]
            setups.append(f"Watch:   Sweep of {sweep_label} {sweep_val:.2f} -> reclaim -> LONG")
            if call_wall:
                setups.append(f"Target:  Call wall {call_wall:.2f} (+{call_wall - spot_nq:.0f} pts)")
        else:
            setups.append("Watch:   LONG bias — wait for sweep of key level below spot")

    elif bias == "short":
        # Find closest ICT level ABOVE spot
        above = [(v, l) for v, l in ict_levels if v > spot_nq]
        if above:
            above.sort(key=lambda x: x[0] - spot_nq)
            sweep_val, sweep_label = above[0]
            setups.append(f"Watch:   Sweep of {sweep_label} {sweep_val:.2f} -> reclaim -> SHORT")
            if put_wall:
                setups.append(f"Target:  Put wall {put_wall:.2f} (-{spot_nq - put_wall:.0f} pts)")
        else:
            setups.append("Watch:   SHORT bias — wait for sweep of key level above spot")

    elif bias in ("fade_short", "fade_long"):
        setups.append(f"Watch:   Positive gamma — fade extremes")
        if call_wall and put_wall:
            setups.append(f"Range:   Short near {call_wall:.2f} / Long near {put_wall:.2f}")

    else:
        setups.append("Watch:   Neutral — fade extremes near walls")
        if call_wall and put_wall:
            setups.append(f"Range:   Long near {put_wall:.2f} / Short near {call_wall:.2f}")

    return setups


# ─── Terminal output ─────────────────────────────────────────────────────────

def _print_brief(b: dict):
    """Print formatted terminal brief."""
    spot_nq = b["spot_nq"]
    spot_ndx = b["spot_ndx"]
    spot_qqq = b["spot_qqq"]
    basis = b["basis"]
    regime = b["regime"]
    bias = b["bias"]
    confluence = b.get("regime_confluence", False)
    wall_mag = b.get("wall_magnetism", "neutral")

    vix_price = b["vix_price"]
    vix_regime = b["vix_regime"]
    vix_note = b["vix_note"]
    suggested_target = b["suggested_target"]

    vix_line = (f"  VIX:           {vix_price:.2f}  [{vix_regime}]  — {vix_note}"
                if vix_price else f"  VIX:           N/A")

    # Earnings warning banner at the very top
    earnings_banner = ""
    if b.get("has_major_earnings"):
        tickers = ", ".join(b["earnings_today"])
        earnings_banner = f"""
{'!' * 58}
  ⚠️  EARNINGS TODAY: {tickers}
  GEX map distorted by earnings premium — SKIP day
  Earnings distort GEX — resume tomorrow
{'!' * 58}"""

    # 0DTE air strings
    air_0_up = f"{_fmt(b['gex_0dte_nearest_above'])}  (Air: {b['gex_0dte_air_up']} pts)" if b.get("gex_0dte_air_up") else "VACUUM"
    air_0_dn = f"{_fmt(b['gex_0dte_nearest_below'])}  (Air: {b['gex_0dte_air_dn']} pts)" if b.get("gex_0dte_air_dn") else "VACUUM"

    # Multi-day air strings
    air_m_up = f"{_fmt(b['gex_multi_nearest_above'])}  (Air: {b['gex_multi_air_up']} pts)" if b.get("gex_multi_air_up") else "VACUUM"
    air_m_dn = f"{_fmt(b['gex_multi_nearest_below'])}  (Air: {b['gex_multi_air_dn']} pts)" if b.get("gex_multi_air_dn") else "VACUUM"

    # Gap warning when price is far from prior-close inventory
    dist_to_inv = b.get("dist_to_inv", 0)
    gap_warning = ""
    if dist_to_inv >= 80:
        gap_warning = f"\n  ** NOTE: Large gap ({dist_to_inv:.0f} pts) from prior-close positioning; walls are inventory, use nearest nodes. **"

    print(f"""{earnings_banner}
{'=' * 58}
  NQ PRE-MARKET BRIEF -- {b['date']}  {b['time_et']} ET
{'=' * 58}
  SPOTS      NQ {spot_nq:.2f}  |  NDX {spot_ndx:.2f}  |  QQQ {spot_qqq:.2f}  |  Basis {basis:+.2f}
{vix_line}
  Suggest target: {suggested_target} pts

  BIAS:          {bias.upper()}   [from {b.get('active_gex_label', '?')}]
  TRADE STATE:   {b['trade_state']}
  REGIME:        {regime.upper()}
  CONFLUENCE:    {"YES" if confluence else "NO"}
  WALL MAG:      {wall_mag.upper()}

-- NEAREST LEVELS (actionable for ORB) -------------------------
  Resistance:        {air_0_up}
  Support:           {air_0_dn}
  Flip:              {_fmt(b['gex_0dte_flip'])}
  Regime:            {b['gex_0dte_regime'].upper()}

-- INVENTORY ZONE (prior-close OI — NOT intraday flow) ---------
  Inv High:          {_fmt(b.get('inv_high'))}
  Inv Low:           {_fmt(b.get('inv_low'))}
  Dist to Inv:       {dist_to_inv:.0f} pts
  Call Wall (inv):   {_fmt(b.get('gex_0dte_call_wall'))}
  Put Wall (inv):    {_fmt(b.get('gex_0dte_put_wall'))}{gap_warning}

-- MULTI-DAY STRUCTURAL ----------------------------------------
  Regime:            {b['gex_multi_regime'].upper()}
  Flip:              {_fmt(b['gex_multi_flip'])}
  Call Wall (inv):   {_fmt(b['gex_multi_call_wall'])}   (+/- 1.5%)
  Put Wall (inv):    {_fmt(b['gex_multi_put_wall'])}    (+/- 1.5%)
  Struct Call Wall:  {_fmt(b.get('gex_multi_call_wall_struct'))}  (+/- 4%)
  Struct Put Wall:   {_fmt(b.get('gex_multi_put_wall_struct'))}   (+/- 4%)
  Global Call Wall:  {_fmt(b.get('gex_multi_call_wall_global'))}
  Global Put Wall:   {_fmt(b.get('gex_multi_put_wall_global'))}
  Nearest Above:     {air_m_up}
  Nearest Below:     {air_m_dn}

-- ICT LEVELS --------------------------------------------------
  PDH:           {_fmt(b['PDH'])}     PDL: {_fmt(b['PDL'])}     PDC: {_fmt(b['PDC'])}
  ONH:           {_fmt(b['ONH'])}     ONL: {_fmt(b['ONL'])}
  Asia:          {_fmt(b['asia_hi'])} / {_fmt(b['asia_lo'])}
  London:        {_fmt(b['london_hi'])} / {_fmt(b['london_lo'])}

-- SETUP WATCH -------------------------------------------------""")

    for setup in b.get("setups", []):
        print(f"  {setup}")

    earnings_note = ""
    if b.get("has_major_earnings"):
        earnings_note = f"\n  ⚠️  Earnings distort GEX — resume tomorrow"

    print(f"""
  Skip if:  No sweep of key level by 10:00 AM ET

  Case Score:    {b['case_score']}/4    Signal: {b['trade_signal']}{earnings_note}
  Target: {suggested_target} pts  |  Stop: {STOP_POINTS} pts  |  R:R = {suggested_target/STOP_POINTS:.1f}:1
{'=' * 58}
""")


# ─── Utilities ───────────────────────────────────────────────────────────────

def _append_csv(b: dict, path: str):
    """Append brief as a row to the running daily log CSV."""
    flat = {k: v for k, v in b.items() if not isinstance(v, (list, dict))}
    flat["equal_highs"] = str(b.get("equal_highs", []))
    flat["equal_lows"] = str(b.get("equal_lows", []))
    flat["setups"] = " | ".join(b.get("setups", []))

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
                        help="Backtest date YYYY-MM-DD (default: today live)")
    parser.add_argument("--parquet", type=str, default="nq_1m.parquet",
                        help="Parquet file for backtest mode")
    args = parser.parse_args()

    if args.date:
        print(f"[MODE] Backtest — {args.date}")
        run_backtest_brief(args.date, args.parquet)
    else:
        print(f"[MODE] Live")
        run_live_brief()


if __name__ == "__main__":
    main()
