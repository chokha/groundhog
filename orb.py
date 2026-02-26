"""
orb.py — ORB simulation engine
Detects sweeps of ICT levels, ORB breaks, entries, MFE/MAE, targets

Runs on each trading day using:
    - ICT levels (pre-computed)
    - GEX levels (pre-computed)
    - NQ 1-min bars for that RTH session
"""

import pandas as pd
import numpy as np
from datetime import date, time
from typing import Optional

from config import (
    RTH_OPEN, RTH_CLOSE,
    ORB_5_MINUTES, ORB_15_MINUTES,
    TARGET_POINTS, STOP_POINTS, SWEEP_BUFFER,
    CASE_1, CASE_2, CASE_3,
    REGIME_NEGATIVE, REGIME_POSITIVE,
    MIN_AIR_POINTS,
)
from gex import classify_case


def compute_orb(bars_1m: pd.DataFrame, trade_date, start_time: str = "09:30", minutes: int = 5) -> dict:
    """
    Compute ORB high/low/mid from 1-min bars for a given trade date.
    Returns None values if ORB window not yet closed (not enough bars).
    """
    td = date.fromisoformat(str(trade_date)) if isinstance(trade_date, str) else trade_date
    parts = start_time.split(":")
    st = time(int(parts[0]), int(parts[1]))

    rth = bars_1m[(bars_1m["date"] == td) & (bars_1m["ts"].dt.time >= st)]
    if len(rth) < minutes:
        return {
            "orb_high": None, "orb_low": None, "orb_mid": None,
            "orb_range": None, "orb_closed": False,
        }

    orb = rth.head(minutes)
    hi = float(orb["high"].max())
    lo = float(orb["low"].min())
    return {
        "orb_high": hi,
        "orb_low": lo,
        "orb_mid": round((hi + lo) / 2, 2),
        "orb_range": round(hi - lo, 1),
        "orb_closed": True,
    }


def simulate_orb_day(
    rth_bars: pd.DataFrame,
    ict: dict,
    gex: dict,
    trade_date: date,
    orb_minutes: int = 5,
    target_pts: float = TARGET_POINTS,
    stop_pts: float   = STOP_POINTS,
) -> dict:
    """
    Simulate one day of ORB trading.

    Logic:
        1. Identify ORB range (first N minutes)
        2. Scan post-ORB bars for sweep of key ICT levels
        3. Detect reclaim of ORB level after sweep
        4. Enter in reclaim direction
        5. Track MFE/MAE until target or stop

    Returns dict of per-day outcome metrics.
    """
    result = _base_result(trade_date, orb_minutes)

    if rth_bars.empty:
        result["skip_reason"] = "no_rth_data"
        return result

    # ── ORB range ─────────────────────────────────────────────────────────────
    orb_key  = f"orb{orb_minutes}"
    orb_hi   = ict.get(f"{orb_key}_hi")
    orb_lo   = ict.get(f"{orb_key}_lo")

    if orb_hi is None or orb_lo is None:
        result["skip_reason"] = "no_orb_data"
        return result

    result["orb_hi"]    = orb_hi
    result["orb_lo"]    = orb_lo
    orb_range = round(orb_hi - orb_lo, 2)
    if orb_range > 500 or orb_range < 0:
        result["orb_range"]   = None
        result["skip_reason"] = "invalid_orb_range"
        return result
    result["orb_range"] = orb_range
    result["rth_open"]  = ict.get("rth_open")

    # ── GEX regime context ────────────────────────────────────────────────────
    result["regime"]      = gex.get("regime")
    result["gamma_flip"]  = gex.get("gamma_flip")
    result["call_wall"]   = gex.get("call_wall")
    result["put_wall"]    = gex.get("put_wall")
    result["air_up"]         = gex.get("air_up")
    result["air_dn"]         = gex.get("air_dn")
    result["wall_magnetism"] = gex.get("wall_magnetism")
    result["bias"]           = gex.get("bias")

    # ── Post-ORB bars ─────────────────────────────────────────────────────────
    post_orb = rth_bars.iloc[orb_minutes:].reset_index(drop=True)
    if post_orb.empty:
        result["skip_reason"] = "no_post_orb_data"
        return result

    # ── ICT level list for sweep scanning ────────────────────────────────────
    from ict import get_all_ict_levels_as_list
    ict_levels = get_all_ict_levels_as_list(ict)

    # ── Sweep detection ───────────────────────────────────────────────────────
    sweep = _detect_sweep(post_orb, ict_levels, orb_hi, orb_lo)

    sweep["bias"] = gex.get("bias")

    result["sweep_occurred"]  = sweep["occurred"]
    result["sweep_direction"] = sweep["direction"]   # "high" or "low" or None
    result["sweep_level"]     = sweep["level"]
    result["sweep_label"]     = sweep["label"]
    result["sweep_bar_idx"]   = sweep["bar_idx"]

    # ── Day type classification ───────────────────────────────────────────────
    day_type = classify_case(gex, sweep["occurred"], sweep["direction"])
    result["day_type"] = day_type

    # ── Trade signal ──────────────────────────────────────────────────────────
    # Only signal if Case 1 (high probability setup)
    if day_type != CASE_1:
        result["trade_signal"] = "skip"
        result["skip_reason"]  = f"day_type_{day_type}"
        # Still compute post-ORB move for reference
        result = _compute_session_stats(result, post_orb, orb_hi, orb_lo)
        return result

    # ── Entry: wait for reclaim of ORB level after sweep ─────────────────────
    entry = _find_entry(post_orb, sweep, orb_hi, orb_lo, sweep["bar_idx"])
    result["entry_price"] = entry["price"]
    result["entry_bar"]   = entry["bar_idx"]
    result["entry_dir"]   = entry["direction"]   # "long" or "short"
    result["trade_signal"] = entry["direction"] or "no_entry"
    if entry["bar_idx"] is not None and sweep["bar_idx"] is not None:
        result["bars_to_entry"] = entry["bar_idx"] - sweep["bar_idx"]
    else:
        result["bars_to_entry"] = None

    if entry["price"] is None:
        result["skip_reason"] = "no_reclaim_entry"
        result = _compute_session_stats(result, post_orb, orb_hi, orb_lo)
        return result

    # ── MFE / MAE / Target / Stop from entry ─────────────────────────────────
    trade = _simulate_trade(
        post_orb, entry, target_pts, stop_pts
    )
    result.update(trade)

    # ── Session stats (regardless of trade) ──────────────────────────────────
    result = _compute_session_stats(result, post_orb, orb_hi, orb_lo)

    return result


# ─── Sweep detection ──────────────────────────────────────────────────────────

def _detect_sweep(
    bars: pd.DataFrame,
    ict_levels: list,
    orb_hi: float,
    orb_lo: float,
) -> dict:
    """
    Scan bars for price sweeping through an ICT level.
    A sweep = price trades THROUGH the level by SWEEP_BUFFER points,
    then closes BACK on the other side within 1-3 bars (engineered stop run).
    """
    null_result = {"occurred": False, "direction": None, "level": None,
                   "label": None, "bar_idx": None}

    # Levels to watch: ONH, ONL, PDH, PDL, equal highs/lows, ORB boundaries
    watch_levels = [(p, lbl) for p, lbl in ict_levels
                    if lbl in ("ONH", "ONL", "PDH", "PDL", "EqHi", "EqLo",
                               "AsiaHi", "AsiaLo", "LonHi", "LonLo")]

    # Add ORB levels
    watch_levels += [(orb_hi, "ORBHi"), (orb_lo, "ORBLo")]
    watch_levels  = sorted(watch_levels, key=lambda x: x[0])

    for i, bar in bars.iterrows():
        hi, lo = bar["high"], bar["low"]

        for level, label in watch_levels:
            # Sweep high (price trades above level then could reverse)
            if hi >= level + SWEEP_BUFFER:
                # Check if close is back below the level (rejection)
                if bar["close"] < level:
                    return {
                        "occurred":  True,
                        "direction": "high",
                        "level":     level,
                        "label":     label,
                        "bar_idx":   i,
                    }

            # Sweep low (price trades below level then could reverse)
            if lo <= level - SWEEP_BUFFER:
                if bar["close"] > level:
                    return {
                        "occurred":  True,
                        "direction": "low",
                        "level":     level,
                        "label":     label,
                        "bar_idx":   i,
                    }

    return null_result


# ─── Entry logic ──────────────────────────────────────────────────────────────

def _find_entry(
    bars: pd.DataFrame,
    sweep: dict,
    orb_hi: float,
    orb_lo: float,
    sweep_bar: Optional[int],
) -> dict:
    """
    After sweep, enter on the next bar's open if it's already on the
    correct side of the level. This gets us in 1 bar after the sweep
    instead of waiting for a close-based reclaim (which is often too late).
    Bias from GEX wall magnetism filters out entries against wall structure.
    """
    null = {"price": None, "bar_idx": None, "direction": None}

    if not sweep["occurred"] or sweep_bar is None:
        return null

    sweep_dir = sweep["direction"]
    bias = sweep.get("bias") or "neutral"

    # Bias filter — don't enter against magnetism direction
    if sweep_dir == "low" and bias == "short":
        return null
    if sweep_dir == "high" and bias == "long":
        return null

    # Look at bars starting from the one right after the sweep
    post_sweep = bars[bars.index > sweep_bar]

    for i, bar in post_sweep.iterrows():
        if sweep_dir == "low":
            # Swept low → expect long → enter on open if already above ORB lo
            if bar["open"] > orb_lo:
                return {
                    "price":     float(bar["open"]),
                    "bar_idx":   i,
                    "direction": "long",
                }
        elif sweep_dir == "high":
            # Swept high → expect short → enter on open if already below ORB hi
            if bar["open"] < orb_hi:
                return {
                    "price":     float(bar["open"]),
                    "bar_idx":   i,
                    "direction": "short",
                }

    return null


# ─── Trade simulation ─────────────────────────────────────────────────────────

def _simulate_trade(
    bars: pd.DataFrame,
    entry: dict,
    target_pts: float,
    stop_pts: float,
) -> dict:
    """
    From entry, track price bar by bar until target or stop is hit.
    Returns MFE, MAE, PnL, hit_target, bars_to_close.
    """
    ep     = entry["price"]
    dirn   = entry["direction"]
    eb     = entry["bar_idx"]

    target = ep + target_pts if dirn == "long" else ep - target_pts
    stop   = ep - stop_pts   if dirn == "long" else ep + stop_pts

    mfe = 0.0
    mae = 0.0
    result = {
        "mfe": 0.0, "mae": 0.0, "pnl": 0.0,
        "hit_target": False, "hit_stop": False,
        "bars_in_trade": 0, "exit_price": None,
    }

    post_entry = bars[bars.index >= eb]

    for n, (i, bar) in enumerate(post_entry.iterrows()):
        # Update MFE/MAE from this bar BEFORE checking exit
        if dirn == "long":
            favorable = bar["high"] - ep
            adverse   = ep - bar["low"]
        else:
            favorable = ep - bar["low"]
            adverse   = bar["high"] - ep

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        # Check target
        if dirn == "long" and bar["high"] >= target:
            result.update({
                "hit_target": True, "exit_price": target,
                "pnl": target_pts, "bars_in_trade": n + 1,
            })
            break
        if dirn == "short" and bar["low"] <= target:
            result.update({
                "hit_target": True, "exit_price": target,
                "pnl": target_pts, "bars_in_trade": n + 1,
            })
            break

        # Check stop
        if dirn == "long" and bar["low"] <= stop:
            result.update({
                "hit_stop": True, "exit_price": stop,
                "pnl": -stop_pts, "bars_in_trade": n + 1,
            })
            break
        if dirn == "short" and bar["high"] >= stop:
            result.update({
                "hit_stop": True, "exit_price": stop,
                "pnl": -stop_pts, "bars_in_trade": n + 1,
            })
            break

    # Cap MFE for winners to avoid inflated values
    if result["hit_target"]:
        mfe = min(mfe, target_pts * 3)

    result["mfe"] = round(mfe, 2)
    result["mae"] = round(mae, 2)

    # End of day without target or stop
    if not result["hit_target"] and not result["hit_stop"] and not post_entry.empty:
        eod_close = float(post_entry["close"].iloc[-1])
        pnl = (eod_close - ep) if dirn == "long" else (ep - eod_close)
        result["exit_price"]    = eod_close
        result["pnl"]           = round(pnl, 2)
        result["bars_in_trade"] = len(post_entry)

    return result


# ─── Session stats ────────────────────────────────────────────────────────────

def _compute_session_stats(result: dict, bars: pd.DataFrame, orb_hi: float, orb_lo: float) -> dict:
    """Add session-level stats regardless of trade."""
    if bars.empty:
        return result
    result["session_range"]   = round(float(bars["high"].max() - bars["low"].min()), 2)
    result["session_hi"]      = float(bars["high"].max())
    result["session_lo"]      = float(bars["low"].min())
    result["session_close"]   = float(bars["close"].iloc[-1])
    return result


# ─── Base result skeleton ─────────────────────────────────────────────────────

def _base_result(trade_date: date, orb_minutes: int) -> dict:
    return {
        "date":             str(trade_date),
        "orb_minutes":      orb_minutes,
        "orb_hi":           None,
        "orb_lo":           None,
        "orb_range":        None,
        "rth_open":         None,
        "regime":           None,
        "gamma_flip":       None,
        "call_wall":        None,
        "put_wall":         None,
        "air_up":           None,
        "air_dn":           None,
        "wall_magnetism":   None,
        "bias":             None,
        "sweep_occurred":   False,
        "sweep_direction":  None,
        "sweep_level":      None,
        "sweep_label":      None,
        "sweep_bar_idx":    None,
        "day_type":         None,
        "trade_signal":     None,
        "entry_price":      None,
        "entry_bar":        None,
        "entry_dir":        None,
        "bars_to_entry":    None,
        "mfe":              None,
        "mae":              None,
        "pnl":              None,
        "hit_target":       None,
        "hit_stop":         None,
        "bars_in_trade":    None,
        "exit_price":       None,
        "session_range":    None,
        "session_hi":       None,
        "session_lo":       None,
        "session_close":    None,
        "skip_reason":      None,
    }
