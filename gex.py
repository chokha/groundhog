"""
gex.py — Compute GEX (Gamma Exposure) map from option chain
Produces: gamma flip, call wall, put wall, secondary shelves, regime, air up/down
"""

import pandas as pd
import numpy as np
from datetime import date as date_type
from typing import Optional

from config import (
    CONTRACT_MULT, NEUTRAL_BAND,
    REGIME_NEGATIVE, REGIME_POSITIVE, REGIME_NEUTRAL,
    MIN_AIR_POINTS
)


def filter_today_expiry(df: pd.DataFrame, today) -> pd.DataFrame:
    """Filter chain to only options expiring today.
    Use expiry date, NOT dte value — ORATS reports dte=1
    at prior close for same-day expiry."""
    if df.empty:
        return df
    d = df.copy()
    d["expiry"] = pd.to_datetime(d["expiry"]).dt.date
    today_date = pd.Timestamp(today).date() if not isinstance(today, date_type) else today
    return d[d["expiry"] == today_date]


def compute_gex_map(
    df: pd.DataFrame,
    spot: float,
    contract_mult: int = CONTRACT_MULT,
    label: str = "",
    dte_min: int = 1,
    dte_max: int = 60,
) -> dict:
    """
    Core GEX computation from an option chain DataFrame.

    Args:
        df:             Option chain with columns: strike, putCall, openInterest, gamma
        spot:           Current underlying price (NQ or QQQ at 9:25 ET)
        contract_mult:  100 for equity options
        label:          "qqq" or "ndx" for logging
        dte_min:        Minimum DTE to include (0 for 0DTE)
        dte_max:        Maximum DTE to include

    Returns dict with:
        gamma_flip      — strike where net cumulative GEX crosses zero
        call_wall       — strike with largest positive GEX contribution
        put_wall        — strike with largest negative GEX contribution
        shelves         — top 5 significant GEX nodes (abs value)
        nearest_above   — closest node above spot (used for air_up)
        nearest_below   — closest node below spot (used for air_dn)
        air_up          — points from spot to next significant node above
        air_dn          — points from spot to next significant node below
        regime          — negative / positive / neutral (relative to spot)
        by_strike       — DataFrame: strike, gex, cum_gex (for charting)
    """
    if df.empty:
        return _empty_gex(label)

    d = df.copy()

    if dte_min == 0 and dte_max == 0:
        d = d[d["dte"] < 1]
        print(f"[GEX] {label}: {len(d)} rows after DTE < 1 (0DTE) filter")
    else:
        d = d[d["dte"].between(dte_min, dte_max)]
        print(f"[GEX] {label}: {len(d)} rows after DTE {dte_min}–{dte_max} filter")
    if d.empty:
        return _empty_gex(label)

    d["oi"]    = d["openInterest"].fillna(0).astype(float)
    d["gamma"] = d["gamma"].fillna(0).astype(float)

    # ── GEX sign convention ───────────────────────────────────────────────────
    # Calls: dealers are SHORT calls → they BUY as price rises → positive GEX
    # Puts:  dealers are SHORT puts  → they SELL as price falls → negative GEX
    is_put = d["putCall"].str.upper().str.startswith("P")
    sign   = (~is_put).astype(float) * 2 - 1    # +1 calls, -1 puts

    # Standard GEX formula: gamma * OI * multiplier * spot²
    d["gex"] = d["gamma"] * d["oi"] * contract_mult * (spot ** 2) * sign

    # ── Aggregate by strike ───────────────────────────────────────────────────
    by_strike = (
        d.groupby("strike", as_index=False)["gex"]
        .sum()
        .sort_values("strike")
        .reset_index(drop=True)
    )
    by_strike["cum_gex"] = by_strike["gex"].cumsum()

    # ── Gamma flip: strike where cumulative net GEX crosses zero ─────────────
    gamma_flip = _find_zero_crossing(by_strike)

    # GLOBAL walls
    call_wall_global = float(by_strike.loc[by_strike["gex"].idxmax(), "strike"])
    put_wall_global  = float(by_strike.loc[by_strike["gex"].idxmin(), "strike"])

    # ACTIVE walls ±1.5% of spot (ORB critical)
    pct_active = 0.015
    lo_a, hi_a = spot * (1 - pct_active), spot * (1 + pct_active)
    near_active = by_strike[(by_strike["strike"] >= lo_a) & (by_strike["strike"] <= hi_a)]

    if near_active.empty:
        call_wall_active = call_wall_global
        put_wall_active  = put_wall_global
    else:
        call_wall_active = float(near_active.loc[near_active["gex"].idxmax(), "strike"])
        put_wall_active  = float(near_active.loc[near_active["gex"].idxmin(), "strike"])

    # STRUCTURAL walls ±4% of spot (context)
    pct_struct = 0.04
    lo_s, hi_s = spot * (1 - pct_struct), spot * (1 + pct_struct)
    near_struct = by_strike[(by_strike["strike"] >= lo_s) & (by_strike["strike"] <= hi_s)]

    if near_struct.empty:
        call_wall_struct = call_wall_global
        put_wall_struct  = put_wall_global
    else:
        call_wall_struct = float(near_struct.loc[near_struct["gex"].idxmax(), "strike"])
        put_wall_struct  = float(near_struct.loc[near_struct["gex"].idxmin(), "strike"])

    # Backward compatible
    call_wall = call_wall_active
    put_wall  = put_wall_active

    # Shelves from active window
    top_k = 8
    if near_active.empty:
        top_active = by_strike.reindex(
            by_strike["gex"].abs().sort_values(ascending=False).index).head(top_k)
    else:
        top_active = near_active.reindex(
            near_active["gex"].abs().sort_values(ascending=False).index).head(top_k)
    shelves_active = sorted(top_active["strike"].tolist())

    if near_struct.empty:
        top_struct = by_strike.reindex(
            by_strike["gex"].abs().sort_values(ascending=False).index).head(top_k)
    else:
        top_struct = near_struct.reindex(
            near_struct["gex"].abs().sort_values(ascending=False).index).head(top_k)
    shelves_struct = sorted(top_struct["strike"].tolist())

    nodes_active = sorted(set(
        shelves_active +
        [call_wall_active, put_wall_active] +
        ([gamma_flip] if gamma_flip is not None else [])
    ))
    nodes_struct = sorted(set(
        shelves_struct +
        [call_wall_struct, put_wall_struct] +
        ([gamma_flip] if gamma_flip is not None else [])
    ))

    # ── Air: distance from spot to next node above/below ─────────────────────
    nodes_above = [n for n in nodes_active if n > spot + 5]
    nodes_below = [n for n in nodes_active if n < spot - 5]
    support_above  = bool(nodes_above)
    support_below  = bool(nodes_below)
    nearest_above  = min(nodes_above) if nodes_above else None
    nearest_below  = max(nodes_below) if nodes_below else None
    air_up = round(nearest_above - spot, 1) if nearest_above is not None else None
    air_dn = round(spot - nearest_below, 1) if nearest_below is not None else None
    all_nodes_above_spot = bool(nodes_active) and all(n > spot for n in nodes_active)

    # ── Wall magnetism ────────────────────────────────────────────────────────
    sig_nodes = sorted(set(shelves_active + [call_wall_active, put_wall_active]))
    walls_above = len([n for n in sig_nodes if n > spot])
    walls_below = len([n for n in sig_nodes if n < spot])
    if walls_above > walls_below * 2:
        wall_magnetism = "up"
    elif walls_below > walls_above * 2:
        wall_magnetism = "down"
    else:
        wall_magnetism = "neutral"
    nearest_wall_above = min([n for n in sig_nodes if n > spot], default=None)
    nearest_wall_below = max([n for n in sig_nodes if n < spot], default=None)
    wall_imbalance = round(walls_above / max(walls_below, 1), 2)

    # ── Regime relative to spot ───────────────────────────────────────────────
    if gamma_flip is None:
        total_net_gex = by_strike["gex"].sum()
        regime = REGIME_NEGATIVE if total_net_gex < 0 else REGIME_POSITIVE
    elif abs(spot - gamma_flip) <= NEUTRAL_BAND:
        regime = REGIME_NEUTRAL
    elif spot < gamma_flip:
        regime = REGIME_NEGATIVE
    else:
        regime = REGIME_POSITIVE

    # ── Bias: regime + wall magnetism combined signal ──────────────────────
    if regime == REGIME_NEGATIVE and wall_magnetism == "up":
        bias = "long"
    elif regime == REGIME_NEGATIVE and wall_magnetism == "down":
        bias = "short"
    elif regime == REGIME_POSITIVE and wall_magnetism == "up":
        bias = "fade_short"
    elif regime == REGIME_POSITIVE and wall_magnetism == "down":
        bias = "fade_long"
    else:
        bias = "neutral"

    # ── Inventory zone (call/put wall bounds) ────────────────────────────────
    inv_low  = min(call_wall_active, put_wall_active)
    inv_high = max(call_wall_active, put_wall_active)
    inside_inventory = inv_low <= spot <= inv_high
    if inside_inventory:
        dist_to_inv = 0.0
    elif spot > inv_high:
        dist_to_inv = round(spot - inv_high, 2)
    else:
        dist_to_inv = round(inv_low - spot, 2)
    vacuum_up = spot > inv_high
    vacuum_dn = spot < inv_low

    return {
        "label":       label,
        "gamma_flip":  gamma_flip,
        "call_wall":         call_wall_active,
        "put_wall":          put_wall_active,
        "shelves":           shelves_active,
        "all_nodes":         nodes_active,
        "call_wall_active":  call_wall_active,
        "put_wall_active":   put_wall_active,
        "call_wall_struct":  call_wall_struct,
        "put_wall_struct":   put_wall_struct,
        "call_wall_global":  call_wall_global,
        "put_wall_global":   put_wall_global,
        "shelves_active":    shelves_active,
        "shelves_struct":    shelves_struct,
        "nodes_active":      nodes_active,
        "nodes_struct":      nodes_struct,
        "nearest_above": nearest_above,
        "nearest_below": nearest_below,
        "air_up":      air_up,
        "air_dn":      air_dn,
        "support_above": support_above,
        "support_below": support_below,
        "all_nodes_above_spot": all_nodes_above_spot,
        "wall_magnetism": wall_magnetism,
        "walls_above_count": walls_above,
        "walls_below_count": walls_below,
        "wall_imbalance": wall_imbalance,
        "nearest_wall_above": nearest_wall_above,
        "nearest_wall_below": nearest_wall_below,
        "regime":      regime,
        "bias":        bias,
        # Inventory zone
        "inv_low":              inv_low,
        "inv_high":             inv_high,
        "inside_inventory":     inside_inventory,
        "vacuum_up":            vacuum_up,
        "vacuum_dn":            vacuum_dn,
        "dist_to_inv":          dist_to_inv,
        "call_wall_inventory":  call_wall_active,
        "put_wall_inventory":   put_wall_active,
        "spot":        spot,
        "by_strike":   by_strike,
    }


def combine_qqq_ndx_gex(qqq_gex: dict, ndx_gex: dict, spot_nq: float, basis: float = 0) -> dict:
    """
    Combine QQQ and NDX GEX maps into a unified level set.
    NDX is primary — same price scale as NQ futures (~20,000).
    NDX levels shifted to NQ scale using basis (NQ - NDX premium).
    Air recomputed vs NQ spot after basis shift.
    QQQ is secondary for confluence confirmation only.
    """
    def to_nq(level):
        return round(level + basis, 2) if level is not None else None

    # Shift NDX nodes to NQ scale
    ndx_nodes = sorted(set(
        (ndx_gex.get("nodes_active") or ndx_gex.get("shelves") or []) +
        [x for x in [
            ndx_gex.get("gamma_flip"),
            ndx_gex.get("call_wall_active"),
            ndx_gex.get("put_wall_active"),
        ] if x is not None]
    ))
    nq_nodes  = [to_nq(n) for n in ndx_nodes]

    # Recompute air and nearest nodes from NQ-shifted nodes vs NQ spot
    nodes_above = [n for n in nq_nodes if n is not None and n > spot_nq + 5]
    nodes_below = [n for n in nq_nodes if n is not None and n < spot_nq - 5]
    nearest_above = min(nodes_above) if nodes_above else None
    nearest_below = max(nodes_below) if nodes_below else None
    air_up = round(nearest_above - spot_nq, 1) if nearest_above is not None else None
    air_dn = round(spot_nq - nearest_below, 1) if nearest_below is not None else None

    result = {
        "primary":        ndx_gex,
        "secondary":      qqq_gex,
        "regime":         ndx_gex.get("regime"),
        "gamma_flip":     to_nq(ndx_gex.get("gamma_flip")),
        "call_wall":         to_nq(ndx_gex.get("call_wall_active") or ndx_gex.get("call_wall")),
        "put_wall":          to_nq(ndx_gex.get("put_wall_active")  or ndx_gex.get("put_wall")),
        "call_wall_struct":  to_nq(ndx_gex.get("call_wall_struct")),
        "put_wall_struct":   to_nq(ndx_gex.get("put_wall_struct")),
        "call_wall_global":  to_nq(ndx_gex.get("call_wall_global")),
        "put_wall_global":   to_nq(ndx_gex.get("put_wall_global")),
        "nearest_above":  nearest_above,
        "nearest_below":  nearest_below,
        "air_up":         air_up,
        "air_dn":         air_dn,
        "wall_magnetism": ndx_gex.get("wall_magnetism"),
        "bias":           ndx_gex.get("bias"),
        "shelves":        nq_nodes,
        "basis":          basis,
        "spot_nq":        spot_nq,
        # Inventory zone (NQ-shifted)
        "inv_low":           to_nq(ndx_gex.get("inv_low")),
        "inv_high":          to_nq(ndx_gex.get("inv_high")),
        "call_wall_inventory": to_nq(ndx_gex.get("call_wall_inventory")),
        "put_wall_inventory":  to_nq(ndx_gex.get("put_wall_inventory")),
        # QQQ confluence
        "qqq_regime":     qqq_gex.get("regime"),
        "regime_confluence": (
            ndx_gex.get("regime") == qqq_gex.get("regime")
        ),
    }

    # Recompute inventory zone vs NQ spot (after basis shift)
    inv_low_nq  = result["inv_low"]
    inv_high_nq = result["inv_high"]
    if inv_low_nq is not None and inv_high_nq is not None:
        inside = inv_low_nq <= spot_nq <= inv_high_nq
        result["inside_inventory"] = inside
        if inside:
            result["dist_to_inv"] = 0.0
        elif spot_nq > inv_high_nq:
            result["dist_to_inv"] = round(spot_nq - inv_high_nq, 2)
        else:
            result["dist_to_inv"] = round(inv_low_nq - spot_nq, 2)
        result["vacuum_up"] = spot_nq > inv_high_nq
        result["vacuum_dn"] = spot_nq < inv_low_nq
    else:
        result["inside_inventory"] = False
        result["dist_to_inv"] = 0.0
        result["vacuum_up"] = False
        result["vacuum_dn"] = False

    return result


def _find_zero_crossing(by_strike: pd.DataFrame) -> Optional[float]:
    """Find strike where cumulative GEX crosses zero (gamma flip)."""
    s = by_strike.reset_index(drop=True)
    for i in range(1, len(s)):
        prev = s.loc[i-1, "cum_gex"]
        curr = s.loc[i,   "cum_gex"]
        if (prev <= 0 and curr >= 0) or (prev >= 0 and curr <= 0):
            # Linear interpolation for precision
            strike_prev = s.loc[i-1, "strike"]
            strike_curr = s.loc[i,   "strike"]
            if curr != prev:
                frac = abs(prev) / (abs(prev) + abs(curr))
                return round(float(strike_prev + frac * (strike_curr - strike_prev)), 2)
            return float(strike_curr)
    return None


def _empty_gex(label: str = "") -> dict:
    """Return empty GEX dict when chain data unavailable."""
    return {
        "label":             label,
        "gamma_flip":        None,
        "call_wall":         None,
        "put_wall":          None,
        "shelves":           [],
        "all_nodes":         [],
        "call_wall_active":  None,
        "put_wall_active":   None,
        "call_wall_struct":  None,
        "put_wall_struct":   None,
        "call_wall_global":  None,
        "put_wall_global":   None,
        "shelves_active":    [],
        "shelves_struct":    [],
        "nodes_active":      [],
        "nodes_struct":      [],
        "nearest_above":     None,
        "nearest_below":     None,
        "air_up":            None,
        "air_dn":            None,
        "support_above":     False,
        "support_below":     False,
        "all_nodes_above_spot": False,
        "walls_above_count": 0,
        "walls_below_count": 0,
        "wall_imbalance":    1.0,
        "nearest_wall_above": None,
        "nearest_wall_below": None,
        "regime":            REGIME_NEUTRAL,
        "bias":              "neutral",
        "wall_magnetism":    "neutral",
        # Inventory zone
        "inv_low":           None,
        "inv_high":          None,
        "inside_inventory":  False,
        "vacuum_up":         False,
        "vacuum_dn":         False,
        "dist_to_inv":       0.0,
        "call_wall_inventory": None,
        "put_wall_inventory":  None,
        "spot":              None,
        "by_strike":         pd.DataFrame(),
    }


def classify_case(gex: dict, sweep_occurred: bool, sweep_direction: Optional[str]) -> str:
    """
    Classify day type based on GEX regime + sweep outcome.
    
    Returns: "case_1", "case_2", or "case_3"
    """
    from config import CASE_1, CASE_2, CASE_3

    regime  = gex.get("regime")
    air_up  = gex.get("air_up") or 0
    air_dn  = gex.get("air_dn") or 0

    if not sweep_occurred:
        return CASE_3

    # Sweep occurred — regime determines expansion vs trap
    if regime == REGIME_NEGATIVE:
        # Check there's enough air in the direction of potential move
        if sweep_direction == "low" and air_up >= MIN_AIR_POINTS:
            return CASE_1   # sweep low + neg gamma + air up = expansion long
        if sweep_direction == "high" and air_dn >= MIN_AIR_POINTS:
            return CASE_1   # sweep high + neg gamma + air dn = expansion short
        return CASE_2       # neg gamma but no air = still trappy

    elif regime == REGIME_POSITIVE:
        return CASE_2       # sweep into positive gamma = trap, expect reversal

    else:
        return CASE_3       # neutral regime, treat as drift
