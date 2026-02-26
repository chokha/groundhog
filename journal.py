"""
journal.py — Post-game journal generator.

Computes outcomes from 1m NQ bars, matches playbook patterns, and saves
a journal entry for the trading day.

Usage:
    python journal.py --date 2026-02-03
    python journal.py --date 2026-02-03 --macro "tariff selloff" --lesson "Gap traps"
"""

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from dashboard import _load_archive, _prepare_yf_bars
from playbook import match_patterns

JOURNAL_DIR = Path(__file__).parent / "journal"
BRIEF_DIR = Path(__file__).parent / "daily_briefs"


# ─── Bar loading ─────────────────────────────────────────────────────────────

def load_bars(trade_date):
    """
    Load NQ 1m bars for trade_date from best available source:
    archive → nq_1m.parquet → nq_1m_databento.parquet → yfinance.
    Returns DataFrame with columns: ts, open, high, low, close, volume, date.
    """
    td = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))
    fetch_start = str(td - timedelta(days=1))
    fetch_end = str(td + timedelta(days=1))

    # 1. Archive
    df = _load_archive("NQ=F", fetch_start, fetch_end)
    if not df.empty:
        day_bars = df[df["date"] == td]
        if not day_bars.empty:
            print(f"  Loaded {len(day_bars)} bars from archive")
            return day_bars

    # 2. nq_1m.parquet
    p = Path("nq_1m.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"])
        if "date" not in df.columns:
            df["date"] = df["ts"].dt.date
        day_bars = df[df["date"] == td]
        if not day_bars.empty:
            print(f"  Loaded {len(day_bars)} bars from nq_1m.parquet")
            return day_bars

    # 3. nq_1m_databento.parquet
    p = Path("nq_1m_databento.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"])
        if "date" not in df.columns:
            df["date"] = df["ts"].dt.date
        day_bars = df[df["date"] == td]
        if not day_bars.empty:
            print(f"  Loaded {len(day_bars)} bars from nq_1m_databento.parquet")
            return day_bars

    # 4. yfinance fallback
    print(f"  Fetching NQ=F 1m bars from yfinance for {td}...")
    raw = yf.Ticker("NQ=F").history(start=fetch_start, end=fetch_end, interval="1m")
    if not raw.empty:
        df = _prepare_yf_bars(raw)
        day_bars = df[df["date"] == td]
        if not day_bars.empty:
            print(f"  Loaded {len(day_bars)} bars from yfinance")
            return day_bars

    return pd.DataFrame()


# ─── Outcome computation ────────────────────────────────────────────────────

def _price_at_time(bars, target_time):
    """Get close price at or just before target_time. Returns None if not found."""
    mask = bars["ts"].dt.time <= target_time
    subset = bars[mask]
    if subset.empty:
        return None
    return round(float(subset.iloc[-1]["close"]), 2)


def compute_outcomes(bars_1m, trade_date, brief):
    """
    Compute trading outcomes from 1m bars.

    Returns dict with: ORB stats, checkpoint prices, moves, MFE/MAE, target hits.
    """
    td = trade_date if isinstance(trade_date, date) else date.fromisoformat(str(trade_date))

    # Filter to RTH only (9:30 - 16:00)
    rth = bars_1m[
        (bars_1m["ts"].dt.time >= time(9, 30)) &
        (bars_1m["ts"].dt.time <= time(16, 0))
    ].copy()

    if rth.empty:
        return {"error": "No RTH bars found"}

    rth_open = round(float(rth.iloc[0]["open"]), 2)

    # ── ORB (first 5 bars: 9:30-9:34) ────────────────────────────────────────
    orb_bars = rth[rth["ts"].dt.time < time(9, 35)]
    if orb_bars.empty:
        return {"error": "No ORB bars found", "rth_open": rth_open}

    orb_high = round(float(orb_bars["high"].max()), 2)
    orb_low = round(float(orb_bars["low"].min()), 2)
    orb_range = round(orb_high - orb_low, 2)

    # ── ORB break detection (first close beyond ORB range after 9:34) ─────────
    post_orb = rth[rth["ts"].dt.time >= time(9, 35)]
    orb_break_dir = None
    orb_break_time = None
    orb_break_price = None

    for _, bar in post_orb.iterrows():
        if bar["close"] > orb_high:
            orb_break_dir = "long"
            orb_break_time = bar["ts"].strftime("%H:%M")
            orb_break_price = round(float(bar["close"]), 2)
            break
        elif bar["close"] < orb_low:
            orb_break_dir = "short"
            orb_break_time = bar["ts"].strftime("%H:%M")
            orb_break_price = round(float(bar["close"]), 2)
            break

    # ── Checkpoint prices ─────────────────────────────────────────────────────
    checkpoints = {
        "price_0935": _price_at_time(rth, time(9, 35)),
        "price_0945": _price_at_time(rth, time(9, 45)),
        "price_1000": _price_at_time(rth, time(10, 0)),
        "price_1030": _price_at_time(rth, time(10, 30)),
        "price_1100": _price_at_time(rth, time(11, 0)),
        "price_eod": _price_at_time(rth, time(16, 0)),
    }

    # ── Moves from RTH open ──────────────────────────────────────────────────
    moves = {}
    for key, price in checkpoints.items():
        move_key = key.replace("price_", "move_")
        if price is not None:
            moves[move_key] = round(price - rth_open, 2)
        else:
            moves[move_key] = None

    # ── MFE / MAE from ORB break price ───────────────────────────────────────
    mfe = None
    mae = None
    if orb_break_price is not None and not post_orb.empty:
        after_break = post_orb[post_orb["ts"].dt.strftime("%H:%M") >= orb_break_time] if orb_break_time else post_orb
        if not after_break.empty:
            if orb_break_dir == "long":
                mfe = round(float(after_break["high"].max()) - orb_break_price, 2)
                mae = round(orb_break_price - float(after_break["low"].min()), 2)
            elif orb_break_dir == "short":
                mfe = round(orb_break_price - float(after_break["low"].min()), 2)
                mae = round(float(after_break["high"].max()) - orb_break_price, 2)

    # ── Target hits ──────────────────────────────────────────────────────────
    suggested_target = brief.get("suggested_target", 80)
    runner_target = suggested_target * 2
    t1_hit = False
    runner_hit = False
    if mfe is not None:
        t1_hit = mfe >= suggested_target
        runner_hit = mfe >= runner_target

    # ── RTH range ────────────────────────────────────────────────────────────
    rth_high = round(float(rth["high"].max()), 2)
    rth_low = round(float(rth["low"].min()), 2)
    day_range = round(rth_high - rth_low, 2)

    result = {
        "rth_open": rth_open,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_range": orb_range,
        "orb_break_dir": orb_break_dir,
        "orb_break_time": orb_break_time,
        **checkpoints,
        **moves,
        "mfe_from_orb_break": mfe,
        "mae_from_orb_break": mae,
        "t1_hit": t1_hit,
        "runner_hit": runner_hit,
        "rth_high": rth_high,
        "rth_low": rth_low,
        "day_range": day_range,
    }

    return result


# ─── Journal generation ─────────────────────────────────────────────────────

def generate_journal(date_str, macro=None, lesson=None):
    """
    Generate a journal entry for the given date.

    Loads brief JSON, loads 1m bars, computes outcomes, matches playbook,
    and saves to journal/YYYY-MM-DD.json.
    """
    td = date.fromisoformat(date_str)

    # Load brief
    brief_path = BRIEF_DIR / f"brief_{date_str}.json"
    if not brief_path.exists():
        print(f"  [ERROR] No brief found at {brief_path}")
        print(f"  Run: python dashboard.py --date {date_str}")
        sys.exit(1)

    with open(brief_path) as f:
        brief = json.load(f)

    print(f"\n  Journal: {date_str}")
    print(f"  Brief:   {brief_path}")

    # Load bars
    bars = load_bars(td)
    if bars.empty:
        print(f"  [ERROR] No 1m bars found for {date_str}")
        sys.exit(1)

    # Compute outcomes
    outcomes = compute_outcomes(bars, td, brief)
    if "error" in outcomes:
        print(f"  [WARN] Outcome computation issue: {outcomes['error']}")

    # Match playbook patterns
    playbook_matches = match_patterns(brief)

    # Build journal entry
    brief_subset = {
        k: brief.get(k) for k in [
            "spot_nq", "vix_price", "vix_regime", "regime", "premarket_bias",
            "gap_from_pdc", "premarket_range", "inside_inventory",
            "PDH", "PDL", "PDC", "ONH", "ONL",
            "suggested_target", "has_major_earnings", "trade_state", "mode",
        ]
    }

    entry = {
        "date": date_str,
        "brief": brief_subset,
        "outcome": outcomes,
        "matched_patterns": [m["pattern_id"] for m in playbook_matches],
        "matched_patterns_detail": playbook_matches,
        "macro_context": macro,
        "lesson": lesson,
    }

    # Save
    JOURNAL_DIR.mkdir(exist_ok=True)
    out_path = JOURNAL_DIR / f"{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2, default=str)

    print(f"  Saved:   {out_path}")
    _print_journal_summary(entry)

    return entry


# ─── Terminal output ─────────────────────────────────────────────────────────

def _print_journal_summary(entry):
    """Print formatted journal summary to terminal."""
    o = entry.get("outcome", {})
    b = entry.get("brief", {})

    print(f"\n  ── POST-GAME JOURNAL: {entry['date']} ──────────────────────")
    print(f"  RTH Open: {o.get('rth_open', 'N/A')}")
    print(f"  ORB:      {o.get('orb_high', '?')} / {o.get('orb_low', '?')}  "
          f"(range: {o.get('orb_range', '?')})")

    brk = o.get("orb_break_dir", "none")
    brk_time = o.get("orb_break_time", "?")
    print(f"  Break:    {brk.upper() if brk else 'NONE'} @ {brk_time}")

    print(f"\n  Checkpoints (move from RTH open):")
    for label, key in [("09:35", "move_0935"), ("09:45", "move_0945"),
                       ("10:00", "move_1000"), ("10:30", "move_1030"),
                       ("11:00", "move_1100"), ("EOD",   "move_eod")]:
        val = o.get(key)
        if val is not None:
            print(f"    {label}: {val:+.2f}")

    mfe = o.get("mfe_from_orb_break")
    mae = o.get("mae_from_orb_break")
    if mfe is not None:
        print(f"\n  MFE: {mfe:+.2f}  MAE: {mae:+.2f}")

    t1 = "HIT" if o.get("t1_hit") else "MISS"
    runner = "HIT" if o.get("runner_hit") else "MISS"
    target = b.get("suggested_target", "?")
    print(f"  T1 ({target} pts): {t1}  |  Runner ({target}x2): {runner}")

    print(f"\n  Day range: {o.get('day_range', '?')}  "
          f"(H: {o.get('rth_high', '?')}  L: {o.get('rth_low', '?')})")

    patterns = entry.get("matched_patterns", [])
    if patterns:
        print(f"\n  Matched patterns: {', '.join(patterns)}")

    macro = entry.get("macro_context")
    if macro:
        print(f"  Macro: {macro}")

    lesson = entry.get("lesson")
    if lesson:
        print(f"  Lesson: {lesson}")

    print(f"  ──────────────────────────────────────────────────────────\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate post-game journal entry")
    parser.add_argument("--date", required=True, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--macro", default=None, help="Macro/news context")
    parser.add_argument("--lesson", default=None, help="Lesson learned")
    args = parser.parse_args()

    generate_journal(args.date, macro=args.macro, lesson=args.lesson)


if __name__ == "__main__":
    main()
