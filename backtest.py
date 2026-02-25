"""
backtest.py — Main backtest loop
Runs GEX + ICT + ORB simulation over full NQ parquet date range.
Outputs per-day CSV and summary statistics.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from config import (
    NQ_PARQUET_PATH, QQQ_TICKER, NDX_TICKER,
    RTH_OPEN, RTH_CLOSE,
    CASE_1, CASE_2, CASE_3,
)
from orats import fetch_qqq_and_ndx, get_prior_trading_date
from gex import compute_gex_map, combine_qqq_ndx_gex
from ict import load_nq_data, compute_ict_levels, get_spots
from orb import simulate_orb_day


def run_backtest(
    parquet_path: str = NQ_PARQUET_PATH,
    output_dir:   str = "backtest_output",
    start_date:   str = None,
    end_date:     str = None,
) -> pd.DataFrame:
    """
    Full backtest over date range in NQ parquet.
    
    Args:
        parquet_path:  Path to NQ 1-min OHLCV parquet
        output_dir:    Directory to write results CSVs
        start_date:    "YYYY-MM-DD" (default: all data)
        end_date:      "YYYY-MM-DD" (default: all data)
    
    Returns:
        DataFrame of per-day results (both 5-min and 15-min ORB)
    """
    Path(output_dir).mkdir(exist_ok=True)

    # ── Load NQ data ──────────────────────────────────────────────────────────
    print("Loading NQ 1-min data...")
    nq = load_nq_data(parquet_path)
    print(f"  Loaded {len(nq):,} bars, {nq['date'].nunique()} trading days")
    print(f"  Date range: {nq['date'].min()} → {nq['date'].max()}")

    # ── Filter date range ─────────────────────────────────────────────────────
    all_dates = sorted(nq["date"].unique())
    if start_date:
        all_dates = [d for d in all_dates if str(d) >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if str(d) <= end_date]

    print(f"\nRunning backtest over {len(all_dates)} trading days...\n")

    all_results = []
    skipped = []

    for i, trade_date in enumerate(all_dates):
        td_str = str(trade_date)
        print(f"  [{i+1}/{len(all_dates)}] {td_str}", end="")

        try:
            # ── Pull ORATS chains (prior close OI) ─────────────────────────
            prior_date = get_prior_trading_date(trade_date)
            chains     = fetch_qqq_and_ndx(str(prior_date), use_cache=True)
            qqq_chain  = chains["qqq"]
            ndx_chain  = chains["ndx"]

            # ── Pre-market spots for each instrument ──────────────────────
            ict = compute_ict_levels(nq, trade_date)
            spots = get_spots(nq, trade_date)
            spot_nq  = spots["spot_nq"]
            spot_qqq = spots["spot_qqq"]
            spot_ndx = spots["spot_ndx"]
            basis    = spots["basis"]

            if spot_nq is None:
                print(f" → SKIP (no pre-market spot)")
                skipped.append({"date": td_str, "reason": "no_spot_925"})
                continue

            # ── Compute GEX maps on native scale per instrument ────────────
            ndx_gex = compute_gex_map(ndx_chain, spot_ndx, label="ndx")
            qqq_gex = compute_gex_map(qqq_chain, spot_qqq, label="qqq")
            gex     = combine_qqq_ndx_gex(qqq_gex, ndx_gex, spot_nq, basis)

            # ── RTH bars for this day ────────────────────────────────────────
            rth_bars = nq[
                (nq["date"] == trade_date) &
                (nq["ts"].dt.time >= RTH_OPEN) &
                (nq["ts"].dt.time <= RTH_CLOSE)
            ].copy().reset_index(drop=True)

            # ── Simulate both ORB timeframes ─────────────────────────────────
            for orb_mins in [5, 15]:
                res = simulate_orb_day(rth_bars, ict, gex, trade_date, orb_minutes=orb_mins)
                # Add confluence flags
                res["regime_confluence"] = gex.get("regime_confluence")
                res["qqq_flip"]          = gex.get("qqq_flip")
                res["qqq_regime"]        = gex.get("qqq_regime")
                # ICT context
                res["PDH"]               = ict.get("PDH")
                res["PDL"]               = ict.get("PDL")
                res["PDC"]               = ict.get("PDC")
                res["ONH"]               = ict.get("ONH")
                res["ONL"]               = ict.get("ONL")
                res["london_hi"]         = ict.get("london_hi")
                res["london_lo"]         = ict.get("london_lo")
                res["spot_925"]          = spot_nq

                all_results.append(res)

            regime = gex.get("regime", "?")
            day_types = [r["day_type"] for r in all_results[-2:]]
            print(f" → regime={regime} | 5m={day_types[0]} | 15m={day_types[1]}")

        except Exception as e:
            print(f" → ERROR: {e}")
            skipped.append({"date": td_str, "reason": str(e)})
            continue

    # ── Build results DataFrame ────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    if df_results.empty:
        print("\nNo results generated.")
        return df_results

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_results.to_csv(f"{output_dir}/backtest_all.csv", index=False)
    
    df_5m  = df_results[df_results["orb_minutes"] == 5]
    df_15m = df_results[df_results["orb_minutes"] == 15]
    df_5m.to_csv(f"{output_dir}/backtest_5m.csv",  index=False)
    df_15m.to_csv(f"{output_dir}/backtest_15m.csv", index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(f"{output_dir}/skipped_days.csv", index=False)

    # ── Print summary ──────────────────────────────────────────────────────────
    print_summary(df_5m, label="5-min ORB")
    print_summary(df_15m, label="15-min ORB")

    return df_results


def print_summary(df: pd.DataFrame, label: str = ""):
    """Print backtest performance summary."""
    if df.empty:
        return

    print(f"\n{'═'*55}")
    print(f"  BACKTEST SUMMARY — {label}")
    print(f"{'═'*55}")

    total_days  = len(df)
    case1       = df[df["day_type"] == CASE_1]
    case2       = df[df["day_type"] == CASE_2]
    case3       = df[df["day_type"] == CASE_3]
    traded      = df[df["entry_price"].notna()]

    print(f"  Total days:      {total_days}")
    print(f"  Case 1 days:     {len(case1)} ({len(case1)/total_days*100:.0f}%)")
    print(f"  Case 2 days:     {len(case2)} ({len(case2)/total_days*100:.0f}%)")
    print(f"  Case 3 days:     {len(case3)} ({len(case3)/total_days*100:.0f}%)")

    if not traded.empty:
        winners = traded[traded["pnl"] > 0]
        losers  = traded[traded["pnl"] <= 0]
        win_rate = len(winners) / len(traded) * 100 if len(traded) > 0 else 0

        print(f"\n  Trades executed: {len(traded)}")
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg winner:      {winners['pnl'].mean():.1f} pts" if not winners.empty else "  Avg winner:      —")
        print(f"  Avg loser:       {losers['pnl'].mean():.1f} pts"  if not losers.empty  else "  Avg loser:       —")
        print(f"  Avg MFE:         {traded['mfe'].mean():.1f} pts")
        print(f"  Avg MAE:         {traded['mae'].mean():.1f} pts")
        print(f"  Total PnL:       {traded['pnl'].sum():.1f} pts")
        print(f"  Hit 100pt tgt:   {traded['hit_target'].sum()} times ({traded['hit_target'].mean()*100:.0f}%)")

    # Regime breakdown
    print(f"\n  Regime breakdown:")
    for regime in ["negative", "positive", "neutral"]:
        r = df[df["regime"] == regime]
        if not r.empty:
            traded_r = r[r["entry_price"].notna()]
            wr = traded_r[traded_r["pnl"] > 0].shape[0] / len(traded_r) * 100 if len(traded_r) > 0 else 0
            print(f"    {regime:10s}: {len(r)} days, {len(traded_r)} trades, {wr:.0f}% WR")

    print(f"{'═'*55}\n")


if __name__ == "__main__":
    import sys
    parquet = sys.argv[1] if len(sys.argv) > 1 else NQ_PARQUET_PATH
    run_backtest(parquet_path=parquet)
