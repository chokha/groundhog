"""
earnings_calendar.py — Check if major NDX components report earnings today

These 10 stocks = ~55% of NDX weighting.
Any one reporting = treat as high-impact day (GEX map distorted by earnings premium).
"""

import pandas as pd
import yfinance as yf

MAJOR_NDX_COMPONENTS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META",
    "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
]


def check_earnings_today(trade_date) -> dict:
    """
    Check if any major NDX component has earnings on trade_date.

    Returns dict with:
        earnings_today:     list of tickers reporting
        has_major_earnings: bool
        warning:            warning string or None
    """
    earnings_today = []

    for ticker in MAJOR_NDX_COMPONENTS:
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                # yfinance returns earnings date(s) — may be dict or DataFrame
                if isinstance(cal, dict):
                    earn_date = cal.get("Earnings Date")
                else:
                    earn_date = cal.get("Earnings Date")

                if earn_date is not None:
                    # Normalize to list of date objects
                    if not hasattr(earn_date, "__iter__") or isinstance(earn_date, str):
                        earn_date = [earn_date]
                    dates = []
                    for d in earn_date:
                        try:
                            dates.append(pd.Timestamp(d).date())
                        except Exception:
                            pass
                    if trade_date in dates:
                        earnings_today.append(ticker)
        except Exception:
            pass

    has_earnings = len(earnings_today) > 0
    warning = None
    if has_earnings:
        warning = (
            f"EARNINGS: {', '.join(earnings_today)} "
            f"— GEX map distorted, SKIP day"
        )

    return {
        "earnings_today":     earnings_today,
        "has_major_earnings": has_earnings,
        "warning":            warning,
    }
