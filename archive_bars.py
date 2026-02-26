"""archive_bars.py — Download and archive 1m bars to local parquet."""
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import pytz

ARCHIVE_DIR = Path(__file__).parent / "archive"
TICKERS = {
    "NQ=F":  "nq_1m_archive.parquet",
    "QQQ":   "qqq_1m_archive.parquet",
    "^VIX":  "vix_1m_archive.parquet",
}

def archive_ticker(ticker: str, filename: str):
    """Download recent 1m bars and append to archive parquet, deduping by ts."""
    path = ARCHIVE_DIR / filename
    # Download last 30 days of 1m data (max available)
    raw = yf.Ticker(ticker).history(period="30d", interval="1m", prepost=True)
    if raw.empty:
        print(f"  {ticker}: no data returned, skipping")
        return
    # Prepare: rename columns, tz-convert to ET, add date
    df = raw.reset_index()
    col_map = {"Datetime": "ts", "Date": "ts",
               "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df = df[["ts", "open", "high", "low", "close", "volume", "date"]]

    # Merge with existing archive
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts"], keep="last")
        combined = combined.sort_values("ts").reset_index(drop=True)
    else:
        combined = df.sort_values("ts").reset_index(drop=True)

    combined.to_parquet(path, index=False)
    date_range = f"{combined['date'].min()} to {combined['date'].max()}"
    print(f"  {ticker}: {len(combined)} rows archived ({date_range})")

def main():
    ARCHIVE_DIR.mkdir(exist_ok=True)
    print(f"Archiving 1m bars at {datetime.now(pytz.timezone('America/New_York'))}")
    for ticker, filename in TICKERS.items():
        archive_ticker(ticker, filename)
    print("Done.")

if __name__ == "__main__":
    main()
