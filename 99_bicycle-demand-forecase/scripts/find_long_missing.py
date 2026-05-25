"""
Long missing period detection script for Ddareungi (Seoul Bike) rental stations.
- A day is considered missing if no data exists for all 24 hours at a station.
- Outputs consecutive missing periods >= LONG_MISSING_TH days per station.
- Prints results as a LONG_MISSING dict ready to copy-paste into the notebook.
"""
 
from pathlib import Path
import pandas as pd
 
# ── Config ───────────────────────────────────────────────────
DATA_DIR        = Path("data")   # directory containing filtered_YYYYMM.csv files
LONG_MISSING_TH = 15             # threshold for long missing (days)
 
 
# ── Data loading ─────────────────────────────────────────────
def load_filtered_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("filtered_2*.csv"))
    if not files:
        raise FileNotFoundError(f"No filtered_*.csv files found in: {data_dir}")
    parts = [
        pd.read_csv(f, dtype={"RENT_DATE": str, "RENT_TIME": str,
                               "RENT_ID": str, "CNT": int})
        for f in files
    ]
    return pd.concat(parts, ignore_index=True)
 
 
# ── Daily missing flag per station ───────────────────────────
def build_daily_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (RENT_DATE, RENT_ID) pair, flag missing=1
    only when zero hours of data exist for that day.
    """
    all_dates = sorted(df["RENT_DATE"].unique())
    all_ids   = sorted(df["RENT_ID"].unique())
 
    # Build full expected (date, station) combinations
    expected = pd.DataFrame(
        [(d, r) for d in all_dates for r in all_ids],
        columns=["RENT_DATE", "RENT_ID"]
    )
 
    # Count distinct hours observed per (date, station)
    observed = (
        df.groupby(["RENT_DATE", "RENT_ID"])["RENT_TIME"]
        .nunique()
        .reset_index()
        .rename(columns={"RENT_TIME": "hours_observed"})
    )
 
    merged = expected.merge(observed, on=["RENT_DATE", "RENT_ID"], how="left")
    merged["hours_observed"] = merged["hours_observed"].fillna(0).astype(int)
    merged["missing"] = (merged["hours_observed"] == 0).astype(int)
    merged["date"] = pd.to_datetime(merged["RENT_DATE"], format="%Y%m%d")
    return merged
 
 
# ── Consecutive missing period detection ─────────────────────
def find_consecutive_missing(daily: pd.DataFrame,
                              threshold: int = LONG_MISSING_TH) -> dict:
    """
    Detect consecutive missing periods >= threshold days per station.
    Returns: {rent_id: [(start_date, end_date, n_days), ...]}
    """
    result = {}
 
    for rent_id, group in daily.groupby("RENT_ID"):
        group = group.sort_values("date").reset_index(drop=True)
 
        long_periods = []
        in_missing   = False
        start_date   = None
        count        = 0
 
        for _, row in group.iterrows():
            if row["missing"] == 1:
                if not in_missing:
                    # Start of a new missing streak
                    in_missing = True
                    start_date = row["date"]
                    count = 1
                else:
                    count += 1
            else:
                if in_missing and count >= threshold:
                    end_date = row["date"] - pd.Timedelta(days=1)
                    long_periods.append((
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        count
                    ))
                in_missing = False
                count = 0
 
        # Handle case where missing streak extends to the last date
        if in_missing and count >= threshold:
            end_date = group["date"].iloc[-1]
            long_periods.append((
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                count
            ))
 
        if long_periods:
            result[rent_id] = long_periods
 
    return result
 
 
# ── Main ─────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_filtered_csvs(DATA_DIR)
    print(f"  Total rows: {len(df):,} | Stations: {df['RENT_ID'].nunique()} | "
          f"Period: {df['RENT_DATE'].min()} ~ {df['RENT_DATE'].max()}")
 
    print("\nBuilding daily missing flags...")
    daily = build_daily_missing(df)
 
    print(f"\nDetecting consecutive missing periods (>= {LONG_MISSING_TH} days)...")
    long_missing = find_consecutive_missing(daily, threshold=LONG_MISSING_TH)
 
    # ── Print results ────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Long missing periods (consecutive >= {LONG_MISSING_TH} days)")
    print("=" * 55)
 
    if not long_missing:
        print(f"  -> No consecutive missing periods >= {LONG_MISSING_TH} days found.")
    else:
        for rent_id, periods in sorted(long_missing.items()):
            for (start, end, days) in periods:
                print(f"  Station {rent_id}: {start} ~ {end}  ({days} days)")
 
    # ── Print copy-pasteable LONG_MISSING dict for notebook ──
    print("\n" + "=" * 55)
    print("  LONG_MISSING dict (copy-paste into notebook)")
    print("=" * 55)
    print("LONG_MISSING = {")
    for rent_id, periods in sorted(long_missing.items()):
        tuples = ", ".join(
            [f'("{start}", "{end}")' for (start, end, _) in periods]
        )
        print(f'    "{rent_id}": [{tuples}],')
    print("}")
 
    return long_missing
 
 
if __name__ == "__main__":
    main()
 