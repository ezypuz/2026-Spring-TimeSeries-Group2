#!/usr/bin/env python3
"""Visualize truly missing days across ./data/filtered_YYYYMM.csv files.

Missing criterion (stricter than visualize_missing.py):
  A (RENT_DATE, RENT_ID) pair is "missing" only when ALL 24 hours
  have no data for that day.  Partial-day gaps are NOT counted.

Outputs:
  missing_daily_heatmap_month_id.png  – missing days (%) by year-month × RENT_ID
  missing_daily_heatmap_dow_id.png    – missing days (%) by day-of-week × RENT_ID
  missing_daily_timeline.png          – daily count of fully-missing stations
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_FILE = DATA_DIR / "target_rent_id.txt"
HOURS = set(range(24))
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_target_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    return [r.zfill(5) for r in raw]


def load_all_filtered(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("filtered_2*.csv"))
    if not files:
        raise FileNotFoundError(f"No filtered_*.csv files found in {data_dir}")
    parts = [
        pd.read_csv(f, dtype={"RENT_DATE": str, "RENT_TIME": int, "RENT_ID": str, "CNT": int})
        for f in files
    ]
    return pd.concat(parts, ignore_index=True)


def build_daily_missing(df: pd.DataFrame, target_ids: list[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by (RENT_DATE, RENT_ID) with a `missing` flag.

    missing=1 only when all 24 hours have no observation for that (date, id) pair.
    """
    all_dates = sorted(df["RENT_DATE"].unique())

    # expected: every (date, rent_id) combination
    expected = pd.DataFrame(
        [(d, r) for d in all_dates for r in target_ids],
        columns=["RENT_DATE", "RENT_ID"],
    )

    # count distinct hours observed per (date, rent_id)
    observed_hours = (
        df.groupby(["RENT_DATE", "RENT_ID"])["RENT_TIME"]
        .nunique()
        .reset_index()
        .rename(columns={"RENT_TIME": "hours_observed"})
    )

    merged = expected.merge(observed_hours, on=["RENT_DATE", "RENT_ID"], how="left")
    merged["hours_observed"] = merged["hours_observed"].fillna(0).astype(int)

    # truly missing = zero hours observed (all 24 hours absent)
    merged["missing"] = (merged["hours_observed"] == 0).astype(int)

    merged["date"] = pd.to_datetime(merged["RENT_DATE"], format="%Y%m%d")
    merged["YEAR_MONTH"] = merged["RENT_DATE"].str[:6]
    merged["DOW"] = merged["date"].dt.dayofweek  # 0=Mon … 6=Sun
    return merged


def plot_heatmap(pivot: pd.DataFrame, xlabel: str, ylabel: str, title: str,
                 xticklabels: list, yticklabels: list, out_path: Path,
                 figsize: tuple = (14, 8), xrotation: int = 45) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Missing days (%)")

    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=xrotation, ha="right", fontsize=8)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 60 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_month_id_heatmap(daily: pd.DataFrame, out_path: Path) -> None:
    pivot = (
        daily.groupby(["YEAR_MONTH", "RENT_ID"])["missing"]
        .mean()
        .mul(100)
        .unstack("RENT_ID")
        .sort_index()
    )
    plot_heatmap(
        pivot,
        xlabel="RENT_ID",
        ylabel="Year-Month",
        title="Fully-missing days (%) by Year-Month × RENT_ID\n"
              "(missing = zero data across all 24 hours)",
        xticklabels=pivot.columns.tolist(),
        yticklabels=pivot.index.tolist(),
        out_path=out_path,
        figsize=(max(10, len(pivot.columns) * 0.7), max(8, len(pivot) * 0.4)),
        xrotation=45,
    )


def plot_dow_id_heatmap(daily: pd.DataFrame, out_path: Path) -> None:
    pivot = (
        daily.groupby(["DOW", "RENT_ID"])["missing"]
        .mean()
        .mul(100)
        .unstack("RENT_ID")
        .reindex(range(7))
    )
    plot_heatmap(
        pivot,
        xlabel="RENT_ID",
        ylabel="Day of week",
        title="Fully-missing days (%) by Day-of-Week × RENT_ID",
        xticklabels=pivot.columns.tolist(),
        yticklabels=DOW_LABELS,
        out_path=out_path,
        figsize=(max(10, len(pivot.columns) * 0.7), 5),
        xrotation=45,
    )


def plot_daily_timeline(daily: pd.DataFrame, out_path: Path) -> None:
    """Count of stations that are fully missing each day."""
    per_day = (
        daily.groupby("date")["missing"]
        .sum()
        .reset_index()
        .rename(columns={"missing": "missing_stations"})
    )

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(per_day["date"], per_day["missing_stations"], width=1,
           color="steelblue", alpha=0.8)

    for year in per_day["date"].dt.year.unique():
        start = pd.Timestamp(year=year, month=1, day=1)
        ax.axvline(start, color="red", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(start, ax.get_ylim()[1], str(year),
                color="red", fontsize=8, va="top")

    ax.set_xlabel("Date")
    ax.set_ylabel("Fully-missing stations")
    ax.set_title(
        f"Daily count of stations with zero data all day  "
        f"(max = {daily['RENT_ID'].nunique()} stations)"
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(TARGET_FILE)

    target_ids = load_target_ids(TARGET_FILE)
    print(f"Target IDs ({len(target_ids)}): {target_ids}")

    print("Loading filtered CSV files...")
    df = load_all_filtered(DATA_DIR)
    print(f"  Total rows: {len(df):,}  |  date range: {df['RENT_DATE'].min()} – {df['RENT_DATE'].max()}")

    print("Building daily missing index...")
    daily = build_daily_missing(df, target_ids)

    total = len(daily)
    n_missing = daily["missing"].sum()
    print(f"  (date, RENT_ID) pairs  : {total:,}")
    print(f"  Fully-missing days     : {n_missing:,}  ({n_missing / total * 100:.1f}%)")

    out_dir = Path(__file__).resolve().parent
    plot_month_id_heatmap(daily, out_dir / "missing_daily_heatmap_month_id.png")
    plot_dow_id_heatmap(daily, out_dir / "missing_daily_heatmap_dow_id.png")
    plot_daily_timeline(daily, out_dir / "missing_daily_timeline.png")


if __name__ == "__main__":
    main()
