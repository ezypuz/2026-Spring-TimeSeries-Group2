#!/usr/bin/env python3
"""Visualize missing (RENT_DATE, RENT_TIME, RENT_ID) combinations
across all ./data/filtered_YYYYMM.csv files.

A combination is considered missing when no row exists for a given
(date, hour, RENT_ID) triplet that was expected to be present.

Outputs:
  missing_heatmap_hour_id.png   – missing rate by hour × RENT_ID
  missing_heatmap_month_id.png  – missing rate by year-month × RENT_ID
  missing_timeline.png          – daily missing count over time
"""

from __future__ import annotations

from pathlib import Path
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_FILE = DATA_DIR / "target_rent_id.txt"
HOURS = list(range(24))


def load_target_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    # zero-pad to 5 digits to match filtered CSV RENT_ID format
    return [r.zfill(5) for r in raw]


def load_all_filtered(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("filtered_2*.csv"))
    if not files:
        raise FileNotFoundError(f"No filtered_*.csv files found in {data_dir}")
    parts = []
    for f in files:
        df = pd.read_csv(f, dtype={"RENT_DATE": str, "RENT_TIME": int, "RENT_ID": str, "CNT": int})
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_missing_df(df: pd.DataFrame, target_ids: list[str]) -> pd.DataFrame:
    all_dates = sorted(df["RENT_DATE"].unique())

    # full expected index
    expected = pd.DataFrame(
        list(itertools.product(all_dates, HOURS, target_ids)),
        columns=["RENT_DATE", "RENT_TIME", "RENT_ID"],
    )
    observed = df[["RENT_DATE", "RENT_TIME", "RENT_ID"]].drop_duplicates()
    observed["_present"] = True

    merged = expected.merge(observed, on=["RENT_DATE", "RENT_TIME", "RENT_ID"], how="left")
    merged["missing"] = merged["_present"].isna().astype(int)
    merged["YEAR_MONTH"] = merged["RENT_DATE"].str[:6]
    merged["RENT_DATE"] = pd.to_datetime(merged["RENT_DATE"], format="%Y%m%d")
    return merged


def plot_hour_id_heatmap(missing: pd.DataFrame, out_path: Path) -> None:
    """Missing rate (%) for each (RENT_ID, hour) pair across all dates."""
    pivot = (
        missing.groupby(["RENT_ID", "RENT_TIME"])["missing"]
        .mean()
        .mul(100)
        .unstack("RENT_TIME")  # columns = hours
    )
    pivot = pivot.reindex(sorted(pivot.index))

    fig, ax = plt.subplots(figsize=(16, max(6, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Missing rate (%)")

    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Hour (RENT_TIME)")
    ax.set_ylabel("RENT_ID")
    ax.set_title("Missing rate by Hour × RENT_ID\n(over all dates in dataset)")

    # annotate cells
    for i in range(len(pivot)):
        for j in range(24):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="black" if val < 60 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_month_id_heatmap(missing: pd.DataFrame, out_path: Path) -> None:
    """Missing rate (%) for each (YEAR_MONTH, RENT_ID) pair."""
    pivot = (
        missing.groupby(["YEAR_MONTH", "RENT_ID"])["missing"]
        .mean()
        .mul(100)
        .unstack("RENT_ID")  # columns = RENT_IDs
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.7), max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Missing rate (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_xlabel("RENT_ID")
    ax.set_ylabel("Year-Month")
    ax.set_title("Missing rate by Year-Month × RENT_ID")

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="black" if val < 60 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_daily_timeline(missing: pd.DataFrame, out_path: Path) -> None:
    """Total missing combinations per day."""
    daily = (
        missing.groupby("RENT_DATE")["missing"]
        .sum()
        .reset_index()
        .rename(columns={"missing": "missing_count"})
    )

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(daily["RENT_DATE"], daily["missing_count"], width=1, color="steelblue", alpha=0.8)

    # year separators
    for year in daily["RENT_DATE"].dt.year.unique():
        start = pd.Timestamp(year=year, month=1, day=1)
        ax.axvline(start, color="red", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(start, ax.get_ylim()[1], str(year), color="red", fontsize=8, va="top")

    ax.set_xlabel("Date")
    ax.set_ylabel("Missing combinations (RENT_ID × hour)")
    ax.set_title("Daily missing count  (max possible per day = 23 IDs × 24 hours = 552)")
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

    print("Building missing index...")
    missing = build_missing_df(df, target_ids)
    total_expected = len(missing)
    total_missing = missing["missing"].sum()
    print(f"  Expected combinations: {total_expected:,}")
    print(f"  Missing combinations : {total_missing:,}  ({total_missing/total_expected*100:.1f}%)")

    out_dir = Path(__file__).resolve().parent
    plot_hour_id_heatmap(missing, out_dir / "missing_heatmap_hour_id.png")
    plot_month_id_heatmap(missing, out_dir / "missing_heatmap_month_id.png")
    plot_daily_timeline(missing, out_dir / "missing_timeline.png")


if __name__ == "__main__":
    main()
