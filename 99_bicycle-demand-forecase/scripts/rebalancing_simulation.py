"""
Rebalancing simulation for Seoul public bicycle (Ddareungi) stations.

Decision date: December 27 (weekday) — hourly predicted demand vs availability
(not cumulative). Rebalancing decision at midnight.

Usage:
    python scripts/rebalancing_simulation.py
"""
import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import gc
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[Warning] XGBoost not installed; XGBoost best models will fall back to Random Forest")

from src.config import DATA_DIR, ALPHA as DEMAND_ALPHA
from src.utils import (
    load_filtered_csvs,
    load_rainfall_binary,
    make_features_rainfall_binary,
    make_series,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Midnight 24h worst-hour mix: 3 DEFICIT + 2 NEUTRAL + 3 SURPLUS (2024-12-27 scan)
REPRESENTATIVE_STATIONS = [
    "02148", "03310", "03802",   # DEFICIT
    "03311", "02128",            # NEUTRAL
    "02172", "02112", "02178",   # SURPLUS
]
DECISION_MONTH = 12
DECISION_DAY = 27  # 2024-12-27 = Friday (weekday)
HORIZON_HOURS = 24  # full decision day (00:00-23:00) for worst-hour scan
AVAIL_ALPHA = 0.5
SPARE_BIKES_FOR_SURPLUS = 2   # integer bikes left after covering demand
PEAK_HOURS = range(17, 22)    # 17:00-21:00 — block donors that peak later

DEMAND_BENCHMARK = ROOT_DIR / "results" / "combined_benchmark.csv"
AVAIL_BENCHMARK = ROOT_DIR / "results" / "available_ml_benchmark_results.csv"
ORDERS_FILE = ROOT_DIR / "results" / "auto_arima_orders_all.csv"
PREDICTIONS_FILE = ROOT_DIR / "results" / "rebalancing_predictions.csv"
FIGURE_FILE = ROOT_DIR / "figures" / "rebalancing_flow.png"

ML_MODELS = {"Random Forest", "XGBoost", "XGBoost (alpha=2.0)", "XGBoost (alpha=0.5)"}
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

STATUS_COLORS = {"DEFICIT": "#d62728", "SURPLUS": "#2ca02c", "NEUTRAL": "#bdbdbd"}


def normalize_station_id(station_id) -> str:
    return str(station_id).strip().zfill(5)


def load_best_model_map(csv_path: Path, asym_col: str) -> dict[str, str]:
    df = pd.read_csv(csv_path, dtype={"station_id": str})
    df["station_id"] = df["station_id"].map(normalize_station_id)
    idx = df.groupby("station_id")[asym_col].idxmin()
    best = df.loc[idx, ["station_id", "model"]]
    return dict(zip(best["station_id"], best["model"]))


def load_available_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("interpolated_available_*.csv"))
    if not files:
        raise FileNotFoundError(f"No interpolated_available_*.csv in {data_dir}")
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["datetime"] = pd.to_datetime(
        df["DATE"].astype(str) + df["TIME"].astype(str).str.zfill(2),
        format="%Y%m%d%H",
    )
    df["RENT_ID"] = df["RENT_ID"].astype(int)
    return df


def make_available_series(df: pd.DataFrame, station_id: str) -> pd.Series:
    rent_id = int(station_id)
    sub = (
        df[df["RENT_ID"] == rent_id]
        .set_index("datetime")["AVAILABLE"]
        .sort_index()
    )
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="h")
    return sub.reindex(full_idx).astype(float)


def resolve_decision_year(series_list: list[pd.Series]) -> int:
    """Pick the latest year where Dec 28 exists in all series."""
    common_end = min(s.index.max() for s in series_list)
    for year in range(common_end.year, common_end.year - 4, -1):
        target = pd.Timestamp(year=year, month=DECISION_MONTH, day=DECISION_DAY)
        if all((s.index.min() <= target) and (s.index.max() >= target + pd.Timedelta(hours=23)) for s in series_list):
            return year
    raise ValueError(
        f"Cannot find {DECISION_MONTH}/{DECISION_DAY} covered by all demand and availability series."
    )


def get_decision_day_index(year: int) -> pd.DatetimeIndex:
    """24 hourly timestamps for Dec 28 of the given year."""
    start = pd.Timestamp(year=year, month=DECISION_MONTH, day=DECISION_DAY, hour=0)
    return pd.date_range(start, periods=24, freq="h")


def asymmetric_xgb_obj(alpha: float):
    def _obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        errors = y_pred - y_true
        grad = np.where(errors < 0, alpha * 2 * errors, 2 * errors)
        hess = np.where(errors < 0, alpha * 2 * np.ones_like(errors), 2 * np.ones_like(errors))
        return grad, hess

    return _obj


def parse_xgb_alpha(model_name: str, default: float) -> float:
    if "alpha=" not in model_name:
        return default
    return float(model_name.split("alpha=")[1].rstrip(")"))


def fit_ml_model(model_name: str, X_train, y_train, X_test, default_alpha: float):
    if model_name == "Random Forest" or (model_name.startswith("XGBoost") and not HAS_XGB):
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    if model_name == "XGBoost":
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        return model.predict(X_test)

    if model_name.startswith("XGBoost (alpha="):
        alpha = parse_xgb_alpha(model_name, default_alpha)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(
            params={"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
                    "colsample_bytree": 0.8, "seed": 42},
            dtrain=dtrain,
            num_boost_round=300,
            obj=asymmetric_xgb_obj(alpha),
            verbose_eval=False,
        )
        return booster.predict(dtest)

    raise ValueError(f"Unsupported ML model: {model_name}")


def fit_predict_series(
    series: pd.Series,
    weather_df: pd.DataFrame,
    model_name: str,
    test_index: pd.DatetimeIndex,
    orders_row: pd.Series | None = None,
    default_alpha: float = 2.0,
) -> pd.Series:
    train_end = test_index[0] - pd.Timedelta(hours=1)
    train_s = series[series.index <= train_end]

    is_ml = model_name in ML_MODELS or model_name.startswith("XGBoost")
    if is_ml:
        df_feat = make_features_rainfall_binary(series, weather_df)
        test_df = df_feat[df_feat.index.isin(test_index)]

        if len(test_df) < len(test_index):
            fallback = df_feat[
                (df_feat.index >= test_index[0]) & (df_feat.index <= test_index[-1])
            ]
            if len(fallback) > len(test_df):
                test_df = fallback.reindex(test_index).dropna()
            if len(test_df) == 0:
                fallback = df_feat[df_feat.index <= test_index[-1]].iloc[-len(test_index):]
                if len(fallback) > 0:
                    test_df = fallback

        if len(test_df) == 0:
            raise ValueError(
                f"No ML test rows for model={model_name}. "
                f"Series range: {series.index.min()} ~ {series.index.max()}, "
                f"requested: {test_index[0]} ~ {test_index[-1]}."
            )

        train_df = df_feat[df_feat.index < test_df.index[0]]
        feat_cols = [c for c in df_feat.columns if c != "CNT"]
        preds = fit_ml_model(
            model_name,
            train_df[feat_cols].values,
            train_df["CNT"].values,
            test_df[feat_cols].values,
            default_alpha,
        )
        return pd.Series(np.clip(preds, 0, None), index=test_df.index, name=series.name)

    if orders_row is None:
        raise ValueError(f"SARIMA/ARIMAX requires order row for model {model_name}")

    order = (int(orders_row["p"]), int(orders_row["d"]), int(orders_row["q"]))
    seasonal_order = (
        int(orders_row["P"]), int(orders_row["D"]),
        int(orders_row["Q"]), int(orders_row["S"]),
    )

    if model_name.startswith("ARIMAX"):
        exog_full = weather_df[["temp", "rain_binary"]].reindex(series.index).ffill()
        exog_full["weekday_binary"] = (exog_full.index.dayofweek < 5).astype(int)
        exog_train = exog_full[exog_full.index <= train_end]
        exog_test = exog_full[exog_full.index.isin(test_index)]
        fit = SARIMAX(
            train_s, exog=exog_train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = fit.forecast(steps=len(test_index), exog=exog_test)
    else:
        fit = SARIMAX(
            train_s, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = fit.forecast(steps=len(test_index))

    return pd.Series(np.clip(fc.values, 0, None), index=test_index, name=series.name)


STATUS_RANK = {"DEFICIT": 0, "NEUTRAL": 1, "SURPLUS": 2}


def aggregate_worst_status(hourly_statuses: list[str]) -> str:
    """DEFICIT if any hour is DEFICIT; else SURPLUS if all hours SURPLUS; else NEUTRAL."""
    if not hourly_statuses:
        return "NEUTRAL"
    if any(st == "DEFICIT" for st in hourly_statuses):
        return "DEFICIT"
    if all(st == "SURPLUS" for st in hourly_statuses):
        return "SURPLUS"
    return "NEUTRAL"


def classify_ops_hour(avail: float, demand: float) -> str:
    """Hourly status: compare inventory to bikes needed (ceil hourly demand)."""
    demand_need = int(np.ceil(demand)) if demand > 0 else 0
    if avail < demand_need:
        return "DEFICIT"
    if avail >= demand_need + SPARE_BIKES_FOR_SURPLUS:
        return "SURPLUS"
    return "NEUTRAL"


def classify_status(
    pred_available: float,
    pred_demand: float,
    avail_median: float,
    demand_median: float,
) -> str:
    return classify_ops_hour(float(pred_available), float(pred_demand))


def classify_at_midnight(
    raw_demand: dict[str, pd.Series],
    raw_available: dict[str, pd.Series],
    decision_ts: pd.Timestamp,
    horizon: int,
) -> tuple[dict[str, str], dict[str, float], dict[str, int], dict[str, str]]:
    """
    Midnight rebalancing via worst hour in the next `horizon` hours:
    for each hour h, classify avail(h) vs ceil(demand(h)); aggregate worst status.
    """
    statuses, avail_now, worst_hour, raw_status = {}, {}, {}, {}

    for sid in raw_demand:
        d = raw_demand[sid]
        a = raw_available[sid]

        if decision_ts not in d.index:
            continue

        if horizon >= 24:
            window_idx = d.index
        else:
            pos = d.index.get_loc(decision_ts)
            end = min(pos + horizon, len(d) - 1)
            window_idx = d.index[pos + 1 : end + 1] if end > pos else d.index[:0]

        hour_statuses = []
        if len(window_idx) == 0:
            avail_now[sid] = float(a.loc[decision_ts])
            worst_hour[sid] = int(decision_ts.hour)
            statuses[sid] = "NEUTRAL"
            raw_status[sid] = "NEUTRAL"
            continue

        worst_ts = window_idx[0]
        worst_rank = STATUS_RANK["SURPLUS"]
        for ts in window_idx:
            st = classify_ops_hour(float(a.loc[ts]), float(d.loc[ts]))
            hour_statuses.append(st)
            rank = STATUS_RANK[st]
            if rank < worst_rank:
                worst_rank = rank
                worst_ts = ts

        avail_now[sid] = float(a.loc[decision_ts])
        worst_hour[sid] = int(worst_ts.hour)
        st = aggregate_worst_status(hour_statuses)
        raw_status[sid] = st
        statuses[sid] = st

    return statuses, avail_now, worst_hour, raw_status


def apply_peak_hour_donor_filter(
    pred_demand: dict[str, pd.Series],
    pred_available: dict[str, pd.Series],
    decision_status: dict[str, str],
) -> dict[str, str]:
    """
    A station tagged SURPLUS at midnight cannot donate if it becomes DEFICIT
    during evening peak hours (e.g. 02129: fine at dawn, short on bikes at 18:00).
    """
    filtered = dict(decision_status)
    for sid, st in decision_status.items():
        if st != "SURPLUS":
            continue
        for ts in pred_demand[sid].index:
            if ts.hour not in PEAK_HOURS:
                continue
            if classify_ops_hour(
                float(pred_available[sid].loc[ts]),
                float(pred_demand[sid].loc[ts]),
            ) == "DEFICIT":
                filtered[sid] = "NEUTRAL"
                print(
                    f"  [Donor filter] {sid}: midnight SURPLUS -> NEUTRAL "
                    f"(DEFICIT at peak hour {ts.hour}:00)"
                )
                break
    return filtered


def station_grid_positions(stations: list[str], n_cols: int = 4) -> dict[str, tuple[float, float]]:
    n = len(stations)
    n_rows = math.ceil(n / n_cols)
    positions = {}
    for i, sid in enumerate(stations):
        row, col = divmod(i, n_cols)
        x = 1.2 + col * 2.6
        y = n_rows - row
        positions[sid] = (x, y)
    return positions


def plot_rebalancing_flow(
    stations: list[str],
    pred_demand: dict[str, pd.Series],
    pred_available: dict[str, pd.Series],
    decision_ts: pd.Timestamp,
    hourly_status: dict[str, list[str]],
    decision_status: dict[str, str],
    output_path: Path,
):
    """
    Top: hourly demand vs availability on Dec 28 (x = hour 0-23).
    Bottom: midnight rebalancing schematic with arrows.
    """
    n = len(stations)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(16, 3.0 * n_rows + 3.5))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.9],
                         hspace=0.55, wspace=0.30)

    date_label = decision_ts.strftime("%Y-%m-%d")
    fig.suptitle(
        f"Rebalancing - {date_label} (weekday) | "
        f"Hourly: raw bars, status via avail vs ceil(demand) | "
        f"Midnight: worst hour in full day 00-23h (same hourly rule)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    for i, sid in enumerate(stations):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        d = pred_demand[sid]
        a = pred_available[sid]
        hours = d.index.hour
        statuses = hourly_status[sid]

        # Background tint by hourly status
        for h in range(24):
            if h >= len(statuses):
                break
            st = statuses[h]
            ax.axvspan(h - 0.5, h + 0.5, color=STATUS_COLORS[st], alpha=0.12, zorder=0)

        ax.bar(hours - 0.18, d.values, width=0.36, color="#1f77b4", alpha=0.85, label="Pred. demand")
        ax.set_ylabel("Demand", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax.twinx()
        ax2.bar(hours + 0.18, a.values, width=0.36, color="#ff7f0e", alpha=0.75, label="Pred. availability")
        ax2.set_ylabel("Availability", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        midnight_status = decision_status.get(sid, "NEUTRAL")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(0.02, 0.97, f"Midnight: {midnight_status}",
                transform=ax.transAxes, fontsize=8, va="top",
                color=STATUS_COLORS[midnight_status], fontweight="bold")

        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour (0-23), snapshot per hour")
        ax.set_title(f"Station {sid}", fontsize=10)

        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)

    # Schematic panel
    ax_flow = fig.add_subplot(gs[n_rows, :])
    positions = station_grid_positions(stations, n_cols=4)
    xs = [positions[s][0] for s in stations]
    ys = [positions[s][1] for s in stations]
    ax_flow.set_xlim(min(xs) - 1, max(xs) + 1)
    ax_flow.set_ylim(0.3, max(ys) + 0.8)
    ax_flow.axis("off")
    ax_flow.set_title(
        f"Midnight rebalancing ({decision_ts.strftime('%Y-%m-%d %H:%M')}): "
        f"worst hour in full day 00-23h | avail vs ceil(demand) -> SURPLUS -> DEFICIT",
        fontsize=10,
    )

    for sid in stations:
        x, y = positions[sid]
        status = decision_status.get(sid, "NEUTRAL")
        circle = plt.Circle(
            (x, y), 0.42, color=STATUS_COLORS[status], alpha=0.3,
            ec=STATUS_COLORS[status], lw=2,
        )
        ax_flow.add_patch(circle)
        ax_flow.text(x, y + 0.08, sid, ha="center", va="center", fontsize=9, fontweight="bold")
        ax_flow.text(x, y - 0.22, status, ha="center", va="center", fontsize=7,
                     color=STATUS_COLORS[status])

    surplus = [s for s, st in decision_status.items() if st == "SURPLUS"]
    deficit = [s for s, st in decision_status.items() if st == "DEFICIT"]
    for src in surplus:
        for dst in deficit:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            ax_flow.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                linewidth=1.8, color="#9467bd", alpha=0.85,
                connectionstyle="arc3,rad=0.12",
            ))

    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS["DEFICIT"], alpha=0.4, label="DEFICIT"),
        mpatches.Patch(color=STATUS_COLORS["SURPLUS"], alpha=0.4, label="SURPLUS"),
        mpatches.Patch(color=STATUS_COLORS["NEUTRAL"], alpha=0.4, label="NEUTRAL"),
        mpatches.Patch(color="#1f77b4", alpha=0.7, label="Demand (bar)"),
        mpatches.Patch(color="#ff7f0e", alpha=0.7, label="Availability (bar)"),
    ]
    ax_flow.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def main():
    print("=" * 60)
    print("Rebalancing simulation - single day (Dec 27, weekday)")
    print("=" * 60)

    demand_best = load_best_model_map(DEMAND_BENCHMARK, f"Asym.RMSE(alpha={DEMAND_ALPHA})")
    avail_best = load_best_model_map(AVAIL_BENCHMARK, f"Asym.RMSE(alpha={AVAIL_ALPHA})")
    orders_df = pd.read_csv(ORDERS_FILE, dtype={"station_id": str})
    orders_df["station_id"] = orders_df["station_id"].map(normalize_station_id)
    orders_map = orders_df.set_index("station_id")

    print("\nLoading data...")
    df_demand = load_filtered_csvs(DATA_DIR)
    df_demand = df_demand[df_demand["datetime"].dt.year >= 2023]
    df_avail = load_available_csvs(DATA_DIR)
    weather_df = load_rainfall_binary(DATA_DIR, threshold=1.0)

    stations = [normalize_station_id(s) for s in REPRESENTATIVE_STATIONS]
    demand_series_map = {sid: make_series(df_demand, sid) for sid in stations}
    avail_series_map = {sid: make_available_series(df_avail, sid) for sid in stations}

    all_series = list(demand_series_map.values()) + list(avail_series_map.values())
    decision_year = resolve_decision_year(all_series)
    day_index = get_decision_day_index(decision_year)
    decision_ts = day_index[0]

    print(f"\nData ranges:")
    print(f"  Demand     : {df_demand['datetime'].min()} ~ {df_demand['datetime'].max()}")
    print(f"  Available  : {df_avail['datetime'].min()} ~ {df_avail['datetime'].max()}")
    print(f"Decision day : {day_index[0].date()} (24 hours, midnight rebalancing)")
    print(f"Stations ({len(stations)}): {', '.join(stations)}")

    pred_demand_raw: dict[str, pd.Series] = {}
    pred_available_raw: dict[str, pd.Series] = {}

    for sid in stations:
        print(f"\n[Station {sid}]")
        d_model = demand_best.get(sid, "Random Forest")
        a_model = avail_best.get(sid, "Random Forest")
        print(f"  Demand model : {d_model}")
        print(f"  Avail model  : {a_model}")

        orders_row = orders_map.loc[sid] if sid in orders_map.index else None

        pred_demand_raw[sid] = fit_predict_series(
            demand_series_map[sid], weather_df, d_model, day_index,
            orders_row=orders_row, default_alpha=DEMAND_ALPHA,
        )
        pred_available_raw[sid] = fit_predict_series(
            avail_series_map[sid], weather_df, a_model, day_index,
            orders_row=None, default_alpha=AVAIL_ALPHA,
        )

        common_idx = pred_demand_raw[sid].index.intersection(pred_available_raw[sid].index)
        pred_demand_raw[sid] = pred_demand_raw[sid].reindex(common_idx)
        pred_available_raw[sid] = pred_available_raw[sid].reindex(common_idx)
        gc.collect()

    # Align all stations to same hourly index
    day_index = pred_demand_raw[stations[0]].index
    for sid in stations[1:]:
        day_index = day_index.intersection(pred_demand_raw[sid].index)
    for sid in stations:
        pred_demand_raw[sid] = pred_demand_raw[sid].reindex(day_index)
        pred_available_raw[sid] = pred_available_raw[sid].reindex(day_index)

    pred_demand = pred_demand_raw
    pred_available = pred_available_raw

    print(
        "\nRounding rules:"
        "\n  Hourly chart & status: raw model output (no rounding)"
        "\n  Midnight rebalancing: worst hour over full 24h (00:00-23:00)"
    )

    decision_status, avail_now, worst_hour, _ = classify_at_midnight(
        pred_demand_raw, pred_available_raw, decision_ts, HORIZON_HOURS
    )
    print("\nDonor safety check (peak hours 17-21):")
    decision_status = apply_peak_hour_donor_filter(
        pred_demand, pred_available, decision_status
    )

    print(f"\nMidnight classification ({decision_ts}):")
    for sid in stations:
        wh = worst_hour.get(sid, -1)
        print(
            f"  {sid}: {decision_status.get(sid, 'N/A')} "
            f"(worst hour {wh:02d}:00, avail@00:00={avail_now.get(sid, float('nan')):.2f})"
        )

    hourly_status: dict[str, list[str]] = {}
    rows = []
    for sid in stations:
        hourly_status[sid] = []
        for ts in day_index:
            av = float(pred_available[sid].loc[ts])
            dm = float(pred_demand[sid].loc[ts])
            st = classify_ops_hour(av, dm)
            hourly_status[sid].append(st)
            rows.append({
                "station_id": sid,
                "datetime": ts,
                "pred_demand": round(dm, 4),
                "pred_available": round(av, 4),
                "status": st,
            })

    pred_df = pd.DataFrame(rows)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nPredictions saved to: {PREDICTIONS_FILE}")

    plot_rebalancing_flow(
        stations, pred_demand, pred_available,
        decision_ts, hourly_status, decision_status, FIGURE_FILE,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
