"""
23개 정류소 전체 파이프라인.

Phase 1. auto_arima — 정류소별 독립적으로 최적 차수 탐색 → reports/orders.json 저장
Phase 2. rolling CV  — 각 정류소의 차수로 SARIMA / ARIMAX / ML 평가 → reports/results_all.csv 저장

중간에 끊겨도 재실행 시 완료된 항목은 건너뜁니다.

실행 방법 (99_bicycle-demand-forecase/ 에서):
    python scripts/run_all_stations.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, TARGET_RENT_IDS, ALPHA, TEST_DAYS
from src.utils import (
    load_filtered_csvs, make_series, load_weather, make_features,
    save_result, load_done_keys,
    rolling_cv_sarima, rolling_cv_ml,
)

REPORTS_DIR  = Path(__file__).parent.parent / "reports"
ORDERS_JSON  = REPORTS_DIR / "orders.json"
RESULTS_CSV  = REPORTS_DIR / "results_all.csv"
EXOG_COLS    = ["temp", "rain"]
N_SPLITS_SARIMA = 3
N_SPLITS_ML     = 5


# ── Phase 1: auto_arima ───────────────────────────────────────────

def find_order(series: pd.Series, exog=None) -> dict:
    """d=0, D=1, m=24 고정 후 p,q,P,Q 탐색."""
    model = auto_arima(
        series.dropna(),
        exogenous=exog,
        d=0, D=1,
        seasonal=True, m=24,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    return {
        "order":          list(model.order),
        "seasonal_order": list(model.seasonal_order),
    }


def phase1_find_orders(df_raw, weather_df) -> dict:
    """정류소별 weekday/weekend 차수 탐색. 이미 있는 정류소는 스킵."""
    orders = json.loads(ORDERS_JSON.read_text()) if ORDERS_JSON.exists() else {}

    for i, station_id in enumerate(TARGET_RENT_IDS, 1):
        if station_id in orders:
            print(f"[{i}/{len(TARGET_RENT_IDS)}] {station_id} — 차수 이미 있음, skip")
            continue

        print(f"[{i}/{len(TARGET_RENT_IDS)}] {station_id} — auto_arima 탐색 중...", flush=True)
        series = make_series(df_raw, station_id)

        station_orders = {}
        for day_type, mask_fn in [
            ("weekday", lambda s: s[s.index.dayofweek < 5]),
            ("weekend", lambda s: s[s.index.dayofweek >= 5]),
        ]:
            s_day = mask_fn(series)
            result = find_order(s_day)
            station_orders[day_type] = result
            print(f"  [{day_type}] order={result['order']}  seasonal={result['seasonal_order']}")

        orders[station_id] = station_orders
        ORDERS_JSON.write_text(json.dumps(orders, indent=2, ensure_ascii=False))

    print(f"\nPhase 1 완료 → {ORDERS_JSON}")
    return orders


# ── Phase 2: rolling CV ───────────────────────────────────────────

def run_station(station_id: str, station_orders: dict,
                df_raw, weather_df, done_keys: set) -> None:
    print(f"\n{'='*55}")
    print(f"  Station {station_id}  [{datetime.now():%H:%M:%S}]")
    print(f"{'='*55}")

    series  = make_series(df_raw, station_id)
    df_feat = make_features(series, weather_df)

    for day_type, mask_fn in [
        ("weekday", lambda s: s[s.index.dayofweek < 5]),
        ("weekend", lambda s: s[s.index.dayofweek >= 5]),
    ]:
        s_day         = mask_fn(series)
        order         = tuple(station_orders[day_type]["order"])
        seasonal_order = tuple(station_orders[day_type]["seasonal_order"])

        # ── SARIMA ────────────────────────────────────────────────
        key = (station_id, day_type, "SARIMA")
        if key not in done_keys:
            print(f"  [{day_type}] SARIMA {order}×{seasonal_order}  (n={N_SPLITS_SARIMA})", flush=True)
            folds = rolling_cv_sarima(
                s_day, order, seasonal_order,
                n_splits=N_SPLITS_SARIMA, test_days=TEST_DAYS, alpha=ALPHA,
            )
            for f in folds:
                save_result({"station_id": station_id, "day_type": day_type,
                             "model": "SARIMA", **f}, RESULTS_CSV)
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [{day_type}] SARIMA already done, skip")

        # ── ARIMAX ────────────────────────────────────────────────
        key = (station_id, day_type, "ARIMAX")
        if key not in done_keys:
            print(f"  [{day_type}] ARIMAX {order}×{seasonal_order}  (n={N_SPLITS_SARIMA})", flush=True)
            exog_day = mask_fn(weather_df.reindex(s_day.index).ffill())[EXOG_COLS]
            folds = rolling_cv_sarima(
                s_day, order, seasonal_order,
                exog_df=exog_day,
                n_splits=N_SPLITS_SARIMA, test_days=TEST_DAYS, alpha=ALPHA,
            )
            for f in folds:
                save_result({"station_id": station_id, "day_type": day_type,
                             "model": "ARIMAX", **f}, RESULTS_CSV)
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [{day_type}] ARIMAX already done, skip")

    # ── ML (평일만) ───────────────────────────────────────────────
    df_day    = df_feat[df_feat.index.dayofweek < 5]
    feat_cols = [c for c in df_day.columns if c != "CNT"]
    X, y      = df_day[feat_cols], df_day["CNT"]

    ml_models = [
        ("RandomForest", RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        )),
        ("XGBoost", xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        )),
    ]

    for model_name, model_obj in ml_models:
        key = (station_id, "weekday", model_name)
        if key not in done_keys:
            print(f"  [weekday] {model_name}  (n={N_SPLITS_ML})", flush=True)
            folds = rolling_cv_ml(X, y, model_obj,
                                  n_splits=N_SPLITS_ML, test_days=TEST_DAYS, alpha=ALPHA)
            for f in folds:
                save_result({"station_id": station_id, "day_type": "weekday",
                             "model": model_name, **f}, RESULTS_CSV)
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [weekday] {model_name} already done, skip")


def phase2_run_all(orders: dict, df_raw, weather_df) -> None:
    done_keys = load_done_keys(RESULTS_CSV)
    if done_keys:
        print(f"체크포인트: {len(done_keys)}개 조합 이미 완료")

    for i, station_id in enumerate(TARGET_RENT_IDS, 1):
        print(f"\n[{i}/{len(TARGET_RENT_IDS)}]", end="")
        run_station(station_id, orders[station_id], df_raw, weather_df, done_keys)

    print(f"\n\nPhase 2 완료 → {RESULTS_CSV}")


# ── main ──────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 시작")
    print(f"  대상: {len(TARGET_RENT_IDS)}개 정류소\n")

    df_raw     = load_filtered_csvs(DATA_DIR)
    weather_df = load_weather(DATA_DIR)

    print("── Phase 1: auto_arima 차수 탐색 ──")
    orders = phase1_find_orders(df_raw, weather_df)

    print("\n── Phase 2: rolling CV 평가 ──")
    phase2_run_all(orders, df_raw, weather_df)

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] 전체 완료")


if __name__ == "__main__":
    main()
