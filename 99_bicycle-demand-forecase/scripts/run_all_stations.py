"""
23개 정류소 전체에 대해 SARIMA / ARIMAX / ML 모델을 rolling CV로 평가.
결과는 reports/results_all.csv 에 누적 저장.
중간에 끊겨도 재실행 시 완료된 정류소는 건너뜁니다.

실행 방법 (프로젝트 루트에서):
    python scripts/run_all_stations.py
"""
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
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

RESULTS_CSV = Path(__file__).parent.parent / "reports" / "results_all.csv"

# auto_arima 결과 (station 02128 기준, 2023+ 데이터)
WEEKDAY_ORDER          = (2, 0, 1)
WEEKDAY_SEASONAL_ORDER = (1, 1, 0, 24)
WEEKEND_ORDER          = (1, 0, 2)
WEEKEND_SEASONAL_ORDER = (1, 1, 0, 24)

N_SPLITS_SARIMA = 3   # SARIMA / ARIMAX rolling CV 횟수
N_SPLITS_ML     = 5   # ML rolling CV 횟수
EXOG_COLS       = ["temp", "rain"]


def run_station(station_id: str, df_raw, weather_df, done_keys: set) -> None:
    print(f"\n{'='*55}")
    print(f"  Station {station_id}  [{datetime.now():%H:%M:%S}]")
    print(f"{'='*55}")

    series  = make_series(df_raw, station_id)
    df_feat = make_features(series, weather_df)

    day_configs = [
        ("weekday", lambda s: s[s.index.dayofweek < 5],
         WEEKDAY_ORDER, WEEKDAY_SEASONAL_ORDER),
        ("weekend", lambda s: s[s.index.dayofweek >= 5],
         WEEKEND_ORDER, WEEKEND_SEASONAL_ORDER),
    ]

    for day_type, mask_fn, order, seasonal_order in day_configs:
        s_day = mask_fn(series)

        # ── SARIMA ────────────────────────────────────────────────
        key = (station_id, day_type, "SARIMA")
        if key not in done_keys:
            print(f"  [{day_type}] SARIMA  (n_splits={N_SPLITS_SARIMA}) ...", flush=True)
            folds = rolling_cv_sarima(
                s_day, order, seasonal_order,
                n_splits=N_SPLITS_SARIMA, test_days=TEST_DAYS, alpha=ALPHA,
            )
            for f in folds:
                save_result(
                    {"station_id": station_id, "day_type": day_type, "model": "SARIMA", **f},
                    RESULTS_CSV,
                )
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [{day_type}] SARIMA  already done, skip")

        # ── ARIMAX ────────────────────────────────────────────────
        key = (station_id, day_type, "ARIMAX")
        if key not in done_keys:
            print(f"  [{day_type}] ARIMAX  (n_splits={N_SPLITS_SARIMA}) ...", flush=True)
            exog_day = mask_fn(weather_df.reindex(s_day.index).ffill())[EXOG_COLS]
            folds = rolling_cv_sarima(
                s_day, order, seasonal_order,
                exog_df=exog_day,
                n_splits=N_SPLITS_SARIMA, test_days=TEST_DAYS, alpha=ALPHA,
            )
            for f in folds:
                save_result(
                    {"station_id": station_id, "day_type": day_type, "model": "ARIMAX", **f},
                    RESULTS_CSV,
                )
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [{day_type}] ARIMAX  already done, skip")

    # ── ML 모델 (평일만) ──────────────────────────────────────────
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
            print(f"  [weekday] {model_name}  (n_splits={N_SPLITS_ML}) ...", flush=True)
            folds = rolling_cv_ml(
                X, y, model_obj,
                n_splits=N_SPLITS_ML, test_days=TEST_DAYS, alpha=ALPHA,
            )
            for f in folds:
                save_result(
                    {"station_id": station_id, "day_type": "weekday",
                     "model": model_name, **f},
                    RESULTS_CSV,
                )
            done_keys.add(key)
            print(f"         done — {len(folds)} folds")
        else:
            print(f"  [weekday] {model_name}  already done, skip")


def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 전체 정류소 분석 시작")
    print(f"  대상: {len(TARGET_RENT_IDS)}개 정류소")
    print(f"  결과: {RESULTS_CSV}")

    df_raw     = load_filtered_csvs(DATA_DIR)
    weather_df = load_weather(DATA_DIR)
    done_keys  = load_done_keys(RESULTS_CSV)

    if done_keys:
        print(f"  체크포인트: {len(done_keys)}개 조합 이미 완료")

    for i, station_id in enumerate(TARGET_RENT_IDS, 1):
        print(f"\n[{i}/{len(TARGET_RENT_IDS)}]", end="")
        run_station(station_id, df_raw, weather_df, done_keys)

    print(f"\n\n[{datetime.now():%Y-%m-%d %H:%M:%S}] 완료")
    print(f"결과 파일: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
