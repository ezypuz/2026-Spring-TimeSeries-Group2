"""
23개 정류소 평일 전체 벤치마크 스크립트.
- auto_arima_orders_weekday.csv에서 정류소별 최적 차수 로드
- SARIMA / ARIMAX / Random Forest / XGBoost / XGBoost(asym) 적합
- 결과를 results/full_benchmark_weekday.csv에 저장
- 체크포인트: 중간에 끊겨도 재시작 가능

Usage:
    python scripts/full_benchmark_weekday.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import gc
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import DATA_DIR, SEASONAL_PERIOD, TEST_DAYS, ALPHA
from src.utils import (
    load_filtered_csvs, make_series,
    load_rainfall_binary,
    asymmetric_rmse,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
RESULTS_DIR    = ROOT_DIR / "results"
ORDERS_FILE    = RESULTS_DIR / "remains_auto_arima_orders_all.csv"
RESULTS_FILE   = RESULTS_DIR / "remains_full_benchmark_all.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 차수 로드 ──────────────────────────────────────────────────────────────────
orders_df = pd.read_csv(ORDERS_FILE, dtype={"station_id": str})
print(f"Loaded {len(orders_df)} station orders")

# ── 체크포인트 ─────────────────────────────────────────────────────────────────
done_stations = set()
if RESULTS_FILE.exists():
    done_df = pd.read_csv(RESULTS_FILE, dtype={"station_id": str})
    # 모든 2개 시계열 모델이 완료된 정류소만 skip
    model_counts = done_df.groupby("station_id")["model"].count()
    done_stations = set(model_counts[model_counts >= 2].index)
    print(f"체크포인트: {len(done_stations)}개 정류소 완료 → 건너뜀")

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_remains_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("interpolated_available_*.csv"))
    if not files:
        raise FileNotFoundError(f"interpolated_available_*.csv 파일이 없습니다: {data_dir}")
    parts = [
        pd.read_csv(f, dtype={"DATE": str, "TIME": str,
                               "RENT_ID": str, "AVAILABLE": int})
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df["RENT_ID"] = df["RENT_ID"].str.zfill(5)
    df["datetime"] = pd.to_datetime(
        df["DATE"] + df["TIME"].str.zfill(2),
        format="%Y%m%d%H"
    )
    df["CNT"] = df["AVAILABLE"]

    return df

print("\nLoading data...")
df_raw     = load_remains_csvs(DATA_DIR)
df_raw     = df_raw[df_raw["datetime"].dt.year >= 2023]
df_raw     = df_raw[["datetime", "RENT_ID", "CNT"]]
weather_df = load_rainfall_binary(DATA_DIR, threshold=1.0)
print(f"Data range: {df_raw['datetime'].min().date()} ~ {df_raw['datetime'].max().date()}")

# ── 평가 함수 ──────────────────────────────────────────────────────────────────
def make_perf(y_test, y_pred, model_name, station_id, alpha=0.5):
    y_test  = np.asarray(y_test)
    y_pred  = np.asarray(y_pred)
    errors  = y_pred - y_test
    rmse    = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae     = float(mean_absolute_error(y_test, y_pred))
    asym    = float(asymmetric_rmse(y_test, y_pred, alpha=alpha))
    under   = float((errors < 0).mean() * 100)
    
    mean_y  = float(np.mean(y_test))
    cv_rmse = float(rmse / mean_y * 100) if mean_y > 0 else float("nan")
    
    # MAPE: y=0인 시점 제외
    mask    = y_test > 0
    mape    = float(np.mean(np.abs(errors[mask] / y_test[mask])) * 100) if mask.any() else float("nan")
    
    print(f"    [{model_name}] RMSE={rmse:.3f} | MAE={mae:.3f} | "
          f"Asym.RMSE={asym:.3f} | Under={under:.1f}% | "
          f"CV-RMSE={cv_rmse:.1f}% | MAPE={mape:.1f}%")
    return {
        "station_id":                station_id,
        "model":                     model_name,
        "RMSE":                      round(rmse, 3),
        "MAE":                       round(mae, 3),
        f"Asym.RMSE(alpha={alpha})": round(asym, 3),
        "Under-pred rate(%)":        round(under, 1),
        "CV-RMSE(%)":                round(cv_rmse, 1),
        "MAPE(%)":                   round(mape, 1),
    }

def save_record(record):
    row = pd.DataFrame([record])
    row.to_csv(
        RESULTS_FILE,
        mode="a",
        header=not RESULTS_FILE.exists(),
        index=False,
    )

def asymmetric_obj(y_pred, dtrain, alpha=2.0):
    y_true = dtrain.get_label()
    errors = y_pred - y_true
    grad = np.where(errors < 0, alpha * 2 * errors, 2 * errors)
    hess = np.where(errors < 0, alpha * 2 * np.ones_like(errors), 2 * np.ones_like(errors))
    return grad, hess

# ── 메인 루프 ──────────────────────────────────────────────────────────────────
for _, row in orders_df.iterrows():
    station_id = str(row["station_id"])

    if station_id in done_stations:
        print(f"\n[SKIP] Station {station_id}")
        continue

    print(f"\n{'='*60}")
    print(f"[Station {station_id}] Processing...")
    print(f"{'='*60}")

    order          = (int(row["p"]), int(row["d"]), int(row["q"]))
    seasonal_order = (int(row["P"]), int(row["D"]), int(row["Q"]), int(row["S"]))
    print(f"  Order: ARIMA{order}x{seasonal_order}")

    try:
        # ── 시계열 생성 ────────────────────────────────────────────
        series  = make_series(df_raw, station_id)

        test_hours = TEST_DAYS * 24
        train_s = series.iloc[:-test_hours]
        test_s  = series.iloc[-test_hours:]

        # 외생변수 (weekday_binary 추가)
        exog_full  = weather_df[["temp", "rain_binary"]].reindex(series.index).ffill()
        exog_full["weekday_binary"] = (exog_full.index.dayofweek < 5).astype(int)
        exog_train = exog_full.iloc[:-test_hours]
        exog_test  = exog_full.iloc[-test_hours:]

        print(f"  Train: {train_s.index[0].date()} ~ {train_s.index[-1].date()} ({len(train_s):,}h)")

        # ── SARIMA ────────────────────────────────────────────────
        print("  SARIMA 적합 중...")
        sarima_fit = SARIMAX(
            train_s, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = sarima_fit.forecast(steps=len(test_s))
        save_record(make_perf(test_s.values, fc.values, "SARIMA", station_id, ALPHA))
        del sarima_fit, fc; gc.collect()

        # ── ARIMAX ────────────────────────────────────────────────
        print("  ARIMAX 적합 중...")
        arimax_fit = SARIMAX(
            train_s, exog=exog_train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = arimax_fit.forecast(steps=len(test_s), exog=exog_test)
        save_record(make_perf(test_s.values, fc.values, "ARIMAX (temp+rain+weekday)", station_id, ALPHA))
        del arimax_fit, fc; gc.collect()
        '''
        # ── ML 피처 생성 ───────────────────────────────────────────
        df_feat  = make_features_rainfall_binary(series, weather_df)
        df_wday  = df_feat[df_feat.index.dayofweek < 5]
        train_df = df_wday.iloc[:-test_hours]
        test_df  = df_wday.iloc[-test_hours:]
        FEAT     = [c for c in df_feat.columns if c != "CNT"]
        X_train, y_train = train_df[FEAT].values, train_df["CNT"].values
        X_test,  y_test  = test_df[FEAT].values,  test_df["CNT"].values
        del df_feat, df_wday, train_df, test_df; gc.collect()

        # ── Random Forest ─────────────────────────────────────────
        print("  Random Forest 적합 중...")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=42,
        )
        rf.fit(X_train, y_train)
        save_record(make_perf(y_test, rf.predict(X_test), "Random Forest", station_id, ALPHA))
        del rf; gc.collect()

        # ── XGBoost ───────────────────────────────────────────────
        if HAS_XGB:
            print("  XGBoost 적합 중...")
            xgb_m = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            xgb_m.fit(X_train, y_train, verbose=False)
            save_record(make_perf(y_test, xgb_m.predict(X_test), "XGBoost", station_id, ALPHA))
            del xgb_m; gc.collect()

            # ── XGBoost Asymmetric ────────────────────────────────
            print("  XGBoost (asymmetric) 적합 중...")
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest  = xgb.DMatrix(X_test,  label=y_test)
            params = {"max_depth": 6, "learning_rate": 0.05,
                      "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42}
            xgb_asym = xgb.train(
                params=params, dtrain=dtrain, num_boost_round=300,
                obj=lambda yp, d: asymmetric_obj(yp, d, alpha=ALPHA),
                verbose_eval=False,
            )
            save_record(make_perf(y_test, xgb_asym.predict(dtest),
                                  f"XGBoost (alpha={ALPHA})", station_id, ALPHA))
            del dtrain, dtest, xgb_asym; gc.collect()
'''
        print(f"  [SAVED] Station {station_id} complete!")

    except Exception as e:
        print(f"  [ERROR] Station {station_id}: {e}")

    finally:
        try:
            del series, train_s, test_s
            del exog_full, exog_train, exog_test
            # del X_train, y_train, X_test, y_test
        except:
            pass
        gc.collect()
# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"완료! Results saved to: {RESULTS_FILE}")