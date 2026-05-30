"""
23개 정류소 평일 AutoARIMA 차수 탐색 스크립트.
- 메모리 최적화: 정류소마다 데이터 로드/해제, gc.collect() 적용
- 정류소 완료마다 즉시 CSV 저장 (행 단위 append)
- 체크포인트: 중간에 끊겨도 재시작 가능

Usage:
    python scripts/auto_arima_search_weekday.py
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
from statsforecast.models import AutoARIMA

from src.config import SEASONAL_PERIOD, TEST_DAYS
from src.utils import make_series

# ── 설정 ──────────────────────────────────────────────────────────────────────
TARGET_RENT_IDS = [
    "02111", "02112", "02116",
]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
ORDERS_FILE = RESULTS_DIR / "remains_auto_arima_orders_all.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 체크포인트 ─────────────────────────────────────────────────────────────────
done_stations = set()
if ORDERS_FILE.exists():
    done_df = pd.read_csv(ORDERS_FILE, dtype={"station_id": str})
    done_stations = set(done_df["station_id"].unique())
    print(f"체크포인트: {len(done_stations)}개 완료 → 건너뜀: {sorted(done_stations)}")

# ── 데이터 로드 (전체 1회만) ───────────────────────────────────────────────────
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
df_raw = load_remains_csvs(DATA_DIR)
df_raw = df_raw[df_raw["datetime"].dt.year >= 2023]

# 필요한 컬럼만 유지
df_raw = df_raw[["datetime", "RENT_ID", "CNT"]]
print(f"Data range: {df_raw['datetime'].min().date()} ~ {df_raw['datetime'].max().date()}")
print(f"Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ── 메인 루프 ──────────────────────────────────────────────────────────────────
for station_id in TARGET_RENT_IDS:
    if station_id in done_stations:
        print(f"\n[SKIP] Station {station_id}")
        continue

    print(f"\n{'='*50}")
    print(f"[Station {station_id}] AutoARIMA 탐색 중...")

    try:
        # 해당 정류소 데이터만 추출
        series  = make_series(df_raw, station_id)

        test_hours = TEST_DAYS * 24
        train = series.iloc[:-test_hours].values  # numpy array로 변환 (메모리 절약)

        print(f"  Train size: {len(train):,}h")

        # AutoARIMA
        model = AutoARIMA(
            season_length=SEASONAL_PERIOD,
            max_p=2, max_q=2,
            max_P=2, max_Q=2,
            d=0, D=1,
            stepwise=True,
            approximation=False,
            ic="aic",
        )
        model.fit(train)

        arma = model.model_["arma"]
        p, q = int(arma[0]), int(arma[1])
        P, Q = int(arma[2]), int(arma[3])
        d, D = int(arma[5]), int(arma[6])
        aic  = round(model.model_["aic"], 3)

        print(f"  Best: ARIMA({p},{d},{q})x({P},{D},{Q},{SEASONAL_PERIOD})  AIC={aic}")

        # 즉시 CSV에 한 행 append
        record = pd.DataFrame([{
            "station_id": station_id,
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q,
            "S": SEASONAL_PERIOD,
            "AIC": aic,
        }])
        record.to_csv(
            ORDERS_FILE,
            mode="a",
            header=not ORDERS_FILE.exists(),
            index=False,
        )
        print(f"  [SAVED]")

    except Exception as e:
        print(f"  [ERROR] {e}")

    finally:
        # 메모리 해제
        try:
            del series, train, model
        except:
            pass
        gc.collect()

# ── 최종 출력 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"완료! Orders saved to: {ORDERS_FILE}")
if ORDERS_FILE.exists():
    print(pd.read_csv(ORDERS_FILE, dtype={"station_id": str}).to_string(index=False))