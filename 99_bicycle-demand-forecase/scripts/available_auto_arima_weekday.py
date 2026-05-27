"""
2023년 available 데이터(평일)로 statsforecast AutoARIMA 차수 탐색
statsforecast는 Cython 기반으로 pmdarima 대비 메모리/속도 효율적
데이터: data/available_2023MM.csv (available_202301.csv ~ available_202312.csv)
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
SEASONAL_PERIOD = 24
TARGET_RENT_ID  = 2128

# ── 2023년 available 데이터 로드 ──────────────────────────────────────────────
files = sorted(DATA_DIR.glob("available_2023*.csv"))
if not files:
    raise FileNotFoundError(f"available_2023*.csv 파일이 없습니다: {DATA_DIR}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["datetime"] = pd.to_datetime(
    df["DATE"].astype(str) + df["TIME"].astype(str).str.zfill(2),
    format="%Y%m%d%H"
)
df["RENT_ID"] = df["RENT_ID"].astype(int)
print(f"로드 파일 수: {len(files)}개")
print(f"기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")

# ── 평일 시계열 생성 ──────────────────────────────────────────────────────────
sub = df[df["RENT_ID"] == TARGET_RENT_ID].set_index("datetime")["AVAILABLE"].sort_index()
full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="h")
series = sub.reindex(full_idx).astype(float)

train = series[series.index.dayofweek < 5].dropna()
print(f"평일 train: {train.index[0].date()} ~ {train.index[-1].date()}  ({len(train):,}시간)")

# ── AutoARIMA 차수 탐색 ───────────────────────────────────────────────────────
print("\nstatsforecast AutoARIMA 실행 중...")
model = AutoARIMA(
    season_length=SEASONAL_PERIOD,
    d=0,
    D=1,
    stepwise=True,
    approximation=False,
    ic="aic",
    trace=True,
)
model = model.fit(train.values)

# 탐색된 차수 추출: arma = [p, q, P, Q, S, d, D]
arma = model.model_["arma"]
p, q, P, Q, S, d, D = int(arma[0]), int(arma[1]), int(arma[2]), int(arma[3]), int(arma[4]), int(arma[5]), int(arma[6])
best_order          = (p, d, q)
best_seasonal_order = (P, D, Q, S)

print(f"\n최적 차수: ARIMA{best_order}x{best_seasonal_order}")
print(f"  p={p}, d={d}, q={q}  |  P={P}, D={D}, Q={Q}, S={S}")
print(f"AIC : {model.model_['aic']:.3f}")
print(f"BIC : {model.model_['bic']:.3f}")
