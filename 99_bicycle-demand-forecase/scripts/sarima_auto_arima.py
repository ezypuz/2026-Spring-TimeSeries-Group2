"""
available_2024MM.csv 파일들을 합쳐 전체 대여소의 시간별 총 가용 대수를 집계하고,
pmdarima의 auto_arima로 SARIMA 차수를 자동 탐색하는 스크립트.
"""

import glob
import os
import pandas as pd
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── 1. 데이터 로드 및 병합 ─────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "available_2023??.csv")))
if not files:
    raise FileNotFoundError(f"available_2024??.csv 파일을 찾을 수 없습니다: {DATA_DIR}")

print(f"로드할 파일 수: {len(files)}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ── 2. 전처리: 시간 인덱스 생성 ───────────────────────────────────────────────
df["datetime"] = pd.to_datetime(
    df["DATE"].astype(str) + df["TIME"].astype(str).str.zfill(2),
    format="%Y%m%d%H",
)

# 전체 대여소 기준 시간별 총 가용 대수 집계
ts = (
    df.groupby("datetime")["AVAILABLE"]
    .sum()
    .asfreq("h")          # 1시간 간격 명시
    .ffill()  # 결측 시각 전방 채움
)

print(f"\n시계열 기간: {ts.index[0]} ~ {ts.index[-1]}")
print(f"총 데이터 포인트: {len(ts)}")
print(f"결측 수: {ts.isna().sum()}")
print(f"\n처음 5행:\n{ts.head()}")

# ── 3. auto_arima로 SARIMA 차수 탐색 ─────────────────────────────────────────
#   - m=24  : 일별 계절성 (시간 단위 데이터)
#   - seasonal=True : 계절성 포함 탐색
#   - stepwise=True : 탐색 속도 향상
#   - information_criterion='aic' : AIC 기준 최적 모델 선택
print("\n[auto_arima] SARIMA 차수 탐색 중... (시간이 다소 걸릴 수 있습니다)")

model = auto_arima(
    ts,
    m=24,
    seasonal=True,
    stepwise=True,
    information_criterion="aic",
    start_p=0, max_p=1,
    start_q=0, max_q=1,
    start_P=0, max_P=1,
    start_Q=0, max_Q=1,
    d=0,    # 자동 차분 차수 결정
    D=1,    # 자동 계절 차분 차수 결정
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    n_jobs=-1,
)

# ── 4. 결과 출력 ──────────────────────────────────────────────────────────────
order    = model.order           # (p, d, q)
s_order  = model.seasonal_order  # (P, D, Q, m)

print("\n" + "=" * 50)
print("최적 SARIMA 차수")
print("=" * 50)
print(f"  비계절 차수  (p, d, q)        = {order}")
print(f"  계절   차수  (P, D, Q, m={s_order[3]}) = {s_order}")
print(f"  AIC                           = {model.aic():.4f}")
print("=" * 50)
print("\n모델 요약:")
print(model.summary())
