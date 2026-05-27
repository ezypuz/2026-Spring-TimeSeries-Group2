"""
2023~2025년 따릉이 대여 데이터(평일)로 statsforecast AutoARIMA 차수 탐색.
statsforecast는 Cython 기반으로 pmdarima 대비 메모리/속도 효율적.

Usage:
    # terminal
    python scripts/sarima_auto_arima_weekend.py

    # ipynb 내부
    %run scripts/sarima_auto_arima_weekend.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, SEASONAL_PERIOD, TARGET_RENT_ID, LONG_MISSING
from src.utils  import load_filtered_csvs, make_series

# ── Load data (2023~2025) ─────────────────────────────────────────────────────
df_raw = load_filtered_csvs(DATA_DIR)
df_raw = df_raw[df_raw["datetime"].dt.year >= 2023]
print(f"Data range : {df_raw['datetime'].min().date()} ~ {df_raw['datetime'].max().date()}")

# ── Build weekend series ──────────────────────────────────────────────────────
series   = make_series(df_raw, str(TARGET_RENT_ID))
series_weekend = series[series.index.dayofweek >= 5].dropna()
print(f"Weekend train : {series_weekend.index[0].date()} ~ {series_weekend.index[-1].date()}  "
      f"({len(series_weekend):,} hours)")

# ── AutoARIMA search ──────────────────────────────────────────────────────────
print("\nRunning statsforecast AutoARIMA...")
model = AutoARIMA(
    season_length=SEASONAL_PERIOD,
    max_p=2, max_q=2,
    max_P=2, max_Q=2,
    d=0,
    D=1,
    stepwise=True,
    approximation=False,
    ic="aic",
    trace=True,
)
model = model.fit(series_weekend.values)

# Extract order: arma = [p, q, P, Q, S, d, D]
arma = model.model_["arma"]
p, q   = int(arma[0]), int(arma[1])
P, Q   = int(arma[2]), int(arma[3])
S      = int(arma[4])
d, D   = int(arma[5]), int(arma[6])

best_order          = (p, d, q)
best_seasonal_order = (P, D, Q, S)

print(f"\n{'='*50}")
print(f"Best order : ARIMA{best_order}x{best_seasonal_order}")
print(f"  p={p}, d={d}, q={q}  |  P={P}, D={D}, Q={Q}, S={S}")
print(f"AIC : {model.model_['aic']:.3f}")
print(f"BIC : {model.model_['bic']:.3f}")
print(f"{'='*50}")
print(f"\n# Paste into 01_sarima_weekend.ipynb Section 6:")
print(f"order          = {best_order}")
print(f"seasonal_order = {best_seasonal_order}")

# 결과를 전역 변수로 노출 (ipynb에서 %run 후 참조 가능)
BEST_ORDER          = best_order
BEST_SEASONAL_ORDER = best_seasonal_order