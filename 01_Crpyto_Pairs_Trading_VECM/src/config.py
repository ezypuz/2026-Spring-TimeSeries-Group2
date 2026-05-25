"""
Project-wide configuration.
All magic numbers, paths, and defaults live here.
"""

import os
from pathlib import Path

# 프로젝트 루트의 .env 파일 로드 (없으면 환경변수 그대로 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수 직접 사용

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).resolve().parent.parent   # 01_Crpyto_Pairs_Trading_VECM/
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ---------------------------------------------------------------------------
# Exchange API  (ccxt 기반 — 거래소 변경 시 EXCHANGE_ID만 바꾸면 됩니다)
# 지원 거래소 예시: "binance", "bybit", "okx", "kraken"
# ---------------------------------------------------------------------------

EXCHANGE_ID         = os.getenv("EXCHANGE_ID", "binance")
EXCHANGE_API_KEY    = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL       = "1h"
DEFAULT_DAYS           = 365
DEFAULT_MIN_VOLUME_USDT = 500_000.0   # 24h 거래량 최소 필터 (USDT)

REQUEST_DELAY     = 0.2   # 초 — 정상 요청 간격
ERROR_RETRY_DELAY = 5.0   # 초 — 에러 후 대기

KLINE_LIMIT = 1000        # Binance API 단일 요청 최대 캔들 수

# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

ADF_MAX_LAGS    = None    # None = auto (AIC 기준)
ADF_ALPHA       = 0.05    # 유의수준
KPSS_NLAGS      = "auto"

# ---------------------------------------------------------------------------
# Cointegration
# ---------------------------------------------------------------------------

JOHANSEN_DET_ORDER = 0    # -1: no const, 0: const outside, 1: const inside
JOHANSEN_K_AR_DIFF = 1    # 차분 래그 수
COINT_ALPHA        = 0.05

# ---------------------------------------------------------------------------
# VECM
# ---------------------------------------------------------------------------

VECM_K_AR_DIFF  = 1
VECM_COINT_RANK = 1
VECM_DET_ORDER  = "co"    # statsmodels deterministic parameter

# ---------------------------------------------------------------------------
# Trading strategy
# ---------------------------------------------------------------------------

ZSCORE_ENTRY    = 2.0     # 진입 임계값 (|z| > ZSCORE_ENTRY)
ZSCORE_EXIT     = 0.0     # 청산 임계값 (|z| < ZSCORE_EXIT)
ZSCORE_WINDOW   = None    # None = 전체 샘플 기준, int = 롤링 윈도우

INITIAL_CAPITAL    = 10_000.0
TRANSACTION_COST   = 0.001   # 편도 0.1%
