"""
Binance USDT Pairs - OHLCV Data Collector
==========================================
Binance에서 거래 가능한 모든 /USDT 페어의 OHLCV 시계열 데이터를 수집합니다.

Usage:
    python 01_data_collection.py
    python 01_data_collection.py --interval 1d --days 365 --min-volume 1000000
"""

import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Binance API 키 (공개 엔드포인트만 사용할 경우 빈 문자열로 둬도 됨)
API_KEY    = ""
API_SECRET = ""

# rate limit 대응: 요청 사이 대기 시간 (초)
REQUEST_DELAY     = 0.2   # 일반 요청 간격
ERROR_RETRY_DELAY = 5.0   # 에러 후 재시도 대기


# ---------------------------------------------------------------------------
# 심볼 목록 조회
# ---------------------------------------------------------------------------

def get_all_usdt_pairs(
    client: Client,
    min_volume_usdt: float = 0.0,
) -> list[str]:
    """
    Binance에서 거래 가능한 모든 USDT 페어 심볼 리스트를 반환합니다.

    Parameters
    ----------
    client : Client
        python-binance Client 인스턴스
    min_volume_usdt : float
        24시간 거래량(USDT) 필터. 0이면 필터링 없음.

    Returns
    -------
    list[str]
        예) ['BTCUSDT', 'ETHUSDT', ...]
    """
    exchange_info = client.get_exchange_info()
    tickers_24h   = {t["symbol"]: float(t["quoteVolume"])
                     for t in client.get_ticker()}

    symbols = []
    for s in exchange_info["symbols"]:
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s["isSpotTradingAllowed"]
        ):
            vol = tickers_24h.get(s["symbol"], 0.0)
            if vol >= min_volume_usdt:
                symbols.append(s["symbol"])

    symbols.sort()
    return symbols


# ---------------------------------------------------------------------------
# OHLCV 수집
# ---------------------------------------------------------------------------

def fetch_crypto_data(
    client: Client,
    symbol: str,
    interval: str = Client.KLINE_INTERVAL_1HOUR,
    start_date: datetime | None = None,
    end_date:   datetime | None = None,
) -> pd.DataFrame | None:
    """
    특정 심볼의 OHLCV 데이터를 수집합니다.
    Binance klines API는 한 번에 최대 1,000개를 반환하므로,
    기간이 길 경우 자동으로 페이지네이션하여 전체 데이터를 가져옵니다.

    Parameters
    ----------
    client   : Client
    symbol   : str   예) 'BTCUSDT'
    interval : str   예) Client.KLINE_INTERVAL_1HOUR ('1h')
    start_date : datetime  수집 시작 시각 (UTC, timezone-aware 권장)
    end_date   : datetime  수집 종료 시각 (UTC, timezone-aware 권장)

    Returns
    -------
    pd.DataFrame | None
        컬럼: open, high, low, close, volume
        인덱스: datetime (UTC)
        실패 시 None 반환
    """
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # millisecond timestamp
    start_ms = int(start_date.timestamp() * 1000)
    end_ms   = int(end_date.timestamp()   * 1000)

    all_klines: list = []
    current_start = start_ms

    while current_start < end_ms:
        try:
            klines = client.get_klines(
                symbol    = symbol,
                interval  = interval,
                startTime = current_start,
                endTime   = end_ms,
                limit     = 1000,
            )
        except BinanceAPIException as e:
            print(f"    [API 오류] {symbol}: {e}")
            return None
        except Exception as e:
            print(f"    [오류] {symbol}: {e}")
            return None

        if not klines:
            break

        all_klines.extend(klines)

        # 마지막 캔들 종료 시각 + 1ms 를 다음 시작점으로
        current_start = klines[-1][6] + 1

        time.sleep(REQUEST_DELAY)

    if not all_klines:
        return None

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    # 타입 변환
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = (
        df.set_index("open_time")
          [["open", "high", "low", "close", "volume"]]
          .sort_index()
    )

    # 중복 인덱스 제거 (페이지네이션 경계에서 간혹 발생)
    df = df[~df.index.duplicated(keep="first")]

    return df


# ---------------------------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------------------------

def save_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    DataFrame을 CSV로 저장합니다.

    Parameters
    ----------
    df       : pd.DataFrame  인덱스가 datetime인 OHLCV 데이터
    filepath : Path          저장할 전체 경로
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)


# ---------------------------------------------------------------------------
# 메인: 전체 페어 일괄 수집
# ---------------------------------------------------------------------------

def main(
    interval:       str   = "1h",
    days:           int   = 365,
    min_volume:     float = 500_000.0,
    output_dir:     Path  = DATA_DIR,
    symbols_subset: list[str] | None = None,
) -> None:
    """
    모든 USDT 페어(또는 지정 심볼)의 OHLCV 데이터를 수집하고 CSV로 저장합니다.

    Parameters
    ----------
    interval       : Binance kline 간격 문자열  예) '1h', '1d', '15m'
    days           : 수집할 최근 일수
    min_volume     : 24h 거래량(USDT) 최소 필터
    output_dir     : CSV 저장 루트 디렉토리
    symbols_subset : 지정 시 해당 심볼만 수집 (None이면 전체)
    """
    # interval 문자열 → Binance Client 상수 매핑
    interval_map = {
        "1m":  Client.KLINE_INTERVAL_1MINUTE,
        "3m":  Client.KLINE_INTERVAL_3MINUTE,
        "5m":  Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h":  Client.KLINE_INTERVAL_1HOUR,
        "2h":  Client.KLINE_INTERVAL_2HOUR,
        "4h":  Client.KLINE_INTERVAL_4HOUR,
        "6h":  Client.KLINE_INTERVAL_6HOUR,
        "8h":  Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d":  Client.KLINE_INTERVAL_1DAY,
        "3d":  Client.KLINE_INTERVAL_3DAY,
        "1w":  Client.KLINE_INTERVAL_1WEEK,
    }
    if interval not in interval_map:
        raise ValueError(f"지원하지 않는 interval: {interval}. 가능한 값: {list(interval_map)}")
    binance_interval = interval_map[interval]

    client = Client(API_KEY, API_SECRET)

    # 날짜 범위
    end_date   = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=days)
    date_tag   = end_date.strftime("%Y%m%d")

    print("=" * 60)
    print(f"  Binance USDT Pairs Data Collector")
    print(f"  interval  : {interval}")
    print(f"  기간      : {start_date.date()} ~ {end_date.date()} ({days}일)")
    print(f"  최소거래량 : {min_volume:,.0f} USDT")
    print(f"  저장 경로  : {output_dir}")
    print("=" * 60)

    # 심볼 목록
    if symbols_subset:
        symbols = symbols_subset
        print(f"\n지정된 심볼 {len(symbols)}개를 수집합니다.")
    else:
        print("\n[1/2] USDT 페어 목록 조회 중...")
        symbols = get_all_usdt_pairs(client, min_volume_usdt=min_volume)
        print(f"  → {len(symbols)}개 페어 발견")

    # 이미 수집된 파일은 스킵
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count    = 0
    fail_count    = 0
    total         = len(symbols)

    print(f"\n[2/2] 데이터 수집 시작 (총 {total}개)\n")

    for idx, symbol in enumerate(symbols, start=1):
        filename = output_dir / f"{symbol}_{interval}_{date_tag}.csv"

        # 이미 존재하면 스킵
        if filename.exists():
            print(f"  [{idx:>4}/{total}] {symbol:<15} → 이미 존재, 스킵")
            skip_count += 1
            continue

        print(f"  [{idx:>4}/{total}] {symbol:<15} 수집 중...", end=" ", flush=True)

        df = fetch_crypto_data(
            client,
            symbol,
            interval  = binance_interval,
            start_date = start_date,
            end_date   = end_date,
        )

        if df is None or df.empty:
            print("데이터 없음, 스킵")
            fail_count += 1
            time.sleep(ERROR_RETRY_DELAY)
            continue

        save_to_csv(df, filename)
        print(f"{len(df):>6}행 저장 완료  →  {filename.name}")
        success_count += 1

    # 결과 요약
    print("\n" + "=" * 60)
    print(f"  수집 완료!")
    print(f"  성공: {success_count}개  /  스킵(기존): {skip_count}개  /  실패: {fail_count}개")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binance USDT 페어 OHLCV 데이터 수집기"
    )
    parser.add_argument(
        "--interval", "-i",
        default="1h",
        help="캔들 간격 (기본값: 1h). 예) 1m, 15m, 1h, 4h, 1d",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="수집할 최근 일수 (기본값: 365)",
    )
    parser.add_argument(
        "--min-volume", "-v",
        type=float,
        default=500_000.0,
        dest="min_volume",
        help="24h 최소 거래량 USDT 필터 (기본값: 500000). 0이면 필터 없음",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DATA_DIR,
        dest="output_dir",
        help=f"CSV 저장 디렉토리 (기본값: {DATA_DIR})",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=None,
        help="수집할 심볼 지정 (미지정 시 전체). 예) BTCUSDT ETHUSDT",
    )
    args = parser.parse_args()

    main(
        interval       = args.interval,
        days           = args.days,
        min_volume     = args.min_volume,
        output_dir     = args.output_dir,
        symbols_subset = args.symbols,
    )
