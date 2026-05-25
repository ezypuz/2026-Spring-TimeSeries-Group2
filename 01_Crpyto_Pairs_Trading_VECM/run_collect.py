"""
CLI entry point: data collection.

Usage examples
--------------
# Collect all USDT pairs, 1-hour candles, last 365 days (defaults)
python run_collect.py

# Daily candles, last 2 years, no volume filter
python run_collect.py --interval 1d --days 730 --min-volume 0

# Specific symbols only
python run_collect.py --symbols BTCUSDT ETHUSDT SOLUSDT

# Custom output directory
python run_collect.py --output-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, DEFAULT_INTERVAL, DEFAULT_DAYS, DEFAULT_MIN_VOLUME_USDT, EXCHANGE_ID
from src.data import CCXTCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto USDT pairs OHLCV data collector (ccxt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exchange",   "-e", default=EXCHANGE_ID,
                        help="ccxt exchange id, e.g. binance bybit okx kraken")
    parser.add_argument("--interval",   "-i", default=DEFAULT_INTERVAL,
                        help="Candle interval, e.g. 1m 15m 1h 4h 1d")
    parser.add_argument("--days",       "-d", type=int, default=DEFAULT_DAYS,
                        help="Number of days back from today")
    parser.add_argument("--min-volume", "-v", type=float,
                        default=DEFAULT_MIN_VOLUME_USDT, dest="min_volume",
                        help="24h quote-volume (USDT) filter; 0 = no filter")
    parser.add_argument("--output-dir", "-o", type=Path, default=DATA_DIR,
                        dest="output_dir",
                        help="Directory to write CSV files")
    parser.add_argument("--symbols",    "-s", nargs="+", default=None,
                        help="Explicit symbol list; omit to collect all pairs")
    args = parser.parse_args()

    collector = CCXTCollector(exchange_id=args.exchange)
    collector.collect_all(
        symbols    = args.symbols,
        interval   = args.interval,
        days       = args.days,
        min_volume = args.min_volume,
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
