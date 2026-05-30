import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera, shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import LONG_MISSING as _DEFAULT_LONG_MISSING


def load_filtered_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("filtered_2*.csv"))
    if not files:
        raise FileNotFoundError(f"filtered_*.csv 파일이 없습니다: {data_dir}")
    parts = [
        pd.read_csv(f, dtype={"RENT_DATE": str, "RENT_TIME": str,
                               "RENT_ID": str, "CNT": int})
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df["datetime"] = pd.to_datetime(
        df["RENT_DATE"] + df["RENT_TIME"].str.zfill(2),
        format="%Y%m%d%H"
    )
    return df


def make_series(df: pd.DataFrame, rent_id: str,
                long_missing: dict = None) -> pd.Series:
    """
    특정 정류소의 시간별 CNT 시계열 생성.
    - 전체 시간 인덱스로 reindex (이용 없는 시간대 = 0)
    - 장기 결측 구간은 NaN 유지 (학습 윈도우에서 제외)
    - 단기(≤72h) 선형 보간, 중기(73~336h) 요일+시간 평균 대체
    """
    if long_missing is None:
        long_missing = _DEFAULT_LONG_MISSING

    sub = (
        df[df["RENT_ID"] == rent_id]
        .set_index("datetime")["CNT"]
        .sort_index()
    )
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="h")
    sub = sub.reindex(full_idx, fill_value=0).astype(float)

    if rent_id in long_missing:
        for (start, end) in long_missing[rent_id]:
            sub[start:end] = np.nan

    sub = sub.interpolate(method="linear", limit=72, limit_direction="both")

    still_na = sub.isna()
    if still_na.any():
        tmp_df = pd.DataFrame({
            "CNT": sub, "DOW": sub.index.dayofweek, "HOUR": sub.index.hour
        })
        mean_table = tmp_df.groupby(["DOW", "HOUR"])["CNT"].mean()
        for idx in sub[still_na].index:
            sub[idx] = mean_table.get((idx.dayofweek, idx.hour), 0)

    return sub


def load_weather(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_weather.csv"))
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="cp949",
                         usecols=["일시", "기온(°C)"],
                         dtype={"기온(°C)": float})
        parts.append(df)
    weather = pd.concat(parts, ignore_index=True)
    weather["datetime"] = pd.to_datetime(weather["일시"])
    weather = (
        weather
        .set_index("datetime")
        .drop(columns=["일시"])
        .rename(columns={"기온(°C)": "temp"})
        .sort_index()
        .resample("h").mean()
    )
    weather["temp"] = weather["temp"].interpolate(method="linear")
    return weather

def load_weather_full(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_weather.csv"))
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="cp949",
                         usecols=["일시", "기온(°C)", "강수량(mm)"],
                         dtype={"기온(°C)": float, "강수량(mm)": float})
        parts.append(df)
    weather = pd.concat(parts, ignore_index=True)
    weather["datetime"] = pd.to_datetime(weather["일시"])
    weather = (
        weather
        .set_index("datetime")
        .drop(columns=["일시"])
        .rename(columns={"기온(°C)": "temp", "강수량(mm)": "rainfall"})
        .sort_index()
        .resample("h").mean()
    )
    weather["temp"] = weather["temp"].interpolate(method="linear")
    weather["rainfall"] = weather["rainfall"].fillna(0)
    return weather

def load_rainfall_binary(data_dir: Path, threshold: float = 1.0) -> pd.DataFrame:
    """
    날씨 CSV에서 기온 + 강수 여부(binary) 로드.
    - threshold mm 이상이면 1, 미만이면 0
    - 11~3월은 3시간 간격 관측: 결측은 앞뒤 관측값이 둘 다 존재하고 둘 다 1일 때만 1, 나머지 0
    - 4~10월: 결측 → 0 (보수적 처리)
    """
    files = sorted(data_dir.glob("*_weather.csv"))
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="cp949",
                         usecols=["일시", "기온(°C)", "강수량(mm)"],
                         dtype={"기온(°C)": float, "강수량(mm)": float})
        parts.append(df)
    weather = pd.concat(parts, ignore_index=True)
    weather["datetime"] = pd.to_datetime(weather["일시"])
    weather = (
        weather
        .set_index("datetime")
        .drop(columns=["일시"])
        .rename(columns={"기온(°C)": "temp", "강수량(mm)": "rainfall"})
        .sort_index()
        .resample("h").mean()
    )
    weather["temp"] = weather["temp"].interpolate(method="linear")

    # 관측값 기준 binary (결측은 NaN 유지)
    raw_binary = (weather["rainfall"] >= threshold).astype(float)
    raw_binary[weather["rainfall"].isna()] = np.nan

    rain_binary = pd.Series(0, index=weather.index)

    for ts in weather.index[weather["rainfall"].isna()]:
        month = ts.month

        # 4~10월: 보수적으로 0
        if 4 <= month <= 10:
            rain_binary[ts] = 0
            continue

        # 11~3월: 앞뒤 3시간 이내 관측값 탐색
        prev_val = None
        next_val = None

        for h in range(1, 4):
            prev_ts = ts - pd.Timedelta(hours=h)
            if prev_ts in raw_binary.index and not np.isnan(raw_binary[prev_ts]):
                # 3시간 이내에 관측값 존재 → 사용
                prev_val = raw_binary[prev_ts]
                break
            # h시간 전도 결측이면 → 간격이 3시간 초과로 볼 수 없으므로 탐색 중단
            if prev_ts in raw_binary.index and np.isnan(raw_binary[prev_ts]):
                continue

        for h in range(1, 4):
            next_ts = ts + pd.Timedelta(hours=h)
            if next_ts in raw_binary.index and not np.isnan(raw_binary[next_ts]):
                next_val = raw_binary[next_ts]
                break
            if next_ts in raw_binary.index and np.isnan(raw_binary[next_ts]):
                continue

        # 앞뒤 둘 다 존재하고 둘 다 1일 때만 1
        if prev_val is not None and next_val is not None:
            rain_binary[ts] = 1 if (prev_val == 1 and next_val == 1) else 0
        else:
            # 앞뒤 중 하나라도 없으면 (간격 초과) → 0
            rain_binary[ts] = 0

    # 관측값 있는 시간대는 threshold 기준 그대로 적용
    observed_mask = weather["rainfall"].notna()
    rain_binary[observed_mask] = (weather["rainfall"][observed_mask] >= threshold).astype(int)

    weather["rain_binary"] = rain_binary
    return weather.drop(columns=["rainfall"])

def make_features(series: pd.Series, weather_df: pd.DataFrame) -> pd.DataFrame:
    """ML용 피처 행렬 생성 (시간 특성 + lag + 기온)."""
    df = pd.DataFrame({"CNT": series})
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["lag_1"]      = df["CNT"].shift(1)
    df["lag_24"]     = df["CNT"].shift(24)
    df["lag_168"]    = df["CNT"].shift(168)
    df["rolling_mean_24"]  = df["CNT"].shift(1).rolling(24).mean()
    df["rolling_mean_168"] = df["CNT"].shift(1).rolling(168).mean()
    df["temp"] = weather_df["temp"].reindex(df.index).ffill()
    return df.dropna()

def make_features_full(series: pd.Series, weather_df: pd.DataFrame) -> pd.DataFrame:
    """ML용 피처 행렬 생성 (시간 특성 + lag + 기온)."""
    df = pd.DataFrame({"CNT": series})
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["lag_1"]      = df["CNT"].shift(1)
    df["lag_24"]     = df["CNT"].shift(24)
    df["lag_168"]    = df["CNT"].shift(168)
    df["rolling_mean_24"]  = df["CNT"].shift(1).rolling(24).mean()
    df["rolling_mean_168"] = df["CNT"].shift(1).rolling(168).mean()
    df["temp"] = weather_df["temp"].reindex(df.index).ffill()
    df["rainfall"] = (weather_df["rainfall"].reindex(df.index).fillna(0))
    return df.dropna()

def make_features_ML(series: pd.Series, weather_df: pd.DataFrame) -> pd.DataFrame:
    """ML용 피처 행렬 생성 (시간 특성 + lag + 기온 + 강수여부 binary)."""
    df = pd.DataFrame({"CNT": series})
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["lag_1"]      = df["CNT"].shift(1)
    df["lag_24"]     = df["CNT"].shift(24)
    df["lag_168"]    = df["CNT"].shift(168)
    df["rolling_mean_24"]  = df["CNT"].shift(1).rolling(24).mean()
    df["rolling_mean_168"] = df["CNT"].shift(1).rolling(168).mean()
    df["temp"]        = weather_df["temp"].reindex(df.index).ffill()
    df["rain_binary"] = weather_df["rain_binary"].reindex(df.index).fillna(0).astype(int)
    return df.dropna()

def adf_test(series: pd.Series, name: str = "", maxlag: int = 6) -> bool:
    result = adfuller(series.dropna(), maxlag=maxlag, autolag="AIC")
    print(f"[ADF Test] {name}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Critical (1%) : {result[4]['1%']:.4f}")
    print(f"  Critical (5%) : {result[4]['5%']:.4f}")
    stationary = result[1] < 0.05
    print(f"  결론          : {'정상 (stationary)' if stationary else '비정상 → 차분 필요'}")
    return stationary


def diagnose_residuals(fitted_model, model_name: str = "SARIMA") -> dict:
    residuals = pd.Series(fitted_model.resid.dropna())

    print(f"{'='*55}")
    print(f"  잔차 진단: {model_name}")
    print(f"{'='*55}")

    lb = acorr_ljungbox(residuals, lags=[10, 20, 48], return_df=True)
    print("\n[Ljung-Box Test] H0: 잔차에 자기상관 없음")
    print(lb[["lb_stat", "lb_pvalue"]].to_string())
    lb_pass = (lb["lb_pvalue"] > 0.05).all()
    print(f"  → {'백색잡음' if lb_pass else '자기상관 존재'}")

    jb_stat, jb_pval = jarque_bera(residuals)
    print(f"\n[Jarque-Bera Test] H0: 정규분포")
    print(f"  통계량={jb_stat:.4f}, p-value={jb_pval:.4f}")
    print(f"  → {'정규성 기각 불가' if jb_pval > 0.05 else '정규성 기각'}")

    sw_sample = residuals.sample(min(len(residuals), 5000), random_state=42)
    sw_stat, sw_pval = shapiro(sw_sample)
    print(f"\n[Shapiro-Wilk Test] H0: 정규분포")
    print(f"  통계량={sw_stat:.4f}, p-value={sw_pval:.4f}")
    print(f"  → {'정규성 기각 불가' if sw_pval > 0.05 else '정규성 기각'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Residual Diagnostics - {model_name}", fontsize=13)
    axes[0, 0].plot(residuals.values, alpha=0.6)
    axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Residual Time Series")
    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Residual Histogram")
    plot_acf(residuals, lags=48, ax=axes[1, 0], title="Residual ACF")
    plot_pacf(residuals, lags=48, ax=axes[1, 1], title="Residual PACF")
    plt.tight_layout()
    plt.show()

    return {"model": model_name, "lb_pass": lb_pass,
            "jb_pval": round(jb_pval, 4), "sw_pval": round(sw_pval, 4)}


def asymmetric_rmse(actual, forecast, alpha: float = 2.0) -> float:
    """과소예측에 alpha배 패널티를 적용한 RMSE."""
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    errors = forecast - actual
    loss = np.where(errors < 0, alpha * errors**2, errors**2)
    return float(np.sqrt(loss.mean()))


def evaluate_forecast(fitted_model, test: pd.Series,
                      model_name: str = "SARIMA",
                      exog_test: pd.DataFrame = None,
                      alpha: float = 2.0) -> dict:
    if exog_test is not None:
        forecast = fitted_model.forecast(steps=len(test), exog=exog_test)
    else:
        forecast = fitted_model.forecast(steps=len(test))

    forecast  = pd.Series(forecast.values, index=test.index)
    errors    = forecast - test

    rmse      = np.sqrt(mean_squared_error(test, forecast))
    mae       = mean_absolute_error(test, forecast)
    asym      = asymmetric_rmse(test, forecast, alpha=alpha)
    under_rate = (errors < 0).mean() * 100

    print(f"[{model_name}]")
    print(f"  RMSE              : {rmse:.3f}")
    print(f"  MAE               : {mae:.3f}")
    print(f"  Asymmetric RMSE   : {asym:.3f}  (alpha={alpha})")
    print(f"  과소예측 비율      : {under_rate:.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(test.index, test.values, label="Actual (y)", alpha=0.7)
    axes[0].plot(forecast.index, forecast.values,
                 label=f"Forecast ({model_name})", linestyle="--", alpha=0.8)
    axes[0].set_title(f"Forecast vs Actual - {model_name}  "
                      f"[RMSE={rmse:.2f} | Asym.RMSE={asym:.2f}]")
    axes[0].legend()
    colors = ["red" if e < 0 else "steelblue" for e in errors]
    axes[1].bar(range(len(errors)), errors.values, color=colors, alpha=0.6, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Forecast Error | Red=Under, Blue=Over | Under-pred={under_rate:.1f}%")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        f"Asym.RMSE(α={alpha})": round(asym, 3),
        "과소예측비율(%)": round(under_rate, 1),
    }


def evaluate_ml_forecast(y_test: pd.Series, y_pred,
                          model_name: str, alpha: float = 2.0) -> dict:
    forecast  = pd.Series(y_pred, index=y_test.index)
    errors    = forecast - y_test

    rmse      = np.sqrt(mean_squared_error(y_test, forecast))
    mae       = mean_absolute_error(y_test, forecast)
    asym      = asymmetric_rmse(y_test.values, y_pred, alpha=alpha)
    under_rate = (errors < 0).mean() * 100
    over_rate = (errors > 0).mean() * 100

    print(f"[{model_name}]")
    print(f"  RMSE            : {rmse:.3f}")
    print(f"  MAE             : {mae:.3f}")
    print(f"  Asymmetric RMSE : {asym:.3f}  (alpha={alpha})")
    print(f"  Under-pred rate : {under_rate:.1f}%")
    print(f"  Over-pred rate : {over_rate:.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(y_test.index, y_test.values, label="Actual (y)", alpha=0.7)
    axes[0].plot(forecast.index, forecast.values,
                 label=f"Forecast ({model_name})", linestyle="--", alpha=0.8)
    axes[0].set_title(f"Forecast vs Actual - {model_name}  "
                      f"[RMSE={rmse:.3f} | Asym.RMSE={asym:.3f}]")
    axes[0].legend()
    colors = ["red" if e < 0 else "steelblue" for e in errors]
    axes[1].bar(range(len(errors)), errors.values, color=colors, alpha=0.6, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Forecast Error | Red=Under, Blue=Over | Under-pred={under_rate:.1f}% | Over-pred={over_rate:.1f}%")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        f"Asym.RMSE(alpha={alpha})": round(asym, 3),
        "Under-pred rate(%)": round(under_rate, 1),
        "Over-pred rate(%)": round(over_rate, 1)
    }

def evaluate_ml_forecast_collect(y_test: pd.Series, y_pred,
                          model_name: str, alpha: float = 2.0, station_id: int = 2128) -> dict:
    forecast  = pd.Series(y_pred, index=y_test.index)
    errors    = forecast - y_test

    rmse      = np.sqrt(mean_squared_error(y_test, forecast))
    mae       = mean_absolute_error(y_test, forecast)
    asym      = asymmetric_rmse(y_test.values, y_pred, alpha=alpha)
    under_rate = (errors < 0).mean() * 100
    over_rate = (errors > 0).mean() * 100

    return {
        "station_id": station_id,
        "model": model_name,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        f"Asym.RMSE(alpha={alpha})": round(asym, 3),
        "Under-pred rate(%)": round(under_rate, 1),
        "Over-pred rate(%)": round(over_rate, 1)
    }