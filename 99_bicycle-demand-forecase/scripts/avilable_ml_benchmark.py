# Import libraries 
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from src.utils import load_rainfall_binary, make_features_rainfall_binary, evaluate_ml_forecast_collect

try:
    import xgboost as xgb
    HAS_XGB = True
    print("XGBoost loaded successfully")
except ImportError:
    HAS_XGB = False
    print("[Warning] XGBoost not found -> pip install xgboost")



# Settings
DATA_DIR        = Path("data")   # interpolated_available_YYYYMM.csv
SEASONAL_PERIOD = 24             # 일일 계절 주기 (시간 단위 → S=24)
TEST_DAYS       = 30             # 마지막 30일을 test set으로 사용
ALPHA = 0.5                      # Underestimate penalty rate 
TARGET_RENT_IDS = [
    2111, 2112, 2116, 2122, 2128, 2129, 2130,
    2139, 2143, 2148, 2159, 2171, 2172, 2178,
    2179, 2185, 2186, 2191, 2198, 2199, 3310, 
    3311, 3802,
]

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

# Define functions
def load_available_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("interpolated_available_*.csv"))
    if not files:
        raise FileNotFoundError(f"interpolated_available_*.csv 파일이 없습니다: {data_dir}")
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["datetime"] = pd.to_datetime(
        df["DATE"].astype(str) + df["TIME"].astype(str).str.zfill(2),
        format="%Y%m%d%H"
    )
    df["RENT_ID"] = df["RENT_ID"].astype(int)
    return df

def make_series(df: pd.DataFrame, rent_id: int) -> pd.Series:
    """
    특정 정류소의 시간별 AVAILABLE 시계열 생성
    interpolated_available_*.csv는 이미 보간 완료 → 결측 없음
    """
    sub = (
        df[df["RENT_ID"] == rent_id]
        .set_index("datetime")["AVAILABLE"]
        .sort_index()
    )
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="h")
    sub = sub.reindex(full_idx)
    return sub.astype(float)

def asymmetric_obj(y_pred, dtrain, alpha=2.0):
    """
    Custom asymmetric loss for XGBoost (gradient & hessian).
    Under-prediction (y_pred < y_true): alpha * 2 * error
    Over-prediction  (y_pred >= y_true): 2 * error
    """
    y_true = dtrain.get_label()
    errors = y_pred - y_true

    grad = np.where(errors < 0,
                    alpha * 2 * errors,
                    2 * errors)
    hess = np.where(errors < 0,
                    alpha * 2 * np.ones_like(errors),
                    2 * np.ones_like(errors))
    return grad, hess

# Load data & Preporcessing.
df_raw = load_available_csvs(DATA_DIR)
weather_df = load_rainfall_binary(DATA_DIR)

# Collect all station results
all_results = []
for TARGET_RENT_ID in TARGET_RENT_IDS:
    print("\n" + "─" * 60)
    print(f"[Station {TARGET_RENT_ID}] Model fitting started...")
    print("─" * 60)

    try:
        series = make_series(df_raw, TARGET_RENT_ID)
        df_feat = make_features_rainfall_binary(series, weather_df)


        # Train / Test Split
        test_hours = TEST_DAYS * 24
        train_df = df_feat.iloc[:-test_hours]
        test_df  = df_feat.iloc[-test_hours:]
        FEATURE_COLS = [c for c in df_feat.columns if c != "CNT"]
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["CNT"]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df["CNT"]


        # =========================================================
        # Random Forest
        # =========================================================
        rf_model = RandomForestRegressor(
            n_estimators=200,     # 트리 개수
            max_depth=10,         # 트리 최대 깊이
            min_samples_leaf=5,   # 리프 노드 최소 샘플 수
            n_jobs=-1,            # 병렬 처리
            random_state=42,
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        perf_rf = evaluate_ml_forecast_collect(y_test, 
                                               y_pred_rf,
                                               model_name="Random Forest", 
                                               alpha=ALPHA, 
                                               station_id=TARGET_RENT_ID
                                               )
        all_results.append(perf_rf)
        
        # =========================================================
        # XGBoost
        # =========================================================
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred_xgb = xgb_model.predict(X_test)
        perf_xgb   = evaluate_ml_forecast_collect(y_test, 
                                               y_pred_xgb,
                                               model_name="XGBoost", 
                                               alpha=ALPHA, 
                                               station_id=TARGET_RENT_ID
                                               )
        all_results.append(perf_xgb)


        # ── XGBoost with Asymmetric Loss Function ────────────────────
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest  = xgb.DMatrix(X_test,  label=y_test)

        params = {
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "seed":             42,
        }

        xgb_asym_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=300,
            obj=lambda y_pred, dtrain: asymmetric_obj(y_pred, dtrain, alpha=ALPHA),
            evals=[(dtest, "test")],
            verbose_eval=False,
        )


        y_pred_xgb_asym= xgb_asym_model.predict(dtest)
        perf_xgb_asym = evaluate_ml_forecast_collect(y_test, 
                                               y_pred_xgb_asym,
                                               model_name=f"XGBoost (alpha={ALPHA})", 
                                               alpha=ALPHA, 
                                               station_id=TARGET_RENT_ID
                                               )
        all_results.append(perf_xgb_asym)

        print(f"[Station {TARGET_RENT_ID}] Complete!")

    except Exception as e:
        print(f"Error at station {TARGET_RENT_ID}: {e}")

print("\n" + "=" * 60)
print("All station benchmarks completed successfully.")
print("Results saved to: results/station_benchmark_results.csv")
print("=" * 60)

# Save benchmark results
Path("results").mkdir(exist_ok=True)
results_df = pd.DataFrame(all_results)
results_df.to_csv(
    "results/available_ml_benchmark_results.csv",
    index=False,
)

print("\nBenchmark completed.")
print(results_df.head())