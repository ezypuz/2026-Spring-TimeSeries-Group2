# 2026 Spring 시계열 분석 프로젝트 — Group 2

서울 공공자전거 **따릉이** 대여 수요를 시계열 모델로 예측하고, 이를 바탕으로 정류소 간 **자전거 재배치 전략**을 수립하는 프로젝트입니다.

---

## 프로젝트 개요

- **데이터**: 23개 정류소의 시간별 대여 건수 (2021.01 ~ 2025.12)
- **핵심 지표**: Asymmetric RMSE (α=2) — 과소예측(자전거 고갈)에 2배 패널티
- **모델**: SARIMA, ARIMAX, Random Forest, XGBoost (비대칭 손실)

---

## 프로젝트 구조

```
99_bicycle-demand-forecase/
├── data/
│   ├── filtered_YYYYMM.csv     # 월별 정류소 대여 기록 (2021~2025)
│   ├── YYYY_weather.csv        # 연도별 기상 데이터 (기온)
│   └── target_rent_id.txt      # 분석 대상 23개 정류소 ID
├── notebooks/
│   ├── 01_sarima_weekday.ipynb # 평일 SARIMA / ARIMAX
│   ├── 02_sarima_weekend.ipynb # 주말 SARIMA / ARIMAX
│   └── 03_ml_models.ipynb      # Random Forest / XGBoost
├── src/
│   ├── config.py               # 공용 설정 (경로, 파라미터, 결측 구간)
│   └── utils.py                # 공용 함수 (데이터 로드, 전처리, 평가)
├── scripts/
│   ├── find_long_missing.py    # 장기 결측 구간 탐지
│   ├── visualize_missing.py    # 결측 히트맵 시각화
│   └── visualize_missing_daily.py
├── reports/
│   ├── missing_value_report.md
│   ├── modeling_progress_report.md
│   └── research_proposal.pdf
└── figures/                    # 시각화 결과물
```

---

## 분석 파이프라인

```
원본 데이터 (filtered_*.csv)
    ↓
결측치 처리 (단기: 선형 보간 / 중기: 요일 평균 / 장기: 제외)
    ↓
평일 / 주말 분리
    ↓
┌─────────────────┬─────────────────┐
│  SARIMA/ARIMAX  │   ML 모델       │
│  (01, 02)       │   (03)          │
└─────────────────┴─────────────────┘
    ↓
Asymmetric RMSE (α=2) 기준 모델 비교
```

---

## 주요 결과 (정류소 02128 기준, 평일)

| 모델 | RMSE | Asym.RMSE (α=2) | 과소예측률 |
|------|------|-----------------|-----------|
| SARIMA | 0.647 | 0.796 | 25.7% |
| ARIMAX | 0.651 | 0.799 | 38.2% |
| Random Forest | 0.601 | 0.762 | 27.9% |
| XGBoost | 0.599 | 0.763 | 27.9% |
| **XGBoost (α=2)** | **0.650** | **0.761** | **26.4%** |

→ XGBoost (비대칭 손실) 이 과소예측률과 Asym.RMSE 모두 균형 잡힌 성능.

---

## 실행 방법

```bash
pip install -r requirements.txt
```

노트북은 `notebooks/` 폴더에서 순서대로 실행합니다.  
`src/config.py`의 `TARGET_RENT_ID`를 변경하면 다른 정류소를 분석할 수 있습니다.
