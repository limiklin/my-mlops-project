import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ================================
# 1. 데이터 불러오기
# ================================
data_path = r"C:\커리어\내 프로젝트\기계학습기초\data.xlsx"
df = pd.read_excel(data_path)

# ================================
# 2. ECLO 계산
#    ECLO = 1*MI + 3*MO + 5*SE + 10*FA
# ================================
df["eclo"] = (
    1 * df["부상신고자수"]
    + 3 * df["경상자수"]
    + 5 * df["중상자수"]
    + 10 * df["사망자수"]
)

# ================================
# 3. 라벨링 매핑 정의
# ================================
label_mappings = {
    "주야": {"주간": 0, "야간": 1},
    "법규위반": {
        "교차로운행방법위반": 0,
        "기타": 1,
        "보행자보호의무위반": 2,
        "신호위반": 3,
        "안전거리미확보": 4,
        "안전운전불이행": 5,
        "중앙선침범": 6,
    },
    "피해운전자 차종": {
        "개인형이동수단(PM)": 0,
        "건설기계": 1,
        "기타불명": 2,
        "보행자": 3,
        "승용": 4,
        "승합": 5,
        "원동기": 6,
        "이륜": 7,
        "자전거": 8,
        "특수": 9,
        "화물": 10,
        "": 11,   # 값 없음 처리
    },
    "노면상태": {"건조": 0, "기타": 1, "적설/습기": 2},
    "기상상태": {"기타": 0, "눈": 1, "맑음": 2, "비": 3, "흐림": 4},
    "도로형태": {
        "교차로-교차로부근": 0,
        "교차로-교차로안": 1,
        "교차로-교차로횡단보도내": 2,
        "기타": 3,
        "단일로-고가도로위": 4,
        "단일로-교량": 5,
        "단일로-기타": 6,
        "단일로-지하차도(도로운저)": 7,
        "단일로-터널안": 8,
        "미분류": 9,
        "주차장-주차장": 10,
    },
    "가해운전자 성별": {"남성": 0, "여성": 1},
    "가해운전자 연령대": {
        "9세~29세": 0,
        "30세~39세": 1,
        "40세~49세": 2,
        "50세~59세": 3,
        "60세~90세": 4,
    },
    "가해운전자 차종": {
    "개인형이동수단(PM)": 0,
    "건설기계": 1,
    "기타불명": 2,
    "승용": 3,
    "승합": 4,
    "원동기": 5,
    "이륜": 6,
    "자전거": 7,
    "특수": 8,
    "화물": 9
    },
    "사고유형": {
    "차대사람-기타": 0,
    "차대사람-횡단중": 1,
    "차대사람-보도통행중": 2,
    "차대사람-차도통행중": 3,
    "차대차-정면충돌": 4,
    "차대차-기타": 5,
    "차대차-추돌": 7,
    "차대차-측면충돌": 8,
    "차대차-후진중충돌": 9,
    "차량단독-공작물충돌": 10,
    "차량단독-기타": 11
    }
}

# ================================
# 4. 라벨링 적용
# ================================
for col, mapping in label_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ================================
# 5. 타깃/피처 분리
#    (예: eclo를 타깃으로 사용)
# ================================
y = df["eclo"]
X = df.drop(columns=["eclo", "사망자수", "중상자수", "경상자수", "부상신고자수"])  # ECLO 계산에 쓴 원본 칼럼 제거

# ================================
# 6. 범주형 변수 원-핫 인코딩
# ================================
X = pd.get_dummies(X)

# ================================
# 7. 데이터 분할
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 8. MLflow 설정
# ================================
mlflow.set_tracking_uri("http://localhost:5001")   # MLflow 서버 주소
mlflow.set_experiment("Traffic Accident Prediction")  # 실험 이름

# ================================
# 9. 모델 학습 및 기록
# ================================
with mlflow.start_run():
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # MLflow에 기록
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # 모델 업로드 (Artifact)
    mlflow.sklearn.log_model(model, artifact_path="model")

print("✅ MLflow에 모델과 실험 기록이 저장되었습니다!")