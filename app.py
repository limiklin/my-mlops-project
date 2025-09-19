# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ---------------------------
# 1. 모델과 컬럼 로드
# ---------------------------
model = joblib.load("traffic_accident_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------------------------
        # 2. JSON 데이터 받기
        # ---------------------------
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame(json_data, index=[0])

        # ---------------------------
        # 3. 컬럼 순서 맞추기
        # ---------------------------
        df = df.reindex(columns=model_columns, fill_value=0)

        # ---------------------------
        # 4. 모델 예측
        # ---------------------------
        prediction = model.predict(df)

        # 회귀 결과는 float이므로 round 처리 (소수점 2자리까지)
        return jsonify({"prediction": int(round(prediction[0]))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 개발 환경에서는 0.0.0.0으로 열어두기
    app.run(host="0.0.0.0", port=5000)
