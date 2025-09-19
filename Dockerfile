# Dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델과 코드 복사
COPY . .

# Flask 서버 실행
CMD ["python", "app.py"]
