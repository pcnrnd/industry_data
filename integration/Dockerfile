# 베이스 이미지로 Python을 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI와 Streamlit의 포트를 오픈
EXPOSE 8000 8501

# 환경변수 설정 (FastAPI와 Streamlit 환경 구분)
ENV FASTAPI_HOST="0.0.0.0"
ENV FASTAPI_PORT="8000"
ENV STREAMLIT_HOST="0.0.0.0"
ENV STREAMLIT_PORT="8501"

# FastAPI 서버와 Streamlit 서버를 동시에 실행할 수 있도록 설정
# 쉘 명령어로 FastAPI와 Streamlit을 백그라운드에서 실행하는 명령어 작성
CMD ["sh", "-c", "uvicorn backend.main:app --host $FASTAPI_HOST --port $FASTAPI_PORT & streamlit run front/app.py --server.address $STREAMLIT_HOST --server.port $STREAMLIT_PORT"]
