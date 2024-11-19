# fastapi 서버 실행 명령어: uvicorn main:app --reload --port 8009 (port 번호 변경 시, gene.py에 있는 port 번호도 함께 수정)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.augmentation_router import router as augmentation_router
from api.recommendation_router import router as recomendation_router

app = FastAPI()

app.include_router(augmentation_router, prefix="/augmentation", tags=['augmentation'])
app.include_router(recomendation_router, prefix="/recommendation", tags=['recommendation'])

app.get('.')
def main():
    return "connection success!"