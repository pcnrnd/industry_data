# fastapi 서버 실행 명령어: uvicorn main:app --reload --port 8009 (port 번호 변경 시, gene.py에 있는 port 번호도 함께 수정)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import json

app = FastAPI()
