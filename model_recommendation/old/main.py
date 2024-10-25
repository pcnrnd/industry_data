# 분류모델, 이상탐지 모델 
# uvicorn main:app --reload
# uvicorn main:app --reload --port 8001
from fastapi import FastAPI, Request
from lib.anomaly_lib import *
from lib.pycaret_clf import *
# from pycaret.anomaly import AnomalyExperiment
# from pycaret.classification import ClassificationExperiment
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import ray

app = FastAPI()

@app.get('/') # server test
def main():
    return 'main'

@app.post('/clf/')
async def clf_test(request: Request): # dict
    data = await request.json() # 데이터를 비동기 방식으로 load
    # df = request['json_data']
    df = pd.read_json(data['json_data'])
    target = data['target']

    # ray를 활용한 머신러닝 분산 학습
    if not ray.is_initialized(): 
        ray.init()
    # model_compare_clf = train_caret_clf(df, df[target])
    model_compare_task = train_caret_clf.remote(df, df[target]) # 분류 모델 객체 선언
    model_compare_clf = ray.get(model_compare_task) # 분류 모델 분산 학습
    ray.shutdown()  # 머신러닝 모델 분산 학습 종료
    df_json = model_compare_clf.to_json() # 학습결과를 json으로 변환
    result_data = json.loads(df_json) # json을 파이썬 객체로 변한
    return JSONResponse(content={'result': result_data})