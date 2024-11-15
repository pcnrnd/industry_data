# 분류모델, 이상탐지 모델 
# uvicorn main:app --reload
# uvicorn main:app --reload --port 8001 / 현재 streamlit 과 8001 port로 연결되어 있음
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from io import StringIO
from lib.models.classification import Classification, compare_clf_models
from lib.models.regression import RegressionModels, compare_reg_models

from lib.prepro import Preprocessing
import pandas as pd
import json

app = FastAPI()

@app.get('/') # server test
def main():
    return 'main'


@app.post('/new_clf')
async def new_clf(request: Request):
    data = await request.json()
    df = pd.read_json(StringIO(data['json_data']))
    target = data['target']
    reg_compare = compare_clf_models(df, target, n_trials=5)
    dumps_data = json.dumps(reg_compare)
    print('result_data: ', dumps_data)
    return JSONResponse(content={'result': dumps_data})


@app.post('/new_reg')
async def new_reg(request: Request):
    data = await request.json()
    df = pd.read_json(StringIO(data['json_data']))
    target = data['target']
    reg_compare = compare_reg_models(df, target, n_trials=5)
    dumps_data = json.dumps(reg_compare)
    print('result_data: ', dumps_data)
    return JSONResponse(content={'result': dumps_data})
