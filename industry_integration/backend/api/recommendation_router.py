from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from io import StringIO
from services.classification import Classification, compare_clf_models
from services.regression import RegressionModels, compare_reg_models

from services.prepro import Preprocessing
import pandas as pd
import json

router = APIRouter()
@router.post('/new_clf')
async def new_clf(request: Request):
    data = await request.json()
    df = pd.read_json(StringIO(data['json_data']))
    target = data['target']

    # if not ray.is_initialized():
    #     ray.init()
    # clf_compare = Classification(X=df, y=df[target]).train_clf()
    # clf_models_data = ClassificationModels(df, target, n_trials=10).run_clf_models()
    # prepro_data = Preprocessing().make_dict(clf_models_data)
    # ray.shutdown()
    # dumps_data = json.dumps(prepro_data)
    # result_data = json.loads(dumps_data)
    ###
    reg_compare = compare_clf_models(df, target, n_trials=5)
    dumps_data = json.dumps(reg_compare)
    print('result_data: ', dumps_data)
    # with open('data.json', 'w') as json_file:
    #     json.dump(dumps_data, json_file, indent=4)

    return JSONResponse(content={'result': dumps_data})

@router.post('/new_reg')
async def new_reg(request: Request):
    data = await request.json()
    df = pd.read_json(StringIO(data['json_data']))
    target = data['target']
    # reg_compare = RegressionModels(df, target, n_trials=10).run_reg_models()
    reg_compare = compare_reg_models(df, target, n_trials=5)
    dumps_data = json.dumps(reg_compare)
    print('result_data: ', dumps_data)
    return JSONResponse(content={'result': dumps_data})
