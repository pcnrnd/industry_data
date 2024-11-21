from fastapi import APIRouter, Request
from services.dataframe import PolarsDataFrame
import duckdb

router = APIRouter()

@router.post()
def add_file_path(request: Request):
    data = request.json()
    table_name = data['table_name']
    abs_path = data['abs_path']
    
    pf = PolarsDataFrame()
    paths = pf.get_all_file_paths(abs_path)
    df = pf.make_polars_dataframe(paths)
    con = duckdb.connect('../databases/file_catalog.db')
    con.execute(f'CREATE TABLE {table_name} AS SELECT FROM {df}')
    return None
