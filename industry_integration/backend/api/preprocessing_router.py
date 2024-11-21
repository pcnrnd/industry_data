from fastapi import APIRouter, Request

router = APIRouter()

@router.post()
def add_file_path(request: Request):
    data = request.json()
    table_name = data['table_name']
    abs_path = data['abs_path']
    
    from services.dataframe import PolarsDataFrame
    import duckdb
    pf = PolarsDataFrame()
    paths = pf.get_all_file_paths(abs_path) # 모든 파일 데이터 경로 추출
    df = pf.make_polars_dataframe(paths) # 추출한 경로 데이터프레임 생성
    con = duckdb.connect('../databases/file_catalog.db')
    con.execute(f'CREATE TABLE {table_name} AS SELECT FROM {df}')

    return {'message': '데이터 저장 완료!'}
