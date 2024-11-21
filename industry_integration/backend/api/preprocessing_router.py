from fastapi import APIRouter, Request

router = APIRouter()

@router.post('/add_file_data')
def add_file_data(request: Request):
    '''
    add unstructured data psths
    '''
    data = request.json()
    table_name = data['table_name']
    abs_path_of_file = data['abs_path']
    
    from services.dataframe import PolarsDataFrame
    import duckdb
    pf = PolarsDataFrame()
    paths = pf.get_all_file_paths(abs_path_of_file) # 모든 파일 데이터 경로 추출
    df = pf.make_polars_dataframe(paths) # 추출한 경로 데이터프레임 생성
    con = duckdb.connect('../databases/file_catalog.db')
    con.execute(f'CREATE TABLE {table_name} AS SELECT FROM {df}')

    return {'message': '데이터 저장 완료!'}

@router.get('/read_file_data')
def read_file_data(request: Request):
    '''
    read unstructured data table
    '''
    data = request.json()
    table_name = data['table_name']
    
    import duckdb
    con = duckdb.connect('../databases/file_catalog.db')
    query_result = con.execute(f'SELECT * FROM {table_name}').fetchall()
    return {'query_result': query_result}