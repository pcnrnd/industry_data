from fastapi import APIRouter, Request
from services.dataframe import PolarsDataFrame
import duckdb

router = APIRouter()

@router.post('/add_file_data')
async def add_file_data(request: Request):
    '''
    add unstructured data psths
    '''
    data = await request.json()
    abs_path_of_file = data['abs_path']
    table_name = data['table_name']


    pf = PolarsDataFrame()
    paths = pf.get_all_file_paths(abs_path_of_file) # 모든 파일 데이터 경로 추출
    df = pf.make_polars_dataframe(paths) # 추출한 경로 데이터프레임 생성
    
    # con = duckdb.connect(database=':default:')
    con = duckdb.connect(database='/databases/database.db')
    con.execute(f'DROP TABLE IF EXISTS {table_name}')
    con.execute(f'CREATE TABLE {table_name} AS SELECT * FROM df')

    return {'message': '데이터 저장 완료!'}

@router.get('/show_tables')
async def show_tables():
    # con = duckdb.connect(database=':default:')
    con = duckdb.connect(database='/databases/database.db')
    con.execute('DROP TABLE IF EXISTS sand_data')
    con.execute('CREATE TABLE sand_data AS SELECT * FROM df')

    # con = duckdb.connect(database=':dafault:')
    tables = con.execute('SHOW TABLES').fetchall()
    return {'tables': tables}
    
@router.get('/read_file_data')
async def read_file_data(request: Request):
    '''
    read unstructured data table
    '''
    data = await request.json()
    table_name = data['table_name']
    
    # con = duckdb.connect(database=':default:')
    con = duckdb.connect(database='/databases/database.db')
    query_result = con.execute(f'SELECT * FROM {table_name}').fetchall()

    return {'query_result': query_result}