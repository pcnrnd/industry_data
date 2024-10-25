# 대시보드 UI
# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run recomendation.py
# 실행방법2: streamlit run recomendation.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# import time # 코드 실행 시간 측정 시 사용
# sqlite:///./database/database.db
import ray
import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.sidebar.title("Details")

# 분류, 이상 탐지 등 추천받을 머신러닝 모델 선택
option = st.sidebar.selectbox(
    'Select Machine Learning Task', ('분류', '이상탐지'))
uploaded_file = None
uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드

# @st.cache_data # resource
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

with st.spinner('Wait for it...'):
    if uploaded_file is None: 
        st.write(
        '''
        ### 머신러닝 실행 방법
        * 분류
        1. Upload csv file 
        2. Select Target column 
        3. Drop cloumns
        4. 제거할 Target 데이터 선택

        * 이상 탐지
        1. Upload csv file
        2. 머신러닝 테스트 실행
        ''')
        
    updated_df = None
    # Uploaded data Dashboard
    if uploaded_file is not None:
        st.subheader('데이터 분석')
        df = load_data(uploaded_file) 
        col_list = df.columns.tolist() # 데이터 전처리 옵션 설정 리스트
        target_feture = ""
        if option == '분류':
            target_feture = st.sidebar.multiselect('Select Target Column', options=col_list)

        data_to_drop = st.sidebar.multiselect('Drop Cloumns', options=col_list)

        tab_eda_df, tab_eda_info, tab_Label_counts = st.tabs(['Original data', 'Null information', 'Target Data Counts']) # tab_Label_counts Labels counts
        with tab_eda_df:
            st.write('Original data')
            st.dataframe(df)
        with tab_eda_info:
            st.write('Null information')
            info_df = pd.DataFrame({'Column names': df.columns,
                                    'Non-Null Count': df.count(),
                                    'Null Count': df.isnull().sum(),
                                    'Dtype': df.dtypes,
                                    })
            info_df.reset_index(inplace=True)
            st.write(info_df.iloc[:, 1:].astype(str))
