import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 데이터프레임 생성 함수
def create_timeline_dataframe(start_date, end_date):
    # 주어진 기간 내의 날짜 생성
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]  # 기간 내의 날짜 생성

    # 10개의 타임라인 생성
    data = {
        'Timeline': [f'Timeline {i + 1}' for i in range(10)],
        'Start Date': [dates[i % num_days] for i in range(10)],  # 각 로우에 대해 시작 날짜 설정
        'End Date': [dates[(i % num_days) + 1] if (i % num_days) < (num_days - 1) else dates[-1] + timedelta(days=1) for i in range(10)],  # 종료 날짜 설정
        'Status': ['Not Started' for _ in range(10)],  # 상태 추가
        'Color': ['#E5EDF9' for _ in range(10)]  # 색상 추가
    }

    # 데이터프레임 생성
    return pd.DataFrame(data)

# 타임라인 시각화 함수
def draw_timeline(df):
    fig, ax = plt.subplots(figsize=(10, 2))

    for index, row in df.iterrows():
        start_date = row['Start Date']
        end_date = row['End Date']
        timeline_name = row['Timeline']
        
        # 타임라인에서 작은 정사각형 그리기
        square_size = 20  # 정사각형 크기
        ax.plot(start_date, 1, marker='s', markersize=square_size, color=row['Color'], label=timeline_name)  # 시작 시각
        ax.plot(end_date, 1, marker='s', markersize=square_size, color=row['Color'])  # 종료 시각
        # ax.text(start_date, 1.02, timeline_name, ha='center', fontsize=8)

    # x축 포맷 설정
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: (datetime.fromtimestamp(x).strftime('%Y-%m-%d'))))
    plt.xticks(rotation=45, ha='right')
    plt.title('Timeline of Nodes')
    plt.ylim(0.9, 1.2)  # y축 범위 조정
    plt.axis('off')  # 축 숨기기
    st.pyplot(fig)
