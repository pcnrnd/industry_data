import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import networkx as nx
# 노드 추가 함수 (구현 필요)
def add_node(node_name, color):
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()
        st.session_state.node_colors = {}
    st.session_state.graph.add_node(node_name)
    st.session_state.node_colors[node_name] = color

# 엣지 추가 함수 (구현 필요)
def add_edge(node1, node2):
    if 'graph' in st.session_state and node1 and node2:
        st.session_state.graph.add_edge(node1, node2)

# 그래프 시각화 함수 (구현 필요)
def draw_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    
    # 노드 색상 설정
    colors = [st.session_state.node_colors[node] for node in graph.nodes()]
    
    nx.draw(graph, pos, with_labels=True, node_color=colors, node_size=2000, font_size=16, font_color='black', font_weight='bold')
    plt.axis('off')
    st.pyplot(plt)

# 데이터프레임 생성 함수
def create_timeline_dataframe(start_date, end_date):
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    data = {
        'Timeline': [f'Timeline {i + 1}' for i in range(10)],
        'Start Date': [dates[i % num_days] for i in range(10)],
        'End Date': [dates[(i % num_days) + 1] if (i % num_days) < (num_days - 1) else dates[-1] + timedelta(days=1) for i in range(10)],
        'Status': ['Not Started' for _ in range(10)],
        'Color': ['#E5EDF9' for _ in range(10)]
    }

    return pd.DataFrame(data)

# 타임라인 시각화 함수
def draw_timeline(df):
    fig, ax = plt.subplots(figsize=(10, 2))

    for index, row in df.iterrows():
        start_date = row['Start Date']
        end_date = row['End Date']
        timeline_name = row['Timeline']
        
        square_size = 20
        ax.plot(start_date, 1, marker='s', markersize=square_size, color=row['Color'], label=timeline_name)
        ax.plot(end_date, 1, marker='s', markersize=square_size, color='red')
        ax.text(start_date, 1.02, timeline_name, ha='center', fontsize=8)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: (datetime.fromtimestamp(x).strftime('%Y-%m-%d'))))
    plt.xticks(rotation=45, ha='right')
    plt.title('Timeline of Nodes')
    plt.ylim(0.9, 1.2)
    plt.axis('off')
    st.pyplot(fig)