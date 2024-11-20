import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx


# 노드 추가 함수
def add_node(node_name, color):
    st.session_state.graph.add_node(node_name)
    st.session_state.node_colors[node_name] = color

# 엣지 추가 함수
def add_edge(node1, node2):
    if node1 and node2:
        st.session_state.graph.add_edge(node1, node2)

# 그래프 시각화
def draw_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    
    # 노드 색상 설정
    colors = [st.session_state.node_colors[node] for node in graph.nodes()]
    
    nx.draw(graph, pos, with_labels=True, node_color=colors, node_size=2000, font_size=16, font_color='black', font_weight='bold')
    plt.axis('off')
    st.pyplot(plt)