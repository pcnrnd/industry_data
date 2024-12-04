import streamlit as st
from PIL import Image, ImageOps
from io import BytesIO
from lib.prepro import Preprocessing
import plotly.express as px

st.set_page_config(layout='wide')

prepro = Preprocessing()

# 이미지 변환 및 다운로드 함수
def get_image_download_buffer(image, format="PNG"):
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer

def main():
    st.title("이미지 데이터 전처리")
    st.sidebar.header("전처리 옵션")

    # 이미지 업로드
    uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    # 원본 이미지
    original_image = Image.open(uploaded_file)

    # 전처리 옵션
    option_rotate = st.sidebar.slider("이미지 회전 (도)", -180, 180, 0)
    option_resize = st.sidebar.checkbox("이미지 크기 조정")
    option_grayscale = st.sidebar.checkbox("흑백 변환")

    # 전처리 시작
    processed_image = original_image.copy()


if __name__ == "__main__":
    main()
