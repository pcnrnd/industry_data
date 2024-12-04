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

    if option_rotate != 0:
        processed_image = processed_image.rotate(option_rotate)
    
    if option_resize:
        new_width = st.sidebar.number_input("가로 크기", value=processed_image.size[0], min_value=1)
        new_height = st.sidebar.number_input("세로 크기", value=processed_image.size[1], min_value=1)
        processed_image = processed_image.resize((new_width, new_height))
    
    if option_grayscale:
        processed_image = ImageOps.grayscale(processed_image)

    if st.sidebar.checkbox("색상 반전 적용"):
        processed_image = ImageOps.invert(processed_image)

    if st.sidebar.checkbox("패딩 추가"):
        padding_size = st.sidebar.slider("패딩 크기", min_value=0, max_value=100, value=10)
        padding_color = st.sidebar.color_picker("패딩 색상", value="#000000")
        processed_image = ImageOps.expand(processed_image, border=padding_size, fill=padding_color)

    if st.sidebar.checkbox("솔라라이즈 효과 적용"):
        solarize_threshold = st.sidebar.slider("솔라라이즈 임계값", min_value=0, max_value=255, value=128)
        processed_image = ImageOps.solarize(processed_image, threshold=solarize_threshold)

    if st.sidebar.checkbox("비율 유지 크기 조정 (Fit)"):
        fit_width = st.sidebar.number_input("Fit 가로 크기", value=processed_image.size[0], min_value=1)
        fit_height = st.sidebar.number_input("Fit 세로 크기", value=processed_image.size[1], min_value=1)
        processed_image = ImageOps.fit(processed_image, (fit_width, fit_height), method=Image.BICUBIC)

    if st.sidebar.checkbox("상하 반전 (Flip)"):
        processed_image = ImageOps.flip(processed_image)

    if st.sidebar.checkbox("좌우 반전 (Mirror)"):
        processed_image = ImageOps.mirror(processed_image)


if __name__ == "__main__":
    main()
