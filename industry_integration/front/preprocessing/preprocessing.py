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


    path = 'G:\industry_data\sand_data\data\Training\original_data\TS_1.모래입자크기분류_1'
    path_data = prepro.get_all_file_paths(path)
    file_df = prepro.make_polars_dataframe(path_data)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image File Size Distribution")
            fig = px.histogram(file_df['file_size'], nbins=100, title="File Size Distribution") # , labels={"value": "File Size (Bytes)"}
            st.plotly_chart(fig)
        with col2:
            st.header("Image Resolution Distribution")
            fig = px.scatter(x=file_df['image_width'], y=file_df['image_height'], labels={"x": "Width (pixels)", "y": "Height (pixels)"}, title="Image Resolutions")
            st.plotly_chart(fig)

    # 좌우 레이아웃 설정
    col1, col2 = st.columns(2)
    with st.container(border=True):
        with col1:
            st.subheader("원본 이미지")
            st.image(original_image, use_column_width=True, caption="원본 이미지")
            
    with st.container(border=True):
        with col2:
            st.subheader("전처리된 이미지")
            st.image(processed_image, use_column_width=True, caption="전처리된 이미지")

    st.subheader('메타데이터(Metadata)')
    st.dataframe(file_df, use_container_width=True)


    # 다운로드
    st.sidebar.markdown("### 처리된 이미지 다운로드")
    output_format = st.sidebar.selectbox("저장 파일 형식", ["PNG", "JPEG"])
    st.sidebar.download_button(
        label="이미지 다운로드",
        data=get_image_download_buffer(processed_image, format=output_format),
        file_name=f"processed_image.{output_format.lower()}",
        mime=f"image/{output_format.lower()}"
    )

    st.sidebar.button(
        '전체 데이터 적용'
    )

if __name__ == "__main__":
    main()
