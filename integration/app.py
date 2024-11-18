# 실행방법: streamlit run app.py

import streamlit as st
import os
import sys

sys.path.append(os.path.abspath('./pages/main/'))
sys.path.append(os.path.abspath('./pages/recommendation/'))
sys.path.append(os.path.abspath('./pages/augmentation/'))

pages = {
    "Home": [
        st.Page('./pages/main/main.py', title='main')
    ],
    "Model": [
        st.Page('./pages/recommendation/recomendation.py', title='Model Recommendation'),
    ],
    'Data': [
        st.Page('./pages/augmentation/gene.py', title='Data Augmentation')
    ]
}

pg = st.navigation(pages)
pg.run()
