# 실행방법: streamlit run app.py

import streamlit as st
import os
import sys

sys.path.append(os.path.abspath('./main/'))
sys.path.append(os.path.abspath('./recommendation/'))
sys.path.append(os.path.abspath('./augmentation/'))

pages = {
    "Home": [
        st.Page('./main/main.py', title='main')
    ],
    "Model": [
        st.Page('./recommendation/recomendation.py', title='Model Recommendation'),
    ],
    'Data': [
        st.Page('./augmentation/gene.py', title='Data Augmentation')
    ]
}

pg = st.navigation(pages)
pg.run()
