# 실행방법: streamlit run app.py

import streamlit as st
import os
import sys

sys.path.append(os.path.abspath('./dashboard/main/'))
sys.path.append(os.path.abspath('./recommendation/'))
sys.path.append(os.path.abspath('./augmentation/'))
sys.path.append(os.path.abspath('./preprocessing/'))

pages = {
    "Home": [
        st.Page('./dashboard/main.py', title='main')
    ],
    "Model": [
        st.Page('./recommendation/recomendation.py', title='Model Recommendation'),
    ],
    'Data': [
        st.Page('./augmentation/gene.py', title='Data Augmentation'),
        st.Page('./preprocessing/preprocessing.py', title='Data preprocessing')
    ]
}

pg = st.navigation(pages)
pg.run()
