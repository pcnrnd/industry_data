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


# # Initialize session state variables if they don't exist
# if 'page' not in st.session_state:
#     st.session_state.page = "Page 1"  # Default page

# # Sidebar navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select a page:", ("Page 1", "Page 2"))

# # Page routing
# if page == "Page 1":
#     run_page1()
# elif page == "Page 2":
#     run_page2()
