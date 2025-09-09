import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path="application_train.csv"):
    df = pd.read_csv(file_path)
    return df