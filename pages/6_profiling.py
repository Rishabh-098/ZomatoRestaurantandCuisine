import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


df = pd.read_csv('Dataset\Combined_data.csv')
st.dataframe(df)
st.title('Automated Exploratory Data Analysis')
profile_report = df.profile_report()
st_profile_report(profile_report)

