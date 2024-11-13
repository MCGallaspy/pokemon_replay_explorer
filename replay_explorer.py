import streamlit as st
import pandas as pd

df = pd.read_parquet('regh_slim.parquet')
df['uploadtime'] = pd.to_datetime(df['uploadtime'], unit='s')
st.header("Data used")
st.dataframe(df)

if st.button("Crunch the numbers"):
    st.header("Results")
    