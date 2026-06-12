import streamlit as st

st.set_page_config(page_title="AI Stock Analyzer")

st.title("📈 AI Stock Analyzer")

ticker = st.text_input("Enter ticker", value="VTI")

if st.button("Analyze"):
    st.success(f"Analyzing {ticker}")
