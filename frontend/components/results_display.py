import streamlit as st

def display_results(response, response_time):
    st.write(f"Response: {response}")
    st.write(f"Response time: {response_time:.2f} seconds")
