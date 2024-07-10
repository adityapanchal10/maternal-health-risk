import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set page configuration
st.set_page_config(
    page_title="HealthyMom App",
    layout="centered", #centered #wide
    initial_sidebar_state="auto", #collapsed
)


# Load the Data set
df = pd.read_csv("./Maternal Health Risk Data Set.csv")

# st.markdown("## Home")

# Main page title and welcome message
st.title("Welcome to HealthyMom App")

st.markdown("""
The **HealthyMom App** is designed to support healthcare professionals in the triage process of maternity hospitals.
By providing predictions of maternal health risks, our app aims to contribute to Maternal Health,
ensuring that expectant mothers receive the appropriate level of care quickly and efficiently.
""")

# Display image on Home
st.image("./MomArt.jpeg", use_column_width=True)

# Sidebar Content
#st.sidebar.markdown("---")
st.sidebar.markdown("### App Developers:")
st.sidebar.write("Aditya Panchal")
st.sidebar.markdown("Mariana Steffens")
st.sidebar.markdown("Navya Reddy Tiyyagura")
st.sidebar.markdown("Se Yeon Kim")
#st.sidebar.markdown("---")
st.sidebar.markdown("##### App developed as part of the Human Centered Data Science course at the Freie Universität Berlin")
st.sidebar.markdown("##### 02.07.2024")

