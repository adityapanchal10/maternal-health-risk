import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="HealthyMom App",
    layout="centered", #centered #wide
    initial_sidebar_state="auto", #collapsed
)

# Main page title and welcome message
st.title("Welcome to HealthyMom App")

st.logo(
    "./love.png",
    icon_image="./heartbeat.gif",
)

st.markdown("""
The **HealthyMom App** is designed to support healthcare professionals in the triage process of maternity hospitals.
By providing predictions of maternal health risks, our app aims to contribute to Maternal Health,
ensuring that expectant mothers receive the appropriate level of care quickly and efficiently.
""")

# Display image on Home
st.image("./MomArt.jpeg", use_column_width=True)

st.sidebar.header("\n")
st.sidebar.subheader("App Developers:", divider="red")
st.sidebar.write("Aditya Panchal")
st.sidebar.markdown("Mariana Steffens")
st.sidebar.markdown("Navya Reddy Tiyyagura")
st.sidebar.markdown("Se Yeon Kim")
st.sidebar.markdown("##### :gray[App developed as part of the Human Centered Data Science course during SoSe-24 at the Freie Universität Berlin]")
st.sidebar.markdown("##### :gray[16.07.2024]")