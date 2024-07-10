import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)


# Set page configuration
st.set_page_config(
    #page_title="Maternal Health Risk Dataset",
    layout="centered", #centered #wide
    initial_sidebar_state="expanded",
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

def main():
    # Page Title
    st.markdown("## About the Dataset")
    st.title("Maternal Health Risk Dataset")
    st.write("\n\n\n")

    st.warning('''If the mother presents a significant symptom not considered
    by the model (e.g., stroke symptoms), disregard the model's prediction and
    base the urgency purely on medical judgment.''', icon="⚠️")

    st.write("""Data was collected from five hospitals and one maternity clinic in Dhaka, Bangladesh.
    Patient health data was collected using wearable sensor devices, and risk levels were classified with the help of medical experts and literature review.
    """)

    # Add DOI link to sidebar
    # st.write(
    #     """
    #     **DOI**: [10.24432/C5DP5D](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
    #     """
    # )
    st.write("[![10.24432/C5DP5D](https://img.shields.io/badge/doi-10.3389%2Ffcomp.2021.734559-blue)](https://doi.org/10.24432/C5DP5D)")
    
    st.write("**Author**: Marzia Ahmed")
    st.write("**Dataset donated on**: 14.08.2023")

    
    # Load Dataset
    df, target = load_data()

    # Sidebar Info
    categorical = ["Age"]
    attributes = df.columns.tolist()
    st.sidebar.header("Selection Options")
    selected_attribute = st.sidebar.selectbox('Select an attribute', attributes)

    # Initialize filtered_df based on the data type of selected_attribute
    st.sidebar.header("Filter Options")

    if df[selected_attribute].dtype == 'object':
        unique_values = df[selected_attribute].unique().tolist()
        selected_values = st.sidebar.multiselect(f'Select {selected_attribute} values', unique_values, default=unique_values)
        filtered_df = df[df[selected_attribute].isin(selected_values)]
    else:
        min_value, max_value = st.sidebar.slider(
            f'Filter {selected_attribute} values',
            int(df[selected_attribute].min()),
            int(df[selected_attribute].max()),
            (int(df[selected_attribute].min()), int(df[selected_attribute].max()))
        )
        filtered_df = df[(df[selected_attribute] >= min_value) & (df[selected_attribute] <= max_value)]

    # Display dataset
    st.markdown("### The Dataset")
    st.write("""Below you can find the dataset used in this application.
    You can filter the data using the Selection & Filter Options on the sidebar.""")
    st.dataframe(filtered_df)


    # Display Attribute Distribution
    st.write("\n\n\n")
    st.write("### Attribute distribution")
    st.write("In the sidebar, select the attribute for which you want to see the distribution.")
    st.write("You can also filter the data based on the attribute values.")
    st.write("The graph shows the distribution of the selected filtered attribute collored by the risk classification level.")

    # Plotting
    custom_palette = {'low risk': '#00B38A', 'high risk': '#EA324C', 'mid risk': '#F2AC42'}

    # Map risk levels to colors
    color_discrete_map = {'low risk': custom_palette['low risk'], 'mid risk': custom_palette['mid risk'], 'high risk': custom_palette['high risk']}

    # Plotting with Plotly
    fig = px.histogram(
        filtered_df, 
        x=selected_attribute, 
        color='RiskLevel', 
        title=f"{selected_attribute} by Risk Classification",
        category_orders={"RiskLevel": ['low risk', 'mid risk', 'high risk']}, 
        barmode='stack',
        height=300, 
        width=600,
        color_discrete_map=color_discrete_map
    )   

    fig.update_traces(marker_line=dict(width=1, color='white'))

    st.plotly_chart(fig)
    
    
    # Create a DataFrame with the given information
    info = {
        "Attribute Name": ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"],
        "Role": ["Feature", "Feature", "Feature", "Feature", "Feature", "Feature", "Target"],
        "Type": ["Integer", "Integer", "Integer", "Integer", "Integer", "Integer", "Categorical"],
        "Description": [
            "Any ages in years when a women during pregnant.",
            "Upper value of Blood Pressure in mmHg, another significant attribute during pregnancy.",
            "Lower value of Blood Pressure in mmHg, another significant attribute during pregnancy.",
            "Blood glucose levels is in terms of a molar concentration (mmol/L)",
            "Body Temperature (F)",
            "A normal resting heart rate (bpm)",
            "Predicted Risk Intensity Level during pregnancy considering the previous attribute."
        ],
        "Missing Values": ["no", "no", "no", "no", "no", "no", "no"]
    }

    df_info = pd.DataFrame(info)

    # Display the table in Streamlit
    st.markdown("### Attribute Information")
    st.write("""
    This table provides detailed information about each variable in the dataset. 

    - **Attribute Name**: The name of the variable.
    - **Role**: Whether the variable is a feature (used for prediction) or the target (the outcome we are predicting).
    - **Type**: The data type of the variable (e.g., Integer, Categorical).
    - **Description**: A brief explanation of what the variable represents.
    - **Missing Values**: Indicates if there are any missing values for the variable.
    """)
    st.dataframe(df_info, hide_index=True)

    

if __name__ == "__main__":
    main()
