import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from joblib import load
import plotly.graph_objects as go
import plotly.io as pio
import time
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.dashboard_components import ImportancesComponent, ShapContributionsTableComponent, ShapContributionsGraphComponent
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

# Train Logistic Regression model
@st.cache_resource
def load_model():
    model = load("./random_forest_model.pkl")
    return model
    
    
def main():
    df_og, target = load_data()
    label_encoder = LabelEncoder()
    df = df_og.copy()
    df[target] = label_encoder.fit_transform(df[target])
    X = df.drop(target, axis=1)
    y = df[target]
    
    if 'X_train' not in st.session_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
    
    st.write("## Indiviual Predictions")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("\n\n")
    st.write("### Feature Importances")
    st.write("Using [ExplainerDashboard](https://github.com/oegedijk/explainerdashboard) for our model, we can see the feature importances.")
    st.write(f"**Model**: Random Forest (Trained on *{len(X_train)}* samples and validated on *{len(X_test)}* samples.)")
    
    model = load_model()
    model = model.fit(X_train, y_train)
    # importances = model.feature_importances_
    # print(importances)  
    
    with st.container():
        st.write("\n\n")
        with st.spinner(text='Loading Explainer...'):
            if 'explainer' not in st.session_state:
                explainer = ClassifierExplainer(model, X_test, y_test)
                st.session_state.explainer = explainer
                explainer.dump("./explainer.joblib")
            else:
                explainer = ClassifierExplainer.from_file("./explainer.joblib")
            
            importances_component = ImportancesComponent(explainer, hide_title=True)
            importances_html = importances_component.to_html()
            st.components.v1.html(importances_html, height=440, width=800, scrolling=False)
       
    st.toast('Explainer loaded', icon="✔️")
    
    st.write("### Contributions for a single point")
    st.write("To see the contributions for a single point, select a sample from the sidebar.")
    
    # Index selector for SHAP contributions
    index = st.sidebar.selectbox("Select an index to view contributions", options=range(len(X_test)))
    st.write(f"Selected index: {index}")
    st.write(f"Predicted class: {model.predict(X_test.iloc[[index]])[0]} ({label_encoder.classes_[model.predict(X_test.iloc[[index]])[0]]})")
    
    sample_df = df_og.loc[[X_test.index[index]]]
    sample_df = sample_df.to_frame().T if isinstance(sample_df, pd.Series) else sample_df

    col1, col2 = st.columns(2)
    
    with col1:
        # Contributions Table Component
        st.write("\n\n")
        st.dataframe(sample_df, hide_index=True)
        st.write("\n\n")
        contributions_table_component = ShapContributionsTableComponent(explainer, title="Contributions Table", index=index)
        contributions_table_html = contributions_table_component.to_html()
        st.components.v1.html(contributions_table_html, height=600, scrolling=False)
        
    with col2:
        # Contributions Graph Component
        st.write("\n\n")
        contributions_graph_component = ShapContributionsGraphComponent(explainer, title="Contributions Plot", index=index)
        contributions_graph_html = contributions_graph_component.to_html()
        st.components.v1.html(contributions_graph_html, height=900, width=500, scrolling=False)
    
    st.balloons()
        
    
if __name__ == "__main__":
    main()
