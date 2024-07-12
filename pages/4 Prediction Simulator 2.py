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
from explainerdashboard.dashboard_components import *
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
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
    
    st.write("## Prediction Simulator")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("To get a better understanding of the model's decision process, it is necessary to understand both how changing that feature impacts the model’s output, and also the distribution of that feature’s values.")
    st.write("This page allows you to explore the model's predictions by changing the feature values.")

    # st.write("This page resembles 'What if... ?' It shows how the model prediction would change based on a change in the attribute values")
    st.write("\n\n")
    st.write("### What if... ?")
    st.write("Select a sample from the sidebar to view and modify the feature values. The model prediction will be updated accordingly.")
    st.write("The new sample values are displayed below, alongwith the change from the original sample values.")
    
    model = load_model()
    model = model.fit(X_train, y_train)
    # importances = model.feature_importances_
    # print(importances)  
    
    if 'explainer' not in st.session_state:
        explainer = ClassifierExplainer(model, X_test, y_test)
        st.session_state.explainer = explainer
        explainer.dump("./explainer.joblib")
    else:
        explainer = ClassifierExplainer.from_file("./explainer.joblib")
    
    # what_if_component = ExplainerDashboard(explainer, 
    #                                         importances=False,
    #                                         model_summary=False,
    #                                         contributions=False,
    #                                         whatif=True,
    #                                         shap_dependence=False,
    #                                         shap_interaction=False,
    #                                         decision_trees=False, 
    #                                         hide_whatifpdp=True,
    #                                         index=0
    #                                     )
    # what_if_html = what_if_component.to_html()
    # st.components.v1.html(what_if_html, width=800, height=2000, scrolling=False)
    
    index = st.sidebar.selectbox("Select a `mother_id` to view and modify", options=range(len(X_test)))
    st.write(f"Selected *mother_id*: {index}")
    
    sample_df = df_og.loc[[X_test.index[index]]]
    sample_df = sample_df.to_frame().T if isinstance(sample_df, pd.Series) else sample_df

    X_test_mod = X_test.copy()
    
    sample = X_test.iloc[index]
    sample_index = X_test.index[index]    
    sample_df = df_og.loc[[X_test.index[index]]]
    
    # Create sliders for each feature to modify the values
    st.sidebar.write("Change feature values here:")
    new_values = {}
    for col in sample.index:
        new_values[col] = st.sidebar.slider(col, int(X[col].min()), int(X[col].max()), int(sample[col]))
    
    # Create a DataFrame with the new values
    new_sample = pd.DataFrame([new_values])
    
    # Display the new feature values
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(label="**Age**", value=new_sample['Age'][0], delta=new_sample['Age'][0] - sample['Age'])
    with c2:
        st.metric(label="**SystolicBP**", value=new_sample['SystolicBP'][0], delta=new_sample['SystolicBP'][0] - sample['SystolicBP'])
    with c3:
        st.metric(label="**DiastolicBP**", value=new_sample['DiastolicBP'][0], delta=new_sample['DiastolicBP'][0] - sample['DiastolicBP'])
    with c4:
        st.metric(label="**BS**", value=new_sample['BS'][0], delta=new_sample['BS'][0] - sample['BS'])
    with c5:
        st.metric(label="**BodyTemp**", value=new_sample['BodyTemp'][0], delta=new_sample['BodyTemp'][0] - sample['BodyTemp'])
    with c6:
        st.metric(label="**HeartRate**", value=new_sample['HeartRate'][0], delta=new_sample['HeartRate'][0] - sample['HeartRate'])
        
    st.write("\n\n\n\n")
    
    col1, col2 = st.columns(2)
    with col1:
        explainer = ClassifierExplainer(model, X_test, y_test)
        prediction_component = ClassifierPredictionSummaryComponent(explainer, title="Original Prediction", index=index, hide_selector=True)
        prediction_component_html = prediction_component.to_html()
        st.components.v1.html(prediction_component_html, height=560, scrolling=False)
        string = ""
        for i, label in enumerate(label_encoder.classes_):
            string += f"{i}: {label},  ‎ " 
        st.write(f"‎ ‎ ‎ ‎ ‎ {string[:-3]}")
        
    with col2: 
        X_test_mod.loc[sample_index] = new_sample.loc[0]
        
        explainer = ClassifierExplainer(model, X_test_mod, y_test)
        prediction_component = ClassifierPredictionSummaryComponent(explainer, title="New Prediction", index=index, hide_selector=True)
        prediction_component_html = prediction_component.to_html()
        st.components.v1.html(prediction_component_html, height=560, scrolling=False)
        string = ""
        for i, label in enumerate(label_encoder.classes_):
            string += f"{i}: {label},  ‎ " 
        st.write(f"‎ ‎ ‎ ‎ ‎ {string[:-3]}")
    
    st.toast('Simulator loaded', icon="✔️")
    st.write("One can see how the output class probabilities change as the feature values are modified.")
    
    st.write("\n\n")
    st.write("### Key Findings")
    st.write("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.")
    
    st.balloons()
        
    
if __name__ == "__main__":
    main()
