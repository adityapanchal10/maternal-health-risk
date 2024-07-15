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
import plotly.graph_objects as go
import plotly.io as pio

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

def create_pie_chart(predictions, title):
    labels = ['High Risk', 'Low Risk', 'Mid Risk']

    color_map = {
        0: "#EA324C",
        1: "#00B38A",
        2: "#F2AC42"
    }
    colors = [color_map[i] for i in range(len(labels))]

    fig = go.Figure(data=[go.Pie(labels=labels, values=predictions, marker=dict(colors=colors), hole=0.4)])

    fig.update_layout(
        title=title,
        margin=dict(b=0),  # Reduce the bottom margin. This way we dont have to set the height and wodth 
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.0,
            xanchor="center",
            x=0.5
        )
    )
    return fig
    
def main():
    df, target = load_data()
    label_encoder = LabelEncoder()
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
        if y_train.dtype == object: # useful if we go to about model first and then come here
            y_train = label_encoder.transform(y_train)
            y_test = label_encoder.transform(y_test)
    
    st.header("Prediction Simulator", anchor="prediction-simulator")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    
    st.write("This page allows you to explore the model's predictions by simulating different values.")
    st.warning('''If the mother presents a significant symptom not considered by the model (e.g., stroke symptoms), disregard the model's prediction and
    base the urgency purely on medical judgment.''', icon="⚠️")
    
    st.write("\n\n")
    st.subheader("What if... ?", anchor="what-if", divider="red")
    st.write("Select a *mother_id* from the sidebar and change the values for the measurements to simulate the health risk prediction. The model prediction will be updated accordingly.")
    st.write("The new sample values are displayed below, alongwith the change from the original sample values. (they can be reset to original values by clicking the `Reset` button in the sidebar)")
    
    model = load_model()
    model = model.fit(X_train, y_train) 
    
    if 'predicted_probs' not in st.session_state:
        pred_probs = model.predict_proba(X_test)
        st.session_state.predicted_probs = pred_probs
        print(pred_probs.shape)
    else:
        pred_probs = st.session_state.predicted_probs
    
    if 'explainer' not in st.session_state:
        explainer = ClassifierExplainer(model, X_test, y_test)
        st.session_state.explainer = explainer
        explainer.dump("./explainer.joblib")
    else:
        explainer = ClassifierExplainer.from_file("./explainer.joblib")
    
    index = st.sidebar.selectbox("Select a `mother_id` to view and modify", options=range(len(X_test)))
    st.write(f"🧸 **Selected *mother_id*: {index}**")

    sample = X_test.iloc[index]
    
    # Create sliders for each feature to modify the values
    st.sidebar.write("Change attribute values here:")
    new_values = {}
    for col in sample.index:
        if col == "BS" or col == "BodyTemp":
            new_values[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(sample[col]), step=0.1)
        else:
            new_values[col] = st.sidebar.slider(col, int(X[col].min()), int(X[col].max()), int(sample[col]))
    # Add a reset button
    reset = st.sidebar.button("Reset")
    if reset:
        for col in sample.index:
            if col == "BS" or col == "BodyTemp":
                new_values[col] = sample[col]
            else:
                new_values[col] = int(sample[col])
    
    # Create a DataFrame with the new values
    new_sample = pd.DataFrame([new_values])
    
    # Display the new feature values
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(label="**Age** :gray[(y)]", value=new_sample['Age'][0], delta=new_sample['Age'][0] - sample['Age'])
    with c2:
        st.metric(label="**SystolicBP** :gray[(mmHg)]", value=new_sample['SystolicBP'][0], delta=new_sample['SystolicBP'][0] - sample['SystolicBP'])
    with c3:
        st.metric(label="**DiastolicBP** :gray[(mmHg)]", value=new_sample['DiastolicBP'][0], delta=new_sample['DiastolicBP'][0] - sample['DiastolicBP'])
    with c4:
        st.metric(label="**BS** :gray[(mmol/L)]", value=new_sample['BS'][0], delta=round(new_sample['BS'][0] - sample['BS'], 2))
    with c5:
        st.metric(label="**BodyTemp** :gray[(°F)]", value=new_sample['BodyTemp'][0], delta=round(new_sample['BodyTemp'][0] - sample['BodyTemp'], 2))
    with c6:
        st.metric(label="**HeartRate** :gray[(bpm)]", value=new_sample['HeartRate'][0], delta=new_sample['HeartRate'][0] - sample['HeartRate'])
        
    st.write("\n\n\n\n")
    
    # Traffic light colors for classes
    color_map = {
        0: {"background": "#EA324CAA", "color": "white"},
        1: {"background": "#00B38AAA", "color": "black"},
        2: {"background": "#F2AC42AA", "color": "black"}
    }
    
    col1, col2 = st.columns(2)
    with col1: 
        # Use custom pie chart instead of the prediction component
        pie_chart = create_pie_chart(pred_probs[index], "Original Prediction")
        st.plotly_chart(pie_chart)
        
        predicted_class = model.predict(X_test.iloc[[index]])[0]
        class_name = label_encoder.classes_[predicted_class]
        color = color_map.get(predicted_class, {"background": "black", "color": "white"})
        st.markdown(f"""
        <div style='background-color: {color['background']}; margin-left:20px; margin-right:2s0px; padding: 10px; border-radius: 5px; color: {color['color']};'>
            Predicted class: {class_name} (class {predicted_class})
        </div>
        """, unsafe_allow_html=True)
        
    with col2: 
        # Use custom pie chart instead of the prediction component
        if np.array_equal(X_test.iloc[[index]].values, new_sample.iloc[[0]].values):
            predicted_probs = pred_probs[index]
        else:
            predicted_probs = model.predict_proba(new_sample.iloc[[0]])[0]
        pie_chart = create_pie_chart(predicted_probs, "New Prediction")
        st.plotly_chart(pie_chart)
        
        predicted_class = model.predict(new_sample.iloc[[0]])[0]
        class_name = label_encoder.classes_[predicted_class]
        color = color_map.get(predicted_class, {"background": "black", "color": "white"})
        st.markdown(f"""
        <div style='background-color: {color['background']}; margin-left:20px; margin-right:20px; padding: 10px; border-radius: 5px; color: {color['color']};'>
            Predicted class: {class_name} (class {predicted_class})
        </div>
        """, unsafe_allow_html=True)       
    
    st.toast('Simulator ready 🕹️', icon="✔️")
    st.write("\n")
    st.write("\n")
    st.write("One can see how the output class probabilities change as the attribute values are modified.")
     
    
if __name__ == "__main__":
    main()
