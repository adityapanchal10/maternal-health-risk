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
        if y_train.dtype == object: # useful if we go to about model first and then come here
            y_train = label_encoder.transform(y_train)
            y_test = label_encoder.transform(y_test)
    
    st.header("Individual Prediction", anchor="individual-prediction")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("This page enables the examination of the prediction and the contribution of each feature to the prediction for an individual sample.")
        
    model = load_model()
    model = model.fit(X_train, y_train)
            
    with st.spinner(text='Loading Explainer...'):
            if 'explainer' not in st.session_state:
                explainer = ClassifierExplainer(model, X_test, y_test)
                st.session_state.explainer = explainer
                explainer.dump("./explainer.joblib")
            else:
                explainer = ClassifierExplainer.from_file("./explainer.joblib")
       
    st.toast('Explainer ready 🔬', icon="✔️")
     
    st.write("Select the *mother_id* from the sidebar to see the prediction health risk for the mother and prediction contribution.")
    st.warning('''If the mother presents a significant symptom not considered by the model (e.g., stroke symptoms), disregard the model's prediction and
    base the urgency purely on medical judgment.''', icon="⚠️")

    # Index selector for SHAP contributions
    index = st.sidebar.selectbox("Select a `mother_id` to view the prediction", options=range(len(X_test)))
    st.write(f"🧸 **Selected *mother_id*: {index}**")

    predicted_class = model.predict(X_test.iloc[[index]])[0]
    class_name = label_encoder.classes_[predicted_class]
    
    # Traffic light colors for classes
    color_map = {
        0: {"background": "#EA324CAA", "color": "white"},
        1: {"background": "#00B38AAA", "color": "black"},
        2: {"background": "#F2AC42AA", "color": "black"}
    }
    color = color_map.get(predicted_class, {"background": "black", "color": "white"})
    
    st.markdown(f"""
    <div style='background-color: {color['background']}; padding: 10px; border-radius: 5px; color: {color['color']};'>
        Predicted class: {class_name} (class {predicted_class})
    </div>
    """, unsafe_allow_html=True)

    st.write("\n\n")
    
    sample_df = df_og.loc[[X_test.index[index]]]
    sample_df = sample_df.to_frame().T if isinstance(sample_df, pd.Series) else sample_df

    # Display attribute values
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(label="**Age** :gray[(y)]", value=sample_df['Age'])
    with c2:
        st.metric(label="**SystolicBP** :gray[(mmHg)]", value=sample_df['SystolicBP'])
    with c3:
        st.metric(label="**DiastolicBP** :gray[(mmHg)]", value=sample_df['DiastolicBP'])
    with c4:
        st.metric(label="**BS** :gray[(mmol/L)]", value=sample_df['BS'])
    with c5:
        st.metric(label="**BodyTemp** :gray[(°F)]", value=sample_df['BodyTemp'])
    with c6:
        st.metric(label="**HeartRate** :gray[(bpm)]", value=sample_df['HeartRate'])
        
    st.write("\n\n\n\n")

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("\n\n")
        contributions_table_component = ShapContributionsTableComponent(explainer, title="Contributions Table", index=index)
        contributions_table_html = contributions_table_component.to_html()
        st.components.v1.html(contributions_table_html, height=730, scrolling=False)
        
    with col2:
        # Contributions Graph Component
        st.write("\n\n")
        contributions_graph_component = ShapContributionsGraphComponent(explainer, title="Contributions Plot", index=index)
        contributions_graph_html = contributions_graph_component.to_html()
        st.components.v1.html(contributions_graph_html, height=730, scrolling=False, width=500) 
    
    st.write("The table shows the contributions of each feature to the prediction for the selected sample. :gray[(Higher the 'Final Predction', lower the predicted risk level)]")
    st.write("And the plot shows how the model makes a prediction for a sample i.e. how each feature contributes to the prediction.")
    st.write("We start from the left that is the baseline a.k.a the average prediction of our model. We then add all the contributions given by the SHAP values one by one and until we end up with the final prediction the right.")
    st.write("A higher value of the 'Prediction %' indicates a final prediction of :green[low risk], while a lower value indicates a final prediction of :red[high risk].")
    st.write("⚠️  **Note:** Before making any decision based on the model's final prediction, please also have a good look at the :violet-background[Confusion Matrix] under the 'About the Model' page. It gives necessary information about the model's performance, namely *false positives* and *false negatives*.")
    st.info("One of the fundamental properties of Shapley values is that their sum yields the difference of actual and average prediction. In our case, it means that SHAP values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained. Also, the baseline remains the same for each point as it is simply the mean of all the predictions of our model", icon="ℹ️")
        
    
if __name__ == "__main__":
    main()
