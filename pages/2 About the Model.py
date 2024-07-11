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

# Display results
def display_results(model, X_train, y_train, X_test, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    cm = confusion_matrix(y_test, y_pred_test)
    
    accuracy_df = pd.DataFrame({
        "Split": ["Train", "Test"],
        "Accuracy": [f"{accuracy_train*100:.2f}%", f"{accuracy_test*100:.2f}%"],
        "# samples": [f"{len(y_train)}", f"{len(y_test)}"]
    })
    st.write("**Accuracy:**")
    accuracy_df = accuracy_df.style.set_properties(**{'text-align': 'left'})
    accuracy_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(accuracy_df, width=500, hide_index=True)

    fig = go.Figure(data=go.Heatmap(
                    z=cm[::-1],
                    x=['High Risk', 'Low Risk', 'Medium Risk'],  
                    y=['Medium Risk', 'Low Risk', 'High Risk'],  
                    hoverongaps=False,
                    text=cm[::-1],
                    colorscale="blues",
                    texttemplate="%{text}"))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title="Predicted",
        yaxis_title="True")

    st.plotly_chart(fig)
    plt.clf()  # Clear the current figure after displaying it
    
def main():
    st.write("## About the Model")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("\n\n")
    st.write("### Model Training and Evaluation")
    st.write("**Model**: [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
    st.write("**Hyperparameters**:")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.write("- [criterion](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=100%20in%200.22.-,criterion,-%7B%E2%80%9Cgini%E2%80%9D%2C%20%E2%80%9Centropy%E2%80%9D%2C%20%E2%80%9Clog_loss): log_loss")
    with c2:
        st.write("- [max_depth](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=is%20tree%2Dspecific.-,max_depth,-int%2C%20default%3DNone): 15")
    with c3:
        st.write("- [max_features](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=is%20not%20provided.-,max_features,-%7B%E2%80%9Csqrt%E2%80%9D%2C%20%E2%80%9Clog2%E2%80%9D%2C%20None): log2")
    with c4:
        st.write("- [n_estimators](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=Parameters%3A-,n_estimators,-int%2C%20default%3D100): 100")
    
    st.write("The above mentioned hyperparameters are the result of hyperparameter tuning using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#gridsearchcv) using a 4 fold cross-validation.")

    df, target = load_data()
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
    st.write(f"Training on **{len(X_train)}** samples and using **{len(X_test)}** samples for validation.")
    model = load_model()
    with st.spinner(text='Training...'):
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
    st.write(f"**Training time**: {training_time:.2f} seconds")    
    st.toast('Training Complete !!', icon="✔️")
        
    st.write("\n\n\n")
    st.write("### Model Performance")
    display_results(model, X_train, y_train, X_test, y_test)
    st.write("One can see that the accuracy is quite good.")
    st.write("However, the accuracy metric only gives the overall corectness of the model.")
    st.write("In order to get a better understanding of the model's performance across different classes, the confusion matrix is more valueable.")
    st.write("The confusion matrix shows the actual v.s. predicted classification for each class.")
    st.write("By looking at the confusion matrix, we can see that our model does a good job in reducing the number of false positives i.e. if the actual is *High Risk*, only a few instances are predicted as *Low Risk* or *Medium Risk*.")
    st.write("This is important because in the context of maternal health, we want to minimize the number of false positives as much as possible i.e. a *High Risk* and *Medium Risk* should not be predicted as *Low Risk* as much as possible.")
    st.write("The inverse is okay i.e. if a *Low Risk* is predicted as *Medium Risk* or *High Risk*, it is not as bad as the former case.")
        
if __name__ == "__main__":
    main()
