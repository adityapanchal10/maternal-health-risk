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
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 
from sklearn.multiclass import OneVsRestClassifier
import fairness_functions as ff

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
    
def get_fairness():
    df, target = load_data()

    # create a list of the conditions
    conditions = [
        (df['Age'] >= 10) & (df['Age'] <= 19),
        (df['Age'] >= 20) & (df['Age'] <= 34),
        (df['Age'] > 34)
        ]
    # create a list of the values we want to assign for each condition
    values = ['teenager', 'adult', 'advanced maternal age']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['AgeGroup'] = np.select(conditions, values, default='unknown')

    # Encode the target variable
    # high risk = 0, low risk = 1, mid risk = 2, adult = 0, advanced maternal age = 1, teenager = 2
    le = LabelEncoder()
    df['AgeGroupEncoded'] = le.fit_transform(df['AgeGroup'])
    df['RiskLevelEncoded'] = le.fit_transform(df['RiskLevel'])

    # Split the data
    X = df.drop(['AgeGroup','RiskLevel', 'RiskLevelEncoded'], axis=1)
    y = df['RiskLevelEncoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Train a One-vs-Rest classifier with RandomForest
    rf_classifier = RandomForestClassifier(random_state=7)
    ovr = OneVsRestClassifier(rf_classifier)
    ovr.fit(X_train, y_train)

    # Make predictions
    y_pred = ovr.predict(X_test)

    # Add predictions to the DataFrame for fairness functions
    df_test = X_test.copy()
    df_test['TrueLabel'] = y_test
    df_test['Prediction'] = y_pred

    # Group Fairness
    high_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 0)
    high_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 0)
    high_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 0)
    mid_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 2)
    mid_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 2)
    mid_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 2)
    low_risk_adult = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 0, 'Prediction', 1)
    low_risk_adv_age = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 1, 'Prediction', 1)
    low_risk_teen = group_fairness_value = ff.group_fairness(df_test, 'AgeGroupEncoded', 2, 'Prediction', 1)

    st.write("\n")
    st.write("Fairness metrics generally apply to binary classification. But here, our target variable has three classes namely - :green[low risk], :orange[mid risk], and :red[high risk].")
    st.write("Hence, we employ the **One-vs-Rest (OvR)** approach. This method converts the multiclass problem into several binary classification problems. For each class, a binary classification problem is created where the class of interest is the positive class, and all other classes are combined as the negative class.")
    st.write("Which in our case becomes:")
    st.write("- :red[high risk] vs. (:orange[mid risk] + :green[low risk])")
    st.write("- :orange[mid risk] vs. (:red[high risk] + :green[low risk])")
    st.write("- :green[low risk] vs. (:red[high risk] + :orange[mid risk])")
    
    st.write("\n\n\n")
    st.subheader("1. Group Fairness", anchor="group-fairness", divider="red")
    group_fairness_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "High Risk": [f"{high_risk_teen*100:.2f}%", f"{high_risk_adult*100:.2f}%", f"{high_risk_adv_age*100:.2f}%"],
        "Mid Risk": [f"{mid_risk_teen*100:.2f}%", f"{mid_risk_adult*100:.2f}%", f"{mid_risk_adv_age*100:.2f}%"],
        "Low Risk": [f"{low_risk_teen*100:.2f}%", f"{low_risk_adult*100:.2f}%", f"{low_risk_adv_age*100:.2f}%"]
    })
    group_fairness_df = group_fairness_df.style.set_properties(**{'text-align': 'left'})
    group_fairness_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(group_fairness_df, width=500, hide_index=True)

    st.write("Group fairness aims to ensure that certain desirable properties or outcomes are evenly distributed among groups defined by sensitive attributes, such as age, gender, race, or socioeconomic status. For the sake of this project, we assumed age to be the sensitive attribute. Each group need to have the same probability of being assigned to the predicted class.")
    st.write("For example, if we investigate the 'Age Group' then all groups, protected and unprotected should ideally have the same probability to receive a high risk, mid risk, and low risk prediction. Mathematically this is stated as followed:")
    st.write(r"$P(RiskPrediction = high \vert Age Group = Advanced Maternal Age) == P(RiskPrediction = high \vert Age Group = Adult)$")
    st.write("However, as demonstrated in the table above, the groups exhibit varying values. This discrepancy is understandable, given that the assumption of age as a sensitive attribute is not entirely viable, since pregnancy is indeed age-dependent [cite some paper(s) here if time].")
    st.write("Hence, this does not necessarily indicate bias, as such variations are inherent in nature.")

    # Predictive Parity
    ppv_adult = ff.predictive_parity(df_test, "AgeGroupEncoded", 0, "Prediction", "TrueLabel")
    ppv_adv_age = ff.predictive_parity(df_test, "AgeGroupEncoded", 1, "Prediction", "TrueLabel")
    ppv_teenager = ff.predictive_parity(df_test, "AgeGroupEncoded", 2, "Prediction", "TrueLabel")

    st.write("\n\n\n")
    st.subheader("2. Predictive Parity", anchor="ppv", divider="red")
    ppv_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "Positive Predictive Value (PPV)": [f"{ppv_teenager*100:.2f}%", f"{ppv_adult*100:.2f}%", f"{ppv_adv_age*100:.2f}%"]
    })
    ppv_df = ppv_df.style.set_properties(**{'text-align': 'left'})
    ppv_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(ppv_df, width=500, hide_index=True)

    st.write("Predictive Parity measures the proportion of positive predictions that are actually correct. A high PPV indicates that we can be sure that a positive prediction is true.")
    st.write("In our example, we would like to analyze whether different age groups are less likely to truly belong to the prediction and whether there is a significant difference among these three groups.")

    # False Positive Error Rate Balance
    fpr_adult = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 0, "Prediction", "TrueLabel")
    fpr_adv_age = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 1, "Prediction", "TrueLabel")
    fpr_teenager = ff.fp_error_rate_balance(df_test, "AgeGroupEncoded", 2, "Prediction", "TrueLabel")

    st.write("\n\n\n")
    st.subheader("3. False Positive Error Rate", anchor="fpr", divider="red")
    fpr_df = pd.DataFrame({
        "Age Group": ["Teenager", "Adult", "Advanced Maternal Age"],
        "False Positive Rate (FPR)": [f"{fpr_teenager*100:.2f}%", f"{fpr_adult*100:.2f}%", f"{fpr_adv_age*100:.2f}%"]
    })
    fpr_df = fpr_df.style.set_properties(**{'text-align': 'left'})
    fpr_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    st.dataframe(fpr_df, width=500, hide_index=True)

    st.write("False Positive Error Rate measures the proportion of negative cases that are incorrectly classified as positive. In other words, it tells you how often a model incorrectly predicts the positive class for cases that should be in the negative class.")
    st.write("We would like to analyze if any age group is favored by having a higher FPR than the other, thus predicting it more often to be prone to other types risks even though they are not.")
    st.toast("Fairness metrics loaded !!", icon="✔️")
    
def main():
    st.header("About the Model", anchor="model")
    st.title("Maternal Health Risk Prediction")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("This page provides information about the model used for prediction, its performance, and the feature importance alongwith information regarding the model's fairness.")
    st.write("\n\n")
    
    tab1, tab2 = st.tabs(["**Model Training and Evaluation**", "**Model Fairness**"])
    
    with tab2:
        get_fairness()
        
    with tab1:
        st.write("\n")
        st.write("**Model**: [Random Forest](https://willkoehrsen.github.io/data%20science/machine%20learning/random-forest-simple-explanation/)")
        
        with st.popover("🚀 **Hyperparameters**"):
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
        st.subheader("Model Performance", anchor="model-performance", divider="red")
        display_results(model, X_train, y_train, X_test, y_test)
    
        st.write("By looking at the confusion matrix, we can see that our model does a good job in reducing the number of false positives i.e. if the actual is *:red[High Risk]*, only a few instances are predicted as *:green[Low Risk]* or *:orange[Medium Risk]*.")
        st.write("This is important because in the context of maternal health, we want to minimize the number of false positives as much as possible i.e. a *:red[High Risk]* and *:orange[Medium Risk]* should not be predicted as *:green[Low Risk]* as much as possible.")
        st.write("The inverse, a false negative, is okay i.e. if a *:green[Low Risk]* is predicted as *:orange[Medium Risk]* or *:red[High Risk]*, it is not as bad as the former case.")    
        with st.expander("💡 Click here to know more about the confusion matrix..."):
            st.write("The accuracy metric only gives the overall correctness of the model.")
            st.write("In order to get a better understanding of the model's performance across different classes, the confusion matrix is more valueable.")
            st.write("The confusion matrix shows the actual v.s. predicted classification for each class.")
            
        st.write("\n\n")
        help_str = "We do not use the model co-efficeints as feature importances because the value of each co-efficient depends on the scale of the input features. For example, if we use months as a unit for Age instead of years, the coefficient for Age will be 12 times smaller which does not make sense.\nThis means that the magnitude of a coefficient is not necessarily a good measure of a feature’s importance.\nHence, SHAP values are used to calculate feature importances."
        st.subheader("Feature Importances", anchor="feature-importances", help=help_str, divider="red")
        st.write("Using [:blue-background[ExplainerDashboard]](https://github.com/oegedijk/explainerdashboard) for our model, we visualize feature importances.")
        
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        model = model.fit(X_train, y_train)
        
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
        
        # st.toast('Explainer loaded', icon="✔️")
        st.write("From the plot above, we can see that the most prominent feature for the model in its decision making is *BS* i.e blood sugar levels")
        st.write("This gives an overview of the model's decision making process. However, if one wants to see the contributions for a single sample, click on 'Individual Prediction' in the sidebar.") 
        
        with st.expander("📚 **General Note**"):
            st.write('We do not use the model co-efficeints as feature importances because the value of each co-efficient depends on the scale of the input features. For example, if we use months as a unit for Age instead of years, the coefficient for Age will be 12 times smaller which does not make sense.')
            st.write("This means that the magnitude of a coefficient is not necessarily a good measure of a feature’s importance.")
            st.write("Hence, SHAP values are used to calculate feature importances.")
        with st.expander("🤯 **What are SHAP values**? 🎲"):
            st.write("Shapley values are a concept from game theory that provide a natural way to compute which features contribute to a prediction or contribute to the uncertainty of a prediction.")
            st.write("A prediction can be explained by assuming that each feature value of the instance is a 'player' in a game where the prediction is the payout.")
            st.info("The SHAP value of a feature is **not** the difference of the predicted value after removing the feature from the model training. It can be interpreted as - given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.", icon="ℹ️")
        
if __name__ == "__main__":
    main()
