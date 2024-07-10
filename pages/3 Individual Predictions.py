import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Maternal Health Risk Data Set.csv")
    target = 'RiskLevel'
    return df, target

# Train Logistic Regression model
@st.cache_resource
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

# Train Kernel Ridge Regression model
@st.cache_resource
def train_kernel_ridge_regression(X_train, y_train):
    model = KernelRidge(kernel='rbf')
    start_time = time.time()
    model.fit(X_train, y_train) 
    training_time = time.time() - start_time
    return model, training_time

# Display results
def display_results(model, X_test, y_test, training_time):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Training Time: {training_time:.4f} seconds")
    
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')
    sns.heatmap(cm, annot=True, ax=ax, fmt=".1f")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()  # Clear the current figure after displaying it

#@st.cache_data
def explain_model(_model, X_train, model_type):
    # Initialize the SHAP explainer based on the model type
    if model_type == "Logistic Regression":
        def model_predict_log_odds(x): 
            p = _model.predict_log_proba(x)
            return p[:, 1] - p[:, 0]
        explainer = shap.KernelExplainer(model_predict_log_odds, shap.kmeans(X_train, 7))
        # Calculate SHAP values
        shap_values = explainer(X_train[:25])
    elif model_type == "Kernel Ridge Regression":
        # KernelExplainer works better for non-linear models
        explainer = shap.KernelExplainer(_model.predict, shap.kmeans(X_train, 7))
        # Calculate SHAP values
        shap_values = explainer(X_train[:25])
    
    # Return the explainer and SHAP values for further use
    return explainer, shap_values

def display_shap_summary_plot(explainer, shap_values, X, y):
    if 'clustering' in st.session_state:
        clust = st.session_state['clustering']
    else:
        clust = shap.utils.hclust(X, y, linkage="single")
        st.session_state['clustering'] = clust
    shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=2, max_display=len(X.columns))
    st.pyplot()
    plt.clf()  # Clear the current figure after displaying it

def display_shap_scatter_plot(shap_values):
    plt.figure(figsize=(15, 6))
    shap.plots.scatter(shap_values[:, :7], ylabel="SHAP value\n(higher means more likely to be diagnosed diabetic)", show=False) 
    st.pyplot()
    plt.clf()  # Clear the current figure after displaying it
    shap.plots.scatter(shap_values[:, 7:14], ylabel="SHAP value\n(higher means more likely to be diagnosed diabetic)", show=False) 
    st.pyplot()
    plt.clf()
    shap.plots.scatter(shap_values[:, 14:], ylabel="SHAP value\n(higher means more likely to be diagnosed diabetic)", show=False) 
    st.pyplot()
    plt.clf()
    
def display_waterfall_plot(shap_values, selected_index):
    shap.plots.waterfall(shap_values[selected_index], max_display=15)
    st.pyplot()
    plt.clf()  # Clear the current figure after displaying it

def main():
    st.title("CDC Diabetes Health Indicators Data Predictor")
    st.write("Select a model from the sidebar for training.")
    st.write("It will then be used for prediction, evaluation, and explanation.")
    st.write("- The [*logistic regression*](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model is trained on the entire dataset till a maximum of 200 iterations.")
    st.write("- The [*kernel ridge regression*](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html) model is trained using a Radial Basis Kernel and only on the first 1000 samples due to memory constraints.")
    st.write("SHAP calculations for explanations could only use 100 values from the train set due to computational constraints...")
    st.write("\n\n\n")
    st.write("### ➾ Model Training and Evaluation")

    df, target = load_data()
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)

    st.sidebar.header("☢ Model Selection")
    model_option = st.sidebar.radio("Select Model", ("Logistic Regression", "Kernel Ridge Regression"))

    if model_option == "Logistic Regression":
        model, training_time = train_logistic_regression(X_train, y_train)
        # r_indices = np.random.choice(len(X_train), 500, replace=False)
        explainer, shap_values = explain_model(model, X_train, "Logistic Regression")
        
        st.write(f"︻デ═一 {model_option}")
        display_results(model, X_test, y_test, training_time)
        
        st.write("\n\n\n")
        st.write("### ➾ Model Explanation (Looking at the co-efficients)")
        lr_coefs = model.coef_[0]
        lr_coefs = pd.Series(lr_coefs, index=X_train.columns)
        lr_coefs = lr_coefs.sort_values(ascending=False)
        st.write("#### Feature Importance")
        st.write("The most common way of understanding a linear model is to examine the coefficients learned for each feature.")
        st.write("These coefficients tell us how much the model output changes when we change each of the input features.")
        st.bar_chart(lr_coefs)
        st.write("However, they are not not a great way to measure the overall importance of a feature.")
        st.write("This is because the value of each coefficient depends on the scale of the input features. For example, if we use months as a unit for Age instead of years, the coefficient for Age will be 12 times smaller which does not make sense.")
        st.write("This means that the magnitude of a coefficient is not necessarily a good measure of a feature’s importance in a linear model.")
        st.write("Hence, it is necessary to understand both how changing that feature impacts the model’s output, and also the distribution of that feature’s values. To do this, we will use [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) values.")
        
        st.write("\n\n\n")
        st.write("### ➾ Model Explanation with SHAP")
        st.write("Shapley values are a concept from game theory that provide a natural way to compute which features contribute to a prediction or contribute to the uncertainty of a prediction.")
        st.write("A prediction can be explained by assuming that each feature value of the instance is a 'player' in a game where the prediction is the payout.")
        st.write("> ***Note***: The Shapley value of a feature value is **not** the difference of the predicted value after removing the feature from the model training. It can be interpreted as - given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.")
        st.write("")
        st.write("##### 1. Summary Plot")
        st.write("We start by plotting the global importance of each feature in the model.")
        display_shap_summary_plot(explainer, shap_values, X_train[:50], y_train[:50])
        st.write("This bar plot shows that GenHealth, HighBP, BMI, and age are the top factors driving the model’s prediction of having diabetes or not.")
        st.write("This is interesting and at first glance looks reasonable. The bar plot also includes a feature redundancy clustering which we will use later.")
        
        st.write("\n\n")
        st.write("##### 2. SHAP Scatter Plots")
        st.write("Let us see how changing the value of a feature impacts the model’s prediction of diabetes or not probabilities.")
        st.write(" If the blue dots follow an increasing pattern, this means that the larger the feature, the higher is the model’s predicted 'having diabetes' probability.")
        display_shap_scatter_plot(shap_values)
        st.write("The scatter plots show the relationship between the feature importance shown in the summary plot and the SHAP scatter plot.")
        st.write("(The blue dots of features such as GenHealth, HighBP, and BMI show an increasing pattern, indicating higher is its importance in prediction, as given by the Summary Plot).")
        st.write("From the scatter plot, it can also be seen that the higher level of Income or Education one has, less likely they are to be diagnosed with diabetes. This is interesting as one would hope that diabetes wouldn't discriminate against rich and poor neither it would consider the education level of the patient. One probable reason for it might be better access to healthcare and a healthier lifestyle for people with higher income and education levels.")
        st.write("\nComing back to the feature redundancy clustering, we can see that the features such as Income, and Education are clustered together. This means that these features are correlated and provide similar information to the model. i.e. the information those features contain about the outcome is redundant and the model could have used either feature.")
        st.write("There are also some features that are idependent like Smoker and HvyAlcoholConsump. Such features do not suffer from [observed confounding](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html#Observed-confounding).")

        st.write("\n\n")
        st.write("##### 3. How is a prediction made?")
        st.write("The below chart shows how the model makes a prediction for a sample (the sample can be selected in the sidebar).")
        st.sidebar.header("☢ Sample Selection")
        indices = np.random.choice(len(X_train[:20]), 20, replace=False)
        selected_index = st.sidebar.selectbox('Select a sample index:', indices)
        if st.sidebar.button('Get random sample'):
            random_index = np.random.choice(len(X_train[:20]))
            st.sidebar.write(f"Randomly selected sample index: {random_index}")
            selected_index = random_index  # This line won't update the selectbox, but one can use selected_index for further processing.
        sample_df = df.iloc[selected_index]
        sample_df = sample_df.to_frame().T
        st.dataframe(sample_df, hide_index=True)
        display_waterfall_plot(shap_values, selected_index)
        st.write("The waterfall plot shows how the model makes a prediction for a sample. The base value is the average prediction of the model. The SHAP values show how each feature contributes to the prediction.")
        st.write("The SHAP values are added one at a time, starting from the bottom, until the current model prediction is reached.")
        st.write("One of the fundamental properties of Shapley values is that they always sum up to the difference between the game outcome when all players are present and the game outcome when no players are present.")
        st.write("In our case, it means that SHAP values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained.")

    else:
        model, training_time = train_kernel_ridge_regression(X_train[:1000], y_train[:1000]) # considering only 1000 samples due to memory constraints
        explainer, shap_values = explain_model(model, X_train, "Kernel Ridge Regression")
        st.write(f"︻デ═一 {model_option}")
        display_results(model, X_test, y_test, training_time)
        
        # Display a summary plot
        st.write("\n\n\n")
        st.write("### ➾ Model Explanation with SHAP")
        st.write("Shapley values are a concept from game theory that provide a natural way to compute which features contribute to a prediction or contribute to the uncertainty of a prediction.")
        st.write("A prediction can be explained by assuming that each feature value of the instance is a 'player' in a game where the prediction is the payout.")
        st.write("> ***Note***: The Shapley value of a feature value is **not** the difference of the predicted value after removing the feature from the model training. It can be interpreted as - given the current set of feature values, the contribution of a feature value to the difference between the actual prediction and the mean prediction is the estimated Shapley value.")
        st.write("")
        st.write("##### 1. Summary Plot")
        st.write("We start by plotting the global importance of each feature in the model")
        display_shap_summary_plot(explainer, shap_values, X_train[:50], y_train[:50])
        st.write("This bar plot shows that BMI, Age, and GenHealth are the top three factors driving the model’s prediction of having diabetes or not.")
        st.write("This is somewhat similar to that of Logistic Regression. There is also a feature redundancy clustering as before.")
        
        st.write("\n\n")
        st.write("##### 2. SHAP Scatter Plots")
        st.write("Let us see how changing the value of a feature impacts the model’s prediction of diabetes or not probabilities.")
        st.write(" If the blue dots follow an increasing pattern, this means that the larger the feature, the higher is the model’s predicted 'having diabetes' probability.")
        display_shap_scatter_plot(shap_values)
        st.write("Observing the trend of the blue dots is a bit difficult than the Logistic Regression model. Once can also note the non-linearity of the model from the scatter plots.")
        st.write("However, we can see the relationship between the feature importance shown in the summary plot and the SHAP scatter plot.")
        st.write("\nClustering here also shows features Income and Education clustered together. Hence, the model could have used either feature in prediction as they provide similar information to the model.")
    
        st.write("\n\n")
        st.write("##### 3. How is a prediction made?")
        st.write("The below chart shows how the model makes a prediction for a sample (the sample can be selected in the sidebar).")
        st.sidebar.header("☢ Sample Selection")
        indices = np.random.choice(len(X_train[:20]), 20, replace=False)
        selected_index = st.sidebar.selectbox('Select a sample index:', indices)
        if st.sidebar.button('Get random sample'):
            random_index = np.random.choice(len(X_train[:20]))
            st.sidebar.write(f"Randomly selected sample index: {random_index}")
            selected_index = random_index  # This line won't update the selectbox, but one can use selected_index for further processing.
        sample_df = df.iloc[selected_index]
        sample_df = sample_df.to_frame().T
        st.dataframe(sample_df, hide_index=True)
        display_waterfall_plot(shap_values, selected_index)
        st.write("The waterfall plot shows how the model makes a prediction for a sample. The base value is the average prediction of the model. The SHAP values show how each feature contributes to the prediction.")
        st.write("The SHAP values are added one at a time, starting from the bottom, until the current model prediction is reached.")
        st.write("One of the fundamental properties of Shapley values is that they always sum up to the difference between the game outcome when all players are present and the game outcome when no players are present.")
        st.write("In our case, it means that SHAP values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained.")
    
    st.write("\n\n\n")
    st.write("### ➾ Conclusion")
    st.write("Though the SHAP values provide a great way to understand how the model makes predictions, it is important to remember that they are not causal. They only make the corelations of the underlying model transparent.")
    st.write("To rely or not on the model predictions is a much more complex decision that requires a lot more than looking at the SHAP values (the units of the model output taken under explanation can lead to very different views of the model behaviour) especially when the model is to be used in a medical scenario.")

if __name__ == "__main__":
    main()
