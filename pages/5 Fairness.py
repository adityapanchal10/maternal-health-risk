import streamlit as st


# if 'clicked1' not in st.session_state:
#     st.session_state.clicked1 = False
    
# if 'clicked2' not in st.session_state:
#     st.session_state.clicked2 = False

# if 'clicked3' not in st.session_state:
#     st.session_state.clicked3 = False
    
# if 'stage' not in st.session_state:
#     st.session_state.stage = 0
    
# def set_state(i):
#     st.session_state.stage = i

# protected = ["Age"]

# # Load dataset
# #@st.cache_data
# def load_data():
#     df = pd.read_csv("./Maternal Health Risk Data Set.csv")
#     target = 'RiskLevel'
#     return df, target

# # Train Logistic Regression model
# #@st.cache_data
# def train_logistic_regression(X_train, y_train):
#     model = LogisticRegression(max_iter=200)
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     training_time = time.time() - start_time
#     return model, training_time

# # Train Kernel Ridge Regression model
# #@st.cache_data
# def train_kernel_ridge_regression(X_train, y_train):
#     model = KernelRidge(kernel='rbf')
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     training_time = time.time() - start_time
#     return model, training_time

# # Get predictions for entire dataset
# def get_predictions(model, X):
#     return np.round(model.predict(X)).astype(int)

# # Remove selected attributes from dataframe
# def remove_attributes(df, attributes):
#     df = df.drop(columns=attributes)
#     return df

# # Display results
# def display_results(model, X_test, y_test, training_time):
#     y_pred = model.predict(X_test)
#     y_pred = np.round(y_pred).astype(int)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     st.write(f"Model used: {model}")
#     st.write(f"Training Time: {training_time:.4f} seconds")
#     st.write(f"Accuracy: {accuracy:.4f}")
    
# def get_fairness_metrics(df, vars, new=False):
#     for i, attribute in enumerate(vars):
#         st.write("\n\n")
#         st.write(f"︻デ═一 {attribute}")
#         st.write("- **Group Fairness**")
#         for val in sorted(df[attribute].unique()):
#             group_fairness = ff.group_fairness(df, attribute, val, "Predictions", 1)
#             st.write(f"P(DiabetesPrediction = 1 | {attribute} = {val}) = {group_fairness:.3f}")
        
#         st.write("- **Conditional statistical parity**")
#         if new:
#             cond = st.selectbox("Select the conditional attribute (default: 'Smoker'): ", ("Smoker", "HighBP", "HighChol", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost"), index=0, key=i+10)
#         else:
#             cond = st.selectbox("Select the conditional attribute (default: 'Smoker'): ", ("Smoker", "HighBP", "HighChol", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost"), index=0, key=i)
#         for val in sorted(df[attribute].unique()):
#             csp = ff.conditional_statistical_parity(df, attribute, val, "Predictions", 1, cond, 1)
#             st.write(f"P(DiabetesPrediction = 1 | {attribute} = {val}, {cond} = 1) = {csp:.3f}")
            
#         st.write("- **Predictive parity**")
#         for val in sorted(df[attribute].unique()):
#             pp = ff.predictive_parity(df, attribute , val, "Predictions", "Diabetes_binary")
#             st.write(f"Predictive parity ({attribute} = {val}) = {pp:.3f}")
            
#         st.write("- **False Positive Rate**")
#         for val in sorted(df[attribute].unique()):
#             fpr = ff.fp_error_rate_balance(df, attribute, val, "Predictions", "Diabetes_binary")
#             st.write(f"False Positive Rate ({attribute} = {val}) = {fpr:.3f}")

# def main():
#     st.title("CDC Diabetes Health Indicators Fairness Analysis")

#     # st.write(st.session_state)
#     df, target = load_data()
#     X = df.drop(target, axis=1)
#     y = df[target]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)

#     st.sidebar.header("☢ Model Selection")
#     model_option = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Kernel Ridge Regression"))

#     if model_option == "Logistic Regression":
#         model, training_time = train_logistic_regression(X_train, y_train)
#     else:
#         model, training_time = train_kernel_ridge_regression(X_train[:3000], y_train[:3000])
        
#     st.write("Select a model from the sidebar for training and click the *Run* button to generate predictions for the entire dataset.")
#     st.write("Again, *logistic regression* is trained on the entire dataset till a maximum of 200 iterations.")
#     st.write("Whereas *kernel ridge regression* is trained using a Radial Basis Kernel and only on the first 3000 samples due to memory constraints.")

#     # Button to get predictions for entire dataset
    
#     if st.sidebar.button("Run") or st.session_state.clicked1:
#         df2 = df
#         predictions = get_predictions(model, X)
#         df2['Predictions'] = predictions
#         cols = df2.columns.tolist()
#         cols = cols[:1] + [cols[-1]] + cols[1:-1]
#         df2 = df2[cols]
#         st.session_state.clicked1 = True
#         #st.session_state.clicked2 = False
#         #st.session_state.clicked3 = False
#         set_state(1)
        
#         st.write("\n\n\n")
#         st.write(f"### ➾ CDC Diabetes Health Indicators Predictions")
#         display_results(model, X_test, y_test, training_time)
    
#         st.write("\n\n\n")
#         st.write(f"### ➾ Fairness Analysis")
#         st.write("Select a protected variable from the sidebar and click on the *Analyse* button to get its fairness metrics .")
    
#     st.sidebar.write("---")
#     st.sidebar.header("☢ PV Selection")
#     attributes_to_analyse = st.sidebar.multiselect("Select the Protected variables to analyse", protected)
#     if st.sidebar.button("Analyse") or st.session_state.clicked2:
#         st.session_state.clicked2 = True
#         # st.session_state.clicked3 = False
#         set_state(2)
#         get_fairness_metrics(df2, attributes_to_analyse)
#         st.write("\n\n")
#         st.write("One can observe that there is bias.")
#         st.write("In order to mitigate this, we employ the strategy of removing the PVs and retraining the model.")
        
#         st.write("\n\n\n")
#         st.write(f"### ➾ Predictions after Removing the PVs")
#         st.write("Select the PVs you want to remove and click the *Remove and Retrain* button to retrain the model.")

#     st.sidebar.write("---")
#     st.sidebar.header("☢ PV Removal")
#     # Multiselect option to remove attributes
#     attributes_to_remove = st.sidebar.multiselect("Select the Protected variables to remove", protected)
    
#     # Button to remove selected attributes and train model again
#     if st.sidebar.button("Remove and Retrain") or st.session_state.clicked3:
#         st.session_state.clicked3 = True
#         st.session_state.clicked2 = True
#         st.session_state.clicked1 = True
#         set_state(3)
#         df_new = remove_attributes(df, attributes_to_remove)
#         X = df_new.drop(target, axis=1)
#         y = df_new[target]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)
#         if model_option == "Logistic Regression":
#             model, training_time = train_logistic_regression(X_train, y_train)
#         else:
#             model, training_time = train_kernel_ridge_regression(X_train[:3000], y_train[:3000])
#         predictions = get_predictions(model, X)
#         df2 = df
#         df2['Predictions'] = predictions
#         cols = df2.columns.tolist()
#         cols = cols[:1] + [cols[-1]] + cols[1:-1]
#         df2 = df2[cols]
    
#         st.write(f"Protected Variables Removed: {attributes_to_remove}")
#         display_results(model, X_test, y_test, training_time)
#         # st.write(df2)
    
#         st.write("\n\n\n")
#         st.write(f"### ➾ Fairness Metrics after Removing the PVs")
#         st.write("New predictions are generated after removing the selected PVs.")
#         st.write("These are then joined with the original dataset to calculate the fairness metrics.")
#         get_fairness_metrics(df2, attributes_to_analyse, new=True)

def main():
    st.title("🚧 Page Under Construction 🔨")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.image("./excavator.gif")

if __name__ == "__main__":
    main()
