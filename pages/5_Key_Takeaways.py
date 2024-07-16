import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.header("Key Takeaways", anchor="conclusion")
    st.title("HealthyMom App")
    st.logo(
        "./love.png",
        icon_image="./heartbeat.gif",
    )
    st.write("This page explores potential applications of the app while also considering situations where its use may not be suitable.")
    
    st.write("\n\n")
    st.subheader("How we imagine that the HealthyMom App could be used?", divider="red")
    st.write("**1. Risk Assessment in Maternity Care:**")
    st.write("- The app could be used to quickly assess the risk levels of expectant mothers based on various health parameters. It could be ideal for triage processes in maternity hospitals to ensure timely and appropriate care for high-risk cases.")
    st.write("**2. Exploration of Predictive Models:**")
    st.write("- Possibility to utilize the app to understand how different health features impact maternal risk predictions.")
    st.write("- It could be beneficial for healthcare professionals who want to explore and validate model predictions by modifying input features using the Prediction Simulator.")
    st.write("**3. Educational Purposes:**")
    st.write("- The app could be a valuable educational tool for data science students to learn about predictive modeling and explanation user interface")
   
    st.write("\n\n\n")
    st.subheader("When NOT to use the app?", divider="red")
    st.write("**1. Non-Predicted Medical Conditions:**")
    st.write("- Do not rely on the app for conditions or symptoms not covered by the model (e.g., stroke symptoms). In such cases, base urgency on professional medical judgment alone.")
    st.write("**2. Overriding Medical Expertise:**")
    st.write("- Do not use the app to override the expertise of healthcare professionals. The app could be a support tool for triage, not a definitive diagnostic tool. Ensure that final decisions regarding patient care are made by qualified medical personnel, taking into account all clinical findings and patient history.")
    st.write("**3. Limited Data Scenarios:**")
    st.write("- Be cautious when using the app in scenarios with data that significantly deviates from the training dataset (e.g., different demographics or health conditions not represented in the data from Dhaka, Bangladesh). A comprehensive evaluation by a data scientist should be performed before using the application in a medical setting.")
    
    st.write("\n\n\n")
    st.subheader("References / Acknowledgements", divider="red")
    st.write("1. Van Katwijk, Cornelis, and Louis LH Peeters. “*Clinical aspects of pregnancy after the age of 35 years: a review of the literature.*” Human reproduction update 4, no. 2 (1998): 185-194.")
    st.write("2. Khalil, Asma, Argyro Syngelaki, Nerea Maiz, Yana Zinevich, and Kypros H. Nicolaides. “*Maternal age and adverse pregnancy outcome: a cohort study.*” Ultrasound in Obstetrics & Gynecology 42, no. 6 (2013): 634-643.")
    st.write("3. **Dataset:** Marzia Ahmed, 14.08.2023, [10.24432/C5DP5D](https://doi.org/10.24432/C5DP5D)")
    st.write("4. Oege Dijk, oegesam, Ray Bell, Lily, Simon-Free, Brandon Serna, rajgupt, et al. “*Oegedijk/explainerdashboard: Explainerdashboard 0.4.2: Dtreeviz V2 Compatiblity*”. Zenodo, February 12, 2023. https://doi.org/10.5281/zenodo.7633294.")
    st.write("5. https://christophm.github.io/interpretable-ml-book/shap.html")
    st.write("6. https://christophm.github.io/interpretable-ml-book/shapley.html")
    
if __name__ == "__main__":
    main()