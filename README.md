### 🌻 HealthyMom App

You can access the web application [here](https://healthy-mom.streamlit.app/) 🌐.

---

#### 📥 Download Repository

Clone the repository to your local machine:

```sh
git clone https://github.com/adityapanchal10/maternal-risk-app.git
```

#### 🛠️ Setup Environment with Python 3.9.19

```sh
conda create -n newenv python=3.9
```

#### 📦 Install Necessary Dependencies

Activate the newly created environment:

```sh
conda activate newenv
```

Then, update the environment using '**env.yml**' if you prefer conda:

```sh
conda env update -n newenv -f env.yml
```

OR install using '**requirements.txt**' if you prefer pip:

```sh
pip install -r requirements.txt
``` 

#### 🚀 Run the Application

Navigate to the directory containing '**0_Home.py**' and execute the following command:

```sh
streamlit run 0_Home.py
```

Once the application is running, you can access it in your web browser by navigating to http://localhost:8501. This will allow you to interact with the HealthyMom app, explore its features, and analyze the predictions and contributions of each feature to the predictions for individual samples.

---

#### References

1. **Data**: https://archive.ics.uci.edu/dataset/863/maternal+health+risk

2. **Explainer Dashboard** ([github](https://github.com/oegedijk/explainerdashboard)): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10835759.svg)](https://doi.org/10.5281/zenodo.10835759) 

3. https://christophm.github.io/interpretable-ml-book/shapley.html

4. https://christophm.github.io/interpretable-ml-book/shap.html

