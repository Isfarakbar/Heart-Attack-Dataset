import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Page setup
st.set_page_config(page_title="Heart Attack Risk Analysis", layout="wide")
st.title("🫀 Heart Attack Risk Analysis App")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart_data.csv")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

df = load_data()

# Sidebar
st.sidebar.header("📌 Navigation")
section = st.sidebar.radio("Go to", ["Dataset Info", "Visualizations", "Statistics", "Probability", "Prediction"])

# Dataset Info
if section == "Dataset Info":
    st.subheader("📊 Dataset Overview")
    st.markdown("""
    This dataset contains anonymized patient records used to assess **heart attack risk**. Features include:
    - Age, Gender
    - Heart Rate
    - Systolic & Diastolic Blood Pressure
    - Blood Sugar
    - CK-MB (Creatine Kinase MB)
    - Troponin Level
    - Diagnosis Result (Positive/Negative)
    """)
    st.dataframe(df.head(10))
    st.write("### Column Information:")
    st.text(df.info())

    st.write("### Summary Statistics:")
    st.dataframe(df.describe())

# Visualizations
elif section == "Visualizations":
    st.subheader("📈 Data Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Gender Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='gender', data=df, ax=ax, palette="Set2")
        ax.set_xticklabels(['Male', 'Female'])
        st.pyplot(fig)

    with col2:
        st.markdown("### Heart Attack Diagnosis Result")
        fig, ax = plt.subplots()
        sns.countplot(x='result', data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

    st.markdown("### Box Plot of Vital Signs")
    vital_cols = ['heart_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'blood_sugar']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[vital_cols], ax=ax)
    ax.set_title("Vital Signs Distribution")
    st.pyplot(fig)

    st.markdown("### Troponin vs CK-MB by Result")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='troponin', y='ck-mb', hue='result', ax=ax)
    st.pyplot(fig)

# Statistics
elif section == "Statistics":
    st.subheader("📏 Descriptive Statistics & Confidence Intervals")

    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, bins=10, color='skyblue', ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    mean_age = df['age'].mean()
    ci_age = stats.t.interval(0.95, len(df['age'])-1, loc=mean_age, scale=stats.sem(df['age']))
    st.write(f"**Mean Age:** {mean_age:.2f}")
    st.write(f"**95% Confidence Interval for Age:** ({ci_age[0]:.2f}, {ci_age[1]:.2f})")

    st.write("### Frequency Table:")
    st.write("**Diagnosis Result:**")
    st.write(df['result'].value_counts())

    st.write("**Gender vs Result Crosstab:**")
    st.write(pd.crosstab(df['gender'], df['result'], rownames=['Gender'], colnames=['Result']))

# Probability
elif section == "Probability":
    st.subheader("🎲 Probability Distribution")

    st.markdown("### Normal Distribution Fit for Troponin Level")
    troponin = df['troponin']
    mu, std = stats.norm.fit(troponin)

    fig, ax = plt.subplots()
    sns.histplot(troponin, bins=20, kde=False, stat='density', color='salmon', ax=ax)
    x = np.linspace(troponin.min(), troponin.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title(f"Normal Fit → Mean: {mu:.2f}, Std Dev: {std:.2f}")
    st.pyplot(fig)

    st.write(f"Estimated Mean = {mu:.2f}")
    st.write(f"Estimated Standard Deviation = {std:.2f}")

# Prediction
elif section == "Prediction":
    st.subheader("🧠 Heart Attack Prediction using Logistic Regression")

    df['result_binary'] = df['result'].apply(lambda x: 1 if x.lower() == 'positive' else 0)
    features = ['age', 'heart_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'blood_sugar', 'ck-mb', 'troponin']
    X = df[features]
    y = df['result_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Model Evaluation")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.write("### Predict Heart Attack Risk")
    with st.form("prediction_form"):
        age = st.number_input("Age", 18, 100)
        heart_rate = st.number_input("Heart Rate", 40, 200)
        sbp = st.number_input("Systolic BP", 80, 250)
        dbp = st.number_input("Diastolic BP", 40, 150)
        sugar = st.number_input("Blood Sugar", 50, 400)
        ckmb = st.number_input("CK-MB", 0.0, 100.0)
        troponin = st.number_input("Troponin", 0.0, 10.0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([[age, heart_rate, sbp, dbp, sugar, ckmb, troponin]], columns=features)
            prediction = model.predict(input_data)[0]
            result = "🟥 High Risk of Heart Attack (Positive)" if prediction == 1 else "🟩 Low Risk (Negative)"
            st.success(f"Prediction Result: {result}")
