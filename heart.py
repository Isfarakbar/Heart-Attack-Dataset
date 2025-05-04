import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Page settings
st.set_page_config(page_title="Heart Attack Data Analysis", layout="wide")
st.title("❤️ Heart Attack Dataset Analysis")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart_data.csv")  # Replace with your CSV file path
    return df

df = load_data()

# Clean and preprocess
df.columns = df.columns.str.strip().str.replace(' ', '_')
df['Result'] = df['Result'].str.lower().str.strip()
df['Gender'] = df['Gender'].map({0: 'Female', 1: 'Male'})

# Encode target variable
le = LabelEncoder()
df['Result_encoded'] = le.fit_transform(df['Result'])

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Info", "Visualizations", "Statistics", "Classification", "Probability"])

# Section 1: Dataset Info
if section == "Dataset Info":
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

    st.write("### Column Info")
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(map(str, buffer)))

    st.write("### Descriptive Statistics")
    st.dataframe(df.describe())

# Section 2: Visualizations
elif section == "Visualizations":
    st.subheader("📈 Visualizations")

    st.markdown("### Heart Attack Result Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Result', data=df, palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.markdown("### Gender Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Gender', data=df, palette='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Age vs Blood Pressure (Scatter)")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='Age', y='Systolic_blood_pressure', hue='Result', data=df, ax=ax3)
    st.pyplot(fig3)

    st.markdown("### CK-MB and Troponin Distribution by Result")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Result', y='Troponin', data=df, ax=ax4)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.boxplot(x='Result', y='CK-MB', data=df, ax=ax5)
    st.pyplot(fig5)

# Section 3: Statistics
elif section == "Statistics":
    st.subheader("📏 Descriptive Stats & Confidence Intervals")

    for col in ['Age', 'Heart_rate', 'Systolic_blood_pressure', 'CK-MB', 'Troponin']:
        vals = df[col]
        mean = vals.mean()
        ci = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=stats.sem(vals))
        st.write(f"**{col}**: Mean = {mean:.2f}, 95% CI = ({ci[0]:.2f}, {ci[1]:.2f})")

# Section 4: Classification
elif section == "Classification":
    st.subheader("📉 Logistic Regression: Heart Attack Prediction")

    features = ['Age', 'Heart_rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Blood_sugar', 'CK-MB', 'Troponin']
    X = df[features]
    y = df['Result_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Section 5: Probability
elif section == "Probability":
    st.subheader("🎲 Probability Distribution - Troponin Levels")

    troponin = df['Troponin']
    mu, std = stats.norm.fit(troponin)
    fig5, ax5 = plt.subplots()
    sns.histplot(troponin, bins=30, kde=False, stat='density', ax=ax5)
    xmin, xmax = ax5.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax5.plot(x, p, 'r', linewidth=2)
    ax5.set_title("Troponin Level Distribution with Normal Fit")
    st.pyplot(fig5)

    st.write(f"Mean = {mu:.2f}, Standard Deviation = {std:.2f}")
