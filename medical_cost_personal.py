import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load dataset
file_path = "insurance.csv"
data = pd.read_csv(file_path)

# Pre-trained model pipeline setup
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
X = data.drop("charges", axis=1)
y = data["charges"]
model_pipeline.fit(X, y)

# Streamlit UI setup
st.title("Insurance Charges Prediction Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "Home",
    "Data Overview": "Data Overview",
    "Exploratory Analysis": "Exploratory Analysis",
    "Predict Charges": "Predict Charges"
}
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

if selected_page == "Home":
    st.image("https://cdn.prod.website-files.com/65fda7b5fdef3cef45c71e36/660a93290bfbfe052077345f_657b26b4dbfa4c1ce2120024_employee-healthcare-cost.png", caption="Insurance Dashboard", use_column_width=True)
    st.header("Welcome to the Insurance Dashboard")
    st.write("Use this app to explore insurance data and predict charges.")
elif selected_page == "Data Overview":
    st.header("Dataset Overview")
    st.write("Here is the first few rows of the dataset:")
    st.dataframe(data.head())

    st.subheader("Dataset Summary")
    st.write(data.describe())
    st.write("### Column Details")
    st.write({col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)})

elif selected_page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['age'], bins=15, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Charges vs. Smoking Status")
    fig, ax = plt.subplots()
    data.groupby('smoker')['charges'].mean().plot(kind='bar', color=['blue', 'orange'], ax=ax)
    ax.set_title("Average Charges by Smoking Status")
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Average Charges")
    st.pyplot(fig)

elif selected_page == "Predict Charges":
    st.header("Predict Insurance Charges")
    st.write("Fill in the details below to predict charges:")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Prediction
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })
        prediction = model_pipeline.predict(input_data)
        st.success(f"The predicted insurance charge is ${prediction[0]:.2f}")
