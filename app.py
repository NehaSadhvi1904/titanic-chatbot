import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
import threading
import uvicorn

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Initialize FastAPI app
app = FastAPI()

# CORS settings for Streamlit-FastAPI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI Endpoint for Titanic Analysis
@app.get("/query")
def query_titanic(question: str):
    question = question.lower()

    if "percentage of passengers were male" in question:
        male_percentage = (df['sex'].value_counts(normalize=True)['male']) * 100
        return {"response": f"{male_percentage:.2f}% of the passengers were male."}

    elif "histogram of passenger ages" in question:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['age'].dropna(), bins=20, kde=True)
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.title("Histogram of Passenger Ages")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        base64_img = base64.b64encode(img.getvalue()).decode()
        return {"response": "Here is the histogram of passenger ages.", "image": base64_img}

    elif "average ticket fare" in question:
        avg_fare = df["fare"].mean()
        return {"response": f"The average ticket fare was ${avg_fare:.2f}"}

    elif "how many passengers embarked from each port" in question:
        embark_counts = df["embark_town"].value_counts().to_dict()
        return {"response": f"Number of passengers per port: {embark_counts}"}

    elif "survival rate" in question:
        survival_rate = (df["survived"].mean()) * 100
        return {"response": f"The overall survival rate was {survival_rate:.2f}%."}

    elif "survival rate for each class" in question:
        class_survival = df.groupby("pclass")["survived"].mean() * 100
        return {"response": f"Survival rate per class: {class_survival.to_dict()}"}

    elif "percentage of males and females survived" in question:
        gender_survival = df.groupby("sex")["survived"].mean() * 100
        return {"response": f"Survival rate by gender: {gender_survival.to_dict()}"}

    elif "boxplot of fares" in question:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df["fare"])
        plt.title("Boxplot of Ticket Fares")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        base64_img = base64.b64encode(img.getvalue()).decode()
        return {"response": "Here is the boxplot of ticket fares.", "image": base64_img}

    elif "how many passengers were in each age group" in question:
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"])
        age_distribution = df["age_group"].value_counts().to_dict()
        return {"response": f"Passenger count by age group: {age_distribution}"}

    return {"response": "I couldn't understand your query. Please try again."}

# Start FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

# ----------- STREAMLIT FRONTEND -----------
st.title("🚢 Titanic Dataset Chatbot")
st.write("Ask me anything about the Titanic dataset!")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input:
        response = query_titanic(user_input)
        st.write(response["response"])

        if "image" in response:
            image_data = base64.b64decode(response["image"])
            st.image(image_data, caption="Generated Visualization", use_column_width=True)
