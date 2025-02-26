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
