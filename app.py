import joblib
import streamlit as st
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define strong negative words that should always return "Negative"
strong_negative_words = {"kill", "murder", "suicide", "bomb", "attack", "assault", "rape", "terrorist"}

# Load pretrained model, vectorizer, and label encoder
model = joblib.load("sentiment_model (1).pkl")
CV = joblib.load("vectorizer (1).pkl")
le = joblib.load("label_encoder (1).pkl")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()
    
    # If any strong negative word is present, return "Negative" directly
    if any(word in strong_negative_words for word in words):
        return "FORCE_NEGATIVE"

    text = " ".join([word for word in words if word not in stop_words])  # Remove stopwords
    return text

# Function to predict sentiment
def predict_sentiment(text):
    text_clean = preprocess_text(text)
    
    # If strong negative word is found, return "Negative"
    if text_clean == "FORCE_NEGATIVE":
        return "Negative"
    
    text_vec = CV.transform([text_clean])
    prediction = model.predict(text_vec)
    return le.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Twitter Sentiment Analysis with Pretrained Model")
st.write("Enter a comment or upload an Excel/CSV file to analyze sentiment.")

# Input for a single comment
user_input = st.text_area("Enter a comment:")
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a comment.")

# Upload Excel/CSV file for batch sentiment analysis
uploaded_file = st.file_uploader("Upload an Excel or CSV file with a 'text' column", type=["xls", "xlsx", "csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(preprocess_text)
        
        # Apply sentiment prediction including the force-negative check
        df['Sentiment'] = df['text_clean'].apply(lambda x: "Negative" if x == "FORCE_NEGATIVE" else le.inverse_transform(model.predict(CV.transform([x])))[0])

        # Display data with sentiment
        st.write(df[['text', 'Sentiment']])
        
        # Count sentiment categories
        sentiment_counts = df['Sentiment'].value_counts()
        
        # Plot Pie Chart
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%', 
            colors=['red', 'yellow', 'green'],  # Customize colors for negative, neutral, positive
            startangle=90
        )
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    else:
        st.error("Uploaded file must have a 'text' column.")
