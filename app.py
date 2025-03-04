import joblib
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load pretrained model, vectorizer, and label encoder
model = joblib.load("sentiment_model (1).pkl")
CV = joblib.load("vectorizer (1).pkl")
le = joblib.load("label_encoder (1).pkl")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to predict sentiment
def predict_sentiment(text):
    text_clean = preprocess_text(text)
    text_vec = CV.transform([text_clean])
    prediction = model.predict(text_vec)
    return le.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Twitter Sentiment Analysis with Pretrained Model")
st.write("Enter a comment or upload an Excel file to analyze sentiment.")

# Input for a single comment
user_input = st.text_area("Enter a comment:")
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a comment.")

# Upload Excel file for batch sentiment analysis
uploaded_file = st.file_uploader("Upload an Excel file with a 'text' column", type=["xls", "xlsx","csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(preprocess_text)
        df_vec = CV.transform(df['text_clean'])
        df['Sentiment'] = le.inverse_transform(model.predict(df_vec))
        st.write(df[['text', 'Sentiment']])
    else:
        st.error("Excel file must have a 'text' column.")
