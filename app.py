import streamlit as st
import pandas as pd
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Remove punctuation and stopwords from text."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load dataset (for demonstration, replace with actual dataset)
@st.cache_data
def load_data():
    df = pd.read_csv('/content/twitter_x_y_train.csv')  # Replace with actual dataset path
    df['text_clean'] = df['text'].apply(preprocess_text)
    return df

# Load data
train_df = load_data()
x = train_df['text_clean']
y = train_df['airline_sentiment']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Vectorization
CV = CountVectorizer(stop_words='english')
x_train_vec = CV.fit_transform(x_train)
x_test_vec = CV.transform(x_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

def predict_sentiment(text):
    """Predict sentiment for a given text."""
    text_clean = preprocess_text(text)
    text_vec = CV.transform([text_clean])
    prediction = model.predict(text_vec)
    return le.inverse_transform(prediction)[0]

# Streamlit App
st.title("Twitter Sentiment Analysis")

# Option 1: Single Comment Analysis
st.header("Analyze a Single Comment")
user_input = st.text_area("Enter a comment:")
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a comment.")

# Option 2: Upload Excel File
st.header("Upload an Excel File")
uploaded_file = st.file_uploader("Upload an Excel file with a 'text' column", type=["xls", "xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(preprocess_text)
        df_vec = CV.transform(df['text_clean'])
        df['Sentiment'] = le.inverse_transform(model.predict(df_vec))
        st.write(df[['text', 'Sentiment']])
    else:
        st.error("Excel file must have a 'text' column.")
