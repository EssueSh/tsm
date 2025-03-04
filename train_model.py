import pandas as pd
import numpy as np
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset (Make sure to have a CSV file with 'text' and 'airline_sentiment' columns)
df = pd.read_csv("/content/twitter_x_y_train.csv")  # Change to your dataset path

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
df['text_clean'] = df['text'].apply(preprocess_text)

# Encode target labels
le = LabelEncoder()
df['airline_sentiment'] = le.fit_transform(df['airline_sentiment'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(df['text_clean'], df['airline_sentiment'], test_size=0.2, random_state=40)

# Convert text to numerical data
CV = CountVectorizer(stop_words='english')
x_train_vec = CV.fit_transform(x_train)
x_test_vec = CV.transform(x_test)

# Train model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# Evaluate model
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model, vectorizer, and label encoder
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(CV, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model and vectorizer saved successfully!")
