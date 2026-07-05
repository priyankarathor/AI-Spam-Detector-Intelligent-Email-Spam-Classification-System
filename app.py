import os
import pickle
import nltk

from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# Download NLTK Data (Only if missing)
# -----------------------------
NLTK_DATA = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA, exist_ok=True)

nltk.data.path.append(NLTK_DATA)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="AI Spam Detector API",
    version="1.0.0"
)

# -----------------------------
# NLP Objects
# -----------------------------
stop_word = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()

# -----------------------------
# Load Model
# -----------------------------
try:
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    print("Error Loading Model:", e)
    raise e

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_text(text):

    tokens = word_tokenize(str(text).lower())

    filtered_words = [
        word for word in tokens
        if word.isalnum() and word not in stop_word
    ]

    stemmed_words = [
        porter_stemmer.stem(word)
        for word in filtered_words
    ]

    return " ".join(stemmed_words)

# -----------------------------
# Request Body
# -----------------------------
class MessageRequest(BaseModel):
    message: str

# -----------------------------
# Home API
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "AI Spam Detection API is Live Now"
    }

# Prediction API
# -----------------------------
@app.post("/predict")
def predict_spam(data: MessageRequest):

    processed_msg = preprocess_text(data.message)

    vector = tfidf.transform([processed_msg])

    prediction = model.predict(vector)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return {
        "input_message": data.message,
        "processed_message": processed_msg,
        "prediction": result
    }