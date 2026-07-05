import os
import pickle
import nltk

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ==========================================================
# Download NLTK Resources
# ==========================================================

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

# For latest NLTK versions
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", download_dir=NLTK_DATA)
    except:
        pass


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="AI Spam Detector API",
    version="1.0.0"
)

# ==========================================================
# CORS
# ==========================================================

origins = [

    # Local Development
    "http://localhost:3000",
    "http://127.0.0.1:3000",

    # Replace with your Vercel URL after deployment
    "https://your-project.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# NLP
# ==========================================================

stop_word = set(stopwords.words("english"))

porter_stemmer = PorterStemmer()

# ==========================================================
# Load Model
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:

    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model Loaded Successfully")

except Exception as e:
    print("Model Loading Error:", e)
    raise e


# ==========================================================
# Text Preprocessing
# ==========================================================

def preprocess_text(text):

    tokens = word_tokenize(str(text).lower())

    filtered_words = [

        word

        for word in tokens

        if word.isalnum() and word not in stop_word

    ]

    stemmed_words = [

        porter_stemmer.stem(word)

        for word in filtered_words

    ]

    return " ".join(stemmed_words)


# ==========================================================
# Request Model
# ==========================================================

class MessageRequest(BaseModel):
    message: str


# ==========================================================
# Home API
# ==========================================================

@app.get("/")
def home():

    return {

        "status": "success",

        "message": "AI Spam Detector API Running",

        "version": "1.0.0"

    }


# ==========================================================
# Health API
# ==========================================================

@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


# ==========================================================
# Prediction API
# ==========================================================

@app.post("/predict")
def predict_spam(data: MessageRequest):

    try:

        processed_msg = preprocess_text(data.message)

        vector = tfidf.transform([processed_msg])

        prediction = model.predict(vector)[0]

        result = "Spam" if prediction == 1 else "Not Spam"

        return {

            "input_message": data.message,

            "processed_message": processed_msg,

            "prediction": result

        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)

        }