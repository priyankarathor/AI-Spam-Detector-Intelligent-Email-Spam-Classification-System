import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = FastAPI()
# //stop words details
stop_word = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


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


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Spam Detection API Running"
    }


@app.post("/predict")
def predict_spam(data: MessageRequest):
    processed_msg = preprocess_text(data.message)

    vector = tfidf.transform([processed_msg])

    prediction = model.predict(vector)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return {
        "message": data.message,
        "prediction": result
    }