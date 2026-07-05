import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_word = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()


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


# Load saved model + vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# User input
msg = input("Enter message: ")

processed_msg = preprocess_text(msg)

vector = tfidf.transform([processed_msg])

prediction = model.predict(vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")