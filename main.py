# User Input → Cleaning → Tokenization → Stopword Removal → Stemming/Lemmatization → Vectorization → Prediction
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources (Run once)
# nltk.download('punkt')
# nltk.download('stopwords')

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

# Keep only first 2 columns
df = df.iloc[:, :2]

# Rename columns
df.columns = ['label', 'message']

# print(df.head())

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ==========================
# Preprocessing
# ==========================
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

# Apply preprocessing on dataset
df['processed_message'] = df['message'].apply(preprocess_text)

# print(df.head())

# ==========================
# Vectorization
# ==========================
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['processed_message'])
y = df['label']

# ==========================
# Train Model
# ==========================
model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully!")

# ==========================
# User Input
# ==========================
msg = input("Enter message: ")

processed_msg = preprocess_text(msg)

vector = tfidf.transform([processed_msg])

prediction = model.predict(vector)

# ==========================
# Result
# ==========================
if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")