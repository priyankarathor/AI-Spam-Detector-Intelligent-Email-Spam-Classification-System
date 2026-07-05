import pandas as pd
import nltk
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Run only first time
# nltk.download('punkt')
# nltk.download('stopwords')

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


# Load Dataset
# ==========================
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

df = df.iloc[:, :2]
df.columns = ['label', 'message']

print(df.head())

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess Dataset
# ==========================
df['processed_message'] = df['message'].apply(preprocess_text)

# Vectorization
# ==========================
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['processed_message'])
y = df['label']

# ==========================
# Train Model
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save Model
# ==========================
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")

print("Vectorizer size:", os.path.getsize("vectorizer.pkl"))
print("Model size:", os.path.getsize("model.pkl"))