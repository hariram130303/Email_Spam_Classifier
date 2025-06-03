import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Load and train models
def train_models():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(url, header=None, names=["label", "message"])
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    df['message_clean'] = df.message.apply(clean_text)

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
    X = tfidf.fit_transform(df['message_clean'])
    y = df['label_num']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_model = MultinomialNB().fit(X_train, y_train)
    svm_model = LinearSVC().fit(X_train, y_train)

    return tfidf, nb_model, svm_model

# Predict using a given model
def predict(message, model, tfidf):
    clean_msg = clean_text(message)
    vector = tfidf.transform([clean_msg])
    pred = model.predict(vector)[0]
    return "Spam" if pred == 1 else "Ham"
