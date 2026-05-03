import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv("raw_emails.csv")
df["text"] = df["subject"] + " " + df["body"]

# TF-IDF with tuning
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Better model for text
model = MultinomialNB()
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Advanced model trained!")