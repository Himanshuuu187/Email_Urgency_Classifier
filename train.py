import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Load data
df = pd.read_csv("raw_emails.csv")

# Basic cleaning
df.dropna(subset=["subject", "body", "label"], inplace=True)
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

# Ensure sender column exists (used in analytics)
if "sender" not in df.columns:
    df["sender"] = "unknown"

if "date" not in df.columns:
    df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

# TF-IDF Vectorizer with tuned params
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,       # Apply log normalization to TF
    strip_accents="unicode",
    analyzer="word",
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train / test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = MultinomialNB(alpha=0.1)   # Lower alpha = less smoothing, better for tuned TF-IDF
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save label classes for reference
label_info = {
    "classes": list(model.classes_),
    "trained_at": datetime.now().isoformat(),
    "num_samples": len(df),
    "features": vectorizer.max_features
}
with open("model_info.json", "w") as f:
    json.dump(label_info, f, indent=2)

print("\n✅ Model trained and saved!")
print(f"   Labels: {list(model.classes_)}")
print(f"   Samples: {len(df)}")
