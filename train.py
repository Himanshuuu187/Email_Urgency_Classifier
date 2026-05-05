import pandas as pd
import pickle
import json
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob

# ─────────────────────────────────────────────
# Load & Clean Data
# ─────────────────────────────────────────────
df = pd.read_csv("raw_emails.csv")

df.dropna(subset=["subject", "body", "label"], inplace=True)
df["subject"] = df["subject"].fillna("")
df["body"]    = df["body"].fillna("")
df["text"]    = df["subject"] + " " + df["body"]

# Ensure optional columns exist
if "sender" not in df.columns:
    df["sender"] = "unknown"
if "date" not in df.columns:
    df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# Sentiment Enrichment (for data exploration)
# ─────────────────────────────────────────────
print("Computing sentiment on training data...")

def get_polarity(text):
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 3)
    except Exception:
        return 0.0

def get_subjectivity(text):
    try:
        return round(TextBlob(str(text)).sentiment.subjectivity, 3)
    except Exception:
        return 0.0

df["polarity"]     = df["body"].apply(get_polarity)
df["subjectivity"] = df["body"].apply(get_subjectivity)

print("\n=== Sentiment Stats by Urgency Label ===")
print(df.groupby("label")[["polarity", "subjectivity"]].mean().round(3))

# ─────────────────────────────────────────────
# TF-IDF + Model Training
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print(cm_df)

# ─────────────────────────────────────────────
# Save Artifacts
# ─────────────────────────────────────────────
pickle.dump(model,      open("model.pkl",      "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

model_info = {
    "classes":      list(model.classes_),
    "trained_at":   datetime.now().isoformat(),
    "num_samples":  len(df),
    "features":     vectorizer.max_features,
    "ngram_range":  list(vectorizer.ngram_range),
    "avg_polarity_by_label": df.groupby("label")["polarity"].mean().round(3).to_dict(),
}
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("\n✅ Model trained and saved!")
print(f"   Labels   : {list(model.classes_)}")
print(f"   Samples  : {len(df)}")
print(f"   Features : {vectorizer.max_features}")
