import streamlit as st
import pickle
import pandas as pd

from gmail_fetch import get_emails

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(layout="wide")
st.title("📧 Gmail AI Urgency Assistant")

# Button to fetch emails
if st.button("Fetch Unread Emails"):

    emails = get_emails()

    results = []

    for email in emails:
        text = email["subject"] + " " + email["body"]

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])

        results.append({
            "Subject": email["subject"],
            "Urgency": prediction,
            "Confidence": round(confidence, 2)
        })

    df = pd.DataFrame(results)

    # 🔥 Priority Inbox View
    st.subheader("📬 Priority Inbox")

    high_df = df[df["Urgency"] == "high"]
    medium_df = df[df["Urgency"] == "medium"]
    low_df = df[df["Urgency"] == "low"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("🔴 HIGH")
        st.dataframe(high_df)

    with col2:
        st.warning("🟡 MEDIUM")
        st.dataframe(medium_df)

    with col3:
        st.success("🟢 LOW")
        st.dataframe(low_df)

    # 📊 Full table
    st.subheader("📊 All Emails")
    st.dataframe(df)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "gmail_results.csv")