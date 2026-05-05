import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer as SKTfidf

from gmail_fetch import get_emails

# ─────────────────────────────────────────────
# Config & Setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gmail AI Urgency Assistant",
    layout="wide",
    page_icon="📧"
)

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()


# ─────────────────────────────────────────────
# Keyword Highlighter
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_keywords(text: str, top_n: int = 7) -> list:
    """Extract top N keywords from text using TF-IDF scoring."""
    if not text or len(text.strip()) < 10:
        return []
    try:
        tfidf = SKTfidf(
            max_features=100,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        tfidf.fit_transform([text])
        scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
        sorted_kws = sorted(scores.items(), key=lambda x: x[1])
        return [kw for kw, _ in sorted_kws[:top_n]]
    except Exception:
        return []


def render_keywords(keywords: list):
    """Render keywords as styled pill badges."""
    if not keywords:
        st.caption("No keywords extracted.")
        return
    badge_html = " ".join(
        f'<span style="background:#1e3a5f;color:#7dd3fc;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78rem;margin:2px;display:inline-block;">'
        f'{kw}</span>'
        for kw in keywords
    )
    st.markdown(badge_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sentiment Detector
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def analyze_sentiment(text: str) -> dict:
    """
    Analyze tone using TextBlob.
    Returns label, polarity (-1 to +1), subjectivity (0 to 1), and emoji.
    """
    if not text or len(text.strip()) < 5:
        return {"label": "Neutral", "emoji": "😐", "polarity": 0.0, "subjectivity": 0.0}

    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    if polarity >= 0.2:
        label, emoji = "Positive", "😊"
    elif polarity <= -0.2:
        label, emoji = "Negative", "😠"
    else:
        label, emoji = "Neutral", "😐"

    return {"label": label, "emoji": emoji, "polarity": polarity, "subjectivity": subjectivity}


def sentiment_color(label: str) -> str:
    return {"Positive": "green", "Negative": "red", "Neutral": "gray"}.get(label, "gray")


# ─────────────────────────────────────────────
# Email Classification
# ─────────────────────────────────────────────
def classify_emails(emails: list) -> pd.DataFrame:
    results = []
    for email in emails:
        text   = email["subject"] + " " + email["body"]
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = round(max(model.predict_proba(vector)[0]), 2)
        sentiment  = analyze_sentiment(email.get("body", ""))

        results.append({
            "Subject":      email["subject"],
            "Sender":       email.get("sender_name", "Unknown"),
            "Sender Email": email.get("sender_email", ""),
            "Date":         email.get("date", ""),
            "Snippet":      email.get("snippet", ""),
            "Body":         email.get("body", ""),
            "Urgency":      prediction,
            "Confidence":   confidence,
            "Sentiment":    sentiment["label"],
            "Polarity":     sentiment["polarity"],
            "Subjectivity": sentiment["subjectivity"],
            "Message ID":   email.get("message_id", ""),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
URGENCY_COLOR = {"high": "🔴", "medium": "🟡", "low": "🟢"}
URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}
COLOR_MAP     = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}
SENT_COLORS   = {"Positive": "#10B981", "Neutral": "#6B7280", "Negative": "#EF4444"}


def urgency_badge(level: str) -> str:
    return URGENCY_COLOR.get(level, "⚪") + " " + level.upper()


# ─────────────────────────────────────────────
# Email Card
# ─────────────────────────────────────────────
def render_email_card(row, idx):
    """Expandable email card with keywords and sentiment analysis."""
    badge     = urgency_badge(row["Urgency"])
    conf_color = "green" if row["Confidence"] >= 0.8 else "orange" if row["Confidence"] >= 0.6 else "red"
    sentiment  = analyze_sentiment(row["Body"])
    s_color    = sentiment_color(sentiment["label"])

    with st.expander(
        f"{badge}  |  {sentiment['emoji']}  |  {row['Subject']}  —  {row['Sender']}  ({row['Date']})"
    ):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**From:** {row['Sender']} `<{row['Sender Email']}>`")
            st.markdown(f"**Date:** {row['Date']}")
            st.markdown(f"**Preview:** {row['Snippet']}")

        with col2:
            st.metric("Urgency", row["Urgency"].upper())
            st.markdown(f"**Confidence:** :{conf_color}[{row['Confidence']}]")
            st.markdown(f"**Tone:** :{s_color}[{sentiment['emoji']} {sentiment['label']}]")
            st.caption(f"Polarity: {sentiment['polarity']:+.2f}  |  Subjectivity: {sentiment['subjectivity']:.2f}")

        st.divider()

        # Keywords
        st.markdown("**🔑 Top Keywords**")
        keywords = extract_keywords(row["Subject"] + " " + row["Body"])
        render_keywords(keywords)

        st.markdown("")

        # Sentiment bars
        st.markdown("**🎭 Sentiment Detail**")
        scol1, scol2 = st.columns(2)
        with scol1:
            st.caption("Polarity  (Negative ← 0 → Positive)")
            norm = int((sentiment["polarity"] + 1) / 2 * 100)
            st.progress(norm, text=f"{sentiment['polarity']:+.2f}")
        with scol2:
            st.caption("Subjectivity  (Objective → Subjective)")
            st.progress(int(sentiment["subjectivity"] * 100), text=f"{sentiment['subjectivity']:.2f}")


# ─────────────────────────────────────────────
# Analytics Dashboard
# ─────────────────────────────────────────────
def render_analytics(df: pd.DataFrame):
    st.subheader("📊 Analytics Dashboard")

    total        = len(df)
    high_count   = len(df[df["Urgency"] == "high"])
    medium_count = len(df[df["Urgency"] == "medium"])
    low_count    = len(df[df["Urgency"] == "low"])
    avg_conf     = df["Confidence"].mean()
    neg_pct      = round(len(df[df["Sentiment"] == "Negative"]) / total * 100) if total else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📬 Total",          total)
    k2.metric("🔴 High",           high_count)
    k3.metric("🟡 Medium",         medium_count)
    k4.metric("🟢 Low",            low_count)
    k5.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")
    k6.metric("😠 Negative Tone",  f"{neg_pct}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Urgency Distribution")
        uc = df["Urgency"].value_counts().reset_index()
        uc.columns = ["Urgency", "Count"]
        fig_pie = px.pie(uc, names="Urgency", values="Count",
                         color="Urgency", color_discrete_map=COLOR_MAP, hole=0.4)
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("#### Sentiment Distribution")
        sc = df["Sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment", "Count"]
        fig_sent = px.pie(sc, names="Sentiment", values="Count",
                          color="Sentiment", color_discrete_map=SENT_COLORS, hole=0.4)
        fig_sent.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_sent, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Sentiment × Urgency Heatmap")
        heat = (df.groupby(["Urgency", "Sentiment"]).size().reset_index(name="Count"))
        fig_heat = px.density_heatmap(
            heat, x="Urgency", y="Sentiment", z="Count",
            color_continuous_scale="Reds",
            category_orders={"Urgency": ["high", "medium", "low"],
                             "Sentiment": ["Negative", "Neutral", "Positive"]}
        )
        fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col4:
        st.markdown("#### Polarity Score by Urgency")
        fig_box = px.box(
            df, x="Urgency", y="Polarity", color="Urgency",
            color_discrete_map=COLOR_MAP,
            category_orders={"Urgency": ["high", "medium", "low"]},
            points="all"
        )
        fig_box.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Top Senders")
        top_senders = (
            df.groupby("Sender")
            .agg(Total=("Urgency", "count"), High=("Urgency", lambda x: (x == "high").sum()))
            .sort_values("Total", ascending=False).head(10).reset_index()
        )
        fig_bar = px.bar(
            top_senders, x="Total", y="Sender", orientation="h",
            color="High", color_continuous_scale="Reds",
            labels={"Total": "Total Emails", "High": "High Urgency"}
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col6:
        st.markdown("#### Urgency Over Time")
        df_dated = df[df["Date"].notna() & (df["Date"] != "")]
        if not df_dated.empty:
            timeline = (
                df_dated.groupby(["Date", "Urgency"]).size()
                .reset_index(name="Count").sort_values("Date")
            )
            fig_line = px.line(
                timeline, x="Date", y="Count", color="Urgency",
                color_discrete_map=COLOR_MAP, markers=True
            )
            fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Date information not available for the timeline chart.")


# ─────────────────────────────────────────────
# Filtering & Search Panel
# ─────────────────────────────────────────────
def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Filter & Search")

    keyword = st.sidebar.text_input("Search keyword (subject / sender)", "")

    urgency_options  = ["All"] + sorted(df["Urgency"].unique().tolist(), key=lambda x: URGENCY_ORDER.get(x, 9))
    urgency_filter   = st.sidebar.selectbox("Urgency Level", urgency_options)

    sentiment_filter = st.sidebar.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])

    senders       = ["All"] + sorted(df["Sender"].dropna().unique().tolist())
    sender_filter = st.sidebar.selectbox("Sender", senders)

    min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    dated = df[df["Date"].notna() & (df["Date"] != "")]
    if not dated.empty:
        try:
            df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
            min_date  = df["_date_parsed"].min().date()
            max_date  = df["_date_parsed"].max().date()
            date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
            if len(date_range) == 2:
                df = df[
                    (df["_date_parsed"].dt.date >= date_range[0]) &
                    (df["_date_parsed"].dt.date <= date_range[1])
                ]
        except Exception:
            pass

    filtered = df.copy()

    if keyword:
        kw = keyword.lower()
        filtered = filtered[
            filtered["Subject"].str.lower().str.contains(kw, na=False) |
            filtered["Sender"].str.lower().str.contains(kw, na=False) |
            filtered["Snippet"].str.lower().str.contains(kw, na=False)
        ]

    if urgency_filter != "All":
        filtered = filtered[filtered["Urgency"] == urgency_filter]

    if sentiment_filter != "All":
        filtered = filtered[filtered["Sentiment"] == sentiment_filter]

    if sender_filter != "All":
        filtered = filtered[filtered["Sender"] == sender_filter]

    filtered = filtered[filtered["Confidence"] >= min_conf]

    st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)} emails**")
    return filtered


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
st.title("📧 Gmail AI Urgency Assistant")
st.caption("Classify urgency, extract keywords, and detect sentiment — 100% free, no API needed.")

tab_inbox, tab_analytics, tab_table = st.tabs(["📬 Priority Inbox", "📊 Analytics", "📋 All Emails"])

st.sidebar.header("⚙️ Settings")
max_emails  = st.sidebar.number_input("Max emails to fetch", min_value=10, max_value=200, value=50, step=10)
gmail_query = st.sidebar.text_input("Gmail search query", value="is:unread")
fetch_btn   = st.sidebar.button("🔄 Fetch Emails", type="primary", use_container_width=True)

if "email_df" not in st.session_state:
    st.session_state["email_df"] = None

if fetch_btn:
    with st.spinner("Fetching and classifying emails..."):
        try:
            emails = get_emails(max_results=int(max_emails), query=gmail_query)
            if emails:
                st.session_state["email_df"] = classify_emails(emails)
                st.sidebar.success(f"✅ Fetched {len(emails)} emails!")
            else:
                st.sidebar.warning("No emails found matching the query.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

df = st.session_state["email_df"]

if df is not None and not df.empty:

    filtered_df = render_filters(df)

    with tab_inbox:
        st.subheader("📬 Priority Inbox")
        for urgency_level in ["high", "medium", "low"]:
            level_df = filtered_df[filtered_df["Urgency"] == urgency_level]
            if level_df.empty:
                continue
            badge = URGENCY_COLOR[urgency_level]
            st.markdown(f"### {badge} {urgency_level.upper()} ({len(level_df)})")
            for idx, (_, row) in enumerate(level_df.iterrows()):
                render_email_card(row, f"{urgency_level}_{idx}")
            st.divider()

    with tab_analytics:
        render_analytics(filtered_df)

    with tab_table:
        st.subheader("📋 All Emails")
        display_cols = ["Subject", "Sender", "Date", "Urgency", "Confidence", "Sentiment", "Polarity", "Snippet"]
        st.dataframe(
            filtered_df[display_cols].sort_values("Urgency", key=lambda x: x.map(URGENCY_ORDER)),
            use_container_width=True,
            height=500
        )
        csv = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", data=csv,
            file_name=f"gmail_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Click **Fetch Emails** in the sidebar to get started.")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Fetch** — Pulls unread emails from Gmail via the Gmail API.
        2. **Classify** — ML model (TF-IDF + Naive Bayes) tags each email as High / Medium / Low urgency.
        3. **Keywords** — Top keywords extracted per email using TF-IDF scoring — shown as pill badges.
        4. **Sentiment** — TextBlob detects tone: Positive 😊 / Neutral 😐 / Negative 😠 with polarity and subjectivity scores.
        5. **Filter** — Sidebar: filter by urgency, sentiment, sender, date range, and confidence threshold.
        6. **Analyze** — Analytics tab: urgency + sentiment charts, heatmap, polarity box plots, top senders, and timeline.
        """)
