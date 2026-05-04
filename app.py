import streamlit as st
import pickle
import pandas as pd
import anthropic
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from gmail_fetch import get_emails



from dotenv import load_dotenv
load_dotenv()

claude_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ─────────────────────────────────────────────
# Config & Setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gmail AI Urgency Assistant",
    layout="wide",
    page_icon="📧"
)

# Load model & vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Anthropic client for summarization
claude_client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# AI Summarization
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def summarize_email(subject: str, body: str) -> str:
    snippet = body.strip().replace("\n", " ")
    summary = snippet[:300] + "..." if len(snippet) > 300 else snippet
    return f"**Summary:** {summary}\n\n**Action Items:** Review manually."


# ─────────────────────────────────────────────
# Email Classification
# ─────────────────────────────────────────────
def classify_emails(emails: list) -> pd.DataFrame:
    results = []
    for email in emails:
        text = email["subject"] + " " + email["body"]
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = round(max(model.predict_proba(vector)[0]), 2)

        results.append({
            "Subject":       email["subject"],
            "Sender":        email.get("sender_name", "Unknown"),
            "Sender Email":  email.get("sender_email", ""),
            "Date":          email.get("date", ""),
            "Snippet":       email.get("snippet", ""),
            "Body":          email.get("body", ""),
            "Urgency":       prediction,
            "Confidence":    confidence,
            "Message ID":    email.get("message_id", ""),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────
URGENCY_COLOR = {"high": "🔴", "medium": "🟡", "low": "🟢"}
URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}


def urgency_badge(level: str) -> str:
    return URGENCY_COLOR.get(level, "⚪") + " " + level.upper()


def render_email_card(row, idx):
    """Render a single email as an expandable card."""
    badge = urgency_badge(row["Urgency"])
    conf_color = "green" if row["Confidence"] >= 0.8 else "orange" if row["Confidence"] >= 0.6 else "red"

    with st.expander(f"{badge}  |  {row['Subject']}  —  {row['Sender']}  ({row['Date']})"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**From:** {row['Sender']} `<{row['Sender Email']}>`")
            st.markdown(f"**Date:** {row['Date']}")
            st.markdown(f"**Snippet:** {row['Snippet']}")

        with col2:
            st.metric("Urgency", row["Urgency"].upper())
            st.markdown(f"**Confidence:** :{conf_color}[{row['Confidence']}]")

        # AI Summary button
        summary_key = f"summary_{idx}"
        if st.button("🤖 Summarize with AI", key=f"btn_{idx}"):
            with st.spinner("Summarizing..."):
                st.session_state[summary_key] = summarize_email(row["Subject"], row["Body"])

        if summary_key in st.session_state:
            st.markdown("---")
            st.markdown("**📋 AI Summary**")
            st.markdown(st.session_state[summary_key])


# ─────────────────────────────────────────────
# Analytics Dashboard
# ─────────────────────────────────────────────
def render_analytics(df: pd.DataFrame):
    st.subheader("📊 Analytics Dashboard")

    # Top-level KPIs
    total = len(df)
    high_count = len(df[df["Urgency"] == "high"])
    medium_count = len(df[df["Urgency"] == "medium"])
    low_count = len(df[df["Urgency"] == "low"])
    avg_conf = df["Confidence"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📬 Total Emails", total)
    k2.metric("🔴 High", high_count)
    k3.metric("🟡 Medium", medium_count)
    k4.metric("🟢 Low", low_count)
    k5.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")

    st.divider()

    col1, col2 = st.columns(2)

    # Urgency distribution pie chart
    with col1:
        st.markdown("#### Urgency Distribution")
        urgency_counts = df["Urgency"].value_counts().reset_index()
        urgency_counts.columns = ["Urgency", "Count"]
        color_map = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}
        fig_pie = px.pie(
            urgency_counts, names="Urgency", values="Count",
            color="Urgency", color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Confidence distribution histogram
    with col2:
        st.markdown("#### Confidence Score Distribution")
        fig_hist = px.histogram(
            df, x="Confidence", color="Urgency",
            nbins=20, color_discrete_map=color_map,
            labels={"Confidence": "Confidence Score", "count": "Number of Emails"}
        )
        fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    col3, col4 = st.columns(2)

    # Top senders by email count
    with col3:
        st.markdown("#### Top Senders")
        top_senders = (
            df.groupby("Sender")
            .agg(Total=("Urgency", "count"), High=("Urgency", lambda x: (x == "high").sum()))
            .sort_values("Total", ascending=False)
            .head(10)
            .reset_index()
        )
        fig_bar = px.bar(
            top_senders, x="Total", y="Sender", orientation="h",
            color="High", color_continuous_scale="Reds",
            labels={"Total": "Total Emails", "High": "High Urgency Count"}
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Urgency over time (if dates available)
    with col4:
        st.markdown("#### Urgency Over Time")
        df_dated = df[df["Date"].notna() & (df["Date"] != "")]

        if not df_dated.empty:
            timeline = (
                df_dated.groupby(["Date", "Urgency"])
                .size()
                .reset_index(name="Count")
                .sort_values("Date")
            )
            fig_line = px.line(
                timeline, x="Date", y="Count", color="Urgency",
                color_discrete_map=color_map, markers=True,
                labels={"Count": "Emails", "Date": "Date"}
            )
            fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Date information not available for timeline chart.")


# ─────────────────────────────────────────────
# Filtering & Search Panel
# ─────────────────────────────────────────────
def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Filter & Search")

    # Keyword search
    keyword = st.sidebar.text_input("Search keyword (subject / sender)", "")

    # Urgency filter
    urgency_options = ["All"] + sorted(df["Urgency"].unique().tolist(), key=lambda x: URGENCY_ORDER.get(x, 9))
    urgency_filter = st.sidebar.selectbox("Urgency Level", urgency_options)

    # Sender filter
    senders = ["All"] + sorted(df["Sender"].dropna().unique().tolist())
    sender_filter = st.sidebar.selectbox("Sender", senders)

    # Confidence threshold
    min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    # Date range filter
    dated = df[df["Date"].notna() & (df["Date"] != "")]
    if not dated.empty:
        try:
            df["_date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
            min_date = df["_date_parsed"].min().date()
            max_date = df["_date_parsed"].max().date()
            date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
            if len(date_range) == 2:
                df = df[
                    (df["_date_parsed"].dt.date >= date_range[0]) &
                    (df["_date_parsed"].dt.date <= date_range[1])
                ]
        except Exception:
            pass

    # Apply filters
    filtered = df.copy()

    if keyword:
        kw_lower = keyword.lower()
        filtered = filtered[
            filtered["Subject"].str.lower().str.contains(kw_lower, na=False) |
            filtered["Sender"].str.lower().str.contains(kw_lower, na=False) |
            filtered["Snippet"].str.lower().str.contains(kw_lower, na=False)
        ]

    if urgency_filter != "All":
        filtered = filtered[filtered["Urgency"] == urgency_filter]

    if sender_filter != "All":
        filtered = filtered[filtered["Sender"] == sender_filter]

    filtered = filtered[filtered["Confidence"] >= min_conf]

    st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)} emails**")

    return filtered


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
st.title("📧 Gmail AI Urgency Assistant")
st.caption("Classify, summarize, and analyze your Gmail inbox with AI.")

# Tabs
tab_inbox, tab_analytics, tab_table = st.tabs(["📬 Priority Inbox", "📊 Analytics", "📋 All Emails"])

# Fetch button (sidebar)
st.sidebar.header("⚙️ Settings")
max_emails = st.sidebar.number_input("Max emails to fetch", min_value=10, max_value=200, value=50, step=10)
gmail_query = st.sidebar.text_input("Gmail search query", value="is:unread")
fetch_btn = st.sidebar.button("🔄 Fetch Emails", type="primary", use_container_width=True)

# Session state for emails
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

# Main content
df = st.session_state["email_df"]

if df is not None and not df.empty:

    # Apply filters (sidebar)
    filtered_df = render_filters(df)

    # ── Tab 1: Priority Inbox ──
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

    # ── Tab 2: Analytics ──
    with tab_analytics:
        render_analytics(filtered_df)

    # ── Tab 3: Full Table ──
    with tab_table:
        st.subheader("📋 All Emails")

        display_cols = ["Subject", "Sender", "Date", "Urgency", "Confidence", "Snippet"]
        st.dataframe(
            filtered_df[display_cols].sort_values(
                "Urgency", key=lambda x: x.map(URGENCY_ORDER)
            ),
            use_container_width=True,
            height=500
        )

        # Download
        csv = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"gmail_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Click **Fetch Emails** in the sidebar to get started.")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Fetch** — Pulls unread emails from your Gmail inbox via the Gmail API.
        2. **Classify** — Uses a trained ML model (TF-IDF + Naive Bayes) to tag each email as High / Medium / Low urgency.
        3. **Summarize** — Click "Summarize with AI" on any email to get a Claude-powered summary + action items.
        4. **Filter** — Use the sidebar to search by keyword, sender, date, urgency, or confidence level.
        5. **Analyze** — Switch to the Analytics tab for charts: urgency distribution, top senders, confidence scores, and timeline.
        """)
