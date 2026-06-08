import os
import json
import base64
import re
import tempfile
from email import message_from_bytes
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate_gmail():
    """
    Authenticate with Gmail API.
    
    Priority order:
      1. st.secrets["GMAIL_TOKEN"]  — for Streamlit Cloud deployment
      2. token.json file            — for local development
      3. credentials.json OAuth flow — for first-time local setup
    
    If the token is expired, it is automatically refreshed and saved back.
    """
    creds = None

    # ── 1. Try Streamlit secrets (Cloud deployment) ──────────────────
    try:
        import streamlit as st
        if "GMAIL_TOKEN" in st.secrets:
            raw = st.secrets["GMAIL_TOKEN"]
            token_data = json.loads(raw) if isinstance(raw, str) else dict(raw)

            creds = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes", SCOPES),
            )
    except Exception:
        pass  # Not running in Streamlit or secret not set — fall through

    # ── 2. Try local token.json ───────────────────────────────────────
    if creds is None and os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # ── 3. Refresh expired token ──────────────────────────────────────
    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed token back to token.json (local only)
                if os.path.exists("token.json"):
                    with open("token.json", "w") as f:
                        f.write(creds.to_json())
            except Exception as e:
                raise RuntimeError(
                    f"Token refresh failed: {e}\n\n"
                    "Your refresh token has been revoked or expired.\n"
                    "Run this locally to get a new token:\n"
                    "  python refresh_token.py\n"
                    "Then update the GMAIL_TOKEN secret on Streamlit Cloud."
                )
        else:
            raise RuntimeError(
                "No valid credentials and no refresh token available.\n"
                "Run 'python refresh_token.py' locally to re-authenticate."
            )

    # ── 4. First-time local OAuth flow (no token at all) ─────────────
    if creds is None:
        if not os.path.exists("credentials.json"):
            raise FileNotFoundError(
                "credentials.json not found. "
                "Download it from Google Cloud Console → APIs & Services → Credentials."
            )
        flow  = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ── Text helpers ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_body(email_msg) -> str:
    body = ""
    if email_msg.is_multipart():
        for part in email_msg.walk():
            ct   = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in disp:
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
                except Exception:
                    continue
    else:
        try:
            body = email_msg.get_payload(decode=True).decode(errors="ignore")
        except Exception:
            body = ""
    return clean_text(body)


def parse_sender(raw_from: str) -> dict:
    if not raw_from:
        return {"name": "Unknown", "email": "unknown@unknown.com"}
    match = re.match(r'^(.*?)\s*<(.+?)>$', raw_from.strip())
    if match:
        name  = match.group(1).strip().strip('"')
        email = match.group(2).strip()
    else:
        name  = raw_from.strip()
        email = raw_from.strip()
    return {"name": name or email, "email": email}


def parse_date(raw_date: str) -> str:
    if not raw_date:
        return None
    try:
        return parsedate_to_datetime(raw_date).strftime("%Y-%m-%d")
    except Exception:
        return None


# ── Main fetch function ───────────────────────────────────────────────

def get_emails(max_results: int = 50, query: str = "is:unread") -> list:
    """
    Fetch emails from Gmail.
    Returns list of dicts: subject, body, snippet, sender_name,
                           sender_email, date, message_id.
    """
    service = authenticate_gmail()

    try:
        results  = service.users().messages().list(
            userId="me", maxResults=max_results, q=query
        ).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to list emails: {e}")

    messages = results.get("messages", [])
    emails   = []

    for msg in messages:
        try:
            msg_data  = service.users().messages().get(
                userId="me", id=msg["id"], format="raw"
            ).execute()
            raw_data  = base64.urlsafe_b64decode(msg_data["raw"])
            email_msg = message_from_bytes(raw_data)

            subject     = clean_text(email_msg.get("subject", "(No Subject)"))
            body        = extract_body(email_msg)
            snippet     = body[:200] + "..." if len(body) > 200 else body
            sender_info = parse_sender(email_msg.get("from", ""))
            date_str    = parse_date(email_msg.get("date", ""))

            emails.append({
                "subject":      subject,
                "body":         body,
                "snippet":      snippet,
                "sender_name":  sender_info["name"],
                "sender_email": sender_info["email"],
                "date":         date_str,
                "message_id":   msg["id"],
            })
        except Exception as e:
            print(f"⚠️  Skipping email {msg['id']}: {e}")
            continue

    return emails
