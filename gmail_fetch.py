import os
import base64
import re
from email import message_from_bytes
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def authenticate_gmail():
    """Authenticate and return Gmail API service."""
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None  # Force re-login if refresh fails

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_body(email_msg) -> str:
    """Extract plain text body from email, handling multipart."""
    body = ""
    if email_msg.is_multipart():
        for part in email_msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in content_disposition:
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
    """Parse sender name and email from the From header."""
    if not raw_from:
        return {"name": "Unknown", "email": "unknown@unknown.com"}

    match = re.match(r'^(.*?)\s*<(.+?)>$', raw_from.strip())
    if match:
        name = match.group(1).strip().strip('"')
        email = match.group(2).strip()
    else:
        name = raw_from.strip()
        email = raw_from.strip()

    return {"name": name or email, "email": email}


def parse_date(raw_date: str) -> str:
    """Parse email date header into ISO format string."""
    if not raw_date:
        return None
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def get_emails(max_results: int = 50, query: str = "is:unread") -> list[dict]:
    """
    Fetch emails from Gmail.

    Args:
        max_results: Max number of emails to fetch (default 50).
        query: Gmail search query (default: unread emails).

    Returns:
        List of dicts with keys: subject, body, sender_name, sender_email, date, snippet.
    """
    service = authenticate_gmail()

    try:
        results = service.users().messages().list(
            userId='me',
            maxResults=max_results,
            q=query
        ).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to list emails: {e}")

    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        try:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='raw'
            ).execute()

            raw_data = base64.urlsafe_b64decode(msg_data['raw'])
            email_msg = message_from_bytes(raw_data)

            subject = clean_text(email_msg.get('subject', '(No Subject)'))
            body = extract_body(email_msg)
            snippet = body[:200] + "..." if len(body) > 200 else body

            sender_info = parse_sender(email_msg.get('from', ''))
            date_str = parse_date(email_msg.get('date', ''))

            emails.append({
                "subject": subject,
                "body": body,
                "snippet": snippet,
                "sender_name": sender_info["name"],
                "sender_email": sender_info["email"],
                "date": date_str,
                "message_id": msg['id']
            })

        except Exception as e:
            print(f"⚠️  Skipping email {msg['id']}: {e}")
            continue

    return emails
