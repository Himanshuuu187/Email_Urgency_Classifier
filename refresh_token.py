"""
Run this script LOCALLY (not on Streamlit Cloud) to get a fresh token.
It will open a browser window for Google login.

Usage:
    python refresh_token.py

After running, copy the printed JSON and paste it into:
  Streamlit Cloud → App Settings → Secrets → GMAIL_TOKEN = '<paste here>'
"""

import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

flow  = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
creds = flow.run_local_server(port=0)

# Save locally
with open("token.json", "w") as f:
    f.write(creds.to_json())

# Print for Streamlit secrets
token_dict = json.loads(creds.to_json())
print("\n✅ New token generated!")
print("\n── Copy this into Streamlit Cloud Secrets as GMAIL_TOKEN ──\n")
print(json.dumps(token_dict, indent=2))
print("\n────────────────────────────────────────────────────────────")
print("In Streamlit Cloud → Settings → Secrets, add:")
print(f'GMAIL_TOKEN = \'{json.dumps(token_dict)}\'')
