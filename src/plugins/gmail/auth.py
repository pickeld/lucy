"""Google OAuth2 authentication helpers for the Gmail plugin.

Handles the full OAuth2 flow:
1. Generate authorization URL for user consent
2. Exchange authorization code for access + refresh tokens
3. Build authenticated Gmail API service from stored refresh token
4. Test authentication by calling the Gmail profile endpoint
"""

import logging
from typing import Any, Dict, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource

logger = logging.getLogger(__name__)

# OAuth2 scopes required by the Gmail plugin.
# - gmail.readonly: read email messages and metadata
# - gmail.labels: read and manage labels (for processed-label tagging)
# - gmail.modify: needed to add labels to processed messages
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Google OAuth2 endpoints
_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Out-of-band redirect for copy-paste code flow (desktop/server apps)
_OOB_REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob"


def get_auth_url(client_id: str, client_secret: str) -> str:
    """Generate a Google OAuth2 authorization URL for user consent.

    The user should open this URL in their browser, sign in with their
    Google account, and approve the requested permissions.  Google will
    then display an authorization code that the user copies back into
    the settings UI.

    Args:
        client_id: Google Cloud OAuth2 client ID
        client_secret: Google Cloud OAuth2 client secret

    Returns:
        Authorization URL string

    Raises:
        ValueError: If client_id or client_secret are empty
    """
    if not client_id or not client_secret:
        raise ValueError("client_id and client_secret are required")

    flow = InstalledAppFlow.from_client_config(
        client_config={
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": _AUTH_URI,
                "token_uri": _TOKEN_URI,
                "redirect_uris": [_OOB_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",  # Force consent to always get a refresh_token
    )
    return auth_url


def exchange_code(
    client_id: str,
    client_secret: str,
    code: str,
) -> Tuple[str, str]:
    """Exchange an authorization code for OAuth2 tokens.

    Args:
        client_id: Google Cloud OAuth2 client ID
        client_secret: Google Cloud OAuth2 client secret
        code: Authorization code obtained from the consent screen

    Returns:
        Tuple of (access_token, refresh_token)

    Raises:
        ValueError: If any parameter is empty
        Exception: If the token exchange fails
    """
    if not client_id or not client_secret or not code:
        raise ValueError("client_id, client_secret, and code are all required")

    flow = InstalledAppFlow.from_client_config(
        client_config={
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": _AUTH_URI,
                "token_uri": _TOKEN_URI,
                "redirect_uris": [_OOB_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials

    refresh_token = creds.refresh_token or ""
    access_token = creds.token or ""

    if not refresh_token:
        logger.warning(
            "No refresh_token returned — the user may have already authorized "
            "this app.  Revoke access at https://myaccount.google.com/permissions "
            "and re-authorize to get a fresh refresh_token."
        )

    logger.info("OAuth2 token exchange successful")
    return access_token, refresh_token


def build_gmail_service(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> Resource:
    """Build an authenticated Gmail API service from stored credentials.

    Uses the refresh_token to obtain a fresh access_token automatically.
    The returned service object can be used to call Gmail API methods.

    Args:
        client_id: Google Cloud OAuth2 client ID
        client_secret: Google Cloud OAuth2 client secret
        refresh_token: OAuth2 refresh token from previous authorization

    Returns:
        googleapiclient.discovery.Resource for Gmail API v1

    Raises:
        ValueError: If any parameter is empty
        Exception: If credentials cannot be refreshed
    """
    if not client_id or not client_secret or not refresh_token:
        raise ValueError(
            "client_id, client_secret, and refresh_token are all required"
        )

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=_TOKEN_URI,
        scopes=SCOPES,
    )

    # Force a token refresh to verify the credentials are valid
    creds.refresh(Request())

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


def test_authentication(service: Resource) -> Dict[str, Any]:
    """Verify the Gmail service can access the user's profile.

    Args:
        service: Authenticated Gmail API service

    Returns:
        Dict with user profile info (emailAddress, messagesTotal, etc.)

    Raises:
        Exception: If the API call fails
    """
    profile = service.users().getProfile(userId="me").execute()
    logger.info(f"Gmail authenticated as: {profile.get('emailAddress', 'unknown')}")
    return profile
