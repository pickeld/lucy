"""Gmail API client wrapper for the Gmail plugin.

Provides a high-level interface over the Google Gmail API for:
- Testing connectivity
- Listing labels/folders
- Fetching messages with pagination
- Downloading attachments
"""

import base64
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GmailClient:
    """Client for the Gmail REST API via google-api-python-client.

    All methods operate on the authenticated user ("me") and handle
    pagination, error reporting, and response normalization.
    """

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        """Initialize the client.  The Gmail service is built lazily.

        Args:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            refresh_token: OAuth2 refresh token
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._service = None

    @property
    def service(self):
        """Lazy-build the authenticated Gmail API service."""
        if self._service is None:
            from .auth import build_gmail_service

            self._service = build_gmail_service(
                self._client_id,
                self._client_secret,
                self._refresh_token,
            )
        return self._service

    # -------------------------------------------------------------------------
    # Connection test
    # -------------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test API connectivity by fetching the user's profile.

        Returns:
            True if the connection and authentication are valid
        """
        try:
            profile = self.service.users().getProfile(userId="me").execute()
            email = profile.get("emailAddress", "unknown")
            total = profile.get("messagesTotal", 0)
            logger.info(f"Gmail connection OK: {email} ({total} messages)")
            return True
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return False

    def get_profile(self) -> Dict[str, Any]:
        """Fetch the authenticated user's Gmail profile.

        Returns:
            Profile dict with emailAddress, messagesTotal, threadsTotal, historyId
        """
        return self.service.users().getProfile(userId="me").execute()

    # -------------------------------------------------------------------------
    # Labels (folders)
    # -------------------------------------------------------------------------

    def get_labels(self) -> List[Dict[str, Any]]:
        """Fetch all Gmail labels (folders) for the user.

        Returns:
            List of label dicts: {id, name, type, messagesTotal, ...}
        """
        try:
            result = self.service.users().labels().list(userId="me").execute()
            labels = result.get("labels", [])

            # Enrich with message counts (the list endpoint only returns
            # id/name/type — need individual get() for counts)
            enriched: List[Dict[str, Any]] = []
            for label in labels:
                enriched.append({
                    "id": label.get("id", ""),
                    "name": label.get("name", ""),
                    "type": label.get("type", "user"),
                })

            # Sort: system labels first (INBOX, SENT, etc.), then user labels
            enriched.sort(key=lambda l: (0 if l["type"] == "system" else 1, l["name"]))
            return enriched
        except Exception as e:
            logger.error(f"Failed to fetch Gmail labels: {e}")
            return []

    def get_label_id_by_name(self, name: str) -> Optional[str]:
        """Find a label ID by its display name (case-insensitive).

        Args:
            name: Label name to search for

        Returns:
            Label ID string, or None if not found
        """
        labels = self.get_labels()
        name_lower = name.lower()
        for label in labels:
            if label["name"].lower() == name_lower:
                return label["id"]
        return None

    def create_label(self, name: str) -> Optional[Dict[str, Any]]:
        """Create a new Gmail label.

        Args:
            name: Label name to create

        Returns:
            Created label dict, or None on failure
        """
        try:
            label = (
                self.service.users()
                .labels()
                .create(
                    userId="me",
                    body={
                        "name": name,
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                )
                .execute()
            )
            logger.info(f"Created Gmail label: '{name}' (id={label.get('id')})")
            return label
        except Exception as e:
            logger.error(f"Failed to create Gmail label '{name}': {e}")
            return None

    def get_or_create_label(self, name: str) -> Optional[str]:
        """Get a label ID by name, creating the label if it doesn't exist.

        Args:
            name: Label name

        Returns:
            Label ID, or None on failure
        """
        label_id = self.get_label_id_by_name(name)
        if label_id:
            return label_id

        created = self.create_label(name)
        return created.get("id") if created else None

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    def list_messages(
        self,
        label_ids: Optional[List[str]] = None,
        query: str = "",
        max_results: int = 100,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List message IDs matching the given criteria.

        Args:
            label_ids: Filter by label IDs (e.g. ["INBOX", "SENT"])
            query: Gmail search query (same syntax as the search bar)
            max_results: Maximum messages to return per page
            page_token: Pagination token from a previous call

        Returns:
            Dict with 'messages' (list of {id, threadId}) and optional
            'nextPageToken'
        """
        try:
            kwargs: Dict[str, Any] = {
                "userId": "me",
                "maxResults": min(max_results, 500),
            }
            if label_ids:
                kwargs["labelIds"] = label_ids
            if query:
                kwargs["q"] = query
            if page_token:
                kwargs["pageToken"] = page_token

            return self.service.users().messages().list(**kwargs).execute()
        except Exception as e:
            logger.error(f"Failed to list Gmail messages: {e}")
            return {"messages": []}

    def get_message(self, message_id: str, format: str = "full") -> Dict[str, Any]:
        """Fetch a single message with full content.

        Args:
            message_id: Gmail message ID
            format: Response format — 'full' (parsed), 'raw', or 'metadata'

        Returns:
            Message dict with payload, headers, body, etc.
        """
        return (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format=format)
            .execute()
        )

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """Download an attachment's raw bytes.

        Args:
            message_id: Gmail message ID
            attachment_id: Attachment ID from the message payload

        Returns:
            Decoded attachment bytes
        """
        result = (
            self.service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )
        data = result.get("data", "")
        return base64.urlsafe_b64decode(data)

    def add_label_to_message(self, message_id: str, label_id: str) -> bool:
        """Add a label to a message.

        Args:
            message_id: Gmail message ID
            label_id: Label ID to add

        Returns:
            True if successful
        """
        try:
            self.service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"addLabelIds": [label_id]},
            ).execute()
            return True
        except Exception as e:
            logger.error(
                f"Failed to add label {label_id} to message {message_id}: {e}"
            )
            return False

    def batch_add_label(self, message_ids: List[str], label_id: str) -> int:
        """Add a label to multiple messages using batch modify.

        Args:
            message_ids: List of Gmail message IDs
            label_id: Label ID to add

        Returns:
            Number of messages successfully modified
        """
        if not message_ids:
            return 0

        # Gmail API batch modify supports up to 1000 messages at a time
        batch_size = 1000
        modified = 0

        for i in range(0, len(message_ids), batch_size):
            batch = message_ids[i : i + batch_size]
            try:
                self.service.users().messages().batchModify(
                    userId="me",
                    body={
                        "ids": batch,
                        "addLabelIds": [label_id],
                    },
                ).execute()
                modified += len(batch)
            except Exception as e:
                logger.error(f"Batch label failed for {len(batch)} messages: {e}")

        return modified
