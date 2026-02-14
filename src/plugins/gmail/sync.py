"""Email synchronization logic for the Gmail plugin.

Fetches emails from selected Gmail folders, parses body + attachments,
sanitizes content, chunks it, and indexes into the RAG vector store.
Follows the same pattern as the Paperless DocumentSyncer.
"""

import base64
import logging
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

from llama_index.core.schema import TextNode

from .client import GmailClient

logger = logging.getLogger(__name__)

# Default label name applied to emails after RAG indexing
DEFAULT_PROCESSED_LABEL = "rag-indexed"

# Maximum characters per chunk for embedding.
MAX_CHUNK_CHARS = 6_000
CHUNK_OVERLAP_CHARS = 200

# Minimum useful content length after sanitization (characters).
MIN_CONTENT_CHARS = 50

# Minimum ratio of word-like characters for quality filtering
MIN_WORD_CHAR_RATIO = 0.40

# Unicode categories to strip (format characters: RTL/LTR marks, etc.)
_STRIP_UNICODE_CATEGORIES = {"Cf"}

# Supported attachment MIME types for text extraction
_TEXT_EXTRACTABLE_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/csv",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Attachment:
    """Parsed email attachment metadata + extracted text."""

    filename: str
    mime_type: str
    size: int
    attachment_id: str  # Gmail attachment ID for downloading
    extracted_text: str = ""


@dataclass
class ParsedEmail:
    """Fully parsed email with body text and attachment info."""

    message_id: str
    thread_id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    date: Optional[datetime]
    body_text: str
    labels: List[str]
    snippet: str
    attachments: List[Attachment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor."""

    def __init__(self):
        super().__init__()
        self._buf = StringIO()

    def handle_data(self, data: str) -> None:
        self._buf.write(data)

    def get_text(self) -> str:
        return self._buf.getvalue()


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# Content sanitization
# ---------------------------------------------------------------------------


def _strip_unicode_control(text: str) -> str:
    """Remove Unicode format characters (category Cf)."""
    return "".join(
        ch for ch in text if unicodedata.category(ch) not in _STRIP_UNICODE_CATEGORIES
    )


def _sanitize_email_content(raw: str) -> str:
    """Clean raw email content for RAG embedding.

    Strips:
    - Unicode control characters
    - Excessive quoting (> lines from reply chains)
    - Email signatures (lines after -- )
    - Excessive whitespace

    Args:
        raw: Raw email body text

    Returns:
        Cleaned text suitable for embedding
    """
    if not raw:
        return ""

    text = _strip_unicode_control(raw)

    # Strip HTML if present
    if "<" in text and ">" in text:
        text = _strip_html(text)

    # Remove excessive reply quoting (lines starting with >)
    lines = text.split("\n")
    cleaned_lines = []
    consecutive_quoted = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            consecutive_quoted += 1
            # Keep first 3 quoted lines for context, skip the rest
            if consecutive_quoted <= 3:
                cleaned_lines.append(stripped.lstrip("> "))
        else:
            consecutive_quoted = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove common email signature delimiter
    sig_markers = ["-- \n", "--\n", "---\n", "Sent from my iPhone", "Sent from my Galaxy"]
    for marker in sig_markers:
        idx = text.find(marker)
        if idx > 0 and idx > len(text) * 0.3:
            # Only strip if the marker is in the latter portion of the email
            text = text[:idx]
            break

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Attachment text extraction
# ---------------------------------------------------------------------------


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF file.

    Args:
        data: PDF file bytes

    Returns:
        Extracted text, or empty string on failure
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e}")
        return ""


def _extract_docx_text(data: bytes) -> str:
    """Extract text from a DOCX file.

    Args:
        data: DOCX file bytes

    Returns:
        Extracted text, or empty string on failure
    """
    try:
        from docx import Document

        doc = Document(BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.warning(f"DOCX text extraction failed: {e}")
        return ""


def _extract_attachment_text(data: bytes, filename: str, mime_type: str) -> str:
    """Extract text content from an attachment based on its type.

    Args:
        data: Attachment file bytes
        filename: Original filename
        mime_type: MIME type of the attachment

    Returns:
        Extracted text, or empty string if not extractable
    """
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return _extract_pdf_text(data)
    elif (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or filename.lower().endswith(".docx")
    ):
        return _extract_docx_text(data)
    elif mime_type.startswith("text/") or filename.lower().endswith((".txt", ".csv")):
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""
    else:
        logger.debug(f"Skipping non-extractable attachment: {filename} ({mime_type})")
        return ""


# ---------------------------------------------------------------------------
# Text chunking (reused from Paperless pattern)
# ---------------------------------------------------------------------------


def _split_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """Split text into chunks that fit within the embedding model's token limit.

    Tries to split on paragraph boundaries first, then sentence boundaries,
    with hard character splits as a fallback.

    Args:
        text: Full text to split
        max_chars: Maximum characters per chunk
        overlap: Character overlap between consecutive chunks

    Returns:
        List of text chunks (at least one element)
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 1

        chunks.append(text[start:boundary])
        start = max(boundary - overlap, boundary)
        if boundary == end:
            start = boundary - overlap

    return chunks


def _is_quality_chunk(chunk: str) -> bool:
    """Check whether a text chunk has enough meaningful content.

    Args:
        chunk: A single text chunk

    Returns:
        True if the chunk passes quality checks
    """
    stripped = chunk.strip()
    if len(stripped) < 20:
        return False

    word_chars = len(re.findall(r"[\w\s.,;:!?'\"-]", stripped, re.UNICODE))
    ratio = word_chars / len(stripped) if stripped else 0

    if ratio < MIN_WORD_CHAR_RATIO:
        logger.debug(
            "Rejecting low-quality chunk (%.0f%% word chars, %d chars)",
            ratio * 100,
            len(stripped),
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Email parser
# ---------------------------------------------------------------------------


def _get_header(headers: List[Dict[str, str]], name: str) -> str:
    """Get an email header value by name (case-insensitive).

    Args:
        headers: List of {name, value} dicts from Gmail API
        name: Header name to find

    Returns:
        Header value, or empty string if not found
    """
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _decode_body_data(data: str) -> str:
    """Decode a base64url-encoded body data string from Gmail API.

    Args:
        data: Base64url-encoded string

    Returns:
        Decoded UTF-8 text
    """
    if not data:
        return ""
    try:
        decoded = base64.urlsafe_b64decode(data)
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_parts(
    payload: Dict[str, Any],
    prefer_plain: bool = True,
) -> str:
    """Recursively extract text body from a Gmail message payload.

    Walks the MIME tree, preferring text/plain over text/html.
    For multipart messages, concatenates all text parts.

    Args:
        payload: Gmail message payload dict
        prefer_plain: If True, prefer text/plain; otherwise HTML

    Returns:
        Extracted body text
    """
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    parts = payload.get("parts", [])

    # Leaf node with body data
    if not parts and body.get("data"):
        if mime_type == "text/plain":
            return _decode_body_data(body["data"])
        elif mime_type == "text/html":
            html = _decode_body_data(body["data"])
            return _strip_html(html)
        return ""

    # Multipart: recurse into parts
    plain_parts: List[str] = []
    html_parts: List[str] = []

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                plain_parts.append(_decode_body_data(data))
        elif part_mime == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:
                html_parts.append(_strip_html(_decode_body_data(data)))
        elif part_mime.startswith("multipart/"):
            # Recurse into nested multipart
            nested = _extract_body_parts(part, prefer_plain)
            if nested:
                plain_parts.append(nested)

    if prefer_plain and plain_parts:
        return "\n\n".join(plain_parts)
    elif html_parts:
        return "\n\n".join(html_parts)
    elif plain_parts:
        return "\n\n".join(plain_parts)
    return ""


def _extract_attachments_metadata(payload: Dict[str, Any]) -> List[Attachment]:
    """Extract attachment metadata from a Gmail message payload.

    Walks the MIME tree looking for parts with a filename header.

    Args:
        payload: Gmail message payload dict

    Returns:
        List of Attachment objects (without extracted text yet)
    """
    attachments: List[Attachment] = []
    parts = payload.get("parts", [])

    for part in parts:
        filename = part.get("filename", "")
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        attachment_id = body.get("attachmentId", "")
        size = body.get("size", 0)

        if filename and attachment_id:
            attachments.append(
                Attachment(
                    filename=filename,
                    mime_type=mime_type,
                    size=size,
                    attachment_id=attachment_id,
                )
            )

        # Recurse into nested multipart parts
        nested_parts = part.get("parts", [])
        if nested_parts:
            nested_payload = {"parts": nested_parts, "mimeType": part.get("mimeType", "")}
            attachments.extend(_extract_attachments_metadata(nested_payload))

    return attachments


def parse_email(message: Dict[str, Any]) -> ParsedEmail:
    """Parse a Gmail API message response into a structured ParsedEmail.

    Args:
        message: Full Gmail message dict from messages.get(format='full')

    Returns:
        ParsedEmail with body text and attachment metadata
    """
    payload = message.get("payload", {})
    headers = payload.get("headers", [])

    # Parse date
    date_str = _get_header(headers, "Date")
    date = None
    if date_str:
        try:
            date = parsedate_to_datetime(date_str)
        except Exception:
            pass

    # Parse recipients
    to_str = _get_header(headers, "To")
    to_addresses = [addr.strip() for addr in to_str.split(",") if addr.strip()]

    # Extract body text
    body_text = _extract_body_parts(payload, prefer_plain=True)

    # Extract attachment metadata
    attachments = _extract_attachments_metadata(payload)

    return ParsedEmail(
        message_id=message.get("id", ""),
        thread_id=message.get("threadId", ""),
        subject=_get_header(headers, "Subject") or "(no subject)",
        from_address=_get_header(headers, "From"),
        to_addresses=to_addresses,
        date=date,
        body_text=body_text,
        labels=message.get("labelIds", []),
        snippet=message.get("snippet", ""),
        attachments=attachments,
    )


# ---------------------------------------------------------------------------
# Email syncer
# ---------------------------------------------------------------------------


class EmailSyncer:
    """Handles syncing emails from Gmail to the RAG vector store.

    Labels processed emails in Gmail with a custom label (default:
    ``rag-indexed``) so they are automatically excluded from future sync
    runs.
    """

    def __init__(self, client: GmailClient, rag):
        """Initialize syncer.

        Args:
            client: GmailClient instance
            rag: LlamaIndexRAG instance
        """
        self.client = client
        self.rag = rag
        self._syncing = False
        self._last_sync = 0
        self._email_count = 0
        self._processed_label_id: Optional[str] = None

    @property
    def is_syncing(self) -> bool:
        return self._syncing

    @property
    def last_sync_time(self) -> int:
        return self._last_sync

    @property
    def synced_count(self) -> int:
        return self._email_count

    def _ensure_processed_label(self, label_name: str) -> Optional[str]:
        """Ensure the processed-emails label exists in Gmail.

        Args:
            label_name: Name of the label to use

        Returns:
            Label ID, or None if creation failed
        """
        if self._processed_label_id is not None:
            return self._processed_label_id

        label_id = self.client.get_or_create_label(label_name)
        if label_id is not None:
            self._processed_label_id = label_id
            logger.info(
                f"Using Gmail label '{label_name}' (id={label_id}) "
                "for processed emails"
            )
        else:
            logger.warning(
                f"Could not get/create Gmail label '{label_name}'. "
                "Emails will still be synced but not labeled."
            )
        return label_id

    def sync_emails(
        self,
        max_emails: int = 500,
        label_ids: Optional[List[str]] = None,
        processed_label_name: str = DEFAULT_PROCESSED_LABEL,
        include_attachments: bool = True,
        force: bool = False,
    ) -> dict:
        """Sync emails from Gmail to the RAG vector store.

        Args:
            max_emails: Maximum emails to sync
            label_ids: Gmail label IDs to fetch from (None = INBOX)
            processed_label_name: Label to mark processed emails
            include_attachments: Whether to extract and index attachment text
            force: If True, skip processed-label exclusion and dedup checks

        Returns:
            Dict with sync results
        """
        if self._syncing:
            return {"status": "already_running"}

        self._syncing = True
        synced = 0
        skipped = 0
        errors = 0
        labeled = 0
        attachment_count = 0

        try:
            # Auto-detect empty collection → force mode
            if not force:
                try:
                    info = self.rag.qdrant_client.get_collection(
                        self.rag.COLLECTION_NAME
                    )
                    if (info.points_count or 0) == 0:
                        logger.info(
                            "Qdrant collection is empty — automatically "
                            "enabling force mode for full re-sync"
                        )
                        force = True
                except Exception as e:
                    logger.debug(f"Could not check collection point count: {e}")

            if force:
                logger.info("Starting Gmail FORCE re-sync (ignoring processed label)...")
            else:
                logger.info("Starting Gmail email sync...")

            # Ensure the processed label exists
            processed_label_id = self._ensure_processed_label(processed_label_name)

            # Build exclusion query: skip already-processed emails
            exclude_query = ""
            if not force and processed_label_id:
                exclude_query = f"-label:{processed_label_name}"

            # Use provided label_ids or default to INBOX
            fetch_labels = label_ids if label_ids else ["INBOX"]

            # Fetch emails page by page
            page_token = None
            page_num = 1

            while synced < max_emails:
                logger.info(f"Fetching page {page_num}...")
                result = self.client.list_messages(
                    label_ids=fetch_labels,
                    query=exclude_query,
                    max_results=min(100, max_emails - synced),
                    page_token=page_token,
                )

                messages = result.get("messages", [])
                if not messages:
                    break

                for msg_stub in messages:
                    if synced >= max_emails:
                        break

                    msg_id = msg_stub.get("id", "")

                    try:
                        source_id = f"gmail:{msg_id}"

                        # Dedup check
                        if not force and self.rag._message_exists(source_id):
                            skipped += 1
                            # Still label it if not labeled yet
                            if processed_label_id:
                                self.client.add_label_to_message(
                                    msg_id, processed_label_id
                                )
                            continue

                        # Fetch full message
                        message = self.client.get_message(msg_id, format="full")
                        parsed = parse_email(message)

                        # Sanitize body
                        body = _sanitize_email_content(parsed.body_text)
                        if len(body) < MIN_CONTENT_CHARS:
                            logger.debug(
                                f"Skipping email '{parsed.subject}' "
                                f"(id={msg_id}): only {len(body)} chars"
                            )
                            skipped += 1
                            # Still label it
                            if processed_label_id:
                                self.client.add_label_to_message(
                                    msg_id, processed_label_id
                                )
                            continue

                        # Determine timestamp
                        ts = (
                            int(parsed.date.timestamp())
                            if parsed.date
                            else int(time.time())
                        )

                        # Build base metadata
                        base_metadata = {
                            "source": "gmail",
                            "source_id": source_id,
                            "content_type": "text",
                            "chat_name": parsed.subject,
                            "sender": parsed.from_address,
                            "timestamp": ts,
                            "folder": ",".join(parsed.labels),
                            "thread_id": parsed.thread_id,
                            "to": ",".join(parsed.to_addresses[:5]),
                            "has_attachments": str(len(parsed.attachments) > 0).lower(),
                            "attachment_names": ",".join(
                                a.filename for a in parsed.attachments[:10]
                            ),
                        }

                        # Chunk and index email body
                        chunks = _split_text(body)
                        chunks = [c for c in chunks if _is_quality_chunk(c)]

                        if not chunks:
                            skipped += 1
                            if processed_label_id:
                                self.client.add_label_to_message(
                                    msg_id, processed_label_id
                                )
                            continue

                        chunk_ok = True
                        for idx, chunk in enumerate(chunks):
                            chunk_meta = dict(base_metadata)
                            chunk_meta["message"] = chunk[:2000]
                            if len(chunks) > 1:
                                chunk_meta["chunk_index"] = str(idx)
                                chunk_meta["chunk_total"] = str(len(chunks))

                            embedding_text = (
                                f"Email: {parsed.subject}\n"
                                f"From: {parsed.from_address}\n\n"
                                f"{chunk}"
                            )

                            node = TextNode(
                                text=embedding_text,
                                metadata=chunk_meta,
                                id_=str(uuid.uuid4()),
                            )

                            if not self.rag.add_node(node):
                                chunk_ok = False

                        if chunk_ok:
                            synced += 1
                            if len(chunks) > 1:
                                logger.info(
                                    f"Indexed email: {parsed.subject} "
                                    f"({len(chunks)} chunks)"
                                )
                            else:
                                logger.info(f"Indexed email: {parsed.subject}")
                        else:
                            errors += 1

                        # Index attachments
                        if include_attachments and parsed.attachments:
                            for att in parsed.attachments:
                                if att.mime_type not in _TEXT_EXTRACTABLE_MIMES:
                                    # Check by extension as fallback
                                    ext = att.filename.lower().rsplit(".", 1)[-1] if "." in att.filename else ""
                                    if ext not in ("pdf", "docx", "txt", "csv"):
                                        continue

                                try:
                                    att_data = self.client.get_attachment(
                                        msg_id, att.attachment_id
                                    )
                                    att_text = _extract_attachment_text(
                                        att_data, att.filename, att.mime_type
                                    )
                                    if len(att_text) < MIN_CONTENT_CHARS:
                                        continue

                                    att_text = _sanitize_email_content(att_text)
                                    att_chunks = _split_text(att_text)
                                    att_chunks = [
                                        c for c in att_chunks if _is_quality_chunk(c)
                                    ]

                                    for aidx, achunk in enumerate(att_chunks):
                                        att_meta = {
                                            "source": "gmail",
                                            "source_id": f"gmail:{msg_id}:att:{att.filename}",
                                            "content_type": "document",
                                            "chat_name": f"{parsed.subject} — {att.filename}",
                                            "sender": parsed.from_address,
                                            "timestamp": ts,
                                            "message": achunk[:2000],
                                            "folder": ",".join(parsed.labels),
                                            "attachment_name": att.filename,
                                        }
                                        if len(att_chunks) > 1:
                                            att_meta["chunk_index"] = str(aidx)
                                            att_meta["chunk_total"] = str(len(att_chunks))

                                        att_node = TextNode(
                                            text=(
                                                f"Email Attachment: {att.filename}\n"
                                                f"From email: {parsed.subject}\n\n"
                                                f"{achunk}"
                                            ),
                                            metadata=att_meta,
                                            id_=str(uuid.uuid4()),
                                        )
                                        self.rag.add_node(att_node)
                                        attachment_count += 1

                                except Exception as ae:
                                    logger.warning(
                                        f"Failed to extract attachment "
                                        f"'{att.filename}' from email "
                                        f"'{parsed.subject}': {ae}"
                                    )

                        # Label the email as processed
                        if processed_label_id:
                            if self.client.add_label_to_message(
                                msg_id, processed_label_id
                            ):
                                labeled += 1

                    except Exception as e:
                        logger.error(f"Error syncing email {msg_id}: {e}")
                        errors += 1

                # Check for next page
                page_token = result.get("nextPageToken")
                if not page_token:
                    break
                page_num += 1

            self._last_sync = int(time.time())
            self._email_count = synced

            logger.info(
                f"Gmail sync complete: {synced} indexed, "
                f"{labeled} labeled, {skipped} skipped, "
                f"{errors} errors, {attachment_count} attachments"
            )

            return {
                "status": "complete",
                "synced": synced,
                "labeled": labeled,
                "skipped": skipped,
                "errors": errors,
                "attachments": attachment_count,
            }

        except Exception as e:
            logger.error(f"Gmail sync failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self._syncing = False
