"""File discovery for audio call recordings.

Provides a scanner for discovering audio files from local directories.
Discovers files, computes content hashes for deduplication, and extracts
metadata from audio file tags (ID3/MP4) and filename patterns.
"""

import errno
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Supported audio file extensions (lowercase, without dot)
DEFAULT_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "wma", "aac", "opus", "webm", "mp4"}

# Buffer size for SHA256 hashing (64 KB)
_HASH_BUFFER_SIZE = 65536


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AudioFileMetadata:
    """Metadata extracted from audio file tags (ID3, MP4, etc.)."""

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    date: Optional[str] = None
    duration_seconds: Optional[float] = None
    genre: Optional[str] = None


@dataclass
class AudioFile:
    """Discovered audio file with metadata for sync processing."""

    filename: str
    path: str
    size: int
    modified_at: datetime
    content_hash: str
    extension: str
    file_metadata: AudioFileMetadata = field(default_factory=AudioFileMetadata)

    @property
    def source_id(self) -> str:
        """Unique source ID for dedup checks in Qdrant."""
        return f"call_recording:{self.content_hash}"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _extract_audio_metadata(file_path: str) -> AudioFileMetadata:
    """Extract metadata from an audio file using mutagen.

    Reads ID3 tags (MP3), MP4 atoms (M4A/MP4), Vorbis comments (OGG/FLAC),
    and other tag formats supported by mutagen.

    Args:
        file_path: Path to the audio file

    Returns:
        AudioFileMetadata with whatever fields could be extracted
    """
    meta = AudioFileMetadata()

    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(file_path, easy=True)
        if audio is None:
            return meta

        # Extract common tags (easy mode normalises tag names)
        if audio.tags:
            meta.title = _first_tag(audio.tags, "title")
            meta.artist = _first_tag(audio.tags, "artist")
            meta.album = _first_tag(audio.tags, "album")
            meta.date = _first_tag(audio.tags, "date")
            meta.genre = _first_tag(audio.tags, "genre")

        # Duration from mutagen's info object
        if hasattr(audio, "info") and audio.info and hasattr(audio.info, "length"):
            meta.duration_seconds = audio.info.length

    except ImportError:
        logger.debug("mutagen not installed — audio metadata extraction disabled")
    except Exception as e:
        logger.debug(f"Audio metadata extraction failed for {file_path}: {e}")

    return meta


def _first_tag(tags: Dict, key: str) -> Optional[str]:
    """Get the first value for a tag key (tags may store lists)."""
    val = tags.get(key)
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def _parse_filename_metadata(filename: str) -> Dict[str, Optional[str]]:
    """Attempt to parse date, time, phone, and participant info from a filename.

    Recognised filename patterns:
    - ``Call_recording_0545479819_260209_104905.m4a``
      (phone=0545479819, date=09/02/2026, time=10:49:05)
    - ``2024-01-15_John_Doe.mp3``
    - ``recording_20240115_143022.wav``
    - ``Call with Alice 2024-01-15.m4a``
    - ``20240115-call-bob.mp3``

    Args:
        filename: The filename (without directory path)

    Returns:
        Dict with optional keys: date_str, time_str, participants, phone_number
    """
    result: Dict[str, Optional[str]] = {
        "date_str": None,
        "time_str": None,
        "participants": None,
        "phone_number": None,
    }

    # Strip extension
    stem = Path(filename).stem

    # ---- Priority pattern: Call_recording_IDENTIFIER_DDMMYY_HHMMSS ----
    # IDENTIFIER is either a phone number (digits) or a contact name (letters/underscores)
    # Examples:
    #   Call_recording_0545479819_260209_104905.m4a  → phone
    #   Call_recording_Amir_Adar_260209_174436.m4a   → contact name
    call_rec_match = re.match(
        r"(?i)call[_\-\s]*recording[_\-\s]*"
        r"(.+?)"                # identifier (phone or name)
        r"[_\-\s]"
        r"(\d{6})"              # DDMMYY
        r"[_\-\s]"
        r"(\d{6})$",            # HHMMSS (must be at end)
        stem,
    )
    if call_rec_match:
        identifier = call_rec_match.group(1).strip("_- ")
        date_raw = call_rec_match.group(2)  # DDMMYY
        time_raw = call_rec_match.group(3)  # HHMMSS

        # Determine if identifier is a phone number or a name
        if re.match(r"^\d{7,15}$", identifier):
            # It's a phone number
            result["phone_number"] = identifier
            result["participants"] = identifier
        else:
            # It's a contact name — replace underscores with spaces
            contact_name = identifier.replace("_", " ").strip()
            result["participants"] = contact_name

        # Parse DDMMYY → YYYY-MM-DD
        try:
            dd = int(date_raw[0:2])
            mm = int(date_raw[2:4])
            yy = int(date_raw[4:6])
            # Two-digit year: 00-49 → 2000-2049, 50-99 → 1950-1999
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
            result["date_str"] = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
        except (ValueError, IndexError):
            pass

        # Parse HHMMSS → HH:MM:SS
        try:
            hh = time_raw[0:2]
            mi = time_raw[2:4]
            ss = time_raw[4:6]
            result["time_str"] = f"{hh}:{mi}:{ss}"
        except (ValueError, IndexError):
            pass

        return result

    # ---- Generic date patterns ----
    # ISO-like: 2024-01-15 or 2024_01_15
    date_match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", stem)
    if date_match:
        result["date_str"] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        # Compact: 20240115
        compact_match = re.search(r"(\d{4})(\d{2})(\d{2})", stem)
        if compact_match:
            result["date_str"] = (
                f"{compact_match.group(1)}-{compact_match.group(2)}-{compact_match.group(3)}"
            )

    # ---- Extract phone number from generic filenames ----
    # Look for 7-15 consecutive digits that could be a phone number
    phone_match = re.search(r"(?<!\d)(\d{7,15})(?!\d)", stem)
    if phone_match:
        candidate = phone_match.group(1)
        # Exclude date-like sequences (YYYYMMDD or DDMMYYYY)
        if not re.match(r"(19|20)\d{6}$", candidate) and not re.match(r"\d{2}(0[1-9]|1[0-2])(19|20)\d{2}$", candidate):
            result["phone_number"] = candidate

    # ---- Extract participant names ----
    name_part = stem
    # Remove date strings
    name_part = re.sub(r"\d{4}[-_]?\d{2}[-_]?\d{2}", "", name_part)
    # Remove time strings (HH:MM:SS or HHMMSS)
    name_part = re.sub(r"\d{2}[-_:]?\d{2}[-_:]?\d{2}", "", name_part)
    # Remove phone numbers (7+ digits)
    name_part = re.sub(r"\d{7,15}", "", name_part)
    # Remove common prefixes/words
    for prefix in ("recording", "call", "rec", "call_with", "call-with", "with"):
        name_part = re.sub(rf"(?i)^{prefix}[_\-\s]*", "", name_part)
        name_part = re.sub(rf"(?i)[_\-\s]*{prefix}[_\-\s]*", " ", name_part)
    # Clean separators and extra spaces
    name_part = re.sub(r"[_\-]+", " ", name_part).strip()

    if name_part and len(name_part) >= 2:
        participants = [
            p.strip()
            for p in re.split(r"\s*[&,+]\s*|\s+and\s+", name_part)
            if p.strip()
        ]
        if participants:
            result["participants"] = ", ".join(participants)

    return result


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file's content.

    Falls back to a metadata-based hash (path + size + mtime) when the
    file cannot be read — e.g. Dropbox CloudStorage file-lock errors
    (``[Errno 35] Resource deadlock avoided`` on macOS).

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash string
    """
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(_HASH_BUFFER_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except OSError as e:
        # Fallback: hash based on path + size + mtime (for locked/online-only files)
        logger.debug(f"Content hash failed for {file_path} ({e}), using metadata hash")
        stat = os.stat(file_path)
        meta_str = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(meta_str.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Local directory scanner
# ---------------------------------------------------------------------------


class LocalFileScanner:
    """Scans a local directory recursively for audio files.

    Args:
        source_path: Root directory to scan
        extensions: Set of audio file extensions to match (lowercase, no dot)
    """

    def __init__(self, source_path: str, extensions: Optional[set] = None):
        self.source_path = source_path
        self.extensions = extensions or DEFAULT_AUDIO_EXTENSIONS

    def scan(self) -> List[AudioFile]:
        """Recursively scan the directory for audio files.

        Returns:
            List of AudioFile objects with content hashes
        """
        root = Path(self.source_path)
        if not root.exists() or not root.is_dir():
            logger.warning(f"Source path does not exist or is not a directory: {self.source_path}")
            return []

        files: List[AudioFile] = []
        locked_count = 0
        error_count = 0

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            ext = path.suffix.lower().lstrip(".")
            if ext not in self.extensions:
                continue

            try:
                try:
                    stat = path.stat()
                except OSError as e:
                    if e.errno == errno.EDEADLK:
                        locked_count += 1
                        logger.debug(f"File locked by cloud sync (Errno 35): {path.name}")
                    else:
                        error_count += 1
                        logger.debug(f"Cannot stat {path}: {e}")
                    continue
                content_hash = compute_file_hash(str(path))
                # Audio metadata extraction is non-critical — skip on lock errors
                try:
                    file_meta = _extract_audio_metadata(str(path))
                except OSError:
                    file_meta = AudioFileMetadata()

                files.append(
                    AudioFile(
                        filename=path.name,
                        path=str(path),
                        size=stat.st_size,
                        modified_at=datetime.fromtimestamp(
                            stat.st_mtime, tz=ZoneInfo("UTC")
                        ),
                        content_hash=content_hash,
                        extension=ext,
                        file_metadata=file_meta,
                    )
                )
            except OSError as e:
                if e.errno == errno.EDEADLK:
                    locked_count += 1
                    logger.debug(f"File locked by cloud sync (Errno 35): {path.name}")
                else:
                    error_count += 1
                    logger.warning(f"Failed to scan file {path}: {e}")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to scan file {path}: {e}")
                continue

        summary = f"Local scan found {len(files)} audio files in {self.source_path}"
        if locked_count:
            summary += f" ({locked_count} skipped — locked by cloud sync)"
        if error_count:
            summary += f" ({error_count} errors)"
        logger.info(summary)

        return files

    def download(self, audio_file: AudioFile) -> Path:
        """Return the existing local path (no download needed)."""
        return Path(audio_file.path)

    def cleanup(self, audio_file: AudioFile, local_path: Path) -> None:
        """No-op for local files."""
        pass

    def test_connection(self) -> bool:
        """Test that the local directory exists and is readable."""
        root = Path(self.source_path)
        return root.exists() and root.is_dir() and os.access(str(root), os.R_OK)
