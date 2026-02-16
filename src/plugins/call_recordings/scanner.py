"""File discovery for audio call recordings.

Provides scanners for discovering audio files from local directories
and Dropbox folders. Each scanner discovers files, computes content
hashes for deduplication, and provides download access.
"""

import hashlib
import logging
import os
import tempfile
from abc import ABC, abstractmethod
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

    Also attempts to read duration from the audio file.

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
            # mutagen couldn't identify the file type
            return _extract_duration_fallback(file_path, meta)

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
        return _extract_duration_fallback(file_path, meta)
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


def _extract_duration_fallback(
    file_path: str, meta: AudioFileMetadata
) -> AudioFileMetadata:
    """Try to extract duration using soundfile as fallback.

    Args:
        file_path: Path to the audio file
        meta: Existing metadata to augment

    Returns:
        Updated AudioFileMetadata
    """
    try:
        import soundfile as sf

        info = sf.info(file_path)
        meta.duration_seconds = info.duration
    except ImportError:
        logger.debug("soundfile not installed — duration extraction skipped")
    except Exception:
        pass  # Not all formats supported by soundfile
    return meta


def _parse_filename_metadata(filename: str) -> Dict[str, Optional[str]]:
    """Attempt to parse date and participant info from a filename.

    Common filename patterns:
    - ``2024-01-15_John_Doe.mp3``
    - ``recording_20240115_143022.wav``
    - ``Call with Alice 2024-01-15.m4a``
    - ``20240115-call-bob.mp3``

    Args:
        filename: The filename (without directory path)

    Returns:
        Dict with optional 'date_str', 'participants' keys
    """
    import re

    result: Dict[str, Optional[str]] = {"date_str": None, "participants": None}

    # Strip extension
    stem = Path(filename).stem

    # Try to extract date patterns
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

    # Try to extract participant names
    # Remove date portion and common prefixes
    name_part = stem
    # Remove date strings
    name_part = re.sub(r"\d{4}[-_]?\d{2}[-_]?\d{2}", "", name_part)
    # Remove time strings (HH:MM:SS or HHMMSS)
    name_part = re.sub(r"\d{2}[-_:]?\d{2}[-_:]?\d{2}", "", name_part)
    # Remove common prefixes/words
    for prefix in ("recording", "call", "rec", "call_with", "call-with", "with"):
        name_part = re.sub(rf"(?i)^{prefix}[_\-\s]*", "", name_part)
        name_part = re.sub(rf"(?i)[_\-\s]*{prefix}[_\-\s]*", " ", name_part)
    # Clean separators and extra spaces
    name_part = re.sub(r"[_\-]+", " ", name_part).strip()

    if name_part and len(name_part) >= 2:
        # Split on common delimiters for multiple participants
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

    Uses buffered reading for memory efficiency with large files.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(_HASH_BUFFER_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Scanner interface
# ---------------------------------------------------------------------------


class BaseFileScanner(ABC):
    """Abstract base class for audio file scanners."""

    @abstractmethod
    def scan(self) -> List[AudioFile]:
        """Discover audio files in the configured source.

        Returns:
            List of AudioFile objects found
        """

    @abstractmethod
    def download(self, audio_file: AudioFile) -> Path:
        """Get a local path to the audio file for transcription.

        For local scanners this returns the existing path.
        For remote scanners this downloads to a temp directory.

        Args:
            audio_file: The AudioFile to download

        Returns:
            Path to the local audio file
        """

    @abstractmethod
    def cleanup(self, audio_file: AudioFile, local_path: Path) -> None:
        """Clean up any temp files after transcription.

        Args:
            audio_file: The AudioFile that was processed
            local_path: The local path returned by download()
        """

    @abstractmethod
    def test_connection(self) -> bool:
        """Test that the source is accessible.

        Returns:
            True if the source is accessible
        """


# ---------------------------------------------------------------------------
# Local directory scanner
# ---------------------------------------------------------------------------


class LocalFileScanner(BaseFileScanner):
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
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            ext = path.suffix.lower().lstrip(".")
            if ext not in self.extensions:
                continue

            try:
                stat = path.stat()
                content_hash = compute_file_hash(str(path))
                file_meta = _extract_audio_metadata(str(path))

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
            except Exception as e:
                logger.warning(f"Failed to scan file {path}: {e}")
                continue

        logger.info(
            f"Local scan found {len(files)} audio files in {self.source_path}"
        )
        return files

    def download(self, audio_file: AudioFile) -> Path:
        """Return the existing local path (no download needed).

        Args:
            audio_file: The local AudioFile

        Returns:
            Path to the existing local file
        """
        return Path(audio_file.path)

    def cleanup(self, audio_file: AudioFile, local_path: Path) -> None:
        """No-op for local files — nothing to clean up."""
        pass

    def test_connection(self) -> bool:
        """Test that the local directory exists and is readable.

        Returns:
            True if directory is accessible
        """
        root = Path(self.source_path)
        return root.exists() and root.is_dir() and os.access(str(root), os.R_OK)


# ---------------------------------------------------------------------------
# Dropbox scanner
# ---------------------------------------------------------------------------


class DropboxFileScanner(BaseFileScanner):
    """Scans a Dropbox folder for audio files.

    Uses the Dropbox Python SDK to list files and download them
    to a temp directory for transcription.

    Args:
        access_token: Dropbox API access token
        folder_path: Dropbox folder path to scan (e.g., "/Call Recordings")
        extensions: Set of audio file extensions to match
    """

    def __init__(
        self,
        access_token: str,
        folder_path: str = "",
        extensions: Optional[set] = None,
    ):
        self.access_token = access_token
        self.folder_path = folder_path.rstrip("/")
        self.extensions = extensions or DEFAULT_AUDIO_EXTENSIONS
        self._dbx = None
        # Track temp dirs for cleanup
        self._temp_dirs: Dict[str, str] = {}

    @property
    def dbx(self):
        """Lazy-initialize the Dropbox client."""
        if self._dbx is None:
            try:
                import dropbox

                self._dbx = dropbox.Dropbox(self.access_token)
            except ImportError:
                raise ImportError(
                    "dropbox package not installed. "
                    "Install with: pip install dropbox"
                )
        return self._dbx

    def scan(self) -> List[AudioFile]:
        """List audio files in the Dropbox folder.

        Uses Dropbox's server-side content_hash for dedup
        (avoids downloading just for hashing).

        Returns:
            List of AudioFile objects with Dropbox content hashes
        """
        import dropbox

        files: List[AudioFile] = []
        try:
            result = self.dbx.files_list_folder(
                self.folder_path or "",
                recursive=True,
            )

            while True:
                for entry in result.entries:
                    if not isinstance(entry, dropbox.files.FileMetadata):
                        continue

                    ext = Path(entry.name).suffix.lower().lstrip(".")
                    if ext not in self.extensions:
                        continue

                    # Use Dropbox's content_hash for dedup
                    content_hash = entry.content_hash or ""

                    modified_at = entry.server_modified
                    if modified_at.tzinfo is None:
                        modified_at = modified_at.replace(tzinfo=ZoneInfo("UTC"))

                    files.append(
                        AudioFile(
                            filename=entry.name,
                            path=entry.path_display or entry.path_lower or "",
                            size=entry.size,
                            modified_at=modified_at,
                            content_hash=content_hash,
                            extension=ext,
                            # Metadata extraction happens after download
                        )
                    )

                if not result.has_more:
                    break
                result = self.dbx.files_list_folder_continue(result.cursor)

        except Exception as e:
            logger.error(f"Dropbox scan failed: {e}")
            return []

        logger.info(
            f"Dropbox scan found {len(files)} audio files in "
            f"'{self.folder_path or '/'}'"
        )
        return files

    def download(self, audio_file: AudioFile) -> Path:
        """Download a Dropbox file to a local temp directory.

        Args:
            audio_file: The Dropbox AudioFile to download

        Returns:
            Path to the downloaded local temp file
        """
        temp_dir = tempfile.mkdtemp(prefix="call_rec_")
        self._temp_dirs[audio_file.content_hash] = temp_dir

        local_path = Path(temp_dir) / audio_file.filename

        logger.info(f"Downloading from Dropbox: {audio_file.path}")
        self.dbx.files_download_to_file(str(local_path), audio_file.path)

        # Now extract metadata from the downloaded file
        audio_file.file_metadata = _extract_audio_metadata(str(local_path))

        return local_path

    def cleanup(self, audio_file: AudioFile, local_path: Path) -> None:
        """Remove temp files after transcription.

        Args:
            audio_file: The processed AudioFile
            local_path: The temp file path to clean up
        """
        import shutil

        temp_dir = self._temp_dirs.pop(audio_file.content_hash, None)
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp dir: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")

    def test_connection(self) -> bool:
        """Test Dropbox connectivity and folder access.

        Returns:
            True if Dropbox is accessible and folder exists
        """
        try:
            # Test account access
            self.dbx.users_get_current_account()
            # Test folder access
            self.dbx.files_list_folder(self.folder_path or "", limit=1)
            return True
        except Exception as e:
            logger.error(f"Dropbox connection test failed: {e}")
            return False
