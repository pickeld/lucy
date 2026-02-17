"""Local Whisper transcription for audio call recordings.

Uses the faster-whisper package (CTranslate2 backend) for local
audio-to-text transcription.  Typically 4-6× faster than the original
openai-whisper on CPU, with identical accuracy.

Optionally runs pyannote.audio speaker diarization to label segments
with speaker identifiers (Speaker A, Speaker B, etc.).

The Whisper model is loaded lazily on first use and cached for
subsequent calls to avoid slow startup times.
"""

import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid Whisper model sizes
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

# Default model size — balance between accuracy and speed
DEFAULT_MODEL_SIZE = "medium"

# Default diarization pipeline
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Compute types ordered by preference per device
_CPU_COMPUTE_TYPE = "int8"       # fastest on CPU, minimal quality loss
_CUDA_COMPUTE_TYPE = "float16"   # optimal for GPU

# Speaker label format
_SPEAKER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Minimum overlap (seconds) below which midpoint proximity is used instead
_OVERLAP_THRESHOLD_S = 0.1


def _is_pyannote_available() -> bool:
    """Check if pyannote.audio is installed and importable."""
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        return True
    except ImportError:
        return False


def _get_pyannote_version() -> Optional[Tuple[int, ...]]:
    """Return the pyannote.audio version as a tuple, or None if unavailable."""
    try:
        import pyannote.audio
        parts = pyannote.audio.__version__.split(".")
        return tuple(int(x) for x in parts[:2])
    except Exception:
        return None


def _verify_ffmpeg_available() -> bool:
    """Check that ffmpeg/ffprobe is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription of an audio file.

    Attributes:
        text: Full transcription text (with speaker labels when diarization is enabled)
        language: Detected language code (e.g., "en", "he", "ar")
        duration_seconds: Audio duration in seconds
        segments: Timestamped text segments from Whisper (with optional 'speaker' key)
        confidence: Average confidence score (0.0 to 1.0).  Derived from
            ``exp(avg_logprob)`` per segment — a practical heuristic, not a
            calibrated probability.
        speakers_detected: Number of distinct speakers found (0 if diarization disabled)
    """

    text: str
    language: Optional[str] = None
    duration_seconds: int = 0
    segments: List[Dict] = field(default_factory=list)
    confidence: float = 1.0
    speakers_detected: int = 0


class WhisperTranscriber:
    """Local Whisper transcriber using faster-whisper (CTranslate2).

    Loads the specified Whisper model lazily on first transcription
    request and caches it for subsequent calls.  Automatically attempts
    GPU (CUDA) initialization and falls back to CPU if CTranslate2
    cannot use the GPU (CUDA/cuDNN version mismatch, missing libraries,
    etc.).

    When diarization is enabled and pyannote.audio is available, runs
    speaker diarization after transcription to label segments with
    speaker identifiers (Speaker A, Speaker B, etc.).

    Args:
        model_size: Whisper model size — "tiny", "base", "small", "medium",
            "large", "large-v2", or "large-v3".
        hf_token: HuggingFace token for gated model downloads.
        enable_diarization: If True, run speaker diarization when available.
        device: Device override — "auto" (default), "cpu", or "cuda".
            When "auto", attempts GPU first and falls back to CPU.
        compute_type: CTranslate2 compute type — "auto" (default), "int8",
            "float16", "float32".  "auto" lets CTranslate2 pick the fastest
            type supported on the current hardware.
        cpu_threads: Number of CPU threads for CTranslate2 inference.
            ``0`` (default) means use all available cores.
        num_workers: Number of parallel decoding workers.  Higher values
            increase throughput at the cost of memory.  ``1`` by default.
        download_root: Directory to cache downloaded models.  ``None`` uses
            the default HuggingFace cache directory.
        local_files_only: If True, never download models from the Hub —
            only use locally cached files.  Useful for air-gapped deployments.
        diarization_model: pyannote pipeline identifier or local path.
            Defaults to ``"pyannote/speaker-diarization-3.1"``.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        hf_token: Optional[str] = None,
        enable_diarization: bool = False,
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    ):
        size = model_size.lower().strip()
        if size not in VALID_MODEL_SIZES:
            logger.warning(
                f"Invalid Whisper model size '{model_size}', "
                f"falling back to '{DEFAULT_MODEL_SIZE}'"
            )
            size = DEFAULT_MODEL_SIZE
        self._model_size = size
        self._model = None
        self._hf_token = hf_token or None
        self._enable_diarization = enable_diarization
        self._diarization_pipeline = None

        # Device / compute config
        self._device = device.lower().strip()
        self._compute_type = compute_type.lower().strip()
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._download_root = download_root
        self._local_files_only = local_files_only
        self._diarization_model = diarization_model

        # Thread-safety locks for lazy initialization
        self._model_lock = threading.Lock()
        self._diarization_lock = threading.Lock()

    @property
    def model_size(self) -> str:
        """Currently configured model size."""
        return self._model_size

    @property
    def diarization_available(self) -> bool:
        """Whether diarization is enabled and pyannote is available."""
        return self._enable_diarization and _is_pyannote_available() and bool(self._hf_token)

    # ------------------------------------------------------------------
    # Lazy model loading (thread-safe)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        """Lazily load the Whisper model on first use.

        Thread-safe: uses a lock to prevent double-loads when called
        from concurrent threads (e.g. the ``ThreadPoolExecutor`` in
        ``CallRecordingSyncer``).

        Device selection strategy:
        - ``device="auto"`` → try GPU first, fall back to CPU on any error
        - ``device="cuda"`` → GPU only (raises on failure)
        - ``device="cpu"`` → CPU only
        """
        if self._model is not None:
            return

        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install with: pip install faster-whisper"
                )

            # Set HF_TOKEN for authenticated model downloads
            token = self._resolve_hf_token()
            if token:
                os.environ["HF_TOKEN"] = token
                logger.info("HF_TOKEN set for model download")

            # Resolve CPU threads
            if self._cpu_threads <= 0:
                import multiprocessing
                cpu_threads = multiprocessing.cpu_count()
            else:
                cpu_threads = self._cpu_threads

            # Build device/compute_type candidates
            candidates = self._build_device_candidates()

            last_error = None
            for device, compute_type in candidates:
                try:
                    logger.info(
                        f"Loading faster-whisper model '{self._model_size}' "
                        f"on {device} ({compute_type}, {cpu_threads} threads, "
                        f"{self._num_workers} workers)..."
                    )
                    model_kwargs = dict(
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        num_workers=self._num_workers,
                    )
                    if self._download_root:
                        model_kwargs["download_root"] = self._download_root
                    if self._local_files_only:
                        model_kwargs["local_files_only"] = self._local_files_only

                    self._model = WhisperModel(self._model_size, **model_kwargs)
                    logger.info(
                        f"faster-whisper model '{self._model_size}' loaded "
                        f"successfully on {device} ({compute_type})"
                    )
                    return
                except Exception as e:
                    last_error = e
                    if device == "cuda":
                        logger.warning(
                            f"GPU model init failed ({e}), "
                            f"falling back to CPU..."
                        )
                    else:
                        # CPU failure is fatal — re-raise
                        raise

            # Should not reach here, but just in case
            if last_error:
                raise last_error

    def _build_device_candidates(self) -> List[Tuple[str, str]]:
        """Build ordered list of (device, compute_type) pairs to try.

        When ``device="auto"``, tries GPU first then CPU.
        When ``compute_type="auto"``, uses "auto" which lets CTranslate2
        select the fastest type for the device/hardware.

        Returns:
            List of (device, compute_type) tuples in preference order.
        """
        if self._device == "cuda":
            ct = self._compute_type if self._compute_type != "auto" else "auto"
            return [("cuda", ct)]

        if self._device == "cpu":
            ct = self._compute_type if self._compute_type != "auto" else _CPU_COMPUTE_TYPE
            return [("cpu", ct)]

        # device="auto" — try GPU first, fall back to CPU
        candidates = []

        # Only attempt GPU if there's a reasonable chance it'll work
        try:
            import torch
            if torch.cuda.is_available():
                cuda_ct = self._compute_type if self._compute_type != "auto" else _CUDA_COMPUTE_TYPE
                candidates.append(("cuda", cuda_ct))
        except ImportError:
            pass

        # Always include CPU as fallback
        cpu_ct = self._compute_type if self._compute_type != "auto" else _CPU_COMPUTE_TYPE
        candidates.append(("cpu", cpu_ct))

        return candidates

    def _resolve_hf_token(self) -> Optional[str]:
        """Resolve the HuggingFace token from multiple sources.

        Checks (in order):
        1. Token passed at init time
        2. HF_TOKEN environment variable
        3. settings_db (live read)

        Returns:
            Token string or None
        """
        if self._hf_token:
            return self._hf_token

        # Check env var (may have been set by another process)
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            self._hf_token = env_token
            return env_token

        # Live read from settings DB (catches tokens set after startup)
        try:
            import settings_db
            token = settings_db.get_setting_value("hf_token") or ""
            if token.strip():
                self._hf_token = token.strip()
                os.environ["HF_TOKEN"] = self._hf_token
                logger.info("HF_TOKEN resolved from settings DB")
                return self._hf_token
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Diarization pipeline loading (thread-safe, version-aware)
    # ------------------------------------------------------------------

    def _ensure_diarization_loaded(self) -> bool:
        """Lazily load the pyannote diarization pipeline.

        Thread-safe and version-aware:
        - pyannote >= 4.0: uses ``token=`` kwarg, FFmpeg-only I/O
        - pyannote 3.x: uses ``use_auth_token=`` kwarg

        Returns True if the pipeline is ready, False if unavailable.
        """
        if self._diarization_pipeline is not None:
            return True

        if not self._enable_diarization:
            return False

        with self._diarization_lock:
            # Double-check after acquiring lock
            if self._diarization_pipeline is not None:
                return True

            token = self._resolve_hf_token()
            if not token:
                logger.warning(
                    "Diarization enabled but HF_TOKEN not set — "
                    "set it in Settings → API Keys to enable speaker detection"
                )
                return False

            try:
                return self._load_diarization_pipeline(token)
            except ImportError:
                logger.info(
                    "pyannote.audio not installed — speaker diarization disabled. "
                    "Install with: pip install pyannote.audio"
                )
                return False
            except Exception as e:
                self._log_diarization_error(e)
                return False

    def _load_diarization_pipeline(self, token: str) -> bool:
        """Perform the actual pipeline loading with version-aware logic.

        Args:
            token: HuggingFace auth token

        Returns:
            True if pipeline loaded successfully

        Raises:
            ImportError: if pyannote.audio is not installed
            Exception: on pipeline load failure
        """
        from pyannote.audio import Pipeline

        pyannote_ver = _get_pyannote_version()
        is_v4_plus = pyannote_ver is not None and pyannote_ver >= (4, 0)

        # --- Compatibility: torch.load weights_only (PyTorch 2.6+) ---
        # Scope the patch to pipeline loading only, then restore.
        torch_load_patched = False
        orig_torch_load = None
        try:
            import torch
            orig_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return orig_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load
            torch_load_patched = True
            logger.debug("Applied scoped torch.load weights_only shim")
        except ImportError:
            pass

        try:
            # --- Compatibility: huggingface_hub shim (only for pyannote < 4.0) ---
            if not is_v4_plus:
                self._apply_huggingface_hub_shim()

            if is_v4_plus:
                # pyannote 4+ requires FFmpeg, no soundfile backend
                if not _verify_ffmpeg_available():
                    logger.warning(
                        "pyannote.audio >= 4.0 requires FFmpeg but it was not "
                        "found on PATH. Diarization may fail."
                    )

            # --- Load the pipeline ---
            # Ensure HF_TOKEN is in env — huggingface_hub reads it as fallback
            os.environ["HF_TOKEN"] = token

            model_id = self._diarization_model

            logger.info(
                f"Loading pyannote diarization pipeline '{model_id}'"
                f" (pyannote version: {pyannote_ver})..."
            )

            # Support loading from a local directory (air-gapped)
            if os.path.isdir(model_id):
                self._diarization_pipeline = Pipeline.from_pretrained(model_id)
            elif is_v4_plus:
                # pyannote 4+ uses token= kwarg
                self._diarization_pipeline = Pipeline.from_pretrained(
                    model_id, token=token,
                )
            else:
                # pyannote 3.x uses use_auth_token= kwarg
                try:
                    self._diarization_pipeline = Pipeline.from_pretrained(
                        model_id, use_auth_token=token,
                    )
                except TypeError:
                    # Some intermediate versions may already use token=
                    self._diarization_pipeline = Pipeline.from_pretrained(
                        model_id, token=token,
                    )

            # Move to GPU if available
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    self._diarization_pipeline.to(_torch.device("cuda"))
                    logger.info("Diarization pipeline using GPU")
            except (ImportError, Exception):
                pass

            logger.info(
                f"pyannote diarization pipeline '{model_id}' loaded successfully"
            )
            return True

        finally:
            # Restore original torch.load immediately
            if torch_load_patched and orig_torch_load is not None:
                try:
                    import torch
                    torch.load = orig_torch_load
                    logger.debug("Restored original torch.load")
                except ImportError:
                    pass

    def _apply_huggingface_hub_shim(self):
        """Patch huggingface_hub for pyannote 3.x use_auth_token compat.

        huggingface_hub v1.0 removed ``use_auth_token`` in favor of
        ``token``.  pyannote 3.x still passes the old kwarg.
        Only applied when pyannote < 4.0.
        """
        try:
            import huggingface_hub as _hfhub

            _orig_dl = _hfhub.hf_hub_download

            def _patched_dl(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    val = kwargs.pop("use_auth_token")
                    if val is not None:
                        kwargs["token"] = val
                return _orig_dl(*args, **kwargs)

            if _orig_dl is not _patched_dl:
                _hfhub.hf_hub_download = _patched_dl
                if hasattr(_hfhub, "file_download"):
                    _hfhub.file_download.hf_hub_download = _patched_dl
                logger.debug(
                    "Applied huggingface_hub use_auth_token shim (pyannote <4.0)"
                )
        except ImportError:
            pass

    def _log_diarization_error(self, error: Exception):
        """Log a diarization loading error with actionable guidance.

        Distinguishes between:
        - Token missing → Settings guidance
        - 403 GatedRepo → HuggingFace model acceptance page
        - Other errors → generic troubleshooting
        """
        error_str = str(error)
        model_id = self._diarization_model

        if "401" in error_str or "Unauthorized" in error_str:
            logger.warning(
                f"Diarization pipeline auth failed: invalid or expired HF token. "
                f"Update it in Settings → API Keys."
            )
        elif "403" in error_str or "GatedRepo" in error_str or "gated" in error_str.lower():
            logger.warning(
                f"Diarization pipeline access denied: you must accept the model "
                f"terms at https://huggingface.co/{model_id} before using it. "
                f"Visit the model page, accept the conditions, then retry."
            )
        elif "404" in error_str or "not found" in error_str.lower():
            logger.warning(
                f"Diarization pipeline not found: '{model_id}'. "
                f"Check the model identifier is correct."
            )
        else:
            logger.warning(
                f"Failed to load diarization pipeline '{model_id}': {error}. "
                f"Make sure you've accepted the model terms at "
                f"https://huggingface.co/{model_id}"
            )

    # ------------------------------------------------------------------
    # Diarization execution
    # ------------------------------------------------------------------

    def _diarize(self, audio_path: Path) -> Optional[List[Tuple[float, float, str]]]:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of (start, end, speaker_label) tuples, or None if
            diarization is unavailable or fails.
        """
        if not self._ensure_diarization_loaded():
            return None

        try:
            logger.info(f"Running speaker diarization on {audio_path.name}...")
            diarization = self._diarization_pipeline(str(audio_path))

            # Convert pyannote output to simple list of (start, end, speaker)
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append((turn.start, turn.end, speaker))

            # Map raw speaker IDs (SPEAKER_00, SPEAKER_01) to friendly labels
            speaker_map = {}
            label_idx = 0
            for _, _, speaker in turns:
                if speaker not in speaker_map:
                    if label_idx < len(_SPEAKER_LABELS):
                        speaker_map[speaker] = f"Speaker {_SPEAKER_LABELS[label_idx]}"
                    else:
                        speaker_map[speaker] = f"Speaker {label_idx + 1}"
                    label_idx += 1

            friendly_turns = [
                (start, end, speaker_map[speaker])
                for start, end, speaker in turns
            ]

            logger.info(
                f"Diarization complete: {len(friendly_turns)} turns, "
                f"{len(speaker_map)} speakers detected"
            )
            return friendly_turns

        except Exception as e:
            logger.warning(f"Diarization failed for {audio_path.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Speaker assignment (overlap + midpoint proximity)
    # ------------------------------------------------------------------

    def _assign_speakers_to_segments(
        self,
        segments: List[Dict],
        diarization_turns: List[Tuple[float, float, str]],
    ) -> List[Dict]:
        """Align whisper segments with diarization speaker labels.

        Uses a two-tier strategy:
        1. When a diarization turn overlaps a segment by more than
           ``_OVERLAP_THRESHOLD_S`` (100 ms), the turn with the greatest
           overlap wins.
        2. When overlap is below the threshold (boundary jitter, rapid
           turns), the *nearest* diarization turn by midpoint proximity
           is chosen instead of defaulting to a fixed speaker.

        Args:
            segments: Whisper segments with start/end/text
            diarization_turns: (start, end, speaker) from pyannote

        Returns:
            Same segments list with added 'speaker' key
        """
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_mid = (seg_start + seg_end) / 2.0

            best_speaker = None
            best_score = -1.0

            for turn_start, turn_end, speaker in diarization_turns:
                # Calculate overlap duration
                overlap_start = max(seg_start, turn_start)
                overlap_end = min(seg_end, turn_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > _OVERLAP_THRESHOLD_S:
                    # Primary strategy: largest overlap wins
                    score = overlap
                else:
                    # Fallback: inverse midpoint distance (closer = higher score)
                    # This handles boundary jitter and near-misses gracefully
                    turn_mid = (turn_start + turn_end) / 2.0
                    distance = abs(seg_mid - turn_mid)
                    # Use a small score so overlap always wins when significant
                    score = _OVERLAP_THRESHOLD_S / (1.0 + distance)

                if score > best_score:
                    best_score = score
                    best_speaker = speaker

            seg["speaker"] = best_speaker or "Unknown Speaker"

        return segments

    def _format_text_with_speakers(self, segments: List[Dict]) -> str:
        """Format transcript text with speaker labels.

        Groups consecutive segments by the same speaker into paragraphs.

        Args:
            segments: Segments with 'speaker' and 'text' keys

        Returns:
            Formatted transcript string like:
            Speaker A: Hello, how are you?
            Speaker B: I'm fine, thanks.
        """
        if not segments:
            return ""

        lines = []
        current_speaker = None
        current_texts = []

        for seg in segments:
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            if not text:
                continue

            if speaker != current_speaker:
                # Flush previous speaker's text
                if current_speaker and current_texts:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [text]
            else:
                current_texts.append(text)

        # Flush last speaker
        if current_speaker and current_texts:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Audio validation
    # ------------------------------------------------------------------

    def _validate_audio(self, path: Path) -> None:
        """Pre-validate that an audio file contains decodable audio.

        Uses ffprobe to check the file has an audio stream with non-zero
        duration.  This catches corrupt, empty, or zero-byte files
        *before* they hit the model.

        Args:
            path: Path to the audio file

        Raises:
            ValueError: If the file has no decodable audio
        """
        # Quick size check
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(
                f"Audio file is empty (0 bytes): {path.name}"
            )

        # Use ffprobe to check for audio stream and duration
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = result.stdout.strip()
            if not output or output == "N/A":
                # No duration means ffprobe couldn't find an audio stream
                # Try format-level duration as fallback
                result2 = subprocess.run(
                    [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = result2.stdout.strip()

            if not output or output == "N/A":
                raise ValueError(
                    f"Audio file contains no decodable audio stream: {path.name}. "
                    f"The file may be corrupt or in an unsupported format."
                )

            duration = float(output)
            if duration < 0.1:
                raise ValueError(
                    f"Audio file is too short to transcribe ({duration:.2f}s): "
                    f"{path.name}. Minimum ~0.1 seconds of audio required."
                )

            logger.debug(f"Audio validated: {path.name} — {duration:.1f}s")

        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe timed out for {path.name} — skipping validation")
        except FileNotFoundError:
            logger.debug("ffprobe not found — skipping audio pre-validation")
        except ValueError:
            raise  # re-raise our own ValueErrors
        except Exception as e:
            logger.debug(f"Audio validation failed for {path.name}: {e} — proceeding anyway")

    # ------------------------------------------------------------------
    # Main transcription entry point
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file using faster-whisper.

        When diarization is enabled and available, also runs speaker
        diarization and labels each segment with a speaker identifier.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Force a specific language code (e.g., "en", "he").
                ``None`` lets Whisper auto-detect.
            beam_size: Beam size for decoding (1 = greedy, 5 = default beam).
            word_timestamps: If True, return word-level timestamps.
                Useful for finer-grained diarization alignment.
            initial_prompt: Optional prompt text to condition the model.
                Useful for domain vocabulary or spelling hints.
            on_progress: Optional callback ``fn(message: str)`` called
                periodically to report transcription progress.  Used by
                the sync layer to write live progress to the DB so the
                UI can display it.

        Returns:
            TranscriptionResult with full text (speaker-labeled when
            diarization is active), language, segments, and confidence.

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ImportError: If faster-whisper is not installed
            ValueError: If the audio file contains no decodable audio
        """
        _report = on_progress or (lambda _msg: None)

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Pre-validate: check audio stream exists and has duration
        _report("Validating audio…")
        self._validate_audio(path)

        _report("Loading model…")
        self._ensure_model_loaded()

        file_size_kb = path.stat().st_size / 1024
        logger.info(f"Transcribing: {path.name} ({file_size_kb:.0f} KB)")
        _report("Starting transcription…")

        # Build transcription kwargs
        transcribe_kwargs = dict(
            beam_size=beam_size,
            condition_on_previous_text=False,  # avoid hallucination loops
            vad_filter=True,          # skip silence — big speedup on recordings
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=word_timestamps,
        )
        if language:
            transcribe_kwargs["language"] = language
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        # Run faster-whisper transcription
        try:
            segments_gen, info = self._model.transcribe(
                str(path),
                **transcribe_kwargs,
            )
        except Exception as e:
            error_msg = str(e)
            if "cannot reshape" in error_msg or "empty" in error_msg.lower():
                raise ValueError(
                    f"Audio file contains no processable audio data: {path.name}. "
                    f"The file may be corrupt, empty, or in an unsupported format."
                ) from e
            raise

        # Total audio duration from info (used for progress percentage)
        audio_duration = info.duration if info.duration else 0

        # Consume the segment generator and collect results
        segments = []
        total_confidence = 0.0
        total_duration = 0.0
        last_progress_time = time.monotonic()
        _PROGRESS_INTERVAL_S = 3.0  # Throttle DB writes to every 3s

        for seg in segments_gen:
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }

            # Include word-level data when requested
            if word_timestamps and seg.words:
                segment_data["words"] = [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]

            segments.append(segment_data)

            # Confidence from avg_logprob — a practical heuristic, not a
            # calibrated probability.  exp(avg_logprob) roughly correlates
            # with segment-level confidence per Whisper maintainers.
            avg_logprob = seg.avg_logprob  # may be None or float (including 0.0)
            if avg_logprob is not None:
                segment_confidence = math.exp(avg_logprob)
            else:
                segment_confidence = 0.5  # missing data sentinel
            total_confidence += segment_confidence

            if seg.end > total_duration:
                total_duration = seg.end

            # Report progress (throttled)
            now = time.monotonic()
            if now - last_progress_time >= _PROGRESS_INTERVAL_S:
                last_progress_time = now
                pos_s = int(seg.end)
                if audio_duration > 0:
                    pct = min(99, int(seg.end / audio_duration * 100))
                    _report(
                        f"Transcribing: {pos_s}s / {int(audio_duration)}s ({pct}%) "
                        f"— {len(segments)} segments"
                    )
                else:
                    _report(f"Transcribing: {pos_s}s — {len(segments)} segments")

        if not segments:
            logger.warning(f"Whisper returned no segments for {path.name}")
            return TranscriptionResult(text="", confidence=0.0)

        _report(f"Transcription done — {len(segments)} segments, post-processing…")

        # Language from detection info
        language_detected = info.language

        # Average confidence across segments
        avg_confidence = (
            total_confidence / len(segments) if segments else 0.5
        )
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        # Use info.duration if available (more accurate), fall back to segment end
        duration_seconds = int(info.duration if info.duration else total_duration)

        # --- Speaker diarization (optional) ---
        speakers_detected = 0
        if self._enable_diarization:
            _report(f"Running speaker diarization ({duration_seconds}s audio)…")
            diarization_turns = self._diarize(path)
            if diarization_turns:
                segments = self._assign_speakers_to_segments(segments, diarization_turns)
                # Count unique speakers
                speakers_detected = len(set(
                    seg.get("speaker", "") for seg in segments
                ))
                # Build speaker-labeled text
                text = self._format_text_with_speakers(segments)
                _report(f"Diarization complete — {speakers_detected} speakers")
            else:
                # Diarization unavailable or failed — plain text
                text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()
        else:
            text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()

        if not text:
            logger.warning(f"Whisper returned empty transcription for {path.name}")
            return TranscriptionResult(text="", confidence=0.0)

        diar_info = f", {speakers_detected} speakers" if speakers_detected else ""
        logger.info(
            f"Transcription complete: {path.name} — "
            f"{len(text)} chars, {duration_seconds}s, "
            f"lang={language_detected}, confidence={avg_confidence:.2f}{diar_info}"
        )

        return TranscriptionResult(
            text=text,
            language=language_detected,
            duration_seconds=duration_seconds,
            segments=segments,
            confidence=avg_confidence,
            speakers_detected=speakers_detected,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_model(self) -> None:
        """Unload the Whisper model and diarization pipeline to free memory.

        Called during plugin shutdown.
        """
        if self._model is not None:
            logger.info(f"Unloading Whisper model '{self._model_size}'")
            del self._model
            self._model = None

        if self._diarization_pipeline is not None:
            logger.info("Unloading diarization pipeline")
            del self._diarization_pipeline
            self._diarization_pipeline = None

        # Try to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
