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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid Whisper model sizes
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

# Default model size — balance between accuracy and speed
DEFAULT_MODEL_SIZE = "medium"

# Compute types ordered by preference per device
_CPU_COMPUTE_TYPE = "int8"       # fastest on CPU, minimal quality loss
_CUDA_COMPUTE_TYPE = "float16"   # optimal for GPU

# Speaker label format
_SPEAKER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _is_pyannote_available() -> bool:
    """Check if pyannote.audio is installed and importable."""
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription of an audio file.

    Attributes:
        text: Full transcription text (with speaker labels when diarization is enabled)
        language: Detected language code (e.g., "en", "he", "ar")
        duration_seconds: Audio duration in seconds
        segments: Timestamped text segments from Whisper (with optional 'speaker' key)
        confidence: Average confidence score (0.0 to 1.0)
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
    request and caches it for subsequent calls.  Automatically uses
    GPU (CUDA) if available, otherwise falls back to CPU with int8
    quantization for maximum throughput.

    When diarization is enabled and pyannote.audio is available, runs
    speaker diarization after transcription to label segments with
    speaker identifiers (Speaker A, Speaker B, etc.).

    Args:
        model_size: Whisper model size — "small", "medium", or "large".
        hf_token: HuggingFace token for model downloads.
        enable_diarization: If True, run speaker diarization when available.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        hf_token: Optional[str] = None,
        enable_diarization: bool = False,
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

    @property
    def model_size(self) -> str:
        """Currently configured model size."""
        return self._model_size

    @property
    def diarization_available(self) -> bool:
        """Whether diarization is enabled and pyannote is available."""
        return self._enable_diarization and _is_pyannote_available() and bool(self._hf_token)

    def _ensure_model_loaded(self):
        """Lazily load the Whisper model on first use.

        Caches the model instance for subsequent transcriptions.
        Uses GPU (float16) if CUDA is available, otherwise CPU (int8).
        """
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )

        # Detect device and pick optimal compute type
        device = "cpu"
        compute_type = _CPU_COMPUTE_TYPE
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = _CUDA_COMPUTE_TYPE
                logger.info("CUDA available — using GPU for Whisper")
            else:
                logger.info(
                    f"CUDA not available — using CPU with {compute_type} quantization"
                )
        except ImportError:
            logger.info(
                f"torch not available for device detection — "
                f"using CPU with {compute_type} quantization"
            )

        # Set HF_TOKEN for authenticated model downloads (higher rate limits)
        token = self._resolve_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
            logger.info("HF_TOKEN set for model download")

        # Use all available CPU cores for CTranslate2 inference
        import multiprocessing
        cpu_threads = multiprocessing.cpu_count()

        logger.info(
            f"Loading faster-whisper model '{self._model_size}' "
            f"on {device} ({compute_type}, {cpu_threads} threads)..."
        )
        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=min(cpu_threads, 4),  # parallel decoding workers
        )
        logger.info(
            f"faster-whisper model '{self._model_size}' loaded successfully "
            f"({cpu_threads} threads)"
        )

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

    def _ensure_diarization_loaded(self) -> bool:
        """Lazily load the pyannote diarization pipeline.

        Returns True if the pipeline is ready, False if unavailable.
        """
        if self._diarization_pipeline is not None:
            return True

        if not self._enable_diarization:
            return False

        token = self._resolve_hf_token()
        if not token:
            logger.warning(
                "Diarization enabled but HF_TOKEN not set — "
                "set it in Settings → API Keys to enable speaker detection"
            )
            return False

        try:
            # Compatibility shim: PyTorch 2.6+ defaults torch.load to
            # weights_only=True, which breaks pyannote model loading.
            # Patch torch.load to use weights_only=False as default.
            try:
                import torch
                _orig_torch_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    if "weights_only" not in kwargs:
                        kwargs["weights_only"] = False
                    return _orig_torch_load(*args, **kwargs)
                torch.load = _patched_torch_load
                logger.debug("Applied torch.load weights_only compatibility shim")
            except ImportError:
                pass

            # Compatibility shim: pyannote.audio references torchaudio APIs
            # that were removed in torchaudio >= 2.6.  Patch them before import.
            try:
                import torchaudio

                _patched = []

                # AudioMetaData was removed — create a placeholder
                if not hasattr(torchaudio, "AudioMetaData"):
                    _dummy_info_cls = getattr(torchaudio, "AudioInfo", None)
                    if _dummy_info_cls is None:
                        from dataclasses import make_dataclass
                        _dummy_info_cls = make_dataclass(
                            "AudioMetaData",
                            [
                                ("sample_rate", int, 0),
                                ("num_frames", int, 0),
                                ("num_channels", int, 0),
                                ("bits_per_sample", int, 0),
                                ("encoding", str, ""),
                            ],
                        )
                    torchaudio.AudioMetaData = _dummy_info_cls
                    _patched.append("AudioMetaData")

                # list_audio_backends was removed — return empty list
                if not hasattr(torchaudio, "list_audio_backends"):
                    torchaudio.list_audio_backends = lambda: ["soundfile"]
                    _patched.append("list_audio_backends")

                # get_audio_backend was removed
                if not hasattr(torchaudio, "get_audio_backend"):
                    torchaudio.get_audio_backend = lambda: "soundfile"
                    _patched.append("get_audio_backend")

                if _patched:
                    logger.debug(
                        f"Applied torchaudio compatibility shims: {', '.join(_patched)}"
                    )
            except ImportError:
                pass

            # Compatibility shim: pyannote.audio 3.x passes deprecated
            # use_auth_token to huggingface_hub which removed it in v1.0.
            # Patch hf_hub_download to accept and convert the old kwarg.
            try:
                import huggingface_hub as _hfhub

                _orig_dl = _hfhub.hf_hub_download
                def _patched_dl(*args, **kwargs):
                    if "use_auth_token" in kwargs:
                        val = kwargs.pop("use_auth_token")
                        # Only set token if explicitly provided (not None)
                        # so the HF_TOKEN env var can still be used as fallback
                        if val is not None:
                            kwargs["token"] = val
                    return _orig_dl(*args, **kwargs)

                if _orig_dl is not _patched_dl:
                    _hfhub.hf_hub_download = _patched_dl
                    # Also patch the internal module copy that pyannote may import from
                    if hasattr(_hfhub, "file_download"):
                        _hfhub.file_download.hf_hub_download = _patched_dl
                    logger.debug("Applied huggingface_hub use_auth_token compatibility shim")
            except ImportError:
                pass

            from pyannote.audio import Pipeline

            # Ensure HF_TOKEN is in env — huggingface_hub reads it automatically
            os.environ["HF_TOKEN"] = token

            logger.info("Loading pyannote speaker diarization pipeline...")
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )

            # Move to GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    self._diarization_pipeline.to(torch.device("cuda"))
                    logger.info("Diarization pipeline using GPU")
            except (ImportError, Exception):
                pass

            logger.info("pyannote diarization pipeline loaded successfully")
            return True

        except ImportError:
            logger.info(
                "pyannote.audio not installed — speaker diarization disabled. "
                "Install with: pip install pyannote.audio"
            )
            return False
        except Exception as e:
            logger.warning(
                f"Failed to load diarization pipeline: {e}. "
                f"Make sure you've accepted the model terms at "
                f"https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            return False

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

    def _assign_speakers_to_segments(
        self,
        segments: List[Dict],
        diarization_turns: List[Tuple[float, float, str]],
    ) -> List[Dict]:
        """Align whisper segments with diarization speaker labels.

        For each whisper segment, finds the diarization turn with the
        greatest time overlap and assigns that turn's speaker label.

        Args:
            segments: Whisper segments with start/end/text
            diarization_turns: (start, end, speaker) from pyannote

        Returns:
            Same segments list with added 'speaker' key
        """
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            best_speaker = None
            best_overlap = 0.0

            for turn_start, turn_end, speaker in diarization_turns:
                # Calculate overlap
                overlap_start = max(seg_start, turn_start)
                overlap_end = min(seg_end, turn_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            seg["speaker"] = best_speaker or "Speaker A"

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

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using faster-whisper.

        When diarization is enabled and available, also runs speaker
        diarization and labels each segment with a speaker identifier.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            TranscriptionResult with full text (speaker-labeled when
            diarization is active), language, segments, and confidence.

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ImportError: If faster-whisper is not installed
            ValueError: If the audio file contains no decodable audio
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Pre-validate: check audio stream exists and has duration
        self._validate_audio(path)

        self._ensure_model_loaded()

        logger.info(f"Transcribing: {path.name} ({path.stat().st_size / 1024:.0f} KB)")

        # Run faster-whisper transcription
        # beam_size=5 is the default (good accuracy); use 1 for greedy (faster)
        # condition_on_previous_text=False avoids hallucination loops and is faster
        try:
            segments_gen, info = self._model.transcribe(
                str(path),
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,          # skip silence — big speedup on recordings
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )
        except Exception as e:
            error_msg = str(e)
            if "cannot reshape" in error_msg or "empty" in error_msg.lower():
                raise ValueError(
                    f"Audio file contains no processable audio data: {path.name}. "
                    f"The file may be corrupt, empty, or in an unsupported format."
                ) from e
            raise

        # Consume the segment generator and collect results
        segments = []
        total_confidence = 0.0
        total_duration = 0.0

        for seg in segments_gen:
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            segments.append(segment_data)

            # Confidence from avg_logprob
            avg_logprob = seg.avg_logprob or 0.0
            segment_confidence = math.exp(avg_logprob) if avg_logprob else 0.5
            total_confidence += segment_confidence

            if seg.end > total_duration:
                total_duration = seg.end

        if not segments:
            logger.warning(f"Whisper returned no segments for {path.name}")
            return TranscriptionResult(text="", confidence=0.0)

        # Language from detection info
        language = info.language

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
            diarization_turns = self._diarize(path)
            if diarization_turns:
                segments = self._assign_speakers_to_segments(segments, diarization_turns)
                # Count unique speakers
                speakers_detected = len(set(
                    seg.get("speaker", "") for seg in segments
                ))
                # Build speaker-labeled text
                text = self._format_text_with_speakers(segments)
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
            f"lang={language}, confidence={avg_confidence:.2f}{diar_info}"
        )

        return TranscriptionResult(
            text=text,
            language=language,
            duration_seconds=duration_seconds,
            segments=segments,
            confidence=avg_confidence,
            speakers_detected=speakers_detected,
        )

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
