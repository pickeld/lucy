"""Local Whisper transcription for audio call recordings.

Uses the faster-whisper package (CTranslate2 backend) for local
audio-to-text transcription.  Typically 4-6× faster than the original
openai-whisper on CPU, with identical accuracy.

The Whisper model is loaded lazily on first use and cached for
subsequent calls to avoid slow startup times.
"""

import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid Whisper model sizes
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

# Default model size — balance between accuracy and speed
DEFAULT_MODEL_SIZE = "medium"

# Compute types ordered by preference per device
_CPU_COMPUTE_TYPE = "int8"       # fastest on CPU, minimal quality loss
_CUDA_COMPUTE_TYPE = "float16"   # optimal for GPU


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription of an audio file.

    Attributes:
        text: Full transcription text
        language: Detected language code (e.g., "en", "he", "ar")
        duration_seconds: Audio duration in seconds
        segments: Timestamped text segments from Whisper
        confidence: Average confidence score (0.0 to 1.0)
    """

    text: str
    language: Optional[str] = None
    duration_seconds: int = 0
    segments: List[Dict] = field(default_factory=list)
    confidence: float = 1.0


class WhisperTranscriber:
    """Local Whisper transcriber using faster-whisper (CTranslate2).

    Loads the specified Whisper model lazily on first transcription
    request and caches it for subsequent calls.  Automatically uses
    GPU (CUDA) if available, otherwise falls back to CPU with int8
    quantization for maximum throughput.

    Args:
        model_size: Whisper model size — "small", "medium", or "large".
            Larger models are more accurate but slower and use more memory.

            Approximate performance (faster-whisper on CPU int8):
            - small:  ~1 GB RAM, fastest, good for clear audio
            - medium: ~2.5 GB RAM, balanced accuracy/speed
            - large:  ~5 GB RAM, best accuracy, slowest
    """

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, hf_token: Optional[str] = None):
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

    @property
    def model_size(self) -> str:
        """Currently configured model size."""
        return self._model_size

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
        if self._hf_token:
            import os
            os.environ["HF_TOKEN"] = self._hf_token
            logger.info("HF_TOKEN set for model download")

        logger.info(
            f"Loading faster-whisper model '{self._model_size}' "
            f"on {device} ({compute_type})..."
        )
        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            f"faster-whisper model '{self._model_size}' loaded successfully"
        )

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

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            TranscriptionResult with full text, language, segments,
            duration, and confidence score

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
        text_parts = []
        total_confidence = 0.0
        total_duration = 0.0

        for seg in segments_gen:
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            segments.append(segment_data)
            text_parts.append(seg.text.strip())

            # Confidence from avg_logprob
            avg_logprob = seg.avg_logprob or 0.0
            segment_confidence = math.exp(avg_logprob) if avg_logprob else 0.5
            total_confidence += segment_confidence

            if seg.end > total_duration:
                total_duration = seg.end

        # Build full text
        text = " ".join(text_parts).strip()

        if not text:
            logger.warning(f"Whisper returned empty transcription for {path.name}")
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

        logger.info(
            f"Transcription complete: {path.name} — "
            f"{len(text)} chars, {duration_seconds}s, "
            f"lang={language}, confidence={avg_confidence:.2f}"
        )

        return TranscriptionResult(
            text=text,
            language=language,
            duration_seconds=duration_seconds,
            segments=segments,
            confidence=avg_confidence,
        )

    def unload_model(self) -> None:
        """Unload the Whisper model to free memory.

        Called during plugin shutdown.
        """
        if self._model is not None:
            logger.info(f"Unloading Whisper model '{self._model_size}'")
            del self._model
            self._model = None

            # Try to free GPU memory
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
