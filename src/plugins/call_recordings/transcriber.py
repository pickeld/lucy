"""Local Whisper transcription for audio call recordings.

Uses the openai-whisper Python package for local audio-to-text
transcription with configurable model size (small, medium, large).

The Whisper model is loaded lazily on first use and cached for
subsequent calls to avoid slow startup times.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid Whisper model sizes
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large"}

# Default model size — balance between accuracy and speed
DEFAULT_MODEL_SIZE = "medium"


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
    """Local Whisper transcriber using openai-whisper package.

    Loads the specified Whisper model lazily on first transcription
    request and caches it for subsequent calls.  Automatically uses
    GPU (CUDA) if available, otherwise falls back to CPU.

    Args:
        model_size: Whisper model size — "small", "medium", or "large".
            Larger models are more accurate but slower and use more memory.

            Approximate performance:
            - small:  ~2 GB VRAM, fastest, good for clear audio
            - medium: ~5 GB VRAM, balanced accuracy/speed
            - large:  ~10 GB VRAM, best accuracy, slowest
    """

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        size = model_size.lower().strip()
        if size not in VALID_MODEL_SIZES:
            logger.warning(
                f"Invalid Whisper model size '{model_size}', "
                f"falling back to '{DEFAULT_MODEL_SIZE}'"
            )
            size = DEFAULT_MODEL_SIZE
        self._model_size = size
        self._model = None

    @property
    def model_size(self) -> str:
        """Currently configured model size."""
        return self._model_size

    def _ensure_model_loaded(self):
        """Lazily load the Whisper model on first use.

        Caches the model instance for subsequent transcriptions.
        Uses GPU if CUDA is available, otherwise CPU.
        """
        if self._model is not None:
            return

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. "
                "Install with: pip install openai-whisper"
            )

        # Detect device
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available — using GPU for Whisper")
            else:
                logger.info("CUDA not available — using CPU for Whisper")
        except ImportError:
            logger.info("torch not available for device detection — using CPU")

        logger.info(
            f"Loading Whisper model '{self._model_size}' on {device}..."
        )
        self._model = whisper.load_model(self._model_size, device=device)
        logger.info(f"Whisper model '{self._model_size}' loaded successfully")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file using local Whisper.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            TranscriptionResult with full text, language, segments,
            duration, and confidence score

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ImportError: If openai-whisper is not installed
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_model_loaded()

        logger.info(f"Transcribing: {path.name} ({path.stat().st_size / 1024:.0f} KB)")

        # Run Whisper transcription
        # fp16=False avoids noisy "FP16 is not supported on CPU" warning
        result = self._model.transcribe(
            str(path),
            verbose=False,
            fp16=False,
            # Enable word timestamps for better segment data
            word_timestamps=False,
        )

        # Extract text
        text = result.get("text", "").strip()
        if not text:
            logger.warning(f"Whisper returned empty transcription for {path.name}")
            return TranscriptionResult(text="", confidence=0.0)

        # Extract language
        language = result.get("language")

        # Extract segments with timestamps
        raw_segments = result.get("segments", [])
        segments = []
        total_confidence = 0.0
        total_duration = 0.0

        for seg in raw_segments:
            segment_data = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }
            segments.append(segment_data)

            # Accumulate for averages
            # Whisper provides avg_logprob per segment; convert to 0-1 confidence
            avg_logprob = seg.get("avg_logprob", 0.0)
            # logprob is negative; closer to 0 = more confident
            # Convert: exp(logprob) gives probability
            import math

            segment_confidence = math.exp(avg_logprob) if avg_logprob else 0.5
            total_confidence += segment_confidence

            seg_end = seg.get("end", 0.0)
            if seg_end > total_duration:
                total_duration = seg_end

        # Average confidence across segments
        avg_confidence = (
            total_confidence / len(segments) if segments else 0.5
        )
        # Clamp to [0, 1]
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        duration_seconds = int(total_duration)

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
