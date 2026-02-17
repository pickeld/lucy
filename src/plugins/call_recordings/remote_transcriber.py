"""Remote transcription via AssemblyAI API.

Provides the same ``transcribe()`` interface as ``WhisperTranscriber``
so it can be used as a drop-in replacement in ``CallRecordingSyncer``.

AssemblyAI advantages over local Whisper:
- Transcription + speaker diarization in a single API call
- Auto language detection
- No local GPU/CPU requirements — works on any hardware
- Models: universal-3-pro (best, $0.12/min), universal-2 ($0.015/min)

Requires: ``pip install assemblyai``
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Speaker label format (matches WhisperTranscriber convention)
_SPEAKER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _friendly_speaker(raw: str) -> str:
    """Convert AssemblyAI speaker label (e.g. 'A') to 'Speaker A'."""
    if raw and len(raw) == 1 and raw.isalpha():
        return f"Speaker {raw.upper()}"
    return f"Speaker {raw}" if raw else "Unknown Speaker"


class AssemblyAITranscriber:
    """Remote transcriber using AssemblyAI's speech-to-text API.

    Provides the same ``transcribe()`` / ``unload_model()`` interface
    as ``WhisperTranscriber`` so ``CallRecordingSyncer`` can use either
    transparently.

    Features:
    - Built-in speaker diarization (no pyannote needed)
    - Auto language detection
    - Word-level timestamps and confidence scores

    Args:
        api_key: AssemblyAI API key
        model: Speech model — "universal-3-pro" (best) or "universal-2" (fast/cheap)
        language: Force language code (e.g. "en", "he"). None for auto-detection.
        enable_diarization: If True, enable speaker diarization
    """

    def __init__(
        self,
        api_key: str,
        model: str = "universal-2",
        language: Optional[str] = None,
        enable_diarization: bool = True,
    ):
        self._api_key = api_key
        self._model = model
        self._language = language or None
        self._enable_diarization = enable_diarization

    @property
    def model_size(self) -> str:
        """Return the AssemblyAI model name (used in metadata)."""
        return f"assemblyai-{self._model}"

    @property
    def diarization_available(self) -> bool:
        """Whether diarization is enabled (always available with AssemblyAI)."""
        return self._enable_diarization

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        beam_size: int = 5,        # ignored — API parameter
        word_timestamps: bool = False,  # ignored — always available
        initial_prompt: Optional[str] = None,  # ignored — not supported
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        """Transcribe an audio file via AssemblyAI API.

        Same signature as ``WhisperTranscriber.transcribe()`` for
        drop-in compatibility.  API-only parameters (beam_size,
        initial_prompt) are accepted but ignored.

        Args:
            audio_path: Path to the audio file
            language: Override language code (or None for auto-detect)
            on_progress: Progress callback ``fn(message: str)``

        Returns:
            TranscriptionResult with text, language, segments, speakers

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ImportError: If assemblyai package not installed
            RuntimeError: On API errors
        """
        # Import here to avoid hard dependency at module level
        from plugins.call_recordings.transcriber import TranscriptionResult

        _report = on_progress or (lambda _msg: None)

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size_kb = path.stat().st_size / 1024
        logger.info(
            f"AssemblyAI transcription: {path.name} ({file_size_kb:.0f} KB) "
            f"model={self._model}"
        )

        _report("Connecting to AssemblyAI…")

        try:
            import assemblyai as aai
        except ImportError:
            raise ImportError(
                "assemblyai package not installed. "
                "Install with: pip install assemblyai"
            )

        # Configure API key
        aai.settings.api_key = self._api_key

        # Build transcription config
        lang = language or self._language
        speech_models = [self._model] if self._model else None

        config_kwargs: Dict = {
            "speaker_labels": self._enable_diarization,
            "language_detection": lang is None,  # auto-detect when no language forced
        }
        if speech_models:
            config_kwargs["speech_models"] = speech_models
        if lang:
            config_kwargs["language_code"] = lang

        config = aai.TranscriptionConfig(**config_kwargs)

        _report(f"Uploading {path.name} to AssemblyAI…")

        # Submit transcription (upload + process)
        transcriber = aai.Transcriber()

        # Start transcription
        start_time = time.monotonic()
        transcript = transcriber.transcribe(str(path), config)
        elapsed = time.monotonic() - start_time

        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            error_msg = transcript.error or "Unknown AssemblyAI error"
            logger.error(f"AssemblyAI transcription failed: {error_msg}")
            raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")

        _report("Processing results…")

        # Extract results
        full_text = transcript.text or ""
        if not full_text.strip():
            logger.warning(f"AssemblyAI returned empty transcription for {path.name}")
            return TranscriptionResult(text="", confidence=0.0)

        # Language detection
        language_detected = None
        if hasattr(transcript, "language_code") and transcript.language_code:
            language_detected = transcript.language_code

        # Duration (AssemblyAI returns milliseconds)
        duration_ms = getattr(transcript, "audio_duration", 0) or 0
        duration_seconds = int(duration_ms / 1000) if duration_ms > 1000 else int(duration_ms)

        # Build segments from words (for compatibility with existing pipeline)
        segments: List[Dict] = []
        avg_confidence = 0.0

        if transcript.words:
            total_conf = sum(
                w.confidence for w in transcript.words if w.confidence
            )
            avg_confidence = total_conf / len(transcript.words) if transcript.words else 0.5

            # Group words into segments (by sentence/pause boundaries)
            # Use utterances if available (diarization), otherwise use words
            if self._enable_diarization and transcript.utterances:
                for utt in transcript.utterances:
                    segments.append({
                        "start": utt.start / 1000.0,  # ms → s
                        "end": utt.end / 1000.0,
                        "text": utt.text.strip(),
                        "speaker": _friendly_speaker(utt.speaker),
                    })
            else:
                # No diarization — group words into sentence-like segments
                current_segment: Dict = {}
                for word in transcript.words:
                    if not current_segment:
                        current_segment = {
                            "start": word.start / 1000.0,
                            "end": word.end / 1000.0,
                            "text": word.text,
                        }
                    else:
                        current_segment["end"] = word.end / 1000.0
                        current_segment["text"] += " " + word.text

                        # Split on sentence boundaries or long gaps
                        gap = (word.start - (current_segment.get("_prev_end", word.start))) / 1000.0
                        text = current_segment["text"]
                        if (
                            text.rstrip().endswith((".", "?", "!"))
                            or len(text) > 200
                            or gap > 1.5
                        ):
                            segments.append(current_segment)
                            current_segment = {}
                    current_segment["_prev_end"] = word.end

                if current_segment:
                    current_segment.pop("_prev_end", None)
                    segments.append(current_segment)

        # Clean up _prev_end from segments
        for seg in segments:
            seg.pop("_prev_end", None)

        # Count speakers
        speakers_detected = 0
        if self._enable_diarization and transcript.utterances:
            speakers_detected = len(set(
                utt.speaker for utt in transcript.utterances if utt.speaker
            ))

        # Format text with speaker labels
        if speakers_detected > 0 and transcript.utterances:
            full_text = self._format_text_with_speakers(transcript.utterances)
            _report(f"Transcription complete — {speakers_detected} speakers detected")
        else:
            _report("Transcription complete")

        logger.info(
            f"AssemblyAI transcription complete: {path.name} — "
            f"{len(full_text)} chars, {duration_seconds}s, "
            f"lang={language_detected}, confidence={avg_confidence:.2f}, "
            f"speakers={speakers_detected}, elapsed={elapsed:.1f}s"
        )

        return TranscriptionResult(
            text=full_text,
            language=language_detected,
            duration_seconds=duration_seconds,
            segments=segments,
            confidence=max(0.0, min(1.0, avg_confidence)),
            speakers_detected=speakers_detected,
        )

    @staticmethod
    def _format_text_with_speakers(utterances) -> str:
        """Format transcript text with speaker labels from AssemblyAI utterances.

        Groups consecutive utterances by the same speaker into paragraphs.

        Returns:
            Formatted transcript string like:
            Speaker A: Hello, how are you?
            Speaker B: I'm fine, thanks.
        """
        if not utterances:
            return ""

        lines = []
        current_speaker = None
        current_texts: List[str] = []

        for utt in utterances:
            speaker = _friendly_speaker(utt.speaker)
            text = utt.text.strip()
            if not text:
                continue

            if speaker != current_speaker:
                if current_speaker and current_texts:
                    lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [text]
            else:
                current_texts.append(text)

        if current_speaker and current_texts:
            lines.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n".join(lines)

    def unload_model(self) -> None:
        """No-op — no local model to unload."""
        pass
