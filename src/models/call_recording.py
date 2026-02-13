"""Call recording document class for RAG system.

This module provides the CallRecordingDocument class for handling
transcribed call recordings in the RAG vector store using LlamaIndex.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator

from .base import BaseRAGDocument, ContentType, DocumentMetadata, Source, SourceType

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode


class CallType(str, Enum):
    """Types of call recordings."""
    
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    CONFERENCE = "conference"
    VOICEMAIL = "voicemail"
    UNKNOWN = "unknown"


class CallRecordingDocument(BaseRAGDocument):
    """Document class for transcribed call recordings.
    
    Extends BaseRAGDocument with call-specific fields for handling
    transcribed audio recordings from phone calls or voice messages.
    
    Attributes:
        recording_id: Unique identifier for the recording
        transcript: Full transcribed text of the call
        duration_seconds: Duration of the call in seconds
        participants: List of call participants
        call_type: Type of call (incoming, outgoing, conference)
        phone_number: Primary phone number associated with the call
        confidence_score: Transcription confidence (0.0 to 1.0)
        audio_file_path: Path to the original audio file
        audio_format: Format of the audio file (mp3, wav, etc.)
        transcription_provider: Provider used for transcription
        language_detected: Language detected in the audio
    """
    
    recording_id: str = Field(..., description="Unique recording identifier")
    transcript: str = Field(..., description="Full transcribed text")
    duration_seconds: int = Field(default=0, description="Call duration in seconds")
    participants: List[str] = Field(default_factory=list, description="Call participants")
    call_type: CallType = Field(default=CallType.UNKNOWN, description="Type of call")
    phone_number: Optional[str] = Field(default=None, description="Primary phone number")
    confidence_score: float = Field(default=1.0, description="Transcription confidence")
    audio_file_path: Optional[str] = Field(default=None, description="Path to audio file")
    audio_format: Optional[str] = Field(default=None, description="Audio format (mp3, wav)")
    transcription_provider: Optional[str] = Field(default=None, description="Transcription service used")
    language_detected: Optional[str] = Field(default=None, description="Detected language")
    
    @field_validator("confidence_score")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v
    
    @field_validator("duration_seconds")
    @classmethod
    def duration_must_be_non_negative(cls, v: int) -> int:
        """Validate duration is non-negative."""
        if v < 0:
            raise ValueError("Duration cannot be negative")
        return v
    
    @classmethod
    def get_source(cls) -> Source:
        """Get the default source for call recordings.
        
        Defaults to MANUAL; override via from_transcription(source=...)
        for recordings from specific plugins (e.g. Source.WHATSAPP).
        """
        return Source.MANUAL
    
    @classmethod
    def get_content_type(cls) -> ContentType:
        """Get the content type for call recordings."""
        return ContentType.CALL_RECORDING
    
    @classmethod
    def get_source_type(cls) -> SourceType:
        """DEPRECATED: Use get_source() instead."""
        return SourceType.CALL_RECORDING
    
    @classmethod
    def from_transcription(
        cls,
        recording_id: str,
        transcript: str,
        participants: List[str],
        duration_seconds: int,
        call_type: CallType = CallType.UNKNOWN,
        phone_number: Optional[str] = None,
        confidence_score: float = 1.0,
        audio_file_path: Optional[str] = None,
        audio_format: Optional[str] = None,
        transcription_provider: Optional[str] = None,
        language_detected: Optional[str] = None,
        recorded_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        source: Source = Source.MANUAL
    ) -> "CallRecordingDocument":
        """Create a CallRecordingDocument from transcription data.
        
        This factory method provides a convenient way to create documents
        from audio transcription pipelines.
        
        Args:
            recording_id: Unique identifier for the recording
            transcript: Transcribed text content
            participants: List of call participants
            duration_seconds: Call duration in seconds
            call_type: Type of call
            phone_number: Primary phone number
            confidence_score: Transcription confidence
            audio_file_path: Path to audio file
            audio_format: Audio format
            transcription_provider: Service used for transcription
            language_detected: Detected language
            recorded_at: When the call was recorded
            tags: Optional tags
            source: Where the recording came from (default: MANUAL)
            
        Returns:
            CallRecordingDocument instance
        """
        if recorded_at is None:
            recorded_at = datetime.now(ZoneInfo("UTC"))
        
        # Determine primary author from participants
        author = participants[0] if participants else "Unknown"
        
        # Create metadata
        metadata = DocumentMetadata(
            source_id=f"call:{recording_id}",
            source=source,
            content_type=ContentType.CALL_RECORDING,
            source_type=SourceType.CALL_RECORDING,  # Legacy compat
            created_at=recorded_at,
            tags=tags or [],
            language=language_detected,
            custom_fields={
                "recording_id": recording_id,
                "call_type": call_type.value,
                "duration_seconds": duration_seconds,
                "participants": participants,
                "confidence_score": confidence_score
            }
        )
        
        return cls(
            content=transcript,
            author=author,
            timestamp=recorded_at,
            metadata=metadata,
            recording_id=recording_id,
            transcript=transcript,
            duration_seconds=duration_seconds,
            participants=participants,
            call_type=call_type,
            phone_number=phone_number,
            confidence_score=confidence_score,
            audio_file_path=audio_file_path,
            audio_format=audio_format,
            transcription_provider=transcription_provider,
            language_detected=language_detected
        )
    
    def format_duration(self) -> str:
        """Format duration as human-readable string.
        
        Returns:
            Formatted duration (e.g., "5:30" or "1:23:45")
        """
        hours, remainder = divmod(self.duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    
    def to_searchable_content(self) -> str:
        """Format call recording for display in search results.
        
        Returns human-readable format with participants and duration.
        
        Returns:
            Formatted string for search result display
        """
        formatted_time = self.format_timestamp()
        duration = self.format_duration()
        participants_str = ", ".join(self.participants) if self.participants else "Unknown"
        
        return (
            f"[{formatted_time}] Call ({duration}) with {participants_str}: "
            f"{self.transcript[:200]}..."
        )
    
    def get_embedding_text(self) -> str:
        """Get optimized text for embedding generation.
        
        Includes participant and call context for better semantic search.
        
        Returns:
            Text optimized for embedding
        """
        parts = []
        
        # Add context header
        call_type_str = self.call_type.value.title()
        participants_str = ", ".join(self.participants) if self.participants else "Unknown"
        parts.append(f"{call_type_str} call with: {participants_str}")
        
        # Add duration context
        parts.append(f"Duration: {self.format_duration()}")
        
        # Add transcript
        parts.append(f"Transcript: {self.transcript}")
        
        return "\n\n".join(parts)
    
    def to_llama_index_node(self) -> "TextNode":
        """Convert to LlamaIndex TextNode with call-specific metadata.
        
        Adds call-specific fields to the standard metadata.
        
        Returns:
            LlamaIndex TextNode with full metadata
        """
        from llama_index.core.schema import TextNode
        
        # Get base metadata
        node_metadata = self.metadata.to_qdrant_payload()
        
        # Add call-specific fields
        node_metadata.update({
            "document_id": self.id,
            "author": self.author,
            "timestamp": int(self.timestamp.timestamp()),
            "recording_id": self.recording_id,
            "duration_seconds": self.duration_seconds,
            "participants": self.participants,
            "call_type": self.call_type.value,
            "phone_number": self.phone_number,
            "confidence_score": self.confidence_score,
            "audio_format": self.audio_format,
            "transcription_provider": self.transcription_provider,
            "language_detected": self.language_detected
        })
        
        return TextNode(
            text=self.get_embedding_text(),
            metadata=node_metadata,
            id_=self.id
        )
