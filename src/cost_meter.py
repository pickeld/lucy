"""In-memory cost meter with SQLite persistence.

Provides a thread-safe CostMeter singleton that:
- Tracks per-event costs in memory (fast lookups)
- Maintains a running session total
- Persists every event to SQLite for historical queries
- Supports snapshotting for per-query cost calculation

Usage:
    from cost_meter import METER, CostEvent
    from time import time

    METER.record_chat(
        provider="openai", model="gpt-4o",
        in_tokens=500, out_tokens=200,
        conversation_id="abc-123",
        request_context="rag_query",
    )

    print(f"Session total: ${METER.session_total:.4f}")
"""

import threading
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from pricing import chat_cost, embed_cost, image_cost, resolve_model_key, whisper_cost

# Maximum in-memory events to keep (oldest are evicted)
MAX_MEMORY_EVENTS = 1000


@dataclass
class CostEvent:
    """A single cost-tracking event."""

    ts: float                   # Unix timestamp
    provider: str               # "openai" | "gemini"
    model: str                  # "gpt-4o", "text-embedding-3-large", etc.
    kind: str                   # "chat" | "embed" | "whisper" | "image"
    in_tokens: int = 0          # Input/prompt tokens
    out_tokens: int = 0         # Output/completion tokens
    total_tokens: int = 0       # Total tokens (for embeddings)
    cost_usd: float = 0.0      # Calculated cost in USD
    conversation_id: str = ""   # Associated conversation UUID
    request_context: str = ""   # "rag_query" | "condense" | "synthesize" | "ingest" | "image_describe" | "transcribe"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict (for API responses)."""
        return {
            "ts": self.ts,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "conversation_id": self.conversation_id,
            "request_context": self.request_context,
        }


class CostMeter:
    """Thread-safe in-memory cost meter with SQLite persistence.

    Maintains a running session total and a bounded event buffer.
    Every event is also written to SQLite for historical queries.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._session_total: float = 0.0
        self._events: List[CostEvent] = []
        self._enabled: bool = True

    @property
    def session_total(self) -> float:
        """Current session cost total in USD."""
        with self._lock:
            return self._session_total

    @property
    def events(self) -> List[CostEvent]:
        """Copy of in-memory event buffer (most recent last)."""
        with self._lock:
            return list(self._events)

    @property
    def enabled(self) -> bool:
        """Whether cost tracking is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def snapshot(self) -> float:
        """Take a snapshot of the current session total.

        Use before a multi-step operation (e.g., RAG query) to later
        compute the delta: ``query_cost = METER.session_total - snapshot``.

        Returns:
            Current session total in USD
        """
        with self._lock:
            return self._session_total

    def add(self, event: CostEvent) -> None:
        """Record a cost event.

        Adds to the in-memory buffer, updates the session total,
        and persists to SQLite.

        Args:
            event: The CostEvent to record
        """
        if not self._enabled:
            return

        with self._lock:
            self._events.append(event)
            self._session_total += event.cost_usd

            # Evict oldest events if buffer is full
            if len(self._events) > MAX_MEMORY_EVENTS:
                self._events = self._events[-MAX_MEMORY_EVENTS:]

        # Persist to SQLite (outside lock to avoid holding it during I/O)
        try:
            import cost_db
            cost_db.insert_cost_event(
                ts=event.ts,
                provider=event.provider,
                model=event.model,
                kind=event.kind,
                in_tokens=event.in_tokens,
                out_tokens=event.out_tokens,
                total_tokens=event.total_tokens,
                cost_usd=event.cost_usd,
                conversation_id=event.conversation_id,
                request_context=event.request_context,
            )
        except Exception:
            # Non-fatal: if persistence fails, in-memory tracking continues
            import logging
            logging.getLogger(__name__).warning(
                "Failed to persist cost event to SQLite", exc_info=True
            )

    # -----------------------------------------------------------------
    # Convenience methods — compute cost and record in one call
    # -----------------------------------------------------------------

    def record_chat(
        self,
        provider: str,
        model: str,
        in_tokens: int,
        out_tokens: int,
        conversation_id: str = "",
        request_context: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> CostEvent:
        """Record a chat/LLM completion cost event.

        Args:
            provider: "openai" or "gemini"
            model: Model name (e.g., "gpt-4o")
            in_tokens: Prompt token count
            out_tokens: Completion token count
            conversation_id: Associated conversation UUID
            request_context: Context label
            meta: Optional metadata dict

        Returns:
            The created CostEvent
        """
        model_key = resolve_model_key(provider, model)
        cost = chat_cost(model_key, in_tokens, out_tokens)

        event = CostEvent(
            ts=time(),
            provider=provider,
            model=model,
            kind="chat",
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost_usd=cost,
            conversation_id=conversation_id,
            request_context=request_context,
            meta=meta or {},
        )
        self.add(event)
        return event

    def record_embed(
        self,
        provider: str,
        model: str,
        total_tokens: int,
        conversation_id: str = "",
        request_context: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> CostEvent:
        """Record an embedding cost event.

        Args:
            provider: "openai" or "gemini"
            model: Embedding model name
            total_tokens: Total tokens embedded
            conversation_id: Associated conversation UUID
            request_context: Context label
            meta: Optional metadata dict

        Returns:
            The created CostEvent
        """
        model_key = resolve_model_key(provider, model)
        cost = embed_cost(model_key, total_tokens)

        event = CostEvent(
            ts=time(),
            provider=provider,
            model=model,
            kind="embed",
            total_tokens=total_tokens,
            cost_usd=cost,
            conversation_id=conversation_id,
            request_context=request_context,
            meta=meta or {},
        )
        self.add(event)
        return event

    def record_whisper(
        self,
        duration_seconds: float,
        model: str = "whisper-1",
        conversation_id: str = "",
        request_context: str = "transcribe",
        meta: Optional[Dict[str, Any]] = None,
    ) -> CostEvent:
        """Record a Whisper transcription cost event.

        Args:
            duration_seconds: Audio duration in seconds
            model: Whisper model name
            conversation_id: Associated conversation UUID
            request_context: Context label
            meta: Optional metadata dict

        Returns:
            The created CostEvent
        """
        cost = whisper_cost(duration_seconds, model)

        event = CostEvent(
            ts=time(),
            provider="openai",
            model=model,
            kind="whisper",
            cost_usd=cost,
            conversation_id=conversation_id,
            request_context=request_context,
            meta=meta or {"duration_seconds": duration_seconds},
        )
        self.add(event)
        return event

    def record_image(
        self,
        model: str = "dall-e-3",
        count: int = 1,
        conversation_id: str = "",
        request_context: str = "image_generate",
        meta: Optional[Dict[str, Any]] = None,
    ) -> CostEvent:
        """Record an image generation cost event.

        Args:
            model: Image model name (e.g., "dall-e-3")
            count: Number of images generated
            conversation_id: Associated conversation UUID
            request_context: Context label
            meta: Optional metadata dict

        Returns:
            The created CostEvent
        """
        model_key = f"openai:{model}"
        cost = image_cost(model_key, count)

        event = CostEvent(
            ts=time(),
            provider="openai",
            model=model,
            kind="image",
            cost_usd=cost,
            conversation_id=conversation_id,
            request_context=request_context,
            meta=meta or {"count": count},
        )
        self.add(event)
        return event

    def get_recent_events(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the N most recent events as dicts (for API responses).

        Args:
            n: Number of recent events

        Returns:
            List of event dicts (most recent first)
        """
        with self._lock:
            recent = self._events[-n:] if n < len(self._events) else list(self._events)
        return [e.to_dict() for e in reversed(recent)]


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------

METER = CostMeter()
