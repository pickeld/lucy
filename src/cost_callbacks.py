"""LlamaIndex callback handler for automatic LLM and embedding cost tracking.

Hooks into LlamaIndex's callback system to intercept all LLM and embedding
calls, extract token usage, compute costs, and emit events to the CostMeter.

Usage:
    from cost_callbacks import create_cost_callback_manager

    Settings.callback_manager = create_cost_callback_manager(
        llm_provider="openai",
        llm_model="gpt-4o",
        embed_model="text-embedding-3-large",
    )
"""

import logging
from time import time
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

logger = logging.getLogger(__name__)


class CostTrackingHandler(BaseCallbackHandler):
    """LlamaIndex callback handler that tracks LLM and embedding costs.

    Listens for LLM and EMBEDDING event completions, extracts token usage
    from the event payload, and emits CostEvents to the global CostMeter.

    Token extraction strategy:
    - LLM calls: Extract from response.raw.usage (OpenAI) or estimate via tiktoken
    - Embedding calls: Estimate tokens from input text length via tiktoken

    Attributes:
        llm_provider: Current LLM provider ("openai" or "gemini")
        llm_model: Current LLM model name
        embed_provider: Embedding provider (always "openai" for now)
        embed_model: Embedding model name
        conversation_id: Current conversation ID for event tagging
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        embed_provider: str = "openai",
        embed_model: str = "text-embedding-3-large",
    ):
        # BaseCallbackHandler requires these four lists
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        # Set by the caller before each query to tag events
        self.conversation_id: str = ""
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy-load tiktoken tokenizer for fallback token counting."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                # Fallback: rough estimate (4 chars per token)
                self._tokenizer = "fallback"
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback estimate.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        tokenizer = self._get_tokenizer()
        if tokenizer == "fallback":
            return max(1, len(text) // 4)
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts. We don't need to track starts."""
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event completes. Extract usage and emit cost event."""
        if payload is None:
            return

        try:
            if event_type == CBEventType.LLM:
                self._handle_llm_event(payload)
            elif event_type == CBEventType.EMBEDDING:
                self._handle_embedding_event(payload)
        except Exception as e:
            logger.debug(f"Cost tracking callback error (non-fatal): {e}")

    def _handle_llm_event(self, payload: Dict[str, Any]) -> None:
        """Extract token usage from an LLM completion event and emit cost.

        Tries multiple strategies to get token counts:
        1. response.raw.usage (OpenAI SDK response)
        2. response.additional_kwargs.usage (some LlamaIndex wrappers)
        3. Fallback: count tokens in prompt + completion text via tiktoken
        """
        from cost_meter import METER

        in_tokens = 0
        out_tokens = 0
        model = self.llm_model
        provider = self.llm_provider

        # Strategy 1: Try to get usage from the LLM response object
        response = payload.get(EventPayload.RESPONSE)
        if response is not None:
            # LlamaIndex wraps responses — try to get the raw response
            raw = getattr(response, "raw", None)
            if raw is not None:
                usage = getattr(raw, "usage", None)
                if usage is not None:
                    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    out_tokens = getattr(usage, "completion_tokens", 0) or 0
                # Get model from raw response if available
                raw_model = getattr(raw, "model", None)
                if raw_model:
                    model = raw_model

            # Strategy 2: Check additional_kwargs
            if in_tokens == 0 and out_tokens == 0:
                additional = getattr(response, "additional_kwargs", {})
                if isinstance(additional, dict):
                    usage = additional.get("usage", {})
                    if isinstance(usage, dict):
                        in_tokens = usage.get("prompt_tokens", 0) or 0
                        out_tokens = usage.get("completion_tokens", 0) or 0

            # Strategy 3: Check raw dict-like responses (Gemini)
            if in_tokens == 0 and out_tokens == 0:
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta is not None:
                    in_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                    out_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        # Strategy 4: Fallback — count tokens from prompt/completion text
        if in_tokens == 0 and out_tokens == 0:
            # Try to get the prompt messages
            messages = payload.get(EventPayload.MESSAGES, [])
            if messages:
                prompt_text = " ".join(
                    getattr(m, "content", str(m)) for m in messages
                )
                in_tokens = self._count_tokens(prompt_text)

            # Try to get completion text
            if response is not None:
                completion_text = getattr(response, "text", "") or str(response)
                out_tokens = self._count_tokens(completion_text)

        # Only record if we have some token counts
        if in_tokens > 0 or out_tokens > 0:
            METER.record_chat(
                provider=provider,
                model=model,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                conversation_id=self.conversation_id,
                request_context="llm_call",
            )
            logger.debug(
                f"Cost tracked: LLM {provider}:{model} "
                f"in={in_tokens} out={out_tokens}"
            )

    def _handle_embedding_event(self, payload: Dict[str, Any]) -> None:
        """Extract token usage from an embedding event and emit cost.

        Embedding events contain the chunks being embedded. We count
        tokens in all chunks to estimate the total embedding tokens.
        """
        from cost_meter import METER

        total_tokens = 0

        # Get the chunks/texts being embedded
        chunks = payload.get(EventPayload.CHUNKS, [])
        if chunks:
            for chunk in chunks:
                if isinstance(chunk, str):
                    total_tokens += self._count_tokens(chunk)
                else:
                    # Could be a TextNode or similar
                    text = getattr(chunk, "text", str(chunk))
                    total_tokens += self._count_tokens(text)

        if total_tokens > 0:
            METER.record_embed(
                provider=self.embed_provider,
                model=self.embed_model,
                total_tokens=total_tokens,
                conversation_id=self.conversation_id,
                request_context="embedding",
            )
            logger.debug(
                f"Cost tracked: Embed {self.embed_provider}:{self.embed_model} "
                f"tokens={total_tokens}"
            )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Required by BaseCallbackHandler — no-op."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Required by BaseCallbackHandler — no-op."""
        pass


# Singleton handler instance (so we can update conversation_id from app.py)
_cost_handler: Optional[CostTrackingHandler] = None


def get_cost_handler() -> Optional[CostTrackingHandler]:
    """Get the singleton cost tracking handler.

    Returns:
        The CostTrackingHandler instance, or None if not yet created
    """
    return _cost_handler


def create_cost_callback_manager(
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    embed_provider: str = "openai",
    embed_model: str = "text-embedding-3-large",
) -> CallbackManager:
    """Create a CallbackManager with cost tracking enabled.

    Args:
        llm_provider: LLM provider name
        llm_model: LLM model name
        embed_provider: Embedding provider name
        embed_model: Embedding model name

    Returns:
        CallbackManager with CostTrackingHandler registered
    """
    global _cost_handler
    _cost_handler = CostTrackingHandler(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
    )
    logger.info(
        f"Cost tracking enabled: LLM={llm_provider}:{llm_model}, "
        f"Embed={embed_provider}:{embed_model}"
    )
    return CallbackManager([_cost_handler])
