"""LLM pricing registry — single source of truth for cost calculations.

Maps provider:model keys to per-1K-token prices (USD). Update this file
when providers change their pricing — no other code needs to change.

Pricing last verified: 2025-01-15

Usage:
    from pricing import chat_cost, embed_cost, resolve_model_key

    key = resolve_model_key("openai", "gpt-4o")
    cost = chat_cost(key, in_tokens=500, out_tokens=200)
"""

from typing import Optional

# Last verified date — bump when you re-check provider pricing pages
PRICING_VERIFIED_DATE = "2025-01-15"

# ---------------------------------------------------------------------------
# Price table: provider:model -> cost per 1K tokens (USD)
#
#   "in"          = input/prompt tokens  ($/1K)
#   "out"         = output/completion tokens ($/1K)
#   "embed"       = embedding tokens ($/1K)
#   "per_minute"  = audio transcription ($/minute)
#   "per_image"   = image generation ($/image, standard quality 1024x1024)
# ---------------------------------------------------------------------------

PRICING = {
    # ── OpenAI Chat Models ──────────────────────────────────────────────
    "openai:gpt-4o":              {"in": 0.0025,   "out": 0.010},
    "openai:gpt-4o-mini":         {"in": 0.00015,  "out": 0.0006},
    "openai:gpt-4-turbo":         {"in": 0.01,     "out": 0.03},
    "openai:gpt-4":               {"in": 0.03,     "out": 0.06},
    "openai:gpt-3.5-turbo":       {"in": 0.0005,   "out": 0.0015},
    "openai:o1":                  {"in": 0.015,    "out": 0.06},
    "openai:o1-mini":             {"in": 0.003,    "out": 0.012},
    "openai:o3-mini":             {"in": 0.0011,   "out": 0.0044},

    # ── OpenAI Embeddings ───────────────────────────────────────────────
    "openai:text-embedding-3-small": {"embed": 0.00002},
    "openai:text-embedding-3-large": {"embed": 0.00013},
    "openai:text-embedding-ada-002":  {"embed": 0.0001},

    # ── OpenAI Audio ────────────────────────────────────────────────────
    "openai:whisper-1":           {"per_minute": 0.006},

    # ── OpenAI Image Generation ─────────────────────────────────────────
    "openai:dall-e-3":            {"per_image": 0.040},
    "openai:dall-e-2":            {"per_image": 0.020},

    # ── Google Gemini Chat Models ───────────────────────────────────────
    "gemini:gemini-pro":          {"in": 0.00125,  "out": 0.005},
    "gemini:gemini-1.5-flash":    {"in": 0.000075, "out": 0.0003},
    "gemini:gemini-1.5-pro":      {"in": 0.00125,  "out": 0.005},
    "gemini:gemini-2.0-flash":    {"in": 0.0001,   "out": 0.0004},
    "gemini:gemini-2.0-flash-lite": {"in": 0.000075, "out": 0.0003},

    # ── Google Gemini Embeddings ────────────────────────────────────────
    "gemini:text-embedding-004":  {"embed": 0.00002},
}

# ---------------------------------------------------------------------------
# Model name aliases — maps common variants to canonical pricing keys
# ---------------------------------------------------------------------------

_ALIASES = {
    # OpenAI aliases
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    # Gemini aliases
    "models/gemini-pro": "gemini-pro",
    "models/gemini-1.5-flash": "gemini-1.5-flash",
    "models/gemini-1.5-pro": "gemini-1.5-pro",
    "models/gemini-2.0-flash": "gemini-2.0-flash",
}


def resolve_model_key(provider: str, model_name: str) -> str:
    """Resolve a provider + model name to a pricing key.

    Handles model name aliases (e.g., date-suffixed OpenAI model names)
    and falls back gracefully if the model isn't in the registry.

    Args:
        provider: Provider name ("openai" or "gemini")
        model_name: Model name as configured in settings

    Returns:
        Pricing key like "openai:gpt-4o"
    """
    provider = provider.lower().strip()
    model = model_name.strip()

    # Try direct lookup first
    key = f"{provider}:{model}"
    if key in PRICING:
        return key

    # Try alias resolution
    canonical = _ALIASES.get(model, model)
    key = f"{provider}:{canonical}"
    if key in PRICING:
        return key

    # Strip "models/" prefix (Gemini SDK sometimes includes it)
    if model.startswith("models/"):
        stripped = model[len("models/"):]
        key = f"{provider}:{stripped}"
        if key in PRICING:
            return key

    # Return the best-effort key (caller should handle KeyError)
    return f"{provider}:{model}"


def chat_cost(model_key: str, in_tokens: int, out_tokens: int) -> float:
    """Calculate cost for a chat/LLM completion call.

    Args:
        model_key: Pricing key like "openai:gpt-4o"
        in_tokens: Number of input/prompt tokens
        out_tokens: Number of output/completion tokens

    Returns:
        Cost in USD (0.0 if model not found in registry)
    """
    p = PRICING.get(model_key)
    if not p or "in" not in p:
        return 0.0
    return (in_tokens / 1000) * p["in"] + (out_tokens / 1000) * p["out"]


def embed_cost(model_key: str, tokens: int) -> float:
    """Calculate cost for an embedding call.

    Args:
        model_key: Pricing key like "openai:text-embedding-3-large"
        tokens: Total tokens embedded

    Returns:
        Cost in USD (0.0 if model not found in registry)
    """
    p = PRICING.get(model_key)
    if not p or "embed" not in p:
        return 0.0
    return (tokens / 1000) * p["embed"]


def whisper_cost(duration_seconds: float, model: str = "whisper-1") -> float:
    """Calculate cost for a Whisper transcription call.

    Args:
        duration_seconds: Audio duration in seconds
        model: Whisper model name

    Returns:
        Cost in USD
    """
    key = f"openai:{model}"
    p = PRICING.get(key)
    if not p or "per_minute" not in p:
        return 0.0
    minutes = duration_seconds / 60.0
    return minutes * p["per_minute"]


def image_cost(model_key: str, count: int = 1) -> float:
    """Calculate cost for image generation.

    Args:
        model_key: Pricing key like "openai:dall-e-3"
        count: Number of images generated

    Returns:
        Cost in USD
    """
    p = PRICING.get(model_key)
    if not p or "per_image" not in p:
        return 0.0
    return count * p["per_image"]


def get_model_price(model_key: str) -> Optional[dict]:
    """Get the raw price entry for a model key.

    Args:
        model_key: Pricing key like "openai:gpt-4o"

    Returns:
        Price dict or None if not found
    """
    return PRICING.get(model_key)


def is_known_model(model_key: str) -> bool:
    """Check if a model key exists in the pricing registry.

    Args:
        model_key: Pricing key like "openai:gpt-4o"

    Returns:
        True if the model has pricing data
    """
    return model_key in PRICING
