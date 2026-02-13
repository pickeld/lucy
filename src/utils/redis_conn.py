import json
from typing import Optional, Any
from config import settings
import redis

# Redis connection singleton
_redis_client = None


def get_redis_client() -> redis.Redis:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=int(settings.redis_port),
            decode_responses=True  # Automatically decode response bytes to str
        )
    return _redis_client


def redis_set(key: str, value: Any, expire: Optional[int] = None) -> None:
    """Store value in Redis, optionally with expiration in seconds."""
    client = get_redis_client()
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value)
    client.set(key, value, ex=int(settings.redis_ttl)
               if expire is None else expire)


def redis_get(key: str, default: Any = None) -> Any:
    """Get value from Redis, return default if not found."""
    client = get_redis_client()
    value = client.get(key)
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def redis_delete(key: str) -> bool:
    """Delete a key from Redis. Returns True if key existed and was deleted."""
    client = get_redis_client()
    return client.delete(key) > 0


def redis_delete_pattern(pattern: str) -> int:
    """Delete all keys matching a pattern. Returns count of deleted keys."""
    client = get_redis_client()
    keys = client.keys(pattern)
    if keys:
        return client.delete(*keys)
    return 0
