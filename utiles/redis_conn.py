import json
from typing import Optional, Any
from config import config
import redis

# Redis connection singleton
_redis_client = None


def get_redis_client() -> redis.Redis:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host='localhost',  # Redis running in Docker, exposed on localhost
            port=6379,        # Default Redis port from docker-compose
            decode_responses=True  # Automatically decode response bytes to str
        )
    return _redis_client


def redis_set(key: str, value: Any, expire: Optional[int] = None) -> None:
    """Store value in Redis, optionally with expiration in seconds."""
    client = get_redis_client()
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value)
    client.set(key, value, ex=int(config.redis_ttl) if expire is None else expire)


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
