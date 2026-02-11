"""WhatsApp group management with Redis caching.

This module provides classes for managing WhatsApp groups, including
fetching group information from WAHA and caching in Redis.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from utils.globals import send_request
from utils.logger import logger
from utils.redis_conn import redis_get, redis_set, redis_delete, redis_delete_pattern


@dataclass
class Group:
    """Represents a WhatsApp group chat.
    
    Attributes:
        id: The group ID (e.g., '120363123456789@g.us')
        name: The display name of the group
    """
    id: Optional[str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        return f"Name: {self.name}"

    def extract(self, data: Dict[str, Any]) -> "Group":
        """Extract group information from API response data.
        
        Args:
            data: Dictionary with group data from WAHA API
            
        Returns:
            Self with extracted data
        """
        self.id = data.get("id", {}).get("_serialized")
        self.name = data.get("name")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary representation.
        
        Returns:
            Dictionary with all group attributes
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class GroupManager:
    """Manager for WhatsApp group operations with Redis caching."""
    
    def __init__(self) -> None:
        self.groups: Dict[str, Group] = {}

    def get_group(self, payload: Dict[str, Any]) -> Group:
        """Get group information from payload, using cache when available.
        
        Args:
            payload: The webhook payload containing group information
            
        Returns:
            Group object with extracted information, or empty Group if not a group message
        """
        _from = payload.get("from")
        if not _from.endswith("@g.us"):
            return Group(id=None, name=None)

        group_data = redis_get(f"group:{_from}")
        if not group_data:
            group_data = self.fetch_group(_from)
            redis_set(f"group:{_from}", group_data)

        group = Group().extract(group_data)
        return group

    def fetch_group(self, group_id: str) -> Dict[str, Any]:
        """Fetch group information from WAHA API.
        
        Args:
            group_id: The WhatsApp group ID
            
        Returns:
            Dictionary with group information
        """
        logger.debug(f"Fetching group info for {group_id}")
        try:
            resp = send_request(
                method="GET", endpoint=f"/api/{config.waha_session_name}/groups/{group_id}")
            return resp
        except Exception as e:
            logger.error(f"WAHA group fetch failed for {group_id}: {e}")
            return {}

    def refresh_group(self, group_id: str) -> Optional[Group]:
        """Clear cache and re-fetch group data from WAHA.
        
        Args:
            group_id: The group ID (e.g., '120363123456789@g.us')
            
        Returns:
            Updated Group object or None if fetch failed
        """
        cache_key = f"group:{group_id}"
        redis_delete(cache_key)
        logger.info(f"Cleared cache for group: {group_id}")
        
        group_data = self.fetch_group(group_id)
        if group_data:
            redis_set(cache_key, group_data)
            return Group().extract(group_data)
        return None

    def clear_all_groups_cache(self) -> int:
        """Clear all cached group data from Redis.
        
        Returns:
            Number of cache entries deleted
        """
        count = redis_delete_pattern("group:*")
        logger.info(f"Cleared {count} group cache entries")
        return count
