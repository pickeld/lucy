from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from utiles.globals import send_request
from utiles.logger import logger
from utiles.redis_conn import redis_get, redis_set


class GroupManager:
    def __init__(self) -> None:
        self.groups = {}

    def get_group(self, payload):
        _from = payload.get("from")
        if not _from.endswith("@g.us"):
            return Group(id=None, name=None)

        group_data = redis_get(f"group:{_from}")
        if not group_data:
            group_data = self.fetch_group(_from)
            redis_set(f"group:{_from}", group_data)

        group = Group().extract(group_data)
        return group

    def fetch_group(self, group_id: str):
        logger.debug(f"Fetching group info for {group_id}")
        try:
            resp = send_request(
                method="GET", endpoint=f"/api/{config.waha_session_name}/groups/{group_id}")
            return resp
        except Exception as e:
            logger.error(f"WAHA group fetch failed for {group_id}: {e}")
            return {}


@dataclass
class Group:
    id: Optional[str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        return f"Name: {self.name}"

    def extract(self, data: Dict[str, Any]) -> "Group":
        self.id = data.get("id", {}).get("_serialized")
        self.name = data.get("name")
        return self

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
