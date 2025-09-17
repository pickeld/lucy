from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from utiles.globals import send_request
from utiles.logger import logger


class GroupManager:
    def __init__(self) -> None:
        self.groups = {}

    def get_group(self, payload) -> "Group":
        group_id = payload.get("from")
        if not group_id:
            raise ValueError("Payload missing 'from' field")
        if group_id not in self.groups:
            self.groups[group_id] = self.fetch_group(group_id)
        return self.groups[group_id]

    def fetch_group(self, group_id: str) -> "Group":
        try:
            resp = send_request(
                method="GET", endpoint=f"/api/{config.waha_session_name}/groups/{group_id}")
            if isinstance(resp, dict):
                logger.debug(f"Fetched group data: {resp}")
                return Group().extract(resp)
            logger.error(
                f"Unexpected WAHA response for {group_id}: {type(resp)}")
        except Exception as e:
            logger.error(f"WAHA group fetch failed for {group_id}: {e}")
            return Group(id=group_id, name=None)


@dataclass
class Group:
    id: Optional[str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        return f"Name: {self.name}"

    def extract(self, data: Dict[str, Any]) -> "Group":
        # populate fields in-place and return self
        self.id = data.get("id")
        self.name = data.get("name")
        return self
