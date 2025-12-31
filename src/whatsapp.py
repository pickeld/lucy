import base64
import os

import httpx

from config import config
from contact import Contact, ContactManager
from groups import Group, GroupManager
from utils.logger import logger

contact_manager = ContactManager()
group_manager = GroupManager()


# make dir for media storage
if not os.path.exists("tmp/images"):
    os.makedirs("tmp/images")

class MediaMessage:
    def __init__(self, payload):
        self.has_media = payload.get("hasMedia", False)
        self.base64 = None
        self.url = None
        self.type = None
        self.saved_path = None
        
        media_type = payload.get("_data", {}).get("type")
        # Exclude unsupported media types: sticker, audio, video, ptv (video notes)
        if self.has_media and media_type not in ["sticker", "audio", "video", "ptv"]:
            self.media = payload.get("media", {})
            self.url = self.media.get('url')
            self.type = self.media.get('mimetype')
            
            # Only fetch media if URL is present
            if not self.url:
                logger.warning(f"Media message has no URL, skipping media download. Media type: {media_type}")
                self.has_media = False
                return
                
            self.base64 = base64.standard_b64encode(httpx.get(
                self.url, headers={"X-Api-Key": config.waha_api_key}).content).decode("utf-8")
            if config.log_level == "DEBUG":
                # save media to file
                extension = self.type.split("/")[-1]
                filename = f"tmp/images/media_{payload.get('id')}.{extension}"
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(self.base64))
                logger.debug(f"Saved media to {filename}")
                self.saved_path = filename

    def __str__(self):
        if self.has_media:
            return self.saved_path
        return "No media"

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class WhatsappMSG:
    def __init__(self, payload):
        self.contact: Contact = contact_manager.get_contact(payload)
        self.group: Group = group_manager.get_group(payload)
        self.is_group = True if self.group.id else False
        self.timestamp = payload.get("timestamp")
        self.message = payload.get("body", None)
        self.media = MediaMessage(payload)
        self.to = payload.get("to", None)

    def __str__(self) -> str:
        return f"{self.group.name}/{self.contact.name}: {self.message} || Media: {True if self.media.has_media else False}"

    def to_dict(self):
        def serialize(value):
            if hasattr(value, "to_dict"):
                return value.to_dict()
            elif isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple, set)):
                return [serialize(v) for v in value]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                return str(value)

        return {
            k: serialize(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

