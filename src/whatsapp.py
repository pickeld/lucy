import base64
import json
import os

import httpx

from config import config
from contact import Contact, ContactManager
from groups import Group, GroupManager
from utiles.globals import send_request
from utiles.logger import logger

contact_manager = ContactManager()
group_manager = GroupManager()


# make dir
if not os.path.exists("tmp"):
    os.makedirs("tmp")

class MediaMessage:
    def __init__(self, payload):
        self.has_media = payload.get("hasMedia", False)
        media_type = payload.get("_data", {}).get("type")
        if self.has_media and media_type not in ["sticker", "audio", "video"]:
            self.media = payload.get("media", {})
            self.url = self.media.get('url')
            self.type = self.media.get('mimetype')
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


class QuotedMessage:
    def __init__(self, quoted_data, recipient):
        self.quoted_data = quoted_data
        self.quoted_msg = quoted_data.get("quotedMsg", {})
        self.type = self.quoted_msg.get("type", "")
        self.body = self.quoted_msg.get("body", "").strip()
        self.kind = self.quoted_msg.get("kind", "")
        self.quoted_stanza_id = quoted_data.get("quotedStanzaID", "")
        self.quoted_participant = quoted_data.get("quotedParticipant", "")
        self.mimetype = self.quoted_msg.get("mimetype", "")
        self.caption = self.quoted_msg.get("caption", "").strip()
        if self.type == "image":
            self.file_extension = self.mimetype.split("/")[-1]
            self.filename = f"true_{recipient}_{self.quoted_stanza_id}_{self.quoted_participant}.{self.file_extension}"
            endpoint = f"/api/files/default/{self.filename}"
            response = send_request(method="GET", endpoint=endpoint)
            self.base64_data = base64.b64encode(
                response.content).decode("ascii")


class WhatsappMSG:
    def __init__(self, payload):
        self.contact: Contact = contact_manager.get_contact(payload)
        self.group: Group = group_manager.get_group(payload)
        self.is_group = True if self.group.id else False
        self.timestamp = payload.get("timestamp")
        self.message = payload.get("body", None)
        self.media = MediaMessage(payload)
        self.to = payload.get("to", None)
        # self.quoted = QuotedMessage(quoted_data=payload.get("quotedMsg", {}), recipient=self.recipient)
        # self.recipient = payload.get("to")

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

    def route(self):
        if self.message.startswith(config.chat_prefix):
            return "chat"
        elif self.message.startswith(config.dalle_prefix):
            return "dalle"
        else:
            return "unknown"

    def reply(self, response: str):
        send_request(method="POST",
                     endpoint="/api/sendText",
                     payload={
                              "chatId": self.recipient,
                              "text": response,
                              "session": config.waha_session_name
                     }
                     )

