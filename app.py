import base64
import requests

from typing import Dict, Union
from flask import Flask, request, jsonify, render_template_string
import httpx
import base64
import requests

from config import config
from utiles.logger import Logger
from memory_agent import MemoryAgent
from providers.dalle import Dalle


logger = Logger()
app = Flask(__name__)


_memory_agents = {}
_contacts = {}


class Group:
    def __init__(self, group_id):
        self.group_id = group_id
        self.data = self.get_group(group_id)
        self.name = self.data.get("name", "")
        self.participants = self.data.get("participants", [])
        self.is_group = True
        self.isMyGroup = True if self.data.get("isMyGroup", False) else False
        self.isMe = True if self.data.get("isMe", False) else False

    def get_group(self, id):
        endpoint = f"/api/session/groups/{id}"
        response = send_request(method="GET", endpoint=endpoint)
        logger.debug(f"Retrieved group info for {id}: {response.json()}")
        return response.json()

    def __str__(self):
        return f"Group ID: {self.group_id}, Name: {self.name}, Participants: {len(self.participants)}"


class Contact:
    def __init__(self, payload):
        self._from = payload.get("from", "")
        self.participant = payload.get("participant", "")
        data = self.get_contact()

        self.isMyContact = data.get("isMyContact", False)
        if self.isMyContact:
            self.name = data.get("name", "")
        else:
            self.name = data.get("pushname", "")

        self.number = data.get("number", "")
        self.isBusiness = data.get("isBusiness", False)
        self.is_group = data.get("isGroup", False)
        self.isUser = data.get("isUser", False)

        self.isMe = data.get("isMe", False)
        self.isBlocked = data.get("isBlocked", False)

    def get_contact(self):
        endpoint = f"/api/contacts"
        contact = self.participant if self.participant and self.participant != "out@c.us" else self._from
        params = {"contactId": contact, "session": config.waha_session_name}
        response = send_request(method="GET", endpoint=endpoint, params=params)
        return response.json()

    def __str__(self):
        return self.__dict__.__str__()


def get_contact(payload):
    try:
        sender = payload["participant"]
    except KeyError:
        sender = payload.get("from", None)

    if sender not in _contacts:
        _contacts[sender] = Contact(payload)
    logger.debug(f"Retrieved contact for sender {sender}: {_contacts[sender]}")
    return _contacts[sender]


def get_memory_agent(recipient: str) -> MemoryAgent:
    if recipient not in _memory_agents:
        _memory_agents[recipient] = MemoryAgent(recipient)
    return _memory_agents[recipient]


def send_request(method: str, endpoint: str, payload: Union[Dict, None] = None, params: Union[Dict, None] = None):
    payload = payload or {}
    params = params or {}

    headers = {"Content-Type": "application/json",
               "X-Api-Key": config.waha_api_key,
               "stream": "true"}
    url = f"{config.waha_base_url}{endpoint}"
    if method.upper() == "POST":
        response = requests.post(url, json=payload, headers=headers, params=params)
    elif method.upper() == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif method.upper() == "PUT":
        response = requests.put(url, json=payload, headers=headers, params=params)
    return response


class MediaMessage:
    def __init__(self, data):
        # logger.debug(f"Message has media: {payload}")
        self.url = data.get('url')
        self.type = data.get('mimetype')
        self.base64 = base64.standard_b64encode(httpx.get(
            self.url, headers={"X-Api-Key": config.waha_api_key}).content).decode("utf-8")


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
        # logger.debug(f"Initializing WhatsappMSG with payload: {payload}")
        self.message = payload.get("body", "").strip()
        self.contact = get_contact(payload=payload)
        self.recipient = payload.get("to", "")
        self._from = payload.get("from", "")
        self.is_group = True if "@g" in self._from else False
        if self.is_group:
            self.group = Group(self._from)
        self.has_media = payload.get("hasMedia", False)
        if self.has_media:
            self.media = MediaMessage(payload.get("media"))
        self.has_quote = payload.get("_data", {}).get("quotedMsg", {})
        if self.has_quote:
            try:
                self.quoted = QuotedMessage(quoted_data=payload.get(
                    "_data", {}), recipient=self.recipient)
            except Exception as e:
                logger.error(f"Error processing quoted message: {e}")
                self.quoted = None
        else:
            self.quoted = None

    def is_valid(self) -> bool:
        allowed_senders = []
        if self.contact.name not in ["Me", allowed_senders]:
            return False
        if not self.message.startswith(config.chat_prefix) and not self.message.startswith(config.dalle_prefix):
            return False
        return True

    def startswith(self, prefix: str) -> bool:
        return self.message.startswith(prefix)

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

    def __str__(self):
        return f"From: {self.contact.name}, To: {self.recipient}, Message: '{self.message}'"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"}), 200


@app.route("/test", methods=["GET"])
def test():
    # send a test message to yourself
    test_message = f"{config.gpt_prefix} Hello from WhatsApp-GPT!"
    send_request(method="POST",
                 endpoint="/api/sendText",
                 payload={
                     "chatId": config.waha_test_recipient,
                     "text": test_message,
                     "session": config.waha_session_name
                 }
                 )
    return jsonify({"status": "test message sent"}), 200

@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.json.get("payload", {}) if request.json else {}

    if payload.get('event') =="message_ack":
        pass
    if not payload.get("from") == config.waha_test_recipient:
        return jsonify({"status": "ignored"}), 200
    whatsapp_msg = WhatsappMSG(payload)
    mem_agent: MemoryAgent = get_memory_agent(whatsapp_msg._from)
    mem_agent.remember(text=whatsapp_msg.message,
                       role=whatsapp_msg.contact.name or "unknown")

    if not whatsapp_msg.is_valid():
        return jsonify({"status": "ignored"}), 200

    try:
        route = whatsapp_msg.route()
        if route == "chat":
            response = mem_agent.send_message(whatsapp_msg)
            whatsapp_msg.reply(str(response))
        elif route == "dalle":
            dalle = Dalle()
            dalle.context = mem_agent.get_recent_text_context()

            dalle.prompt = whatsapp_msg.message[len(
                config.dalle_prefix):].strip()
            image_url = dalle.request()

            send_request(method="POST",
                         endpoint="/api/sendImage",
                         payload={
                             "chatId": payload.get("to"),
                             "file": {"url": image_url},
                             "session": config.waha_session_name
                         })

        else:
            logger.debug(
                f"Message did not match any route: {whatsapp_msg.message}")
            return jsonify({"status": "no matching handler"}), 200
        return jsonify({"status": "ok"}), 200

    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        raise
        return jsonify({"error": str(e)}), 400


@app.route("/pair", methods=["GET"])
def pair():
    session_name = config.waha_session_name

    response = send_request("GET", f"/api/sessions/{session_name}")

    status_data = response.json()
    logger.debug(f"Status data: {status_data}")

    if status_data.get("status") == "WORKING" and status_data.get("engine", {}).get("state") == "CONNECTED":
        return "<h1>Session 'default' is already connected.</h1>", 200

    if status_data.get("status") != "SCAN_QR_CODE":
        send_request(method="POST", endpoint="/api/sessions/start",
                     payload={"name": session_name})

        # Step 3: Configure webhook
        send_request("PUT", f"/api/sessions/{session_name}", {
            "config": {
                "webhooks": [
                    {
                        "url": config.waha_webhook_url,
                        "events": ["message.any", "session.status"]
                    }
                ]
            }
        })

    # Step 4: Get QR code
    qr_response = send_request("GET", f"/api/{session_name}/auth/qr")
    if isinstance(qr_response, dict) and "error" in qr_response:
        return f"Failed to get QR code: {qr_response['error']}", 500

    qr_image_data = qr_response.content
    if qr_image_data:
        qr_base64 = base64.b64encode(qr_image_data).decode("utf-8")
        html = f"<h1>Scan to Pair WhatsApp</h1><img src='data:image/png;base64,{qr_base64}'>"
        return render_template_string(html)
    else:
        return "QR code not available yet. Please refresh in a few seconds.", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.port, debug=True if config.log_level=="DEBUG" else False)
