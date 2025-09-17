import base64
from typing import Dict, Union

import httpx
import requests
import json
from flask import Flask, jsonify, render_template_string, request

from config import config
from contact import Contact
from memory_agent import MemoryAgent
from providers.dalle import Dalle
from utiles.globals import send_request
from utiles.logger import logger
from whatsapp import WhatsappMSG

app = Flask(__name__)


# _memory_agents = {}


# def get_memory_agent(recipient: str) -> MemoryAgent:
#     if recipient not in _memory_agents:
#         _memory_agents[recipient] = MemoryAgent(recipient)
#     return _memory_agents[recipient]

def pass_filter(payload):
    if payload.get('event') == "message_ack" or \
        payload.get("from").endswith("@newsletter") or \
        payload.get("from").endswith("@broadcast") or \
        payload.get("_data", {}).get("type") == "e2e_notification":
            return False
        
    return True

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"}), 200


@app.route("/test", methods=["GET"])
def test():
    # send a test message to yourself
    test_message = f"my name is david and i 39 years old, i have two kids, mia which is 4 and ben which is 6 and i work as a software developer."
    send_request(method="POST",
                 endpoint="/api/sendText",
                 payload={
                     "chatId": "972547755011@c.us",
                     "text": test_message,
                     "session": config.waha_session_name
                 }
                 )
    return jsonify({"status": "test message sent"}), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.json.get("payload", {})
    try:
        if pass_filter(payload) is False:
            return jsonify({"status": "ok"}), 200
        msg = WhatsappMSG(payload)
        logger.info(msg)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error processing webhook: {e} || Payload: {payload}")
        return jsonify({"error": str(e)}), 200
    # # from_ = payload.get('from')
    # # contact: Contact = get_or_create_contact(contact_id=from_)
    # print(msg)
    # return jsonify({"status": "ok"}), 200

    # if not contact.is_me:
    #     logger.debug(f"Ignoring message from {from_}")
    #     return jsonify({"status": "ignored"}), 200

    # whatsapp_msg = WhatsappMSG(payload)
    # print(whatsapp_msg)
    # mem_agent: MemoryAgent = get_memory_agent(whatsapp_msg._from)

    # mem_agent.remember(text=whatsapp_msg.message,
    #                    role=whatsapp_msg.contact.name or "unknown")

    # if not whatsapp_msg.is_valid():
    #     return jsonify({"status": "ignored"}), 200

    # try:
    #     route = whatsapp_msg.route()
    #     if route == "chat":
    #         response = mem_agent.send_message(whatsapp_msg)
    #         whatsapp_msg.reply(str(response))
    #     elif route == "dalle":
    #         dalle = Dalle()
    #         dalle.context = mem_agent.get_recent_text_context()

    #         dalle.prompt = whatsapp_msg.message[len(
    #             config.dalle_prefix):].strip()
    #         image_url = dalle.request()

    #         send_request(method="POST",
    #                      endpoint="/api/sendImage",
    #                      payload={
    #                          "chatId": payload.get("to"),
    #                          "file": {"url": image_url},
    #                          "session": config.waha_session_name
    #                      })

    #     else:
    #         logger.debug(
    #             f"Message did not match any route: {whatsapp_msg.message}")
    #         return jsonify({"status": "no matching handler"}), 200
    #     return jsonify({"status": "ok"}), 200

    # except Exception as e:
    #     logger.error(f"Failed to process message: {e}")
    #     raise
    #     return jsonify({"error": str(e)}), 400


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
    app.run(host="0.0.0.0", port=5002,
            debug=True if config.log_level == "DEBUG" else False)
