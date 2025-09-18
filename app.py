import base64
from typing import Dict, Union

import httpx
import requests
import json
from flask import Flask, jsonify, render_template_string, request

from config import config
from contact import Contact
from memory_agent import MemoryManager
from providers.dalle import Dalle
from utiles.globals import send_request
from utiles.logger import logger
from whatsapp import WhatsappMSG

app = Flask(__name__)


memory_manager = MemoryManager()


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
        # logger.debug(f"Received webhook payload: {payload}")
        msg = WhatsappMSG(payload)
        agent = memory_manager.get_agent(msg)
        if msg.message:
            agent.remember(timestamp=msg.timestamp,
                           sender=msg.contact.name or msg.contact.number, msg=msg.message)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 200


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
