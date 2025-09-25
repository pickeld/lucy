import base64
import time
from typing import Dict, Union

import httpx
import requests
import json
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from config import config
from contact import Contact
from memory_agent import MemoryManager
from providers.dalle import Dalle
from utiles.globals import send_request
from utiles.logger import logger
import traceback

from whatsapp import WhatsappMSG

app = Flask(__name__)


memory_manager = MemoryManager()


def pass_filter(payload):
    if payload.get('event') == "message_ack" or \
            payload.get("from").endswith("@newsletter") or \
            payload.get("from").endswith("@broadcast") or \
            payload.get("_data", {}).get("type") in ["e2e_notification", "notification_template"]:
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
    # logger.info(f"Received webhook: {request.json}")
    payload = request.json.get("payload", {})
    try:
        if pass_filter(payload) is False:
            return jsonify({"status": "ok"}), 200

        msg = WhatsappMSG(payload)
        # logger.debug(f"Received: {msg.__dict__}")
        agent = memory_manager.get_agent(msg)
        if msg.message:
            agent.remember(timestamp=msg.timestamp,
                           sender=msg.contact.name or msg.contact.number, msg=msg.message)
        logger.debug(f"Processed message: {msg}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(
            f"Error processing webhook: {e} ::: {payload}\n{trace} :: {payload}")
        return jsonify({"error": str(e), "traceback": trace}), 500


@app.route("/", methods=["GET"])
def index():
    try:
        response: dict = send_request(
            "GET", f"/api/sessions/{config.waha_session_name}")
        logger.debug(f"Session status: {response}")

        if response.get("status") == "WORKING" and response.get("engine", {}).get("state") == "CONNECTED":
            return "<h1>Session 'default' is already connected.</h1>", 200
        elif response.get("status") == "SCAN_QR_CODE":
            return redirect("/qr_code")
        else:
            return redirect("/pair")
    # catch 404 error
    except Exception as e:
        logger.error(f"Error checking session status: {e}")
        return redirect("/pair")


@app.route("/qr_code", methods=["GET"])
def qr_code():
    qr_response = send_request(
        "GET", f"/api/{config.waha_session_name}/auth/qr")
    qr_image_data = qr_response.content
    if qr_image_data:
        qr_base64 = base64.b64encode(qr_image_data).decode("utf-8")
        html = f"<h1>Scan to Pair WhatsApp</h1><img src='data:image/png;base64,{qr_base64}'>"
        return render_template_string(html)
    else:
        return "QR code not available yet. Please refresh in a few seconds.", 200


@app.route("/pair", methods=["GET"])
def pair():
    session_name = config.waha_session_name
    send_request(method="POST", endpoint="/api/sessions/start",
                 payload={"name": session_name})

    send_request("PUT", f"/api/sessions/{session_name}", {
        "config": {
            "webhooks": [
                {
                    "url": config.webhook_url,
                    "events": ["message.any", "session.status"]
                }
            ]
        }
    })
    time.sleep(2)
    return redirect("/qr_code")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765,
            debug=True if config.log_level == "DEBUG" else False)
