#!/usr/bin/env python3
"""Send a WhatsApp message to a phone number via WAHA API.

Usage:
    python scripts/send_message.py PHONE_NUMBER "Your message here"
    
Example:
    python scripts/send_message.py 972501234567 "Hello from the bot!"
"""

import sys
import os

# Add src to path so we can import from there
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import config
from utils.globals import send_request


def send_message(phone_number: str, message: str):
    """Send a WhatsApp text message.
    
    Args:
        phone_number: Phone number without + or country code spaces (e.g., "972501234567")
        message: The message text to send
        
    Returns:
        API response dictionary
    """
    # Ensure proper chat ID format
    if not phone_number.endswith("@c.us"):
        chat_id = f"{phone_number}@c.us"
    else:
        chat_id = phone_number
    
    response = send_request(
        method="POST",
        endpoint="/api/sendText",
        payload={
            "chatId": chat_id,
            "text": message,
            "session": config.waha_session_name
        }
    )
    
    return response


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing arguments")
        print("Usage: python scripts/send_message.py PHONE_NUMBER \"Your message\"")
        sys.exit(1)
    
    phone_number = sys.argv[1]
    message = sys.argv[2]
    
    print(f"Sending message to {phone_number}...")
    print(f"Message: {message}")
    
    try:
        result = send_message(phone_number, message)
        print(f"✅ Message sent successfully!")
        print(f"Response: {result}")
    except Exception as e:
        print(f"❌ Failed to send message: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
