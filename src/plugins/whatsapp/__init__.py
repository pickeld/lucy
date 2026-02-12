"""WhatsApp channel plugin.

This package provides the WhatsApp integration as a channel plugin,
using WAHA (WhatsApp HTTP API) as the backend.

Main exports:
    - WhatsAppPlugin: The ChannelPlugin implementation
    - create_whatsapp_message: Factory function for message objects
"""

from plugins.whatsapp.plugin import WhatsAppPlugin

__all__ = ["WhatsAppPlugin"]
