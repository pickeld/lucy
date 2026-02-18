"""Celery application configuration for Lucy background tasks.

Uses Redis as both broker and result backend (DB 1, separate from
the main application's DB 0).

Task queues:
    default  — lightweight tasks (WhatsApp message processing, entity extraction)
    heavy    — CPU/GPU-bound tasks (Whisper transcription, large document sync)

Usage (worker):
    celery -A tasks worker --loglevel=info -Q default,heavy

Usage (from application code):
    from tasks.whatsapp import process_whatsapp_message
    process_whatsapp_message.delay(payload)
"""

import os

from celery import Celery
from celery.signals import worker_process_init

# ---------------------------------------------------------------------------
# Redis broker URL
# ---------------------------------------------------------------------------
# Reads the same REDIS_HOST / REDIS_PORT that the main app uses, but targets
# DB 1 to avoid key collisions with the app's cache/state in DB 0.

_redis_host = os.environ.get("REDIS_HOST", "localhost")
_redis_port = os.environ.get("REDIS_PORT", "6379")
_redis_db = os.environ.get("CELERY_REDIS_DB", "1")
_broker_url = f"redis://{_redis_host}:{_redis_port}/{_redis_db}"

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

app = Celery("lucy")

app.conf.update(
    # Broker + result backend
    broker_url=_broker_url,
    result_backend=_broker_url,

    # Serialization — JSON only for transparency and debuggability
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Reliability: acknowledge AFTER task completes, not before.
    # If the worker crashes mid-task, the message returns to the queue.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Don't prefetch — important for long-running tasks (transcription)
    # so one slow task doesn't block the next in the same worker.
    worker_prefetch_multiplier=1,

    # Retry defaults
    task_default_retry_delay=30,       # 30 seconds base delay
    task_max_retries=3,                # 3 retries before giving up

    # Result expiry — keep results for 1 hour (for status checks)
    result_expires=3600,

    # Task routing — direct heavy tasks to the 'heavy' queue
    task_routes={
        "tasks.transcription.*": {"queue": "heavy"},
        "tasks.whatsapp.*": {"queue": "default"},
    },

    # Default queue for unrouted tasks
    task_default_queue="default",

    # Worker concurrency defaults (can be overridden via CLI)
    # The heavy queue worker should run with --concurrency=1 for Whisper
    worker_concurrency=4,

    # Task time limits (seconds)
    task_soft_time_limit=300,      # 5 min soft limit (raises SoftTimeLimitExceeded)
    task_time_limit=600,           # 10 min hard kill

    # Logging
    worker_hijack_root_logger=False,  # Don't override our logging config

    # Explicitly register task modules (autodiscover doesn't work well
    # when the Celery app itself lives inside the tasks package)
    include=[
        "tasks.whatsapp",
        "tasks.transcription",
    ],
)


# ---------------------------------------------------------------------------
# Worker process initialization — discover and load plugins so that tasks
# like ``transcribe_recording`` can access plugin syncers via the registry.
# ---------------------------------------------------------------------------

@worker_process_init.connect
def _init_plugins_in_worker(**kwargs):
    """Initialize the plugin registry when a Celery worker process starts.

    Creates a minimal Flask app (required by the plugin lifecycle contract)
    and runs the same discovery + loading sequence that ``app.py`` does.
    Blueprint registration happens harmlessly — the worker never serves HTTP.
    """
    from flask import Flask
    from plugins.registry import plugin_registry

    # Create a minimal Flask app to satisfy plugin initialize() signatures
    _flask_app = Flask("lucy-worker")

    plugin_registry.discover_plugins()
    plugin_registry.load_enabled_plugins(_flask_app)
