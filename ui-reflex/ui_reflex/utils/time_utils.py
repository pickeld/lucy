"""Time formatting and grouping helpers for conversation display.

Reused from the Streamlit UI with no changes to logic.
"""

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any


def relative_time(timestamp_str: str | None) -> str:
    """Convert an ISO timestamp string to a relative time display."""
    if not timestamp_str:
        return ""
    try:
        ts = timestamp_str.replace("T", " ").split(".")[0]
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        elif seconds < 604800:
            return f"{seconds // 86400}d ago"
        else:
            return dt.strftime("%d/%m/%Y")
    except Exception:
        return ""


def _time_group_label(dt: datetime, now: datetime) -> str:
    """Determine which time group a datetime belongs to."""
    today = now.date()
    d = dt.date()
    delta_days = (today - d).days

    if delta_days == 0:
        return "Today"
    elif delta_days == 1:
        return "Yesterday"
    elif delta_days <= 7:
        return "Previous 7 Days"
    elif delta_days <= 30:
        return "Previous 30 Days"
    else:
        return dt.strftime("%B %Y")


def group_conversations_by_time(
    conversations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group conversations by time period — returns a flat list for rx.foreach.

    Returns a list of dicts: [{"label": "Today", "conversations": [...]}, ...]
    """
    now = datetime.now(timezone.utc)
    groups: dict[str, list[dict[str, Any]]] = {}

    ORDERED_LABELS = [
        "Today",
        "Yesterday",
        "Previous 7 Days",
        "Previous 30 Days",
    ]

    for convo in conversations:
        ts_str = convo.get("updated_at") or convo.get("created_at") or ""
        if ts_str:
            try:
                ts = ts_str.replace("T", " ").split(".")[0]
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                dt = now
        else:
            dt = now

        label = _time_group_label(dt, now)
        groups.setdefault(label, []).append(convo)

    result: list[dict[str, Any]] = []
    for label in ORDERED_LABELS:
        if label in groups:
            result.append({"label": label, "conversations": groups.pop(label)})

    remaining = sorted(groups.keys(), reverse=True)
    for label in remaining:
        result.append({"label": label, "conversations": groups[label]})

    return result


def export_chat_history(messages: list[dict[str, str]]) -> str:
    """Export chat history as a formatted text string."""
    lines = []
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    return "\n".join(lines)
