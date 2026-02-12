"""Time formatting and grouping helpers for conversation display.

Provides ``relative_time()`` for compact display and
``group_conversations_by_time()`` for ChatGPT-style time grouping.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


def relative_time(timestamp_str: Optional[str]) -> str:
    """Convert an ISO timestamp string to a relative time display.

    Examples: 'just now', '5m ago', '2h ago', '3d ago', '01/02/2024'
    """
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
    """Determine which time group a datetime belongs to.

    Returns one of: 'Today', 'Yesterday', 'Previous 7 Days',
    'Previous 30 Days', or a month-year string like 'January 2024'.
    """
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
    conversations: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group a list of conversations by time period (ChatGPT-style).

    Expects each conversation dict to have an ``updated_at`` key
    (ISO timestamp string). Returns an ordered dict of
    ``{group_label: [conversations]}``.

    Groups (in order): Today, Yesterday, Previous 7 Days,
    Previous 30 Days, then month-year buckets.
    """
    from collections import OrderedDict

    now = datetime.now(timezone.utc)
    groups: Dict[str, List[Dict[str, Any]]] = {}

    # Desired display order for the well-known groups
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
                dt = now  # Fallback: treat as current
        else:
            dt = now

        label = _time_group_label(dt, now)
        groups.setdefault(label, []).append(convo)

    # Build output in deterministic order
    result: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for label in ORDERED_LABELS:
        if label in groups:
            result[label] = groups.pop(label)

    # Remaining groups (month-year) sorted reverse-chronologically
    remaining = sorted(groups.keys(), reverse=True)
    for label in remaining:
        result[label] = groups[label]

    return result


def export_chat_history(messages: List[Dict[str, str]]) -> str:
    """Export chat history as a formatted text string."""
    lines = []
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]\n{msg['content']}\n")
    return "\n".join(lines)
