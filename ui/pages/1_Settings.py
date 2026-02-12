"""Settings page for WhatsApp-GPT.

Provides a UI to view system health, edit all configuration settings
(stored in SQLite), and save changes via the backend API.
"""

import streamlit as st
import requests

# Configuration
API_BASE_URL = "http://localhost:8765"

st.set_page_config(
    page_title="Settings — WhatsApp RAG Assistant",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Settings")
st.caption("Manage all application configuration — stored in SQLite, editable live.")

api_url = st.session_state.get("api_url", API_BASE_URL)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_health(url: str) -> dict:
    """Fetch system health status."""
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        if resp.status_code == 200 or resp.status_code == 503:
            return resp.json()
    except Exception as e:
        return {"status": "error", "dependencies": {"error": str(e)}}
    return {"status": "unknown", "dependencies": {}}


def fetch_config(url: str) -> dict:
    """Fetch all settings grouped by category."""
    try:
        resp = requests.get(f"{url}/config", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def save_config(url: str, updates: dict) -> dict:
    """Save settings via PUT /config."""
    try:
        resp = requests.put(
            f"{url}/config",
            json={"settings": updates},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def reset_category(url: str, category: str = "") -> dict:
    """Reset settings to defaults."""
    try:
        payload = {"category": category} if category else {}
        resp = requests.post(f"{url}/config/reset", json=payload, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# Category metadata for display
CATEGORY_ICONS = {
    "secrets": "🔑",
    "llm": "🤖",
    "rag": "🔍",
    "whatsapp": "💬",
    "infrastructure": "🏗️",
    "app": "🔧",
    "tracing": "📊",
}

CATEGORY_LABELS = {
    "secrets": "API Keys & Secrets",
    "llm": "LLM Configuration",
    "rag": "RAG Configuration",
    "whatsapp": "WhatsApp Configuration",
    "infrastructure": "Infrastructure",
    "app": "App Configuration",
    "tracing": "Tracing — LangSmith",
}

# Select-type options
SELECT_OPTIONS = {
    "llm_provider": ["openai", "gemini"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}

# Category display order
CATEGORY_ORDER = ["secrets", "llm", "rag", "whatsapp", "infrastructure", "app", "tracing"]


# =============================================================================
# SYSTEM HEALTH SECTION
# =============================================================================

st.markdown("---")
st.subheader("🟢 System Health")

health = fetch_health(api_url)
deps = health.get("dependencies", {})

health_cols = st.columns(len(deps) if deps else 3)
for i, (name, status) in enumerate(deps.items()):
    with health_cols[i]:
        is_ok = "connected" in str(status).lower()
        icon = "🟢" if is_ok else "🔴"
        st.metric(
            label=name.upper(),
            value=f"{icon} {'OK' if is_ok else 'Error'}",
        )
        if not is_ok:
            st.caption(str(status))

overall = health.get("status", "unknown")
if overall == "up":
    st.success("All systems operational")
elif overall == "degraded":
    st.warning("Some services are degraded")
else:
    st.error(f"System status: {overall}")


# =============================================================================
# SETTINGS FORM
# =============================================================================

st.markdown("---")

all_config = fetch_config(api_url)

if not all_config:
    st.error(
        "Could not load settings from the API. "
        "Make sure the backend is running at the configured API URL."
    )
    st.stop()

# Collect all form values
form_values = {}

# Sort categories by defined order
sorted_categories = sorted(
    all_config.keys(),
    key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99,
)

for category in sorted_categories:
    settings_dict = all_config[category]
    icon = CATEGORY_ICONS.get(category, "📋")
    label = CATEGORY_LABELS.get(category, category.title())

    with st.expander(f"{icon} {label}", expanded=(category != "infrastructure")):
        # Sort keys within category for consistent display
        for key in sorted(settings_dict.keys()):
            info = settings_dict[key]
            value = info.get("value", "")
            setting_type = info.get("type", "text")
            description = info.get("description", "")

            # Render appropriate widget based on type
            if setting_type == "secret":
                form_values[key] = st.text_input(
                    label=key,
                    value=value,
                    type="password",
                    help=description,
                    key=f"setting_{key}",
                )

            elif setting_type == "select":
                options = SELECT_OPTIONS.get(key, [value])
                current_index = options.index(value) if value in options else 0
                form_values[key] = st.selectbox(
                    label=key,
                    options=options,
                    index=current_index,
                    help=description,
                    key=f"setting_{key}",
                )

            elif setting_type == "bool":
                form_values[key] = str(
                    st.toggle(
                        label=key,
                        value=str(value).lower() in ("true", "1", "yes", "on"),
                        help=description,
                        key=f"setting_{key}",
                    )
                ).lower()

            elif setting_type == "float":
                # Determine slider range based on key
                if "temperature" in key:
                    min_val, max_val, step = 0.0, 2.0, 0.1
                elif "score" in key:
                    min_val, max_val, step = 0.0, 1.0, 0.05
                else:
                    min_val, max_val, step = 0.0, 10.0, 0.1

                try:
                    current_float = float(value)
                except (ValueError, TypeError):
                    current_float = min_val

                form_values[key] = str(
                    st.slider(
                        label=key,
                        min_value=min_val,
                        max_value=max_val,
                        value=current_float,
                        step=step,
                        help=description,
                        key=f"setting_{key}",
                    )
                )

            elif setting_type == "int":
                try:
                    current_int = int(value)
                except (ValueError, TypeError):
                    current_int = 0

                form_values[key] = str(
                    st.number_input(
                        label=key,
                        value=current_int,
                        step=1,
                        help=description,
                        key=f"setting_{key}",
                    )
                )

            else:  # text
                form_values[key] = st.text_input(
                    label=key,
                    value=value,
                    help=description,
                    key=f"setting_{key}",
                )

        # Reset button per category
        if st.button(f"🔄 Reset {label} to Defaults", key=f"reset_{category}"):
            result = reset_category(api_url, category)
            if result.get("status") == "ok":
                st.success(f"Reset {result.get('reset_count', 0)} settings to defaults")
                st.rerun()
            else:
                st.error(f"Reset failed: {result.get('error', 'Unknown error')}")


# =============================================================================
# SAVE BUTTON
# =============================================================================

st.markdown("---")

col_save, col_reset_all = st.columns([1, 1])

with col_save:
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        # Find which values actually changed
        changes = {}
        for category in sorted_categories:
            for key, info in all_config[category].items():
                old_value = info.get("value", "")
                new_value = form_values.get(key, old_value)
                # For secrets, skip if the masked value hasn't changed
                if info.get("type") == "secret" and new_value == old_value:
                    continue
                if str(new_value) != str(old_value):
                    changes[key] = str(new_value)

        if not changes:
            st.info("No changes detected.")
        else:
            result = save_config(api_url, changes)
            if result.get("status") == "ok":
                updated = result.get("updated", [])
                st.success(f"✅ Saved {len(updated)} setting(s): {', '.join(updated)}")
                st.rerun()
            else:
                st.error(f"Save failed: {result.get('error', 'Unknown error')}")

with col_reset_all:
    if st.button("🔄 Reset ALL to Defaults", use_container_width=True):
        result = reset_category(api_url)
        if result.get("status") == "ok":
            st.success(f"Reset {result.get('reset_count', 0)} settings to defaults")
            st.rerun()
        else:
            st.error(f"Reset failed: {result.get('error', 'Unknown error')}")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "⚠️ **Note:** Changes to API keys, infrastructure hosts, and embedding models "
    "may require a service restart to take full effect. "
    "LLM model/temperature changes take effect on the next query."
)
