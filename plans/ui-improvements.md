# WhatsApp-GPT UI: Improvements & New Features Plan

## Executive Summary

This document outlines recommended improvements and new features for the Streamlit-based UI ([`ui/app.py`](../ui/app.py)). The current UI provides RAG Q&A with multi-turn conversations, semantic search, and a Settings page.

**Last Updated:** 2026-02-12

---

## Current UI Overview

### Existing Features
- Basic RAG Q&A chat interface
- Configurable API URL
- Number of context documents (k) slider
- Filter by chat name dropdown
- Filter by sender dropdown
- Filter by time range dropdown
- Chat history with clear button
- RAG stats display

### Current Architecture
```mermaid
flowchart LR
    subgraph Streamlit UI
        Chat[Chat Interface]
        Search[Search Tab]
        Sidebar[Sidebar Filters]
        Stats[RAG Stats]
        Settings[Settings Page]
    end
    
    subgraph Flask API
        RAG[/rag/query]
        RS[/rag/search]
        Chats[/rag/chats]
        Senders[/rag/senders]
        RStats[/rag/stats]
        CFG[/config]
        Health[/health]
    end
    
    Chat --> RAG
    Search --> RS
    Sidebar --> Chats
    Sidebar --> Senders
    Stats --> RStats
    Settings --> CFG
    Settings --> Health
```

---

## Part 1: User Experience Improvements

### 1.1 Visual Design & Theming

**Current Issues:**
- Default Streamlit styling
- No custom branding
- Limited visual hierarchy

**Recommendations:**
- [ ] Add custom CSS for improved styling
- [ ] Implement dark/light mode toggle
- [ ] Add WhatsApp-style message bubbles
- [ ] Custom color theme matching WhatsApp green (#25D366)
- [ ] Add loading animations/skeletons

**Example Implementation:**
```python
st.markdown("""
<style>
.user-message {
    background-color: #DCF8C6;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
    text-align: right;
}
.assistant-message {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)
```

### 1.2 Responsive Layout

**Current Issues:**
- Single `centered` layout option
- Poor mobile experience
- No adaptive sidebar

**Recommendations:**
- [ ] Switch to `wide` layout with columns
- [ ] Add collapsible sidebar for mobile
- [ ] Implement responsive breakpoints
- [ ] Add touch-friendly controls

### 1.3 Error Handling & Feedback

**Current Issues:**
- Generic error messages
- No retry mechanism in UI
- Connection errors not handled gracefully

**Recommendations:**
- [ ] Add toast notifications for errors
- [ ] Implement automatic retry with backoff
- [ ] Add connection status indicator
- [ ] Show helpful error recovery suggestions

---

## Part 2: Feature Enhancements

### 2.1 Multi-Page Application (High Priority)

**Current State:** Multi-page structure started — main app + Settings page.

**Completed ✅:**
- [x] Convert to Streamlit multi-page app structure
- [x] ⚙️ Settings & Configuration page (`ui/pages/1_Settings.py`)

**Remaining:**
- [ ] Add dedicated pages:
  - 📊 Dashboard (home/overview)
  - 📱 Message Browser
  - 📈 Analytics

**File Structure:**
```
ui/
├── app.py              # Main entry (dashboard)
├── pages/
│   ├── 1_💬_RAG_QA.py
│   ├── 2_📱_Messages.py
│   ├── 3_📈_Analytics.py
│   └── 4_⚙️_Settings.py
└── components/
    ├── chat_bubble.py
    ├── filters.py
    └── charts.py
```

### 2.2 Message Browser (High Priority)

**Feature:**
- [ ] View actual WhatsApp messages stored in RAG
- [ ] Search and filter messages
- [ ] Display message metadata (timestamp, sender, chat)
- [ ] Preview images/attachments
- [ ] Export selected messages

**API Endpoints Needed:**
```python
GET /rag/messages?chat_name=...&sender=...&limit=50&offset=0
GET /rag/message/{message_id}
```

### 2.3 Analytics Dashboard (Medium Priority)

**Feature:**
- [ ] Messages per day/week/month chart
- [ ] Top active chats
- [ ] Top senders
- [ ] Word cloud from messages
- [ ] RAG query performance metrics
- [ ] Response time trends

**Charts to Implement:**
```python
# Using plotly
import plotly.express as px

# Messages over time
fig = px.line(df, x='date', y='count', title='Messages Over Time')

# Top chats pie chart
fig = px.pie(df, values='count', names='chat', title='Messages by Chat')
```

### 2.4 Real-Time Message Feed (Medium Priority)

**Feature:**
- [ ] WebSocket connection for live updates
- [ ] Show new messages as they arrive
- [ ] Notification sound/badge for new messages
- [ ] Auto-refresh toggle

**Implementation:**
```python
# Using streamlit-websocket or polling
import time

if st.session_state.auto_refresh:
    time.sleep(5)
    st.rerun()
```

### 2.5 Direct Chat Interface (High Priority)

**Feature:**
- [ ] Send messages directly to WhatsApp chats from UI
- [ ] AI-assisted response suggestions
- [ ] Reply to specific messages
- [ ] Support for media sending

**API Endpoints Needed:**
```python
POST /whatsapp/send
{
    "chat_id": "...",
    "message": "...",
    "reply_to": "..."  # optional
}
```

### 2.6 Configuration Management (Medium Priority)

**Feature:**
- [ ] View/edit bot configuration
- [ ] Manage AI response triggers
- [ ] Set per-chat preferences
- [ ] Toggle features (voice, images, etc.)
- [ ] Manage allowed/blocked contacts

**Settings to Expose:**
- AI trigger prefix (e.g., `??`)
- Auto-reply mode per chat
- Response language preference
- LLM model selection
- Temperature/creativity settings

### 2.7 RAG Collection Management (Low Priority)

**Feature:**
- [ ] View all RAG collections
- [ ] Delete/rebuild collections
- [ ] Upload documents to RAG
- [ ] Manage vector indexes
- [ ] Export/import RAG data

### 2.8 Context Sources Display (Medium Priority)

**Current Issue:** RAG answers don't show source messages

**Feature:**
- [ ] Display source messages used for RAG answers
- [ ] Show relevance scores
- [ ] Clickable sources to view full context
- [ ] Citation-style references

**Example UI:**
```
Answer: [Generated answer here]

📎 Sources:
1. [John in Family Chat, 2 days ago] "..." (95% relevant)
2. [Jane in Work Group, 1 week ago] "..." (87% relevant)
```

---

## Part 3: Technical Improvements

### 3.1 Session Management

**Completed (partial) ✅:**
- [x] Conversation state maintained via `st.session_state` + backend `conversation_id`
- [x] Filter selections persisted per conversation via Redis hash

**Remaining:**
- [ ] Add session persistence using cookies/localStorage
- [ ] Add optional authentication

### 3.2 Caching & Performance ✅ DONE

**Completed ✅:**
- [x] `@st.cache_data(ttl=300)` on `get_chat_list()` and `get_sender_list()`
- [x] `@st.cache_data(ttl=60)` on `get_rag_stats()`

### 3.3 API Client Refactoring

**Current Issues:**
- Inline API calls scattered in code
- Inconsistent error handling
- No request timeout configuration

**Recommendations:**
- [ ] Create dedicated API client class
- [ ] Centralize error handling
- [ ] Add request/response logging
- [ ] Implement retry logic

**Example:**
```python
# ui/api_client.py
class WhatsAppGPTClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
    
    def query_rag(self, question: str, **filters) -> dict:
        # Implementation with error handling
        pass
    
    def get_chats(self) -> list:
        pass
```

### 3.4 Component Architecture

**Recommendations:**
- [ ] Extract reusable components
- [ ] Create filter component
- [ ] Create message display component
- [ ] Create stats card component

---

## Part 4: Integration Features

### 4.1 Direct AI Chat (High Priority)

**Feature:**
- [ ] Chat directly with the LangGraph AI from UI
- [ ] Maintain conversation context
- [ ] Access same tools as WhatsApp bot
- [ ] Test AI responses before deploying

### 4.2 Webhook Monitor (Low Priority)

**Feature:**
- [ ] View incoming webhook events
- [ ] Debug message processing
- [ ] View error logs
- [ ] Replay failed webhooks

### 4.3 Contact/Group Management (Medium Priority)

**Feature:**
- [ ] View all contacts and groups
- [ ] Edit contact names/aliases
- [ ] Set per-contact AI preferences
- [ ] Block/unblock contacts

---

## Part 5: Security & Access Control

### 5.1 Authentication

**Recommendations:**
- [ ] Add login page with password protection
- [ ] Support for multiple users
- [ ] Role-based access (admin, viewer)
- [ ] Session timeout

**Implementation Options:**
1. Streamlit native authentication (`st.experimental_user`)
2. HTTP Basic Auth with nginx
3. Custom authentication with database

### 5.2 Audit Logging

**Recommendations:**
- [ ] Log all UI actions
- [ ] Track who sent which messages
- [ ] Record configuration changes
- [ ] Export audit logs

---

## Implementation Priority Matrix

| Priority | Feature | Impact | Effort | Status |
|----------|---------|--------|--------|--------|
| ~~🔴 High~~ | ~~Multi-Page App Structure~~ | High | Medium | ✅ Started (Settings page) |
| 🔴 High | Direct Chat Interface | High | Medium | ⏳ Pending |
| 🔴 High | Message Browser | High | Medium | ⏳ Pending |
| 🟡 Medium | Analytics Dashboard | Medium | Medium | ⏳ Pending |
| 🟡 Medium | Context Sources Display | Medium | Low | ⏳ Pending |
| ~~🟡 Medium~~ | ~~Configuration Management~~ | Medium | Medium | ✅ Done (Settings page) |
| 🟡 Medium | Visual Design & Theming | Medium | Low | ⏳ Pending |
| 🟢 Low | Real-Time Message Feed | Low | High | ⏳ Pending |
| 🟢 Low | Authentication | Low | Medium | ⏳ Pending |
| 🟢 Low | RAG Collection Management | Low | Medium | ⏳ Pending |

---

## Quick Wins

| Task | Effort | Impact | Status |
|------|--------|--------|--------|
| Add `@st.cache_data` decorators | Low | Medium | ✅ Done |
| Add custom WhatsApp-style CSS | Low | Medium | ⏳ Pending |
| Show source documents in answers | Low | High | ⏳ Pending |
| Add connection status indicator | Low | Medium | ⏳ Pending |
| Add export chat history button | Low | Medium | ⏳ Pending |
| Improve error messages | Low | Medium | ⏳ Pending |

---

## API Endpoints Needed

The following new API endpoints would enhance UI capabilities:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/messages` | GET | Paginated message browser |
| `/rag/message/{id}` | GET | Single message with full context |
| `/whatsapp/send` | POST | Send message to WhatsApp |
| `/whatsapp/contacts` | GET | List all contacts |
| `/whatsapp/groups` | GET | List all groups |
| `/config` | GET/PUT | Bot configuration |
| `/ai/chat` | POST | Direct AI chat endpoint |
| `/analytics/overview` | GET | Dashboard statistics |
| `/analytics/messages-by-day` | GET | Time-series data |

---

## Proposed File Structure

```
ui/
├── app.py                      # Main dashboard page
├── api_client.py               # Centralized API client
├── config.py                   # UI configuration
├── pages/
│   ├── 1_💬_RAG_QA.py         # RAG Q&A (current functionality)
│   ├── 2_📱_Messages.py        # Message browser
│   ├── 3_🤖_AI_Chat.py        # Direct AI chat
│   ├── 4_📈_Analytics.py       # Analytics dashboard
│   ├── 5_👥_Contacts.py        # Contact management
│   └── 6_⚙️_Settings.py       # Configuration
├── components/
│   ├── __init__.py
│   ├── chat_bubble.py          # WhatsApp-style message bubble
│   ├── filters.py              # Reusable filter components
│   ├── stats_card.py           # Metric display cards
│   └── charts.py               # Chart components
├── styles/
│   └── main.css                # Custom CSS
└── assets/
    └── logo.png                # Branding assets
```

---

## Questions for Discussion

1. **Authentication:** Is user authentication needed for the UI?
2. **Multi-user:** Should multiple users have access with different permissions?
3. **Mobile:** Is mobile support a priority?
4. **Real-time:** How important is real-time message updates?
5. **Analytics:** What metrics are most valuable to track?

---

## Dependencies to Add

```txt
# UI enhancements
plotly>=5.18.0          # Interactive charts
streamlit-extras>=0.3.5  # Additional components
streamlit-option-menu>=0.3.6  # Better navigation
pandas>=2.0.0           # Data manipulation
```

---

## References

- [Streamlit Multi-Page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [Streamlit Components](https://streamlit.io/components)
- [Plotly Python](https://plotly.com/python/)
- [WhatsApp Design Guidelines](https://developers.facebook.com/docs/whatsapp/guides/design)
