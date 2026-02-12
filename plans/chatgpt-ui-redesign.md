# ChatGPT-like UI Redesign Plan

## Current State

The current UI is a **Streamlit** multi-page app (`ui/app.py` + `ui/pages/`) with:
- Default Streamlit styling — generic, not modern
- Cluttered sidebar mixing navigation, settings, filters, and stats
- Chat/Search tabs instead of a focused chat interface
- No dark theme, no empty state landing, no streaming
- No time-grouped conversation history (Today, Yesterday, etc.)
- WhatsApp-inspired green theming that feels disconnected

## Recommended Approach: Replace Streamlit with Standalone SPA

### Why Not Keep Streamlit?

Streamlit's architecture (full-page reruns on every interaction) fundamentally prevents smooth ChatGPT-like UX:
- Cannot independently theme sidebar dark vs main area light
- Cannot hide Streamlit's navigation/header/footer cleanly
- Cannot do streaming with typing indicators
- Cannot do hover-to-reveal actions on conversation items
- Multi-page navigation pollutes the sidebar
- Limited responsive design control

### Why Standalone HTML/CSS/JS?

- **Full pixel-perfect control** over the UI
- **Zero backend changes** — Flask API already has all endpoints
- Can be served directly by Flask (add a static route)
- Lightweight — no complex build tools required
- Modern CSS for exact ChatGPT replication
- Supports streaming, animations, hover effects
- Optional lightweight libraries: `marked.js` for markdown, `highlight.js` for code blocks

---

## Architecture

```mermaid
graph TB
    subgraph Frontend [New SPA Frontend]
        HTML[index.html]
        CSS[styles.css]
        JS[app.js]
        MD[marked.js + highlight.js]
    end

    subgraph Backend [Existing Flask API - No Changes]
        CONV[/conversations endpoints]
        RAG[/rag/query endpoint]
        SEARCH[/rag/search endpoint]
        STATS[/rag/stats endpoint]
        FILTERS[/rag/chats + /rag/senders]
        CONFIG[/config endpoints]
        HEALTH[/health endpoint]
    end

    Frontend -->|fetch API| Backend
```

### File Structure

```
ui/
├── static/
│   ├── index.html          # Main SPA entry point
│   ├── css/
│   │   └── styles.css      # ChatGPT-inspired styles
│   ├── js/
│   │   ├── app.js          # Main application logic
│   │   ├── api.js          # API client wrapper
│   │   ├── chat.js         # Chat rendering and interaction
│   │   ├── sidebar.js      # Sidebar/conversation management
│   │   ├── settings.js     # Settings modal/panel
│   │   └── markdown.js     # Markdown rendering setup
│   └── lib/
│       ├── marked.min.js   # Markdown parser
│       └── highlight.min.js # Code syntax highlighting
├── app.py                  # Keep as fallback or remove
└── pages/                  # Keep as fallback or remove
```

### Serving Strategy

Add a Flask route in `src/app.py` to serve the SPA:

```python
@app.route("/ui")
@app.route("/ui/<path:path>")
def serve_ui(path="index.html"):
    return send_from_directory("../ui/static", path)
```

---

## UI Design Specification

### Layout — Three-Panel Design

```
┌─────────────────┬────────────────────────────────────┐
│                  │                                    │
│    SIDEBAR       │          MAIN CHAT AREA            │
│    260px fixed   │          flex: 1                   │
│                  │                                    │
│  ┌────────────┐  │  ┌──────────────────────────────┐  │
│  │ + New Chat │  │  │     Centered container       │  │
│  └────────────┘  │  │     max-width: 768px         │  │
│                  │  │                              │  │
│  Search...       │  │  ┌──────────────────────┐    │  │
│                  │  │  │  Message bubbles      │    │  │
│  ── Today ──     │  │  │  user / assistant     │    │  │
│  Conv title 1    │  │  │  with avatars         │    │  │
│  Conv title 2    │  │  └──────────────────────┘    │  │
│                  │  │                              │  │
│  ── Yesterday ── │  │                              │  │
│  Conv title 3    │  │  ┌──────────────────────┐    │  │
│                  │  │  │  Input bar            │    │  │
│  ── Previous ──  │  │  │  fixed at bottom      │    │  │
│  Conv title 4    │  │  └──────────────────────┘    │  │
│                  │  │                              │  │
│  ┌────────────┐  │  └──────────────────────────────┘  │
│  │ ⚙ Settings │  │                                    │
│  │ 👤 Profile  │  │                                    │
│  └────────────┘  │                                    │
└─────────────────┴────────────────────────────────────┘
```

### Color Scheme — Dark Sidebar, Light Main

| Element | Color | CSS Variable |
|---------|-------|-------------|
| Sidebar background | #171717 | --sidebar-bg |
| Sidebar text | #ECECEC | --sidebar-text |
| Sidebar hover | #2A2A2A | --sidebar-hover |
| Sidebar active | #343541 | --sidebar-active |
| Main background | #FFFFFF | --main-bg |
| Main text | #374151 | --main-text |
| User message bg | #F7F7F8 | --user-msg-bg |
| Assistant message bg | #FFFFFF | --assistant-msg-bg |
| Input bar bg | #FFFFFF | --input-bg |
| Input border | #D1D5DB | --input-border |
| Accent/brand | #10A37F | --accent |
| Code block bg | #1E1E1E | --code-bg |

### Key UI Components

#### 1. Sidebar
- **New Chat** button with `+` icon at top
- **Search bar** to filter conversations
- **Conversation list** grouped by time period:
  - Today
  - Yesterday
  - Previous 7 Days
  - Previous 30 Days
  - Older
- Each conversation item shows:
  - Title (truncated to ~35 chars)
  - Hover reveals: rename icon, delete icon
- **Bottom section**: Settings gear, health status indicator

#### 2. Empty State (No Active Chat)
- Centered logo/icon
- App title: "WhatsApp RAG Assistant"
- 3-4 suggestion cards the user can click:
  - "What did [name] say about..."
  - "Summarize the last week of messages"
  - "Search for messages about..."
  - "Who mentioned [topic] recently?"

#### 3. Chat Messages
- **User messages**: Right-aligned text, subtle background, user avatar
- **Assistant messages**: Left-aligned, white background, bot avatar
- **Markdown rendering**: Headers, bold, italic, lists, links
- **Code blocks**: Syntax highlighted with copy button
- **Sources**: Collapsible "View sources" section after assistant messages
- **Loading**: Animated typing indicator dots while waiting

#### 4. Input Bar
- Fixed at bottom of chat area
- Rounded textarea that auto-grows up to ~200px
- Send button (arrow icon) on the right
- Disabled state while waiting for response
- Filter indicator chips above input bar when filters are active

#### 5. Settings Panel
- Slides in as a modal/drawer from sidebar
- Sections for: API Config, Filters, RAG Settings
- Quick filter dropdowns: Chat/Group, Sender, Time Range
- Health status with dependency indicators
- Save/Reset buttons

#### 6. Filter System
- Filter chips displayed above the input bar
- Click chip `×` to remove a filter
- Filters panel accessible via a funnel icon in the input bar area

---

## Existing API Endpoints (No Backend Changes Needed)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/conversations` | GET | List conversations |
| `/conversations/:id` | GET | Get conversation with messages |
| `/conversations` | POST | Create conversation |
| `/conversations/:id` | PUT | Rename conversation |
| `/conversations/:id` | DELETE | Delete conversation |
| `/rag/query` | POST | Chat query with RAG |
| `/rag/search` | POST | Semantic search |
| `/rag/stats` | GET | Vector store stats |
| `/rag/chats` | GET | List of chat names for filters |
| `/rag/senders` | GET | List of sender names for filters |
| `/rag/messages` | GET | Browse raw messages |
| `/health` | GET | System health check |
| `/config` | GET | Get all settings |
| `/config` | PUT | Update settings |
| `/config/reset` | POST | Reset to defaults |

---

## Implementation Tasks

### Phase 1: Core Chat Interface
1. Create `ui/static/index.html` — HTML skeleton with sidebar + main layout
2. Create `ui/static/css/styles.css` — Full ChatGPT-inspired dark sidebar / light main theme
3. Create `ui/static/js/api.js` — API client with all endpoint wrappers
4. Create `ui/static/js/app.js` — Main app controller, routing, state management
5. Create `ui/static/js/chat.js` — Chat message rendering, markdown, loading states
6. Create `ui/static/js/sidebar.js` — Conversation list, time grouping, search, CRUD
7. Add Flask route to serve the SPA from `src/app.py`

### Phase 2: Polish and Features
8. Create `ui/static/js/settings.js` — Settings modal with health check and config editing
9. Add empty state with suggestion cards
10. Add filter system with chips and filter panel
11. Add markdown rendering with `marked.js` + `highlight.js`
12. Add typing indicator animation during response loading
13. Add keyboard shortcuts (Enter to send, Escape to cancel, Ctrl+N for new chat)

### Phase 3: Advanced UX
14. Add conversation search/filter in sidebar
15. Add mobile-responsive layout (collapsible sidebar)
16. Add semantic search as a secondary mode (accessible from input bar)
17. Add export chat functionality
18. Add CORS configuration to Flask if needed for development

---

## What Gets Replaced vs Kept

| Component | Action |
|-----------|--------|
| `ui/app.py` | **Replaced** — Streamlit no longer used |
| `ui/pages/1_Settings.py` | **Replaced** — Settings modal in new UI |
| `ui/pages/2_Messages.py` | **Replaced** — Integrated into search mode |
| `ui/components/` | **Removed** — Empty directory |
| `src/app.py` | **Keep + Add** static serving route |
| All other backend files | **Unchanged** |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| CORS issues when developing frontend separately | Add Flask-CORS or serve from same origin |
| Losing Streamlit features we use | The Flask API already provides all data |
| RTL text support for Hebrew content | CSS `direction: auto` on message elements |
| No streaming support in current API | Phase 1 uses loading indicator; streaming can be added later |
