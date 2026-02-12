# ChatGPT-like UI Redesign Plan (Streamlit)

## Goal
Redesign the existing Streamlit UI to closely match ChatGPT's look and feel while staying within the Streamlit framework.

## Current State Issues
1. Default Streamlit styling — generic, not modern
2. Sidebar is cluttered: navigation + settings + filters + stats + conversations all mixed
3. Chat/Search tabs split the main interface
4. No dark sidebar theme
5. No empty state / landing page with suggestions
6. No time-grouped conversation history
7. Streamlit page navigation visible in sidebar
8. WhatsApp green theming feels disconnected from ChatGPT goal

## What Streamlit CAN Do (With Heavy CSS)
- Dark sidebar with light main area via CSS overrides on `[data-testid=stSidebar]`
- Hide Streamlit header, footer, hamburger menu, page navigation via CSS
- `st.chat_message` + `st.chat_input` for chat bubbles
- `st.write_stream` for streaming responses
- Time-grouped conversation list in sidebar
- Hover effects via CSS `:hover` on sidebar elements
- Empty state with centered content and clickable suggestion buttons
- Markdown rendering built-in
- Custom theme via `.streamlit/config.toml`

## What Will Be Compromised (Streamlit Limitations)
- Hover-to-reveal rename/delete icons: will use small always-visible icon buttons instead
- Smooth animations/transitions: limited to CSS-only effects
- Auto-growing textarea: `st.chat_input` is fixed height
- True modal/drawer for settings: will use expander or conditional rendering
- Full responsive mobile layout: partial control only

---

## Architecture

```mermaid
graph LR
    subgraph Streamlit UI
        APP[ui/app.py - Single Page]
        THEME[.streamlit/config.toml]
        CSS[Custom CSS Injection]
    end

    subgraph Backend [Existing Flask API - No Changes]
        CONV[/conversations]
        RAG[/rag/query]
        HEALTH[/health]
        CONFIG[/config]
    end

    APP -->|requests| Backend
```

### Restructured File Layout

```
ui/
├── app.py                  # Main and ONLY page — consolidated single-page chat UI
├── .streamlit/
│   └── config.toml         # Custom Streamlit theme
├── components/
│   ├── sidebar.py          # Sidebar: new chat, conversation list, settings toggle
│   ├── chat.py             # Chat area: messages, input, empty state
│   ├── settings_panel.py   # Settings: filters, config, health status
│   └── styles.py           # All custom CSS as Python string constants
├── utils/
│   ├── api.py              # API client wrapper functions
│   └── time_utils.py       # Time grouping and formatting helpers
└── pages/                  # REMOVED or emptied — single page app
```

---

## UI Layout Specification

```
┌──────────────────────┬──────────────────────────────────────┐
│  DARK SIDEBAR        │  LIGHT MAIN AREA                     │
│  bg: #171717         │  bg: #FFFFFF                          │
│  width: ~300px       │                                      │
│                      │  ┌──────────────────────────────┐    │
│  [+ New Chat]        │  │   Centered Chat Container     │    │
│                      │  │   max-width: 768px            │    │
│  ── Today ────       │  │                               │    │
│  ▸ Conv title 1      │  │   👤 User message             │    │
│  ▸ Conv title 2      │  │                               │    │
│                      │  │   🤖 Assistant message         │    │
│  ── Yesterday ───    │  │      📎 Sources [expand]       │    │
│  ▸ Conv title 3      │  │                               │    │
│                      │  │   👤 User message             │    │
│  ── Previous 7d ──   │  │                               │    │
│  ▸ Conv title 4      │  │   🤖 Assistant message         │    │
│                      │  │      ··· typing indicator      │    │
│                      │  │                               │    │
│                      │  └──────────────────────────────┘    │
│                      │                                      │
│  ─────────────────   │  ┌──────────────────────────────┐    │
│  ⚙ Settings          │  │ [Filter chips: Chat | Sender] │    │
│  🟢 API Connected    │  │ Ask about your WhatsApp msgs… │    │
│                      │  └──────────────────────────────┘    │
└──────────────────────┴──────────────────────────────────────┘
```

### Empty State (No Active Chat)

```
┌──────────────────────────────────────┐
│                                      │
│          💬                           │
│   WhatsApp RAG Assistant             │
│                                      │
│   ┌──────────┐  ┌──────────┐        │
│   │ What did  │  │ Summarize│        │
│   │ X say     │  │ last week│        │
│   │ about...  │  │ messages │        │
│   └──────────┘  └──────────┘        │
│   ┌──────────┐  ┌──────────┐        │
│   │ Search    │  │ Who said │        │
│   │ messages  │  │ something│        │
│   │ about...  │  │ about... │        │
│   └──────────┘  └──────────┘        │
│                                      │
│  [Ask about your WhatsApp msgs...]   │
└──────────────────────────────────────┘
```

---

## Detailed CSS Theme

### Dark Sidebar

```css
/* Dark sidebar background */
[data-testid="stSidebar"] {
    background-color: #171717;
    color: #ECECEC;
}
/* Sidebar text and labels */
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #ECECEC;
}
/* Sidebar buttons — ghost style */
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent;
    color: #ECECEC;
    border: 1px solid #444;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2A2A2A;
}
/* New Chat button — accent */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: transparent;
    border: 1px solid #555;
    color: #ECECEC;
}
```

### Hide Streamlit Chrome

```css
/* Hide header, footer, hamburger menu */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
/* Hide page navigation in sidebar */
[data-testid="stSidebarNav"] { display: none !important; }
/* Remove default padding */
.block-container { padding-top: 1rem; }
```

### Chat Message Styling

```css
/* Centered chat container */
.block-container {
    max-width: 768px;
    margin: 0 auto;
}
/* User messages — subtle gray background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #F7F7F8;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
/* Assistant messages — white/clean */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF;
    padding: 12px 16px;
    margin-bottom: 8px;
}
```

### Conversation List Items

```css
/* Active conversation highlight */
.chat-item-active button {
    background-color: #2A2A2A !important;
    border-left: 3px solid #10A37F !important;
}
/* Conversation time group headers */
.time-group-header {
    color: #888;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 8px 12px 4px;
}
```

---

## Streamlit Theme Config

### `.streamlit/config.toml`

```toml
[theme]
base = "light"
primaryColor = "#10A37F"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F7F7F8"
textColor = "#374151"
font = "sans serif"

[server]
headless = true
```

---

## Implementation Tasks

### Phase 1: Restructure and Core Theme
1. Create `.streamlit/config.toml` with ChatGPT-like light theme
2. Create `ui/components/styles.py` with all CSS constants for dark sidebar, hidden chrome, chat styling
3. Create `ui/utils/api.py` — consolidated API client (extract from current app.py)
4. Create `ui/utils/time_utils.py` — time grouping logic (Today, Yesterday, Previous 7 Days, etc.)
5. Restructure `ui/app.py` as single-page app: remove tabs, consolidate chat as primary view

### Phase 2: Sidebar Redesign
6. Create `ui/components/sidebar.py` — dark-themed sidebar component with:
   - New Chat button (ghost style with border)
   - Time-grouped conversation list
   - Compact rename/delete action buttons per conversation
   - Settings toggle at bottom
   - Health status indicator at bottom
7. Remove multi-page navigation (delete or empty `ui/pages/`)

### Phase 3: Chat Area Redesign
8. Create `ui/components/chat.py` — chat area component with:
   - Empty state with centered logo and suggestion cards
   - Chat message rendering with proper styling
   - Source citations as collapsible expanders
   - Active filter chips displayed above input
9. Add streaming response support using `st.write_stream`

### Phase 4: Settings and Filters
10. Create `ui/components/settings_panel.py` — settings as sidebar expander:
    - Filter dropdowns (Chat/Group, Sender, Time Range)
    - API configuration
    - Health status dashboard
    - Stats display
11. Add filter chip display and removal in chat area

### Phase 5: Polish
12. Add keyboard interaction improvements
13. Add conversation search/filter in sidebar
14. Fine-tune CSS for spacing, typography, and responsive behavior
15. Test RTL text support for Hebrew content
16. Add export chat functionality

---

## Component Responsibility Map

| Component | Responsibility |
|-----------|---------------|
| `ui/app.py` | Page config, CSS injection, layout orchestration, session state init |
| `ui/components/sidebar.py` | New chat, conversation list with time groups, settings toggle, health |
| `ui/components/chat.py` | Empty state, message display, input handling, filter chips, streaming |
| `ui/components/settings_panel.py` | Filters, config editing, health dashboard, stats |
| `ui/components/styles.py` | All CSS constants: dark sidebar, chat bubbles, hidden chrome, etc. |
| `ui/utils/api.py` | All `requests` calls to Flask backend |
| `ui/utils/time_utils.py` | `relative_time()`, `group_by_time_period()`, timestamp formatting |

---

## Session State Design

```python
# Core state
st.session_state.messages          # List of chat messages
st.session_state.conversation_id   # Current active conversation UUID
st.session_state.active_filters    # Dict of active RAG filters

# UI state
st.session_state.show_settings     # Toggle settings panel visibility
st.session_state.renaming_id       # Which conversation is being renamed
st.session_state.sidebar_search    # Sidebar search filter text

# Config
st.session_state.api_url           # Backend API URL
st.session_state.k_results         # Number of context documents
```

---

## Migration Notes

- The existing `ui/app.py` logic (API calls, session state, chat rendering) is preserved and refactored into components
- `ui/pages/1_Settings.py` functionality moves into `settings_panel.py`
- `ui/pages/2_Messages.py` functionality can be accessed via semantic search in the chat input
- No backend changes required whatsoever
