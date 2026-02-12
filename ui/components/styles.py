"""All custom CSS for the ChatGPT-inspired Streamlit UI.

Centralises every CSS override in one place so the main app
and components only need to call ``inject_styles()``.
"""

import streamlit as st


def inject_styles() -> None:
    """Inject all custom CSS into the Streamlit page."""
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CSS CONSTANTS
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
/* =======================================================================
   0. HIDE STREAMLIT CHROME
   ======================================================================= */

/* Header bar (contains hamburger + app title) */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Footer "Made with Streamlit" */
footer {
    display: none !important;
}

/* Hamburger menu */
#MainMenu {
    display: none !important;
}

/* Multi-page navigation links in sidebar */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Reduce top padding now that header is hidden */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
}

/* =======================================================================
   1. DARK SIDEBAR
   ======================================================================= */

section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    color: #ECECEC !important;
    border-right: 1px solid #2A2A2A !important;
}

/* Sidebar inner wrapper */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #171717 !important;
}

/* All text elements inside sidebar */
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] span {
    color: #ECECEC !important;
}

/* Sidebar caption / muted text */
section[data-testid="stSidebar"] .stCaption p {
    color: #888 !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {
    border-color: #333 !important;
    margin: 8px 0 !important;
}

/* =======================================================================
   2. SIDEBAR BUTTONS
   ======================================================================= */

/* All sidebar buttons — clean, borderless by default (conversation items) */
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: #ECECEC !important;
    border: none !important;
    border-radius: 8px !important;
    transition: background-color 0.15s ease;
    font-size: 0.85rem !important;
    text-align: left !important;
    padding: 8px 12px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #212121 !important;
}

/* Active/disabled conversation button — subtle highlight */
section[data-testid="stSidebar"] .stButton > button:disabled {
    background-color: #2A2A2A !important;
    color: #ECECEC !important;
    opacity: 1 !important;
    border-left: 3px solid #10A37F !important;
    cursor: default !important;
}

/* Primary button (New Chat) — distinct with dashed border */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: transparent !important;
    border: 1px dashed #555 !important;
    color: #ECECEC !important;
    text-align: center !important;
    padding: 10px 16px !important;
    margin-bottom: 8px !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #2A2A2A !important;
    border-style: solid !important;
}

/* Three-dot (⋯) menu button — small, subtle, blends with sidebar */
/* Targets the narrow column button in conversation rows */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton > button {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #555 !important;
    padding: 2px !important;
    min-height: 0 !important;
    font-size: 1rem !important;
    border-radius: 4px !important;
    line-height: 1 !important;
    opacity: 0.3;
    transition: opacity 0.15s ease;
}

section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton > button:hover {
    opacity: 1 !important;
    color: #ECECEC !important;
    background-color: #333 !important;
}

/* Inline action buttons (Rename / Delete) — small, muted */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button[data-testid="stBaseButton-secondary"] {
    font-size: 0.75rem !important;
    padding: 4px 8px !important;
    min-height: 0 !important;
    opacity: 0.8 !important;
    color: #AAA !important;
    border: 1px solid #444 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stPopover [data-testid="stPopoverBody"] .stButton > button:hover {
    background-color: #3A3A3A !important;
}

/* Conversation rows — tighter alignment */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
}

/* =======================================================================
   3. SIDEBAR TEXT INPUTS
   ======================================================================= */

section[data-testid="stSidebar"] .stTextInput input {
    background-color: #212121 !important;
    color: #ECECEC !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #888 !important;
}

section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #10A37F !important;
    box-shadow: 0 0 0 1px #10A37F !important;
}

/* =======================================================================
   4. SIDEBAR SELECT BOXES
   ======================================================================= */

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #212121 !important;
    color: #ECECEC !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
}

/* =======================================================================
   5. MAIN CHAT AREA
   ======================================================================= */

/* Centered chat container */
.main .block-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* =======================================================================
   6. CHAT MESSAGE BUBBLES
   ======================================================================= */

/* User messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #F7F7F8 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 4px !important;
    border: none !important;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 4px !important;
    border: none !important;
}

/* Chat message text */
[data-testid="stChatMessage"] .stMarkdown p {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #374151 !important;
}

/* Code blocks inside chat */
[data-testid="stChatMessage"] pre {
    background-color: #1E1E1E !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

[data-testid="stChatMessage"] code {
    font-size: 0.85rem !important;
}

/* =======================================================================
   7. CHAT INPUT BAR
   ======================================================================= */

[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid #D1D5DB !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #10A37F !important;
    box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.15) !important;
}

/* =======================================================================
   8. FILTER CHIPS
   ======================================================================= */

.filter-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #E8F5E9;
    color: #2E7D32;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
}

.filter-chip-icon {
    cursor: pointer;
    opacity: 0.7;
}

.filter-chip-icon:hover {
    opacity: 1;
}

/* =======================================================================
   9. EMPTY STATE
   ======================================================================= */

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    padding: 2rem;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.8;
}

.empty-state-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
}

.empty-state-subtitle {
    font-size: 0.95rem;
    color: #9CA3AF;
    margin-bottom: 2rem;
}

/* Suggestion cards grid */
.suggestions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    max-width: 500px;
    width: 100%;
}

.suggestion-card {
    background-color: #F7F7F8;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    transition: background-color 0.15s ease, border-color 0.15s ease;
    text-align: left;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.4;
}

.suggestion-card:hover {
    background-color: #ECECEC;
    border-color: #D1D5DB;
}

/* =======================================================================
   10. SOURCE CITATIONS
   ======================================================================= */

.source-item {
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}

.source-meta {
    font-size: 0.75rem;
    color: #9CA3AF;
    margin-top: 4px;
}

/* =======================================================================
   11. TYPING INDICATOR
   ======================================================================= */

.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 8px 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background-color: #9CA3AF;
    border-radius: 50%;
    animation: typing-bounce 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing-bounce {
    0%, 80%, 100% {
        transform: scale(0.6);
        opacity: 0.4;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}

/* =======================================================================
   12. HEALTH STATUS INDICATORS
   ======================================================================= */

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.status-dot-green { background-color: #10A37F; }
.status-dot-yellow { background-color: #F59E0B; }
.status-dot-red { background-color: #EF4444; }

/* =======================================================================
   13. SETTINGS PANEL (sidebar expander)
   ======================================================================= */

section[data-testid="stSidebar"] .stExpander {
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    background-color: #1E1E1E !important;
}

section[data-testid="stSidebar"] .stExpander summary {
    color: #ECECEC !important;
}

section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] {
    background-color: #1E1E1E !important;
}

/* =======================================================================
   14. SCROLLBAR STYLING
   ======================================================================= */

section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 6px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
    background: transparent;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background-color: #444;
    border-radius: 3px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
    background-color: #555;
}

/* =======================================================================
   15. EXPANDER (sources) IN MAIN AREA
   ======================================================================= */

.main .stExpander {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}

/* =======================================================================
   16. RTL SUPPORT
   ======================================================================= */

[data-testid="stChatMessage"] .stMarkdown p {
    direction: auto;
    unicode-bidi: plaintext;
}

section[data-testid="stSidebar"] .stButton > button {
    direction: auto;
    unicode-bidi: plaintext;
}

/* =======================================================================
   17. MISC OVERRIDES
   ======================================================================= */

/* Remove default Streamlit horizontal rule styling */
.main hr {
    margin: 8px 0 !important;
    border-color: #E5E7EB !important;
}

/* Smoother font rendering */
body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Download button in sidebar */
section[data-testid="stSidebar"] .stDownloadButton > button {
    background-color: transparent !important;
    color: #ECECEC !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background-color: #2A2A2A !important;
}

/* =======================================================================
   18. SUGGESTION BUTTONS (main area, empty state)
   ======================================================================= */

/* Target buttons in the main content area that are NOT in chat messages */
.main .stButton > button {
    background-color: #F7F7F8 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    color: #374151 !important;
    font-size: 0.85rem !important;
    padding: 14px 16px !important;
    text-align: left !important;
    line-height: 1.4 !important;
    min-height: 60px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

.main .stButton > button:hover {
    background-color: #ECECEC !important;
    border-color: #D1D5DB !important;
}

/* =======================================================================
   19. SIDEBAR COLLAPSE BUTTON
   ======================================================================= */

button[data-testid="stSidebarCollapseButton"] {
    color: #ECECEC !important;
}

button[data-testid="stSidebarCollapseButton"] svg {
    fill: #ECECEC !important;
    stroke: #ECECEC !important;
}

/* When sidebar is collapsed, the expand button should be dark */
.main button[data-testid="stSidebarCollapseButton"] {
    color: #374151 !important;
}

.main button[data-testid="stSidebarCollapseButton"] svg {
    fill: #374151 !important;
    stroke: #374151 !important;
}

/* =======================================================================
   20. CONVERSATION LIST ITEM SPACING
   ======================================================================= */

/* Tighter columns in sidebar conversation rows */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 2px !important;
    align-items: center !important;
}

/* Reduce vertical gap between sidebar elements */
section[data-testid="stSidebar"] .stElementContainer {
    margin-bottom: 0px !important;
}

/* =======================================================================
   21. CHAT MESSAGE IMPROVEMENTS
   ======================================================================= */

/* Slightly more contrast for user messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #F3F4F6 !important;
}

/* Better spacing between messages */
[data-testid="stChatMessage"] {
    margin-bottom: 2px !important;
}

/* Chat avatar sizing */
[data-testid="stChatMessage"] [data-testid*="chatAvatarIcon"] {
    width: 28px !important;
    height: 28px !important;
}

/* =======================================================================
   22. MAIN AREA CLEAR/FILTER BUTTONS — smaller, less prominent
   ======================================================================= */

/* Override for small action buttons in chat area header */
.main [data-testid="stHorizontalBlock"] .stButton > button {
    min-height: auto !important;
    padding: 6px 12px !important;
    font-size: 0.8rem !important;
}
"""
