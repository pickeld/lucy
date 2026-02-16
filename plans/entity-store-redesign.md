# Entity Store Redesign — Native Reflex + Settings-Style UI

## Problem

The current Entity Store page at `/entities` is an **iframe** embedding a standalone dark-themed HTML page served by Flask (`/entities/ui`). This creates several issues:

1. **Visual mismatch** — Dark theme clashes with the light Reflex UI
2. **Separate page behavior** — The iframe doesn't share state with the Reflex app
3. **Poor fact visualization** — Raw table layout isn't intuitive for understanding a person's profile
4. **No integration** — Can't leverage Reflex's reactive state or navigation patterns

## Goal

Rebuild the entity store as a **native Reflex component** that:
- Matches the settings page's design language (light theme, card-based, tabbed)
- Behaves like settings (Reflex component, not a separate page)
- Presents person facts in an intuitive, profile-card style layout
- Groups facts semantically so you can quickly understand who someone is

## Architecture

```mermaid
graph TD
    A[/entities route] --> B[entities_page component]
    B --> C[AppState entity vars + handlers]
    C --> D[api_client entity functions]
    D --> E[Flask /entities/* API]
    E --> F[entity_db.py SQLite]

    B --> G[Header: back + title + stats badges]
    B --> H[Tabs: People | All Facts]
    H --> I[People tab: search + person cards]
    H --> J[All Facts tab: filterable table]
    I --> K[Person detail: profile + facts + aliases + relationships]
```

## Files to Modify

| File | Change |
|------|--------|
| `ui-reflex/ui_reflex/api_client.py` | Add 8 entity API functions |
| `ui-reflex/ui_reflex/state.py` | Add entity state vars + ~12 event handlers |
| `ui-reflex/ui_reflex/components/entities_page.py` | **New file** — full entity page component |
| `ui-reflex/ui_reflex/ui_reflex.py` | Replace iframe with native component |
| `ui-reflex/ui_reflex/components/__init__.py` | Export new component if needed |

No backend changes needed — all Flask `/entities/*` endpoints already exist.

---

## Detailed Design

### 1. API Client (`api_client.py`)

Add these async functions matching existing Flask routes:

```python
# Entity endpoints
async def fetch_entities(query: str | None = None) -> list[dict]
    # GET /entities?q=...
    # Returns: {"persons": [...], "count": N}

async def fetch_entity_stats() -> dict
    # GET /entities/stats
    # Returns: {"persons": N, "aliases": N, "facts": N, "relationships": N}

async def fetch_entity(person_id: int) -> dict
    # GET /entities/{person_id}
    # Returns: full person with aliases, facts_detail, relationships

async def delete_entity(person_id: int) -> dict
    # DELETE /entities/{person_id}

async def add_entity_fact(person_id: int, key: str, value: str) -> dict
    # POST /entities/{person_id}/facts  body: {key, value}

async def add_entity_alias(person_id: int, alias: str) -> dict
    # POST /entities/{person_id}/aliases  body: {alias}

async def seed_entities() -> dict
    # POST /entities/seed  body: {confirm: true}

async def fetch_all_facts(key: str | None = None) -> dict
    # GET /entities/facts/all?key=...
    # Returns: {"facts": [...], "available_keys": [...]}
```

### 2. State (`state.py`)

New state variables:

```python
# --- Entity store ---
entity_persons: list[dict[str, str]] = []      # Person list for grid
entity_stats: dict[str, str] = {}               # Stats counters
entity_search: str = ""                          # Search input
entity_selected_id: int = 0                      # Currently selected person ID
entity_detail: dict[str, Any] = {}               # Full person detail
entity_tab: str = "people"                       # Active tab
entity_loading: bool = False                     # Loading spinner
entity_detail_loading: bool = False              # Detail loading
entity_save_message: str = ""                    # Toast/status message

# For add fact/alias forms
entity_new_fact_key: str = ""
entity_new_fact_value: str = ""
entity_new_alias: str = ""

# All facts view
entity_all_facts: list[dict[str, str]] = []
entity_fact_keys: list[str] = []                 # Available keys for filter
entity_fact_key_filter: str = ""                 # Selected key filter
```

New event handlers:

```python
async def on_entities_load()         # Page load — fetch persons + stats
async def load_entities(query?)      # Fetch/search person list
async def load_entity_stats()        # Fetch stats
async def select_entity(person_id)   # Load full person detail
async def close_entity_detail()      # Clear selection
async def delete_entity(person_id)   # Delete + refresh
async def add_entity_fact()          # Add fact from form
async def add_entity_alias()         # Add alias from form
async def seed_entities()            # Trigger WhatsApp seed
async def load_all_facts()           # Fetch all facts table
def set_entity_search(value)         # Search input setter
def set_entity_tab(value)            # Tab setter
def set_entity_fact_key_filter(v)    # Fact key filter setter
def set_entity_new_fact_key(v)       # Form field setters
def set_entity_new_fact_value(v)
def set_entity_new_alias(v)
```

Key computed vars:

```python
@rx.var
def entity_facts_grouped(self) -> list[dict]
    # Groups facts from entity_detail into semantic categories:
    # Identity, Location, Work, Family, Contact, Other
    # Each group: {category, icon, facts: [{key, value, confidence, source}]}

@rx.var
def entity_aliases_list(self) -> list[dict]
    # Aliases from entity_detail formatted for rendering

@rx.var
def entity_relationships_list(self) -> list[dict]
    # Relationships from entity_detail
```

### 3. Entity Page Component (`entities_page.py`)

Layout matches settings page exactly:

```
┌─────────────────────────────────────────────┐
│ ← Entity Store     Stats: 5154 👤 210 📋 14 🔗│
├─────────────────────────────────────────────┤
│  👤 People  │  📋 All Facts                  │
├─────────────────────────────────────────────┤
│                                              │
│  [Search persons...]  [🌱 Seed] [🔄 Refresh]│
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Person 1 │ │ Person 2 │ │ Person 3 │      │
│  │ aliases  │ │ aliases  │ │ aliases  │      │
│  │ 3 facts  │ │ 5 facts  │ │ 0 facts  │      │
│  └─────────┘ └─────────┘ └─────────┘       │
│                                              │
│  ═══════════════════════════════════════════ │
│  PERSON DETAIL (when selected)               │
│                                              │
│  ┌─ 🏷️ Identity ──────────────────────────┐ │
│  │ Gender: female    Born: 1994-03-15      │ │
│  │ ID: 038041612                           │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─ 💼 Work ──────────────────────────────┐  │
│  │ Title: Product Manager                  │ │
│  │ Employer: Wix                           │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─ 📝 Aliases ───────────────────────────┐  │
│  │ [Shiran ×] [שירן ×] [Shiran W ×]       │ │
│  │ [Add alias...]  [+ Add]                 │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─ 🔗 Relationships ────────────────────┐  │
│  │ spouse → David Pickel                   │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

#### Component Structure

```python
def entities_page() -> rx.Component:
    # Mirrors settings_page() structure exactly
    rx.box(
        rx.flex(
            _header()               # Back button + title + stat badges
            _status_message()       # Save/action confirmations
            rx.tabs.root(
                rx.tabs.list(
                    "👤 People"     # value="people"
                    "📋 All Facts"  # value="facts"
                )
                rx.tabs.content(_people_tab())
                rx.tabs.content(_facts_tab())
            )
        )
    )
```

#### Fact Category Grouping

Facts are grouped into semantic categories for intuitive reading:

| Category | Icon | Fact Keys |
|----------|------|-----------|
| Identity | 🏷️ | gender, birth_date, id_number |
| Location | 🏠 | city, address, country |
| Work | 💼 | job_title, employer, industry |
| Family | 👨‍👩‍👧 | marital_status + relationships |
| Contact | 📧 | email, phone |
| Business | 🏢 | is_business |
| Other | 📝 | Everything else not in above categories |

Each fact displays:
- **Human-readable label** (birth_date → "Birthday", job_title → "Job Title")
- **Value** in readable format
- **Confidence badge** (color-coded: green >0.7, yellow 0.4-0.7, red <0.4)
- **Source icon** (💬 WhatsApp, 📄 Paperless, ✏️ Manual)

#### Person Card (in grid)

```
┌───────────────────────────────┐
│  Shiran Waintrob              │
│  [שירן] [Shiran] [+2 more]   │
│  📋 5 facts  📝 4 aliases     │
│  📱 +972501234567             │
└───────────────────────────────┘
```

Uses same card styling as settings `_section_card()` with hover effect.

#### Person Detail Panel

When a person card is clicked, a detail panel expands below the grid
(similar to how the current HTML detail-panel works, but styled as
settings cards).

Each fact category is a `_section_card()`:
- Facts rendered as key-value pairs with labels, not raw table rows
- Confidence shown as a small colored dot
- Source shown as an icon
- Inline add fact/alias forms at the bottom of their sections

### 4. Route Update (`ui_reflex.py`)

```python
# Replace iframe-based entities() with:
def entities() -> rx.Component:
    return layout(entities_page())

# Update page registration:
app.add_page(
    entities,
    route="/entities",
    title="Entities — RAG Assistant",
    on_load=AppState.on_entities_load,  # New handler
)
```

### 5. Fact Label Mapping

Add to state.py (similar to SETTING_LABELS):

```python
FACT_LABELS: dict[str, str] = {
    "birth_date": "Birthday",
    "gender": "Gender",
    "city": "City",
    "job_title": "Job Title",
    "employer": "Employer",
    "marital_status": "Marital Status",
    "email": "Email",
    "id_number": "ID Number",
    "is_business": "Business Account",
    "age": "Age",
    "address": "Address",
    "country": "Country",
    "industry": "Industry",
    "phone": "Phone",
    "recent_topic": "Recent Topic",
    "recent_mood": "Recent Mood",
}
```

---

## Visual Comparison

| Aspect | Current (iframe) | New (Reflex) |
|--------|-----------------|-------------|
| Theme | Dark (#0f0f23) | Light (white, matching settings) |
| Framework | Vanilla JS + HTML | Reflex components |
| Layout | Full-page iframe | Settings-style scrollable panel |
| Navigation | Separate page feel | Integrated with sidebar, back button |
| Person list | Dark grid cards | Light cards with settings-card styling |
| Fact display | Raw table | Grouped category cards with labels |
| Aliases | Plain inline spans | Colored bubbles (like paperless tags) |
| Actions | Inline JS handlers | Reflex event handlers with state |
| Responsiveness | Basic grid | Consistent with rest of UI |

## Implementation Order

1. API client functions (no dependencies)
2. State vars + handlers (depends on API client)
3. Entity page component (depends on state)
4. Route wiring (depends on component)
5. Test in browser
