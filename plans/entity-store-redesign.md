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
- Uses a master-detail side panel layout for browsing + viewing person details
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
    H --> I[People tab: master-detail side panel]
    H --> J[All Facts tab: filterable table]
    I --> K[Left: person card list]
    I --> L[Right: detail panel with grouped facts]
```

## Files to Modify

| File | Change |
|------|--------|
| `src/entity_db.py` | Add `delete_fact()`, `delete_alias()`, `update_fact()` DB functions |
| `src/app.py` | Add DELETE `/entities/<id>/facts/<key>`, DELETE `/entities/<id>/aliases/<alias_id>`, PUT `/entities/<id>/facts` endpoints |
| `ui-reflex/ui_reflex/api_client.py` | Add ~11 entity API functions (including delete/update fact, delete alias) |
| `ui-reflex/ui_reflex/state.py` | Add entity state vars + ~18 event handlers |
| `ui-reflex/ui_reflex/components/entities_page.py` | **New file** — full entity page component with inline edit/delete |
| `ui-reflex/ui_reflex/ui_reflex.py` | Replace iframe with native component |
| `ui-reflex/ui_reflex/components/__init__.py` | Export new component if needed |

**Backend changes needed** — delete fact/alias and update fact endpoints do not exist yet.

---

## Detailed Design

### 1. API Client (`api_client.py`)

Add these async functions matching existing Flask routes:

```python
# Entity endpoints — existing backend routes
async def fetch_entities(query: str | None = None) -> list[dict]
    # GET /entities?q=...

async def fetch_entity_stats() -> dict
    # GET /entities/stats

async def fetch_entity(person_id: int) -> dict
    # GET /entities/{person_id}

async def delete_entity(person_id: int) -> dict
    # DELETE /entities/{person_id}

async def add_entity_fact(person_id: int, key: str, value: str) -> dict
    # POST /entities/{person_id}/facts

async def add_entity_alias(person_id: int, alias: str) -> dict
    # POST /entities/{person_id}/aliases

async def seed_entities() -> dict
    # POST /entities/seed

async def fetch_all_facts(key: str | None = None) -> dict
    # GET /entities/facts/all?key=...

# Entity endpoints — NEW backend routes needed for inline edit/delete
async def update_entity_fact(person_id: int, key: str, value: str) -> dict
    # PUT /entities/{person_id}/facts  body: {key, value}
    # (reuses set_fact which is already upsert)

async def delete_entity_fact(person_id: int, fact_key: str) -> dict
    # DELETE /entities/{person_id}/facts/{fact_key}

async def delete_entity_alias(person_id: int, alias_id: int) -> dict
    # DELETE /entities/{person_id}/aliases/{alias_id}
```

### 2. State (`state.py`)

New state variables:

```python
# --- Entity store ---
entity_persons: list[dict[str, str]] = []
entity_stats: dict[str, str] = {}
entity_search: str = ""
entity_selected_id: int = 0
entity_detail: dict[str, Any] = {}
entity_tab: str = "people"
entity_loading: bool = False
entity_detail_loading: bool = False
entity_save_message: str = ""

# For add fact/alias forms
entity_new_fact_key: str = ""
entity_new_fact_value: str = ""
entity_new_alias: str = ""

# All facts view
entity_all_facts: list[dict[str, str]] = []
entity_fact_keys: list[str] = []
entity_fact_key_filter: str = ""
```

New event handlers:

```python
async def on_entities_load()         # Page load
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
    # Groups facts into semantic categories:
    # Identity, Location, Work, Family, Contact, Other

@rx.var
def entity_aliases_list(self) -> list[dict]
    # Aliases from entity_detail formatted for rendering

@rx.var
def entity_relationships_list(self) -> list[dict]
    # Relationships from entity_detail
```

### 3. Entity Page Component (`entities_page.py`)

#### Master-Detail Side Panel Layout

When a person is selected, the People tab splits into a **two-column layout**:
the person list compresses to the left (~40%), and a detail panel appears on the
right (~60%). When no person is selected, the person grid takes the full width.

**No person selected — full-width grid:**

```
+--------------------------------------------------------------+
| <- Entity Store          Stats: 5154 P  210 F  14 R          |
+--------------------------------------------------------------+
|  P People  |  F All Facts                                    |
+--------------------------------------------------------------+
|  [Search persons...]  [Seed from WhatsApp] [Refresh]         |
|                                                               |
|  +--------------+ +--------------+ +--------------+          |
|  | Person 1      | | Person 2      | | Person 3      |       |
|  | aliases       | | aliases       | | aliases       |       |
|  | F 3  A 4      | | F 5  A 2      | | F 0  A 3      |       |
|  +--------------+ +--------------+ +--------------+          |
|  +--------------+ +--------------+ +--------------+          |
|  | Person 4      | | Person 5      | | Person 6      |       |
|  +--------------+ +--------------+ +--------------+          |
+--------------------------------------------------------------+
```

**Person selected — master-detail side panel:**

```
+--------------------------------------------------------------+
| <- Entity Store          Stats: 5154 P  210 F  14 R          |
+--------------------------------------------------------------+
|  P People  |  F All Facts                                    |
+--------------------------+-----------------------------------+
| [Search...]  [Seed] [R]  |  X Shiran Waintrob    [Delete]   |
|                           |                                   |
| +--------------------+   |  +- Identity ----------------+    |
| |> Shiran Waintrob   |   |  | Gender: female            |    |
| |  shiran / Shiran   |   |  | Birthday: 1994-03-15      |    |
| |  F 5  A 4          |   |  +----------------------------+    |
| +--------------------+   |                                   |
| +--------------------+   |  +- Work --------------------+    |
| |  Eden Peretz       |   |  | Job: Product Manager      |    |
| |  F 0  A 3          |   |  | Employer: Wix             |    |
| +--------------------+   |  +----------------------------+    |
| +--------------------+   |                                   |
| |  Michal            |   |  +- Aliases -----------------+    |
| |  F 0  A 3          |   |  | [Shiran x] [shiran x]    |    |
| +--------------------+   |  | [Add alias...] [+ Add]    |    |
| ...                       |  +----------------------------+    |
|                           |                                   |
|                           |  +- Relationships ----------+    |
|                           |  | spouse -> David Pickel    |    |
|                           |  +----------------------------+    |
+--------------------------+-----------------------------------+
```

#### Component Structure

```python
def entities_page() -> rx.Component:
    # Mirrors settings_page() outer structure
    rx.box(
        rx.flex(
            _header()               # Back button + title + stat badges
            _status_message()       # Save/action confirmations
            rx.tabs.root(
                rx.tabs.list(
                    "People"        # value="people"
                    "All Facts"     # value="facts"
                )
                rx.tabs.content(_people_tab())
                rx.tabs.content(_facts_tab())
            )
        )
    )

def _people_tab() -> rx.Component:
    # Toolbar: search + seed + refresh
    # Conditional two-column layout:
    rx.cond(
        AppState.entity_selected_id > 0,
        # Two columns: narrow person list + detail panel
        rx.flex(
            _person_list_column(),   # ~40% width, scrollable
            _person_detail_panel(),  # ~60% width, scrollable
        ),
        # Full width: person grid
        _person_grid_full(),
    )
```

The **detail panel** uses a vertical card layout matching settings `_section_card()`
pattern. It has a sticky header with the person name, close button, and delete.
The panel content scrolls independently of the person list.

#### Fact Category Grouping

Facts are grouped into semantic categories for intuitive reading:

| Category | Icon | Fact Keys |
|----------|------|-----------|
| Identity | tag | gender, birth_date, id_number |
| Location | map-pin | city, address, country |
| Work | briefcase | job_title, employer, industry |
| Family | users | marital_status + relationships |
| Contact | mail | email, phone |
| Business | building | is_business |
| Other | file-text | Everything else not in above categories |

Each fact displays:
- **Human-readable label** (birth_date -> Birthday, job_title -> Job Title)
- **Value** in readable format
- **Confidence badge** (color-coded: green >0.7, yellow 0.4-0.7, red <0.4)
- **Source icon** (message-circle for WhatsApp, file-text for Paperless, pencil for Manual)

#### Person Card (in grid)

```
+-------------------------------+
|  Shiran Waintrob              |
|  [shiran] [Shiran] [+2 more] |
|  F 5 facts  A 4 aliases      |
|  phone: +972501234567         |
+-------------------------------+
```

Uses same card styling as settings `_section_card()` with hover effect.

#### Person Detail Side Panel

When a person card is clicked, the page transitions from full-width grid to a
**two-column master-detail layout**. The person list compresses to ~40% width
(single column of cards), and a detail panel slides in at ~60% width on the right.

Clicking the X close button or pressing Escape collapses back to full-width.

The detail panel contains stacked `_section_card()` blocks:
- **Header** — Person name, WhatsApp ID, phone, close/delete buttons
- **Fact category cards** — Grouped facts with inline edit/delete (see below)
- **Aliases card** — Removable bubbles (X to delete) + add input
- **Relationships card** — Visual connection badges
- **Add Fact form** — Key + value inputs at the bottom

#### Inline Fact Edit and Delete

Each fact row in the detail panel supports edit and delete:

**Normal mode** (default):
```
| Birthday     1994-03-15     [green dot] [WA icon]  [pencil] [trash] |
```

**Edit mode** (after clicking pencil):
```
| Birthday     [1994-03-15____]                      [save]   [cancel] |
```

State tracking for inline edit:
- `entity_editing_fact_key: str = ""` — which fact key is being edited (empty = none)
- `entity_editing_fact_value: str = ""` — the new value being typed

When pencil is clicked: enter edit mode, pre-fill current value.
When save is clicked: call `update_entity_fact()` (reuses existing upsert POST endpoint).
When trash is clicked: confirm, then call `delete_entity_fact()` (new DELETE endpoint).

Each alias bubble has an X button that calls `delete_entity_alias()` (new DELETE endpoint).

### 0. Backend Changes (entity_db.py + app.py)

These DB functions and Flask endpoints need to be added before frontend work:

**entity_db.py new functions:**

```python
def delete_fact(person_id: int, fact_key: str) -> bool:
    """Delete a single fact by person_id and fact_key."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM person_facts WHERE person_id = ? AND fact_key = ?",
            (person_id, fact_key),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def delete_alias(alias_id: int) -> bool:
    """Delete a single alias by its ID."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM person_aliases WHERE id = ?",
            (alias_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
```

**app.py new endpoints:**

```python
@app.route("/entities/<int:person_id>/facts/<fact_key>", methods=["DELETE"])
def delete_entity_fact(person_id, fact_key):
    deleted = entity_db.delete_fact(person_id, fact_key)
    if not deleted:
        return jsonify({"error": "Fact not found"}), 404
    return jsonify({"status": "ok"}), 200

@app.route("/entities/<int:person_id>/aliases/<int:alias_id>", methods=["DELETE"])
def delete_entity_alias_by_id(person_id, alias_id):
    deleted = entity_db.delete_alias(alias_id)
    if not deleted:
        return jsonify({"error": "Alias not found"}), 404
    return jsonify({"status": "ok"}), 200
```

Note: Fact update (edit) reuses the existing `POST /entities/<id>/facts` endpoint,
which already does upsert via `entity_db.set_fact()`.

### 4. Route Update (`ui_reflex.py`)

```python
# Replace iframe-based entities() with:
def entities() -> rx.Component:
    return layout(entities_page())

# Update page registration:
app.add_page(
    entities,
    route="/entities",
    title="Entities - RAG Assistant",
    on_load=AppState.on_entities_load,
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

| Aspect | Current iframe | New Reflex |
|--------|---------------|------------|
| Theme | Dark #0f0f23 | Light white, matching settings |
| Framework | Vanilla JS + HTML | Reflex components |
| Layout | Full-page iframe | Settings-style scrollable panel |
| Detail view | Expands below grid | Side panel right of list |
| Navigation | Separate page feel | Integrated sidebar + back button |
| Person list | Dark grid cards | Light cards with settings-card styling |
| Fact display | Raw table | Grouped category cards with labels |
| Aliases | Plain inline spans | Colored bubbles like paperless tags |
| Actions | Inline JS handlers | Reflex event handlers with state |

## Implementation Order

1. Backend: Add `delete_fact()` and `delete_alias()` to entity_db.py
2. Backend: Add DELETE endpoints for facts and aliases to app.py
3. Frontend: Add entity API client functions to api_client.py
4. Frontend: Add entity state vars + event handlers to state.py
5. Frontend: Create entities_page.py component with side panel + inline edit/delete
6. Frontend: Update ui_reflex.py route to use native component
7. Test in browser
