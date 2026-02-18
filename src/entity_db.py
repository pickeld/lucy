"""SQLite-backed entity store for person knowledge management.

Accumulates structured knowledge about people over time from WhatsApp
messages, Paperless documents, and other sources. Provides:

- Person records with canonical names and WhatsApp IDs
- Multi-script name aliases for cross-script disambiguation (שירן ↔ Shiran)
- Key-value facts (birth_date, city, job, etc.) with confidence scores
- Person-to-person relationships (friend, spouse, parent, etc.)
- Entity deduplication by phone (primary), email (secondary), name (tertiary)
- Entity merging: combine duplicate persons into one record
- Automatic Hebrew+English display name merging

Database location: data/settings.db (shared with settings_db, conversations_db)
"""

import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from settings_db import DB_PATH
from utils.logger import logger


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection (connection-per-request for thread safety).

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Script detection helper
# ---------------------------------------------------------------------------

_HEBREW_RE = re.compile(r'[\u0590-\u05FF]')
_LATIN_RE = re.compile(r'[a-zA-Z]')

# Patterns that indicate a garbage / non-person name
_GARBAGE_NAME_PATTERNS = [
    re.compile(r'^\W+$'),                    # Pure punctuation/symbols
    re.compile(r'^\d+$'),                     # Pure digits
    re.compile(r'^\(.*\)$'),                  # Wrapped in parens like ('')
    re.compile(r'^[\'"]+$'),                  # Just quotes
    re.compile(r'^\*\w{0,2}$'),              # Star-prefixed short codes like *K
    re.compile(r'^.{0,1}$'),                 # Single char or empty
    re.compile(r'^[\U0001F600-\U0001F9FF\s]+$'),  # Pure emoji
]


def _detect_script(text: str) -> str:
    """Detect the primary script of a text string.

    Returns:
        'hebrew', 'latin', or 'mixed'
    """
    has_hebrew = bool(_HEBREW_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_hebrew and has_latin:
        return "mixed"
    elif has_hebrew:
        return "hebrew"
    elif has_latin:
        return "latin"
    return "unknown"


def _is_valid_person_name(name: str) -> bool:
    """Check if a name is a valid person/contact name.

    Filters out garbage like punctuation-only, single chars, pure digits,
    star-prefix short codes, and other non-name strings that WhatsApp
    contacts sometimes have.

    Args:
        name: The candidate name string

    Returns:
        True if the name looks like a real person/group name
    """
    if not name or len(name.strip()) < 2:
        return False

    stripped = name.strip()

    for pattern in _GARBAGE_NAME_PATTERNS:
        if pattern.match(stripped):
            return False

    # Must contain at least one letter (any script)
    if not _HEBREW_RE.search(stripped) and not _LATIN_RE.search(stripped):
        # Allow CJK and other scripts too — just reject pure non-letter strings
        # Check for any Unicode letter category
        has_letter = any(c.isalpha() for c in stripped)
        if not has_letter:
            return False

    return True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_entity_db() -> None:
    """Create entity tables if they don't exist.

    Safe to call multiple times — uses IF NOT EXISTS.
    Also runs migrations (e.g., adding email column).
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                whatsapp_id TEXT,
                phone TEXT,
                email TEXT,
                is_group BOOLEAN DEFAULT FALSE,
                confidence REAL DEFAULT 0.5,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(canonical_name)
            )
        """)

        # Migration: add email column if missing (for existing DBs)
        try:
            conn.execute("SELECT email FROM persons LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE persons ADD COLUMN email TEXT")
            logger.info("Migration: added email column to persons table")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS person_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                script TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'auto',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                UNIQUE(person_id, alias)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS person_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source_type TEXT DEFAULT 'extracted',
                source_ref TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                UNIQUE(person_id, fact_key)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS person_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                related_person_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source_ref TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                FOREIGN KEY (related_person_id) REFERENCES persons(id) ON DELETE CASCADE,
                UNIQUE(person_id, related_person_id, relationship_type)
            )
        """)

        # Person-asset graph: junction table linking persons to their assets
        # (messages, documents, call recordings) in the vector store.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS person_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                asset_ref TEXT NOT NULL,
                role TEXT DEFAULT 'sender',
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                UNIQUE(person_id, asset_ref, role)
            )
        """)

        # Asset-asset graph: edges between assets for cross-channel coherence.
        # Relation types: thread_member, attachment_of, chunk_of, reply_to,
        # references, transcript_of.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_asset_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_asset_ref TEXT NOT NULL,
                dst_asset_ref TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                provenance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(src_asset_ref, dst_asset_ref, relation_type)
            )
        """)

        # Indexes for fast lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_whatsapp ON persons(whatsapp_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(canonical_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_phone ON persons(phone)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_email ON persons(email)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aliases_alias ON person_aliases(alias COLLATE NOCASE)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aliases_person ON person_aliases(person_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_person ON person_facts(person_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_key ON person_facts(fact_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_assets_person ON person_assets(person_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_assets_ref ON person_assets(asset_ref)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_assets_type ON person_assets(asset_type)"
        )
        # Asset-asset graph indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aae_src ON asset_asset_edges(src_asset_ref)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aae_dst ON asset_asset_edges(dst_asset_ref)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_aae_type ON asset_asset_edges(relation_type)"
        )

        conn.commit()
        logger.info("Entity database tables initialized")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Display name helpers — Hebrew + English merging
# ---------------------------------------------------------------------------

def _build_display_name(conn: sqlite3.Connection, person_id: int) -> Optional[str]:
    """Build a bilingual display name from a person's aliases.

    If a person has both Hebrew and Latin aliases, combine them:
        "Shiran Waintrob / שירן ויינטרוב"

    Only returns a new display name if both scripts are present
    and the canonical name doesn't already contain both scripts.

    Args:
        conn: Active SQLite connection
        person_id: Person to build display name for

    Returns:
        New display name string, or None if no change needed
    """
    row = conn.execute(
        "SELECT canonical_name FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    if not row:
        return None

    current_name = row["canonical_name"]
    current_script = _detect_script(current_name)

    # Already bilingual — no change needed
    if current_script == "mixed":
        return None

    # Gather aliases by script
    alias_rows = conn.execute(
        "SELECT alias, script FROM person_aliases WHERE person_id = ?",
        (person_id,),
    ).fetchall()

    hebrew_names: list[str] = []
    latin_names: list[str] = []

    for a in alias_rows:
        alias_text = a["alias"]
        alias_script = a["script"]
        # Skip numeric/phone aliases
        if alias_text.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            continue
        if alias_script == "hebrew":
            hebrew_names.append(alias_text)
        elif alias_script == "latin":
            latin_names.append(alias_text)

    if not hebrew_names or not latin_names:
        return None  # Need both scripts

    # Pick the longest name from each script (likely the full name)
    best_hebrew = max(hebrew_names, key=len)
    best_latin = max(latin_names, key=len)

    # Build: "Latin Name / Hebrew Name"
    return f"{best_latin} / {best_hebrew}"


def update_display_name(person_id: int) -> Optional[str]:
    """Recalculate and update the display name for a person.

    Checks if the person has aliases in both Hebrew and Latin scripts
    and updates canonical_name to show both (e.g., "Shiran / שירן").

    Args:
        person_id: The person to update

    Returns:
        New display name if updated, None if no change
    """
    conn = _get_connection()
    try:
        new_name = _build_display_name(conn, person_id)
        if new_name:
            # Check that no other person already has this exact name
            existing = conn.execute(
                "SELECT id FROM persons WHERE canonical_name = ? AND id != ?",
                (new_name, person_id),
            ).fetchone()
            if not existing:
                conn.execute(
                    "UPDATE persons SET canonical_name = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_name, person_id),
                )
                conn.commit()
                logger.info(f"Updated display name for person {person_id}: {new_name}")
                return new_name
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Person CRUD — identifier-based dedup (phone → email → name)
# ---------------------------------------------------------------------------

def _normalize_phone(phone: str) -> str:
    """Normalize a phone number for comparison.

    Strips whitespace, dashes, parens, and leading + or 0.
    E.g., "+972-50-123-4567" → "97250123456"
    """
    if not phone:
        return ""
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    # Strip leading zeros (e.g., 0501234567 → 501234567)
    cleaned = cleaned.lstrip('0')
    return cleaned


def find_person_by_phone(phone: str) -> Optional[int]:
    """Find a person ID by phone number (normalized comparison).

    Args:
        phone: Phone number to search for

    Returns:
        Person ID, or None
    """
    if not phone:
        return None

    normalized = _normalize_phone(phone)
    if not normalized:
        return None

    conn = _get_connection()
    try:
        # Check persons.phone column
        rows = conn.execute("SELECT id, phone FROM persons WHERE phone IS NOT NULL").fetchall()
        for r in rows:
            if _normalize_phone(r["phone"]) == normalized:
                return r["id"]

        # Check aliases with numeric script (phone aliases)
        alias_rows = conn.execute(
            "SELECT person_id, alias FROM person_aliases WHERE script = 'numeric'"
        ).fetchall()
        for a in alias_rows:
            if _normalize_phone(a["alias"]) == normalized:
                return a["person_id"]

        return None
    finally:
        conn.close()


def find_person_by_email(email: str) -> Optional[int]:
    """Find a person ID by email address (case-insensitive).

    Searches both persons.email column and the 'email' fact.

    Args:
        email: Email address to search for

    Returns:
        Person ID, or None
    """
    if not email:
        return None

    email_lower = email.strip().lower()
    conn = _get_connection()
    try:
        # Check persons.email column
        row = conn.execute(
            "SELECT id FROM persons WHERE LOWER(email) = ?",
            (email_lower,),
        ).fetchone()
        if row:
            return row["id"]

        # Check person_facts for email fact
        fact_row = conn.execute(
            "SELECT person_id FROM person_facts WHERE fact_key = 'email' AND LOWER(fact_value) = ?",
            (email_lower,),
        ).fetchone()
        if fact_row:
            return fact_row["person_id"]

        return None
    finally:
        conn.close()


def get_or_create_person(
    canonical_name: str,
    whatsapp_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    is_group: bool = False,
) -> int:
    """Get a person ID using identifier cascade, or create a new record.

    Lookup priority:
    1. Phone number (primary unique ID)
    2. Email address (secondary unique ID)
    3. Canonical name (tertiary, legacy)

    If found by any identifier, updates other fields if currently NULL
    and updates last_seen. Also auto-creates aliases and attempts to
    build a bilingual display name (Hebrew + English).

    Args:
        canonical_name: Primary display name (e.g., "Shiran Waintrob")
        whatsapp_id: WhatsApp ID (e.g., "972501234567@c.us")
        phone: Phone number (e.g., "+972501234567")
        email: Email address
        is_group: Whether this is a group entity

    Returns:
        Person ID (integer)
    """
    # Guard: if whatsapp_id is a Linked ID (@lid), the phone param
    # may actually be the LID digits — not a real phone number.
    # Discard it to avoid storing garbage in the phone column.
    if phone and whatsapp_id and whatsapp_id.endswith("@lid"):
        lid_digits = whatsapp_id.replace("@lid", "")
        if phone.lstrip("+") == lid_digits:
            phone = None

    conn = _get_connection()
    try:
        person_id: Optional[int] = None

        # 1. Try phone (primary identifier)
        if phone and not is_group:
            pid = find_person_by_phone(phone)
            if pid is not None:
                person_id = pid

        # 2. Try email (secondary identifier)
        if person_id is None and email and not is_group:
            pid = find_person_by_email(email)
            if pid is not None:
                person_id = pid

        # 3. Try canonical name (tertiary / legacy)
        if person_id is None:
            row = conn.execute(
                "SELECT id FROM persons WHERE canonical_name = ?",
                (canonical_name,),
            ).fetchone()
            if row:
                person_id = row["id"]

        if person_id is not None:
            # Update fields if they're provided and currently NULL
            updates = []
            params: list = []
            if whatsapp_id:
                updates.append("whatsapp_id = COALESCE(whatsapp_id, ?)")
                params.append(whatsapp_id)
            if phone:
                updates.append("phone = COALESCE(phone, ?)")
                params.append(phone)
            if email:
                updates.append("email = COALESCE(email, ?)")
                params.append(email)
            updates.append("last_seen = CURRENT_TIMESTAMP")
            params.append(person_id)
            conn.execute(
                f"UPDATE persons SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

            # Add the incoming name as alias if it's different from canonical
            existing_name = conn.execute(
                "SELECT canonical_name FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            if existing_name and existing_name["canonical_name"] != canonical_name:
                _safe_add_alias(conn, person_id, canonical_name)
                conn.commit()

            # Try to build bilingual display name
            new_display = _build_display_name(conn, person_id)
            if new_display:
                dup = conn.execute(
                    "SELECT id FROM persons WHERE canonical_name = ? AND id != ?",
                    (new_display, person_id),
                ).fetchone()
                if not dup:
                    conn.execute(
                        "UPDATE persons SET canonical_name = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                        (new_display, person_id),
                    )
                    conn.commit()
        else:
            # Create new person
            cursor = conn.execute(
                """INSERT INTO persons (canonical_name, whatsapp_id, phone, email, is_group)
                   VALUES (?, ?, ?, ?, ?)""",
                (canonical_name, whatsapp_id, phone, email, is_group),
            )
            person_id = cursor.lastrowid
            conn.commit()

            # Auto-create aliases from canonical name
            if person_id is not None:
                _auto_create_aliases(conn, person_id, canonical_name)
                conn.commit()

        return person_id  # type: ignore[return-value]
    finally:
        conn.close()


def _safe_add_alias(conn: sqlite3.Connection, person_id: int, alias: str) -> None:
    """Add an alias within an existing connection, ignoring duplicates."""
    script = _detect_script(alias)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO person_aliases (person_id, alias, script, source)
               VALUES (?, ?, ?, ?)""",
            (person_id, alias, script, "auto"),
        )
    except sqlite3.IntegrityError:
        pass


def _auto_create_aliases(
    conn: sqlite3.Connection, person_id: int, canonical_name: str
) -> None:
    """Auto-create aliases from a canonical name.

    Creates aliases for:
    - The full canonical name
    - The first name (first word)
    - Detects script (Hebrew/Latin) for each

    Args:
        conn: Active SQLite connection
        person_id: The person to add aliases for
        canonical_name: Full name to derive aliases from
    """
    parts = canonical_name.strip().split()
    if not parts:
        return

    aliases_to_add = set()
    # Full name
    aliases_to_add.add(canonical_name.strip())
    # First name
    if parts[0]:
        aliases_to_add.add(parts[0])

    for alias in aliases_to_add:
        script = _detect_script(alias)
        try:
            conn.execute(
                """INSERT OR IGNORE INTO person_aliases (person_id, alias, script, source)
                   VALUES (?, ?, ?, ?)""",
                (person_id, alias, script, "auto"),
            )
        except sqlite3.IntegrityError:
            pass  # Alias already exists


def get_person(person_id: int) -> Optional[Dict[str, Any]]:
    """Get a person with all facts and aliases.

    Also computes a display_name that merges Hebrew + English aliases
    alongside the canonical_name.

    Args:
        person_id: The person's database ID

    Returns:
        Dict with person data, facts, aliases, relationships, and display_name
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        if not row:
            return None

        person = dict(row)

        # Fetch aliases (include id for delete support)
        alias_rows = conn.execute(
            "SELECT id, alias, script, source FROM person_aliases WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        person["aliases"] = [dict(a) for a in alias_rows]

        # Build bilingual display name from aliases
        person["display_name"] = _compute_display_name(person["canonical_name"], person["aliases"])

        # Fetch facts
        fact_rows = conn.execute(
            "SELECT fact_key, fact_value, confidence, source_type, source_ref, extracted_at "
            "FROM person_facts WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        person["facts"] = {f["fact_key"]: f["fact_value"] for f in fact_rows}
        person["facts_detail"] = [dict(f) for f in fact_rows]

        # Fetch relationships
        rel_rows = conn.execute(
            """SELECT r.relationship_type, r.confidence, p.canonical_name AS related_name
               FROM person_relationships r
               JOIN persons p ON p.id = r.related_person_id
               WHERE r.person_id = ?""",
            (person_id,),
        ).fetchall()
        person["relationships"] = [dict(r) for r in rel_rows]

        # Fetch asset counts by type
        asset_rows = conn.execute(
            """SELECT asset_type, COUNT(*) as cnt
               FROM person_assets WHERE person_id = ?
               GROUP BY asset_type""",
            (person_id,),
        ).fetchall()
        person["asset_counts"] = {r["asset_type"]: r["cnt"] for r in asset_rows}

        return person
    finally:
        conn.close()


def _compute_display_name(canonical_name: str, aliases: List[Dict[str, Any]]) -> str:
    """Compute a bilingual display name from canonical name + aliases.

    If the person has both Hebrew and Latin aliases, returns:
        "English Name / Hebrew Name"
    Otherwise returns the canonical name as-is.

    This is a read-only computation — does NOT update the DB.
    """
    current_script = _detect_script(canonical_name)
    if current_script == "mixed":
        return canonical_name  # Already bilingual

    hebrew_names: list[str] = []
    latin_names: list[str] = []

    for a in aliases:
        alias_text = a.get("alias", "")
        alias_script = a.get("script", "")
        # Skip numeric/phone aliases
        if alias_text.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            continue
        if alias_script == "hebrew":
            hebrew_names.append(alias_text)
        elif alias_script == "latin":
            latin_names.append(alias_text)

    if not hebrew_names or not latin_names:
        return canonical_name  # Can't make bilingual

    best_hebrew = max(hebrew_names, key=len)
    best_latin = max(latin_names, key=len)

    return f"{best_latin} / {best_hebrew}"


def get_person_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Look up a person by canonical_name or any alias.

    Args:
        name: Name to search for (case-insensitive)

    Returns:
        Full person dict (same as get_person), or None
    """
    conn = _get_connection()
    try:
        # First try canonical_name (exact, case-insensitive)
        row = conn.execute(
            "SELECT id FROM persons WHERE canonical_name = ? COLLATE NOCASE",
            (name,),
        ).fetchone()

        if not row:
            # Try aliases
            alias_row = conn.execute(
                "SELECT person_id FROM person_aliases WHERE alias = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            if alias_row:
                row = alias_row

        if row:
            person_id = row["person_id"] if "person_id" in row.keys() else row["id"]
            return get_person(person_id)

        return None
    finally:
        conn.close()


def get_person_by_whatsapp_id(whatsapp_id: str) -> Optional[Dict[str, Any]]:
    """Look up a person by WhatsApp ID.

    Args:
        whatsapp_id: WhatsApp contact ID (e.g., "972501234567@c.us")

    Returns:
        Full person dict, or None
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM persons WHERE whatsapp_id = ?", (whatsapp_id,)
        ).fetchone()
        if row:
            return get_person(row["id"])
        return None
    finally:
        conn.close()


def delete_person(person_id: int) -> bool:
    """Delete a person and all associated data (cascades to aliases, facts, relationships).

    Args:
        person_id: The person to delete

    Returns:
        True if a person was deleted
    """
    conn = _get_connection()
    try:
        cursor = conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Alias management
# ---------------------------------------------------------------------------

def add_alias(
    person_id: int,
    alias: str,
    script: Optional[str] = None,
    source: str = "auto",
) -> bool:
    """Add a name alias to a person.

    Args:
        person_id: The person to add the alias for
        alias: The alias text (e.g., "שירן", "Shiran")
        script: Script type ("hebrew", "latin", "mixed") — auto-detected if None
        source: Where this alias came from ("whatsapp_contact", "extracted", "manual")

    Returns:
        True if the alias was added (False if it already existed)
    """
    if script is None:
        script = _detect_script(alias)

    conn = _get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO person_aliases (person_id, alias, script, source)
               VALUES (?, ?, ?, ?)""",
            (person_id, alias, script, source),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def resolve_name(name: str) -> List[Dict[str, Any]]:
    """Find all persons matching a name or alias.

    Used for disambiguation: given a first name, returns all
    persons who have that name as a canonical_name or alias.

    Args:
        name: Name to search for (case-insensitive)

    Returns:
        List of person dicts with id, canonical_name, and aliases
    """
    conn = _get_connection()
    try:
        # Find person IDs matching by canonical_name or alias
        person_ids = set()

        # Match canonical name
        rows = conn.execute(
            "SELECT id FROM persons WHERE canonical_name LIKE ? COLLATE NOCASE",
            (f"%{name}%",),
        ).fetchall()
        for r in rows:
            person_ids.add(r["id"])

        # Match aliases
        alias_rows = conn.execute(
            "SELECT person_id FROM person_aliases WHERE alias = ? COLLATE NOCASE",
            (name,),
        ).fetchall()
        for r in alias_rows:
            person_ids.add(r["person_id"])

        # Build results
        results = []
        for pid in person_ids:
            person_row = conn.execute(
                "SELECT id, canonical_name, whatsapp_id, phone FROM persons WHERE id = ?",
                (pid,),
            ).fetchone()
            if person_row:
                person = dict(person_row)
                alias_list = conn.execute(
                    "SELECT alias, script FROM person_aliases WHERE person_id = ?",
                    (pid,),
                ).fetchall()
                person["aliases"] = [dict(a) for a in alias_list]
                results.append(person)

        return results
    finally:
        conn.close()


def search_persons(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search persons by name or alias prefix (for autocomplete/search UI).

    Args:
        query: Search string (prefix match)
        limit: Max results

    Returns:
        List of person summary dicts
    """
    conn = _get_connection()
    try:
        person_ids = set()

        # Search canonical names
        rows = conn.execute(
            "SELECT id FROM persons WHERE canonical_name LIKE ? COLLATE NOCASE LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        for r in rows:
            person_ids.add(r["id"])

        # Search aliases
        alias_rows = conn.execute(
            "SELECT DISTINCT person_id FROM person_aliases "
            "WHERE alias LIKE ? COLLATE NOCASE LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        for r in alias_rows:
            person_ids.add(r["person_id"])

        results = []
        for pid in list(person_ids)[:limit]:
            person_row = conn.execute(
                "SELECT id, canonical_name, whatsapp_id, phone, last_seen FROM persons WHERE id = ?",
                (pid,),
            ).fetchone()
            if person_row:
                person = dict(person_row)
                # Count aliases and facts
                alias_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM person_aliases WHERE person_id = ?",
                    (pid,),
                ).fetchone()["cnt"]
                fact_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM person_facts WHERE person_id = ?",
                    (pid,),
                ).fetchone()["cnt"]
                person["alias_count"] = alias_count
                person["fact_count"] = fact_count
                results.append(person)

        return sorted(results, key=lambda p: p["canonical_name"])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fact management
# ---------------------------------------------------------------------------

def set_fact(
    person_id: int,
    key: str,
    value: str,
    confidence: float = 0.5,
    source_type: str = "extracted",
    source_ref: Optional[str] = None,
) -> None:
    """Upsert a fact for a person.

    If a fact with the same key already exists:
    - Higher confidence overwrites lower confidence
    - Equal confidence updates the value (newer wins)

    Args:
        person_id: The person this fact belongs to
        key: Fact key (e.g., "birth_date", "city", "job_title")
        value: Fact value (e.g., "1994-03-15", "Tel Aviv", "Product Manager")
        confidence: How confident we are in this fact (0.0-1.0)
        source_type: Where this fact came from ("whatsapp", "paperless", "manual", "inferred")
        source_ref: Reference to source (e.g., "chat:972501234567@c.us:1708012345")
    """
    conn = _get_connection()
    try:
        # Check existing fact
        existing = conn.execute(
            "SELECT confidence FROM person_facts WHERE person_id = ? AND fact_key = ?",
            (person_id, key),
        ).fetchone()

        if existing:
            # Only overwrite if new confidence >= existing
            if confidence >= existing["confidence"]:
                conn.execute(
                    """UPDATE person_facts
                       SET fact_value = ?, confidence = ?, source_type = ?,
                           source_ref = ?, extracted_at = CURRENT_TIMESTAMP
                       WHERE person_id = ? AND fact_key = ?""",
                    (value, confidence, source_type, source_ref, person_id, key),
                )
        else:
            conn.execute(
                """INSERT INTO person_facts
                   (person_id, fact_key, fact_value, confidence, source_type, source_ref)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, key, value, confidence, source_type, source_ref),
            )

        # Update person's last_updated timestamp
        conn.execute(
            "UPDATE persons SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
            (person_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_fact(person_id: int, key: str) -> Optional[str]:
    """Get a single fact value for a person.

    Args:
        person_id: The person
        key: Fact key

    Returns:
        Fact value string, or None
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT fact_value FROM person_facts WHERE person_id = ? AND fact_key = ?",
            (person_id, key),
        ).fetchone()
        return row["fact_value"] if row else None
    finally:
        conn.close()


def get_all_facts(person_id: int) -> Dict[str, str]:
    """Get all facts for a person as a key→value dict.

    Args:
        person_id: The person

    Returns:
        Dict of fact_key → fact_value
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT fact_key, fact_value FROM person_facts WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        return {r["fact_key"]: r["fact_value"] for r in rows}
    finally:
        conn.close()


def delete_fact(person_id: int, fact_key: str) -> bool:
    """Delete a single fact for a person by key.

    Args:
        person_id: The person
        fact_key: The fact key to delete

    Returns:
        True if the fact was deleted
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM person_facts WHERE person_id = ? AND fact_key = ?",
            (person_id, fact_key),
        )
        if cursor.rowcount > 0:
            conn.execute(
                "UPDATE persons SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (person_id,),
            )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_alias(alias_id: int) -> bool:
    """Delete a single alias by its row ID.

    Args:
        alias_id: The alias row ID

    Returns:
        True if the alias was deleted
    """
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


# ---------------------------------------------------------------------------
# Relationship management
# ---------------------------------------------------------------------------

def add_relationship(
    person_id: int,
    related_person_id: int,
    relationship_type: str,
    confidence: float = 0.5,
    source_ref: Optional[str] = None,
) -> bool:
    """Add a relationship between two persons.

    Args:
        person_id: Source person
        related_person_id: Related person
        relationship_type: Type (e.g., "friend", "spouse", "parent", "child", "colleague")
        confidence: Confidence score (0-1)
        source_ref: Source reference

    Returns:
        True if relationship was added
    """
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO person_relationships
               (person_id, related_person_id, relationship_type, confidence, source_ref)
               VALUES (?, ?, ?, ?, ?)""",
            (person_id, related_person_id, relationship_type, confidence, source_ref),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def get_relationships(person_id: int) -> List[Dict[str, Any]]:
    """Get all relationships for a person.

    Args:
        person_id: The person

    Returns:
        List of relationship dicts with related person name
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT r.relationship_type, r.confidence, r.source_ref,
                      p.id as related_id, p.canonical_name as related_name
               FROM person_relationships r
               JOIN persons p ON p.id = r.related_person_id
               WHERE r.person_id = ?""",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def expand_person_ids_with_relationships(
    person_ids: List[int],
    max_depth: int = 1,
) -> List[int]:
    """Expand a list of person IDs by traversing relationships.

    Given [42] (Shiran), if Shiran has relationships spouse→17 (David)
    and parent→23 (Mia), returns [42, 17, 23].

    Useful for queries like "tell me about Shiran's family" where we
    want to also retrieve assets belonging to related persons.

    Args:
        person_ids: Starting person IDs
        max_depth: How many relationship hops to follow (default: 1)

    Returns:
        Expanded list of person IDs (includes originals + related)
    """
    if not person_ids or max_depth < 1:
        return list(person_ids)

    conn = _get_connection()
    try:
        expanded: set = set(person_ids)
        frontier = set(person_ids)

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: set = set()
            for pid in frontier:
                rows = conn.execute(
                    "SELECT related_person_id FROM person_relationships WHERE person_id = ?",
                    (pid,),
                ).fetchall()
                for r in rows:
                    rid = r["related_person_id"]
                    if rid not in expanded:
                        expanded.add(rid)
                        next_frontier.add(rid)
                # Also check reverse relationships
                rows_rev = conn.execute(
                    "SELECT person_id FROM person_relationships WHERE related_person_id = ?",
                    (pid,),
                ).fetchall()
                for r in rows_rev:
                    rid = r["person_id"]
                    if rid not in expanded:
                        expanded.add(rid)
                        next_frontier.add(rid)
            frontier = next_frontier

        return list(expanded)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bulk / summary queries
# ---------------------------------------------------------------------------

def get_all_persons_summary() -> List[Dict[str, Any]]:
    """Get all persons with summary info for system prompt injection.

    Returns a lightweight list suitable for building the system prompt's
    known contacts section. Each entry includes aliases and fact count
    but not full fact details (to save tokens).

    Returns:
        List of person summary dicts sorted by canonical_name
    """
    conn = _get_connection()
    try:
        persons = conn.execute(
            "SELECT id, canonical_name, whatsapp_id, phone, is_group, last_seen "
            "FROM persons ORDER BY canonical_name"
        ).fetchall()

        results = []
        for p in persons:
            pid = p["id"]
            person = dict(p)

            # Aliases (fetch with script for display_name computation)
            aliases = conn.execute(
                "SELECT alias, script FROM person_aliases WHERE person_id = ?",
                (pid,),
            ).fetchall()
            alias_dicts = [{"alias": a["alias"], "script": a["script"]} for a in aliases]
            person["aliases"] = [a["alias"] for a in aliases]

            # Compute bilingual display name
            person["display_name"] = _compute_display_name(
                person["canonical_name"], alias_dicts,
            )

            # Key facts (just key-value, no metadata)
            facts = conn.execute(
                "SELECT fact_key, fact_value FROM person_facts WHERE person_id = ?",
                (pid,),
            ).fetchall()
            person["facts"] = {f["fact_key"]: f["fact_value"] for f in facts}

            results.append(person)

        return results
    finally:
        conn.close()


def get_person_context(name: str) -> Optional[str]:
    """Build a concise context string for a person (for system prompt injection).

    E.g., "Shiran Waintrob (שירן): female, ~32, friend, recently stressed about surgery"

    Args:
        name: Person name or alias

    Returns:
        Context string, or None if person not found
    """
    person = get_person_by_name(name)
    if not person:
        return None

    parts = [person["canonical_name"]]

    # Add non-canonical aliases in parentheses
    other_aliases = [
        a["alias"]
        for a in person.get("aliases", [])
        if a["alias"] != person["canonical_name"]
    ]
    if other_aliases:
        parts[0] += f" ({', '.join(other_aliases[:3])})"

    # Add key facts
    facts = person.get("facts", {})
    fact_parts = []
    for key in ["gender", "age", "birth_date", "city", "job_title", "recent_topic", "recent_mood"]:
        if key in facts:
            if key == "birth_date":
                fact_parts.append(f"born {facts[key]}")
            elif key == "recent_topic":
                fact_parts.append(f"recent: {facts[key]}")
            elif key == "recent_mood":
                fact_parts.append(f"mood: {facts[key]}")
            else:
                fact_parts.append(facts[key])

    if fact_parts:
        parts.append(", ".join(fact_parts))

    # Add relationships
    rels = person.get("relationships", [])
    if rels:
        rel_strs = [f"{r['relationship_type']} of {r['related_name']}" for r in rels[:2]]
        parts.append(", ".join(rel_strs))

    return " — ".join(parts)


def get_all_facts_global(
    fact_key: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Get all facts across all persons, optionally filtered by key.

    Returns a flat list of facts with person name attached, sorted
    by person name then fact key.

    Args:
        fact_key: Optional filter to only return facts with this key
        limit: Max results

    Returns:
        List of fact dicts with person_name, fact_key, fact_value, etc.
    """
    conn = _get_connection()
    try:
        if fact_key:
            rows = conn.execute(
                """SELECT f.fact_key, f.fact_value, f.confidence, f.source_type,
                          f.source_ref, f.extracted_at, f.person_id,
                          p.canonical_name AS person_name
                   FROM person_facts f
                   JOIN persons p ON p.id = f.person_id
                   WHERE f.fact_key = ?
                   ORDER BY p.canonical_name, f.fact_key
                   LIMIT ?""",
                (fact_key, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT f.fact_key, f.fact_value, f.confidence, f.source_type,
                          f.source_ref, f.extracted_at, f.person_id,
                          p.canonical_name AS person_name
                   FROM person_facts f
                   JOIN persons p ON p.id = f.person_id
                   ORDER BY p.canonical_name, f.fact_key
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_fact_keys() -> List[str]:
    """Get all distinct fact keys used across all persons.

    Returns:
        Sorted list of unique fact_key values
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT fact_key FROM person_facts ORDER BY fact_key"
        ).fetchall()
        return [r["fact_key"] for r in rows]
    finally:
        conn.close()


def get_stats() -> Dict[str, int]:
    """Get entity store statistics.

    Returns:
        Dict with counts of persons, aliases, facts, relationships
    """
    conn = _get_connection()
    try:
        persons = conn.execute("SELECT COUNT(*) as cnt FROM persons").fetchone()["cnt"]
        aliases = conn.execute("SELECT COUNT(*) as cnt FROM person_aliases").fetchone()["cnt"]
        facts = conn.execute("SELECT COUNT(*) as cnt FROM person_facts").fetchone()["cnt"]
        rels = conn.execute("SELECT COUNT(*) as cnt FROM person_relationships").fetchone()["cnt"]
        assets = conn.execute("SELECT COUNT(*) as cnt FROM person_assets").fetchone()["cnt"]
        return {
            "persons": persons,
            "aliases": aliases,
            "facts": facts,
            "relationships": rels,
            "person_assets": assets,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# WhatsApp contact seeding
# ---------------------------------------------------------------------------

def seed_from_whatsapp_contacts(contacts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Seed the entity store from WhatsApp contact data.

    Takes a list of contact dicts (from WAHA API) and creates
    person records with aliases. Existing persons are updated
    (whatsapp_id/phone filled in if missing).

    Args:
        contacts: List of contact dicts with keys:
            id, name, pushname, number, isBusiness, isMyContact

    Returns:
        Dict with counts: created, updated, skipped
    """
    created = 0
    updated = 0
    skipped = 0

    for contact in contacts:
        name = contact.get("name") or contact.get("pushname")
        whatsapp_id = contact.get("id")
        phone = contact.get("number")

        if not name or not _is_valid_person_name(name):
            skipped += 1
            continue

        # Skip system/status contacts
        if whatsapp_id and (
            whatsapp_id.endswith("@broadcast")
            or whatsapp_id.endswith("@newsletter")
            or whatsapp_id == "status@broadcast"
        ):
            skipped += 1
            continue

        # Detect WhatsApp Linked ID (LID) contacts.
        # Their "number" field contains the LID digits (e.g. 196121158754445),
        # NOT a real phone number.  Don't store it as phone.
        if whatsapp_id and whatsapp_id.endswith("@lid") and phone:
            lid_digits = whatsapp_id.replace("@lid", "")
            if phone == lid_digits:
                logger.debug(
                    f"Skipping LID number as phone for '{name}': {phone}"
                )
                phone = None

        conn = _get_connection()
        try:
            existing = conn.execute(
                "SELECT id FROM persons WHERE canonical_name = ?",
                (name,),
            ).fetchone()

            if existing:
                updated += 1
            else:
                created += 1
        finally:
            conn.close()

        # get_or_create_person handles upsert logic
        person_id = get_or_create_person(
            canonical_name=name,
            whatsapp_id=whatsapp_id,
            phone=phone,
        )

        # Add pushname as alias if different from name
        pushname = contact.get("pushname")
        if pushname and pushname != name:
            add_alias(person_id, pushname, source="whatsapp_pushname")

        # Add phone as alias for lookup convenience (only real phone numbers)
        if phone:
            add_alias(person_id, phone, script="numeric", source="whatsapp_contact")

        # Set is_business fact
        if contact.get("isBusiness"):
            set_fact(
                person_id,
                "is_business",
                "true",
                confidence=1.0,
                source_type="whatsapp",
            )

    logger.info(
        f"Entity seeding complete: {created} created, {updated} updated, {skipped} skipped"
    )
    return {"created": created, "updated": updated, "skipped": skipped}


# ---------------------------------------------------------------------------
# Entity merging
# ---------------------------------------------------------------------------

def merge_persons(
    target_id: int,
    source_ids: List[int],
) -> Dict[str, Any]:
    """Merge multiple person records into one target person.

    The target person keeps its canonical_name and core fields.
    From each source person, we absorb:
    - All aliases (re-pointed to target)
    - All facts (only if target doesn't have that fact_key yet,
      or source has higher confidence)
    - All relationships (re-pointed to target)
    - Phone/email/whatsapp_id (if target's are NULL)

    Source persons are deleted after merge.

    Args:
        target_id: The person ID to keep (merge target)
        source_ids: List of person IDs to merge INTO the target

    Returns:
        Dict with merge summary: aliases_moved, facts_moved, relationships_moved,
        sources_deleted, new_display_name
    """
    if not source_ids:
        return {"error": "No source IDs provided"}

    # Remove target from sources if accidentally included
    source_ids = [sid for sid in source_ids if sid != target_id]
    if not source_ids:
        return {"error": "No source IDs to merge (all were the target)"}

    conn = _get_connection()
    try:
        # Verify target exists
        target = conn.execute(
            "SELECT id, canonical_name, phone, email, whatsapp_id FROM persons WHERE id = ?",
            (target_id,),
        ).fetchone()
        if not target:
            return {"error": f"Target person {target_id} not found"}

        aliases_moved = 0
        facts_moved = 0
        rels_moved = 0
        sources_deleted = 0

        for source_id in source_ids:
            source = conn.execute(
                "SELECT id, canonical_name, phone, email, whatsapp_id FROM persons WHERE id = ?",
                (source_id,),
            ).fetchone()
            if not source:
                continue

            # 1. Move aliases — re-point to target, skip duplicates
            source_aliases = conn.execute(
                "SELECT id, alias, script, source FROM person_aliases WHERE person_id = ?",
                (source_id,),
            ).fetchall()
            for alias_row in source_aliases:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO person_aliases (person_id, alias, script, source)
                           VALUES (?, ?, ?, ?)""",
                        (target_id, alias_row["alias"], alias_row["script"], alias_row["source"]),
                    )
                    aliases_moved += 1
                except sqlite3.IntegrityError:
                    pass

            # Also add source's canonical_name as an alias on the target
            try:
                source_name = source["canonical_name"]
                source_script = _detect_script(source_name)
                conn.execute(
                    """INSERT OR IGNORE INTO person_aliases (person_id, alias, script, source)
                       VALUES (?, ?, ?, ?)""",
                    (target_id, source_name, source_script, "merge"),
                )
            except sqlite3.IntegrityError:
                pass

            # 2. Move facts — only if target doesn't have them or source has higher confidence
            source_facts = conn.execute(
                "SELECT fact_key, fact_value, confidence, source_type, source_ref "
                "FROM person_facts WHERE person_id = ?",
                (source_id,),
            ).fetchall()
            for fact_row in source_facts:
                existing = conn.execute(
                    "SELECT confidence FROM person_facts WHERE person_id = ? AND fact_key = ?",
                    (target_id, fact_row["fact_key"]),
                ).fetchone()
                if not existing:
                    # Target doesn't have this fact — add it
                    conn.execute(
                        """INSERT INTO person_facts
                           (person_id, fact_key, fact_value, confidence, source_type, source_ref)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (target_id, fact_row["fact_key"], fact_row["fact_value"],
                         fact_row["confidence"], fact_row["source_type"], fact_row["source_ref"]),
                    )
                    facts_moved += 1
                elif fact_row["confidence"] > existing["confidence"]:
                    # Source has higher confidence — overwrite
                    conn.execute(
                        """UPDATE person_facts
                           SET fact_value = ?, confidence = ?, source_type = ?,
                               source_ref = ?, extracted_at = CURRENT_TIMESTAMP
                           WHERE person_id = ? AND fact_key = ?""",
                        (fact_row["fact_value"], fact_row["confidence"],
                         fact_row["source_type"], fact_row["source_ref"],
                         target_id, fact_row["fact_key"]),
                    )
                    facts_moved += 1

            # 3. Move relationships — re-point to target
            source_rels = conn.execute(
                "SELECT related_person_id, relationship_type, confidence, source_ref "
                "FROM person_relationships WHERE person_id = ?",
                (source_id,),
            ).fetchall()
            for rel_row in source_rels:
                related_id = rel_row["related_person_id"]
                # Don't create self-referencing relationship
                if related_id == target_id:
                    continue
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO person_relationships
                           (person_id, related_person_id, relationship_type, confidence, source_ref)
                           VALUES (?, ?, ?, ?, ?)""",
                        (target_id, related_id, rel_row["relationship_type"],
                         rel_row["confidence"], rel_row["source_ref"]),
                    )
                    rels_moved += 1
                except sqlite3.IntegrityError:
                    pass

            # Also re-point reverse relationships (where source was the related_person).
            # First delete any that would conflict with existing target relationships,
            # then update the remaining ones.
            conn.execute(
                """DELETE FROM person_relationships
                   WHERE related_person_id = ?
                     AND person_id != ?
                     AND (person_id, relationship_type) IN (
                         SELECT person_id, relationship_type
                         FROM person_relationships
                         WHERE related_person_id = ?
                     )""",
                (source_id, target_id, target_id),
            )
            conn.execute(
                """UPDATE OR IGNORE person_relationships
                   SET related_person_id = ?
                   WHERE related_person_id = ? AND person_id != ?""",
                (target_id, source_id, target_id),
            )

            # 4. Absorb identifiers if target's are NULL
            if not target["phone"] and source["phone"]:
                conn.execute(
                    "UPDATE persons SET phone = ? WHERE id = ?",
                    (source["phone"], target_id),
                )
            if not target["email"] and source["email"]:
                conn.execute(
                    "UPDATE persons SET email = ? WHERE id = ?",
                    (source["email"], target_id),
                )
            if not target["whatsapp_id"] and source["whatsapp_id"]:
                conn.execute(
                    "UPDATE persons SET whatsapp_id = ? WHERE id = ?",
                    (source["whatsapp_id"], target_id),
                )

            # 5. Delete source person (cascades aliases, facts, relationships)
            conn.execute("DELETE FROM persons WHERE id = ?", (source_id,))
            sources_deleted += 1

        # Update target's last_updated
        conn.execute(
            "UPDATE persons SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
            (target_id,),
        )

        # Try to build bilingual display name after merge
        new_display = _build_display_name(conn, target_id)
        if new_display:
            dup = conn.execute(
                "SELECT id FROM persons WHERE canonical_name = ? AND id != ?",
                (new_display, target_id),
            ).fetchone()
            if not dup:
                conn.execute(
                    "UPDATE persons SET canonical_name = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_display, target_id),
                )

        conn.commit()

        # Read final display name
        final_row = conn.execute(
            "SELECT canonical_name FROM persons WHERE id = ?", (target_id,)
        ).fetchone()
        final_name = final_row["canonical_name"] if final_row else ""

        logger.info(
            f"Entity merge: {sources_deleted} persons merged into {target_id} "
            f"({aliases_moved} aliases, {facts_moved} facts, {rels_moved} rels)"
        )
        return {
            "target_id": target_id,
            "aliases_moved": aliases_moved,
            "facts_moved": facts_moved,
            "relationships_moved": rels_moved,
            "sources_deleted": sources_deleted,
            "display_name": final_name,
        }
    finally:
        conn.close()


def find_merge_candidates(limit: int = 50) -> List[Dict[str, Any]]:
    """Find potential duplicate persons that could be merged.

    Detection strategies (in priority order):
    1. Same phone number
    2. Same WhatsApp ID
    3. Same email (persons table or facts)
    4. Shared alias text (two persons with the same alias)
    5. Similar names — first name matches across different persons
       (e.g., "David" alias on person A == "David" alias on person B)

    Args:
        limit: Maximum number of candidate groups to return

    Returns:
        List of merge candidate groups, each with 'reason' and 'persons'
    """
    conn = _get_connection()
    try:
        candidates: list[Dict[str, Any]] = []
        seen_groups: set[frozenset[int]] = set()

        def _add_candidate(reason: str, ids: List[int]) -> None:
            """Helper to add a candidate group, deduplicating by ID set."""
            key = frozenset(ids)
            if key not in seen_groups and len(ids) >= 2:
                seen_groups.add(key)
                persons = _get_mini_persons(conn, ids)
                if len(persons) >= 2:
                    candidates.append({
                        "reason": reason,
                        "persons": persons,
                    })

        # 1. Same phone number
        phone_rows = conn.execute(
            """SELECT phone, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
               FROM persons WHERE phone IS NOT NULL AND phone != ''
               GROUP BY phone HAVING cnt > 1 LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in phone_rows:
            ids = [int(x) for x in row["ids"].split(",")]
            _add_candidate(f"📱 Same phone: {row['phone']}", ids)

        # 2. Same WhatsApp ID
        wa_rows = conn.execute(
            """SELECT whatsapp_id, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
               FROM persons WHERE whatsapp_id IS NOT NULL AND whatsapp_id != ''
               GROUP BY whatsapp_id HAVING cnt > 1 LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in wa_rows:
            ids = [int(x) for x in row["ids"].split(",")]
            _add_candidate(f"💬 Same WhatsApp: {row['whatsapp_id']}", ids)

        # 3a. Same email (persons table)
        email_rows = conn.execute(
            """SELECT email, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
               FROM persons WHERE email IS NOT NULL AND email != ''
               GROUP BY LOWER(email) HAVING cnt > 1 LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in email_rows:
            ids = [int(x) for x in row["ids"].split(",")]
            _add_candidate(f"📧 Same email: {row['email']}", ids)

        # 3b. Same email via facts
        email_fact_rows = conn.execute(
            """SELECT fact_value, GROUP_CONCAT(person_id) as ids, COUNT(*) as cnt
               FROM person_facts WHERE fact_key = 'email'
               GROUP BY LOWER(fact_value) HAVING cnt > 1 LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in email_fact_rows:
            ids = [int(x) for x in row["ids"].split(",")]
            _add_candidate(f"📧 Same email (fact): {row['fact_value']}", ids)

        # 4. Shared alias — two different persons with the exact same alias text
        #    (at least 2 words to avoid false positives on common first names)
        shared_alias_rows = conn.execute(
            """SELECT alias, GROUP_CONCAT(DISTINCT person_id) as ids, COUNT(DISTINCT person_id) as cnt
               FROM person_aliases
               WHERE script != 'numeric' AND alias LIKE '% %'
               GROUP BY alias COLLATE NOCASE HAVING cnt > 1 LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in shared_alias_rows:
            ids = [int(x) for x in row["ids"].split(",")]
            _add_candidate(f"🏷️ Same alias: \"{row['alias']}\"", ids)

        # 5. Name similarity — find persons whose first name (first word of
        #    canonical_name) matches an alias of another person.
        #    This catches cases like "דוד כהן" and "David Cohen" where both
        #    have an alias "David" / "דוד" that doesn't literally match but
        #    the first name was added as alias in the other script.
        if len(candidates) < limit:
            _find_name_similarity_candidates(conn, candidates, seen_groups, limit)

        return candidates[:limit]
    finally:
        conn.close()


def _find_name_similarity_candidates(
    conn: sqlite3.Connection,
    candidates: list,
    seen_groups: set,
    limit: int,
) -> None:
    """Find persons with matching names across different person records.

    Detection strategies:
    1. Full name match — two persons share an identical full alias
       (e.g., "David Cohen" on person A == "David Cohen" on person B)
    2. Canonical name match — same canonical_name text across persons
       (can happen from different import sources)

    Full-name matches are prioritized over single-word (first name) matches
    because they are stronger duplicate signals.
    """
    # Fetch all non-numeric aliases
    all_aliases = conn.execute(
        "SELECT person_id, alias, script FROM person_aliases WHERE script IN ('hebrew', 'latin')"
    ).fetchall()

    # Build alias → person_ids mapping for full names only (2+ words)
    # Single first names like "David" cause too many false positives
    alias_to_persons: dict[str, set[int]] = {}
    for row in all_aliases:
        alias = row["alias"].strip()
        pid = row["person_id"]
        # Require at least 2 words (first + surname)
        if " " not in alias or len(alias) < 3:
            continue
        # Normalize: lowercase for Latin, exact for Hebrew
        key = alias.lower() if row["script"] == "latin" else alias
        if key not in alias_to_persons:
            alias_to_persons[key] = set()
        alias_to_persons[key].add(pid)

    # Find full names shared by multiple persons
    for alias_text, person_ids in sorted(alias_to_persons.items()):
        if len(person_ids) < 2:
            continue
        if len(candidates) >= limit:
            break

        ids = sorted(person_ids)
        key = frozenset(ids)
        if key in seen_groups:
            continue

        seen_groups.add(key)
        persons = _get_mini_persons(conn, ids)
        if len(persons) >= 2:
            candidates.append({
                "reason": f"👤 Same full name: \"{alias_text}\"",
                "persons": persons,
            })


def _get_mini_persons(conn: sqlite3.Connection, ids: List[int]) -> List[Dict[str, Any]]:
    """Get minimal person info for a list of IDs (used in merge candidates)."""
    result: list[Dict[str, Any]] = []
    for pid in ids:
        row = conn.execute(
            "SELECT id, canonical_name, phone, email, whatsapp_id FROM persons WHERE id = ?",
            (pid,),
        ).fetchone()
        if row:
            p = dict(row)
            # Count facts and aliases
            p["alias_count"] = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_aliases WHERE person_id = ?", (pid,)
            ).fetchone()["cnt"]
            p["fact_count"] = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_facts WHERE person_id = ?", (pid,)
            ).fetchone()["cnt"]
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Cleanup — remove garbage persons
# ---------------------------------------------------------------------------

def get_graph_data(limit: int = 100) -> Dict[str, Any]:
    """Build a graph representation of persons, relationships, and asset counts.

    Returns nodes (persons) and edges (relationships) suitable for
    rendering with a graph visualization library (e.g. cytoscape.js).

    Each node includes:
    - id, label (canonical_name), facts count, aliases count
    - asset_counts: dict of asset_type → count from person_assets

    Each edge includes:
    - source, target, relationship_type, confidence

    Args:
        limit: Max persons to include

    Returns:
        Dict with 'nodes' list and 'edges' list
    """
    conn = _get_connection()
    try:
        # Fetch persons prioritized by connectivity:
        # persons with relationships, assets, or facts come first
        persons = conn.execute(
            """SELECT p.id, p.canonical_name, p.phone, p.is_group,
                      COALESCE(asset_cnt.cnt, 0) AS _assets,
                      COALESCE(fact_cnt.cnt, 0) AS _facts,
                      COALESCE(rel_cnt.cnt, 0) AS _rels
               FROM persons p
               LEFT JOIN (SELECT person_id, COUNT(*) as cnt FROM person_assets GROUP BY person_id) asset_cnt
                   ON asset_cnt.person_id = p.id
               LEFT JOIN (SELECT person_id, COUNT(*) as cnt FROM person_facts GROUP BY person_id) fact_cnt
                   ON fact_cnt.person_id = p.id
               LEFT JOIN (SELECT person_id, COUNT(*) as cnt FROM person_relationships GROUP BY person_id) rel_cnt
                   ON rel_cnt.person_id = p.id
               WHERE p.is_group = FALSE
               ORDER BY (_rels > 0) DESC, (_assets > 0) DESC, (_facts > 0) DESC, p.canonical_name
               LIMIT ?""",
            (limit,),
        ).fetchall()

        nodes = []
        person_ids_in_graph: set = set()

        for p in persons:
            pid = p["id"]
            person_ids_in_graph.add(pid)

            # Count aliases
            alias_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_aliases WHERE person_id = ?",
                (pid,),
            ).fetchone()["cnt"]

            # Count facts
            fact_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_facts WHERE person_id = ?",
                (pid,),
            ).fetchone()["cnt"]

            # Asset counts by type
            asset_rows = conn.execute(
                """SELECT asset_type, COUNT(*) as cnt
                   FROM person_assets WHERE person_id = ?
                   GROUP BY asset_type""",
                (pid,),
            ).fetchall()
            asset_counts = {r["asset_type"]: r["cnt"] for r in asset_rows}
            total_assets = sum(asset_counts.values())
            # Format asset summary as a flat string for Reflex compatibility
            # (Reflex foreach can't handle nested dicts well)
            asset_summary = ", ".join(
                f"{cnt} {atype}" for atype, cnt in sorted(asset_counts.items())
            ) if asset_counts else ""

            nodes.append({
                "id": str(pid),
                "label": p["canonical_name"],
                "phone": p["phone"] or "",
                "alias_count": str(alias_count),
                "fact_count": str(fact_count),
                "total_assets": str(total_assets),
                "asset_summary": asset_summary,
            })

        # Fetch relationships between persons in the graph
        edges = []
        if person_ids_in_graph:
            placeholders = ",".join("?" for _ in person_ids_in_graph)
            ids_list = list(person_ids_in_graph)
            rel_rows = conn.execute(
                f"""SELECT r.person_id, r.related_person_id,
                           r.relationship_type, r.confidence,
                           p1.canonical_name AS source_name,
                           p2.canonical_name AS target_name
                    FROM person_relationships r
                    JOIN persons p1 ON p1.id = r.person_id
                    JOIN persons p2 ON p2.id = r.related_person_id
                    WHERE r.person_id IN ({placeholders})
                      AND r.related_person_id IN ({placeholders})""",
                ids_list + ids_list,
            ).fetchall()

            for r in rel_rows:
                conf = r["confidence"]
                edges.append({
                    "source": r["source_name"],
                    "target": r["target_name"],
                    "source_id": r["person_id"],
                    "target_id": r["related_person_id"],
                    "relationship_type": r["relationship_type"],
                    "confidence": f"{int(conf * 100)}%" if conf else "",
                })

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


def get_full_graph_data(
    limit_persons: int = 100,
    limit_assets_per_person: int = 10,
    include_asset_edges: bool = True,
) -> Dict[str, Any]:
    """Build a full graph with person nodes, asset nodes, and all edge types.

    Returns a graph suitable for interactive visualization (Neo4j-style).

    Node types:
        - person: {id, type, label, phone, fact_count, alias_count, total_assets}
        - asset: {id, type, label, source, timestamp, asset_type}

    Edge types:
        - identity↔identity: {source, target, type, confidence}
        - identity↔asset: {source, target, type, role}
        - asset↔asset: {source, target, type, confidence}

    Args:
        limit_persons: Max person nodes to include
        limit_assets_per_person: Max assets per person
        include_asset_edges: Whether to include asset↔asset edges

    Returns:
        Dict with 'nodes' list and 'edges' list
    """
    conn = _get_connection()
    try:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        person_ids_in_graph: set = set()
        asset_refs_in_graph: set = set()

        # --- Person nodes (same query as get_graph_data but simplified) ---
        persons = conn.execute(
            """SELECT p.id, p.canonical_name, p.phone, p.is_group
               FROM persons p
               WHERE p.is_group = FALSE
               ORDER BY p.canonical_name
               LIMIT ?""",
            (limit_persons,),
        ).fetchall()

        for p in persons:
            pid = p["id"]
            person_ids_in_graph.add(pid)

            alias_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_aliases WHERE person_id = ?",
                (pid,),
            ).fetchone()["cnt"]

            fact_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM person_facts WHERE person_id = ?",
                (pid,),
            ).fetchone()["cnt"]

            asset_rows = conn.execute(
                """SELECT asset_type, COUNT(*) as cnt
                   FROM person_assets WHERE person_id = ?
                   GROUP BY asset_type""",
                (pid,),
            ).fetchall()
            total_assets = sum(r["cnt"] for r in asset_rows)

            nodes.append({
                "id": f"person:{pid}",
                "type": "person",
                "label": p["canonical_name"],
                "phone": p["phone"] or "",
                "alias_count": alias_count,
                "fact_count": fact_count,
                "total_assets": total_assets,
            })

        # --- Identity↔identity edges ---
        if person_ids_in_graph:
            placeholders = ",".join("?" for _ in person_ids_in_graph)
            ids_list = list(person_ids_in_graph)
            rel_rows = conn.execute(
                f"""SELECT r.person_id, r.related_person_id,
                           r.relationship_type, r.confidence
                    FROM person_relationships r
                    WHERE r.person_id IN ({placeholders})
                      AND r.related_person_id IN ({placeholders})""",
                ids_list + ids_list,
            ).fetchall()

            for r in rel_rows:
                edges.append({
                    "source": f"person:{r['person_id']}",
                    "target": f"person:{r['related_person_id']}",
                    "type": r["relationship_type"],
                    "edge_category": "identity_identity",
                    "confidence": r["confidence"],
                })

        # --- Asset nodes + identity↔asset edges ---
        for pid in person_ids_in_graph:
            asset_links = conn.execute(
                """SELECT asset_type, asset_ref, role, confidence
                   FROM person_assets
                   WHERE person_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (pid, limit_assets_per_person),
            ).fetchall()

            for link in asset_links:
                aref = link["asset_ref"]
                atype = link["asset_type"]

                # Add asset node if not already added
                if aref not in asset_refs_in_graph:
                    asset_refs_in_graph.add(aref)

                    # Derive a short label from the asset_ref
                    label = aref
                    if ":" in aref:
                        parts = aref.split(":")
                        label = parts[-1][:30] if parts else aref[:30]

                    nodes.append({
                        "id": f"asset:{aref}",
                        "type": "asset",
                        "asset_type": atype,
                        "label": label,
                        "source": atype,
                    })

                # Identity↔asset edge
                edges.append({
                    "source": f"person:{pid}",
                    "target": f"asset:{aref}",
                    "type": link["role"],
                    "edge_category": "identity_asset",
                    "confidence": link["confidence"],
                })

        # --- Asset↔asset edges ---
        if include_asset_edges and asset_refs_in_graph:
            placeholders = ",".join("?" for _ in asset_refs_in_graph)
            refs_list = list(asset_refs_in_graph)

            aa_rows = conn.execute(
                f"""SELECT src_asset_ref, dst_asset_ref, relation_type, confidence
                    FROM asset_asset_edges
                    WHERE src_asset_ref IN ({placeholders})
                       OR dst_asset_ref IN ({placeholders})
                    LIMIT 500""",
                refs_list + refs_list,
            ).fetchall()

            for r in aa_rows:
                src = r["src_asset_ref"]
                dst = r["dst_asset_ref"]

                # Add missing nodes for assets discovered via edges
                for ref in (src, dst):
                    if ref not in asset_refs_in_graph and not ref.startswith("thread:"):
                        asset_refs_in_graph.add(ref)
                        nodes.append({
                            "id": f"asset:{ref}",
                            "type": "asset",
                            "asset_type": "linked",
                            "label": ref.split(":")[-1][:30] if ":" in ref else ref[:30],
                            "source": "linked",
                        })

                edges.append({
                    "source": f"asset:{src}",
                    "target": f"asset:{dst}",
                    "type": r["relation_type"],
                    "edge_category": "asset_asset",
                    "confidence": r["confidence"],
                })

        logger.info(
            f"Full graph: {len(nodes)} nodes ({len(person_ids_in_graph)} persons, "
            f"{len(asset_refs_in_graph)} assets), {len(edges)} edges"
        )

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


def cleanup_garbage_persons() -> Dict[str, Any]:
    """Remove persons with invalid/garbage names from the entity store.

    Identifies persons whose canonical_name fails _is_valid_person_name()
    and deletes them (cascades to aliases, facts, relationships).

    Returns:
        Dict with 'deleted' count and 'names' list of removed names
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, canonical_name FROM persons"
        ).fetchall()

        garbage_ids: list[int] = []
        garbage_names: list[str] = []
        for row in rows:
            name = row["canonical_name"]
            if not _is_valid_person_name(name):
                garbage_ids.append(row["id"])
                garbage_names.append(name)

        if garbage_ids:
            placeholders = ",".join("?" for _ in garbage_ids)
            conn.execute(
                f"DELETE FROM persons WHERE id IN ({placeholders})",
                garbage_ids,
            )
            conn.commit()

        logger.info(
            f"Entity cleanup: removed {len(garbage_ids)} garbage persons"
        )
        return {"deleted": len(garbage_ids), "names": garbage_names}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Person-asset graph CRUD
# ---------------------------------------------------------------------------

def link_person_asset(
    person_id: int,
    asset_type: str,
    asset_ref: str,
    role: str = "sender",
    confidence: float = 1.0,
) -> bool:
    """Link a person to an asset (message, document, call recording).

    Creates an entry in the person_assets junction table and is
    idempotent — duplicate (person_id, asset_ref, role) tuples
    are silently ignored.

    Args:
        person_id: The person's database ID
        asset_type: Asset kind ('whatsapp_msg', 'document', 'call_recording', 'gmail')
        asset_ref: Qdrant point source_id (e.g. '972501234567@c.us:1708012345')
        role: How this person relates to the asset
              ('sender', 'recipient', 'mentioned', 'participant', 'owner')
        confidence: How confident we are in this link (0.0-1.0)

    Returns:
        True if the link was created (False if it already existed)
    """
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO person_assets
               (person_id, asset_type, asset_ref, role, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (person_id, asset_type, asset_ref, role, confidence),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def link_persons_to_asset(
    person_ids: List[int],
    asset_type: str,
    asset_ref: str,
    role: str = "sender",
    confidence: float = 1.0,
) -> int:
    """Link multiple persons to the same asset in a single transaction.

    Args:
        person_ids: List of person database IDs
        asset_type: Asset kind
        asset_ref: Qdrant point source_id
        role: Relationship role
        confidence: Confidence score

    Returns:
        Number of links created
    """
    if not person_ids:
        return 0

    conn = _get_connection()
    try:
        created = 0
        for pid in person_ids:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO person_assets
                       (person_id, asset_type, asset_ref, role, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pid, asset_type, asset_ref, role, confidence),
                )
                created += conn.total_changes
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return created
    finally:
        conn.close()


def get_person_asset_refs(
    person_id: int,
    asset_type: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Get all asset references linked to a person.

    Args:
        person_id: The person's database ID
        asset_type: Optional filter by asset type
        role: Optional filter by role
        limit: Max results

    Returns:
        List of dicts with asset_type, asset_ref, role, confidence
    """
    conn = _get_connection()
    try:
        query = "SELECT asset_type, asset_ref, role, confidence, created_at FROM person_assets WHERE person_id = ?"
        params: list = [person_id]

        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        if role:
            query += " AND role = ?"
            params.append(role)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_asset_person_ids(asset_ref: str) -> List[Dict[str, Any]]:
    """Get all persons linked to a specific asset.

    Args:
        asset_ref: The Qdrant point source_id

    Returns:
        List of dicts with person_id, role, confidence, and person canonical_name
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT pa.person_id, pa.role, pa.confidence,
                      p.canonical_name AS person_name
               FROM person_assets pa
               JOIN persons p ON p.id = pa.person_id
               WHERE pa.asset_ref = ?""",
            (asset_ref,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_person_asset_count(person_id: int) -> Dict[str, int]:
    """Get asset counts grouped by type for a person.

    Args:
        person_id: The person's database ID

    Returns:
        Dict mapping asset_type → count (e.g. {'whatsapp_msg': 42, 'document': 3})
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT asset_type, COUNT(*) as cnt
               FROM person_assets WHERE person_id = ?
               GROUP BY asset_type""",
            (person_id,),
        ).fetchall()
        return {r["asset_type"]: r["cnt"] for r in rows}
    finally:
        conn.close()


def delete_person_asset(person_id: int, asset_ref: str, role: Optional[str] = None) -> bool:
    """Remove a person-asset link.

    Args:
        person_id: The person's database ID
        asset_ref: The asset reference to unlink
        role: Optional specific role to remove (removes all roles if None)

    Returns:
        True if any links were removed
    """
    conn = _get_connection()
    try:
        if role:
            cursor = conn.execute(
                "DELETE FROM person_assets WHERE person_id = ? AND asset_ref = ? AND role = ?",
                (person_id, asset_ref, role),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM person_assets WHERE person_id = ? AND asset_ref = ?",
                (person_id, asset_ref),
            )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Asset-asset graph CRUD
# ---------------------------------------------------------------------------

def link_assets(
    src_asset_ref: str,
    dst_asset_ref: str,
    relation_type: str,
    confidence: float = 1.0,
    provenance: Optional[str] = None,
) -> bool:
    """Create an edge between two assets.

    Idempotent — duplicate (src, dst, relation_type) tuples are silently
    ignored.  Valid relation types: thread_member, attachment_of, chunk_of,
    reply_to, references, transcript_of.

    Args:
        src_asset_ref: Source asset's Qdrant source_id
        dst_asset_ref: Destination asset's Qdrant source_id
        relation_type: Edge type (e.g. 'attachment_of', 'thread_member')
        confidence: Confidence score (0.0-1.0)
        provenance: Where this edge was derived from

    Returns:
        True if the edge was created (False if it already existed)
    """
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO asset_asset_edges
               (src_asset_ref, dst_asset_ref, relation_type, confidence, provenance)
               VALUES (?, ?, ?, ?, ?)""",
            (src_asset_ref, dst_asset_ref, relation_type, confidence, provenance),
        )
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def link_assets_batch(
    edges: List[Dict[str, Any]],
) -> int:
    """Create multiple asset-asset edges in a single transaction.

    Each edge dict should have keys: src_asset_ref, dst_asset_ref,
    relation_type, and optionally confidence and provenance.

    Args:
        edges: List of edge dicts

    Returns:
        Number of edges created
    """
    if not edges:
        return 0

    conn = _get_connection()
    try:
        created = 0
        for edge in edges:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO asset_asset_edges
                       (src_asset_ref, dst_asset_ref, relation_type, confidence, provenance)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        edge["src_asset_ref"],
                        edge["dst_asset_ref"],
                        edge["relation_type"],
                        edge.get("confidence", 1.0),
                        edge.get("provenance"),
                    ),
                )
                created += conn.total_changes
            except (sqlite3.IntegrityError, KeyError):
                pass
        conn.commit()
        return created
    finally:
        conn.close()


def get_asset_neighbors(
    asset_ref: str,
    relation_types: Optional[List[str]] = None,
    direction: str = "both",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get neighboring assets connected by edges.

    Args:
        asset_ref: The asset's Qdrant source_id
        relation_types: Optional filter by edge types
        direction: 'outgoing', 'incoming', or 'both'
        limit: Max results

    Returns:
        List of dicts with neighbor_ref, relation_type, direction,
        confidence, provenance
    """
    conn = _get_connection()
    try:
        results: List[Dict[str, Any]] = []

        # Build relation_type filter clause
        type_clause = ""
        type_params: list = []
        if relation_types:
            placeholders = ",".join("?" for _ in relation_types)
            type_clause = f" AND relation_type IN ({placeholders})"
            type_params = list(relation_types)

        if direction in ("outgoing", "both"):
            rows = conn.execute(
                f"""SELECT dst_asset_ref AS neighbor_ref, relation_type,
                           confidence, provenance
                    FROM asset_asset_edges
                    WHERE src_asset_ref = ?{type_clause}
                    ORDER BY created_at DESC LIMIT ?""",
                [asset_ref] + type_params + [limit],
            ).fetchall()
            for r in rows:
                d = dict(r)
                d["direction"] = "outgoing"
                results.append(d)

        if direction in ("incoming", "both"):
            remaining = limit - len(results)
            if remaining > 0:
                rows = conn.execute(
                    f"""SELECT src_asset_ref AS neighbor_ref, relation_type,
                               confidence, provenance
                        FROM asset_asset_edges
                        WHERE dst_asset_ref = ?{type_clause}
                        ORDER BY created_at DESC LIMIT ?""",
                    [asset_ref] + type_params + [remaining],
                ).fetchall()
                for r in rows:
                    d = dict(r)
                    d["direction"] = "incoming"
                    results.append(d)

        return results
    finally:
        conn.close()


def get_thread_members(
    thread_id: str,
    limit: int = 100,
) -> List[str]:
    """Get all asset refs that belong to a thread.

    Looks for edges where either src or dst matches the thread_id
    and relation_type is 'thread_member'.

    Args:
        thread_id: The thread identifier
        limit: Max results

    Returns:
        List of asset_ref strings in the thread
    """
    conn = _get_connection()
    try:
        # Thread members are stored as edges from thread_id to asset_ref
        rows = conn.execute(
            """SELECT DISTINCT dst_asset_ref AS asset_ref
               FROM asset_asset_edges
               WHERE src_asset_ref = ? AND relation_type = 'thread_member'
               LIMIT ?""",
            (thread_id, limit),
        ).fetchall()
        return [r["asset_ref"] for r in rows]
    finally:
        conn.close()


def delete_asset_edge(
    src_asset_ref: str,
    dst_asset_ref: str,
    relation_type: Optional[str] = None,
) -> bool:
    """Remove an asset-asset edge.

    Args:
        src_asset_ref: Source asset reference
        dst_asset_ref: Destination asset reference
        relation_type: Optional specific type to remove (removes all if None)

    Returns:
        True if any edges were removed
    """
    conn = _get_connection()
    try:
        if relation_type:
            cursor = conn.execute(
                "DELETE FROM asset_asset_edges WHERE src_asset_ref = ? AND dst_asset_ref = ? AND relation_type = ?",
                (src_asset_ref, dst_asset_ref, relation_type),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM asset_asset_edges WHERE src_asset_ref = ? AND dst_asset_ref = ?",
                (src_asset_ref, dst_asset_ref),
            )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_asset_edge_stats() -> Dict[str, int]:
    """Get asset-asset edge counts by relation type.

    Returns:
        Dict mapping relation_type → count
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT relation_type, COUNT(*) as cnt
               FROM asset_asset_edges
               GROUP BY relation_type
               ORDER BY cnt DESC"""
        ).fetchall()
        return {r["relation_type"]: r["cnt"] for r in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auto-initialize on import
# ---------------------------------------------------------------------------

init_entity_db()
