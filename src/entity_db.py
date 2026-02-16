"""SQLite-backed entity store for person knowledge management.

Accumulates structured knowledge about people over time from WhatsApp
messages, Paperless documents, and other sources. Provides:

- Person records with canonical names and WhatsApp IDs
- Multi-script name aliases for cross-script disambiguation (שירן ↔ Shiran)
- Key-value facts (birth_date, city, job, etc.) with confidence scores
- Person-to-person relationships (friend, spouse, parent, etc.)

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
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                whatsapp_id TEXT,
                phone TEXT,
                is_group BOOLEAN DEFAULT FALSE,
                confidence REAL DEFAULT 0.5,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(canonical_name)
            )
        """)

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

        # Indexes for fast lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_whatsapp ON persons(whatsapp_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(canonical_name)"
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

        conn.commit()
        logger.info("Entity database tables initialized")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Person CRUD
# ---------------------------------------------------------------------------

def get_or_create_person(
    canonical_name: str,
    whatsapp_id: Optional[str] = None,
    phone: Optional[str] = None,
    is_group: bool = False,
) -> int:
    """Get a person ID by canonical_name, or create a new record.

    If the person already exists (by canonical_name), updates whatsapp_id/phone
    if they are provided and currently NULL, and updates last_seen.

    Also auto-creates aliases from the canonical name parts and any
    name variants (first name, full name in detected scripts).

    Args:
        canonical_name: Primary display name (e.g., "Shiran Waintrob")
        whatsapp_id: WhatsApp ID (e.g., "972501234567@c.us")
        phone: Phone number (e.g., "+972501234567")
        is_group: Whether this is a group entity

    Returns:
        Person ID (integer)
    """
    conn = _get_connection()
    try:
        # Try to find existing person
        row = conn.execute(
            "SELECT id FROM persons WHERE canonical_name = ?",
            (canonical_name,),
        ).fetchone()

        if row:
            person_id = row["id"]
            # Update fields if they're provided and currently NULL
            updates = []
            params: list = []
            if whatsapp_id:
                updates.append("whatsapp_id = COALESCE(whatsapp_id, ?)")
                params.append(whatsapp_id)
            if phone:
                updates.append("phone = COALESCE(phone, ?)")
                params.append(phone)
            updates.append("last_seen = CURRENT_TIMESTAMP")
            params.append(person_id)
            conn.execute(
                f"UPDATE persons SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
        else:
            # Create new person
            cursor = conn.execute(
                """INSERT INTO persons (canonical_name, whatsapp_id, phone, is_group)
                   VALUES (?, ?, ?, ?)""",
                (canonical_name, whatsapp_id, phone, is_group),
            )
            person_id = cursor.lastrowid
            conn.commit()

            # Auto-create aliases from canonical name
            _auto_create_aliases(conn, person_id, canonical_name)
            conn.commit()

        return person_id
    finally:
        conn.close()


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

    Args:
        person_id: The person's database ID

    Returns:
        Dict with person data, facts, aliases, and relationships, or None
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

        return person
    finally:
        conn.close()


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

            # Aliases
            aliases = conn.execute(
                "SELECT alias, script FROM person_aliases WHERE person_id = ?",
                (pid,),
            ).fetchall()
            person["aliases"] = [a["alias"] for a in aliases]

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
        return {
            "persons": persons,
            "aliases": aliases,
            "facts": facts,
            "relationships": rels,
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

        # Add phone as alias for lookup convenience
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
# Cleanup — remove garbage persons
# ---------------------------------------------------------------------------

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
# Auto-initialize on import
# ---------------------------------------------------------------------------

init_entity_db()
