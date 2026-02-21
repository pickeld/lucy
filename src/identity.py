"""Object-oriented Identity wrapper around entity_db person operations.

Provides a cached, object-oriented API for working with person entities.
Uses an identity-map pattern: ``Identity.get(42)`` returns a cached instance
so repeated lookups within the same process avoid redundant DB reads.

Key features:
- **Identity map cache** with configurable TTL (default 300s)
- **Lazy-loaded collections** (facts, aliases, relationships, asset_counts)
- **Write-through invalidation** — mutations clear local caches
- **Factory methods** for every lookup strategy (name, phone, email, whatsapp)
- **Delegates to entity_db** — no direct SQL; purely a convenience wrapper

Usage::

    from identity import Identity

    # Find or create
    person = Identity.find_or_create("Shiran Waintrob", phone="+972501234567")
    person.set_fact("city", "Tel Aviv", confidence=0.8)
    person.add_alias("שירן", source="whatsapp_pushname")

    # Cached re-fetch (no DB hit if still fresh)
    same = Identity.get(person.id)
    assert same is person

    # Lazy property access
    print(person.display_name)   # "Shiran Waintrob / שירן וינטרוב"
    print(person.facts)          # {"city": "Tel Aviv", ...}
    print(person.context_string())
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from utils.logger import logger


# ---------------------------------------------------------------------------
# Identity class
# ---------------------------------------------------------------------------

class Identity:
    """Cached, object-oriented wrapper around a person entity.

    Do NOT instantiate directly — use the factory class methods:
    ``Identity.get()``, ``Identity.get_by_name()``, ``Identity.find_or_create()``, etc.
    """

    # ── Class-level identity map cache ──────────────────────────────────────
    _cache: Dict[int, "Identity"] = {}
    _cache_timestamps: Dict[int, float] = {}
    CACHE_TTL: float = 300.0  # seconds; set to 0 to disable staleness checks

    # =====================================================================
    # Factory methods (class methods)
    # =====================================================================

    @classmethod
    def get(cls, person_id: int) -> Optional["Identity"]:
        """Get an Identity by person ID, returning a cached instance if fresh.

        Args:
            person_id: The person's database ID.

        Returns:
            An ``Identity`` instance, or ``None`` if no such person exists.
        """
        # Check cache
        cached = cls._cache.get(person_id)
        if cached is not None:
            if cls._is_fresh(person_id):
                return cached

        # Load from DB
        import identity_db
        data = identity_db.get_person(person_id)
        if data is None:
            # Remove stale cache entry if person was deleted
            cls._cache.pop(person_id, None)
            cls._cache_timestamps.pop(person_id, None)
            return None

        return cls._wrap(data)

    @classmethod
    def get_by_name(cls, name: str) -> Optional["Identity"]:
        """Look up a person by canonical name or any alias.

        Args:
            name: Name to search for (case-insensitive).

        Returns:
            An ``Identity`` instance, or ``None``.
        """
        import identity_db
        data = identity_db.get_person_by_name(name)
        if data is None:
            return None
        return cls._wrap(data)

    @classmethod
    def get_by_whatsapp_id(cls, whatsapp_id: str) -> Optional["Identity"]:
        """Look up a person by WhatsApp ID.

        Args:
            whatsapp_id: WhatsApp contact ID (e.g. ``"972501234567@c.us"``).

        Returns:
            An ``Identity`` instance, or ``None``.
        """
        import identity_db
        data = identity_db.get_person_by_whatsapp_id(whatsapp_id)
        if data is None:
            return None
        return cls._wrap(data)

    @classmethod
    def get_by_phone(cls, phone: str) -> Optional["Identity"]:
        """Look up a person by phone number (normalized comparison).

        Args:
            phone: Phone number to search for.

        Returns:
            An ``Identity`` instance, or ``None``.
        """
        import identity_db
        pid = identity_db.find_person_by_phone(phone)
        if pid is None:
            return None
        return cls.get(pid)

    @classmethod
    def get_by_email(cls, email: str) -> Optional["Identity"]:
        """Look up a person by email address (case-insensitive).

        Args:
            email: Email address to search for.

        Returns:
            An ``Identity`` instance, or ``None``.
        """
        import identity_db
        pid = identity_db.find_person_by_email(email)
        if pid is None:
            return None
        return cls.get(pid)

    @classmethod
    def find_or_create(
        cls,
        name: str,
        *,
        whatsapp_id: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        is_group: bool = False,
    ) -> "Identity":
        """Get or create a person using the identifier cascade, then wrap.

        Delegates to ``identity_db.get_or_create_person()`` which deduplicates
        by phone → email → name.

        Args:
            name: Canonical display name.
            whatsapp_id: WhatsApp ID.
            phone: Phone number.
            email: Email address.
            is_group: Whether this is a group entity.

        Returns:
            An ``Identity`` instance (always non-None).
        """
        import identity_db
        pid = identity_db.get_or_create_person(
            canonical_name=name,
            whatsapp_id=whatsapp_id,
            phone=phone,
            email=email,
            is_group=is_group,
        )
        # The person definitely exists now — load full data
        result = cls.get(pid)
        assert result is not None, f"get_or_create_person returned {pid} but get() returned None"
        return result

    @classmethod
    def resolve(
        cls,
        *,
        name: Optional[str] = None,
        whatsapp_id: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Optional["Identity"]:
        """Resolve a person to an Identity using all available identifiers.

        Tries identifiers in priority order (most specific → least):
        WhatsApp ID → phone → email → name.

        Delegates to ``person_resolver.resolve_person()``.

        Args:
            name: Display name or alias.
            whatsapp_id: WhatsApp contact ID.
            phone: Phone number.
            email: Email address.

        Returns:
            An ``Identity`` instance, or ``None`` if not found.
        """
        import person_resolver
        pid = person_resolver.resolve_person(
            name=name,
            whatsapp_id=whatsapp_id,
            phone=phone,
            email=email,
        )
        if pid is None:
            return None
        return cls.get(pid)

    @classmethod
    def search(cls, query: str, limit: int = 20) -> List["Identity"]:
        """Search persons by name/alias substring (for autocomplete/search UI).

        Args:
            query: Search string (substring match).
            limit: Maximum results.

        Returns:
            List of ``Identity`` instances, sorted by canonical name.
        """
        import identity_db
        summaries = identity_db.search_persons(query, limit=limit)
        results: List[Identity] = []
        for s in summaries:
            pid = s.get("id")
            if pid is not None:
                ident = cls.get(pid)
                if ident is not None:
                    results.append(ident)
        return results

    @classmethod
    def all_summary(cls) -> List["Identity"]:
        """Get all persons as Identity instances (lightweight).

        Loads via ``identity_db.get_all_persons_summary()`` and wraps each
        into a cached Identity. Note: the summary data has less detail
        than ``get_person()`` — lazy properties will reload on access.

        Returns:
            List of ``Identity`` instances sorted by canonical name.
        """
        import identity_db
        summaries = identity_db.get_all_persons_summary()
        results: List[Identity] = []
        for data in summaries:
            pid = data.get("id")
            if pid is not None:
                ident = cls._wrap_summary(data)
                results.append(ident)
        return results

    @classmethod
    def resolve_names(cls, names: List[str]) -> List["Identity"]:
        """Resolve a list of names to Identity instances.

        Skips names that can't be resolved.

        Args:
            names: List of display names.

        Returns:
            List of resolved ``Identity`` instances (may be shorter than input).
        """
        results: List[Identity] = []
        seen: set = set()
        for name in names:
            if not name or not name.strip():
                continue
            ident = cls.resolve(name=name.strip())
            if ident is not None and ident.id not in seen:
                seen.add(ident.id)
                results.append(ident)
        return results

    # =====================================================================
    # Cache management (class methods)
    # =====================================================================

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the entire identity map cache."""
        cls._cache.clear()
        cls._cache_timestamps.clear()
        logger.debug("Identity cache cleared")

    @classmethod
    def invalidate(cls, person_id: int) -> None:
        """Remove a specific person from the cache.

        The next ``get()`` call for this ID will reload from DB.

        Args:
            person_id: The person's database ID.
        """
        cls._cache.pop(person_id, None)
        cls._cache_timestamps.pop(person_id, None)

    @classmethod
    def preload(cls, person_ids: List[int]) -> List["Identity"]:
        """Preload multiple persons into the cache.

        Args:
            person_ids: List of person IDs to preload.

        Returns:
            List of loaded ``Identity`` instances (skips IDs not found).
        """
        results: List[Identity] = []
        for pid in person_ids:
            ident = cls.get(pid)
            if ident is not None:
                results.append(ident)
        return results

    # =====================================================================
    # Constructor (private — use factory methods)
    # =====================================================================

    def __init__(self, person_data: Dict[str, Any]) -> None:
        """Initialize from a person data dict (as returned by identity_db.get_person).

        Args:
            person_data: Dict with keys id, canonical_name, whatsapp_id,
                phone, email, is_group, display_name, and optionally
                facts, facts_detail, aliases, relationships, asset_counts.
        """
        self._id: int = person_data["id"]
        self._canonical_name: str = person_data.get("canonical_name", "")
        self._display_name: Optional[str] = person_data.get("display_name")
        self._whatsapp_id: Optional[str] = person_data.get("whatsapp_id")
        self._phone: Optional[str] = person_data.get("phone")
        self._email: Optional[str] = person_data.get("email")
        self._is_group: bool = bool(person_data.get("is_group", False))
        self._first_seen: Optional[str] = person_data.get("first_seen")
        self._last_seen: Optional[str] = person_data.get("last_seen")
        self._last_updated: Optional[str] = person_data.get("last_updated")
        self._confidence: Optional[float] = person_data.get("confidence")

        # Lazy-loaded collections — None means "not yet loaded"
        # If person_data already contains them (from get_person), pre-populate.
        self._facts: Optional[Dict[str, str]] = person_data.get("facts") if "facts" in person_data else None
        self._facts_detail: Optional[List[Dict[str, Any]]] = (
            person_data.get("facts_detail") if "facts_detail" in person_data else None
        )
        self._aliases: Optional[List[Dict[str, Any]]] = (
            person_data.get("aliases") if "aliases" in person_data else None
        )
        self._relationships: Optional[List[Dict[str, Any]]] = (
            person_data.get("relationships") if "relationships" in person_data else None
        )
        self._asset_counts: Optional[Dict[str, int]] = (
            person_data.get("asset_counts") if "asset_counts" in person_data else None
        )

    # =====================================================================
    # Core properties (eagerly loaded)
    # =====================================================================

    @property
    def id(self) -> int:
        """The person's database ID."""
        return self._id

    @property
    def name(self) -> str:
        """The person's canonical name."""
        return self._canonical_name

    @property
    def display_name(self) -> str:
        """Bilingual display name (e.g. ``"Shiran Waintrob / שירן וינטרוב"``).

        Falls back to ``name`` if no bilingual variant is available.
        """
        return self._display_name or self._canonical_name

    @property
    def whatsapp_id(self) -> Optional[str]:
        """WhatsApp contact ID (e.g. ``"972501234567@c.us"``)."""
        return self._whatsapp_id

    @property
    def phone(self) -> Optional[str]:
        """Phone number."""
        return self._phone

    @property
    def email(self) -> Optional[str]:
        """Email address."""
        return self._email

    @property
    def is_group(self) -> bool:
        """Whether this is a group entity (vs. individual person)."""
        return self._is_group

    @property
    def first_seen(self) -> Optional[str]:
        """Timestamp when this person was first seen."""
        return self._first_seen

    @property
    def last_seen(self) -> Optional[str]:
        """Timestamp when this person was last seen."""
        return self._last_seen

    @property
    def last_updated(self) -> Optional[str]:
        """Timestamp of the last update to this person's record."""
        return self._last_updated

    # =====================================================================
    # Lazy-loaded collection properties
    # =====================================================================

    @property
    def facts(self) -> Dict[str, str]:
        """All facts as a ``{key: value}`` dict. Lazy-loaded and cached."""
        if self._facts is None:
            import identity_db
            self._facts = identity_db.get_all_facts(self._id)
        return self._facts

    @property
    def facts_detail(self) -> List[Dict[str, Any]]:
        """Full fact records with metadata (confidence, source, quote). Lazy-loaded."""
        if self._facts_detail is None:
            # Reload full person data to get facts_detail
            import identity_db
            data = identity_db.get_person(self._id)
            if data and "facts_detail" in data:
                self._facts_detail = data["facts_detail"]
            else:
                self._facts_detail = []
        return self._facts_detail or []

    @property
    def aliases(self) -> List[Dict[str, Any]]:
        """Alias records (each with ``alias``, ``script``, ``source``). Lazy-loaded."""
        if self._aliases is None:
            import identity_db
            data = identity_db.get_person(self._id)
            if data and "aliases" in data:
                self._aliases = data["aliases"]
            else:
                self._aliases = []
        return self._aliases or []

    @property
    def alias_names(self) -> List[str]:
        """Flat list of alias strings (convenience accessor)."""
        return [a.get("alias", "") for a in self.aliases if a.get("alias")]

    @property
    def relationships(self) -> List[Dict[str, Any]]:
        """Relationship records with related person names. Lazy-loaded."""
        if self._relationships is None:
            import identity_db
            self._relationships = identity_db.get_relationships(self._id)
        return self._relationships

    @property
    def asset_counts(self) -> Dict[str, int]:
        """Asset counts by type (e.g. ``{"whatsapp_msg": 42, "document": 3}``). Lazy-loaded."""
        if self._asset_counts is None:
            import identity_db
            self._asset_counts = identity_db.get_person_asset_count(self._id)
        return self._asset_counts

    # =====================================================================
    # Fact operations
    # =====================================================================

    def get_fact(self, key: str) -> Optional[str]:
        """Get a single fact value by key.

        Uses the cached ``facts`` dict if available, otherwise reads from DB.

        Args:
            key: Fact key (e.g. ``"birth_date"``, ``"city"``).

        Returns:
            The fact value string, or ``None``.
        """
        if self._facts is not None:
            return self._facts.get(key)
        import identity_db
        return identity_db.get_fact(self._id, key)

    def set_fact(
        self,
        key: str,
        value: str,
        confidence: float = 0.5,
        source_type: str = "extracted",
        source_ref: Optional[str] = None,
        source_quote: Optional[str] = None,
    ) -> None:
        """Upsert a fact for this person.

        Higher confidence overwrites lower confidence; equal confidence
        updates the value (newer wins).

        Args:
            key: Fact key.
            value: Fact value.
            confidence: Confidence score (0.0–1.0).
            source_type: Source type (``"whatsapp"``, ``"paperless"``, ``"manual"``).
            source_ref: Reference to source.
            source_quote: Original text snippet.
        """
        import identity_db
        identity_db.set_fact(
            person_id=self._id,
            key=key,
            value=value,
            confidence=confidence,
            source_type=source_type,
            source_ref=source_ref,
            source_quote=source_quote,
        )
        # Invalidate local fact caches
        self._facts = None
        self._facts_detail = None

    def delete_fact(self, key: str) -> bool:
        """Delete a fact by key.

        Args:
            key: Fact key to delete.

        Returns:
            ``True`` if the fact was deleted.
        """
        import identity_db
        result = identity_db.delete_fact(self._id, key)
        if result:
            self._facts = None
            self._facts_detail = None
        return result

    # =====================================================================
    # Alias operations
    # =====================================================================

    def add_alias(
        self,
        alias: str,
        script: Optional[str] = None,
        source: str = "auto",
    ) -> bool:
        """Add a name alias.

        Args:
            alias: The alias text (e.g. ``"שירן"``, ``"Shiran"``).
            script: Script type — auto-detected if ``None``.
            source: Where this alias came from.

        Returns:
            ``True`` if the alias was added.
        """
        import identity_db
        result = identity_db.add_alias(
            person_id=self._id,
            alias=alias,
            script=script,
            source=source,
        )
        if result:
            self._aliases = None
            self._display_name = None  # May change with new alias
        return result

    def delete_alias(self, alias_id: int) -> bool:
        """Delete an alias by its row ID.

        Args:
            alias_id: The alias row ID.

        Returns:
            ``True`` if the alias was deleted.
        """
        import identity_db
        result = identity_db.delete_alias(alias_id)
        if result:
            self._aliases = None
            self._display_name = None
        return result

    # =====================================================================
    # Relationship operations
    # =====================================================================

    def add_relationship(
        self,
        related: Union["Identity", int],
        rel_type: str,
        confidence: float = 0.5,
        source_ref: Optional[str] = None,
    ) -> bool:
        """Add a relationship to another person.

        Args:
            related: The related person (``Identity`` instance or person ID).
            rel_type: Relationship type (e.g. ``"spouse"``, ``"parent"``).
            confidence: Confidence score.
            source_ref: Source reference.

        Returns:
            ``True`` if the relationship was added.
        """
        related_id = related.id if isinstance(related, Identity) else related
        import identity_db
        result = identity_db.add_relationship(
            person_id=self._id,
            related_person_id=related_id,
            relationship_type=rel_type,
            confidence=confidence,
            source_ref=source_ref,
        )
        if result:
            self._relationships = None
            # Also invalidate the related person's cache
            if isinstance(related, Identity):
                related._relationships = None
            elif related_id in Identity._cache:
                Identity._cache[related_id]._relationships = None
        return result

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships (delegates to the ``relationships`` property).

        Returns:
            List of relationship dicts with related person name.
        """
        return self.relationships

    def expand_related(self, max_depth: int = 1) -> List["Identity"]:
        """Expand this person's identity graph via relationships.

        Traverses relationships up to ``max_depth`` hops and returns
        all connected persons (including self).

        Args:
            max_depth: Number of relationship hops to follow.

        Returns:
            List of ``Identity`` instances (includes self).
        """
        import identity_db
        expanded_ids = identity_db.expand_person_ids_with_relationships(
            [self._id], max_depth=max_depth,
        )
        return Identity.preload(expanded_ids)

    # =====================================================================
    # Mutation operations
    # =====================================================================

    def rename(self, new_name: str) -> Optional[str]:
        """Rename this person's canonical name.

        Args:
            new_name: New canonical name.

        Returns:
            The new name if updated, ``None`` if failed (not found or conflicts).
        """
        import identity_db
        result = identity_db.rename_person(self._id, new_name)
        if result:
            self._canonical_name = result
            self._display_name = None  # Will be recomputed
        return result

    def merge_from(self, sources: List[Union["Identity", int]]) -> Dict[str, Any]:
        """Merge other persons into this identity.

        Absorbs aliases, facts, relationships, and identifiers from
        source persons, then deletes them.

        Args:
            sources: List of source ``Identity`` instances or person IDs.

        Returns:
            Merge summary dict from ``identity_db.merge_persons()``.
        """
        source_ids = [
            s.id if isinstance(s, Identity) else s
            for s in sources
        ]
        import identity_db
        result = identity_db.merge_persons(self._id, source_ids)

        # Remove merged sources from cache
        for sid in source_ids:
            Identity._cache.pop(sid, None)
            Identity._cache_timestamps.pop(sid, None)

        # Refresh self — everything may have changed
        self.refresh()

        return result

    def delete(self) -> bool:
        """Delete this person and all associated data.

        Returns:
            ``True`` if the person was deleted.
        """
        import identity_db
        result = identity_db.delete_person(self._id)
        if result:
            Identity._cache.pop(self._id, None)
            Identity._cache_timestamps.pop(self._id, None)
        return result

    def refresh(self) -> None:
        """Force reload all data from the database.

        Clears all lazy caches and reloads the core record.
        """
        import identity_db
        data = identity_db.get_person(self._id)
        if data is None:
            # Person was deleted
            Identity._cache.pop(self._id, None)
            Identity._cache_timestamps.pop(self._id, None)
            return

        self._canonical_name = data.get("canonical_name", "")
        self._display_name = data.get("display_name")
        self._whatsapp_id = data.get("whatsapp_id")
        self._phone = data.get("phone")
        self._email = data.get("email")
        self._is_group = bool(data.get("is_group", False))
        self._first_seen = data.get("first_seen")
        self._last_seen = data.get("last_seen")
        self._last_updated = data.get("last_updated")
        self._confidence = data.get("confidence")

        # Re-populate lazy caches from full data
        self._facts = data.get("facts") if "facts" in data else None
        self._facts_detail = data.get("facts_detail") if "facts_detail" in data else None
        self._aliases = data.get("aliases") if "aliases" in data else None
        self._relationships = data.get("relationships") if "relationships" in data else None
        self._asset_counts = data.get("asset_counts") if "asset_counts" in data else None

        # Update cache timestamp
        Identity._cache_timestamps[self._id] = time.monotonic()

    # =====================================================================
    # Context and serialization
    # =====================================================================

    def context_string(self) -> str:
        """Build a concise context string for system prompt injection.

        E.g. ``"Shiran Waintrob (שירן): female, born 1994-03-15, Tel Aviv"``

        Returns:
            Context string summarizing this person.
        """
        import identity_db
        result = identity_db.get_person_context(self._canonical_name)
        return result or self.display_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a full dict (triggers lazy loading of all collections).

        Returns:
            Dict with all person data including facts, aliases, relationships.
        """
        return {
            "id": self._id,
            "canonical_name": self._canonical_name,
            "display_name": self.display_name,
            "whatsapp_id": self._whatsapp_id,
            "phone": self._phone,
            "email": self._email,
            "is_group": self._is_group,
            "first_seen": self._first_seen,
            "last_seen": self._last_seen,
            "last_updated": self._last_updated,
            "confidence": self._confidence,
            "facts": self.facts,
            "facts_detail": self.facts_detail,
            "aliases": self.aliases,
            "relationships": self.relationships,
            "asset_counts": self.asset_counts,
        }

    # =====================================================================
    # Dunder methods
    # =====================================================================

    def __repr__(self) -> str:
        return f"Identity(id={self._id}, name={self._canonical_name!r})"

    def __str__(self) -> str:
        return self.display_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Identity):
            return self._id == other._id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._id)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @classmethod
    def _wrap(cls, person_data: Dict[str, Any]) -> "Identity":
        """Wrap person data into a cached Identity instance.

        If an instance for this person_id already exists in the cache,
        it is refreshed in-place rather than replaced.

        Args:
            person_data: Full person dict from ``identity_db.get_person()``.

        Returns:
            Cached ``Identity`` instance.
        """
        pid = person_data["id"]
        existing = cls._cache.get(pid)
        if existing is not None:
            # Update the existing instance in-place
            existing._canonical_name = person_data.get("canonical_name", "")
            existing._display_name = person_data.get("display_name")
            existing._whatsapp_id = person_data.get("whatsapp_id")
            existing._phone = person_data.get("phone")
            existing._email = person_data.get("email")
            existing._is_group = bool(person_data.get("is_group", False))
            existing._first_seen = person_data.get("first_seen")
            existing._last_seen = person_data.get("last_seen")
            existing._last_updated = person_data.get("last_updated")
            existing._confidence = person_data.get("confidence")
            existing._facts = person_data.get("facts") if "facts" in person_data else None
            existing._facts_detail = person_data.get("facts_detail") if "facts_detail" in person_data else None
            existing._aliases = person_data.get("aliases") if "aliases" in person_data else None
            existing._relationships = person_data.get("relationships") if "relationships" in person_data else None
            existing._asset_counts = person_data.get("asset_counts") if "asset_counts" in person_data else None
            cls._cache_timestamps[pid] = time.monotonic()
            return existing

        instance = cls(person_data)
        cls._cache[pid] = instance
        cls._cache_timestamps[pid] = time.monotonic()
        return instance

    @classmethod
    def _wrap_summary(cls, summary_data: Dict[str, Any]) -> "Identity":
        """Wrap summary data (from get_all_persons_summary) into a cached instance.

        Summary data is lighter than full person data — it includes
        facts as a dict but not facts_detail, and aliases as a flat list
        of strings rather than dicts with metadata.

        If a full instance already exists in cache, returns it unchanged.

        Args:
            summary_data: Person summary dict from ``identity_db.get_all_persons_summary()``.

        Returns:
            Cached ``Identity`` instance.
        """
        pid = summary_data.get("id")
        if pid is None:
            raise ValueError("Summary data missing 'id' key")

        existing = cls._cache.get(pid)
        if existing is not None and cls._is_fresh(pid):
            return existing

        # Convert summary aliases (flat string list) to the dict format
        raw_aliases = summary_data.get("aliases", [])
        if raw_aliases and isinstance(raw_aliases[0], str):
            alias_dicts = [{"alias": a, "script": "unknown", "source": "summary"} for a in raw_aliases]
        else:
            alias_dicts = raw_aliases

        # Build a person_data-like dict
        person_data: Dict[str, Any] = {
            "id": pid,
            "canonical_name": summary_data.get("canonical_name", ""),
            "display_name": summary_data.get("display_name"),
            "whatsapp_id": summary_data.get("whatsapp_id"),
            "phone": summary_data.get("phone"),
            "email": summary_data.get("email"),
            "is_group": summary_data.get("is_group", False),
            "last_seen": summary_data.get("last_seen"),
            "facts": summary_data.get("facts", {}),
            "aliases": alias_dicts,
        }

        return cls._wrap(person_data)

    @classmethod
    def _is_fresh(cls, person_id: int) -> bool:
        """Check if a cached entry is still within the TTL.

        Args:
            person_id: Person ID to check.

        Returns:
            ``True`` if the cache entry is fresh (or TTL is disabled).
        """
        if cls.CACHE_TTL <= 0:
            return True
        ts = cls._cache_timestamps.get(person_id)
        if ts is None:
            return False
        return (time.monotonic() - ts) < cls.CACHE_TTL
