# Source Display Cleanup Plan

## Problem

1. **Messy source formatting** — The context shown to the LLM (and by extension, Lucy's responses) currently formats sources inconsistently:
   ```
   [WhatsApp | 31/12/2024 10:30] sender in chat_name: message
   [Paperless | 15/02/2026 17:00] Document 'Title':
   document text...
   ```

2. **Wrong date for Paperless documents** — The `timestamp` field is set to `int(time.time())` (the sync time), not the document's actual creation date from Paperless.

## Desired Format

```
SOURCE | DATE | item text
```

Examples:
```
WHATSAPP | 31/12/2024 10:30 | David in Family Group: Hey everyone
PAPERLESS | 15/03/2023 | Document 'Divorce Agreement': content...
GMAIL | 01/01/2025 14:22 | user@email.com: Subject line - body...
```

## Changes Required

### 1. Fix Paperless timestamp (src/plugins/paperless/sync.py)

**File:** `src/plugins/paperless/sync.py`, line 443  
**Current:** `"timestamp": int(time.time())`  
**Fix:** Parse `doc["created"]` (ISO 8601 string like `"2023-03-15T00:00:00+02:00"`) into a Unix timestamp and use that instead. Fall back to `int(time.time())` only if the `created` field is missing or unparseable.

```python
# Parse document creation date from Paperless API
created_str = doc.get("created", "")
if created_str:
    try:
        from datetime import datetime as _dt
        created_dt = _dt.fromisoformat(created_str)
        doc_timestamp = int(created_dt.timestamp())
    except (ValueError, TypeError):
        doc_timestamp = int(time.time())
else:
    doc_timestamp = int(time.time())

base_metadata = {
    ...
    "timestamp": doc_timestamp,
    ...
}
```

### 2. Reformat `_extract_text_from_payload()` (src/llamaindex_rag.py)

**File:** `src/llamaindex_rag.py`, lines 1024-1092  
**Method:** `LlamaIndexRAG._extract_text_from_payload()`

Change ALL return paths to use the new `SOURCE | DATE | text` format:

| Return path | Current format | New format |
|---|---|---|
| Paperless with `_node_content` + sender | `[Paperless \| date] sender in chat: text` | `PAPERLESS \| date \| sender in chat: text` |
| Paperless with `_node_content` no sender | `[Paperless \| date] Document 'chat': text` | `PAPERLESS \| date \| Document 'chat': text` |
| WhatsApp / fallback with `message` | `[WhatsApp \| date] sender in chat: msg` | `WHATSAPP \| date \| sender in chat: msg` |
| Generic with `_node_content` + sender | `[Source \| date] sender in chat: text` | `SOURCE \| date \| sender in chat: text` |
| Generic with `_node_content` no sender | `[Source \| date] Document 'chat': text` | `SOURCE \| date \| Document 'chat': text` |

The source labels should be **UPPERCASE** to match the user's preference.

### 3. Update `_SOURCE_LABELS` dict (src/llamaindex_rag.py)

**File:** `src/llamaindex_rag.py`, lines 1010-1022

Change labels to uppercase:
```python
_SOURCE_LABELS: Dict[str, str] = {
    "whatsapp": "WHATSAPP",
    "paperless": "PAPERLESS",
    "gmail": "GMAIL",
    "telegram": "TELEGRAM",
    "email": "EMAIL",
    ...
}
```

### 4. Note on existing data

**Already-synced Paperless documents** will still have `timestamp = sync_time` in Qdrant. The only way to fix these is to re-sync Paperless (`force=True`). This is not a code change — just a manual step the user will need to run after deploying the fix.

## Files Modified

| File | Change |
|---|---|
| `src/plugins/paperless/sync.py` | Parse `doc["created"]` into `timestamp` instead of `int(time.time())` |
| `src/llamaindex_rag.py` | Reformat `_extract_text_from_payload()` and update `_SOURCE_LABELS` |

## Testing Notes

- After deploying, run a Paperless force re-sync to update timestamps for existing documents
- WhatsApp and Gmail timestamps are already correct and need no re-sync
