# Contact Disambiguation Plan

## Problem

When a user asks "מה דורון שאל אותי?" (What did Doron ask me?), Lucy returns results for **דורון עלאני** (from Paperless documents) instead of **Doron Yazkirovich** (from WhatsApp). This happens because:

1. **Cross-language name mismatch**: The Hebrew token "דורון" matches "דורון עלאני" via fulltext search (score 0.95), but doesn't match the English-spelled "Doron Yazkirovich" in the `sender` field
2. **No disambiguation**: The LLM has no awareness of the full contact/sender list, so it can't detect that "Doron" is ambiguous and ask the user to clarify

### Expected Behavior
When a name in the query matches multiple people in the archive, Lucy should ask a clarifying question like:
> "I found multiple people named Doron in your archive. Did you mean:
> 1. Doron Yazkirovich
> 2. דורון עלאני
>
> Please specify which one."

## Solution: LLM-Native Disambiguation (No Custom Code)

Instead of building custom transliteration/matching code, we leverage the LLM's built-in multilingual understanding. GPT-4o/Gemini natively know that "דורון" = "Doron" across scripts.

### Changes Made

**1. System Prompt Enhancement** (`src/settings_db.py`)
- Added instruction #9 to the default system prompt template telling the LLM to ask clarifying questions when a first name matches multiple people in the known contacts list
- The instruction covers cross-script awareness (Hebrew ↔ English names)

**2. Dynamic Contact List Injection** (`src/llamaindex_rag.py`)
- Modified `_build_system_prompt()` to append the full sender list (fetched dynamically from `get_sender_list()` → Redis cache) after the template is formatted
- Format: `Known Contacts (N people): Name1, Name2, ...`
- Token cost: ~2K tokens for 500 contacts — negligible (~$0.005/query at GPT-4o pricing)
- Falls back gracefully if the sender list can't be fetched

### How It Works

```
User: "מה דורון שאל אותי?"

System Prompt includes:
  - Instruction #9: "When a name matches multiple people, ask to clarify..."
  - Known Contacts: "..., Doron Yazkirovich, ..., דורון עלאני, ..."

LLM sees "דורון" in the query → recognizes it matches both
  "Doron Yazkirovich" and "דורון עלאני" in the contacts list →
  asks: "I found multiple people named דורון/Doron:
    1) Doron Yazkirovich
    2) דורון עלאני
  Which one did you mean?"

User: "doron yazkirovitch"

LLM now searches with the disambiguated name → finds the correct results
```

### Files Modified

| File | Change |
|------|--------|
| `src/llamaindex_rag.py` | `_build_system_prompt()` — appends dynamic sender list from Redis cache |
| `src/settings_db.py` | `_DEFAULT_SYSTEM_PROMPT` — added disambiguation instruction #9 |

### Why This Approach

- **No custom transliteration code** — the LLM already handles Hebrew ↔ English name matching natively
- **No name_matcher.py** — no new utility files needed
- **Minimal code changes** — only 2 files modified, ~20 lines added
- **Self-maintaining** — as new contacts are added to the archive, they automatically appear in the sender list via Redis cache
- **Conversation-aware** — the LLM can use conversation history to resolve ambiguity without asking (e.g., if the user mentioned "Doron Yazkirovich" in a previous turn)
