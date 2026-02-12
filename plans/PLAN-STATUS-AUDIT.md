# Plan Status Audit — 2026-02-12

## Summary

| # | Plan File | Status | Action |
|---|-----------|--------|--------|
| 1 | `context-and-reasoning-architecture.md` | ✅ SUPERSEDED & APPLIED | **DELETE** |
| 2 | `llamaindex-chat-engine-migration.md` | ✅ FULLY APPLIED | **DELETE** |
| 3 | `rag-document-classes.md` | ✅ FULLY APPLIED | **DELETE** |
| 4 | `remove-threads-manager.md` | ✅ FULLY APPLIED | **DELETE** |
| 5 | `settings-page-plan.md` | ✅ FULLY APPLIED | **DELETE** |
| 6 | `whatsapp-msg-json-schema.md` | ✅ FULLY APPLIED | **DELETE** |
| 7 | `current-improvements-plan.md` | ⚠️ PARTIALLY APPLIED | **KEEP — has pending work** |
| 8 | `improvements-and-features.md` | ⚠️ PARTIALLY APPLIED | **KEEP — has pending work** |
| 9 | `rag-optimization-plan.md` | ⚠️ PARTIALLY APPLIED | **KEEP — has pending work** |
| 10 | `ui-improvements.md` | ⚠️ MOSTLY NOT APPLIED | **KEEP — has pending work** |

---

## Detailed Analysis

### ✅ Plans to DELETE (Fully Applied)

#### 1. `context-and-reasoning-architecture.md`
- Plan itself marked **SUPERSEDED** at the top
- Custom `src/session/` module deleted
- Replaced by LlamaIndex `CondensePlusContextChatEngine` + `RedisChatStore`
- Query reformulation handled by chat engine's condense step

#### 2. `llamaindex-chat-engine-migration.md`
- `llama-index-storage-chat-store-redis` in `requirements.txt` ✅
- `WhatsAppRetriever` class in `llamaindex_rag.py` ✅
- `create_chat_engine()` method with `CondensePlusContextChatEngine` ✅
- `chat_store` property with `RedisChatStore` ✅
- `ChatMemoryBuffer` with token limit ✅
- `/rag/query` simplified to use `chat_engine.chat()` ✅
- All `/session/*` endpoints removed ✅
- `src/session/` directory deleted ✅
- Filter state via Redis hash in `app.py` ✅

#### 3. `rag-document-classes.md`
- `SourceType` enum in `models/base.py` ✅
- `DocumentMetadata` in `models/base.py` ✅
- `BaseRAGDocument` abstract class in `models/base.py` ✅
- `WhatsAppMessageDocument` in `models/whatsapp.py` ✅
- `FileDocument` in `models/document.py` ✅
- `CallRecordingDocument` in `models/call_recording.py` ✅
- `__init__.py` package exports ✅
- `add_document()` / `add_documents()` in `llamaindex_rag.py` ✅
- Note: Document loaders were listed as "Pending" in the plan itself

#### 4. `remove-threads-manager.md`
- `src/conversation_memory.py` deleted ✅
- No ThreadsManager import/usage in `app.py` ✅
- Direct `rag.add_message()` in webhook handler ✅
- No `/threads` endpoints ✅

#### 5. `settings-page-plan.md`
- `src/settings_db.py` with full SQLite CRUD ✅
- `config.py` refactored to `Settings.__getattr__` → SQLite ✅
- `GET/PUT /config`, `GET /config/categories`, `POST /config/reset` endpoints ✅
- Enhanced `/health` checks Redis, Qdrant, WAHA ✅
- `ui/pages/1_Settings.py` with full settings form ✅
- `.env.example` updated with "FIRST-RUN SEED ONLY" comment ✅
- `.gitignore` has `data/` ✅
- `docker-compose.yml` has `./data:/app/data` volume mount ✅

#### 6. `whatsapp-msg-json-schema.md`
- Full class hierarchy: `WhatsappMSG` → `TextMessage`, `ImageMessage`, `VoiceMessage`, etc. ✅
- `ContentType` enum ✅
- `create_whatsapp_message()` factory function ✅
- `to_json()` on all subclasses ✅
- `to_rag_document()` on base and media classes ✅

---

### ⚠️ Plans to KEEP (Partially Applied)

#### 7. `current-improvements-plan.md`

**Done:**
- 1.1 Fix `pass_filter()` null safety ✅
- 1.2 Fix Dockerfile port → 8765 ✅
- 1.3 Enable app in docker-compose ✅
- 1.5 PostgreSQL commented out ✅
- 2.1 Cache chat/sender lists in Redis ✅
- 2.2 Upgraded to `text-embedding-3-large` ✅
- 2.3 `@st.cache_data` in UI ✅
- 2.4 Using python-dotenv ✅
- 2.5 Logger uses format specifiers ✅
- 3.5 Query reformulation via chat engine ✅
- 4.2 Message deduplication ✅
- 4.3 Dependency health checks ✅

**Still Pending:**
- 1.4 README project structure partially outdated — still references `src/session/` and `localhost:5002`
- 3.1 AI response to WhatsApp via `??` prefix — no prefix detection in webhook
- 3.2 Voice message transcription — `VoiceMessage.transcription` always None
- 3.3 Image description — `ImageMessage.description` always None
- 3.6 Tavily web search fallback — not implemented
- 4.1 Async webhook processing — still synchronous
- 4.4 Rate limiting — no Flask-Limiter
- 4.5 Standardize singleton patterns — still mixed
- 5.1 Source citations in RAG answers — UI doesn't show sources
- 5.2 Shared session between UI tabs — not implemented
- 5.3 Message browser page — not implemented

#### 8. `improvements-and-features.md`

**Done:**
- Error handling / custom exceptions ✅
- Type hints and docstrings ✅
- Renamed `utiles` → `utils` ✅
- Fixed config loading ✅
- Fixed RAG singleton ✅

**Still Pending:**
- Circuit breaker pattern
- API documentation / OpenAPI
- Testing suite (pytest)
- Async/await migration
- Message queue integration
- Semantic caching
- Authentication / authorization
- Input validation with Pydantic request models
- AI response activation
- Voice message support
- Multi-modal AI
- Scheduled messages
- Conversation summarization
- Admin dashboard
- Local LLM support (Ollama)
- Observability (Prometheus, Grafana)
- Deployment improvements
- Backup and recovery

#### 9. `rag-optimization-plan.md`

**Done (Phase 1):**
- Issue 2: Lean embedding text ✅
- Issue 3: Upgraded embedding model ✅
- Issue 4: RRF hybrid search merge ✅
- Issue 5: Score threshold filter ✅
- Issue 7: Payload indexes ✅
- Issue 8: Message deduplication ✅
- Issue 12: Metadata-only search path ✅
- Issue 13: Redis cache TTL refresh ✅
- Issue 14: Enhanced LLM prompt ✅

**Still Pending:**
- Issue 1: Conversation chunking / sliding window — not implemented
- Issue 6: Filter low-value messages from embedding
- Issue 10: Rename collection to `knowledge_base` — still `whatsapp_messages`
- Issue 11: Field-aware full-text search scoring

#### 10. `ui-improvements.md`

**Done:**
- `@st.cache_data` caching ✅
- Multi-page app structure started (Settings page) ✅
- Some custom CSS ✅

**Still Pending (almost everything):**
- WhatsApp-style message bubbles
- Dark/light mode toggle
- Message browser page
- Analytics dashboard
- Direct chat interface
- Context sources display
- Real-time message feed
- Configuration management (partially via Settings page)
- RAG collection management
- API client refactoring
- Component architecture
- Authentication
- Responsive layout
