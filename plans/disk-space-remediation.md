# Disk Space Exhaustion — Root Cause Analysis & Remediation Plan

## Summary

The application experienced cascading failures due to **Docker host disk space exhaustion**. This caused three dependent services — Qdrant, Redis, and the call recordings scanner — to fail, resulting in prolonged 503 health check responses and a dropped WhatsApp webhook message.

---

## Root Cause Analysis

```mermaid
graph TD
    A[Docker host disk full] --> B[Qdrant: WAL buffer overflow]
    A --> C[Redis: RDB snapshot failure]
    A --> D[Dropbox mount: file lock errors]
    B --> E[All index creation fails with 500]
    C --> F[Write operations blocked - MISCONF]
    F --> G[WhatsApp webhook message dropped]
    D --> H[All call recording scans fail - Errno 35]
    E --> I[/health returns 503]
    C --> I
    I --> J[Docker healthcheck fails]
    J --> K[UI service may not start - depends_on service_healthy]
```

### Issue 1: Qdrant — No Space Left on Device

**Error:** `Service internal error: No space left on device: WAL buffer size exceeds available disk space`

- Occurs on every startup in `_ensure_text_indexes` and `_ensure_payload_indexes`
- 11 different index creation attempts all fail with HTTP 500
- The Qdrant collection `lucy_data` itself still exists and can be queried
- Index creation is treated as non-fatal in the app code — logged at DEBUG level
- **Impact:** Degraded search performance, but not a hard failure for queries

### Issue 2: Redis — RDB Snapshot Failure

**Error:** `MISCONF Redis is configured to save RDB snapshots, but it is currently unable to persist to disk`

- Redis refuses all WRITE commands when `stop-writes-on-bgsave-error` is enabled (default)
- The WhatsApp webhook handler crashes trying to `redis_set` a contact alias
- The rate limiter (backed by Redis) may also fail
- **Impact:** Messages are dropped, contact resolution fails

### Issue 3: Call Recordings — Resource Deadlock (Errno 35)

**Error:** `[Errno 35] Resource deadlock avoided` on ~300+ `.m4a` files

- Files are mounted from Dropbox CloudStorage: `/Users/dpickel/Library/CloudStorage/Dropbox/call_recordings`
- Dropbox uses extended attributes and file locks for cloud sync
- When disk is full, Dropbox cannot complete sync operations, leaving files in a locked state
- The scanner already has a fallback for `OSError` in `compute_file_hash`, but `_extract_audio_metadata` via mutagen fails at file open
- **Impact:** Zero call recordings can be scanned or transcribed

### Issue 4: Health Check — Prolonged 503

**Timeline from logs:**
- `20:37:42` — First 503
- `20:44:26` — Redis MISCONF error on webhook
- `20:46:35` — First 200 (after ~9 minutes of 503s)

The `/health` endpoint checks Redis ping, Qdrant connectivity, and all plugin health checks. Any single failure returns 503, which causes:
- Docker healthcheck failure (affects `ui` service via `condition: service_healthy`)
- Monitoring/alerting noise

---

## Remediation Plan

### Phase 1: Immediate — Free Disk Space (Ops Task)

These are manual operations on the Docker host:

1. **Check current disk usage:**
   ```bash
   df -h
   docker system df
   ```

2. **Prune unused Docker resources:**
   ```bash
   docker system prune -a --volumes  # CAUTION: removes unused volumes
   # Or more targeted:
   docker image prune -a
   docker builder prune -a
   ```

3. **Check Docker volume sizes:**
   ```bash
   docker system df -v
   ```

4. **Inspect Qdrant storage specifically:**
   ```bash
   docker exec qdrant du -sh /qdrant/storage/
   ```

5. **Fix Redis immediately:**
   ```bash
   docker exec redis redis-cli CONFIG SET stop-writes-on-bgsave-error no
   ```
   Then once disk is freed:
   ```bash
   docker exec redis redis-cli BGSAVE
   docker exec redis redis-cli CONFIG SET stop-writes-on-bgsave-error yes
   ```

### Phase 2: Docker Compose Hardening

Update `docker-compose.yml` to prevent uncontrolled disk growth:

1. **Add Qdrant storage limits and WAL configuration:**
   ```yaml
   qdrant:
     environment:
       - QDRANT__STORAGE__WAL__WAL_CAPACITY_MB=64
       - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=2
   ```

2. **Add Redis memory and persistence configuration:**
   ```yaml
   redis:
     command: >
       redis-server
       --maxmemory 256mb
       --maxmemory-policy allkeys-lru
       --save 900 1
       --save 300 10
       --stop-writes-on-bgsave-error no
   ```

3. **Add Docker logging limits to all services:**
   ```yaml
   x-logging: &default-logging
     logging:
       driver: json-file
       options:
         max-size: "10m"
         max-file: "3"
   ```

### Phase 3: Health Check Improvements

Modify `/health` in `src/app.py` to be more resilient:

1. **Separate liveness from readiness:** Add a `/health/live` endpoint that always returns 200 if the process is running, and keep `/health` as the readiness probe
2. **Add disk space check** to the health endpoint to provide early warning
3. **Make the Docker healthcheck use the liveness probe** so the container doesn't get marked unhealthy due to transient dependency issues:
   ```yaml
   healthcheck:
     test: ["CMD", "python", "-c", "import requests; r=requests.get('http://localhost:8765/health/live', timeout=5); r.raise_for_status()"]
   ```

### Phase 4: Call Recordings Scanner & Transcriber Hardening

#### Scanner (`src/plugins/call_recordings/scanner.py`)

The scanner already handles `OSError` for content hashing but needs broader protection:

1. **Wrap the entire per-file scan in a more specific OSError handler** that captures Errno 35 specifically and logs at WARNING not via the generic exception path
2. **Add a scan summary** that reports how many files were skipped due to lock errors vs successfully scanned

#### Transcriber — Dropbox File Lock Fix (`src/plugins/call_recordings/sync.py`)

The transcription chain fails because ffmpeg cannot open Dropbox-locked files:

```
File mounted from Dropbox → Errno 35 lock → ffmpeg fails → Whisper fails
→ 400 returned → UI shows error → user retries rapidly → same failure 6+ times
```

**Root fix — copy-before-transcribe:**

In `transcribe_file()` in `sync.py`, before calling `self.transcriber.transcribe()`:

1. **Pre-check file readability** by attempting to open the file with `open(file_path, 'rb')`
2. If `OSError` with errno 35 (EDEADLK), **copy the file to a temp directory** using `shutil.copy2()` to a path like `/tmp/call_rec_{hash}.{ext}`
3. Pass the temp copy path to `self.transcriber.transcribe()` instead
4. Clean up the temp file after transcription completes or fails
5. If the copy itself fails (still locked), return a specific `"error_type": "file_locked"` in the response so the UI can display a user-friendly message like *File is being synced by Dropbox — try again later*

#### Transcriber (`src/plugins/call_recordings/transcriber.py`)

Suppress the noisy Whisper FP16 warning by passing `fp16=False` to `model.transcribe()` — harmless but pollutes logs on every call.

### Phase 5: Monitoring & Prevention

1. **Add a disk space check** to the health endpoint:
   ```python
   import shutil
   usage = shutil.disk_usage("/")
   free_pct = usage.free / usage.total * 100
   if free_pct < 10:
       status["dependencies"]["disk"] = f"warning: {free_pct:.1f}% free"
   ```

2. **Add Qdrant collection size** to `/rag/stats` endpoint for visibility in the UI

3. **Consider volume size limits** in Docker if the host supports it

---

## Files to Modify

| File | Change |
|------|--------|
| `docker-compose.yml` | Redis config, Qdrant WAL limits, logging limits, healthcheck update |
| `src/app.py` | Add `/health/live` endpoint, disk space warning in `/health` |
| `src/plugins/call_recordings/scanner.py` | Better Errno 35 handling, scan summary with skip counts |
| `src/plugins/call_recordings/sync.py` | Copy-before-transcribe for Dropbox-locked files, `file_locked` error type |
| `src/plugins/call_recordings/transcriber.py` | Pass `fp16=False` to suppress CPU warning |

## Priority Order

1. **Free disk space now** (manual ops)
2. **Fix Redis MISCONF** (one command)
3. **Docker compose hardening** (prevents recurrence)
4. **Transcriber Dropbox lock fix** (files can't be transcribed at all without this)
5. **Health check improvements** (reduces blast radius)
6. **Scanner improvements** (quality of life)
7. **Monitoring** (early warning)
