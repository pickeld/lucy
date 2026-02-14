# Lucy

**Lucy** is a personal AI assistant powered by LLM models (OpenAI, Gemini) with a plugin-based architecture. It features a RAG (Retrieval-Augmented Generation) knowledge base backed by Qdrant, a Reflex web UI for chat and settings management, and extensible plugins for WhatsApp, Paperless-ngx, and more.

## Key Features

### AI & RAG
- **Multi-Provider LLM**: OpenAI GPT and Google Gemini support
- **Knowledge Base**: LlamaIndex RAG with Qdrant vector store for contextual conversations
- **Image Generation**: DALL-E and Imagen support
- **Context Management**: Intelligent conversation memory and state tracking

### Plugin System
- **WhatsApp Plugin**: Full WhatsApp integration via WAHA API — message handling, groups, contacts
- **Paperless Plugin**: Paperless-ngx document ingestion and search
- **Extensible**: Easy-to-build plugin architecture for new integrations

### Web UI (Reflex)
- **Chat Interface**: Conversational RAG Q&A with previous chat history
- **Settings Management**: Full configuration UI with import/export
- **Cost Tracking**: Real-time LLM cost monitoring per request, session, and daily

### Technical Features
- **Redis Backend**: Efficient caching and state management
- **Docker Support**: Containerized deployment with Docker Compose
- **SQLite Settings**: Database-backed configuration (seeded from `.env` on first run)
- **Comprehensive Logging**: Detailed system logging and monitoring

## Project Structure

```
lucy/
├── src/                          # Main application code
│   ├── app.py                    # Flask app: webhooks, RAG, config & cache API endpoints
│   ├── config.py                 # Settings class — reads from SQLite via settings_db
│   ├── settings_db.py            # SQLite-backed settings database (seeds from .env on first run)
│   ├── llamaindex_rag.py         # LlamaIndex RAG with Qdrant + CondensePlusContextChatEngine
│   ├── cost_meter.py             # LLM cost tracking and metering
│   ├── models/                   # Pydantic v2 document models for RAG
│   │   ├── base.py               # BaseRAGDocument, DocumentMetadata, SourceType
│   │   ├── whatsapp.py           # WhatsAppMessageDocument
│   │   ├── document.py           # FileDocument (PDF, DOCX, etc.)
│   │   └── call_recording.py     # CallRecordingDocument
│   ├── plugins/                  # Plugin system
│   │   ├── base.py               # Base plugin interface
│   │   ├── registry.py           # Plugin discovery and registration
│   │   ├── whatsapp/             # WhatsApp integration plugin
│   │   └── paperless/            # Paperless-ngx integration plugin
│   └── utils/                    # Utility modules
│       ├── exceptions.py         # Custom exception hierarchy
│       ├── globals.py            # HTTP helpers with retry logic
│       ├── logger.py             # Logging configuration
│       └── redis_conn.py         # Redis connection management
├── ui-reflex/                    # Reflex web UI
│   ├── ui_reflex/
│   │   ├── ui_reflex.py          # Main UI layout
│   │   ├── state.py              # Application state management
│   │   ├── api_client.py         # Backend API client
│   │   └── components/           # UI components (chat, sidebar, settings, etc.)
│   └── rxconfig.py               # Reflex configuration
├── data/                         # SQLite database (gitignored, auto-created)
├── plans/                        # Architecture & improvement plans
├── scripts/                      # Utility scripts
├── tests/                        # End-to-end tests
├── docker-compose.yml            # Redis, WAHA, Qdrant, App services
├── Dockerfile                    # Application container
├── .env.example                  # First-run seed for settings database
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Docker**: For running Redis, Qdrant, and optional WAHA services
- **OpenAI API Key** and/or **Google API Key**: For LLM access

### System Requirements

- **Memory**: Minimum 4GB RAM recommended
- **Storage**: At least 1GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/pickeld/lucy.git
   cd lucy
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your credentials.

4. **Start infrastructure services:**
   ```sh
   docker-compose up -d
   ```
   To also start the WhatsApp plugin (WAHA):
   ```sh
   docker compose --profile whatsapp up -d
   ```

5. **Run the application:**
   ```sh
   PYTHONPATH=src python src/app.py
   ```

6. **Launch the Reflex UI** (optional):
   ```sh
   cd ui-reflex && reflex run
   ```

## Configuration

### Environment Configuration

The `.env` file is used **only on first startup** to seed the SQLite settings database. After that, all configuration is managed through the Settings UI page or the `PUT /config` API endpoint.

```env
# See .env.example for all available settings
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
WAHA_API_KEY=your-waha-api-key
REDIS_HOST=localhost
QDRANT_HOST=localhost
```

See `.env.example` for a complete list of available configuration options.

## Development

### Development Setup

1. **Create a virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Start required services:**
   ```sh
   docker-compose up -d
   ```

4. **Run the application in debug mode:**
   ```sh
   PYTHONPATH=src python src/app.py
   ```

## License

MIT License

---

**Note:** This project is for educational and prototyping purposes. Use responsibly and comply with all applicable API terms of service.
