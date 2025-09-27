# whatsapp-gpt

A Flask-based integration between WhatsApp (via the WAHA API) and OpenAI's GPT models. This project allows you to pair your WhatsApp account, receive webhook events, and interact with OpenAI's GPT for chat-based automation.

## Key Features

### WhatsApp Integration
- **Easy Pairing**: QR code-based WhatsApp account pairing
- **Message Handling**: Comprehensive webhook system for all message types
- **Group Support**: Full group chat management and automation
- **Contact Management**: Complete contact operations and tracking

### AI Capabilities
- **OpenAI Integration**: Advanced conversation handling with GPT models
- **DALL-E Support**: Image generation and processing capabilities
- **Context Management**: Intelligent conversation memory and state tracking
- **Template System**: Flexible response templating and formatting

### Technical Features
- **Redis Backend**: Efficient caching and state management
- **Docker Support**: Containerized deployment with Docker Compose
- **Modular Design**: Easily extensible provider system
- **Comprehensive Logging**: Detailed system logging and monitoring

### Media Handling
- **Image Generation**: DALL-E integration for image creation
- **Media Processing**: Support for various media types via WAHA API
- **File Sharing**: Document and media file sharing capabilities

## Project Structure

The project is organized as follows:

- **`src/app.py`**: Main Flask application for WhatsApp integration and webhook handling.
- **`src/config.py`**: Loads environment variables and configuration settings.
- **`src/contact.py`**: Manages WhatsApp contact-related functionality.
- **`src/groups.py`**: Handles WhatsApp group operations and management.
- **`src/memory_agent.py`**: Implements semantic memory and context management.
- **`src/templates.py`**: Contains message templates and response formatting.
- **`src/whatsapp.py`**: Core WhatsApp integration functionality.
- **`src/providers/`**: Contains provider-specific integrations:
  - `dalle.py`: DALL-E image generation integration
  - `prompts.py`: Manages system prompts and templates
- **`src/utiles/`**: Utility modules:
  - `classes.py`: Core class definitions
  - `globals.py`: Global variables and constants
  - `logger.py`: Logging configuration
  - `redis_conn.py`: Redis connection management
- **`docker-compose.yml`**: Sets up the WAHA WhatsApp API service.
- **`.env.example`**: Example environment configuration file.
- **`requirements.txt`**: Lists Python dependencies required for the project.
- **`README.md`**: Documentation for the project.

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Docker**: For running WAHA and Redis services
- **Redis**: Used for caching and state management
- **OpenAI API Key**: For GPT model access
- **WAHA API Key**: For WhatsApp integration (obtained after setup)

### System Requirements

- **Memory**: Minimum 4GB RAM recommended
- **Storage**: At least 1GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/whatsapp-gpt.git
   cd whatsapp-gpt
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your credentials.

4. **Start WAHA (WhatsApp API) via Docker:**
   ```sh
   docker-compose up -d
   ```

5. **Run the Flask app:**
   ```sh
   python app.py
   ```

6. **Pair WhatsApp:**
   - Visit `http://localhost:5002/pair` in your browser and scan the QR code with your WhatsApp app.

## Usage
### Basic Usage
- Incoming WhatsApp messages are automatically processed via the `/webhook` endpoint
- Messages are intelligently handled based on type (text, media, group, etc.)
- Responses are generated using OpenAI's GPT models and custom templates
- Support for both individual chats and group conversations

### Advanced Features
- Contact management via dedicated API endpoints
- Group creation and management capabilities
- Template-based response system
- Redis-backed state management
- Extensible provider system for AI services


## Configuration

### Environment Configuration

The project uses environment variables for configuration, managed via a `.env` file. Required configurations include:

#### Core Settings
```env
# WhatsApp API Configuration
WAHA_API_URL=http://localhost:3000
WAHA_API_KEY=your-waha-api-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application Settings
LOG_LEVEL=DEBUG
FLASK_ENV=development
PORT=5002
```

See `.env.example` for a complete list of available configuration options. Make sure to:
1. Copy `.env.example` to `.env`
2. Update the values according to your setup
3. Never commit your `.env` file to version control
## Development

### Project Organization

The project follows a modular architecture for maintainability and scalability:

```
whatsapp-gpt/
├── src/                    # Main application code
│   ├── app.py             # Flask application entry point
│   ├── config.py          # Configuration management
│   ├── contact.py         # Contact operations
│   ├── groups.py          # Group management
│   ├── memory_agent.py    # Context management
│   ├── templates.py       # Message templates
│   ├── whatsapp.py        # WhatsApp core functionality
│   ├── providers/         # External service integrations
│   │   ├── dalle.py       # DALL-E integration
│   │   └── prompts.py     # System prompts
│   └── utiles/            # Utility modules
│       ├── classes.py     # Core classes
│       ├── globals.py     # Global variables
│       ├── logger.py      # Logging setup
│       └── redis_conn.py  # Redis connection
├── helpers/               # Helper scripts
├── scripts/               # Deployment scripts
└── docker-compose.yml     # Container orchestration
```

### Development Setup

1. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Start required services:**
   ```sh
   docker-compose up -d
   ```

4. **Run the application in debug mode:**
   ```sh
   python src/app.py
   ```


## Project Overview

This project integrates WhatsApp messaging with OpenAI's GPT models using Flask. It enables automated responses and webhook handling for WhatsApp messages. Key components include:

### Core Components
- **WAHA API**: Provides WhatsApp integration via a robust API layer
- **OpenAI GPT**: Handles AI-driven responses and conversation management
- **Flask**: Manages the application and webhook endpoints
- **Redis**: Efficient data caching and state management

### Features
- **Contact Management**: Complete WhatsApp contact operations and user tracking
- **Group Support**: Comprehensive WhatsApp group management capabilities
- **Template System**: Flexible message templates and response formatting
- **Memory Agent**: Contextual memory management for enhanced conversations
- **Provider Integration**: Modular design for AI service providers (DALL-E, etc.)

### Planned Features
- **Semantic Memory**: Integration with vector databases for advanced context management
- **Ollama Support**: Local LLM integration capabilities
- **Gemini Support**: Google's Gemini AI model integration
- **TTS and STT**: Text-to-speech and speech-to-text functionalities

## License

MIT License

---

**Note:** This project is for educational and prototyping purposes. Use responsibly and comply with WhatsApp and OpenAI.