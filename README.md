# CARE-BOT

**Context-Aware, Multilingual AI-Powered Home Assistance System**

CARE-BOT is an intelligent, stationary, voice-based home assistance system designed to assist elderly residents and working professionals. It combines local offline AI intelligence, speech processing (STT/TTS), context-aware multi-turn conversation, persistent scheduling, and deterministic safety engines.

---

## 1. System Requirements & Hardware Target

* **Compute Host:** Apple Silicon Mac (M-series, ARM64)
* **Memory:** 16 GB Unified Memory
* **Python Runtime:** Python 3.12+
* **Form Factor:** Stationary smart-speaker assistant (No cameras; NFC/RFID physical identity selection)
* **Languages:** English, Hindi, and Bengali

---

## 2. Project Architecture (Phase 3: User Management & Context Engine)

```
CARE-BOT/
├── .env.example            # Environment configuration template
├── .gitignore              # Multi-tier ignore rules (venv, db, models, secrets)
├── pyproject.toml          # Project metadata and test configuration
├── requirements.txt        # Verified runtime dependencies
├── README.md               # System documentation and operational guide
│
├── app/
│   ├── __init__.py         # Package root with version tracking
│   ├── config.py           # Pydantic Settings management (.env)
│   ├── main.py             # FastAPI factory, lifespan manager & CORS
│   │
│   ├── api/                # HTTP & WebSocket routing
│   │   ├── deps.py         # Dependency injection (Auth, RBAC, DB, Device Key)
│   │   └── v1/
│   │       ├── api.py      # Central v1 route aggregator
│   │       ├── auth.py     # Login & token authentication endpoints
│   │       ├── health.py   # Diagnostics & database connectivity probe
│   │       ├── users.py    # User profiles, preferences & NFC lookup endpoints
│   │       └── conversations.py # Dialogue sessions, messages & context endpoints
│   │
│   ├── core/               # System Core
│   │   ├── security.py     # PBKDF2 600k hashing & HMAC-SHA256 JWT tokens
│   │   └── context_engine.py # Context compilation & prompt synthesis for LLMs
│   │
│   ├── database/           # Persistence Layer
│   │   ├── base.py         # SQLAlchemy 2.0 DeclarativeBase & TimestampMixin
│   │   ├── session.py      # Engine, sessionmaker, SQLite WAL mode pragmas
│   │   └── models/
│   │       ├── user.py     # User & UserPreferences SQLAlchemy models
│   │       └── conversation.py # ConversationSession & Message models
│   │
│   ├── conversation/       # Conversation Context Module
│   │   ├── schemas.py      # Session and Message Pydantic validation schemas
│   │   └── service.py      # Session and message history business logic
│   │
│   ├── users/              # User Management Module
│   │   ├── schemas.py      # User and Preferences Pydantic validation schemas
│   │   └── service.py      # User registration & preferences business logic
│   │
│   ├── intelligence/       # Local LLM connector & decision engine (Phase 4)
│   ├── language/           # Multilingual NLP & script normalizers (Phase 5)
│   ├── safety/             # Deterministic emergency trie & guardrails (Phase 6)
│   ├── services/           # External integrations (weather, adapters)
│   └── speech/             # Offline STT, TTS & audio streaming (Phase 7)
│
└── tests/                  # Automated Pytest Suite (44 Unit & Integration Tests)
    ├── conftest.py         # In-memory SQLite fixtures, TestClient & auth token helpers
    ├── test_config.py      # Settings and environment validation tests
    ├── test_database.py    # Database connection & transaction tests
    ├── test_security.py    # PBKDF2 600k hashing, salt & JWT token tests
    ├── test_security_audit.py # RBAC, token auth, impersonation & NFC audit tests
    ├── test_users.py       # User CRUD, operating modes & preferences tests
    ├── test_conversations.py # Session persistence & cross-user isolation tests
    ├── test_context_engine.py # Context synthesis & language prompt tests
    └── test_api/
        └── test_health.py  # Health check & diagnostic endpoint tests
```

---

## 3. Quick Start & Setup Guide

### 3.1 Activate Environment
The project uses Python 3.12 inside the local virtual environment:

```bash
cd ~/CARE-BOT
source .venv/bin/activate
```

### 3.2 Configure Environment Variables
Copy the template to create your local `.env`:

```bash
cp .env.example .env
```

### 3.3 Run the Development Server
Launch the backend using Uvicorn:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will start at:
* **Root API:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Interactive OpenAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4. API Endpoints Overview

### Authentication (`/api/v1/auth`)
* `POST /api/v1/auth/login`: Authenticate with username and password, returns signed HMAC-SHA256 access token.
* `GET /api/v1/auth/me`: Resolve current authenticated user profile and permissions from Bearer token.

### User Management (`/api/v1/users`)
* `POST /api/v1/users`: Register a new user profile with tailored defaults and 600,000-iteration PBKDF2 password hash.
* `GET /api/v1/users`: List registered users (requires authentication; Caregivers see all, Residents see themselves).
* `GET /api/v1/users/{user_id}`: Fetch user profile and preferences (ownership or Caregiver role enforced).
* `PUT /api/v1/users/{user_id}`: Update user profile and switch operating mode (ownership or Caregiver enforced; Family Viewers blocked).
* `PUT /api/v1/users/{user_id}/preferences`: Update language, speech rate, and accessibility settings.
* `DELETE /api/v1/users/{user_id}`: Delete user profile and cascade delete sessions.
* `GET /api/v1/users/lookup/nfc/{nfc_card_id}`: Resolve user identity via hardware NFC card tap (requires `X-Device-Key` or authenticated session; returns minimal non-PII identification payload).

### Conversation Context (`/api/v1/conversations`)
* `POST /api/v1/conversations/users/{user_id}/sessions`: Create a dialogue session.
* `GET /api/v1/conversations/users/{user_id}/sessions`: List sessions for a user.
* `GET /api/v1/conversations/users/{user_id}/sessions/{session_id}`: Get session details (enforces ownership).
* `PUT /api/v1/conversations/users/{user_id}/sessions/{session_id}`: Update FSM state or context metadata.
* `POST /api/v1/conversations/users/{user_id}/sessions/{session_id}/messages`: Append user utterance (rejects client impersonation of assistant messages).
* `GET /api/v1/conversations/users/{user_id}/sessions/{session_id}/messages`: Retrieve chronological dialogue.
* `GET /api/v1/conversations/users/{user_id}/context`: Compile structured context payload for local LLM inference.

---

## 5. Running Automated Tests

Run the complete test suite:

```bash
pytest -v
```

All 44 unit and security integration tests execute in under 0.7 seconds against an isolated in-memory SQLite database.
