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

## 2. Project Architecture (Phase 2 Foundation)

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
│   │   ├── deps.py         # Dependency injection providers (DB, settings)
│   │   └── v1/
│   │       ├── api.py      # Central v1 route aggregator
│   │       └── health.py   # Diagnostics & database connectivity probe
│   │
│   ├── database/           # Persistence Layer
│   │   ├── base.py         # SQLAlchemy 2.0 DeclarativeBase & TimestampMixin
│   │   └── session.py      # Engine, sessionmaker, SQLite WAL mode pragmas
│   │
│   ├── core/               # State machine & scheduling (Phase 3 & 6)
│   ├── intelligence/       # Local LLM connector & decision engine (Phase 4)
│   ├── language/           # Multilingual NLP & script normalizers (Phase 5)
│   ├── safety/             # Deterministic emergency trie & guardrails (Phase 6)
│   ├── services/           # External integrations (weather, adapters)
│   ├── speech/             # Offline STT, TTS & audio streaming (Phase 7)
│   └── users/              # User profiles & operating modes (Phase 3)
│
└── tests/                  # Automated Pytest Suite
    ├── conftest.py         # In-memory SQLite fixtures & TestClient
    ├── test_config.py      # Settings and environment validation tests
    ├── test_database.py    # Database connection & transaction tests
    └── test_api/
        └── test_health.py  # Health check & diagnostic endpoint tests
```

---

## 3. Quick Start & Setup Guide

### 3.1 Activate Existing Environment
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

Key configuration options in `.env`:
* `DATABASE_URL`: Defaults to `sqlite:///./carebot.db`.
* `ENVIRONMENT`: Set to `development`, `testing`, or `production`.
* `DEBUG`: `true` to enable OpenAPI documentation (`/docs`).

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

## 4. Health Check Specification

The `GET /health` endpoint verifies system readiness and database connectivity:

```bash
curl -s http://127.0.0.1:8000/health | jq
```

**Sample Response:**
```json
{
  "status": "healthy",
  "service": "CARE-BOT",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-19T03:57:38.123456+00:00",
  "database": {
    "status": "connected",
    "engine": "sqlite",
    "latency_ms": 0.42
  },
  "system": {
    "python_version": "3.12.16",
    "platform": "macOS-15.x-arm64-arm-64bit",
    "machine": "arm64"
  }
}
```

---

## 5. Running Automated Tests

Run the test suite using pytest:

```bash
pytest
```

Run with verbose output and test coverage:

```bash
pytest -v
```

---

## 6. Git Branching Model & Contribution Guidelines

* `main`: Protected production-ready code.
* `feature/backend-foundation`: Current Phase 2 development branch.
* **Strict Rule:** Never commit secrets, `.env` files, SQLite databases, or large AI weights (`*.gguf`, `*.onnx`) to Git.
