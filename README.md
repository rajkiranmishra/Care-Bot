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
│   ├── intelligence/       # Intelligence Engine (Phase 4.1 & 4.2)
│   │   ├── schemas.py      # Pydantic schemas for intents, actions, LLM responses & chat turns
│   │   ├── llm_client.py   # LocalLLMClient protocol & MockLLMClient
│   │   ├── emergency.py    # Deterministic pre-LLM emergency signal detection
│   │   ├── engine.py       # ConversationEngine orchestrator (Phase 4.2)
│   │   ├── prompt_builder.py # Persona prompt builder with JSON schema guidance
│   │   ├── parser.py       # Defensive JSON extractor and safe fallback parser
│   │   ├── intent.py       # Intent classification interfaces & routing
│   │   ├── validator.py    # ActionValidator whitelist & parameter boundary checker
│   │   └── permissions.py  # PermissionChecker enforcing RBAC on proposed actions
│   │
│   ├── language/           # Multilingual NLP & script normalizers (Phase 5)
│   ├── safety/             # Deterministic emergency trie & guardrails (Phase 6)
│   ├── services/           # External integrations (weather, adapters)
│   └── speech/             # Offline STT, TTS & audio streaming (Phase 7)
│
└── tests/                  # Automated Pytest Suite (125 Unit & Integration Tests)
    ├── conftest.py         # In-memory SQLite fixtures, TestClient & auth token helpers
    ├── test_config.py      # Settings, secret validation & environment tests
    ├── test_database.py    # Database connection & transaction tests
    ├── test_security.py    # PBKDF2 600k hashing, salt, JWT token & RFC claims tests
    ├── test_security_audit.py # RBAC, token auth, multi-process bootstrap, legacy DB & NFC audit tests
    ├── test_users.py       # User CRUD, operating modes & preferences tests
    ├── test_conversations.py # Session persistence & cross-user isolation tests
    ├── test_context_engine.py # Context synthesis & language prompt tests
    ├── test_conversation_engine.py # Phase 4.2 ConversationEngine unit & integration tests (26 Tests)
    ├── test_api/
    │   └── test_health.py  # Health check & diagnostic endpoint tests
    └── test_intelligence/  # Phase 4.1 Intelligence & Emergency Tests (41 Tests)
        ├── test_emergency.py # Pre-LLM emergency detection, silence policy & regression tests
        ├── test_parser.py    # Defensive JSON extraction & safe fallback tests
        ├── test_prompt_builder.py # Prompt builder & few-shot exemplars tests
        ├── test_validator.py # Action whitelist & parameter schema boundary tests
        ├── test_permissions.py # RBAC action permission & prompt injection tests
        └── test_llm_client.py # LocalLLMClient protocol & MockLLMClient tests
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
* `POST /api/v1/conversations/users/{user_id}/chat`: Execute authenticated, multi-turn conversational turn orchestrated by `ConversationEngine` with deterministic pre-LLM emergency signal interception, context synthesis, sliding dialogue history, and RBAC action permissions.
* `POST /api/v1/conversations/users/{user_id}/emergency/resolve`: Authorized caregiver resolution procedure to clear an active resident emergency and stand down alert state.

---

## 5. Running Automated Tests

Run the complete test suite:

```bash
pytest -v
```

All 125 unit, integration, and intelligence security tests execute in under 1.0 seconds against an isolated in-memory SQLite database.

---

## 6. Security Model, Privilege Boundaries & Documented Limitations

### 6.1 Confirmed Security Protections
1. **Privilege Escalation Prevention & Multi-Process Bootstrap Security:**
   * Public self-registration (`POST /api/v1/users`) is strictly restricted to the `resident` role. An unauthenticated client cannot claim administrator privileges merely because the database is empty.
   * Initial administrator provisioning requires an explicit, cryptographically secure one-time secret (`X-Bootstrap-Token`) matching `BOOTSTRAP_TOKEN`.
   * **Multi-Process Atomic Protection:** Both in-process mutex locking (`_bootstrap_lock`) and database-level primary key uniqueness (`SystemBootstrap` with `id=1`) enforce that even with multiple Uvicorn workers (`--workers 4`) or multiple distributed instances, only a single initial administrator can ever commit.
   * **Transactional Integrity:** `SystemBootstrap(id=1)` insertion, user creation, and preferences initialization execute within the same database transaction (`auto_commit=False`). If user creation fails (e.g. duplicate conflict or invalid field), the bootstrap marker rolls back completely.
   * **Legacy Database Protection:** If upgrading an existing database where a caregiver already exists, the endpoint checks both `User.role == CAREGIVER` AND `SystemBootstrap.id == 1`. An empty `SystemBootstrap` table on an existing deployment cannot be exploited to bootstrap another administrator.
   * The bootstrap mechanism is permanently disabled once the first administrator account is created.
   * Ongoing assignment of elevated roles requires an authenticated `Caregiver` session or authorized device provisioning via `X-Device-Key`.
   * The bootstrap secret is never logged, serialized in API models, or committed to Git.
2. **Assistant Message Impersonation Block:**
   * Clients cannot inject synthetic assistant or system utterances into conversation history. External message posts are restricted to `role: "user"`.
3. **Password Security:**
   * Passwords hashed using PBKDF2-HMAC-SHA256 with 600,000 rounds (OWASP 2023) and 16-byte cryptographically secure random salt.
   * Passwords, hashes, and salts are never returned in responses, serialized schemas, or logs.
4. **Physical NFC Hardware Security:**
   * NFC lookup endpoint requires either a valid hardware `X-Device-Key` (verified via constant-time `hmac.compare_digest`) or an authenticated session token.
   * Possession of a card identifier alone never exposes private user records, returning only minimal non-PII identity info.
   * `DEVICE_API_KEY` defaults to `None`; no insecure static default is present in settings.
5. **Production Secret Enforcement:**
   * In `ENVIRONMENT=production`, application startup fails fast if `SECRET_KEY` is under 32 characters or contains dev/insecure markers, if `DEVICE_API_KEY` is not explicitly configured, or if `BOOTSTRAP_TOKEN` is weak.

### 6.2 Bootstrap Token Lifecycle & Disaster Recovery
1. **Token Generation:**
   Generate a 256-bit CSPRNG secret on the host machine:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. **Secure Configuration:**
   Place the token in `.env` on the host:
   ```bash
   echo 'BOOTSTRAP_TOKEN="<generated_token>"' >> .env
   chmod 600 .env
   ```
3. **Execution:**
   Submit initial administrator registration:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/users \
     -H "Content-Type: application/json" \
     -H "X-Bootstrap-Token: <generated_token>" \
     -d '{"username": "admin", "full_name": "System Administrator", "password": "SecurePassword123!", "role": "caregiver"}'
   ```
4. **Disablement & Cleanup:**
   Once `SystemBootstrap(id=1)` is committed to the database, remove or comment out `BOOTSTRAP_TOKEN` from `.env` and restart the server so the secret no longer resides in process memory.
5. **Administrator Recovery Procedure:**
   If the administrator account is locked or credentials are lost, **never re-open public bootstrap over the network**. Recovery must be performed via local host access (e.g. running a dedicated maintenance script with root permissions or updating the password hash directly in SQLite on the local filesystem).

### 6.3 Documented System Limitations (Phase 3)
* **Caregiver-to-Resident Authorization Scope:**
  * Any user assigned the `CAREGIVER` role currently possesses system-wide oversight across all residents in the local deployment.
  * Explicit fine-grained caregiver-to-resident assignments (e.g. a `CaregiverResidentMapping` join model) are not yet implemented in Phase 3.
  * Do not claim per-resident multi-tenant authorization until explicit relationship mappings are implemented.

### 6.4 Database Compatibility & Concurrency Boundaries
* **SQLite (Default Local Target):**
  * Configured with `PRAGMA journal_mode=WAL;` and `PRAGMA foreign_keys=ON;`.
  * Write transactions are serialized at the database file level. Primary key uniqueness on `SystemBootstrap(id=1)` combined with `_bootstrap_lock` guarantees atomic bootstrap.
  * Verified for local stationary host operation on macOS Apple Silicon. Under high concurrency across multiple processes, SQLite writes serialize or raise `OperationalError: database is locked` if busy timeouts elapse.
* **PostgreSQL (Supported Production Target):**
  * In PostgreSQL, multi-version concurrency control (MVCC) with primary key index constraints on `SystemBootstrap.id=1` guarantees atomicity across distributed workers/processes.
  * No cross-process or distributed cluster guarantees are claimed beyond single relational database ACID constraints.

### 6.5 Recommendations for Production
1. **JWT Library Migration:**
   * The current pure-standard-library implementation (`hmac`, `hashlib`, `json`, `base64`) provides a lightweight, zero-dependency, auditable RFC 7519 HMAC-SHA256 implementation suitable for the initial prototype.
   * For production, especially when adopting asymmetric signing (RS256/EdDSA) or integrating external OIDC identity providers, migrate to a dedicated, actively maintained library like `PyJWT` backed by `cryptography`.
2. **Granular Caregiver Delegation:**
   * Implement a `CaregiverResidentMapping` relationship table allowing household administrators to delegate access to specific caregivers or family viewers on a per-resident basis.
3. **Hardware Keystore:**
   * Store `DEVICE_API_KEY` in the host's secure enclave / keychain rather than plaintext environment files on hardware deployments.

---

## 7. Phase 4.1 Intelligence Architecture & Pre-LLM Emergency Layer

### 7.1 Deterministic Pre-LLM Emergency Detection
The emergency signal detection layer executes **before** LLM inference:
1. **Deterministic & Resilient:** Functions independently of local LLM availability, connection drops, or model errors.
2. **Separation of Detection from Notification:** `EmergencySignalDetector` strictly classifies signals and returns diagnostics. It does **not** dispatch notifications, dial emergency services, or initiate external side-effects. Notification execution is decoupled and deferred to a separately authorized service.
3. **Multilingual Phrase Matching:** Detects falls, acute chest pain, and distress pleas in English, Hindi (Devanagari & Hinglish), and Bengali (Bengali script & Romanized).
4. **Phonetic & Transcription Error Resilience:** Normalizes elongated syllables (`"heeeelp"`, `"bachaaao"`) and phonetic variations.
5. **False Positive Suppression:** Contextual filters suppress benign recreational or task-oriented uses of "help" (e.g. crosswords, finding glasses, movie quotes).
6. **Refined Silence Safety Invariant & Escalation Policy:**
   * Silence must **never** be interpreted as proof that the user is safe (`is_safe_confirmed=False`).
   * Silence alone without prior distress does **not** automatically establish a critical emergency (remains `NORMAL`).
   * Distinguishes distinct `NORMAL`, `UNCERTAIN`, and `CRITICAL` states.
   * Unanswered welfare checks follow a configurable `WelfareEscalationPolicy` (considering prior distress signals, user vulnerability multiplier, and consecutive unanswered prompt thresholds).
   * Silence following a confirmed emergency (e.g. fall) never downgrades or resets the emergency status.
   * Only an explicit, affirmative confirmation from the user (e.g. *"I am fine"*, *"false alarm"*, *"sab theek hai"*) sets `is_safe_confirmed=True` for ordinary uncertain welfare checks.
   * **Confirmed Emergency Safety Invariant:** An affirmative phrase (*"I am fine"*) spoken after a confirmed CRITICAL emergency does **not** clear the emergency state. It records the verbal affirmation in `user_affirmation` and sets `pending_resolution=True`, requiring authorized caregiver resolution before the alert can be dismissed.

### 7.2 Safety Invariants for Local LLM Integration
1. **Advisory Only (No Direct DB or Execution Access):** The local LLM receives read-only textual prompts and outputs structured JSON text. It has zero database sockets, zero SQL execution authority, and zero ORM access.
2. **Action Whitelist:** Models may only propose actions from an enumerated whitelist (`none`, `create_reminder`, `query_weather`, `trigger_emergency_alert`, `health_log`). Arbitrary tools or dynamic function names are rejected.
3. **Parameter Schema Boundaries:** Parameters are validated by `ActionValidator` with strict bounds (e.g. string lengths, time formats).
4. **Server-Side Authorization Invariant:** Model outputs can **never** override server-side authorization. `PermissionChecker` validates every proposed action against the authenticated user's actual role (Family Viewers cannot mutate state).
5. **Fail-Safe Response Parsing:** `ResponseParser` extracts JSON from markdown or chatter, tolerates missing non-critical fields, and falls back to safe conversational defaults upon syntax errors without crashing route handlers.

---

## 8. Phase 4.2 Conversation Engine Orchestration Architecture

### 8.1 Orchestration Workflow & Safety Order
The `ConversationEngine` (`app/intelligence/engine.py`) coordinates multi-turn dialogue across user identity, operating modes, context synthesis, dialogue history, and local model inference:

```
[User Utterance (HTTP / WebSocket)]
                 │
                 ▼
[Step 1: Deterministic Pre-LLM Emergency Evaluation]
        │                                 │
   (CRITICAL / URGENT)             (NORMAL / UNCERTAIN)
        │                                 │
        ▼                                 ▼
[Emergency Fast-Path]            [Step 2: Context Engine Synthesis]
• Bypass LLM completely         • Profile, mode, preferences, sliding history
• Preserve active alert          • Build prompt via PromptBuilder
• Safe, truthful response                 │
• Record in DB                            ▼
                                 [Step 3: Local LLM Inference]
                                 • MockLLMClient (zero external model download)
                                 • Async timeout & failure protection
                                          │
                                          ▼
                                 [Step 4: Defensive Response Parsing]
                                 • ResponseParser extracts structured JSON
                                 • Markdown / chatter resilience
                                          │
                                          ▼
                                 [Step 5: Action & Boundary Validation]
                                 • ActionValidator enforces whitelist
                                 • Bounds check string & numeric inputs
                                          │
                                          ▼
                                 [Step 6: Server-Side RBAC Enforcement]
                                 • PermissionChecker validates user permissions
                                 • Family Viewer read-only block
                                          │
                                          ▼
                                 [Step 7: Database Persistence & FSM Update]
                                 • Persist user & assistant messages to DB
                                 • Advance session state to ACTIVE_CONVERSATION
                                          │
                                          ▼
                                 [Step 8: Structured Turn Response Returned]
```

### 8.2 Safety Invariants & Emergency State Machines
1. **Deterministic Pre-LLM Emergency Interception:**
   Emergency detection (`EmergencySignalDetector.evaluate_utterance`) strictly executes before context compilation or model invocation. Critical distress signals bypass LLM execution completely, guaranteeing sub-second response times and eliminating latency or failure risks.
2. **Durable Emergency State Persistence:**
   An active `CRITICAL` or `URGENT` emergency is committed to the database (`active_emergencies` table) and associated with the resident's account (`user_id`), NOT a transient conversation session.
   Starting a new dialogue session, changing operating modes, restarting the application host, or sending subsequent non-emergency messages CANNOT silently clear or reset an active emergency.
3. **Confirmed Emergency Retention Invariant:**
   If a user utters an affirmative phrase (e.g. *"I am fine"*, *"sab theek hai"*, *"aami bhalo aachi"*) while an active `CRITICAL` emergency is recorded, the engine sets `pending_resolution=True` and preserves `active_emergency=True`. The active emergency remains in effect until an authorized caregiver completes the resolution procedure. Elderly residents who have suffered a fall or head injury may reflexively report feeling fine; verbal affirmations alone must never cancel an acute physical alert.
4. **Honest Emergency Communication & Multi-Tier State Tracking:**
   The engine strictly distinguishes between:
   * **Emergency Detected:** Recognized by deterministic regex/phonetic matcher.
   * **Recorded Locally:** Committed to local database (`recorded_locally: True`).
   * **Notification Attempted:** External network notification initiated (`notification_attempted: False` in local phase).
   * **Notification Delivered:** Confirmed received by caregiver or external service (`notification_delivered: False` in local phase).
   The system NEVER claims that a caregiver is being notified or has been notified unless confirmed by a verified external delivery integration.
5. **Personal Information Protection (Zero Hallucination):**
   The engine and `MockLLMClient` NEVER invent resident-specific personal data (such as medication names, times, reminders, doctor appointments, or medical history). When verified service integrations are unavailable, the assistant explicitly states that it cannot access the information.
6. **Zero External Inference Dependencies:**
   Phase 4.2 integrates `MockLLMClient` with deterministic responses, allowing full unit, integration, and end-to-end HTTP testing without requiring Ollama runtime installations, background model daemons, or downloading multi-gigabyte neural weights.

### 8.3 Authenticated Chat Turn API (`POST /api/v1/conversations/users/{user_id}/chat`)
* **Endpoint:** `POST /api/v1/conversations/users/{user_id}/chat`
* **Authentication:** `Authorization: Bearer <JWT>`
* **Authorization Scope:** The authenticated user must match `user_id`, or have the `CAREGIVER` role. Family Viewers have read-only permissions and are rejected with HTTP 403 Forbidden.
* **Identity Security:** The target `user_id` is bound exclusively from the validated URL path parameter. Any arbitrary `user_id` in request payloads is ignored.

#### Example Request: Unsupported Medication Query (Honest Access Report)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/conversations/users/1/chat \
  -H "Authorization: Bearer <RESIDENT_JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What medications do I have scheduled for this morning?"
  }'
```

#### Example Response (No Hallucination):
```json
{
  "session_id": "c3f8e65e-3bd7-463e-a89e-ecf8f3b20721",
  "user_message_id": 1,
  "user_utterance": "What medications do I have scheduled for this morning?",
  "assistant_message_id": 2,
  "spoken_response": "I do not have access to your medication schedule or medical records. Please consult your caregiver or healthcare provider.",
  "intent": "reminder_query",
  "proposed_action": {
    "action_type": "none",
    "parameters": {},
    "requires_confirmation": false
  },
  "emergency_signal": {
    "detected": false,
    "severity": "normal",
    "trigger_phrase": null,
    "confidence": 0.0,
    "reasoning": "No emergency signal detected",
    "is_safe_confirmed": false,
    "requires_confirmation": false,
    "requires_immediate_escalation": false,
    "repeat_count": 1,
    "user_affirmation": null,
    "pending_resolution": false,
    "recorded_locally": false,
    "notification_attempted": false,
    "notification_delivered": false
  },
  "session_state": "active_conversation",
  "active_emergency": false,
  "pending_resolution": false,
  "recorded_locally": false,
  "notification_attempted": false,
  "notification_delivered": false
}
```

#### Example Emergency Fast-Path Response (Truthful Notification State):
```json
{
  "session_id": "c3f8e65e-3bd7-463e-a89e-ecf8f3b20721",
  "user_message_id": 3,
  "user_utterance": "I fell down and cannot get up, help!",
  "assistant_message_id": 4,
  "spoken_response": "Emergency signal registered locally: English: Fall detected (unable to rise). Please remain calm and stay in a safe position. (System note: Emergency recorded locally. External notification has not been dispatched.)",
  "intent": "emergency",
  "proposed_action": {
    "action_type": "trigger_emergency_alert",
    "parameters": {
      "severity": "critical",
      "trigger_phrase": "fell down and cannot get up",
      "repeat_count": 1
    },
    "requires_confirmation": false
  },
  "emergency_signal": {
    "detected": true,
    "severity": "critical",
    "trigger_phrase": "fell down and cannot get up",
    "confidence": 1.0,
    "reasoning": "English: Fall detected (unable to rise)",
    "is_safe_confirmed": false,
    "requires_confirmation": false,
    "requires_immediate_escalation": true,
    "repeat_count": 1,
    "user_affirmation": null,
    "pending_resolution": false,
    "recorded_locally": true,
    "notification_attempted": false,
    "notification_delivered": false
  },
  "session_state": "active_conversation",
  "active_emergency": true,
  "pending_resolution": false,
  "recorded_locally": true,
  "notification_attempted": false,
  "notification_delivered": false
}
```

---

### 8.4 Caregiver Emergency Resolution Workflow (`POST /api/v1/conversations/users/{user_id}/emergency/resolve`)

An active critical emergency can **only** be cleared through an authorized caregiver resolution procedure.

* **Endpoint:** `POST /api/v1/conversations/users/{user_id}/emergency/resolve`
* **Authorization Scope:** Strictly restricted to authenticated users with the `CAREGIVER` role. Residents and Family Viewers receive HTTP 403 Forbidden.

#### Example Caregiver Resolution Request:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/conversations/users/1/emergency/resolve \
  -H "Authorization: Bearer <CAREGIVER_JWT>" \
  -H "Content-Type: application/json"
```

#### Example Resolution Response:
```json
{
  "status": "resolved",
  "emergency_id": 1,
  "user_id": 1,
  "severity": "critical",
  "resolved_by": 2,
  "resolved_at": "2026-11-26T01:25:00.000000+00:00"
}
```

---

### 8.5 Implemented vs. Planned Functionality Matrix

| Capability | Current Status | Architectural Implementation / Future Target |
| :--- | :--- | :--- |
| **Pre-LLM Emergency Detection** | **Implemented** | Deterministic multilingual regex and phonetic normalization executed before LLM inference |
| **Durable Emergency Persistence** | **Implemented** | Session-independent `active_emergencies` database table anchored to `User` |
| **Emergency Delivery Honesty** | **Implemented** | 4-state tracking (`detected`, `recorded_locally`, `notification_attempted`, `notification_delivered`) |
| **Caregiver Resolution Workflow** | **Implemented** | Dedicated endpoint `POST /emergency/resolve` restricted to `UserRole.CAREGIVER` |
| **Personal Data Protection** | **Implemented** | Engine & MockLLM refuse to fabricate medication names, schedules, or appointments |
| **Transaction Integrity** | **Implemented** | Atomic dual-message turn persistence with immediate database rollback on failure |
| **Sliding Window History** | **Implemented** | Context engine compiles recent chronological messages for few-shot prompt builder |
| **Live Caregiver Push Notifications** | *Planned (Phase 5/6)* | External webhook/SMS/push delivery adapter with delivery receipts |
| **Emergency EMS / 911 Dispatch** | *Planned (Phase 6)* | Hardware telephony / e911 emergency services integration |
| **Resident Medication & Reminder DB** | *Planned (Phase 6)* | Persistent cron schedule & medication adherence tracking database |

