# Slack-to-Cohere RAG System Architecture (MVP)

## Architecture Philosophy

This architecture prioritizes **practical implementation over theoretical perfection**. Instead of a complex microservices sprawl, we use **3 focused services** that handle the complete Slack-to-RAG workflow with proper multi-tenancy and Slack-native integration patterns.

## Core Design Principles

1. **Slack-Native Integration**: Built around Slack's OAuth, Events API, and UX patterns
2. **Multi-Tenant by Design**: Each `team_id` is a separate tenant with isolated data
3. **Thread-Aware Processing**: Respects Slack's conversation semantics and threading
4. **MVP-First Architecture**: Minimal services that can scale when needed
5. **Security & Compliance**: Proper signature verification, encryption, and audit trails

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Slack Workspace                         │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│   │ /ask    │  │ @bot    │  │ Actions │  │ App Home       │   │
│   │ command │  │ mention │  │ buttons │  │ (settings)     │   │
│   └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Internet / Slack API                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Your Application                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Service A  │  │   Service B  │  │      Service C       │  │
│  │   Ingestion  │  │   Query API  │  │   Slack Frontend     │  │
│  │   & Sync     │  │              │  │   (UX Layer)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Shared Data Layer                       │  │
│  │                                                             │  │
│  │  ┌────────────┐ ┌──────────────┐ ┌─────────────────────┐  │  │
│  │  │PostgreSQL  │ │ Vector DB    │ │     Redis           │  │  │
│  │  │(metadata)  │ │(Pinecone/    │ │(cache + queue)      │  │  │
│  │  │            │ │ Weaviate)    │ │                     │  │  │
│  │  └────────────┘ └──────────────┘ └─────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Service Architecture Details

### Service A: Ingestion & Sync Worker

**Purpose**: Handle all Slack data ingestion, processing, and vector storage

**Components**:
```python
# Main responsibilities
class IngestionService:
    - Slack Events webhook handler
    - Message processing pipeline  
    - File download & text extraction
    - Cohere embedding generation
    - Vector database writes
    - Edit/delete handling
    - Backfill jobs for new installs
```

**Key Features**:
- **Signature Verification**: Validates all Slack webhooks with signing secret
- **Multi-tenant Processing**: Routes messages by `team_id` namespace
- **Thread-Aware Indexing**: Maintains thread context and reply relationships
- **File Handling**: Downloads files with bot token authorization, extracts text
- **Real-time Updates**: Processes message edits/deletes to maintain data integrity
- **Batch Processing**: Efficient embedding generation and database writes

**Data Flow**:
```
Slack Event → Signature Check → Team Validation → Message Process → 
Text Extract → Chunk & Embed → Vector Store → Update Metadata
```

**Technical Implementation**:
```python
# Event processing pipeline
@app.post("/slack/events")
async def handle_slack_events(request: Request):
    # 1. Verify signature
    if not verify_slack_signature(request):
        raise HTTPException(401, "Invalid signature")
    
    event = await request.json()
    
    # 2. Handle different event types
    if event["type"] == "url_verification":
        return {"challenge": event["challenge"]}
    
    # 3. Queue for async processing
    await redis_queue.enqueue("process_event", event)
    return {"status": "ok"}

# Background worker
async def process_event(event: dict):
    team_id = event["team_id"]
    
    # Route by event type
    if event["event"]["type"] == "message":
        await process_message_event(team_id, event["event"])
    elif event["event"]["type"] == "message_changed":
        await handle_message_edit(team_id, event["event"])
    elif event["event"]["type"] == "message_deleted":
        await handle_message_deletion(team_id, event["event"])
```

### Service B: Query API

**Purpose**: Handle user queries, RAG pipeline, and response generation

**Components**:
```python
# Core RAG functionality
class QueryService:
    - Query preprocessing & validation
    - Hybrid retrieval (BM25 + vector)
    - Thread context expansion
    - Cohere reranking & generation
    - Citation formatting
    - Response streaming
```

**RAG Pipeline Architecture**:
```
Query → Validate Tenant → Filter Channels → BM25 Pre-filter → 
Vector Search → Cohere Rerank → Thread Expand → Generate Response → 
Format Citations → Return JSON
```

**Technical Implementation**:
```python
class SlackRAGService:
    def __init__(self):
        self.cohere = cohere.Client(COHERE_API_KEY)
        self.vector_db = get_vector_db()
        self.bm25_index = get_bm25_index()
    
    async def answer_question(self, team_id: str, query: str, **filters) -> RAGResponse:
        # 1. Validate tenant and permissions
        installation = await get_installation(team_id)
        if not installation or not installation.is_active:
            raise HTTPException(404, "App not installed")
        
        # 2. Apply channel allowlist filter
        allowed_channels = installation.channel_allowlist or []
        base_filter = {
            "team_id": team_id,
            "deleted_at": {"$exists": False}
        }
        
        if filters.get("channel_id"):
            if filters["channel_id"] not in allowed_channels:
                raise PermissionError("Channel not allowed")
            base_filter["channel_id"] = filters["channel_id"]
        else:
            base_filter["channel_id"] = {"$in": allowed_channels}
        
        # 3. Hybrid retrieval
        # Step 1: BM25 keyword search (fast pre-filter)
        keyword_candidates = await self.bm25_index.search(
            query, filters=base_filter, top_k=100
        )
        
        # Step 2: Vector similarity on candidates
        query_embedding = await self.cohere.embed(
            [query], 
            model="embed-english-v3.0",
            input_type="search_query"
        )
        
        vector_results = await self.vector_db.query(
            namespace=team_id,
            vector=query_embedding.embeddings[0],
            filter=base_filter,
            top_k=20,
            include_metadata=True
        )
        
        # Step 3: Cohere rerank for relevance
        documents = [r.metadata["text"] for r in vector_results]
        reranked = self.cohere.rerank(
            query=query,
            documents=documents,
            top_k=5,
            model="rerank-english-v3.0"
        )
        
        # 4. Expand with thread context
        core_results = [vector_results[r.index] for r in reranked.results]
        expanded_context = await self.expand_thread_context(core_results)
        
        # 5. Generate response
        context_text = self.format_context_for_generation(expanded_context)
        
        response = self.cohere.generate(
            model="command-r-plus",
            prompt=self.build_rag_prompt(query, context_text),
            max_tokens=500,
            temperature=0.3,
            stop_sequences=["\n\nHuman:", "\n\nUser:"]
        )
        
        # 6. Format with citations
        answer = response.generations[0].text
        sources = self.extract_source_citations(expanded_context)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence_score=reranked.results[0].relevance_score,
            processing_time=time.time() - start_time
        )
```

### Service C: Slack Frontend (UX Layer)

**Purpose**: Handle all Slack-specific user interactions and interface

**Components**:
```python
# Slack UX handlers
class SlackFrontend:
    - Slash command processing
    - Interactive message handling  
    - App Home tab management
    - OAuth installation flow
    - Settings & channel management
    - Ephemeral response previews
```

**User Experience Flow**:
```
User Action → Command Parse → Permission Check → Query Service → 
Format Response → Show Preview → User Confirm → Post to Channel
```

**Key UX Patterns**:

**1. Slash Commands**:
```python
@app.post("/slack/commands/ask")
async def handle_ask_command(request: Request):
    form = await request.form()
    
    # Parse: /ask what did Sarah decide about pricing? #marketing last 7d
    query_parts = parse_ask_command(form["text"])
    
    # Generate response
    rag_response = await query_service.answer_question(
        team_id=form["team_id"],
        query=query_parts.question,
        channel_id=query_parts.channel,
        time_window_days=query_parts.days or 30
    )
    
    # Show ephemeral preview first
    return {
        "response_type": "ephemeral",
        "blocks": build_preview_blocks(rag_response) + [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Post Answer"},
                        "style": "primary", 
                        "value": f"post:{hash(rag_response)}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✏️ Edit Query"},
                        "value": f"edit:{hash(query_parts.question)}"
                    }
                ]
            }
        ]
    }
```

**2. Message Actions**:
```python
@app.post("/slack/actions")
async def handle_interactive_messages(request: Request):
    payload = json.loads((await request.form())["payload"])
    
    if payload["callback_id"] == "summarize_thread":
        # Get thread context
        thread_ts = payload["message"].get("thread_ts") or payload["message"]["ts"]
        
        # Generate thread summary
        summary = await query_service.summarize_thread(
            team_id=payload["team"]["id"],
            channel_id=payload["channel"]["id"],
            thread_ts=thread_ts
        )
        
        # Post as reply in thread
        await slack_client.chat_postMessage(
            token=get_bot_token(payload["team"]["id"]),
            channel=payload["channel"]["id"],
            thread_ts=thread_ts,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"🧵 **Thread Summary**\n\n{summary.text}"}
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"📊 {summary.message_count} messages • ⏱️ {summary.time_span}"}
                    ]
                }
            ]
        )
```

**3. App Home (Admin Settings)**:
```python
def build_app_home_view(team_id: str, user_id: str) -> dict:
    installation = await get_installation(team_id)
    user_info = await get_user_info(team_id, user_id)
    
    is_admin = user_info.is_admin or user_id == installation.installer_user_id
    
    if not is_admin:
        return build_user_home_view(installation)
    
    # Admin settings interface
    return {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🤖 RAG Assistant Settings"}
            },
            {
                "type": "section", 
                "text": {"type": "mrkdwn", "text": "*Channel Access Control*\nChoose which channels the bot can search:"},
                "accessory": {
                    "type": "multi_channels_select",
                    "placeholder": {"type": "plain_text", "text": "Select channels..."},
                    "initial_channels": installation.channel_allowlist or [],
                    "action_id": "update_channel_allowlist"
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Data Retention*\nMessages older than {installation.retention_days} days are automatically removed."},
                "accessory": {
                    "type": "static_select",
                    "options": [
                        {"text": {"type": "plain_text", "text": "30 days"}, "value": "30"},
                        {"text": {"type": "plain_text", "text": "90 days"}, "value": "90"},
                        {"text": {"type": "plain_text", "text": "1 year"}, "value": "365"}
                    ],
                    "initial_option": {"text": {"type": "plain_text", "text": f"{installation.retention_days} days"}, "value": str(installation.retention_days)},
                    "action_id": "update_retention"
                }
            }
        ]
    }
```

## Data Architecture

### Multi-Tenant Data Model

**PostgreSQL Schema**:
```sql
-- Installation tracking (one per workspace)
CREATE TABLE slack_installations (
    team_id VARCHAR(20) PRIMARY KEY,
    installer_user_id VARCHAR(20) NOT NULL,
    bot_access_token TEXT NOT NULL, -- encrypted
    scopes TEXT[] NOT NULL,
    installation_time TIMESTAMP DEFAULT NOW(),
    channel_allowlist TEXT[], -- null = all channels
    retention_days INTEGER DEFAULT 90,
    is_active BOOLEAN DEFAULT true,
    settings JSONB DEFAULT '{}'
);

-- Message metadata (searchable fields)
CREATE TABLE slack_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id VARCHAR(20) NOT NULL,
    channel_id VARCHAR(20) NOT NULL,
    message_ts VARCHAR(20) NOT NULL, -- Slack timestamp
    thread_ts VARCHAR(20), -- null for non-threaded
    user_id VARCHAR(20) NOT NULL,
    message_type VARCHAR(20) DEFAULT 'message',
    text_preview TEXT, -- first 500 chars for search
    has_files BOOLEAN DEFAULT false,
    reply_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP, -- tombstone for deletions
    vector_id VARCHAR(50), -- reference to vector DB
    
    UNIQUE(team_id, channel_id, message_ts),
    INDEX(team_id, channel_id, indexed_at),
    INDEX(thread_ts) WHERE thread_ts IS NOT NULL
);

-- User info cache
CREATE TABLE slack_users (
    team_id VARCHAR(20) NOT NULL,
    user_id VARCHAR(20) NOT NULL,
    username VARCHAR(50),
    display_name VARCHAR(100),
    is_admin BOOLEAN DEFAULT false,
    last_updated TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY(team_id, user_id)
);
```

**Vector Database Namespacing**:
```python
# Pinecone namespace pattern
namespace = f"team_{team_id}"

# Vector metadata structure
vector_metadata = {
    "team_id": "T1234567890",
    "channel_id": "C9876543210", 
    "message_ts": "1625123456.001500",
    "thread_ts": "1625123400.001200",  # null if not in thread
    "user_id": "U1111111111",
    "is_thread_root": False,
    "reply_depth": 2,
    "permalink": "https://workspace.slack.com/archives/C9876543210/p1625123456001500",
    "visibility": "public",  # or "private"
    "has_files": True,
    "indexed_at": "2024-01-15T10:30:00Z"
}
```

### Thread-Aware Context Management

**Thread Context Expansion**:
```python
async def expand_thread_context(self, core_messages: List[VectorMatch]) -> List[SlackMessage]:
    """
    Expand core results with thread neighbors for better context
    
    Strategy:
    1. Group messages by thread_ts
    2. For each thread, get ±3 messages around each core message  
    3. Deduplicate and maintain chronological order
    4. Respect token limits for prompt context
    """
    expanded = []
    
    # Group by thread
    threads = defaultdict(list)
    for msg in core_messages:
        thread_ts = msg.metadata.get("thread_ts") or msg.metadata["message_ts"]
        threads[thread_ts].append(msg)
    
    for thread_ts, thread_messages in threads.items():
        # Get complete thread
        full_thread = await self.db.query(
            """
            SELECT * FROM slack_messages 
            WHERE team_id = %s AND thread_ts = %s AND deleted_at IS NULL
            ORDER BY message_ts ASC
            """,
            (thread_messages[0].metadata["team_id"], thread_ts)
        )
        
        # Find positions of core messages
        core_positions = set()
        for core_msg in thread_messages:
            for i, full_msg in enumerate(full_thread):
                if full_msg.message_ts == core_msg.metadata["message_ts"]:
                    core_positions.add(i)
        
        # Expand with neighbors (±3 messages)
        expanded_positions = set()
        for pos in core_positions:
            for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(full_thread):
                    expanded_positions.add(neighbor_pos)
        
        # Add expanded messages
        for pos in sorted(expanded_positions):
            expanded.append(full_thread[pos])
    
    return expanded[:20]  # Limit to reasonable context size
```

## Security & Compliance Architecture

### Authentication Flow
```python
# Slack signature verification
def verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minute window
        return False
    
    sig_basestring = f"v0:{timestamp}:{request_body.decode()}"
    computed_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(computed_sig, signature)

# Token encryption for storage
def encrypt_token(token: str) -> str:
    f = Fernet(ENCRYPTION_KEY)
    return f.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    f = Fernet(ENCRYPTION_KEY)
    return f.decrypt(encrypted_token.encode()).decode()
```

### Data Access Control
```python
async def enforce_channel_acl(team_id: str, channel_id: str) -> bool:
    """Enforce channel access control lists"""
    installation = await get_installation(team_id)
    
    if not installation.channel_allowlist:
        # No allowlist = all channels allowed (if bot has access)
        return True
    
    return channel_id in installation.channel_allowlist

async def apply_retention_policy(team_id: str):
    """Remove messages older than retention policy"""
    installation = await get_installation(team_id)
    cutoff_date = datetime.utcnow() - timedelta(days=installation.retention_days)
    
    # Soft delete in PostgreSQL
    await db.execute(
        """
        UPDATE slack_messages 
        SET deleted_at = NOW() 
        WHERE team_id = %s AND indexed_at < %s AND deleted_at IS NULL
        """,
        (team_id, cutoff_date)
    )
    
    # Hard delete from vector DB
    old_messages = await db.fetch(
        """
        SELECT vector_id FROM slack_messages 
        WHERE team_id = %s AND deleted_at IS NOT NULL
        """,
        (team_id,)
    )
    
    if old_messages:
        vector_ids = [msg['vector_id'] for msg in old_messages]
        await vector_db.delete(namespace=f"team_{team_id}", ids=vector_ids)
```

## Deployment Architecture

### Single Container Deployment (MVP)
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start application
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose for Development
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/slack_rag
      - REDIS_URL=redis://redis:6379
      - SLACK_CLIENT_ID=${SLACK_CLIENT_ID}
      - SLACK_CLIENT_SECRET=${SLACK_CLIENT_SECRET}
      - SLACK_SIGNING_SECRET=${SLACK_SIGNING_SECRET}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - .:/app
      - /app/__pycache__
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=slack_rag
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  worker:
    build: .
    command: celery -A app.worker worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/slack_rag
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
  redis_data:
```

## Monitoring & Performance

### Key Metrics
```python
# Business metrics
QUERY_LATENCY = Histogram('slack_query_duration_seconds', 'Query processing time')
QUERY_SUCCESS_RATE = Counter('slack_queries_total', 'Query results', ['status', 'team_id'])
CITATION_ACCURACY = Gauge('citation_accuracy_ratio', 'Fraction of valid citations')

# Technical metrics
COHERE_API_CALLS = Counter('cohere_api_calls_total', 'Cohere API usage', ['model', 'operation'])
SLACK_RATE_LIMITS = Counter('slack_rate_limit_hits_total', 'Rate limit encounters')
VECTOR_DB_LATENCY = Histogram('vector_db_query_seconds', 'Vector search time')
INGESTION_LAG = Histogram('message_ingestion_lag_seconds', 'Event processing delay')

# Usage tracking
@QUERY_LATENCY.time()
async def process_query(team_id: str, query: str):
    try:
        result = await rag_service.answer_question(team_id, query)
        QUERY_SUCCESS_RATE.labels(status='success', team_id=team_id).inc()
        return result
    except Exception as e:
        QUERY_SUCCESS_RATE.labels(status='error', team_id=team_id).inc()
        raise
```

This simplified architecture balances practicality with scalability, providing a solid foundation for a Slack-native RAG system that can grow from MVP to enterprise scale.