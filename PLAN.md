# Slack-to-Cohere RAG Integration - MVP Implementation Plan

## Project Overview

This plan outlines building a **practical, Slack-native RAG system** that integrates with Cohere AI for intelligent content summarization and question-answering. Focus is on **MVP first** with proper Slack OAuth, multi-tenant architecture, and user-friendly Slack UX patterns.

## Core Philosophy: Slack-Native, Not Generic

- **Slack OAuth ("Add to Slack")** - Not generic login popups
- **Multi-tenant by design** - Each `team_id` is a separate tenant
- **Thread-aware indexing** - Respect Slack's conversation semantics
- **Events API-driven** - Real-time updates, not polling
- **Minimal viable architecture** - 3 services, not 7+ microservices

## Phase 0: Slack App Setup & OAuth (Week 1)

### 0.1 Slack App Configuration
- [ ] Create Slack App at api.slack.com with proper naming and description
- [ ] Configure OAuth scopes (minimal required):
  - `channels:history` - Read public channel history
  - `channels:read` - List public channels
  - `groups:history` - Read private channel history (when invited)
  - `files:read` - Download shared files
  - `app_mentions:read` - Detect @mentions
  - `chat:write` - Post responses
  - `commands` - Handle slash commands
- [ ] Set up redirect URLs for OAuth flow
- [ ] Configure Events API subscriptions:
  - `message.channels`, `message.groups`, `app_mention`
  - `message_changed`, `message_deleted` (handle edits/deletes)
  - `reaction_added`, `file_shared`

### 0.2 OAuth Implementation ("Add to Slack")
```python
# OAuth flow handler
@app.get("/slack/install")
async def install_slack():
    return RedirectResponse(
        f"https://slack.com/oauth/v2/authorize?"
        f"client_id={SLACK_CLIENT_ID}&"
        f"scope={OAUTH_SCOPES}&"
        f"redirect_uri={REDIRECT_URI}"
    )

@app.get("/slack/oauth/callback")
async def oauth_callback(code: str):
    # Exchange code for tokens
    response = slack_client.oauth_v2_access(
        client_id=SLACK_CLIENT_ID,
        client_secret=SLACK_CLIENT_SECRET,
        code=code,
        redirect_uri=REDIRECT_URI
    )
    
    # Store installation
    await store_installation(
        team_id=response["team"]["id"],
        installer_user_id=response["authed_user"]["id"],
        bot_token=response["access_token"],
        scopes=response["scope"],
        installation_time=datetime.utcnow()
    )
```

### 0.3 Multi-Tenant Data Model
```python
@dataclass
class SlackInstallation:
    team_id: str              # Primary tenant identifier
    installer_user_id: str    # Who installed the app
    bot_access_token: str     # Encrypted bot token
    scopes: List[str]         # Granted permissions
    installation_time: datetime
    channel_allowlist: List[str] = None  # Admin-selected channels
    retention_days: int = 90  # Data retention policy
    is_active: bool = True
```

## Phase 1: Slack Integration & Ingestion (Week 2-3)

### 1.1 Events API Handler with Signature Verification
```python
import hmac
import hashlib
from fastapi import HTTPException

@app.post("/slack/events")
async def handle_slack_events(request: Request):
    body = await request.body()
    
    # Verify Slack signature
    slack_signature = request.headers.get("X-Slack-Signature")
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    
    if not verify_slack_signature(body, timestamp, slack_signature):
        raise HTTPException(401, "Invalid signature")
    
    event = await request.json()
    
    # Handle URL verification
    if event.get("type") == "url_verification":
        return {"challenge": event["challenge"]}
    
    # Queue event processing
    await queue_event_processing(event)
    return {"status": "ok"}
```

### 1.2 Thread-Aware Message Processing
```python
class SlackMessageProcessor:
    async def process_message_event(self, event: Dict):
        team_id = event["team_id"]
        message = event["event"]
        
        # Handle different message types
        if message.get("subtype") == "message_changed":
            await self.handle_message_edit(team_id, message)
        elif message.get("subtype") == "message_deleted":
            await self.handle_message_deletion(team_id, message)
        else:
            await self.index_new_message(team_id, message)
    
    async def index_new_message(self, team_id: str, message: Dict):
        # Extract thread context
        thread_ts = message.get("thread_ts", message["ts"])
        is_thread_root = thread_ts == message["ts"]
        
        # Build semantic unit (turn-based)
        semantic_unit = SlackSemanticUnit(
            team_id=team_id,
            channel_id=message["channel"],
            message_ts=message["ts"],
            thread_ts=thread_ts,
            user_id=message["user"],
            text=message["text"],
            is_thread_root=is_thread_root,
            reply_depth=await self.calculate_reply_depth(thread_ts, message["ts"]),
            permalink=await self.build_permalink(team_id, message),
            visibility="public" if message["channel"].startswith("C") else "private"
        )
        
        # Process files if present
        if "files" in message:
            for file in message["files"]:
                file_text = await self.extract_file_text(team_id, file)
                semantic_unit.attachments.append(file_text)
        
        # Generate embeddings and store
        await self.embed_and_store(semantic_unit)
```

### 1.3 Backfill Strategy for New Installs
```python
async def backfill_workspace_history(team_id: str):
    installation = await get_installation(team_id)
    bot_token = installation.bot_access_token
    
    # Get channels (respect allowlist if set)
    channels = await get_allowed_channels(team_id)
    
    for channel in channels:
        print(f"Backfilling {channel['name']} ({channel['id']})")
        
        # Fetch history with pagination
        cursor = None
        while True:
            response = slack_client.conversations_history(
                token=bot_token,
                channel=channel["id"],
                limit=200,
                cursor=cursor,
                oldest=get_retention_cutoff(team_id)
            )
            
            # Process messages in batches
            await process_message_batch(team_id, response["messages"])
            
            if not response.get("has_more"):
                break
            cursor = response["response_metadata"]["next_cursor"]
            
            # Rate limiting
            await asyncio.sleep(1)
```

### 1.4 File Handling with Authorized Downloads
```python
async def extract_file_text(self, team_id: str, file: Dict) -> str:
    """Extract text from Slack files with proper authorization"""
    if file["mimetype"].startswith("text/"):
        # Download text files
        installation = await get_installation(team_id)
        headers = {"Authorization": f"Bearer {installation.bot_access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(file["url_private"], headers=headers)
            return response.text
            
    elif file["mimetype"] == "application/pdf":
        # OCR/PDF extraction
        return await extract_pdf_text(file["url_private"], team_id)
        
    else:
        # Store metadata only
        return f"[File: {file['name']} ({file['mimetype']})]"
```

## Phase 2: Retrieval & RAG Implementation (Week 4-5)

### 2.1 Hybrid Retrieval Pipeline
```python
class SlackRAGService:
    async def answer_question(
        self, 
        team_id: str, 
        query: str, 
        channel_filter: str = None,
        time_window_days: int = 90
    ) -> RAGResponse:
        
        # 1. Pre-filter by tenant + ACLs + time
        base_filters = {
            "team_id": team_id,
            "timestamp": {"$gte": get_time_cutoff(time_window_days)},
            "deleted_at": {"$exists": False}  # Exclude tombstones
        }
        
        if channel_filter:
            base_filters["channel_id"] = channel_filter
            
        # Verify channel access
        allowed_channels = await get_allowed_channels(team_id)
        if channel_filter and channel_filter not in allowed_channels:
            raise PermissionError("Channel not in allowlist")
        
        # 2. BM25 keyword pre-filter (fast)
        keyword_candidates = await self.bm25_search(
            query, filters=base_filters, top_k=50
        )
        
        # 3. Semantic similarity with Cohere Embed
        query_embedding = await self.cohere.embed([query], input_type="search_query")
        
        semantic_results = await self.vector_db.similarity_search(
            namespace=team_id,
            vector=query_embedding[0],
            filter=base_filters,
            top_k=20,
            include_metadata=True
        )
        
        # 4. Cohere Rerank for relevance
        documents = [r.metadata["text"] for r in semantic_results]
        reranked = self.cohere.rerank(
            query=query,
            documents=documents,
            top_k=5,
            model="rerank-english-v3.0"
        )
        
        # 5. Thread expansion (get neighboring messages)
        expanded_context = await self.expand_thread_context(
            [semantic_results[r.index] for r in reranked.results]
        )
        
        # 6. Generate response with citations
        response = await self.generate_with_citations(query, expanded_context)
        
        return response
```

### 2.2 Thread Context Expansion
```python
async def expand_thread_context(self, core_messages: List) -> List[SlackSemanticUnit]:
    """Expand with ±3 messages in same thread for context"""
    expanded = []
    
    for msg in core_messages:
        thread_ts = msg.metadata["thread_ts"]
        
        # Get thread neighbors
        neighbors = await self.vector_db.query(
            namespace=msg.metadata["team_id"],
            filter={
                "thread_ts": thread_ts,
                "deleted_at": {"$exists": False}
            },
            sort=[("message_ts", 1)],
            limit=20  # Reasonable thread size
        )
        
        # Find position and expand ±3
        msg_index = next(i for i, n in enumerate(neighbors) 
                        if n.metadata["message_ts"] == msg.metadata["message_ts"])
        
        start = max(0, msg_index - 3)
        end = min(len(neighbors), msg_index + 4)
        
        expanded.extend(neighbors[start:end])
    
    return deduplicate_by_ts(expanded)
```

## Phase 3: Slack UX Implementation (Week 6-7)

### 3.1 Slash Commands
```python
@app.post("/slack/commands/ask")
async def handle_ask_command(request: Request):
    form = await request.form()
    team_id = form["team_id"]
    user_id = form["user_id"]
    text = form["text"]  # Query + optional params
    
    # Parse command: /ask <question> [#channel] [last 30d]
    query, channel, days = parse_ask_command(text)
    
    # Show ephemeral preview first
    preview_response = await rag_service.answer_question(
        team_id, query, channel, days
    )
    
    preview_blocks = build_preview_blocks(preview_response)
    
    return {
        "response_type": "ephemeral",
        "blocks": preview_blocks + [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Post Answer"},
                        "style": "primary",
                        "value": f"post_answer:{hash(query)}"
                    },
                    {
                        "type": "button", 
                        "text": {"type": "plain_text", "text": "Edit Query"},
                        "value": f"edit_query:{hash(query)}"
                    }
                ]
            }
        ]
    }
```

### 3.2 Message Actions
```python
@app.post("/slack/actions")
async def handle_message_actions(request: Request):
    payload = json.loads((await request.form())["payload"])
    
    if payload["callback_id"] == "summarize_thread":
        thread_ts = payload["message"]["thread_ts"] or payload["message"]["ts"]
        team_id = payload["team"]["id"]
        
        # Generate thread summary
        summary = await rag_service.summarize_thread(team_id, thread_ts)
        
        # Post as threaded reply
        await slack_client.chat_postMessage(
            token=get_bot_token(team_id),
            channel=payload["channel"]["id"],
            thread_ts=thread_ts,
            blocks=build_summary_blocks(summary)
        )
```

### 3.3 App Home (Settings & Channel Management)
```python
@app.post("/slack/events")  # app_home_opened event
async def handle_app_home(event: Dict):
    user_id = event["user"]
    team_id = event["team_id"]
    
    # Check if user is admin
    is_admin = await check_admin_permissions(team_id, user_id)
    
    if is_admin:
        home_view = build_admin_home_view(team_id)
    else:
        home_view = build_user_home_view(team_id)
    
    await slack_client.views_publish(
        token=get_bot_token(team_id),
        user_id=user_id,
        view=home_view
    )

def build_admin_home_view(team_id: str) -> Dict:
    installation = await get_installation(team_id)
    
    return {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🤖 RAG Assistant Settings"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Channel Allowlist*\nSelect which channels the bot can access:"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "multi_channels_select",
                        "placeholder": {"type": "plain_text", "text": "Choose channels..."},
                        "initial_channels": installation.channel_allowlist or [],
                        "action_id": "update_allowlist"
                    }
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Data Retention*\nCurrently: {installation.retention_days} days"}
            },
            # Usage statistics, cost monitoring, etc.
        ]
    }
```

## Phase 4: Operations & Evaluation (Week 8)

### 4.1 Monitoring & Metrics
```python
# Key metrics to track
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
slack_queries_total = Counter('slack_queries_total', 'Total queries', ['team_id', 'channel'])
response_quality = Histogram('response_quality_score', 'User-rated response quality')
citation_accuracy = Gauge('citation_accuracy', 'Fraction of citations that are valid')

# Technical metrics  
cohere_api_cost = Counter('cohere_api_cost_usd', 'Cohere API costs', ['model', 'operation'])
slack_rate_limits = Counter('slack_rate_limits_hit', 'Rate limit encounters', ['method'])
ingestion_lag = Histogram('message_ingestion_lag_seconds', 'Time from Slack event to indexed')
```

### 4.2 Evaluation Framework
```python
@dataclass
class EvalQuestion:
    query: str
    expected_sources: List[str]  # Message permalinks
    expected_answer_contains: List[str]  # Key phrases
    team_id: str
    channel_id: str

# Gold standard evaluation set
EVAL_QUESTIONS = [
    EvalQuestion(
        query="What did Sarah decide about the pricing strategy?",
        expected_sources=["https://workspace.slack.com/archives/C123/p1625123456"],
        expected_answer_contains=["pricing", "Sarah", "strategy"],
        team_id="T123",
        channel_id="C123"
    )
]

async def run_evaluation():
    results = []
    for q in EVAL_QUESTIONS:
        response = await rag_service.answer_question(
            q.team_id, q.query, q.channel_id
        )
        
        # Check if expected sources are cited
        source_recall = len(set(response.source_urls) & set(q.expected_sources)) / len(q.expected_sources)
        
        # Check answer quality
        contains_keywords = all(kw.lower() in response.answer.lower() for kw in q.expected_answer_contains)
        
        results.append({
            "query": q.query,
            "source_recall": source_recall,
            "contains_keywords": contains_keywords,
            "response_time": response.processing_time
        })
    
    return results
```

### 4.3 Admin Tools & Data Management
```python
# Tenant data export (GDPR compliance)
@app.post("/admin/export/{team_id}")
async def export_team_data(team_id: str, admin_token: str):
    verify_admin_token(admin_token)
    
    # Export all indexed messages
    messages = await vector_db.query(
        namespace=team_id,
        filter={"deleted_at": {"$exists": False}},
        include_metadata=True
    )
    
    # Export installation data
    installation = await get_installation(team_id)
    
    export_data = {
        "installation": installation.dict(),
        "indexed_messages": len(messages),
        "export_time": datetime.utcnow().isoformat()
    }
    
    return export_data

# Tenant data deletion
@app.delete("/admin/purge/{team_id}")
async def purge_team_data(team_id: str, admin_token: str):
    verify_admin_token(admin_token)
    
    # Delete from vector DB
    await vector_db.delete(namespace=team_id, delete_all=True)
    
    # Delete installation record
    await delete_installation(team_id)
    
    return {"status": "purged", "team_id": team_id}
```

## Simplified Architecture (3 Services, Not 7+)

### Service A: Ingestion & Sync Worker
- **Purpose**: Handle Slack Events, process messages, manage files
- **Components**: 
  - Event webhook handler with signature verification
  - Background job queue (Redis/Celery)
  - Message processor with thread awareness
  - File downloader with OCR/text extraction
  - Vector DB writer

### Service B: Query API  
- **Purpose**: Handle queries, RAG pipeline, response generation
- **Components**:
  - FastAPI REST endpoints
  - Hybrid retrieval (BM25 + vector similarity)
  - Cohere integration (embed, rerank, generate)
  - Citation formatter with Slack permalinks

### Service C: Slack Frontend
- **Purpose**: Slack app UX (slash commands, App Home, message actions)
- **Components**:
  - Slash command handlers
  - Interactive message handlers
  - App Home views for settings
  - OAuth installation flow

## Technology Stack (Minimal)

### Core Requirements
```txt
# Backend
fastapi>=0.104.0
slack-sdk>=3.26.0
cohere>=4.37.0
redis>=5.0.0
asyncpg>=0.29.0  # PostgreSQL for metadata
pinecone-client>=2.2.4  # or weaviate-client
pydantic>=2.5.0

# Utilities
python-jose[cryptography]>=3.3.0  # JWT
httpx>=0.25.0  # HTTP client
celery[redis]>=5.3.0  # Background jobs
prometheus-client>=0.19.0  # Metrics

# Text processing
pypdf>=3.17.0  # PDF text extraction
python-docx>=0.8.11  # Word doc extraction
```

### Infrastructure Options

**Option 1: Single Container (MVP)**
- FastAPI + Redis + PostgreSQL + Celery worker
- Deploy on Railway, Fly.io, or similar

**Option 2: Serverless (Scale)**
- Query API: Vercel/Netlify functions
- Ingestion: AWS Lambda + SQS
- Storage: Pinecone + PlanetScale

**Option 3: Traditional VPS**
- Docker Compose on DigitalOcean/Linode
- Add Kubernetes only when traffic demands it

## Success Metrics (MVP Focus)

### User Satisfaction
- **Response relevance**: >80% of answers cite correct messages
- **Response time**: <3 seconds for 95% of queries
- **User adoption**: >50% of workspace members try it within first week

### Technical Health
- **Uptime**: >99% availability
- **Ingestion lag**: <30 seconds from Slack event to searchable
- **Cost efficiency**: <$0.10 per query (Cohere API + infrastructure)

### Business Validation
- **Retention**: >60% of teams use it weekly after first month
- **Growth**: Organic "Add to Slack" installs from directory
- **Feedback**: Positive reviews in Slack App Directory

## Security Checklist

- [ ] **Slack signature verification** for all webhooks
- [ ] **Encrypted token storage** with key rotation
- [ ] **Tenant isolation** (namespace by team_id)
- [ ] **Channel ACL enforcement** at query and ingestion time
- [ ] **Audit logging** (who asked what, which channels accessed)
- [ ] **PII detection** and optional redaction before embedding
- [ ] **Data retention policies** with automatic purging
- [ ] **HTTPS everywhere** with proper certificate management

## Future Enhancements (Post-MVP)

### Phase 5: Advanced Features
- Socket Mode for behind-firewall deployments
- Enterprise Grid support for large orgs  
- Advanced summarization (daily/weekly digests)
- Multi-language support with language detection
- Custom model fine-tuning on workspace data

### Phase 6: Enterprise
- SSO integration (SAML/OIDC)
- Advanced compliance (SOC 2, HIPAA)
- On-premise deployment options
- Advanced analytics and insights
- Integration with other tools (Notion, Confluence)

This MVP-focused plan prioritizes **working software over comprehensive architecture** while maintaining proper Slack integration patterns and multi-tenant security. The goal is to have a usable app that workspace admins can install and users love within 8 weeks.