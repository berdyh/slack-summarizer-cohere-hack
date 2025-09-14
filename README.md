# 🤖 Slack RAG Bot - AI-Powered Knowledge Assistant

A **hackathon-ready** Slack bot that brings **AI-powered question answering** to your workspace using **Cohere's RAG system**. Ask questions about your Slack history and get intelligent answers with source citations.

##  What It Does

- **Smart Search**: Ask questions about past Slack conversations in natural language
- **AI Responses**: Powered by Cohere's Command and Embed models for accurate, contextual answers
- **Source Citations**: Shows which messages the AI used to generate answers
- **Real-time**: Automatically indexes new messages as they're posted
- **Secure**: Proper Slack OAuth with signature verification

## Demo Commands

```
/ask What did we decide about the pricing strategy?
/ask Who was working on the mobile app features?
/ask What are the main concerns about the new deployment?
```

## Architecture

```
Slack Workspace → OAuth Install → Message Indexing → Cohere Embeddings 
                                      ↓
User Query → Vector Search → Context Assembly → Cohere Generation → Response
```

### Tech Stack
- **Backend**: FastAPI (Python 3.11+)
- **AI/ML**: Cohere (Embed v3.0 + Command R)
- **Integration**: Slack Web API + OAuth 2.0
- **Storage**: In-memory vectors (hackathon MVP)
- **Deployment**: Single container, ngrok for local dev

## Quick Start (3 minutes)

### 1. Prerequisites
```bash
# Install dependencies
pip install fastapi uvicorn httpx cohere numpy python-multipart

# Get Cohere API key from https://cohere.ai
# Create Slack App at https://api.slack.com/apps
```

### 2. Slack App Setup
Create a new Slack app with these scopes:
```
Bot Token Scopes:
- channels:history (read channel messages)
- channels:read (list channels)  
- chat:write (post responses)
- commands (handle /ask command)
```

Add Slash Command:
- **Command**: `/ask`
- **Request URL**: `https://your-ngrok-url.ngrok.io/slack/commands`
- **Description**: Ask questions about your Slack history

### 3. Environment Configuration
```bash
# Create .env file
SLACK_CLIENT_ID=your_client_id
SLACK_CLIENT_SECRET=your_client_secret  
SLACK_SIGNING_SECRET=your_signing_secret
COHERE_API_KEY=your_cohere_api_key
```

### 4. Run Locally
```bash
# Start the server
python main.py

# In another terminal, expose with ngrok
ngrok http 8000

# Update Slack app URLs with the ngrok HTTPS URL
```

### 5. Install & Test
1. Visit `http://localhost:8000` 
2. Click "Add to Slack"
3. Complete OAuth installation
4. Wait ~30 seconds for message indexing
5. In Slack, try: `/ask What have we been discussing lately?`

## 📱 User Experience

### Installation Flow
1. **Web Landing Page**: Simple "Add to Slack" button
2. **OAuth Authorization**: Standard Slack install process  
3. **Auto-Indexing**: Bot immediately starts indexing recent messages
4. **Ready to Use**: `/ask` command becomes available

### Query Experience
```
User: /ask What did Sarah mention about the budget?

Bot: Based on recent messages, Sarah mentioned that the Q4 budget 
needs to be finalized by Friday and we should prioritize the 
infrastructure costs. She was concerned about the hosting expenses 
going over the allocated amount.

Sources: Found 3 relevant messages
```

## Development

### Project Structure
```
slack-rag-bot/
├── main.py              # Single-file FastAPI app
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── README.md            # This file
└── docs/                # Planning documents
    ├── plan.md          # Implementation plan
    ├── architecture.md  # System architecture
    └── cursor-ai-prompt.md # Development guide
```

### Key Components

**OAuth & Installation**:
```python
@app.get("/slack/install") 
async def slack_install():
    # Redirects to Slack OAuth with proper scopes
    
@app.get("/slack/oauth")
async def slack_oauth(code: str):
    # Exchanges code for bot token, starts indexing
```

**Message Indexing**:
```python
async def index_workspace_messages(team_id: str, bot_token: str):
    # Fetches recent messages from accessible channels
    # Generates embeddings with Cohere Embed API
    # Stores vectors in memory with metadata
```

**RAG Pipeline**:
```python
async def answer_question(team_id: str, question: str) -> RAGResponse:
    # 1. Generate query embedding
    # 2. Find similar message vectors (cosine similarity)
    # 3. Assemble context from top matches
    # 4. Generate answer with Cohere Command
    # 5. Return formatted response with sources
```

### Slack Integration

**Slash Command Handler**:
```python
@app.post("/slack/commands")
async def slack_slash_command(team_id: str, text: str, command: str):
    # Verifies Slack signature
    # Processes /ask command
    # Returns formatted response
```

**Security**:
- Slack request signature verification
- OAuth token encryption (in production)
- Team-based data isolation
- Rate limiting and error handling

## Hackathon Demo Script

### Setup (2 minutes)
1. **Show Landing Page**: "Simple one-click installation"
2. **Install in Demo Workspace**: Click through OAuth flow
3. **Show Indexing**: "Bot automatically learns from your messages"

### Live Demo (3 minutes)
1. **Basic Query**: `/ask What projects are we working on?`
2. **Specific Question**: `/ask Who mentioned the deployment issues?`
3. **Show Sources**: Highlight how answers include citations
4. **Health Check**: Visit `/health` to show technical metrics

### Technical Deep Dive (2 minutes)
1. **AI Pipeline**: Explain Cohere embeddings → similarity search → generation
2. **Slack Integration**: Show proper OAuth, events, slash commands
3. **Extensibility**: "Easy to add more features like file search, threads"