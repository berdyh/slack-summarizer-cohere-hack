"""
Slack-to-Cohere RAG Integration - MVP Implementation
Following the architecture from architecture.md and plan from PLAN.md

This is the main FastAPI application that implements the 3-service architecture:
- Service A: Ingestion & Sync Worker (message processing, embeddings)
- Service B: Query API (RAG pipeline, response generation)  
- Service C: Slack Frontend (OAuth, slash commands, UX)

Run with: uvicorn main:app --reload --port 8000
"""

import os
import json
import time
import asyncio
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Core dependencies
from fastapi import FastAPI, Request, HTTPException, Form, Depends, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import cohere
from pydantic import BaseModel

# Import HTML response functions
from html_responses import home_page, oauth_success_response, oauth_error_response

# Import Phase 4 modules
from monitoring import (
    metrics_collector, log_installation_event, log_security_event, 
    log_performance_issue, estimate_cohere_cost, get_metrics_response
)
from evaluation import RAGEvaluator, run_automated_evaluation, generate_evaluation_report
from admin_tools import (
    DataExporter, DataManager, SystemHealthMonitor, AuditManager,
    generate_admin_token, verify_admin_token, log_admin_action
)

# Import data models
from models import SlackInstallation, SlackUser, SlackMessage, RAGResponse

# Import Phase 5 database module
from database import db_manager, get_installation, store_installation, get_user, store_user

# Environment setup
SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET") 
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Validate required environment variables
if not all([SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, SLACK_SIGNING_SECRET, COHERE_API_KEY]):
    raise ValueError("Missing required environment variables. Check SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, SLACK_SIGNING_SECRET, COHERE_API_KEY")

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Lifespan event handler for database initialization
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    try:
        await db_manager.initialize()
        print("Database initialization complete")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Continue with in-memory fallback for development
    yield
    # Cleanup code can go here if needed

# Initialize services
app = FastAPI(
    title="Slack RAG Bot", 
    version="0.1.0",
    description="AI-powered question answering for Slack workspaces using Cohere RAG",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (will be replaced with proper database in production)
INSTALLATIONS = {}  # team_id -> SlackInstallation
MESSAGE_VECTORS = defaultdict(list)  # team_id -> list of (vector, metadata)
VECTOR_CACHE = {}  # text_hash -> vector for caching
USER_CACHE = {}  # (team_id, user_id) -> SlackUser

# === Data Models ===
# Models are now imported from models.py to avoid circular imports

# === Phase 0: OAuth Implementation ===

@app.get("/", response_class=HTMLResponse)
async def home():
    """Landing page with 'Add to Slack' button"""
    return home_page()

@app.get("/slack/install") 
async def slack_install():
    """Redirect to Slack OAuth with proper scopes"""
    # Minimal required scopes as per PLAN.md
    scope = "channels:history,channels:read,groups:history,files:read,app_mentions:read,chat:write,commands"
    redirect_uri = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/slack/oauth"
    
    oauth_url = (
        f"https://slack.com/oauth/v2/authorize?"
        f"client_id={SLACK_CLIENT_ID}&"
        f"scope={scope}&"
        f"redirect_uri={redirect_uri}"
    )
    return RedirectResponse(oauth_url)

@app.get("/slack/oauth")
async def slack_oauth(code: str):
    """Handle OAuth callback and install app"""
    print(f"OAuth callback received with code: {code[:20]}...")
    redirect_uri = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/slack/oauth"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://slack.com/api/oauth.v2.access", data={
                "client_id": SLACK_CLIENT_ID,
                "client_secret": SLACK_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri
            })
        
        data = response.json()
        print(f"OAuth response: {data}")
        
        if not data.get("ok"):
            error_msg = data.get('error', 'Unknown error occurred')
            print(f"OAuth failed: {error_msg}")
            return oauth_error_response(f"OAuth failed: {error_msg}")
        
        team_id = data["team"]["id"]
        team_domain = data["team"].get("domain", "workspace")  # Extract team domain
        bot_token = data["access_token"]
        installer_user_id = data["authed_user"]["id"]
        scopes = data["scope"].split(",")
        
        print(f"Creating installation for team: {team_id} (domain: {team_domain})")
        
        # Create installation record
        installation = SlackInstallation(
            team_id=team_id,
            team_domain=team_domain,
            installer_user_id=installer_user_id,
            bot_access_token=bot_token,
            scopes=scopes,
            installation_time=datetime.now(),
            settings={}
        )
        
        # Store installation in database and in-memory (for backward compatibility)
        try:
            await store_installation(installation)
        except Exception as e:
            print(f"Failed to store installation in database: {e}")
            print("Continuing with in-memory storage only")
        
        INSTALLATIONS[team_id] = installation
        print(f"Installation stored. Total installations: {len(INSTALLATIONS)}")
        
        # Start background indexing
        asyncio.create_task(index_workspace_messages(team_id, bot_token))
        
        return oauth_success_response(data)
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        import traceback
        traceback.print_exc()
        return oauth_error_response(f"OAuth callback error: {str(e)}")

# === Message Indexing (Part of Service A) ===

async def index_workspace_messages(team_id: str, bot_token: str):
    """Enhanced backfill strategy for new installations"""
    print(f"Starting backfill indexing for team {team_id}")
    
    try:
        async with httpx.AsyncClient() as client:
            # Get team info for domain
            team_info_response = await client.get(
                "https://slack.com/api/team.info",
                headers={"Authorization": f"Bearer {bot_token}"}
            )
            team_info = team_info_response.json()
            team_domain = team_info.get("team", {}).get("domain", "workspace")
            
            # Get channels with pagination
            channels = []
            cursor = None
            
            while True:
                channels_response = await client.get(
                    "https://slack.com/api/conversations.list",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    params={
                        "types": "public_channel",
                        "limit": 100,
                        "cursor": cursor
                    }
                )
                
                channels_data = channels_response.json()
                if not channels_data.get("ok"):
                    print(f"Failed to get channels: {channels_data.get('error')}")
                    break
                
                channels.extend(channels_data.get("channels", []))
                
                # Check if we have more channels
                if not channels_data.get("response_metadata", {}).get("next_cursor"):
                    break
                cursor = channels_data["response_metadata"]["next_cursor"]
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Limit to first 10 channels for MVP
            channels = channels[:10]
            print(f"Found {len(channels)} channels to index")
            
            # Debug: Show channel details
            accessible_channels = []
            for channel in channels:
                is_member = channel.get('is_member', False)
                print(f"  - #{channel['name']} (ID: {channel['id']}) - Member: {is_member}")
                if is_member:
                    accessible_channels.append(channel)
            
            if not accessible_channels:
                print("No accessible channels found. The bot needs to be invited to channels to read messages.")
                print("To fix this:")
                print("   1. Go to any channel in Slack")
                print("   2. Type: /invite @YourBotName")
                print("   3. Or add the bot to channels via channel settings")
                print("   4. Then restart the indexing by visiting the help page")
                
                # Post a message to the installer about this issue
                try:
                    installation = INSTALLATIONS.get(team_id)
                    if installation:
                        await post_slack_message(
                            installation.bot_access_token,
                            installation.installer_user_id,  # DM the installer
                            "**RAG Bot Setup Required**\n\n"
                            "I've been installed but I need to be invited to channels to read messages.\n\n"
                            "**To get started:**\n"
                            "1. Go to any channel where you want me to read messages\n"
                            "2. Type: `/invite @YourBotName`\n"
                            "3. Or add me via channel settings\n"
                            "4. Then try asking me a question!\n\n"
                            "**Example:** `@YourBotName What have we been discussing lately?`"
                        )
                except Exception as e:
                    print(f"Failed to send setup message: {e}")
                return
            
            all_messages = []
            
            # Get recent messages from each accessible channel with pagination
            for channel in accessible_channels:
                channel_id = channel["id"]
                channel_name = channel["name"]
                
                print(f"Indexing #{channel_name}")
                
                # Get channel history with pagination
                cursor = None
                channel_messages = []
                
                while True:
                    history_response = await client.get(
                        "https://slack.com/api/conversations.history",
                        headers={"Authorization": f"Bearer {bot_token}"},
                        params={
                            "channel": channel_id,
                            "limit": 200,
                            "cursor": cursor,
                            "oldest": (time.time() - 30*24*3600)  # Last 30 days
                        }
                    )
                    
                    history_data = history_response.json()
                    if not history_data.get("ok"):
                        print(f"Failed to get history for #{channel_name}: {history_data.get('error')}")
                        break
                    
                    messages = history_data.get("messages", [])
                    print(f"Retrieved {len(messages)} messages from #{channel_name}")
                    channel_messages.extend(messages)
                    
                    # Check if we have more messages
                    if not history_data.get("response_metadata", {}).get("next_cursor"):
                        break
                    cursor = history_data["response_metadata"]["next_cursor"]
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                
                # Process messages
                print(f"📝 Processing {len(channel_messages)} messages from #{channel_name}")
                for msg in channel_messages:
                    print(f"  - Message type: {msg.get('type')}, subtype: {msg.get('subtype')}, bot_id: {msg.get('bot_id')}")
                    print(f"  - Text length: {len(msg.get('text', ''))}")
                    print(f"  - Text preview: {msg.get('text', '')[:100]}...")
                    
                    # More lenient filtering - include more message types
                    if (msg.get("type") == "message" and 
                        "text" in msg and 
                        len(msg["text"].strip()) > 5 and  # Lower threshold
                        msg.get("subtype") not in ["bot_message", "channel_join", "channel_leave"] and  # Skip only specific subtypes
                        not msg.get("bot_id")):  # Skip bot messages
                        
                        # Build permalink
                        permalink = f"https://{team_domain}.slack.com/archives/{channel_id}/p{msg['ts'].replace('.', '')}"
                        
                        all_messages.append(SlackMessage(
                            team_id=team_id,
                            channel_id=channel_id,
                            user_id=msg.get("user", "unknown"),
                            text=msg["text"],
                            ts=msg["ts"],
                            thread_ts=msg.get("thread_ts"),
                            has_files=bool(msg.get("files")),
                            permalink=permalink
                        ))
                        print(f"  Added message: {msg['text'][:50]}...")
                    else:
                        # More detailed skip reasons
                        skip_reasons = []
                        if msg.get("type") != "message":
                            skip_reasons.append(f"type={msg.get('type')}")
                        if "text" not in msg:
                            skip_reasons.append("no text")
                        elif len(msg["text"].strip()) <= 5:
                            skip_reasons.append(f"text too short ({len(msg['text'].strip())} chars)")
                        if msg.get("subtype") in ["bot_message", "channel_join", "channel_leave"]:
                            skip_reasons.append(f"subtype={msg.get('subtype')}")
                        if msg.get("bot_id"):
                            skip_reasons.append("bot message")
                        
                        print(f"  Skipped message: {msg.get('text', '')[:50]}... (reasons: {', '.join(skip_reasons)})")
            
            print(f"Processing {len(all_messages)} messages for embedding")
            
            # Generate embeddings in batches
            if all_messages:
                await generate_and_store_embeddings_batch(team_id, all_messages)
            
            print(f"Backfill indexing complete for team {team_id}")
            
    except Exception as e:
        print(f"Backfill indexing failed for team {team_id}: {e}")

async def generate_and_store_embeddings_batch(team_id: str, messages: List[SlackMessage]):
    """Generate embeddings in batches to handle large message sets"""
    batch_size = 50  # Process in batches of 50
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        texts = [msg.text for msg in batch]
        
        try:
            # Generate embeddings with Cohere
            response = cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            
            embeddings = response.embeddings
            print(f"Generated {len(embeddings)} embeddings for batch {i//batch_size + 1}")
            
            # Store in memory with metadata
            for msg, embedding in zip(batch, embeddings):
                vector_data = {
                    "vector": embedding,
                    "message": asdict(msg),
                    "full_text": msg.text,  # Store original text
                    "indexed_at": datetime.now().isoformat()
                }
                MESSAGE_VECTORS[team_id].append(vector_data)
            
            # Store in LanceDB
            try:
                await db_manager.store_message_vectors(team_id, batch, [np.array(emb) for emb in embeddings])
            except Exception as e:
                print(f"Failed to store vectors in LanceDB: {e}")
                print("Continuing with in-memory storage only")
            
            # Rate limiting between batches
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Embedding generation failed for batch {i//batch_size + 1}: {e}")
    
    print(f"Stored {len(messages)} vectors for team {team_id}")

async def generate_and_store_embeddings(team_id: str, messages: List[SlackMessage]):
    """Generate embeddings and store in memory"""
    texts = [msg.text for msg in messages]
    
    try:
        # Generate embeddings with Cohere
        response = cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        embeddings = response.embeddings
        print(f"Generated {len(embeddings)} embeddings")
        
        # Store in memory with metadata
        for msg, embedding in zip(messages, embeddings):
            vector_data = {
                "vector": embedding,
                "message": asdict(msg),
                "indexed_at": datetime.utcnow().isoformat()
            }
            MESSAGE_VECTORS[team_id].append(vector_data)
        
        print(f"Stored {len(embeddings)} vectors for team {team_id}")
        
    except Exception as e:
        print(f"Embedding generation failed: {e}")

# === Phase 1: Slack Events API Handler ===

def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify Slack request signature for security"""
    if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minute window
        return False
    
    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    computed_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(computed_sig, signature)

@app.post("/slack/events")
async def handle_slack_events(request: Request):
    """Handle Slack Events API with signature verification"""
    try:
        body = await request.body()
        print(f"Received request body: {body.decode()}")
        
        # Handle URL verification FIRST (before signature verification)
        if body:
            try:
                event_data = json.loads(body)
                print(f"Parsed event data: {event_data}")
                
                if event_data.get("type") == "url_verification":
                    challenge = event_data.get("challenge")
                    print(f"URL verification challenge: {challenge}")
                    return {"challenge": challenge}
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {"error": "Invalid JSON"}
        
        # Verify Slack signature for actual events
        slack_signature = request.headers.get("x-slack-signature", "")
        slack_timestamp = request.headers.get("x-slack-request-timestamp", "")
        
        if not verify_slack_signature(body, slack_timestamp, slack_signature):
            print(f"Invalid signature: {slack_signature}")
            raise HTTPException(401, "Invalid signature")
        
        # Parse event data for actual events
        event_data = json.loads(body)
        
        # Queue event processing for async handling
        if event_data.get("type") == "event_callback":
            event = event_data.get("event", {})
            team_id = event_data.get("team_id")
            
            # Process different event types
            if event.get("type") == "message":
                asyncio.create_task(process_message_event(team_id, event))
            elif event.get("type") == "app_mention":
                asyncio.create_task(process_app_mention_event(team_id, event))
            elif event.get("type") == "app_home_opened":
                # Handle App Home opened event
                user_id = event.get("user")
                if team_id and user_id:
                    asyncio.create_task(handle_app_home_opened(team_id, user_id))
            elif event.get("type") == "member_joined_channel":
                # Handle when bot gets added to a channel
                asyncio.create_task(handle_member_joined_channel(team_id, event))
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"Error in handle_slack_events: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")

async def process_message_event(team_id: str, event: Dict):
    """Process incoming message events with thread awareness"""
    print(f"Processing message event for team: {team_id}")
    print(f"Current installations: {list(INSTALLATIONS.keys())}")
    
    if team_id not in INSTALLATIONS:
        print(f"Team {team_id} not found in installations")
        return
    
    # Handle different message types
    if event.get("subtype") == "message_changed":
        await handle_message_edit(team_id, event)
    elif event.get("subtype") == "message_deleted":
        await handle_message_deletion(team_id, event)
    else:
        await index_new_message(team_id, event)

async def index_new_message(team_id: str, event: Dict):
    """Index a new message with thread context"""
    # Skip bot messages and messages without text
    # Only skip specific subtypes, not all subtypes
    skip_subtypes = ["bot_message", "channel_join", "channel_leave", "channel_topic", "channel_purpose"]
    if (event.get("subtype") in skip_subtypes or 
        event.get("bot_id") or 
        not event.get("text") or 
        len(event["text"].strip()) < 10):
        print(f"Skipping message: subtype={event.get('subtype')}, bot_id={event.get('bot_id')}, text_len={len(event.get('text', ''))}")
        return
    
    print(f"Indexing message: {event.get('text', '')[:100]}... (subtype: {event.get('subtype')}, user: {event.get('user')})")
    
    # Extract thread context
    thread_ts = event.get("thread_ts", event["ts"])
    is_thread_root = thread_ts == event["ts"]
    
    # Create SlackMessage object with thread awareness
    installation = INSTALLATIONS[team_id]
    # Use team_domain if available, otherwise use a default
    team_domain = getattr(installation, 'team_domain', 'workspace')
    permalink = f"https://{team_domain}.slack.com/archives/{event['channel']}/p{event['ts'].replace('.', '')}"
    
    message = SlackMessage(
        team_id=team_id,
        channel_id=event["channel"],
        user_id=event.get("user", "unknown"),
        text=event["text"],
        ts=event["ts"],
        thread_ts=thread_ts,
        has_files=bool(event.get("files")),
        message_type="message",
        permalink=permalink
    )
    
    # Generate embedding and store
    await process_single_message(message)

async def index_channel_history(team_id: str, channel_id: str):
    """Index history for a specific channel when bot is added"""
    if team_id not in INSTALLATIONS:
        return
    
    installation = INSTALLATIONS[team_id]
    bot_token = installation.bot_access_token
    
    print(f"Starting to index channel {channel_id} history for team {team_id}")
    
    try:
        # Get channel info
        channel_info = await get_channel_info(bot_token, channel_id)
        channel_name = channel_info.get("name", channel_id)
        
        print(f"Indexing channel #{channel_name} ({channel_id})")
        
        # Get channel messages
        messages = await get_channel_messages(bot_token, channel_id, limit=1000)
        
        if not messages:
            print(f"No messages found in channel {channel_id}")
            return
        
        print(f"Found {len(messages)} messages in channel {channel_id}")
        
        # Process messages in batches
        batch_size = 50
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            await generate_and_store_embeddings_batch(team_id, batch)
            print(f"Processed batch {i//batch_size + 1}/{(len(messages) + batch_size - 1)//batch_size}")
        
        # Post completion message
        await post_slack_message(
            bot_token,
            channel_id,
            f"Channel indexing complete! I've indexed {len(messages)} messages from #{channel_name}. You can now ask me questions about this channel's history."
        )
        
        print(f"Completed indexing channel {channel_id} with {len(messages)} messages")
        
    except Exception as e:
        print(f"Error indexing channel {channel_id}: {e}")
        # Post error message to channel
        try:
            await post_slack_message(
                bot_token,
                channel_id,
                f"Sorry, I encountered an error while indexing this channel's history: {str(e)}"
            )
        except:
            pass

async def handle_message_edit(team_id: str, event: Dict):
    """Handle message edits by updating the existing message"""
    message_ts = event["message"]["ts"]
    new_text = event["message"]["text"]
    
    # Find and update the message in our storage
    if team_id in MESSAGE_VECTORS:
        for i, vector_data in enumerate(MESSAGE_VECTORS[team_id]):
            if vector_data["message"]["ts"] == message_ts:
                # Update the message text
                MESSAGE_VECTORS[team_id][i]["message"]["text"] = new_text
                
                # Regenerate embedding for the updated text
                try:
                    response = cohere_client.embed(
                        texts=[new_text],
                        model="embed-english-v3.0",
                        input_type="search_document"
                    )
                    MESSAGE_VECTORS[team_id][i]["vector"] = response.embeddings[0]
                    MESSAGE_VECTORS[team_id][i]["indexed_at"] = datetime.now().isoformat()
                    print(f"Updated message {message_ts} for team {team_id}")
                except Exception as e:
                    print(f"Failed to update embedding for message {message_ts}: {e}")
                break

async def handle_message_deletion(team_id: str, event: Dict):
    """Handle message deletions by removing from storage"""
    message_ts = event["deleted_ts"]
    
    # Remove the message from our storage
    if team_id in MESSAGE_VECTORS:
        MESSAGE_VECTORS[team_id] = [
            vector_data for vector_data in MESSAGE_VECTORS[team_id]
            if vector_data["message"]["ts"] != message_ts
        ]
        print(f"Deleted message {message_ts} for team {team_id}")

async def handle_member_joined_channel(team_id: str, event: Dict):
    """Handle when bot gets added to a channel"""
    if team_id not in INSTALLATIONS:
        return
    
    user_id = event.get("user")
    channel_id = event.get("channel")
    
    # Check if the bot itself was added to the channel
    if user_id and channel_id:
        installation = INSTALLATIONS[team_id]
        bot_user_id = installation.bot_user_id
        
        if user_id == bot_user_id:
            print(f"Bot added to channel {channel_id} in team {team_id}")
            
            # Post a message to the channel announcing indexing
            try:
                await post_slack_message(
                    installation.bot_access_token,
                    channel_id,
                    "Hello! I'm now indexing the channel history to help answer questions. This may take a few minutes depending on the channel size."
                )
                
                # Start indexing the channel history
                asyncio.create_task(index_channel_history(team_id, channel_id))
                
            except Exception as e:
                print(f"Error posting to channel {channel_id}: {e}")

async def process_app_mention_event(team_id: str, event: Dict):
    """Process app mention events for direct interaction"""
    if team_id not in INSTALLATIONS:
        return
    
    # Extract question from mention
    text = event.get("text", "")
    # Remove the bot mention from the text
    question = text.replace(f"<@{event.get('bot_id', '')}>", "").strip()
    
    # Handle /ask commands in mentions
    if question.startswith("/ask"):
        question = question.replace("/ask", "").strip()
    
    if not question:
        question = "Hello! How can I help you?"
    
    # Process as a query
    try:
        rag_response = await answer_question(team_id, question)
        
        # Post response back to the channel
        installation = INSTALLATIONS[team_id]
        
        # Build response with source links
        response_text = f"**Answer:** {rag_response.answer}\n\n"
        
        if rag_response.sources:
            response_text += f"**Sources:** Found {len(rag_response.sources)} relevant messages\n"
            # Add actual source links (these are now clickable Slack links)
            for i, source in enumerate(rag_response.sources[:3], 1):  # Show first 3 sources
                response_text += f"  {i}. {source}\n"
            if len(rag_response.sources) > 3:
                response_text += f"  ... and {len(rag_response.sources) - 3} more\n"
        else:
            response_text += f"**Sources:** No specific sources found\n"
        
        await post_slack_message(
            installation.bot_access_token,
            event["channel"],
            response_text,
            thread_ts=event.get("thread_ts")
        )
    except Exception as e:
        print(f"Error processing mention: {e}")

async def process_single_message(message: SlackMessage):
    """Process a single message for embedding and storage"""
    try:
        # Process file attachments if present
        file_text = ""
        if message.has_files:
            file_text = await extract_file_text(message.team_id, message)
        
        # Combine message text with file content
        full_text = message.text
        if file_text:
            full_text += f"\n\n[File content: {file_text}]"
        
        # Generate embedding
        response = cohere_client.embed(
            texts=[full_text],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        embedding = response.embeddings[0]
        
        # Store in memory
        vector_data = {
            "vector": embedding,
            "message": asdict(message),
            "full_text": full_text,  # Store the combined text
            "indexed_at": datetime.now().isoformat()
        }
        MESSAGE_VECTORS[message.team_id].append(vector_data)
        
        print(f"Indexed new message from team {message.team_id}")
        
    except Exception as e:
        print(f"Failed to process message: {e}")

async def extract_file_text(team_id: str, message: SlackMessage) -> str:
    """Extract text from Slack files with proper authorization"""
    if not message.has_files:
        return ""
    
    installation = INSTALLATIONS.get(team_id)
    if not installation:
        return ""
    
    try:
        # Get file information from Slack
        async with httpx.AsyncClient() as client:
            # Note: In a real implementation, you'd need to get file info first
            # For now, we'll return a placeholder
            return "[File attached - content extraction not implemented in MVP]"
            
    except Exception as e:
        print(f"Failed to extract file text: {e}")
        return "[File processing error]"

async def post_slack_message(bot_token: str, channel: str, text: str, thread_ts: str = None):
    """Post a message to Slack channel"""
    async with httpx.AsyncClient() as client:
        payload = {
            "token": bot_token,
            "channel": channel,
            "text": text
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
            
        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            data=payload
        )
        
        return response.json()

async def get_channel_info(bot_token: str, channel_id: str) -> Dict:
    """Get channel information"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://slack.com/api/conversations.info",
            params={
                "token": bot_token,
                "channel": channel_id
            }
        )
        result = response.json()
        if result.get("ok"):
            return result.get("channel", {})
        return {}

async def get_channel_messages(bot_token: str, channel_id: str, limit: int = 1000) -> List[Dict]:
    """Get channel messages"""
    messages = []
    cursor = None
    
    async with httpx.AsyncClient() as client:
        while len(messages) < limit:
            params = {
                "token": bot_token,
                "channel": channel_id,
                "limit": min(200, limit - len(messages))
            }
            if cursor:
                params["cursor"] = cursor
                
            response = await client.get(
                "https://slack.com/api/conversations.history",
                params=params
            )
            
            result = response.json()
            if not result.get("ok"):
                break
                
            batch_messages = result.get("messages", [])
            messages.extend(batch_messages)
            
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
                
    return messages[:limit]

# === Phase 1: Slash Commands ===

@app.post("/slack/commands")
async def handle_slash_commands(
    request: Request,
    team_id: str = Form(...),
    user_id: str = Form(...),
    text: str = Form(...),
    command: str = Form(...)
):
    """Handle slash commands with signature verification and enhanced parsing"""
    
    # Verify signature
    body = await request.body()
    slack_signature = request.headers.get("x-slack-signature", "")
    slack_timestamp = request.headers.get("x-slack-request-timestamp", "")
    
    if not verify_slack_signature(body, slack_timestamp, slack_signature):
        raise HTTPException(401, "Invalid signature")
    
    if command != "/ask":
        return {"text": "Unknown command. Use `/ask [your question]`"}
    
    if not text.strip():
        return {
            "text": "Usage: `/ask [your question]`\n\nExamples:\n• `/ask What did we discuss about the new feature?`\n• `/ask #general What was decided in the meeting?`\n• `/ask #engineering last 7d Who worked on the API?`"
        }
    
    # Check if app is installed
    if team_id not in INSTALLATIONS:
        return {"text": "App not properly installed. Please reinstall from the web interface."}
    
    # Check if we have indexed messages
    if team_id not in MESSAGE_VECTORS or not MESSAGE_VECTORS[team_id]:
        return {"text": "Still indexing messages... Please try again in a minute!"}
    
    try:
        # Parse command with channel and time filters
        parsed_query = parse_ask_command(text.strip())
        
        # Process the question with enhanced parameters
        rag_response = await answer_question(
            team_id=team_id,
            question=parsed_query["question"],
            channel_filter=parsed_query["channel"],
            time_window_days=parsed_query["days"]
        )
        
        # Format response for Slack with enhanced information
        response_text = f"**Answer:** {rag_response.answer}\n\n"
        
        if rag_response.sources:
            response_text += f"**Sources:** Found {len(rag_response.sources)} relevant messages\n"
            # Add actual source links (these are now clickable Slack links)
            for i, source in enumerate(rag_response.sources[:3], 1):  # Show first 3 sources
                response_text += f"  {i}. {source}\n"
            if len(rag_response.sources) > 3:
                response_text += f"  ... and {len(rag_response.sources) - 3} more\n"
        else:
            response_text += f"**Sources:** No specific sources found (using general knowledge)\n"
        
        # Add query context
        if parsed_query["channel"]:
            response_text += f"**Channel filter:** #{parsed_query['channel']}\n"
        if parsed_query["days"] != 30:
            response_text += f"**Time window:** Last {parsed_query['days']} days\n"
        
        response_text += f"**Response time:** {rag_response.processing_time:.2f}s\n"
        response_text += f"**Confidence:** {rag_response.confidence:.2f}"
        
        return {
            "response_type": "in_channel",  # Post publicly
            "text": response_text
        }
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"text": f"Error: {str(e)}"}

def parse_ask_command(text: str) -> Dict[str, Any]:
    """Parse /ask command with optional channel and time filters
    
    Examples:
    - /ask What did we discuss about the new feature?
    - /ask #general What was decided in the meeting?
    - /ask #engineering last 7d Who worked on the API?
    - /ask last 30d What are the main concerns?
    """
    import re
    
    # Default values
    result = {
        "question": text,
        "channel": None,
        "days": 30
    }
    
    # Extract channel filter (#channel)
    channel_match = re.search(r'#(\w+)', text)
    if channel_match:
        result["channel"] = channel_match.group(1)
        # Remove channel from question
        result["question"] = re.sub(r'#\w+\s*', '', result["question"]).strip()
    
    # Extract time window (last Xd, last X days, etc.)
    time_patterns = [
        r'last\s+(\d+)\s*d(?:ays?)?',
        r'(\d+)\s*d(?:ays?)?\s+ago',
        r'past\s+(\d+)\s*d(?:ays?)?'
    ]
    
    for pattern in time_patterns:
        time_match = re.search(pattern, text, re.IGNORECASE)
        if time_match:
            days = int(time_match.group(1))
            result["days"] = min(days, 365)  # Cap at 1 year
            # Remove time filter from question
            result["question"] = re.sub(pattern, '', result["question"], flags=re.IGNORECASE).strip()
            break
    
    # Clean up question
    result["question"] = re.sub(r'\s+', ' ', result["question"]).strip()
    
    return result

# === Phase 2: Enhanced RAG Implementation ===

async def answer_question(team_id: str, question: str, channel_filter: str = None, time_window_days: int = 30) -> RAGResponse:
    """Enhanced RAG pipeline with hybrid retrieval and Cohere reranking"""
    start_time = time.time()
    
    # 1. Validate tenant and permissions
    installation = INSTALLATIONS.get(team_id)
    if not installation or not installation.is_active:
        raise HTTPException(404, "App not installed or inactive")
    
    # 2. Apply channel allowlist filter
    allowed_channels = installation.channel_allowlist or []
    base_filter = {
        "team_id": team_id,
        "deleted_at": {"$exists": False}  # Exclude deleted messages
    }
    
    if channel_filter:
        if allowed_channels and channel_filter not in allowed_channels:
            raise PermissionError("Channel not allowed")
        base_filter["channel_id"] = channel_filter
    elif allowed_channels:
        # Filter to allowed channels only
        pass  # We'll filter in the similarity search
    
    # 3. Get time cutoff for filtering
    time_cutoff = time.time() - (time_window_days * 24 * 3600)
    
    # 4. Generate query embedding first
    query_vector = await generate_query_embedding(question)
    
    # 5. Hybrid retrieval: BM25 keyword pre-filter + vector similarity
    # Try LanceDB first, fallback to in-memory
    try:
        # Use LanceDB for vector search
        lance_results = await db_manager.search_vectors(
            team_id=team_id,
            query_vector=query_vector,
            limit=50,
            channel_filter=channel_filter,
            time_cutoff=time_cutoff
        )
        
        if lance_results:
            # Convert LanceDB results to expected format
            vectors = []
            for result in lance_results:
                vector_data = {
                    "vector": np.array(result["vector"]),
                    "message": {
                        "team_id": result["team_id"],
                        "channel_id": result["channel_id"],
                        "user_id": result["user_id"],
                        "text": result["text"],
                        "ts": result["message_ts"],
                        "thread_ts": result["thread_ts"],
                        "permalink": f"https://{getattr(installation, 'team_domain', 'workspace')}.slack.com/archives/{result['channel_id']}/p{result['message_ts'].replace('.', '')}"
                    },
                    "full_text": result["text"],
                    "indexed_at": result["indexed_at"]
                }
                vectors.append(vector_data)
        else:
            # Fallback to in-memory storage
            vectors = MESSAGE_VECTORS[team_id]
    except Exception as e:
        print(f"LanceDB search failed, using in-memory fallback: {e}")
        vectors = MESSAGE_VECTORS[team_id]
    
    if not vectors:
        raise HTTPException(404, "No indexed messages found")
    
    # Step 1: BM25 keyword pre-filter (only for specific technical queries)
    # For conversational questions, skip BM25 and go directly to vector search
    is_conversational = any(phrase in question.lower() for phrase in [
        "what have we been discussing", "what did we talk about", "recent discussion",
        "lately", "recently", "what's been happening", "catch me up", "what have we discussed",
        "what did we discuss", "what was discussed", "what are we discussing", "what's the discussion",
        "what's been discussed", "what did we cover", "what have we covered", "what was covered"
    ])
    
    print(f"RAG Debug - Searching through {len(vectors)} indexed messages for team {team_id}")
    print(f"RAG Debug - Question: {question}")
    print(f"RAG Debug - Is conversational: {is_conversational}")
    
    # Debug: Show sample of indexed messages
    if vectors:
        print(f"RAG Debug - Sample indexed messages:")
        for i, vector_data in enumerate(vectors[:3]):
            text_preview = vector_data["message"]["text"][:50]
            print(f"  {i+1}. {text_preview}... (user: {vector_data['message']['user_id']})")
    else:
        print("RAG Debug - No vectors found!")
    
    if is_conversational or len(question.split()) > 8:  # Longer questions benefit from full vector search
        search_vectors = vectors
        print(f"RAG Debug - Using full vector search ({len(search_vectors)} vectors)")
    else:
        keyword_candidates = await bm25_keyword_search(question, vectors, time_cutoff, channel_filter, allowed_channels)
        # Use keyword candidates if we found good matches, otherwise use all vectors
        search_vectors = keyword_candidates if len(keyword_candidates) >= 5 else vectors
        print(f"RAG Debug - BM25 found {len(keyword_candidates)} candidates, using {len(search_vectors)} vectors for search")
    
    similarities = []
    for i, vector_data in enumerate(search_vectors):
        # Apply time filter
        indexed_at = vector_data.get("indexed_at", "")
        if indexed_at:
            try:
                indexed_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00')).timestamp()
                if indexed_time < time_cutoff:
                    continue
            except:
                pass  # Skip time filtering if parsing fails
        
        # Apply channel filter
        if channel_filter and vector_data["message"]["channel_id"] != channel_filter:
            continue
        if allowed_channels and vector_data["message"]["channel_id"] not in allowed_channels:
            continue
        
        similarity = cosine_similarity(query_vector, vector_data["vector"])
        similarities.append((similarity, i, vector_data))
    
    # Get top 20 candidates for reranking
    # For conversational questions, boost recent messages
    if is_conversational:
        # Add recency boost to similarity scores
        current_time = time.time()
        for i, (similarity, idx, vector_data) in enumerate(similarities):
            indexed_at = vector_data.get("indexed_at", "")
            if indexed_at:
                try:
                    indexed_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00')).timestamp()
                    # Boost recent messages (within last 7 days get +0.1 boost)
                    if current_time - indexed_time < 7 * 24 * 3600:
                        similarities[i] = (similarity + 0.1, idx, vector_data)
                except:
                    pass
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_candidates = similarities[:20]
    
    # Debug: Show top similarity scores
    print(f"RAG Debug - Top similarity scores:")
    for i, (score, idx, vector_data) in enumerate(top_candidates[:5]):
        text_preview = vector_data["message"]["text"][:50]
        print(f"  {i+1}. Score: {score:.3f} - {text_preview}...")
    
    # Use lower threshold for conversational questions
    min_threshold = 0.05 if is_conversational else 0.1
    
    if not top_candidates or top_candidates[0][0] < min_threshold:
        return RAGResponse(
            answer="I couldn't find relevant information to answer your question. Try rephrasing or asking about different topics that might be in the indexed messages.",
            sources=[],
            confidence=0.0,
            processing_time=time.time() - start_time,
            query=question
        )
    
    # 5. Cohere rerank for relevance
    documents = []
    for similarity, idx, vector_data in top_candidates:
        text_content = vector_data.get("full_text", vector_data["message"]["text"])
        documents.append(text_content)
    
    try:
        reranked = cohere_client.rerank(
            query=question,
            documents=documents,
            top_n=5,
            model="rerank-english-v3.0"
        )
    except Exception as e:
        print(f"Reranking failed, using original results: {e}")
        # Fallback to original similarity results
        reranked_results = [type('obj', (object,), {'index': i, 'relevance_score': sim}) 
                           for i, (sim, _, _) in enumerate(top_candidates[:5])]
    else:
        reranked_results = reranked.results
    
    # 6. Thread context expansion
    core_results = [top_candidates[r.index] for r in reranked_results]
    expanded_context = await expand_thread_context(team_id, core_results)
    
    # Debug: Print what context we're using
    print(f"RAG Debug - Found {len(expanded_context)} context messages")
    for i, ctx in enumerate(expanded_context[:3]):  # Show first 3
        print(f"  Context {i+1}: {ctx['message']['text'][:100]}...")
    
    # 7. Generate response with enhanced context
    response = await generate_with_citations(question, expanded_context)
    
    return RAGResponse(
        answer=response["answer"],
        sources=response["sources"],
        confidence=reranked_results[0].relevance_score if reranked_results else 0.0,
        processing_time=time.time() - start_time,
        query=question
    )

async def bm25_keyword_search(query: str, vectors: List[Dict], time_cutoff: float, channel_filter: str = None, allowed_channels: List[str] = None) -> List[Dict]:
    """Enhanced BM25-style keyword pre-filtering with better matching"""
    query_words = set(query.lower().split())
    candidates = []
    
    # Remove common stop words for better matching
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "when", "where", "why", "who"}
    query_words = query_words - stop_words
    
    for vector_data in vectors:
        text = vector_data.get("full_text", vector_data["message"]["text"]).lower()
        
        # Apply filters first
        if channel_filter and vector_data["message"]["channel_id"] != channel_filter:
            continue
        if allowed_channels and vector_data["message"]["channel_id"] not in allowed_channels:
            continue
        
        # Apply time filter
        indexed_at = vector_data.get("indexed_at", "")
        if indexed_at:
            try:
                indexed_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00')).timestamp()
                if indexed_time < time_cutoff:
                    continue
            except:
                pass
        
        # Enhanced keyword matching
        text_words = set(text.split())
        overlap = len(query_words.intersection(text_words))
        
        # Also check for partial matches and synonyms
        if overlap > 0:
            candidates.append((overlap, vector_data))
        else:
            # Check for partial word matches (e.g., "document" matches "documents")
            partial_matches = 0
            for query_word in query_words:
                for text_word in text_words:
                    if query_word in text_word or text_word in query_word:
                        partial_matches += 0.5
                        break
            
            if partial_matches > 0:
                candidates.append((partial_matches, vector_data))
    
    # Sort by relevance and return top candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [candidate[1] for candidate in candidates[:50]]

async def generate_query_embedding(question: str) -> List[float]:
    """Generate query embedding with Cohere"""
    try:
        query_response = cohere_client.embed(
            texts=[question],
            model="embed-english-v3.0", 
            input_type="search_query"
        )
        return query_response.embeddings[0]
    except Exception as e:
        raise HTTPException(500, f"Failed to generate query embedding: {e}")

async def expand_thread_context(team_id: str, core_messages: List) -> List[Dict]:
    """Expand core results with thread neighbors for better context"""
    expanded = []
    
    # Group messages by thread
    threads = defaultdict(list)
    for similarity, idx, vector_data in core_messages:
        thread_ts = vector_data["message"].get("thread_ts") or vector_data["message"]["ts"]
        threads[thread_ts].append((similarity, idx, vector_data))
    
    for thread_ts, thread_messages in threads.items():
        # Get all messages in this thread
        thread_vectors = []
        for vector_data in MESSAGE_VECTORS[team_id]:
            msg_thread_ts = vector_data["message"].get("thread_ts") or vector_data["message"]["ts"]
            if msg_thread_ts == thread_ts:
                thread_vectors.append(vector_data)
        
        # Sort by timestamp
        thread_vectors.sort(key=lambda x: float(x["message"]["ts"]))
        
        # Find positions of core messages
        core_positions = set()
        for similarity, idx, vector_data in thread_messages:
            for i, thread_vector in enumerate(thread_vectors):
                if thread_vector["message"]["ts"] == vector_data["message"]["ts"]:
                    core_positions.add(i)
                    break
        
        # Expand with neighbors (±3 messages)
        expanded_positions = set()
        for pos in core_positions:
            for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                neighbor_pos = pos + offset
                if 0 <= neighbor_pos < len(thread_vectors):
                    expanded_positions.add(neighbor_pos)
        
        # Add expanded messages
        for pos in sorted(expanded_positions):
            expanded.append(thread_vectors[pos])
    
    # Deduplicate by message timestamp
    seen_ts = set()
    deduplicated = []
    for vector_data in expanded:
        ts = vector_data["message"]["ts"]
        if ts not in seen_ts:
            seen_ts.add(ts)
            deduplicated.append(vector_data)
    
    return deduplicated[:20]  # Limit to reasonable context size

async def generate_with_citations(query: str, context_messages: List[Dict]) -> Dict:
    """Generate answer with proper citations"""
    if not context_messages:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "sources": []
        }
    
    # Prepare context with proper formatting
    context_parts = []
    sources = []
    
    for i, vector_data in enumerate(context_messages):
        msg_data = vector_data["message"]
        text_content = vector_data.get("full_text", msg_data["text"])
        
        # Add thread context if available
        thread_info = ""
        if msg_data.get("thread_ts") and msg_data["thread_ts"] != msg_data["ts"]:
            thread_info = " (in thread)"
        
        context_parts.append(f"Message {i+1}{thread_info}: {text_content}")
        
        # Create source citation with more details
        channel_id = msg_data["channel_id"]
        user_id = msg_data.get("user_id", "unknown")
        timestamp = msg_data.get("ts", "")
        
        # Try to get channel name for better source display
        channel_name = f"#{channel_id}"  # Default fallback
        try:
            # This would need to be cached or looked up
            channel_name = f"#{channel_id}"
        except:
            pass
        
        # Create a more informative source citation with clickable link
        if msg_data.get("permalink"):
            # Create clickable link using Slack's link format
            source_text = f"<{msg_data['permalink']}|Message in {channel_name}>"
        else:
            # Fallback without link
            source_text = f"Message in {channel_name} by user {user_id}"
        
        sources.append(source_text)
    
    context = "\n\n".join(context_parts)
    
    # Debug: Print context being sent to AI
    print(f"RAG Debug - Context being sent to AI:")
    print(f"Context length: {len(context)} characters")
    print(f"Context preview: {context[:200]}...")
    
    # Enhanced prompt for better responses
    prompt = f"""Based on the following Slack messages, please answer the user's question. Be helpful and provide any relevant information you can find.

Slack Messages:
{context}

User Question: {query}

Please provide a helpful answer based on the Slack messages above. If the messages contain relevant information, use it to answer the question. Be specific about what was discussed in the messages. If the messages don't directly answer the question but contain related information, provide that information. Only say you don't have enough information if the messages are completely unrelated to the question.

Answer:"""

    try:
        response = cohere_client.generate(
            model="command-light",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            stop_sequences=["\n\nHuman:", "\n\nUser:", "\n\nQuestion:"]
        )
        
        answer = response.generations[0].text.strip()
        
        # Debug: Print AI response
        print(f"RAG Debug - AI Response:")
        print(f"Response: {answer}")
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to generate answer: {e}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

# === Health Check ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "installations": len(INSTALLATIONS),
        "total_vectors": sum(len(vectors) for vectors in MESSAGE_VECTORS.values()),
        "teams_indexed": list(INSTALLATIONS.keys()),
        "phase": "Phase 3 - Slack UX Implementation",
        "features": [
            "Hybrid retrieval (BM25 + vector similarity)",
            "Cohere reranking for relevance",
            "Thread context expansion",
            "Enhanced citation formatting",
            "Channel and time filtering",
            "Advanced slash command parsing",
            "App Home interface with admin controls",
            "Interactive message actions",
            "Thread summarization",
            "User permission management"
        ],
        "endpoints": {
            "home": "/",
            "help": "/help",
            "health": "/health",
            "installations": "/installations",
            "slack_install": "/slack/install",
            "slack_oauth": "/slack/oauth",
            "slack_events": "/slack/events",
            "slack_commands": "/slack/commands",
            "slack_interactive": "/slack/interactive"
        }
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Test endpoint working", "timestamp": datetime.utcnow().isoformat()}

@app.get("/installations")
async def get_installations():
    """Debug endpoint to check current installations"""
    return {
        "installations": list(INSTALLATIONS.keys()),
        "total_installations": len(INSTALLATIONS),
        "message_vectors": {team_id: len(vectors) for team_id, vectors in MESSAGE_VECTORS.items()}
    }

@app.get("/debug/{team_id}")
async def debug_team(team_id: str):
    """Debug endpoint to check team-specific data"""
    if team_id not in INSTALLATIONS:
        return {"error": "Team not found"}
    
    installation = INSTALLATIONS[team_id]
    vectors = MESSAGE_VECTORS.get(team_id, [])
    
    # Sample a few messages for debugging
    sample_messages = []
    for i, vector_data in enumerate(vectors[:5]):  # First 5 messages
        sample_messages.append({
            "index": i,
            "text_preview": vector_data["message"]["text"][:100],
            "channel_id": vector_data["message"]["channel_id"],
            "user_id": vector_data["message"]["user_id"],
            "ts": vector_data["message"]["ts"],
            "indexed_at": vector_data.get("indexed_at", "unknown")
        })
    
    return {
        "team_id": team_id,
        "installation": {
            "team_name": getattr(installation, 'team_name', 'Unknown'),
            "installer_user_id": installation.installer_user_id,
            "installation_time": installation.installation_time.isoformat(),
            "is_active": installation.is_active,
            "scopes": installation.scopes
        },
        "message_stats": {
            "total_messages": len(vectors),
            "sample_messages": sample_messages
        },
        "channels": list(set(v["message"]["channel_id"] for v in vectors))
    }

@app.post("/debug/{team_id}/reindex")
async def debug_reindex_team(team_id: str):
    """Debug endpoint to manually trigger reindexing"""
    if team_id not in INSTALLATIONS:
        return {"error": "Team not found"}
    
    installation = INSTALLATIONS[team_id]
    
    # Clear existing vectors for this team
    MESSAGE_VECTORS[team_id] = []
    
    # Trigger reindexing
    asyncio.create_task(index_workspace_messages(team_id, installation.bot_access_token))
    
    return {
        "message": f"Reindexing started for team {team_id}",
        "status": "started"
    }

@app.get("/help", response_class=HTMLResponse)
async def help_page():
    """Help page with instructions for inviting bot to channels"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slack RAG Bot - Help</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .section { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .code { background: #e9ecef; padding: 10px; border-radius: 4px; font-family: monospace; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Slack RAG Bot - Help & Setup</h1>
        
        <div class="warning">
            <h3>Bot Not Reading Messages?</h3>
            <p>If the bot isn't finding any messages, it likely needs to be invited to channels first.</p>
        </div>
        
        <div class="section">
            <h3>How to Invite Bot to Channels</h3>
            <ol>
                <li><strong>Go to any channel</strong> where you want the bot to read messages</li>
                <li><strong>Type this command:</strong> <span class="code">/invite @YourBotName</span></li>
                <li><strong>Or use channel settings:</strong>
                    <ul>
                        <li>Click the channel name at the top</li>
                        <li>Go to "Settings" → "Integrations"</li>
                        <li>Add your bot</li>
                    </ul>
                </li>
            </ol>
        </div>
        
        <div class="section">
            <h3>💬 How to Use the Bot</h3>
            <h4>Primary Method - Bot Mentions (Recommended):</h4>
            <ul>
                <li><span class="code">@YourBotName What did we discuss about the new feature?</span></li>
                <li><span class="code">@YourBotName Who worked on the API integration?</span></li>
                <li><span class="code">@YourBotName #general What was decided in the meeting?</span> (search specific channel)</li>
                <li><span class="code">@YourBotName #engineering last 7d Who worked on the API?</span> (time filter)</li>
            </ul>
            
            <h4>Alternative Method - Slash Commands:</h4>
            <ul>
                <li><span class="code">/ask What did we discuss about the new feature?</span></li>
                <li><span class="code">/ask Who worked on the API integration?</span></li>
                <li><span class="code">/ask #general What was decided in the meeting?</span> (search specific channel)</li>
                <li><span class="code">/ask #engineering last 7d Who worked on the API?</span> (time filter)</li>
            </ul>
            
            <div class="warning">
                <strong>Note:</strong> If slash commands don't work, use bot mentions instead. Bot mentions are more reliable and work in all channels.
            </div>
        </div>
        
        <div class="section">
            <h3>🔧 Troubleshooting</h3>
            <h4>Bot says "Still indexing messages..."</h4>
            <p>Wait 1-2 minutes after installation. The bot needs time to read and process messages.</p>
            
            <h4>Bot says "No indexed messages found"</h4>
            <p>Make sure the bot is invited to channels and there are recent messages (last 30 days).</p>
            
            <h4>Bot can't access channels</h4>
            <p>Check that the bot has the required permissions: <code>channels:history</code>, <code>channels:read</code></p>
            
            <h4>Slash commands not working</h4>
            <p><strong>Use bot mentions instead:</strong> Type <code>@YourBotName</code> followed by your question. This is more reliable than slash commands.</p>
            
            <h4>Bot not responding to mentions</h4>
            <p>Make sure the bot is properly installed and has the <code>app_mentions:read</code> permission.</p>
        </div>
        
        <div class="section">
            <h3>Check Status</h3>
            <p><a href="/health">View system health</a> | <a href="/installations">View installations</a></p>
        </div>
        
        <p><a href="/">← Back to Home</a></p>
    </body>
    </html>
    """

# === Phase 3: App Home & Interactive Features ===

async def get_user_info(team_id: str, user_id: str, bot_token: str) -> SlackUser:
    """Get or cache user information"""
    cache_key = (team_id, user_id)
    
    if cache_key in USER_CACHE:
        return USER_CACHE[cache_key]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://slack.com/api/users.info",
                headers={"Authorization": f"Bearer {bot_token}"},
                params={"user": user_id}
            )
        
        data = response.json()
        if data.get("ok"):
            user_data = data["user"]
            slack_user = SlackUser(
                team_id=team_id,
                user_id=user_id,
                username=user_data.get("name"),
                display_name=user_data.get("real_name"),
                is_admin=user_data.get("is_admin", False),
                last_updated=datetime.utcnow()
            )
            USER_CACHE[cache_key] = slack_user
            return slack_user
    except Exception as e:
        print(f"Error fetching user info: {e}")
    
    # Fallback user
    return SlackUser(team_id=team_id, user_id=user_id, last_updated=datetime.utcnow())

async def is_user_admin(team_id: str, user_id: str) -> bool:
    """Check if user is admin (installer or workspace admin)"""
    installation = INSTALLATIONS.get(team_id)
    if not installation:
        return False
    
    # Installer is always admin
    if user_id == installation.installer_user_id:
        return True
    
    # Check if user is workspace admin
    try:
        user_info = await get_user_info(team_id, user_id, installation.bot_access_token)
        return user_info.is_admin
    except:
        return False

async def handle_app_home_opened(team_id: str, user_id: str):
    """Handle App Home opened event"""
    try:
        is_admin = await is_user_admin(team_id, user_id)
        home_view = build_app_home_view(team_id, user_id, is_admin)
        
        installation = INSTALLATIONS.get(team_id)
        if installation:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://slack.com/api/views.publish",
                    headers={"Authorization": f"Bearer {installation.bot_access_token}"},
                    json={
                        "user_id": user_id,
                        "view": home_view
                    }
                )
            
            if response.status_code == 200:
                print(f"Published App Home view for user {user_id} in team {team_id}")
            else:
                print(f"Failed to publish App Home view: {response.text}")
    except Exception as e:
        print(f"Error handling app home opened: {e}")

def build_app_home_view(team_id: str, user_id: str, is_admin: bool) -> dict:
    """Build App Home view based on user permissions"""
    installation = INSTALLATIONS.get(team_id)
    if not installation:
        return {"type": "home", "blocks": []}
    
    if not is_admin:
        # User view - basic info and usage
        return {
            "type": "home",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "RAG Assistant"}
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Welcome to the RAG Assistant!*\n\nI can help you find information from your Slack conversations using AI.\n\n*How to use:*\n• Use `/ask [your question]` to ask questions\n• Mention me with `@RAGBot [question]` for direct queries\n• I'll search through recent messages and provide AI-powered answers"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Current Status:*\n• Bot is active\n• {len(MESSAGE_VECTORS.get(team_id, []))} messages indexed\n• Ready to answer questions"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "*Tip:* Ask specific questions for better results!"}
                    ]
                }
            ]
        }
    
    # Admin view - settings and controls
    channels_text = "All channels" if not installation.channel_allowlist else f"{len(installation.channel_allowlist)} selected channels"
    
    return {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                    "text": {"type": "plain_text", "text": "RAG Assistant - Admin Settings"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Channel Access Control*\nCurrently indexing: {channels_text}\n\n*Data Retention*\nMessages older than {installation.retention_days} days are automatically removed."
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Statistics"},
                        "action_id": "view_stats",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reindex Messages"},
                        "action_id": "reindex_messages"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Current Status:*\n• ✅ Bot is active\n• 📊 {len(MESSAGE_VECTORS.get(team_id, []))} messages indexed\n• 🏢 Team: {installation.team_id}\n• 📅 Installed: {installation.installation_time.strftime('%Y-%m-%d %H:%M')}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "🔧 *Admin controls available* - Use buttons above to manage settings"}
                ]
            }
        ]
    }

@app.post("/slack/interactive")
async def handle_interactive_components(request: Request):
    """Handle interactive components (buttons, selects, etc.)"""
    body = await request.body()
    
    # Verify signature
    slack_signature = request.headers.get("x-slack-signature", "")
    slack_timestamp = request.headers.get("x-slack-request-timestamp", "")
    
    if not verify_slack_signature(body, slack_timestamp, slack_signature):
        raise HTTPException(401, "Invalid signature")
    
    form_data = await request.form()
    payload = json.loads(form_data["payload"])
    
    team_id = payload.get("team", {}).get("id")
    user_id = payload.get("user", {}).get("id")
    action_id = payload.get("actions", [{}])[0].get("action_id")
    
    if not team_id or not user_id:
        return {"text": "Error: Missing team or user information"}
    
    installation = INSTALLATIONS.get(team_id)
    if not installation:
        return {"text": "Error: App not installed"}
    
    # Check admin permissions for admin actions
    if action_id in ["view_stats", "reindex_messages"]:
        is_admin = await is_user_admin(team_id, user_id)
        if not is_admin:
            return {"text": "❌ Admin permissions required for this action"}
    
    if action_id == "view_stats":
        # Show statistics
        total_messages = len(MESSAGE_VECTORS.get(team_id, []))
        channels_count = len(installation.channel_allowlist) if installation.channel_allowlist else "All"
        
        stats_text = f"""📊 *RAG Assistant Statistics*

*Team:* {team_id}
*Total Messages Indexed:* {total_messages}
*Channels Monitored:* {channels_count}
*Retention Policy:* {installation.retention_days} days
*Installation Date:* {installation.installation_time.strftime('%Y-%m-%d %H:%M')}
*Bot Status:* {'✅ Active' if installation.is_active else '❌ Inactive'}

*Recent Activity:*
• Messages are indexed in real-time
• AI responses use Cohere embeddings and generation
• Thread context is preserved for better answers"""
        
        return {
            "response_type": "ephemeral",
            "text": stats_text
        }
    
    elif action_id == "reindex_messages":
        # Trigger reindexing
        asyncio.create_task(index_workspace_messages(team_id, installation.bot_access_token))
        
        return {
            "response_type": "ephemeral",
            "text": "🔄 Reindexing started! This may take a few minutes. You'll be notified when complete."
        }
    
    elif action_id == "summarize_thread":
        # Handle thread summarization
        channel_id = payload.get("channel", {}).get("id")
        message_ts = payload.get("message", {}).get("ts")
        thread_ts = payload.get("message", {}).get("thread_ts", message_ts)
        
        if channel_id and thread_ts:
            asyncio.create_task(summarize_thread(team_id, channel_id, thread_ts, installation.bot_access_token))
            return {
                "response_type": "ephemeral",
                "text": "🧵 Generating thread summary... This may take a moment."
            }
    
    return {"text": "Unknown action"}

async def summarize_thread(team_id: str, channel_id: str, thread_ts: str, bot_token: str):
    """Generate a summary of a thread"""
    try:
        # Get thread messages from our indexed data
        thread_messages = []
        for vector_data in MESSAGE_VECTORS.get(team_id, []):
            msg = vector_data["message"]
            if (msg.channel_id == channel_id and 
                (msg.thread_ts == thread_ts or msg.ts == thread_ts)):
                thread_messages.append(msg)
        
        if not thread_messages:
            # Fallback: fetch from Slack API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/conversations.replies",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    params={
                        "channel": channel_id,
                        "ts": thread_ts,
                        "limit": 50
                    }
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    messages = data.get("messages", [])
                    thread_messages = [
                        SlackMessage(
                            team_id=team_id,
                            channel_id=channel_id,
                            user_id=msg.get("user", "unknown"),
                            text=msg.get("text", ""),
                            ts=msg["ts"],
                            thread_ts=msg.get("thread_ts")
                        )
                        for msg in messages
                        if msg.get("text") and len(msg["text"].strip()) > 5
                    ]
        
        if len(thread_messages) < 2:
            summary_text = "This thread doesn't have enough messages to summarize."
        else:
            # Sort by timestamp
            thread_messages.sort(key=lambda x: float(x.ts))
            
            # Create context for summarization
            context = "\n\n".join([
                f"Message {i+1}: {msg.text}" 
                for i, msg in enumerate(thread_messages[:10])  # Limit to first 10 messages
            ])
            
            # Generate summary using Cohere
            prompt = f"""Please provide a concise summary of this Slack thread discussion. Focus on the main points, decisions made, and key takeaways.

Thread messages:
{context}

Summary:"""

            try:
                response = cohere_client.generate(
                    model="command-r",
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.3
                )
                summary_text = response.generations[0].text.strip()
            except Exception as e:
                summary_text = f"Error generating summary: {str(e)}"
        
        # Post summary as threaded reply
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}"},
                json={
                    "channel": channel_id,
                    "thread_ts": thread_ts,
                    "text": f"🧵 *Thread Summary*\n\n{summary_text}",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"🧵 *Thread Summary*\n\n{summary_text}"
                            }
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"📊 {len(thread_messages)} messages • 🤖 Generated by RAG Assistant"
                                }
                            ]
                        }
                    ]
                }
            )
        
        print(f"Posted thread summary for {thread_ts} in channel {channel_id}")
        
    except Exception as e:
        print(f"Error summarizing thread: {e}")
        # Post error message
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    json={
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "text": "❌ Error generating thread summary. Please try again later."
                    }
                )
        except:
            pass

# === Phase 4: Operations & Evaluation Endpoints ===

# Initialize admin tools
data_exporter = DataExporter(INSTALLATIONS, MESSAGE_VECTORS)
data_manager = DataManager(INSTALLATIONS, MESSAGE_VECTORS)
health_monitor = SystemHealthMonitor(INSTALLATIONS, MESSAGE_VECTORS)
audit_manager = AuditManager()

# === Monitoring Endpoints ===

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return get_metrics_response()

@app.get("/health/detailed")
async def detailed_health():
    """Detailed system health check"""
    health_status = await health_monitor.get_system_health()
    return health_status

@app.get("/health/installation/{team_id}")
async def installation_health(team_id: str):
    """Get health status for a specific installation"""
    health_status = await health_monitor.get_installation_health(team_id)
    return health_status

# === Evaluation Endpoints ===

@app.post("/evaluate/{team_id}")
async def evaluate_team(team_id: str):
    """Run automated evaluation for a team"""
    try:
        evaluator = RAGEvaluator(answer_question)
        report = await evaluator.evaluate_team(team_id)
        
        # Log evaluation
        log_installation_event(
            team_id=team_id,
            event_type="evaluation_completed",
            evaluation_score=report.average_overall_score
        )
        
        return {
            "status": "completed",
            "team_id": team_id,
            "evaluation_time": report.evaluation_time.isoformat(),
            "summary": {
                "total_questions": report.total_questions,
                "successful_questions": report.successful_questions,
                "average_overall_score": report.average_overall_score,
                "average_response_time": report.average_response_time
            },
            "recommendations": report.recommendations
        }
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

@app.get("/evaluate/{team_id}/report")
async def get_evaluation_report(team_id: str):
    """Get evaluation report for a team"""
    try:
        evaluator = RAGEvaluator(answer_question)
        report = await evaluator.evaluate_team(team_id)
        report_text = generate_evaluation_report(report)
        
        return {
            "team_id": team_id,
            "report": report_text,
            "raw_data": asdict(report)
        }
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {str(e)}")

# === Admin Endpoints ===

@app.post("/admin/token")
async def create_admin_token(team_id: str, admin_user_id: str):
    """Create admin token for team management"""
    token = generate_admin_token(team_id, admin_user_id)
    return {
        "token": token,
        "expires_in_hours": 24,
        "team_id": team_id
    }

@app.get("/admin/export/{team_id}")
async def export_team_data(team_id: str, admin_token: str = Header(None)):
    """Export all data for a team (GDPR compliance)"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    export_data = await data_exporter.export_team_data(team_id, token_info["admin_user_id"])
    return export_data

@app.get("/admin/export/{team_id}/csv")
async def export_team_data_csv(team_id: str, admin_token: str = Header(None)):
    """Export team data as CSV file"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    return await data_exporter.export_team_data_csv(team_id, token_info["admin_user_id"])

@app.get("/admin/stats/{team_id}")
async def get_team_statistics(team_id: str, admin_token: str = Header(None)):
    """Get comprehensive statistics for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    stats = await data_exporter.get_team_statistics(team_id)
    return stats

@app.delete("/admin/purge/{team_id}")
async def purge_team_data(team_id: str, admin_token: str = Header(None)):
    """Purge all data for a team (GDPR compliance)"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    result = await data_manager.purge_team_data(team_id, token_info["admin_user_id"])
    return result

@app.put("/admin/retention/{team_id}")
async def update_retention_policy(team_id: str, retention_days: int, admin_token: str = Header(None)):
    """Update data retention policy for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    result = await data_manager.update_retention_policy(team_id, retention_days, token_info["admin_user_id"])
    return result

@app.put("/admin/allowlist/{team_id}")
async def update_channel_allowlist(team_id: str, channel_allowlist: List[str], admin_token: str = Header(None)):
    """Update channel allowlist for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    result = await data_manager.update_channel_allowlist(team_id, channel_allowlist, token_info["admin_user_id"])
    return result

@app.put("/admin/deactivate/{team_id}")
async def deactivate_installation(team_id: str, admin_token: str = Header(None)):
    """Deactivate installation for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    result = await data_manager.deactivate_installation(team_id, token_info["admin_user_id"])
    return result

@app.put("/admin/reactivate/{team_id}")
async def reactivate_installation(team_id: str, admin_token: str = Header(None)):
    """Reactivate installation for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    result = await data_manager.reactivate_installation(team_id, token_info["admin_user_id"])
    return result

@app.get("/admin/audit/{team_id}")
async def get_audit_log(team_id: str, admin_token: str = Header(None), limit: int = 100):
    """Get audit log for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    audit_log = await audit_manager.get_audit_log(team_id, limit)
    return audit_log

@app.get("/admin/compliance/{team_id}")
async def get_compliance_report(team_id: str, admin_token: str = Header(None)):
    """Get compliance report for a team"""
    if not admin_token:
        raise HTTPException(401, "Admin token required")
    
    token_info = verify_admin_token(admin_token)
    if token_info["team_id"] != team_id:
        raise HTTPException(403, "Token not valid for this team")
    
    compliance_report = await audit_manager.get_compliance_report(team_id)
    return compliance_report

# === Enhanced Health Check with Metrics ===

@app.get("/health")
async def health():
    """Enhanced health check with system metrics"""
    health_status = await health_monitor.get_system_health()
    
    # Update system metrics
    indexed_messages = {team_id: len(vectors) for team_id, vectors in MESSAGE_VECTORS.items()}
    metrics_collector.update_system_metrics(len(INSTALLATIONS), indexed_messages)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "installations": len(INSTALLATIONS),
        "total_vectors": sum(len(vectors) for vectors in MESSAGE_VECTORS.values()),
        "system_health": health_status
    }

# === Run the Application ===

if __name__ == "__main__":
    print("Starting Slack RAG Bot...")
    print("Visit http://localhost:8000 to install the bot")
    print("Phase 4: Operations & Evaluation - COMPLETE")
    uvicorn.run(app, host="0.0.0.0", port=8000)
