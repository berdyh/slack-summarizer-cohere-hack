"""
Data models for the Slack RAG system
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any

@dataclass
class SlackInstallation:
    """Multi-tenant installation data model"""
    team_id: str              # Primary tenant identifier
    team_domain: str          # Team domain for building permalinks
    installer_user_id: str    # Who installed the app
    bot_access_token: str     # Bot token (should be encrypted in production)
    scopes: List[str]         # Granted permissions
    installation_time: datetime
    channel_allowlist: List[str] = None  # Admin-selected channels
    retention_days: int = 90  # Data retention policy
    is_active: bool = True
    settings: Dict[str, Any] = None  # Additional settings

@dataclass
class SlackUser:
    """User information model"""
    team_id: str
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    is_admin: bool = False

@dataclass
class SlackMessage:
    """Message data model"""
    team_id: str
    channel_id: str
    user_id: str
    text: str
    ts: str
    thread_ts: Optional[str] = None
    message_type: str = "message"
    has_files: bool = False
    permalink: Optional[str] = None

@dataclass 
class RAGResponse:
    """RAG response model"""
    answer: str
    sources: List[str]
    confidence: float
    processing_time: float
    query: str = ""  # Optional query field for debugging
