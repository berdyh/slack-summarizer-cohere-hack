"""
Phase 5: Production Database Integration
LanceDB for vector storage and PostgreSQL for metadata storage

This module replaces the in-memory storage with persistent databases:
- LanceDB: Vector embeddings and similarity search
- PostgreSQL: Installation metadata, user cache, and system data
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Database dependencies
import lancedb
import pyarrow as pa
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY

# Import data models
from models import SlackInstallation, SlackUser, SlackMessage, RAGResponse

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/slack_rag")
LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", "./data/lancedb")

# SQLAlchemy setup
Base = declarative_base()

class SlackInstallationDB(Base):
    """PostgreSQL model for Slack installations"""
    __tablename__ = "slack_installations"
    
    team_id = Column(String(20), primary_key=True)
    installer_user_id = Column(String(20), nullable=False)
    bot_access_token = Column(Text, nullable=False)  # Should be encrypted in production
    scopes = Column(ARRAY(String), nullable=False)
    installation_time = Column(DateTime, default=datetime.utcnow)
    channel_allowlist = Column(ARRAY(String))
    retention_days = Column(Integer, default=90)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default={})

class SlackUserDB(Base):
    """PostgreSQL model for Slack users"""
    __tablename__ = "slack_users"
    
    team_id = Column(String(20), primary_key=True)
    user_id = Column(String(20), primary_key=True)
    username = Column(String(50))
    display_name = Column(String(100))
    is_admin = Column(Boolean, default=False)
    last_updated = Column(DateTime, default=datetime.utcnow)

class SlackMessageDB(Base):
    """PostgreSQL model for message metadata"""
    __tablename__ = "slack_messages"
    
    id = Column(String, primary_key=True)  # UUID
    team_id = Column(String(20), nullable=False)
    channel_id = Column(String(20), nullable=False)
    message_ts = Column(String(20), nullable=False)
    thread_ts = Column(String(20))
    user_id = Column(String(20), nullable=False)
    message_type = Column(String(20), default="message")
    text_preview = Column(Text)  # First 500 chars for search
    has_files = Column(Boolean, default=False)
    reply_count = Column(Integer, default=0)
    indexed_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime)
    vector_id = Column(String(50))  # Reference to LanceDB
    permalink = Column(Text)

class DatabaseManager:
    """Main database manager for both LanceDB and PostgreSQL"""
    
    def __init__(self):
        self.lance_db = None
        self.lance_table = None
        self.pg_engine = None
        self.pg_session = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize both databases"""
        if self._initialized:
            return
        
        # Initialize LanceDB
        await self._init_lancedb()
        
        # Initialize PostgreSQL
        await self._init_postgresql()
        
        self._initialized = True
        print("Database initialization complete")
    
    async def _init_lancedb(self):
        """Initialize LanceDB for vector storage"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(LANCE_DB_PATH, exist_ok=True)
            
            # Connect to LanceDB
            self.lance_db = lancedb.connect(LANCE_DB_PATH)
            
            # Define schema for vector table
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("team_id", pa.string()),
                pa.field("channel_id", pa.string()),
                pa.field("message_ts", pa.string()),
                pa.field("thread_ts", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32())),
                pa.field("indexed_at", pa.timestamp('us')),
                pa.field("deleted_at", pa.timestamp('us'))
            ])
            
            # Create or get table
            try:
                self.lance_table = self.lance_db.create_table("slack_messages", schema=schema)
                print("Created new LanceDB table")
            except Exception:
                self.lance_table = self.lance_db.open_table("slack_messages")
                print("Opened existing LanceDB table")
                
        except Exception as e:
            print(f"LanceDB initialization failed: {e}")
            raise
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL for metadata storage"""
        try:
            # Create async engine
            self.pg_engine = create_async_engine(DATABASE_URL, echo=False)
            
            # Create session factory
            async_session = sessionmaker(
                self.pg_engine, class_=AsyncSession, expire_on_commit=False
            )
            self.pg_session = async_session
            
            # Create tables
            async with self.pg_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            print("PostgreSQL initialization complete")
            
        except Exception as e:
            print(f"PostgreSQL initialization failed: {e}")
            print("Continuing with in-memory storage only")
            self.pg_engine = None
            self.pg_session = None
    
    # === Installation Management ===
    
    async def store_installation(self, installation: SlackInstallation):
        """Store installation in PostgreSQL"""
        if not self.pg_session:
            print(f"PostgreSQL not available, skipping installation storage for team {installation.team_id}")
            return
            
        try:
            async with self.pg_session() as session:
                db_installation = SlackInstallationDB(
                    team_id=installation.team_id,
                    installer_user_id=installation.installer_user_id,
                    bot_access_token=installation.bot_access_token,
                    scopes=installation.scopes,
                    installation_time=installation.installation_time,
                    channel_allowlist=installation.channel_allowlist,
                    retention_days=installation.retention_days,
                    is_active=installation.is_active,
                    settings=installation.settings or {}
                )
                
                session.add(db_installation)
                await session.commit()
                print(f"Stored installation for team {installation.team_id}")
        except Exception as e:
            print(f"Failed to store installation in PostgreSQL: {e}")
            print("Continuing with in-memory storage only")
    
    async def get_installation(self, team_id: str) -> Optional[SlackInstallation]:
        """Get installation from PostgreSQL"""
        if not self.pg_session:
            return None
            
        try:
            async with self.pg_session() as session:
                result = await session.execute(
                    text("SELECT * FROM slack_installations WHERE team_id = :team_id"),
                    {"team_id": team_id}
                )
                row = result.fetchone()
                
                if row:
                    return SlackInstallation(
                        team_id=row.team_id,
                        installer_user_id=row.installer_user_id,
                        bot_access_token=row.bot_access_token,
                        scopes=row.scopes,
                        installation_time=row.installation_time,
                        channel_allowlist=row.channel_allowlist,
                        retention_days=row.retention_days,
                        is_active=row.is_active,
                        settings=row.settings
                    )
                return None
        except Exception as e:
            print(f"Failed to get installation from PostgreSQL: {e}")
            return None
    
    async def update_installation(self, team_id: str, updates: Dict[str, Any]):
        """Update installation in PostgreSQL"""
        async with self.pg_session() as session:
            await session.execute(
                text("""
                    UPDATE slack_installations 
                    SET channel_allowlist = :channel_allowlist,
                        retention_days = :retention_days,
                        is_active = :is_active,
                        settings = :settings
                    WHERE team_id = :team_id
                """),
                {
                    "team_id": team_id,
                    "channel_allowlist": updates.get("channel_allowlist"),
                    "retention_days": updates.get("retention_days"),
                    "is_active": updates.get("is_active"),
                    "settings": json.dumps(updates.get("settings", {}))
                }
            )
            await session.commit()
    
    # === User Management ===
    
    async def store_user(self, user: SlackUser):
        """Store user in PostgreSQL"""
        async with self.pg_session() as session:
            db_user = SlackUserDB(
                team_id=user.team_id,
                user_id=user.user_id,
                username=user.username,
                display_name=user.display_name,
                is_admin=user.is_admin,
                last_updated=datetime.utcnow()
            )
            
            session.add(db_user)
            await session.commit()
    
    async def get_user(self, team_id: str, user_id: str) -> Optional[SlackUser]:
        """Get user from PostgreSQL"""
        async with self.pg_session() as session:
            result = await session.execute(
                text("SELECT * FROM slack_users WHERE team_id = :team_id AND user_id = :user_id"),
                {"team_id": team_id, "user_id": user_id}
            )
            row = result.fetchone()
            
            if row:
                return SlackUser(
                    team_id=row.team_id,
                    user_id=row.user_id,
                    username=row.username,
                    display_name=row.display_name,
                    is_admin=row.is_admin
                )
            return None
    
    # === Vector Storage ===
    
    async def store_message_vectors(self, team_id: str, messages: List[SlackMessage], vectors: List[np.ndarray]):
        """Store message vectors in LanceDB"""
        if not messages or not vectors:
            return
        
        # Prepare data for LanceDB
        data = []
        for msg, vector in zip(messages, vectors):
            data.append({
                "id": f"{team_id}_{msg.channel_id}_{msg.ts}",
                "team_id": team_id,
                "channel_id": msg.channel_id,
                "message_ts": msg.ts,
                "thread_ts": msg.thread_ts,
                "user_id": msg.user_id,
                "text": msg.text,
                "vector": vector.tolist(),
                "indexed_at": datetime.utcnow(),
                "deleted_at": None
            })
        
        # Create Arrow table and insert
        table = pa.Table.from_pylist(data)
        self.lance_table.add(table)
        
        print(f"Stored {len(messages)} vectors for team {team_id}")
    
    async def search_vectors(self, team_id: str, query_vector: np.ndarray, limit: int = 20, 
                           channel_filter: Optional[str] = None, time_cutoff: Optional[datetime] = None) -> List[Dict]:
        """Search vectors in LanceDB"""
        # Build filter conditions
        filters = [f"team_id = '{team_id}'", "deleted_at IS NULL"]
        
        if channel_filter:
            filters.append(f"channel_id = '{channel_filter}'")
        
        if time_cutoff:
            filters.append(f"indexed_at >= '{time_cutoff.isoformat()}'")
        
        filter_expr = " AND ".join(filters)
        
        # Perform vector search
        results = self.lance_table.search(query_vector.tolist()).where(filter_expr).limit(limit).to_pandas()
        
        # Convert to list of dicts
        return results.to_dict('records')
    
    async def delete_message_vectors(self, team_id: str, message_ids: List[str]):
        """Soft delete message vectors"""
        for msg_id in message_ids:
            # Update deleted_at timestamp
            self.lance_table.update(
                where=f"id = '{msg_id}'",
                values={"deleted_at": datetime.utcnow()}
            )
    

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for backward compatibility
async def get_installation(team_id: str) -> Optional[SlackInstallation]:
    """Get installation - backward compatibility"""
    await db_manager.initialize()
    return await db_manager.get_installation(team_id)

async def store_installation(installation: SlackInstallation):
    """Store installation - backward compatibility"""
    await db_manager.initialize()
    await db_manager.store_installation(installation)

async def get_user(team_id: str, user_id: str) -> Optional[SlackUser]:
    """Get user - backward compatibility"""
    await db_manager.initialize()
    return await db_manager.get_user(team_id, user_id)

async def store_user(user: SlackUser):
    """Store user - backward compatibility"""
    await db_manager.initialize()
    await db_manager.store_user(user)
