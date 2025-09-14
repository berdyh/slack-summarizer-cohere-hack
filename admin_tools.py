"""
Admin Tools for Slack RAG Bot
Implements data management, tenant operations, and administrative functions
"""

import json
import hashlib
import hmac
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from fastapi import HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
import io
import csv

# Import monitoring for audit logging
from monitoring import log_security_event, log_installation_event, metrics_collector

# === Admin Authentication ===

ADMIN_TOKENS = {}  # In production, store in secure database
ADMIN_ACTIONS_LOG = []  # Audit log for admin actions

def generate_admin_token(team_id: str, admin_user_id: str) -> str:
    """Generate admin token for team management"""
    timestamp = str(int(time.time()))
    payload = f"{team_id}:{admin_user_id}:{timestamp}"
    token = hashlib.sha256(payload.encode()).hexdigest()
    
    ADMIN_TOKENS[token] = {
        "team_id": team_id,
        "admin_user_id": admin_user_id,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }
    
    log_security_event(
        team_id=team_id,
        event_type="admin_token_generated",
        severity="info",
        admin_user_id=admin_user_id
    )
    
    return token

def verify_admin_token(token: str) -> Dict[str, Any]:
    """Verify admin token and return token info"""
    if token not in ADMIN_TOKENS:
        raise HTTPException(401, "Invalid admin token")
    
    token_info = ADMIN_TOKENS[token]
    
    # Check expiration
    if datetime.now() > token_info["expires_at"]:
        del ADMIN_TOKENS[token]
        raise HTTPException(401, "Admin token expired")
    
    return token_info

def log_admin_action(team_id: str, action: str, admin_user_id: str, details: Dict[str, Any] = None):
    """Log admin action for audit trail"""
    action_log = {
        "timestamp": datetime.now().isoformat(),
        "team_id": team_id,
        "admin_user_id": admin_user_id,
        "action": action,
        "details": details or {}
    }
    
    ADMIN_ACTIONS_LOG.append(action_log)
    
    log_security_event(
        team_id=team_id,
        event_type=f"admin_action_{action}",
        severity="info",
        admin_user_id=admin_user_id,
        **details or {}
    )

# === Data Export Functions ===

class DataExporter:
    """Handle data export operations for GDPR compliance and backups"""
    
    def __init__(self, installations: Dict, message_vectors: Dict):
        self.installations = installations
        self.message_vectors = message_vectors
    
    async def export_team_data(self, team_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Export all data for a team (GDPR compliance)"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        installation = self.installations[team_id]
        messages = self.message_vectors.get(team_id, [])
        
        # Prepare export data
        export_data = {
            "export_info": {
                "team_id": team_id,
                "export_time": datetime.now().isoformat(),
                "exported_by": admin_user_id,
                "data_types": ["installation", "messages", "vectors"]
            },
            "installation": {
                "team_id": installation.team_id,
                "installer_user_id": installation.installer_user_id,
                "installation_time": installation.installation_time.isoformat(),
                "scopes": installation.scopes,
                "channel_allowlist": installation.channel_allowlist,
                "retention_days": installation.retention_days,
                "is_active": installation.is_active,
                "settings": installation.settings
            },
            "messages": {
                "total_count": len(messages),
                "messages": []
            }
        }
        
        # Export message data (without vectors for size)
        for i, vector_data in enumerate(messages):
            message = vector_data.get("message", {})
            export_data["messages"]["messages"].append({
                "index": i,
                "message_id": getattr(message, 'ts', f"msg_{i}"),
                "channel_id": getattr(message, 'channel_id', 'unknown'),
                "user_id": getattr(message, 'user_id', 'unknown'),
                "text": getattr(message, 'text', '')[:500],  # Truncate for export
                "timestamp": getattr(message, 'ts', ''),
                "thread_ts": getattr(message, 'thread_ts', None),
                "indexed_at": vector_data.get("indexed_at", ""),
                "has_vector": "vector" in vector_data
            })
        
        log_admin_action(
            team_id=team_id,
            action="data_export",
            admin_user_id=admin_user_id,
            details={"message_count": len(messages)}
        )
        
        return export_data
    
    async def export_team_data_csv(self, team_id: str, admin_user_id: str) -> StreamingResponse:
        """Export team data as CSV file"""
        
        export_data = await self.export_team_data(team_id, admin_user_id)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Message ID", "Channel ID", "User ID", "Text", 
            "Timestamp", "Thread TS", "Indexed At"
        ])
        
        # Write message data
        for msg in export_data["messages"]["messages"]:
            writer.writerow([
                msg["message_id"],
                msg["channel_id"],
                msg["user_id"],
                msg["text"],
                msg["timestamp"],
                msg["thread_ts"] or "",
                msg["indexed_at"]
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=slack_data_{team_id}_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    
    async def get_team_statistics(self, team_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a team"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        installation = self.installations[team_id]
        messages = self.message_vectors.get(team_id, [])
        
        # Calculate statistics
        total_messages = len(messages)
        unique_channels = set()
        unique_users = set()
        total_text_length = 0
        
        for vector_data in messages:
            message = vector_data.get("message", {})
            if hasattr(message, 'channel_id'):
                unique_channels.add(message.channel_id)
            if hasattr(message, 'user_id'):
                unique_users.add(message.user_id)
            if hasattr(message, 'text'):
                total_text_length += len(message.text)
        
        # Calculate age of oldest message
        oldest_message_time = None
        newest_message_time = None
        
        for vector_data in messages:
            indexed_at = vector_data.get("indexed_at")
            if indexed_at:
                try:
                    msg_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))
                    if oldest_message_time is None or msg_time < oldest_message_time:
                        oldest_message_time = msg_time
                    if newest_message_time is None or msg_time > newest_message_time:
                        newest_message_time = msg_time
                except:
                    pass
        
        return {
            "team_id": team_id,
            "installation_info": {
                "installed_at": installation.installation_time.isoformat(),
                "installer_user_id": installation.installer_user_id,
                "is_active": installation.is_active,
                "retention_days": installation.retention_days
            },
            "message_statistics": {
                "total_messages": total_messages,
                "unique_channels": len(unique_channels),
                "unique_users": len(unique_users),
                "total_text_length": total_text_length,
                "average_message_length": total_text_length / total_messages if total_messages > 0 else 0,
                "oldest_message": oldest_message_time.isoformat() if oldest_message_time else None,
                "newest_message": newest_message_time.isoformat() if newest_message_time else None
            },
            "channels": list(unique_channels),
            "users": list(unique_users)
        }

# === Data Management Functions ===

class DataManager:
    """Handle data management operations"""
    
    def __init__(self, installations: Dict, message_vectors: Dict):
        self.installations = installations
        self.message_vectors = message_vectors
    
    async def purge_team_data(self, team_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Purge all data for a team (GDPR compliance)"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        # Count data before deletion
        message_count = len(self.message_vectors.get(team_id, []))
        
        # Delete installation
        del self.installations[team_id]
        
        # Delete message vectors
        if team_id in self.message_vectors:
            del self.message_vectors[team_id]
        
        log_admin_action(
            team_id=team_id,
            action="data_purge",
            admin_user_id=admin_user_id,
            details={"deleted_messages": message_count}
        )
        
        return {
            "status": "purged",
            "team_id": team_id,
            "deleted_messages": message_count,
            "purged_at": datetime.now().isoformat()
        }
    
    async def update_retention_policy(self, team_id: str, retention_days: int, admin_user_id: str) -> Dict[str, Any]:
        """Update data retention policy for a team"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        old_retention = self.installations[team_id].retention_days
        self.installations[team_id].retention_days = retention_days
        
        log_admin_action(
            team_id=team_id,
            action="retention_policy_update",
            admin_user_id=admin_user_id,
            details={
                "old_retention_days": old_retention,
                "new_retention_days": retention_days
            }
        )
        
        return {
            "status": "updated",
            "team_id": team_id,
            "old_retention_days": old_retention,
            "new_retention_days": retention_days,
            "updated_at": datetime.now().isoformat()
        }
    
    async def update_channel_allowlist(self, team_id: str, channel_allowlist: List[str], admin_user_id: str) -> Dict[str, Any]:
        """Update channel allowlist for a team"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        old_allowlist = self.installations[team_id].channel_allowlist
        self.installations[team_id].channel_allowlist = channel_allowlist
        
        log_admin_action(
            team_id=team_id,
            action="channel_allowlist_update",
            admin_user_id=admin_user_id,
            details={
                "old_allowlist": old_allowlist,
                "new_allowlist": channel_allowlist
            }
        )
        
        return {
            "status": "updated",
            "team_id": team_id,
            "old_allowlist": old_allowlist,
            "new_allowlist": channel_allowlist,
            "updated_at": datetime.now().isoformat()
        }
    
    async def deactivate_installation(self, team_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Deactivate installation for a team"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        self.installations[team_id].is_active = False
        
        log_admin_action(
            team_id=team_id,
            action="installation_deactivated",
            admin_user_id=admin_user_id
        )
        
        return {
            "status": "deactivated",
            "team_id": team_id,
            "deactivated_at": datetime.now().isoformat()
        }
    
    async def reactivate_installation(self, team_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Reactivate installation for a team"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        self.installations[team_id].is_active = True
        
        log_admin_action(
            team_id=team_id,
            action="installation_reactivated",
            admin_user_id=admin_user_id
        )
        
        return {
            "status": "reactivated",
            "team_id": team_id,
            "reactivated_at": datetime.now().isoformat()
        }

# === System Health and Monitoring ===

class SystemHealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self, installations: Dict, message_vectors: Dict):
        self.installations = installations
        self.message_vectors = message_vectors
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        total_installations = len(self.installations)
        active_installations = len([inst for inst in self.installations.values() if inst.is_active])
        
        total_messages = sum(len(vectors) for vectors in self.message_vectors.values())
        
        # Calculate memory usage (simplified)
        import sys
        memory_usage = sys.getsizeof(self.installations) + sys.getsizeof(self.message_vectors)
        
        # Check for potential issues
        issues = []
        
        if active_installations == 0:
            issues.append("No active installations")
        
        if total_messages == 0:
            issues.append("No indexed messages")
        
        if memory_usage > 100 * 1024 * 1024:  # 100MB
            issues.append("High memory usage")
        
        # Calculate health score
        health_score = 100
        if issues:
            health_score -= len(issues) * 20
        
        return {
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "health_score": max(0, health_score),
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_installations": total_installations,
                "active_installations": active_installations,
                "total_messages": total_messages,
                "memory_usage_bytes": memory_usage
            },
            "issues": issues,
            "recommendations": self._get_recommendations(issues)
        }
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations based on identified issues"""
        recommendations = []
        
        if "No active installations" in issues:
            recommendations.append("Consider promoting the app to get more installations")
        
        if "No indexed messages" in issues:
            recommendations.append("Check message indexing process and channel permissions")
        
        if "High memory usage" in issues:
            recommendations.append("Consider implementing data archiving or database migration")
        
        if not issues:
            recommendations.append("System is running optimally")
        
        return recommendations
    
    async def get_installation_health(self, team_id: str) -> Dict[str, Any]:
        """Get health status for a specific installation"""
        
        if team_id not in self.installations:
            raise HTTPException(404, f"No installation found for team {team_id}")
        
        installation = self.installations[team_id]
        messages = self.message_vectors.get(team_id, [])
        
        issues = []
        
        if not installation.is_active:
            issues.append("Installation is deactivated")
        
        if len(messages) == 0:
            issues.append("No indexed messages")
        
        # Check if installation is recent
        days_since_install = (datetime.now() - installation.installation_time).days
        if days_since_install > 30 and len(messages) == 0:
            issues.append("Installation is old but has no indexed messages")
        
        health_score = 100
        if issues:
            health_score -= len(issues) * 25
        
        return {
            "team_id": team_id,
            "status": "healthy" if health_score >= 75 else "warning" if health_score >= 50 else "critical",
            "health_score": max(0, health_score),
            "installation_info": {
                "is_active": installation.is_active,
                "installed_at": installation.installation_time.isoformat(),
                "days_since_install": days_since_install
            },
            "message_info": {
                "total_messages": len(messages),
                "channels_indexed": len(set(getattr(msg.get("message", {}), 'channel_id', '') for msg in messages))
            },
            "issues": issues
        }

# === Audit and Compliance ===

class AuditManager:
    """Handle audit logging and compliance reporting"""
    
    def __init__(self):
        self.audit_log = ADMIN_ACTIONS_LOG
    
    async def get_audit_log(self, team_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        
        log_entries = self.audit_log
        
        if team_id:
            log_entries = [entry for entry in log_entries if entry["team_id"] == team_id]
        
        # Sort by timestamp (newest first)
        log_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return log_entries[:limit]
    
    async def get_compliance_report(self, team_id: str) -> Dict[str, Any]:
        """Generate compliance report for a team"""
        
        team_audit_log = await self.get_audit_log(team_id)
        
        # Analyze audit log
        data_exports = len([entry for entry in team_audit_log if entry["action"] == "data_export"])
        data_purges = len([entry for entry in team_audit_log if entry["action"] == "data_purge"])
        policy_changes = len([entry for entry in team_audit_log if "update" in entry["action"]])
        
        return {
            "team_id": team_id,
            "report_generated_at": datetime.now().isoformat(),
            "audit_summary": {
                "total_actions": len(team_audit_log),
                "data_exports": data_exports,
                "data_purges": data_purges,
                "policy_changes": policy_changes
            },
            "compliance_status": {
                "has_audit_trail": len(team_audit_log) > 0,
                "data_export_capability": True,
                "data_purge_capability": True,
                "policy_management": True
            },
            "recent_actions": team_audit_log[:10]  # Last 10 actions
        }
