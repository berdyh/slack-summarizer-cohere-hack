"""
Monitoring and Metrics Module for Slack RAG Bot
Implements Prometheus-style metrics and structured logging
"""

import time
import structlog
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# === Prometheus Metrics ===

# Business metrics
slack_queries_total = Counter(
    'slack_queries_total', 
    'Total number of Slack queries processed', 
    ['team_id', 'channel_id', 'query_type', 'status']
)

response_quality_score = Histogram(
    'response_quality_score', 
    'User-rated response quality (0-5 scale)',
    buckets=[0, 1, 2, 3, 4, 5]
)

citation_accuracy = Gauge(
    'citation_accuracy_ratio', 
    'Fraction of citations that are valid and relevant'
)

# Technical metrics
cohere_api_calls_total = Counter(
    'cohere_api_calls_total', 
    'Total Cohere API calls', 
    ['model', 'operation', 'status']
)

cohere_api_cost_usd = Counter(
    'cohere_api_cost_usd', 
    'Cohere API costs in USD', 
    ['model', 'operation']
)

slack_rate_limits_hit = Counter(
    'slack_rate_limits_hit_total', 
    'Slack API rate limit encounters', 
    ['method', 'endpoint']
)

message_ingestion_lag_seconds = Histogram(
    'message_ingestion_lag_seconds', 
    'Time from Slack event to indexed',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

query_processing_duration_seconds = Histogram(
    'query_processing_duration_seconds',
    'Time to process a query end-to-end',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

vector_search_duration_seconds = Histogram(
    'vector_search_duration_seconds',
    'Time to perform vector similarity search',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

embedding_generation_duration_seconds = Histogram(
    'embedding_generation_duration_seconds',
    'Time to generate embeddings',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# System metrics
active_installations = Gauge(
    'active_installations_total',
    'Number of active Slack installations'
)

indexed_messages_total = Gauge(
    'indexed_messages_total',
    'Total number of indexed messages',
    ['team_id']
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

# === Metrics Collection Classes ===

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    team_id: str
    channel_id: Optional[str]
    query_type: str  # 'slash_command', 'app_mention', 'thread_summary'
    start_time: float
    end_time: Optional[float] = None
    status: str = 'processing'  # 'success', 'error', 'timeout'
    response_time: Optional[float] = None
    embedding_time: Optional[float] = None
    search_time: Optional[float] = None
    generation_time: Optional[float] = None
    sources_found: int = 0
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

    def finish(self, status: str = 'success', error_message: Optional[str] = None):
        """Mark query as finished and record metrics"""
        self.end_time = time.time()
        self.response_time = self.end_time - self.start_time
        self.status = status
        self.error_message = error_message
        
        # Record metrics
        slack_queries_total.labels(
            team_id=self.team_id,
            channel_id=self.channel_id or 'unknown',
            query_type=self.query_type,
            status=status
        ).inc()
        
        if self.response_time:
            query_processing_duration_seconds.observe(self.response_time)
        
        if self.embedding_time:
            embedding_generation_duration_seconds.observe(self.embedding_time)
        
        if self.search_time:
            vector_search_duration_seconds.observe(self.search_time)
        
        if self.generation_time:
            embedding_generation_duration_seconds.observe(self.generation_time)
        
        # Log the query
        logger.info(
            "query_completed",
            team_id=self.team_id,
            channel_id=self.channel_id,
            query_type=self.query_type,
            status=status,
            response_time=self.response_time,
            sources_found=self.sources_found,
            confidence_score=self.confidence_score,
            error_message=error_message
        )

@dataclass
class IngestionMetrics:
    """Metrics for message ingestion"""
    team_id: str
    channel_id: str
    message_count: int
    start_time: float
    end_time: Optional[float] = None
    embedding_time: Optional[float] = None
    storage_time: Optional[float] = None
    errors: int = 0

    def finish(self):
        """Mark ingestion as finished and record metrics"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        message_ingestion_lag_seconds.observe(total_time)
        
        if self.embedding_time:
            embedding_generation_duration_seconds.observe(self.embedding_time)
        
        logger.info(
            "ingestion_completed",
            team_id=self.team_id,
            channel_id=self.channel_id,
            message_count=self.message_count,
            total_time=total_time,
            embedding_time=self.embedding_time,
            storage_time=self.storage_time,
            errors=self.errors
        )

# === Metrics Collection Functions ===

class MetricsCollector:
    """Centralized metrics collection and management"""
    
    def __init__(self):
        self.active_queries: Dict[str, QueryMetrics] = {}
        self.active_ingestions: Dict[str, IngestionMetrics] = {}
    
    def start_query(self, team_id: str, channel_id: Optional[str], query_type: str) -> str:
        """Start tracking a query and return query_id"""
        query_id = f"{team_id}_{int(time.time() * 1000)}"
        metrics = QueryMetrics(
            team_id=team_id,
            channel_id=channel_id,
            query_type=query_type,
            start_time=time.time()
        )
        self.active_queries[query_id] = metrics
        
        logger.info(
            "query_started",
            query_id=query_id,
            team_id=team_id,
            channel_id=channel_id,
            query_type=query_type
        )
        
        return query_id
    
    def finish_query(self, query_id: str, status: str = 'success', error_message: Optional[str] = None):
        """Finish tracking a query"""
        if query_id in self.active_queries:
            self.active_queries[query_id].finish(status, error_message)
            del self.active_queries[query_id]
    
    def update_query_metrics(self, query_id: str, **kwargs):
        """Update query metrics with timing or other data"""
        if query_id in self.active_queries:
            metrics = self.active_queries[query_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def start_ingestion(self, team_id: str, channel_id: str, message_count: int) -> str:
        """Start tracking message ingestion"""
        ingestion_id = f"{team_id}_{channel_id}_{int(time.time() * 1000)}"
        metrics = IngestionMetrics(
            team_id=team_id,
            channel_id=channel_id,
            message_count=message_count,
            start_time=time.time()
        )
        self.active_ingestions[ingestion_id] = metrics
        
        logger.info(
            "ingestion_started",
            ingestion_id=ingestion_id,
            team_id=team_id,
            channel_id=channel_id,
            message_count=message_count
        )
        
        return ingestion_id
    
    def finish_ingestion(self, ingestion_id: str):
        """Finish tracking message ingestion"""
        if ingestion_id in self.active_ingestions:
            self.active_ingestions[ingestion_id].finish()
            del self.active_ingestions[ingestion_id]
    
    def update_ingestion_metrics(self, ingestion_id: str, **kwargs):
        """Update ingestion metrics"""
        if ingestion_id in self.active_ingestions:
            metrics = self.active_ingestions[ingestion_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def record_cohere_api_call(self, model: str, operation: str, status: str, cost_usd: float = 0.0):
        """Record Cohere API usage"""
        cohere_api_calls_total.labels(
            model=model,
            operation=operation,
            status=status
        ).inc()
        
        if cost_usd > 0:
            cohere_api_cost_usd.labels(
                model=model,
                operation=operation
            ).inc(cost_usd)
        
        logger.info(
            "cohere_api_call",
            model=model,
            operation=operation,
            status=status,
            cost_usd=cost_usd
        )
    
    def record_slack_rate_limit(self, method: str, endpoint: str):
        """Record Slack API rate limit hit"""
        slack_rate_limits_hit.labels(
            method=method,
            endpoint=endpoint
        ).inc()
        
        logger.warning(
            "slack_rate_limit_hit",
            method=method,
            endpoint=endpoint
        )
    
    def update_system_metrics(self, installations: int, indexed_messages: Dict[str, int]):
        """Update system-level metrics"""
        active_installations.set(installations)
        
        for team_id, count in indexed_messages.items():
            indexed_messages_total.labels(team_id=team_id).set(count)
        
        # Update memory usage (simplified)
        import psutil
        memory_usage_bytes.set(psutil.Process().memory_info().rss)
    
    def record_response_quality(self, score: float):
        """Record user-rated response quality"""
        response_quality_score.observe(score)
        
        logger.info(
            "response_quality_rated",
            score=score
        )
    
    def record_citation_accuracy(self, accuracy: float):
        """Record citation accuracy ratio"""
        citation_accuracy.set(accuracy)
        
        logger.info(
            "citation_accuracy_updated",
            accuracy=accuracy
        )

# Global metrics collector instance
metrics_collector = MetricsCollector()

# === Prometheus Endpoint ===

def get_metrics_response() -> Response:
    """Generate Prometheus metrics response"""
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

# === Utility Functions ===

def log_installation_event(team_id: str, event_type: str, **kwargs):
    """Log installation-related events"""
    logger.info(
        "installation_event",
        team_id=team_id,
        event_type=event_type,
        **kwargs
    )

def log_security_event(team_id: str, event_type: str, severity: str = "info", **kwargs):
    """Log security-related events"""
    logger.warning(
        "security_event",
        team_id=team_id,
        event_type=event_type,
        severity=severity,
        **kwargs
    )

def log_performance_issue(team_id: str, issue_type: str, **kwargs):
    """Log performance issues"""
    logger.warning(
        "performance_issue",
        team_id=team_id,
        issue_type=issue_type,
        **kwargs
    )

# === Cost Estimation (Simplified) ===

# Approximate Cohere API costs (as of 2024)
COHERE_COSTS = {
    "embed-english-v3.0": {
        "per_1k_tokens": 0.0001,  # $0.10 per 1M tokens
        "per_request": 0.0001     # Base cost per request
    },
    "command-r": {
        "per_1k_tokens": 0.0015,  # $1.50 per 1M tokens
        "per_request": 0.0001
    },
    "rerank-english-v3.0": {
        "per_1k_tokens": 0.0001,
        "per_request": 0.0001
    }
}

def estimate_cohere_cost(model: str, tokens: int, operation: str) -> float:
    """Estimate Cohere API cost for a request"""
    if model not in COHERE_COSTS:
        return 0.0
    
    cost_info = COHERE_COSTS[model]
    token_cost = (tokens / 1000) * cost_info["per_1k_tokens"]
    request_cost = cost_info["per_request"]
    
    return token_cost + request_cost
