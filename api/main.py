from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import re
from threading import Lock
import time
from typing import Annotated, Any, Deque, Optional
from urllib.parse import urlparse
from uuid import uuid4

import jwt
import numpy as np
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response, Security, WebSocket, WebSocketDisconnect, status
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from brain.audit import AuditLogger
from brain import knowledge_ingestion
from brain.autonomous_learning import KnowledgeAutomationService
from brain.config import AppConfig
from brain.llm_bridge import OptionalLLMBridge
from brain.logging_utils import configure_logging
from brain.system import IntegratedArtificialBrain
from brain.text_interface import TextBrainInterface
from brain.token_store import RefreshTokenStore
from brain.websocket_manager import WebSocketConnectionManager

try:
    import redis
except Exception:  # pragma: no cover - optional dependency at runtime
    redis = None


logger = logging.getLogger('iabs.api')
http_bearer = HTTPBearer(auto_error=False)

REQUEST_COUNT = Counter('iabs_http_requests_total', 'Total HTTP requests', ['method', 'path', 'status'])
REQUEST_LATENCY = Histogram('iabs_http_request_duration_seconds', 'HTTP request latency', ['method', 'path'])
CHAT_REQUESTS = Counter('iabs_chat_requests_total', 'Total chat requests')
CYCLE_REQUESTS = Counter('iabs_cycle_requests_total', 'Total cycle requests')
FEEDBACK_REQUESTS = Counter('iabs_feedback_requests_total', 'Total feedback requests')
WEBSOCKET_MESSAGES = Counter('iabs_websocket_messages_total', 'Total websocket chat messages')
AUTH_LOGINS = Counter('iabs_auth_logins_total', 'Successful auth logins', ['role'])
AUTH_REFRESHES = Counter('iabs_auth_refresh_total', 'Successful refresh token exchanges', ['role'])
AUTH_LOGOUTS = Counter('iabs_auth_logout_total', 'Successful refresh token revocations', ['role'])
AUDIT_EVENTS = Counter('iabs_audit_events_total', 'Total audit events', ['event_type', 'outcome'])

REQUEST_ID_HEADER = 'X-Request-ID'
SAFE_REQUEST_ID_PATTERN = re.compile(r'^[A-Za-z0-9._:-]{1,128}$')
ALLOWED_WEBHOOK_METHODS = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}
BASE_SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'X-Permitted-Cross-Domain-Policies': 'none',
    'Cache-Control': 'no-store',
}


class RateLimiterProtocol:
    backend_name = 'unknown'

    def check(self, key: str) -> tuple[bool, int, float]:  # pragma: no cover - interface only
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiterProtocol):
    backend_name = 'memory'

    def __init__(self, requests: int, window_seconds: int):
        self.requests = max(1, int(requests))
        self.window_seconds = max(1, int(window_seconds))
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._events[key]
            boundary = now - self.window_seconds
            while bucket and bucket[0] <= boundary:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(0.0, self.window_seconds - (now - bucket[0]))
                return False, 0, retry_after
            bucket.append(now)
            remaining = self.requests - len(bucket)
            if len(self._events) > 2048:
                stale_keys = [name for name, events in self._events.items() if not events or events[-1] <= boundary]
                for name in stale_keys:
                    self._events.pop(name, None)
            return True, remaining, 0.0


class RedisRateLimiter(RateLimiterProtocol):
    backend_name = 'redis'

    def __init__(self, config: AppConfig, fallback: InMemoryRateLimiter):
        self.requests = max(1, int(config.rate_limit_requests))
        self.window_seconds = max(1, int(config.rate_limit_window_seconds))
        self.prefix = config.redis_rate_limit_prefix
        self.fallback = fallback
        self.client = None
        if redis is not None:
            try:
                self.client = redis.Redis.from_url(
                    config.redis_url,
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                )
                self.client.ping()
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                logger.warning('Redis rate limiter unavailable, falling back to memory: %s', exc)
                self.client = None

    def check(self, key: str) -> tuple[bool, int, float]:
        if self.client is None:
            return self.fallback.check(key)
        try:
            window_id = int(time.time() // self.window_seconds)
            redis_key = f'{self.prefix}:{key}:{window_id}'
            current = int(self.client.incr(redis_key))
            if current == 1:
                self.client.expire(redis_key, self.window_seconds)
            ttl = int(self.client.ttl(redis_key))
            if ttl < 0:
                ttl = self.window_seconds
                self.client.expire(redis_key, self.window_seconds)
            allowed = current <= self.requests
            remaining = max(0, self.requests - current)
            return allowed, remaining, float(ttl)
        except Exception as exc:  # pragma: no cover - depends on runtime infra
            logger.warning('Redis rate limiter failed during request, falling back to memory: %s', exc)
            return self.fallback.check(key)


class CycleInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            'examples': [
                {
                    'visual_input': [[1.0] * 64 for _ in range(64)],
                    'audio_input': [0.25] * 1024,
                    'position': [1.0, 2.0],
                    'importance': 0.82,
                }
            ]
        }
    )

    visual_input: list[list[float]] = Field(..., description='مصفوفة 64x64')
    audio_input: list[float] = Field(..., description='متجه بطول 1024')
    position: list[float] = Field(..., min_length=2, max_length=2)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_shapes(self) -> 'CycleInput':
        if len(self.visual_input) != 64 or any(len(row) != 64 for row in self.visual_input):
            raise ValueError('visual_input يجب أن تكون مصفوفة 64x64 بالكامل')
        if len(self.audio_input) != 1024:
            raise ValueError('audio_input يجب أن يكون بطول 1024')
        numeric_values = [value for row in self.visual_input for value in row]
        numeric_values.extend(self.audio_input)
        numeric_values.extend(self.position)
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError('جميع المدخلات يجب أن تحتوي على أرقام منتهية فقط')
        return self


class ConsolidationInput(BaseModel):
    min_importance: float = Field(default=0.7, ge=0.0, le=1.0, examples=[0.7])


class ChatInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            'examples': [
                {
                    'text': 'حلل الرسالة دي واحتفظ بيها في الذاكرة.',
                    'importance': 0.9,
                    'image_refs': ['https://example.com/frame-1.png'],
                    'audio_refs': ['https://example.com/context.mp3'],
                    'tags': ['project', 'memory'],
                }
            ]
        }
    )

    text: str = Field(..., min_length=1, max_length=4000, examples=['إزيك يا دماغ؟'])
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    image_refs: list[str] = Field(default_factory=list, max_length=6)
    audio_refs: list[str] = Field(default_factory=list, max_length=6)
    tags: list[str] = Field(default_factory=list, max_length=8)


class EpisodeInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    image_refs: list[str] = Field(default_factory=list, max_length=8)
    audio_refs: list[str] = Field(default_factory=list, max_length=8)
    tags: list[str] = Field(default_factory=list, max_length=10)
    related_memory_keys: list[str] = Field(default_factory=list, max_length=10)
    importance: float = Field(default=0.78, ge=0.0, le=1.0)


class FeedbackInput(BaseModel):
    reward: float = Field(..., ge=-1.0, le=1.0, examples=[1.0, -0.5])
    memory_key: str | None = Field(default=None, min_length=3, max_length=256)
    feedback_text: str | None = Field(default=None, max_length=1000)


class PersonalityUpdateInput(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    tone: str | None = Field(default=None, min_length=1, max_length=40)
    style: str | None = Field(default=None, min_length=1, max_length=40)
    identity_statement: str | None = Field(default=None, min_length=1, max_length=400)
    behavioral_rules: list[str] | None = Field(default=None, max_length=10)
    preferred_language: str | None = Field(default=None, min_length=2, max_length=20)


class SelfImproveInput(BaseModel):
    trigger: str = Field(default='manual', min_length=2, max_length=60)


class TheoryOfMindInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


def _trim_required_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError('القيمة النصية مطلوبة')
    return text


def _trim_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in re.split(r'[,\n]+', value) if item.strip()]
    else:
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(raw_items))


def _validate_http_target_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('target_url لازم يكون http أو https صالح')
    return value


def _validate_http_method(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in ALLOWED_WEBHOOK_METHODS:
        raise ValueError(f"method لازم تكون واحدة من: {', '.join(sorted(ALLOWED_WEBHOOK_METHODS))}")
    return value


def _validate_header_name(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if not re.fullmatch(r'[A-Za-z0-9-]{3,80}', candidate):
        raise ValueError('اسم الـ header يجب أن يحتوي على أحرف لاتينية أو أرقام أو - فقط')
    return candidate


def _sanitize_request_id(value: Any) -> str:
    candidate = str(value or '').strip()
    if SAFE_REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid4().hex


def _apply_security_headers(response: Response, config: AppConfig) -> Response:
    for header_name, header_value in BASE_SECURITY_HEADERS.items():
        response.headers.setdefault(header_name, header_value)
    if config.environment == 'production':
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response


async def _security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    config: AppConfig = request.app.state.config
    return _apply_security_headers(response, config)


class ActionHookCreateInput(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    event: str = Field(..., min_length=3, max_length=80)
    action_type: str = Field(default='webhook', min_length=3, max_length=40)
    target_url: str | None = Field(default=None, max_length=500)
    method: str = Field(default='POST', min_length=3, max_length=10)
    headers: dict[str, str] = Field(default_factory=dict)
    payload_template: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list, max_length=10)
    cooldown_seconds: int = Field(default=0, ge=0, le=86400)
    active: bool = True

    _normalize_name = field_validator('name', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_event = field_validator('event', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_action_type = field_validator('action_type', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_target_url = field_validator('target_url', mode='before')(lambda cls, value: _trim_optional_text(value))
    _validate_target_url = field_validator('target_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_method = field_validator('method', mode='before')(lambda cls, value: _trim_required_text(value).upper())
    _validate_method = field_validator('method')(lambda cls, value: _validate_http_method(value))
    _normalize_keywords = field_validator('keywords', mode='before')(lambda cls, value: _normalize_keywords(value))


class ActionHookUpdateInput(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    event: str | None = Field(default=None, min_length=3, max_length=80)
    action_type: str | None = Field(default=None, min_length=3, max_length=40)
    target_url: str | None = Field(default=None, max_length=500)
    method: str | None = Field(default=None, min_length=3, max_length=10)
    headers: dict[str, str] | None = None
    payload_template: dict[str, Any] | None = None
    keywords: list[str] | None = Field(default=None, max_length=10)
    cooldown_seconds: int | None = Field(default=None, ge=0, le=86400)
    active: bool | None = None

    _normalize_name = field_validator('name', mode='before')(lambda cls, value: _trim_optional_text(value))
    _normalize_event = field_validator('event', mode='before')(lambda cls, value: _trim_optional_text(value))
    _normalize_action_type = field_validator('action_type', mode='before')(lambda cls, value: _trim_optional_text(value))
    _normalize_target_url = field_validator('target_url', mode='before')(lambda cls, value: _trim_optional_text(value))
    _validate_target_url = field_validator('target_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_method = field_validator('method', mode='before')(lambda cls, value: _trim_optional_text(value).upper() if value is not None else None)
    _validate_method = field_validator('method')(lambda cls, value: _validate_http_method(value))
    _normalize_keywords = field_validator('keywords', mode='before')(lambda cls, value: _normalize_keywords(value) if value is not None else None)


class ActionHookTriggerInput(BaseModel):
    event: str = Field(..., min_length=3, max_length=80)
    text: str = Field(default='', max_length=4000)
    decision: str | None = Field(default=None, max_length=120)
    topics: list[str] = Field(default_factory=list, max_length=10)
    dry_run: bool = True
    allow_network: bool = False


class GoalCreateInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str = Field(default='', max_length=500)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    target_keywords: list[str] | None = Field(default=None, max_length=10)


class GoalUpdateInput(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    status: str | None = Field(default=None, min_length=3, max_length=20)
    priority: float | None = Field(default=None, ge=0.0, le=1.0)
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = Field(default=None, max_length=300)
    target_keywords: list[str] | None = Field(default=None, max_length=10)


class KnowledgeIngestTextInput(BaseModel):
    text: str = Field(..., min_length=40, max_length=200000)
    source_name: str = Field(..., min_length=2, max_length=160)
    source_url: str | None = Field(default=None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=12)
    chunk_size_chars: int = Field(default=900, ge=300, le=2500)
    overlap_chars: int = Field(default=120, ge=0, le=600)
    min_chunk_chars: int = Field(default=180, ge=80, le=1200)
    importance: float = Field(default=0.76, ge=0.0, le=1.0)

    _normalize_text = field_validator('text', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_source_name = field_validator('source_name', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_source_url = field_validator('source_url', mode='before')(lambda cls, value: _trim_optional_text(value))
    _validate_source_url = field_validator('source_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_tags = field_validator('tags', mode='before')(lambda cls, value: _normalize_keywords(value))


class KnowledgeIngestUrlInput(BaseModel):
    target_url: str = Field(..., min_length=8, max_length=500)
    source_name: str | None = Field(default=None, min_length=2, max_length=160)
    tags: list[str] = Field(default_factory=list, max_length=12)
    chunk_size_chars: int = Field(default=900, ge=300, le=2500)
    overlap_chars: int = Field(default=120, ge=0, le=600)
    min_chunk_chars: int = Field(default=180, ge=80, le=1200)
    importance: float = Field(default=0.76, ge=0.0, le=1.0)

    _normalize_target_url = field_validator('target_url', mode='before')(lambda cls, value: _trim_required_text(value))
    _validate_target_url = field_validator('target_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_source_name = field_validator('source_name', mode='before')(lambda cls, value: _trim_optional_text(value))
    _normalize_tags = field_validator('tags', mode='before')(lambda cls, value: _normalize_keywords(value))


class KnowledgeAutomationSourceInput(BaseModel):
    name: str = Field(..., min_length=2, max_length=160)
    feed_url: str = Field(..., min_length=8, max_length=500)
    mode: str = Field(default='auto', min_length=3, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    active: bool = True
    interval_seconds: int = Field(default=86400, ge=300, le=604800)
    max_items_per_run: int = Field(default=5, ge=1, le=12)
    importance: float = Field(default=0.77, ge=0.3, le=0.95)

    _normalize_name = field_validator('name', mode='before')(lambda cls, value: _trim_required_text(value))
    _normalize_feed_url = field_validator('feed_url', mode='before')(lambda cls, value: _trim_required_text(value))
    _validate_feed_url = field_validator('feed_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_mode = field_validator('mode', mode='before')(lambda cls, value: _trim_required_text(value).lower())
    _normalize_tags = field_validator('tags', mode='before')(lambda cls, value: _normalize_keywords(value))
    _normalize_keywords = field_validator('keywords', mode='before')(lambda cls, value: _normalize_keywords(value))


class KnowledgeAutomationSourceUpdateInput(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    feed_url: str | None = Field(default=None, min_length=8, max_length=500)
    mode: str | None = Field(default=None, min_length=3, max_length=20)
    tags: list[str] | None = Field(default=None, max_length=12)
    keywords: list[str] | None = Field(default=None, max_length=12)
    active: bool | None = None
    interval_seconds: int | None = Field(default=None, ge=300, le=604800)
    max_items_per_run: int | None = Field(default=None, ge=1, le=12)
    importance: float | None = Field(default=None, ge=0.3, le=0.95)

    _normalize_name = field_validator('name', mode='before')(lambda cls, value: _trim_optional_text(value))
    _normalize_feed_url = field_validator('feed_url', mode='before')(lambda cls, value: _trim_optional_text(value))
    _validate_feed_url = field_validator('feed_url')(lambda cls, value: _validate_http_target_url(value))
    _normalize_mode = field_validator('mode', mode='before')(lambda cls, value: _trim_optional_text(value).lower() if value is not None else None)
    _normalize_tags = field_validator('tags', mode='before')(lambda cls, value: _normalize_keywords(value) if value is not None else None)
    _normalize_keywords = field_validator('keywords', mode='before')(lambda cls, value: _normalize_keywords(value) if value is not None else None)


class KnowledgeAutomationRunInput(BaseModel):
    force: bool = True


class AuthRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            'examples': [
                {
                    'username': 'admin',
                    'password': 'change-me-now',
                }
            ]
        }
    )

    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)

    _normalize_username = field_validator('username', mode='before')(lambda cls, value: _trim_required_text(value))


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)

    _normalize_refresh_token = field_validator('refresh_token', mode='before')(lambda cls, value: _trim_required_text(value))


class WebSocketChatInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    _normalize_text = field_validator('text', mode='before')(lambda cls, value: _trim_required_text(value))


TokenPayload = dict[str, Any]

CYCLE_EXAMPLES = {
    'default': {
        'summary': 'دورة حسية كاملة',
        'value': {
            'visual_input': [[1.0] * 64 for _ in range(64)],
            'audio_input': [0.25] * 1024,
            'position': [1.0, 2.0],
            'importance': 0.82,
        },
    }
}

CHAT_EXAMPLES = {
    'default': {
        'summary': 'رسالة شات بسيطة',
        'value': {
            'text': 'إزيك يا دماغ؟',
            'importance': 0.9,
        },
    }
}

AUTH_EXAMPLES = {
    'default': {
        'summary': 'تسجيل دخول إداري',
        'value': {
            'username': 'admin',
            'password': 'change-me-now',
        },
    }
}

PROTECTED_PATHS = [
    '/status', '/wake', '/sleep', '/brain/save-state', '/brain/load-state', '/brain/diagnostics', '/brain/personality',
    '/brain/affect', '/brain/user-model', '/brain/user-model/rebuild', '/brain/theory-of-mind', '/brain/config/posture',
    '/brain/action-hooks', '/brain/action-hooks/{hook_id}', '/brain/action-hooks/trigger', '/brain/action-hooks/events',
    '/brain/goals', '/brain/goals/{goal_id}', '/brain/context', '/brain/self-improve', '/brain/sleep-report', '/brain/roadmap',
    '/brain/dashboard', '/brain/next-actions', '/brain/anomalies',
    '/memory/save', '/memory/reload', '/memory/latest', '/memory/consolidate', '/memory/search', '/memory/recent',
    '/memory/episodes', '/memory/episodes/search', '/knowledge/ingest/text', '/knowledge/ingest/url', '/knowledge/search',
    '/knowledge/sources', '/knowledge/sources/{ingest_id}', '/knowledge/analytics', '/knowledge/briefing', '/knowledge/verify',
    '/knowledge/automation/status', '/knowledge/automation/run', '/knowledge/automation/sources', '/knowledge/automation/sources/{source_id}', '/knowledge/automation/sources/{source_id}/run',
    '/cycle', '/chat', '/feedback', '/memory/insights', '/memory/export', '/memory/{key}',
    '/auth/me', '/audit/summary', '/audit/recent'
]

OPENAPI_TAGS_METADATA = [
    {'name': 'Auth', 'description': 'JWT login / refresh / logout / current identity.'},
    {'name': 'System', 'description': 'System health, readiness, posture, diagnostics, and runtime state.'},
    {'name': 'Brain', 'description': 'Core brain controls, affect, goals, dashboard, roadmap, and planning.'},
    {'name': 'Memory', 'description': 'Memory storage, retrieval, export, episodes, and insights.'},
    {'name': 'Knowledge', 'description': 'Knowledge ingestion, search, analytics, briefings, and verification.'},
    {'name': 'Audit', 'description': 'Audit trails and governance visibility.'},
    {'name': 'Realtime', 'description': 'Realtime chat and websocket interaction.'},
]

OPENAPI_OPERATION_HINTS = {
    '/': {'get': {'tags': ['System'], 'summary': 'Root runtime summary', 'description': 'Quick summary of version, storage, security posture, and major platform features.'}},
    '/healthz': {'get': {'tags': ['System'], 'summary': 'Liveness probe'}},
    '/readyz': {'get': {'tags': ['System'], 'summary': 'Readiness probe'}},
    '/auth/token': {'post': {'tags': ['Auth'], 'summary': 'Issue access and refresh tokens'}},
    '/auth/refresh': {'post': {'tags': ['Auth'], 'summary': 'Rotate refresh token'}},
    '/auth/logout': {'post': {'tags': ['Auth'], 'summary': 'Revoke refresh token'}},
    '/brain/config/posture': {'get': {'tags': ['System'], 'summary': 'Runtime hardening posture', 'description': 'Sanitized runtime settings with hardening score, warnings, and deployment recommendations.'}},
    '/brain/capabilities': {'get': {'tags': ['Brain'], 'summary': 'Capability matrix'}},
    '/brain/roadmap': {'get': {'tags': ['Brain'], 'summary': 'Roadmap and phased next steps'}},
    '/brain/action-hooks': {'get': {'tags': ['Brain'], 'summary': 'List action hooks'}, 'post': {'tags': ['Brain'], 'summary': 'Create action hook'}},
    '/brain/action-hooks/trigger': {'post': {'tags': ['Brain'], 'summary': 'Trigger action hooks'}},
    '/memory/search': {'get': {'tags': ['Memory'], 'summary': 'Search memory records'}},
    '/memory/insights': {'get': {'tags': ['Memory'], 'summary': 'Memory insights snapshot'}},
    '/knowledge/ingest/text': {'post': {'tags': ['Knowledge'], 'summary': 'Ingest raw text as knowledge'}},
    '/knowledge/ingest/url': {'post': {'tags': ['Knowledge'], 'summary': 'Fetch URL then ingest into knowledge'}},
    '/knowledge/search': {'get': {'tags': ['Knowledge'], 'summary': 'Search ingested chunks'}},
    '/knowledge/analytics': {'get': {'tags': ['Knowledge'], 'summary': 'Knowledge analytics'}},
    '/knowledge/briefing': {'get': {'tags': ['Knowledge'], 'summary': 'Generate knowledge briefing'}},
    '/knowledge/verify': {'get': {'tags': ['Knowledge'], 'summary': 'Verify claim against ingested knowledge', 'description': 'Returns ranked sources, evidence excerpts, provenance chain, confidence explanation, contradiction clusters, and consensus summary.'}},
    '/knowledge/automation/status': {'get': {'tags': ['Knowledge'], 'summary': 'Knowledge automation status'}},
    '/knowledge/automation/run': {'post': {'tags': ['Knowledge'], 'summary': 'Run all autonomous knowledge sources'}},
    '/knowledge/automation/sources': {'get': {'tags': ['Knowledge'], 'summary': 'List autonomous knowledge sources'}, 'post': {'tags': ['Knowledge'], 'summary': 'Register autonomous knowledge source'}},
    '/knowledge/automation/sources/{source_id}': {'patch': {'tags': ['Knowledge'], 'summary': 'Update autonomous knowledge source'}, 'delete': {'tags': ['Knowledge'], 'summary': 'Delete autonomous knowledge source'}},
    '/knowledge/automation/sources/{source_id}/run': {'post': {'tags': ['Knowledge'], 'summary': 'Run autonomous knowledge source now'}},
    '/audit/summary': {'get': {'tags': ['Audit'], 'summary': 'Audit summary'}},
    '/audit/recent': {'get': {'tags': ['Audit'], 'summary': 'Recent audit events'}},
    '/ws/chat': {'get': {'tags': ['Realtime'], 'summary': 'WebSocket chat stream'}},
}


def create_rate_limiter(config: AppConfig) -> RateLimiterProtocol:
    memory_limiter = InMemoryRateLimiter(config.rate_limit_requests, config.rate_limit_window_seconds)
    if config.rate_limit_backend == 'redis':
        return RedisRateLimiter(config, memory_limiter)
    return memory_limiter


def _security_recommendations(config: AppConfig) -> list[str]:
    recommendations: list[str] = []
    posture = config.security_posture_summary()
    if posture['critical_warning_count']:
        recommendations.append('عالج التحذيرات الحرجة أولاً: استبدل الأسرار الافتراضية وفعّل المصادقة قبل أي نشر فعلي.')
    if posture['plaintext_user_count']:
        recommendations.append('حوّل كلمات المرور النصية إلى password_hash باستخدام PBKDF2 قبل النشر.')
    if not posture['memory_encryption_enabled']:
        recommendations.append('فعّل IABS_MEMORY_ENCRYPTION_KEY لحماية الذاكرة أثناء التخزين.')
    if posture['jwt_secret_length'] < 32:
        recommendations.append('ارفع طول JWT secret إلى 32 حرفاً على الأقل مع قيمة عشوائية قوية.')
    if config.environment == 'production' and config.rate_limit_backend != 'redis':
        recommendations.append('في وضع الإنتاج استخدم Redis لثبات الـ rate limiting عبر أكثر من instance.')
    if config.environment == 'production' and config.log_format != 'json':
        recommendations.append('استخدم JSON logging في الإنتاج لتسهيل الرصد والتدقيق.')
    if posture['hardening_score'] < 80:
        recommendations.append('استهدف hardening score أعلى من 80 قبل النشر العام.')
    if not recommendations:
        recommendations.append('الوضع الحالي جيد ولا توجد توصيات عاجلة إضافية.')
    return recommendations


def _request_context_from_request(request: Request | None) -> dict[str, Any]:
    if request is None:
        return {}
    request_id = _sanitize_request_id(
        request.headers.get(REQUEST_ID_HEADER) or getattr(request.state, 'request_id', None) or uuid4().hex
    )
    request.state.request_id = request_id
    client_host = request.client.host if request.client else 'unknown'
    return {
        'request_id': request_id,
        'method': request.method,
        'path': request.url.path,
        'client_host': client_host,
    }


def _audit(
    request: Request | None,
    event_type: str,
    *,
    actor: str,
    role: str,
    outcome: str,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    if request is None:
        return
    audit_logger: AuditLogger = request.app.state.audit_logger
    audit_logger.log(
        event_type,
        actor=actor,
        role=role,
        outcome=outcome,
        action=action,
        details=details,
        request_context=_request_context_from_request(request),
    )
    AUDIT_EVENTS.labels(event_type=event_type, outcome=outcome).inc()


def _create_token(
    config: AppConfig,
    subject: str,
    role: str,
    token_type: str,
    expires_delta: timedelta,
    *,
    token_id: str | None = None,
) -> tuple[str, datetime, str, datetime]:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + expires_delta
    resolved_jti = token_id or uuid4().hex
    payload = {
        'sub': subject,
        'role': role,
        'type': token_type,
        'jti': resolved_jti,
        'iat': issued_at,
        'exp': expires_at,
    }
    token = jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)
    return token, expires_at, resolved_jti, issued_at


def _create_access_token(config: AppConfig, subject: str, role: str) -> tuple[str, datetime, str, datetime]:
    return _create_token(
        config=config,
        subject=subject,
        role=role,
        token_type='access',
        expires_delta=timedelta(minutes=config.jwt_exp_minutes),
    )


def _create_refresh_token(config: AppConfig, subject: str, role: str) -> tuple[str, datetime, str, datetime]:
    return _create_token(
        config=config,
        subject=subject,
        role=role,
        token_type='refresh',
        expires_delta=timedelta(minutes=config.jwt_refresh_exp_minutes),
    )


def _issue_token_pair(
    config: AppConfig,
    token_store: RefreshTokenStore,
    *,
    subject: str,
    role: str,
    previous_refresh_jti: str | None = None,
) -> dict[str, Any]:
    access_token, access_expires_at, access_jti, access_issued_at = _create_access_token(config, subject, role)
    refresh_token, refresh_expires_at, refresh_jti, refresh_issued_at = _create_refresh_token(config, subject, role)
    if previous_refresh_jti:
        token_store.rotate(
            current_jti=previous_refresh_jti,
            new_jti=refresh_jti,
            subject=subject,
            role=role,
            issued_at=refresh_issued_at.timestamp(),
            expires_at=refresh_expires_at.timestamp(),
        )
    else:
        token_store.register(
            refresh_jti,
            subject,
            role,
            refresh_issued_at.timestamp(),
            refresh_expires_at.timestamp(),
        )
    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_type': 'bearer',
        'role': role,
        'access_jti': access_jti,
        'refresh_jti': refresh_jti,
        'expires_in_seconds': config.jwt_exp_minutes * 60,
        'refresh_expires_in_seconds': config.jwt_refresh_exp_minutes * 60,
        'expires_at': access_expires_at.isoformat(),
        'refresh_expires_at': refresh_expires_at.isoformat(),
    }


def _decode_token(config: AppConfig, token: str, expected_type: str = 'access') -> TokenPayload:
    try:
        payload = jwt.decode(token, config.jwt_secret, algorithms=[config.jwt_algorithm])
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='رمز الوصول غير صالح') from exc
    if not payload.get('sub'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='الرمز لا يحتوي على مستخدم صالح')
    token_type = payload.get('type', 'access')
    if expected_type and token_type != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='نوع الرمز غير صحيح لهذه العملية')
    payload['role'] = str(payload.get('role', 'user'))
    payload['jti'] = str(payload.get('jti', ''))
    return payload


async def _require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(http_bearer),
) -> TokenPayload:
    config: AppConfig = request.app.state.config
    if not config.auth_enabled:
        return {'sub': 'anonymous', 'role': 'admin', 'auth_disabled': True, 'type': 'access'}
    if credentials is None or credentials.scheme.lower() != 'bearer':
        _audit(request, 'auth_required', actor='anonymous', role='anonymous', outcome='denied', action='missing_bearer_token')
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='مطلوب Bearer token')
    try:
        return _decode_token(config, credentials.credentials, expected_type='access')
    except HTTPException as exc:
        _audit(request, 'auth_invalid', actor='anonymous', role='anonymous', outcome='denied', action='invalid_access_token', details={'detail': exc.detail})
        raise


def require_permissions(*permissions: str):
    required_permissions = {item.strip() for item in permissions if item.strip()}

    async def dependency(
        request: Request,
        user: TokenPayload = Depends(_require_user),
    ) -> TokenPayload:
        config: AppConfig = request.app.state.config
        user_role = str(user.get('role', 'user')).lower()
        if config.role_has_permissions(user_role, required_permissions):
            return user
        _audit(
            request,
            'access_denied',
            actor=str(user.get('sub', 'unknown')),
            role=user_role,
            outcome='denied',
            action='permission_check',
            details={'required_permissions': sorted(required_permissions)},
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='ليس لديك صلاحية كافية لتنفيذ هذا الإجراء')

    return dependency


async def _rate_limit(request: Request, call_next):
    config: AppConfig = request.app.state.config
    limiter: RateLimiterProtocol = request.app.state.rate_limiter
    public_paths = {'/healthz', '/readyz', '/metrics', '/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc'}
    if not config.rate_limit_enabled or request.url.path in public_paths:
        return await call_next(request)
    client_host = request.client.host if request.client else 'unknown'
    key = f'{client_host}:{request.url.path}'
    allowed, remaining, retry_after = limiter.check(key)
    if not allowed:
        response = Response(
            content=json.dumps({'detail': 'تم تجاوز حد الطلبات، حاول مرة أخرى بعد قليل.'}, ensure_ascii=False),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type='application/json',
            headers={
                'Retry-After': str(int(retry_after) + 1),
                'X-RateLimit-Limit': str(config.rate_limit_requests),
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Backend': getattr(limiter, 'backend_name', 'unknown'),
            },
        )
        _request_context_from_request(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return _apply_security_headers(response, config)
    response = await call_next(request)
    response.headers['X-RateLimit-Limit'] = str(config.rate_limit_requests)
    response.headers['X-RateLimit-Remaining'] = str(max(0, remaining))
    response.headers['X-RateLimit-Backend'] = getattr(limiter, 'backend_name', 'unknown')
    return _apply_security_headers(response, config)


async def _request_size_guard(request: Request, call_next):
    config: AppConfig = request.app.state.config
    if request.method in {'POST', 'PUT', 'PATCH'}:
        max_size = max(1, int(config.max_request_body_bytes))
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                if int(content_length) > max_size:
                    response = Response(
                        content=json.dumps({'detail': 'حجم الطلب أكبر من الحد المسموح به.'}, ensure_ascii=False),
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        media_type='application/json',
                    )
                    _request_context_from_request(request)
                    response.headers[REQUEST_ID_HEADER] = request.state.request_id
                    return _apply_security_headers(response, config)
            except ValueError:
                pass
        body = await request.body()
        if len(body) > max_size:
            response = Response(
                content=json.dumps({'detail': 'حجم الطلب أكبر من الحد المسموح به.'}, ensure_ascii=False),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                media_type='application/json',
            )
            _request_context_from_request(request)
            response.headers[REQUEST_ID_HEADER] = request.state.request_id
            return _apply_security_headers(response, config)

        async def receive() -> dict[str, Any]:
            return {'type': 'http.request', 'body': body, 'more_body': False}

        request._receive = receive
    return await call_next(request)


async def _metrics_middleware(request: Request, call_next):
    request_id = _sanitize_request_id(request.headers.get(REQUEST_ID_HEADER) or uuid4().hex)
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path
    method = request.method
    REQUEST_COUNT.labels(method=method, path=path, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
    response.headers[REQUEST_ID_HEADER] = request_id
    config: AppConfig = request.app.state.config
    return _apply_security_headers(response, config)


def create_app(config: AppConfig | None = None) -> FastAPI:
    runtime_config = config or AppConfig.from_env()
    runtime_config.ensure_directories()
    configure_logging(runtime_config.log_level, runtime_config.log_format)
    security_posture = runtime_config.security_posture_summary()
    if security_posture['warning_count']:
        logger.warning('Security posture warnings: %s', '; '.join(security_posture['warnings']))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        brain = IntegratedArtificialBrain(
            seed=runtime_config.seed,
            memory_capacity=runtime_config.memory_capacity,
            storage_path=runtime_config.memory_path,
            storage_dsn=runtime_config.memory_dsn,
            state_path=runtime_config.state_path,
            autoload_state=runtime_config.autoload_state,
            postgres_pool_min_size=runtime_config.postgres_pool_min_size,
            postgres_pool_max_size=runtime_config.postgres_pool_max_size,
            memory_encryption_key=runtime_config.memory_encryption_key,
        )
        llm_bridge = OptionalLLMBridge(
            enabled=runtime_config.llm_enabled,
            api_key=runtime_config.llm_api_key,
            api_base=runtime_config.llm_api_base,
            model=runtime_config.llm_model,
            timeout_seconds=runtime_config.llm_timeout_seconds,
            temperature=runtime_config.llm_temperature,
            max_tokens=runtime_config.llm_max_tokens,
        )
        app.state.config = runtime_config
        app.state.brain = brain
        app.state.brain_lock = Lock()
        app.state.text_interface = TextBrainInterface(brain, llm_bridge=llm_bridge)
        app.state.rate_limiter = create_rate_limiter(runtime_config)
        app.state.audit_logger = AuditLogger(runtime_config.audit_log_path)
        app.state.token_store = RefreshTokenStore(runtime_config.refresh_token_store_path)
        app.state.websocket_manager = WebSocketConnectionManager()
        app.state.knowledge_automation = KnowledgeAutomationService(
            runtime_config.knowledge_registry_path,
            enabled=runtime_config.knowledge_automation_enabled,
            poll_interval_seconds=runtime_config.knowledge_scheduler_poll_seconds,
        )
        await app.state.knowledge_automation.start(brain=brain, brain_lock=app.state.brain_lock)
        logger.info('IABS application started with version %s', runtime_config.app_version)
        try:
            yield
        finally:
            await app.state.websocket_manager.close_all(reason='server_shutdown')
            await app.state.knowledge_automation.stop()
            with app.state.brain_lock:
                brain.save_state()
                brain.close()
            app.state.token_store.save()
            logger.info('IABS application stopped and brain state saved')

    app = FastAPI(
        title=runtime_config.app_title,
        version=runtime_config.app_version,
        lifespan=lifespan,
        swagger_ui_parameters={'displayRequestDuration': True, 'defaultModelsExpandDepth': 1},
    )

    app.middleware('http')(_request_size_guard)
    app.middleware('http')(_rate_limit)
    app.middleware('http')(_metrics_middleware)
    app.middleware('http')(_security_headers_middleware)

    def runtime() -> tuple[IntegratedArtificialBrain, Lock, TextBrainInterface]:
        return app.state.brain, app.state.brain_lock, app.state.text_interface

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=(
                f'واجهة IABS v{runtime_config.app_version} مع JWT access/refresh tokens وrotation + revocation وRBAC قائم على permissions '
                'وRate Limiting بذاكرة محلية أو Redis وAudit Logging وPrometheus metrics '
                'وWebSocket chat streaming وذاكرة قابلة للبحث وتتبع أهداف وحالة affective state ودورة نوم للتعلم '
                'مع طبقة ردود إبداعية اختيارية عبر LLM وتشفير اختياري للذاكرة أثناء التخزين، '
                'بالإضافة إلى ingest معرفي مع trust scoring وتحليل تعارضات واستعلام verify للمصادر.'
            ),
            routes=app.routes,
            tags=OPENAPI_TAGS_METADATA,
        )
        schema.setdefault('components', {}).setdefault('securitySchemes', {})['HTTPBearer'] = {
            'type': 'http',
            'scheme': 'bearer',
            'bearerFormat': 'JWT',
        }
        schema.setdefault('info', {})['x-iabs-release'] = {
            'version': runtime_config.app_version,
            'hardening_score': runtime_config.security_posture_summary().get('hardening_score'),
        }
        for path in PROTECTED_PATHS:
            if path in schema.get('paths', {}):
                for operation in schema['paths'][path].values():
                    if isinstance(operation, dict):
                        operation['security'] = [{'HTTPBearer': []}]
        for path, methods in OPENAPI_OPERATION_HINTS.items():
            if path not in schema.get('paths', {}):
                continue
            for method, hint in methods.items():
                operation = schema['paths'][path].get(method)
                if isinstance(operation, dict):
                    operation.update(hint)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi

    @app.get('/')
    def root() -> dict[str, Any]:
        return {
            'message': 'Integrated Artificial Brain API is running',
            'version': app.version,
            'auth': runtime_config.public_auth_summary(),
            'storage': runtime_config.storage_summary(),
            'security': runtime_config.security_posture_summary(),
            'features': [
                'jwt_access_and_rotating_refresh_tokens',
                'refresh_token_revocation',
                'permission_based_rbac',
                'memory_search',
                'brain_diagnostics',
                'online_feedback_learning',
                'goal_tracking',
                'affective_state_model',
                'sleep_cycle_consolidation',
                'prometheus_metrics',
                'websocket_chat_streaming',
                'audit_logging',
                'redis_rate_limiting_optional',
                'postgresql_memory_backend_optional',
                'llm_creative_bridge_optional',
                'encrypted_memory_at_rest_optional',
                'prompt_anomaly_detection',
                'neural_dashboard_snapshot',
                'next_action_planner',
                'roadmap_planner',
                'action_hooks_crud',
                'request_body_size_guard',
                'runtime_config_posture_endpoint',
                'security_response_headers',
                'security_posture_hardening_score',
                'memory_insights_endpoint',
                'audit_summary_endpoint',
                'knowledge_ingestion_text',
                'knowledge_ingestion_url',
                'arabic_chunking_pipeline',
                'duplicate_fingerprint_guard',
                'knowledge_sources_summary',
                'knowledge_source_details',
                'knowledge_source_delete',
                'knowledge_analytics',
                'knowledge_source_trust_scoring',
                'knowledge_query_verification',
                'knowledge_contradiction_detection',
                'knowledge_confidence_explanations',
                'knowledge_consensus_summary',
                'knowledge_provenance_chain',
                'knowledge_contradiction_clustering',
                'knowledge_automation_registry',
                'knowledge_scheduler_background_loop',
                'adaptive_source_scoring',
                'topic_momentum_tracking',
                'autonomous_ingest_runs',
                'openapi_tag_catalog',
            ],
            'docs': '/docs',
            'metrics': '/metrics',
            'websocket': '/ws/chat',
        }

    @app.get('/healthz')
    def healthz() -> dict[str, Any]:
        brain, _, _ = runtime()
        memory_health = brain.memory.healthcheck()
        status_value = 'ok' if memory_health.get('status') == 'ok' else 'degraded'
        return {
            'status': status_value,
            'brain_state': brain.state,
            'memory_backend': brain.memory.storage_backend,
            'memory_health': memory_health,
            'storage_exists': bool(runtime_config.state_path.parent.exists()),
            'llm': app.state.text_interface.llm_status(),
            'security': runtime_config.security_posture_summary(),
        }

    @app.get('/readyz')
    def readyz() -> dict[str, Any]:
        brain, _, _ = runtime()
        return {
            'status': 'ready' if brain.memory.healthcheck().get('status') != 'error' else 'degraded',
            'brain_state': brain.state,
            'has_auth_users': bool(runtime_config.auth_users),
            'data_directory': str(runtime_config.state_path.parent),
            'memory_stats': brain.memory.stats(),
            'audit_log_path': str(runtime_config.audit_log_path),
            'rate_limit_backend': getattr(app.state.rate_limiter, 'backend_name', 'unknown'),
            'token_store': app.state.token_store.stats(),
            'storage': runtime_config.storage_summary(),
            'llm': app.state.text_interface.llm_status(),
            'security': runtime_config.security_posture_summary(),
        }

    @app.get('/metrics', response_class=PlainTextResponse)
    def metrics() -> Response:
        if not runtime_config.metrics_enabled:
            raise HTTPException(status_code=404, detail='Prometheus metrics are disabled')
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post('/auth/token')
    def login(
        payload: Annotated[AuthRequest, Body(openapi_examples=AUTH_EXAMPLES)],
        request: Request,
    ) -> dict[str, Any]:
        if not runtime_config.auth_enabled:
            _audit(request, 'auth_login', actor='anonymous', role='admin', outcome='success', action='auth_disabled_login')
            return {
                'access_token': 'auth-disabled',
                'refresh_token': 'auth-disabled',
                'token_type': 'bearer',
                'role': 'admin',
                'expires_in_seconds': runtime_config.jwt_exp_minutes * 60,
                'refresh_expires_in_seconds': runtime_config.jwt_refresh_exp_minutes * 60,
            }
        user = runtime_config.verify_credentials(payload.username, payload.password)
        if not user:
            _audit(request, 'auth_login', actor=payload.username, role='unknown', outcome='failure', action='login_attempt')
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='بيانات الدخول غير صحيحة')
        role = user['role']
        token_store: RefreshTokenStore = request.app.state.token_store
        response = _issue_token_pair(runtime_config, token_store, subject=payload.username, role=role)
        AUTH_LOGINS.labels(role=role).inc()
        _audit(request, 'auth_login', actor=payload.username, role=role, outcome='success', action='login_success')
        return response

    @app.post('/auth/refresh')
    def refresh_token_endpoint(payload: RefreshRequest, request: Request) -> dict[str, Any]:
        if not runtime_config.auth_enabled:
            return {
                'access_token': 'auth-disabled',
                'refresh_token': 'auth-disabled',
                'token_type': 'bearer',
                'role': 'admin',
                'expires_in_seconds': runtime_config.jwt_exp_minutes * 60,
                'refresh_expires_in_seconds': runtime_config.jwt_refresh_exp_minutes * 60,
            }
        token_payload = _decode_token(runtime_config, payload.refresh_token, expected_type='refresh')
        refresh_jti = str(token_payload.get('jti', ''))
        token_store: RefreshTokenStore = request.app.state.token_store
        if not refresh_jti or not token_store.is_active(refresh_jti):
            _audit(request, 'auth_refresh', actor=str(token_payload.get('sub', 'unknown')), role=str(token_payload.get('role', 'user')), outcome='failure', action='refresh_token_inactive')
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Refresh token منتهي أو تم سحبه')
        role = str(token_payload.get('role', 'user'))
        response = _issue_token_pair(
            runtime_config,
            token_store,
            subject=str(token_payload['sub']),
            role=role,
            previous_refresh_jti=refresh_jti,
        )
        AUTH_REFRESHES.labels(role=role).inc()
        _audit(request, 'auth_refresh', actor=str(token_payload['sub']), role=role, outcome='success', action='refresh_token_rotated', details={'previous_refresh_jti': refresh_jti, 'new_refresh_jti': response['refresh_jti']})
        return response

    @app.post('/auth/logout')
    def logout(payload: RefreshRequest, request: Request) -> dict[str, Any]:
        if not runtime_config.auth_enabled:
            return {'status': 'logged_out'}
        token_payload = _decode_token(runtime_config, payload.refresh_token, expected_type='refresh')
        refresh_jti = str(token_payload.get('jti', ''))
        token_store: RefreshTokenStore = request.app.state.token_store
        revoked = token_store.revoke(refresh_jti, reason='logout') if refresh_jti else False
        AUTH_LOGOUTS.labels(role=str(token_payload.get('role', 'user'))).inc()
        _audit(request, 'auth_logout', actor=str(token_payload.get('sub', 'unknown')), role=str(token_payload.get('role', 'user')), outcome='success' if revoked else 'failure', action='refresh_token_revoke', details={'refresh_jti': refresh_jti})
        return {'status': 'logged_out', 'revoked': revoked}

    @app.get('/auth/me')
    def auth_me(user: TokenPayload = Depends(require_permissions('auth:read'))) -> dict[str, Any]:
        return {
            'username': user['sub'],
            'role': user.get('role', 'user'),
            'token_type': user.get('type', 'access'),
            'permissions': sorted(runtime_config.permissions_for_role(str(user.get('role', 'user')))),
        }

    @app.get('/status')
    def status_endpoint(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return brain.system_status()

    @app.get('/brain/diagnostics')
    def brain_diagnostics(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            diagnostics = brain.diagnostics()
        diagnostics['security'] = runtime_config.security_posture_summary()
        diagnostics['runtime'] = runtime_config.sanitized_runtime_settings()
        diagnostics['recommendations'] = _security_recommendations(runtime_config)
        return diagnostics

    @app.get('/brain/config/posture')
    def brain_config_posture(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        return {
            'security': runtime_config.security_posture_summary(),
            'runtime': runtime_config.sanitized_runtime_settings(),
            'recommendations': _security_recommendations(runtime_config),
        }

    @app.get('/brain/personality')
    def brain_personality(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'personality': brain.get_personality_profile()}

    @app.get('/brain/affect')
    def brain_affect(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'affect': brain.get_affect_state()}

    @app.get('/brain/capabilities')
    def brain_capabilities(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, text_interface = runtime()
        with brain_lock:
            return {
                'memory': brain.memory.stats(),
                'llm': text_interface.llm_status(),
                'features': {
                    'goal_tracking': True,
                    'affective_state': True,
                    'sleep_cycle': True,
                    'creative_reply_layer': True,
                    'self_critic_loop': True,
                    'multimodal_episode_memory': True,
                    'temporal_linking': True,
                    'prompt_anomaly_detection': True,
                    'neural_dashboard_snapshot': True,
                    'next_action_planner': True,
                    'roadmap_planner': True,
                    'user_modeling': True,
                    'theory_of_mind_lite': True,
                    'action_hooks_gateway': True,
                    'action_hooks_crud': True,
                    'request_body_size_guard': True,
                    'runtime_config_posture_endpoint': True,
                    'security_response_headers': True,
                    'security_posture_hardening_score': True,
                    'memory_insights_endpoint': True,
                    'audit_summary_endpoint': True,
                    'dream_engine': True,
                    'knowledge_ingestion_text': True,
                    'knowledge_ingestion_url': True,
                    'arabic_chunking_pipeline': True,
                    'duplicate_fingerprint_guard': True,
                    'knowledge_sources_summary': True,
                    'knowledge_source_details': True,
                    'knowledge_source_delete': True,
                    'knowledge_analytics': True,
                    'knowledge_briefing': True,
                    'knowledge_source_trust_scoring': True,
                    'knowledge_query_verification': True,
                    'knowledge_contradiction_detection': True,
                    'knowledge_confidence_explanations': True,
                    'knowledge_consensus_summary': True,
                    'knowledge_provenance_chain': True,
                    'knowledge_contradiction_clustering': True,
                    'knowledge_automation_registry': True,
                    'knowledge_scheduler_background_loop': bool(runtime_config.knowledge_automation_enabled),
                    'adaptive_source_scoring': True,
                    'topic_momentum_tracking': True,
                    'autonomous_ingest_runs': True,
                    'openapi_tag_catalog': True,
                    'memory_encryption_at_rest': brain.memory.encryption_status().get('enabled', False),
                },
            }

    @app.get('/brain/user-model')
    def brain_user_model(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'user_model': brain.get_user_model(), 'last_tom_inference': brain.last_tom_inference}

    @app.post('/brain/user-model/rebuild')
    def rebuild_brain_user_model(
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            profile = brain.rebuild_user_model()
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='rebuild_user_model')
        return {'user_model': profile}

    @app.post('/brain/theory-of-mind')
    def brain_theory_of_mind(
        payload: TheoryOfMindInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            inference = brain.infer_user_mind(payload.text)
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='theory_of_mind_inference')
        return {'theory_of_mind': inference, 'user_model': brain.get_user_model()}

    @app.get('/brain/action-hooks')
    def brain_action_hooks(
        active_only: bool = Query(default=False),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'hooks': brain.list_action_hooks(active_only=active_only), 'summary': brain.action_hooks_overview()}

    @app.post('/brain/action-hooks')
    def create_brain_action_hook(
        payload: ActionHookCreateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            hook = brain.register_action_hook(
                name=payload.name,
                event=payload.event,
                action_type=payload.action_type,
                target_url=payload.target_url,
                method=payload.method,
                headers=payload.headers,
                payload_template=payload.payload_template,
                keywords=payload.keywords,
                cooldown_seconds=payload.cooldown_seconds,
                active=payload.active,
            )
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='create_action_hook', details={'hook_id': hook['hook_id']})
        return {'hook': hook, 'summary': brain.action_hooks_overview()}

    @app.post('/brain/action-hooks/trigger')
    def trigger_brain_action_hooks(
        payload: ActionHookTriggerInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            summary = brain.trigger_action_hooks(
                event=payload.event,
                text=payload.text,
                decision=payload.decision,
                topics=payload.topics,
                dry_run=payload.dry_run,
                allow_network=payload.allow_network,
            )
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='trigger_action_hooks', details={'event': payload.event, 'matched_hooks': summary['matched_hooks']})
        return {'action_hook_summary': summary, 'hooks': brain.list_action_hooks()}

    @app.get('/brain/action-hooks/events')
    def brain_action_hook_events(
        limit: int = Query(default=10, ge=1, le=50),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'events': brain.recent_action_hook_events(limit=limit), 'summary': brain.action_hooks_overview()}

    @app.get('/brain/action-hooks/{hook_id}')
    def brain_action_hook(
        hook_id: str,
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                return {'hook': brain.get_action_hook(hook_id), 'summary': brain.action_hooks_overview()}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch('/brain/action-hooks/{hook_id}')
    def update_brain_action_hook(
        hook_id: str,
        payload: ActionHookUpdateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                hook = brain.update_action_hook(hook_id, payload.model_dump(exclude_unset=True))
            _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='update_action_hook', details={'hook_id': hook_id})
            return {'hook': hook, 'summary': brain.action_hooks_overview()}
        except (ValueError, RuntimeError) as exc:
            _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='update_action_hook', details={'hook_id': hook_id, 'detail': str(exc)})
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete('/brain/action-hooks/{hook_id}')
    def delete_brain_action_hook(
        hook_id: str,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                deleted_hook = brain.delete_action_hook(hook_id)
                summary = brain.action_hooks_overview()
            _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='delete_action_hook', details={'hook_id': hook_id})
            return {'deleted': True, 'hook': deleted_hook, 'summary': summary}
        except (ValueError, RuntimeError) as exc:
            _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='delete_action_hook', details={'hook_id': hook_id, 'detail': str(exc)})
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get('/brain/roadmap')
    def brain_roadmap(
        limit: int = Query(default=4, ge=1, le=8),
        include_completed: bool = Query(default=False),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return brain.roadmap_snapshot(limit=limit, include_completed=include_completed)

    @app.post('/brain/personality')
    def update_brain_personality(
        payload: PersonalityUpdateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            personality = brain.update_personality_profile(payload.model_dump(exclude_none=True))
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='update_personality')
        return {'personality': personality}

    @app.get('/brain/goals')
    def brain_goals(
        status: str | None = Query(default=None),
        limit: int = Query(default=10, ge=1, le=50),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'goals': brain.list_goals(status=status, limit=limit), 'summary': brain.goals_overview()}

    @app.post('/brain/goals')
    def create_brain_goal(
        payload: GoalCreateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            goal = brain.create_goal(
                payload.title,
                payload.description,
                priority=payload.priority,
                target_keywords=payload.target_keywords,
            )
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='create_goal', details={'goal_id': goal['goal_id']})
        return {'goal': goal, 'summary': brain.goals_overview()}

    @app.patch('/brain/goals/{goal_id}')
    def update_brain_goal(
        goal_id: str,
        payload: GoalUpdateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            goal = brain.update_goal(goal_id, payload.model_dump(exclude_none=True))
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='update_goal', details={'goal_id': goal_id})
        return {'goal': goal, 'summary': brain.goals_overview()}

    @app.get('/brain/context')
    def brain_context(
        limit: int = Query(default=5, ge=1, le=10),
        query: str | None = Query(default=None, max_length=300),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'context': brain.context_snapshot(limit=limit, query=query)}

    @app.post('/brain/self-improve')
    def brain_self_improve(
        payload: SelfImproveInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('system:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            event = brain.run_self_improvement(trigger=payload.trigger)
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='self_improve', details={'trigger': payload.trigger})
        return {'self_improvement': event, 'status': brain.system_status()}

    @app.get('/brain/sleep-report')
    def brain_sleep_report(
        limit: int = Query(default=5, ge=1, le=20),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'reports': brain.recent_sleep_reports(limit=limit), 'last_report': brain.last_sleep_report}

    @app.get('/brain/dashboard')
    def brain_dashboard(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'dashboard': brain.dashboard_snapshot()}

    @app.get('/brain/next-actions')
    def brain_next_actions(
        limit: int = Query(default=5, ge=1, le=10),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return brain.next_actions_snapshot(limit=limit)

    @app.get('/brain/anomalies')
    def brain_anomalies(
        limit: int = Query(default=10, ge=1, le=50),
        _: TokenPayload = Depends(require_permissions('system:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'anomalies': brain.recent_anomalies(limit=limit), 'summary': brain.anomaly_overview()}

    @app.post('/wake')
    def wake(request: Request, user: TokenPayload = Depends(require_permissions('system:write'))) -> dict[str, str]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            brain.wake_up()
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='wake')
        return {'state': brain.state}

    @app.post('/sleep')
    def sleep(request: Request, user: TokenPayload = Depends(require_permissions('system:write'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            sleep_report = brain.sleep()
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='sleep')
        return {'state': brain.state, 'sleep_report': sleep_report}

    @app.post('/brain/save-state')
    def save_brain_state(request: Request, user: TokenPayload = Depends(require_permissions('system:write'))) -> dict[str, str]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            path = brain.save_state()
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='save_state', details={'path': str(path)})
        return {'status': 'saved', 'path': str(path)}

    @app.post('/brain/load-state')
    def load_brain_state(request: Request, user: TokenPayload = Depends(require_permissions('system:write'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            loaded = brain.load_state(force_awake=True)
            response = {
                'loaded': loaded,
                'state': brain.state,
                'cycles_completed': brain.cycles_completed,
            }
        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='load_state', details={'loaded': loaded})
        return response

    @app.post('/memory/save')
    def save_memory(request: Request, user: TokenPayload = Depends(require_permissions('memory:write'))) -> dict[str, str]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            brain.memory.save()
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='memory_save')
        return {'status': 'saved'}

    @app.post('/memory/reload')
    def reload_memory(request: Request, user: TokenPayload = Depends(require_permissions('memory:write'))) -> dict[str, str]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            brain.memory.load()
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='memory_reload')
        return {'status': 'reloaded'}

    @app.get('/memory/latest')
    def latest_memory(_: TokenPayload = Depends(require_permissions('memory:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            latest = brain.memory.latest_short_term()
            if latest is None:
                return {'memory': None}
            return {
                'memory': {
                    'key': latest.key,
                    'data': latest.data,
                    'importance': latest.importance,
                    'timestamp': latest.timestamp,
                    'source': latest.source,
                }
            }

    @app.get('/memory/recent')
    def recent_memory(
        limit: int = Query(default=5, ge=1, le=20),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            memories = brain.memory.recent_memories(limit=limit)
            return {
                'memories': [
                    {
                        'key': item.key,
                        'data': item.data,
                        'importance': item.importance,
                        'timestamp': item.timestamp,
                        'source': item.source,
                    }
                    for item in memories
                ]
            }

    @app.get('/memory/search')
    def search_memory(
        query: str = Query(..., min_length=1, max_length=300),
        limit: int = Query(default=5, ge=1, le=20),
        min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
        source: str | None = Query(default=None),
        strategy: str = Query(default='hybrid', pattern='^(lexical|semantic|hybrid)$'),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            matches = brain.memory.search_memories(query, limit=limit, min_importance=min_importance, source=source, strategy=strategy)
            return {
                'query': query,
                'strategy': strategy,
                'count': len(matches),
                'matches': [
                    {
                        'key': item.key,
                        'data': item.data,
                        'importance': item.importance,
                        'timestamp': item.timestamp,
                        'source': item.source,
                    }
                    for item in matches
                ],
            }

    @app.post('/memory/episodes')
    def create_episode_memory(
        payload: EpisodeInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            episode = brain.record_episode(
                text=payload.text,
                image_refs=payload.image_refs,
                audio_refs=payload.audio_refs,
                tags=payload.tags,
                importance=payload.importance,
                related_memory_keys=payload.related_memory_keys,
                metadata={'created_by': str(user['sub']), 'entrypoint': 'api'},
            )
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='episode_create', details={'memory_key': episode['key']})
        return {'episode': episode, 'status': 'stored'}

    @app.get('/memory/episodes/search')
    def search_episode_memory(
        query: str = Query(..., min_length=1, max_length=300),
        limit: int = Query(default=5, ge=1, le=20),
        min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            matches = brain.search_episodes(query, limit=limit, min_importance=min_importance)
            return {
                'query': query,
                'count': len(matches),
                'matches': matches,
            }

    @app.post('/memory/consolidate')
    def consolidate_memory(
        payload: ConsolidationInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            promoted = brain.memory.consolidate_recent_memories(min_importance=payload.min_importance)
            response = {
                'promoted': promoted,
                'memory_stats': brain.memory.stats(),
            }
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='memory_consolidate', details={'promoted': promoted})
        return response

    @app.post('/knowledge/ingest/text')
    def knowledge_ingest_text(
        payload: KnowledgeIngestTextInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                knowledge = knowledge_ingestion.ingest_text_into_memory(
                    brain,
                    text=payload.text,
                    source_name=payload.source_name,
                    source_url=payload.source_url,
                    tags=payload.tags,
                    chunk_size_chars=payload.chunk_size_chars,
                    overlap_chars=payload.overlap_chars,
                    min_chunk_chars=payload.min_chunk_chars,
                    importance=payload.importance,
                )
                response = {
                    'knowledge': knowledge,
                    'memory_stats': brain.memory.stats(),
                }
            _audit(
                request,
                'memory_action',
                actor=str(user['sub']),
                role=str(user['role']),
                outcome='success',
                action='knowledge_ingest_text',
                details={'ingest_id': knowledge.get('ingest_id'), 'chunk_count': knowledge.get('chunk_count')},
            )
            return response
        except (ValueError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning('Knowledge text ingest failed: %s', exc)
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_ingest_text', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/knowledge/ingest/url')
    async def knowledge_ingest_url(
        payload: KnowledgeIngestUrlInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            fetched = await knowledge_ingestion.fetch_url_text(payload.target_url)
            with brain_lock:
                resolved_name = payload.source_name or fetched.get('title') or payload.target_url
                knowledge = knowledge_ingestion.ingest_text_into_memory(
                    brain,
                    text=fetched['text'],
                    source_name=resolved_name,
                    source_url=fetched['final_url'],
                    tags=payload.tags,
                    chunk_size_chars=payload.chunk_size_chars,
                    overlap_chars=payload.overlap_chars,
                    min_chunk_chars=payload.min_chunk_chars,
                    importance=payload.importance,
                    metadata={
                        'fetched_from_url': fetched['final_url'],
                        'content_type': fetched['content_type'],
                        'status_code': fetched['status_code'],
                        'page_title': fetched.get('title'),
                        'source_type': 'url',
                    },
                )
                knowledge['fetch'] = {
                    'final_url': fetched['final_url'],
                    'content_type': fetched['content_type'],
                    'status_code': fetched['status_code'],
                    'title': fetched.get('title'),
                }
                response = {
                    'knowledge': knowledge,
                    'memory_stats': brain.memory.stats(),
                }
            _audit(
                request,
                'memory_action',
                actor=str(user['sub']),
                role=str(user['role']),
                outcome='success',
                action='knowledge_ingest_url',
                details={'ingest_id': knowledge.get('ingest_id'), 'source_url': knowledge.get('source_url')},
            )
            return response
        except (ValueError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning('Knowledge URL ingest failed: %s', exc)
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_ingest_url', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get('/knowledge/sources')
    def knowledge_sources(
        limit: int = Query(default=10, ge=1, le=50),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return knowledge_ingestion.knowledge_sources_summary(brain, limit=limit)

    @app.get('/knowledge/sources/{ingest_id}')
    def knowledge_source_details(
        ingest_id: str,
        chunk_limit: int = Query(default=20, ge=1, le=200),
        include_chunks: bool = Query(default=True),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                return knowledge_ingestion.get_knowledge_source_details(
                    brain,
                    ingest_id=ingest_id,
                    chunk_limit=chunk_limit,
                    include_chunks=include_chunks,
                )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get('/knowledge/analytics')
    def knowledge_analytics_endpoint(
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return knowledge_ingestion.knowledge_analytics(brain)

    @app.get('/knowledge/briefing')
    def knowledge_briefing_endpoint(
        query: str | None = Query(default=None, min_length=1, max_length=300),
        source_limit: int = Query(default=4, ge=1, le=10),
        highlight_limit: int = Query(default=6, ge=1, le=20),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return knowledge_ingestion.knowledge_briefing(
                brain,
                query=query,
                source_limit=source_limit,
                highlight_limit=highlight_limit,
            )

    @app.get('/knowledge/verify')
    def knowledge_verify_endpoint(
        query: str = Query(..., min_length=3, max_length=300),
        source_limit: int = Query(default=5, ge=1, le=10),
        evidence_limit: int = Query(default=8, ge=1, le=20),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                return knowledge_ingestion.knowledge_verify(
                    brain,
                    query=query,
                    source_limit=source_limit,
                    evidence_limit=evidence_limit,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete('/knowledge/sources/{ingest_id}')
    def delete_knowledge_source(
        ingest_id: str,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:delete')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                payload = knowledge_ingestion.delete_knowledge_source(brain, ingest_id=ingest_id)
                payload['memory_stats'] = brain.memory.stats()
            _audit(
                request,
                'memory_action',
                actor=str(user['sub']),
                role=str(user['role']),
                outcome='success' if payload.get('deleted') else 'failure',
                action='knowledge_source_delete',
                details={
                    'ingest_id': ingest_id,
                    'deleted_chunk_count': payload.get('deleted_chunk_count', 0),
                },
            )
            return payload
        except ValueError as exc:
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_source_delete', details={'ingest_id': ingest_id, 'detail': str(exc)})
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get('/knowledge/search')
    def knowledge_search(
        query: str = Query(..., min_length=1, max_length=300),
        limit: int = Query(default=5, ge=1, le=20),
        min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return knowledge_ingestion.search_ingested_chunks(
                brain,
                query=query,
                limit=limit,
                min_importance=min_importance,
            )

    @app.get('/knowledge/automation/status')
    def knowledge_automation_status(
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        return service.status()

    @app.get('/knowledge/automation/sources')
    def knowledge_automation_sources(
        _: TokenPayload = Depends(require_permissions('memory:read')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        return service.list_sources()

    @app.post('/knowledge/automation/sources')
    def knowledge_automation_register_source(
        payload: KnowledgeAutomationSourceInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        try:
            source = service.register_source(payload.model_dump())
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='knowledge_automation_register', details={'source_id': source['source_id'], 'feed_url': source['feed_url']})
            return {'source': source, 'status': service.status()}
        except ValueError as exc:
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_automation_register', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch('/knowledge/automation/sources/{source_id}')
    def knowledge_automation_update_source(
        source_id: str,
        payload: KnowledgeAutomationSourceUpdateInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        try:
            source = service.update_source(source_id, payload.model_dump(exclude_none=True))
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='knowledge_automation_update', details={'source_id': source_id})
            return {'source': source, 'status': service.status()}
        except ValueError as exc:
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_automation_update', details={'source_id': source_id, 'detail': str(exc)})
            raise HTTPException(status_code=404 if 'غير موجود' in str(exc) else 400, detail=str(exc)) from exc

    @app.delete('/knowledge/automation/sources/{source_id}')
    def knowledge_automation_delete_source(
        source_id: str,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:delete')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        try:
            source = service.delete_source(source_id)
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='knowledge_automation_delete', details={'source_id': source_id})
            return {'deleted': True, 'source': source, 'status': service.status()}
        except ValueError as exc:
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_automation_delete', details={'source_id': source_id, 'detail': str(exc)})
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post('/knowledge/automation/sources/{source_id}/run')
    async def knowledge_automation_run_source(
        source_id: str,
        payload: KnowledgeAutomationRunInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        brain, brain_lock, _ = runtime()
        try:
            summary = await service.run_source(source_id, brain, brain_lock, force=payload.force)
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='knowledge_automation_run_source', details={'source_id': source_id, 'status': summary.get('status'), 'ingested': summary.get('ingested', 0)})
            return {'run': summary, 'status': service.status(), 'memory_stats': brain.memory.stats()}
        except ValueError as exc:
            _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='knowledge_automation_run_source', details={'source_id': source_id, 'detail': str(exc)})
            raise HTTPException(status_code=400 if 'غير مفعل' in str(exc) else 404, detail=str(exc)) from exc

    @app.post('/knowledge/automation/run')
    async def knowledge_automation_run_all(
        payload: KnowledgeAutomationRunInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('memory:write')),
    ) -> dict[str, Any]:
        service: KnowledgeAutomationService = app.state.knowledge_automation
        brain, brain_lock, _ = runtime()
        summary = await service.run_all(brain, brain_lock, force=payload.force)
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='knowledge_automation_run_all', details={'run_count': summary.get('run_count', 0)})
        return {'automation': summary, 'status': service.status(), 'memory_stats': brain.memory.stats()}

    @app.post('/cycle')
    def cycle(
        payload: Annotated[CycleInput, Body(openapi_examples=CYCLE_EXAMPLES)],
        request: Request,
        user: TokenPayload = Depends(require_permissions('cycle:run', 'memory:write')),
    ) -> dict[str, Any]:
        visual = np.asarray(payload.visual_input, dtype=np.float64)
        audio = np.asarray(payload.audio_input, dtype=np.float64)
        position = np.asarray(payload.position, dtype=np.float64)
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                result = brain.live_cycle(visual, audio, position, importance=payload.importance)
                CYCLE_REQUESTS.inc()
                response = {
                    'decision': result.decision,
                    'decision_index': result.decision_index,
                    'confidence': result.confidence,
                    'top_probabilities': result.top_probabilities,
                    'memory_key': brain.last_memory_key,
                    'status': brain.system_status(),
                }
            _audit(request, 'brain_cycle', actor=str(user['sub']), role=str(user['role']), outcome='success', action='cycle', details={'memory_key': brain.last_memory_key})
            return response
        except (ValueError, RuntimeError) as exc:
            logger.warning('Cycle request failed: %s', exc)
            _audit(request, 'brain_cycle', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='cycle', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/chat')
    def chat(
        payload: Annotated[ChatInput, Body(openapi_examples=CHAT_EXAMPLES)],
        request: Request,
        user: TokenPayload = Depends(require_permissions('chat:use', 'memory:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, text_interface = runtime()
        try:
            with brain_lock:
                result = text_interface.process_text(
                    payload.text,
                    importance=payload.importance,
                    image_refs=payload.image_refs,
                    audio_refs=payload.audio_refs,
                    tags=payload.tags,
                )
                CHAT_REQUESTS.inc()
                response = {
                    'reply': result.reply,
                    'decision': result.decision,
                    'confidence': result.confidence,
                    'top_probabilities': result.top_probabilities,
                    'memory_key': result.memory_key,
                    'related_memories': result.related_memories,
                    'inferred_importance': result.inferred_importance,
                    'context_summary': result.context_summary,
                    'personality': result.personality,
                    'affect_state': result.affect_state,
                    'goals_overview': result.goals_overview,
                    'user_model': result.user_model,
                    'theory_of_mind': result.theory_of_mind,
                    'action_hook_summary': result.action_hook_summary,
                    'reply_mode': result.reply_mode,
                    'reply_metadata': result.reply_metadata,
                    'self_critique': result.self_critique,
                    'episode_memory_key': result.episode_memory_key,
                    'episode_summary': result.episode_summary,
                    'anomaly_report': result.anomaly_report,
                    'deterministic_reply': result.deterministic_reply,
                    'status': brain.system_status(),
                }
            _audit(request, 'chat_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='chat', details={'memory_key': result.memory_key})
            return response
        except (ValueError, RuntimeError) as exc:
            logger.warning('Chat request failed: %s', exc)
            _audit(request, 'chat_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='chat', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/feedback')
    def feedback(
        payload: FeedbackInput,
        request: Request,
        user: TokenPayload = Depends(require_permissions('feedback:write')),
    ) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        try:
            with brain_lock:
                result = brain.apply_feedback(
                    payload.reward,
                    memory_key=payload.memory_key,
                    feedback_text=payload.feedback_text,
                    source='api_feedback',
                )
                FEEDBACK_REQUESTS.inc()
                response = {
                    'feedback': result,
                    'status': brain.system_status(),
                }
            _audit(request, 'feedback_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='feedback_apply', details={'memory_key': result['memory_key'], 'reward': result['reward']})
            return response
        except (ValueError, RuntimeError) as exc:
            logger.warning('Feedback request failed: %s', exc)
            _audit(request, 'feedback_action', actor=str(user['sub']), role=str(user['role']), outcome='failure', action='feedback_apply', details={'detail': str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get('/memory/insights')
    def memory_insights(request: Request, user: TokenPayload = Depends(require_permissions('memory:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            payload = {
                'insights': brain.memory.insights(),
                'stats': brain.memory.stats(),
            }
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='memory_insights')
        return payload

    @app.get('/memory/export')
    def export_memory(request: Request, user: TokenPayload = Depends(require_permissions('memory:export'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            payload = {'memories': json.loads(brain.memory.export_memories_json())}
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='memory_export')
        return payload

    @app.get('/memory/{key}')
    def memory(key: str, _: TokenPayload = Depends(require_permissions('memory:read'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            return {'key': key, 'value': brain.memory.recall(key)}

    @app.delete('/memory/{key}')
    def delete_memory(key: str, request: Request, user: TokenPayload = Depends(require_permissions('memory:delete'))) -> dict[str, Any]:
        brain, brain_lock, _ = runtime()
        with brain_lock:
            deleted = brain.memory.delete_memory(key)
            payload = {'key': key, 'deleted': deleted, 'memory_stats': brain.memory.stats()}
        _audit(request, 'memory_action', actor=str(user['sub']), role=str(user['role']), outcome='success' if deleted else 'failure', action='memory_delete', details={'key': key})
        return payload

    @app.get('/audit/summary')
    def audit_summary(
        limit: int = Query(default=100, ge=1, le=500),
        event_type: str | None = Query(default=None),
        outcome: str | None = Query(default=None),
        actor: str | None = Query(default=None),
        role: str | None = Query(default=None),
        _: TokenPayload = Depends(require_permissions('audit:read')),
    ) -> dict[str, Any]:
        audit_logger: AuditLogger = app.state.audit_logger
        return {'summary': audit_logger.summary(limit=limit, event_type=event_type, outcome=outcome, actor=actor, role=role)}

    @app.get('/audit/recent')
    def recent_audit_events(
        limit: int = Query(default=20, ge=1, le=200),
        _: TokenPayload = Depends(require_permissions('audit:read')),
    ) -> dict[str, Any]:
        audit_logger: AuditLogger = app.state.audit_logger
        events = audit_logger.recent(limit=limit)
        return {
            'events': events,
            'count': len(events),
        }

    @app.websocket('/ws/chat')
    async def websocket_chat(websocket: WebSocket) -> None:
        config: AppConfig = app.state.config
        request_id = _sanitize_request_id(websocket.headers.get(REQUEST_ID_HEADER) or uuid4().hex)
        actor = 'anonymous'
        role = 'anonymous'
        if config.auth_enabled:
            token = websocket.query_params.get('token')
            if not token:
                await websocket.accept()
                await websocket.send_json({'type': 'error', 'detail': 'مطلوب token في query string'})
                await websocket.close(code=1008)
                return
            try:
                user = _decode_token(config, token, expected_type='access')
                actor = str(user.get('sub', 'unknown'))
                role = str(user.get('role', 'user')).lower()
                if not config.role_has_permissions(role, {'ws:use', 'memory:write'}):
                    await websocket.accept()
                    await websocket.send_json({'type': 'error', 'detail': 'ليس لديك صلاحية لاستخدام WebSocket chat'})
                    await websocket.close(code=1008)
                    return
            except HTTPException as exc:
                await websocket.accept()
                await websocket.send_json({'type': 'error', 'detail': exc.detail})
                await websocket.close(code=1008)
                return
        await websocket.accept()
        manager: WebSocketConnectionManager = app.state.websocket_manager
        connection_id = await manager.register(websocket)
        brain, brain_lock, text_interface = runtime()
        await websocket.send_json({'type': 'ready', 'message': 'WebSocket chat channel is ready', 'request_id': request_id})
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                except Exception:
                    await websocket.send_json({'type': 'error', 'detail': 'صيغة الرسالة غير صالحة', 'request_id': request_id})
                    continue
                try:
                    payload = WebSocketChatInput.model_validate(message)
                except Exception as exc:
                    await websocket.send_json({'type': 'error', 'detail': str(exc), 'request_id': request_id})
                    continue
                try:
                    with brain_lock:
                        result = text_interface.process_text(payload.text, importance=payload.importance)
                        status_payload = brain.system_status()
                    WEBSOCKET_MESSAGES.inc()
                    chunks = [result.reply[i:i + 40] for i in range(0, len(result.reply), 40)] or ['']
                    for index, chunk in enumerate(chunks, start=1):
                        await websocket.send_json({'type': 'chunk', 'index': index, 'content': chunk, 'request_id': request_id})
                    await websocket.send_json(
                        {
                            'type': 'final',
                            'reply': result.reply,
                            'decision': result.decision,
                            'confidence': result.confidence,
                            'top_probabilities': result.top_probabilities,
                            'memory_key': result.memory_key,
                            'related_memories': result.related_memories,
                            'inferred_importance': result.inferred_importance,
                            'context_summary': result.context_summary,
                            'personality': result.personality,
                            'affect_state': result.affect_state,
                            'goals_overview': result.goals_overview,
                            'user_model': result.user_model,
                            'theory_of_mind': result.theory_of_mind,
                            'action_hook_summary': result.action_hook_summary,
                            'reply_mode': result.reply_mode,
                            'reply_metadata': result.reply_metadata,
                            'self_critique': result.self_critique,
                            'episode_memory_key': result.episode_memory_key,
                            'episode_summary': result.episode_summary,
                            'anomaly_report': result.anomaly_report,
                            'deterministic_reply': result.deterministic_reply,
                            'status': status_payload,
                            'request_id': request_id,
                        }
                    )
                except RuntimeError as exc:
                    await websocket.send_json({'type': 'error', 'detail': str(exc), 'request_id': request_id})
        finally:
            await manager.unregister(connection_id)
            logger.info('WebSocket disconnected actor=%s role=%s request_id=%s', actor, role, request_id)

    return app


app = create_app()
