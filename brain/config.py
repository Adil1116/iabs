from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / 'data'
DEFAULT_AUDIT_LOG_PATH = DEFAULT_DATA_DIR / 'audit.log'
DEFAULT_REFRESH_TOKEN_STORE_PATH = DEFAULT_DATA_DIR / 'refresh_tokens.json'
PBKDF2_SCHEME = 'pbkdf2_sha256'
DEFAULT_AUTH_PASSWORD = 'change-me-now'
DEFAULT_JWT_SECRET = 'change-this-secret-in-production-please-use-32-chars'
MIN_RECOMMENDED_JWT_SECRET_LENGTH = 32


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _parse_auth_users(raw: str | None) -> dict[str, dict[str, str]]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    users: dict[str, dict[str, str]] = {}
    for username, user_data in payload.items():
        if not isinstance(username, str) or not isinstance(user_data, dict):
            continue
        role = str(user_data.get('role', 'user')).strip().lower() or 'user'
        password = str(user_data.get('password', '')).strip()
        password_hash = str(user_data.get('password_hash', '')).strip()
        if not password and not password_hash:
            continue
        normalized: dict[str, str] = {'role': role}
        if password:
            normalized['password'] = password
        if password_hash:
            normalized['password_hash'] = password_hash
        users[username.strip()] = normalized
    return users


def _default_role_permissions() -> dict[str, set[str]]:
    return {
        'admin': {
            'system:read', 'system:write',
            'memory:read', 'memory:write', 'memory:delete', 'memory:export',
            'chat:use', 'cycle:run', 'ws:use', 'feedback:write',
            'auth:read', 'audit:read',
        },
        'operator': {
            'system:read', 'system:write',
            'memory:read', 'memory:write', 'memory:export',
            'chat:use', 'cycle:run', 'ws:use', 'feedback:write',
            'auth:read',
        },
        'auditor': {
            'system:read', 'memory:read', 'auth:read', 'audit:read',
        },
        'reader': {
            'system:read', 'memory:read', 'chat:use', 'cycle:run', 'ws:use',
        },
    }


def _parse_role_permissions(raw: str | None) -> dict[str, set[str]]:
    permissions = _default_role_permissions()
    if not raw:
        return permissions
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return permissions
    if not isinstance(payload, dict):
        return permissions
    for role, values in payload.items():
        if not isinstance(role, str):
            continue
        if isinstance(values, list):
            permissions[role.strip().lower()] = {str(item).strip() for item in values if str(item).strip()}
    return permissions


def _parse_path_or_none(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _load_env_file(path: str | None) -> None:
    env_path = _strip_or_none(path)
    if not env_path:
        return
    candidate = Path(env_path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        return
    for raw_line in candidate.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', maxsplit=1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def _resolve_memory_dsn() -> str | None:
    for env_name in ('IABS_MEMORY_DSN', 'DATABASE_URL', 'SUPABASE_DB_URL', 'SUPABASE_DATABASE_URL'):
        resolved = _strip_or_none(os.getenv(env_name))
        if resolved:
            return resolved
    return None


def _mask_dsn(dsn: str | None) -> str | None:
    if not dsn:
        return None
    if '@' not in dsn or '://' not in dsn:
        return dsn
    scheme, rest = dsn.split('://', maxsplit=1)
    credentials, location = rest.split('@', maxsplit=1)
    if ':' in credentials:
        username, _ = credentials.split(':', maxsplit=1)
        return f'{scheme}://{username}:***@{location}'
    return f'{scheme}://***@{location}'


def _normalize_environment(value: str | None) -> str:
    candidate = (value or 'development').strip().lower()
    if candidate in {'development', 'staging', 'production'}:
        return candidate
    return 'development'


@dataclass
class AppConfig:
    seed: int = 42
    memory_capacity: int = 1000
    memory_path: Path | None = DEFAULT_DATA_DIR / 'memory_store.sqlite'
    memory_dsn: str | None = None
    state_path: Path = DEFAULT_DATA_DIR / 'brain_state.json'
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH
    refresh_token_store_path: Path = DEFAULT_REFRESH_TOKEN_STORE_PATH
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 5
    host: str = '0.0.0.0'
    port: int = 8000
    log_level: str = 'INFO'
    log_format: str = 'json'
    autoload_state: bool = True
    app_title: str = 'Integrated Artificial Brain API'
    app_version: str = '2.16.0'
    environment: str = 'development'
    auth_enabled: bool = True
    auth_username: str = 'admin'
    auth_password: str = DEFAULT_AUTH_PASSWORD
    auth_default_role: str = 'admin'
    auth_users: dict[str, dict[str, str]] = field(default_factory=dict)
    role_permissions: dict[str, set[str]] = field(default_factory=_default_role_permissions)
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = 'HS256'
    jwt_exp_minutes: int = 60
    jwt_refresh_exp_minutes: int = 60 * 24 * 7
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    rate_limit_backend: str = 'memory'
    redis_url: str = 'redis://redis:6379/0'
    redis_rate_limit_prefix: str = 'iabs:ratelimit'
    metrics_enabled: bool = True
    max_request_body_bytes: int = 1024 * 1024
    memory_search_default_limit: int = 5
    memory_encryption_key: str | None = None
    llm_enabled: bool = False
    llm_api_key: str | None = None
    llm_api_base: str = 'https://api.openai.com/v1'
    llm_model: str | None = None
    llm_timeout_seconds: float = 12.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 350
    knowledge_automation_enabled: bool = True
    knowledge_scheduler_poll_seconds: int = 300
    knowledge_registry_path: Path = DEFAULT_DATA_DIR / 'knowledge_automation.json'

    def __post_init__(self) -> None:
        self.log_level = self.log_level.upper()
        self.log_format = (self.log_format or 'json').strip().lower()
        self.environment = _normalize_environment(self.environment)
        self.auth_default_role = self.auth_default_role.strip().lower() or 'user'
        self.rate_limit_backend = (self.rate_limit_backend or 'memory').strip().lower()
        self.postgres_pool_min_size = max(1, int(self.postgres_pool_min_size))
        self.postgres_pool_max_size = max(self.postgres_pool_min_size, int(self.postgres_pool_max_size))
        self.max_request_body_bytes = max(1024, int(self.max_request_body_bytes))
        self.memory_encryption_key = (self.memory_encryption_key or '').strip() or None
        self.llm_api_key = (self.llm_api_key or '').strip() or None
        self.llm_model = (self.llm_model or '').strip() or None
        self.llm_api_base = (self.llm_api_base or 'https://api.openai.com/v1').strip().rstrip('/')
        self.llm_timeout_seconds = max(2.0, float(self.llm_timeout_seconds))
        self.llm_temperature = float(min(1.2, max(0.0, self.llm_temperature)))
        self.llm_max_tokens = max(64, int(self.llm_max_tokens))
        self.llm_enabled = bool(self.llm_enabled and self.llm_api_key and self.llm_model)
        self.knowledge_scheduler_poll_seconds = max(15, int(self.knowledge_scheduler_poll_seconds))
        self.knowledge_registry_path = Path(self.knowledge_registry_path).expanduser().resolve()
        normalized_permissions: dict[str, set[str]] = {}
        source_permissions = self.role_permissions or _default_role_permissions()
        for role, permissions in source_permissions.items():
            if not isinstance(role, str):
                continue
            cleaned_permissions = {str(item).strip() for item in permissions if str(item).strip()}
            normalized_permissions[role.strip().lower()] = cleaned_permissions
        self.role_permissions = normalized_permissions or _default_role_permissions()
        normalized_users: dict[str, dict[str, str]] = {}
        source_users = self.auth_users or {
            self.auth_username: {
                'password': self.auth_password,
                'role': self.auth_default_role,
            }
        }
        for username, user_data in source_users.items():
            if not isinstance(username, str) or not isinstance(user_data, dict):
                continue
            cleaned_username = username.strip()
            password = str(user_data.get('password', '')).strip()
            password_hash = str(user_data.get('password_hash', '')).strip()
            role = str(user_data.get('role', self.auth_default_role)).strip().lower() or self.auth_default_role
            if cleaned_username and (password or password_hash):
                normalized_user = {'role': role}
                if password:
                    normalized_user['password'] = password
                if password_hash:
                    normalized_user['password_hash'] = password_hash
                normalized_users[cleaned_username] = normalized_user
        if not normalized_users:
            normalized_users[self.auth_username] = {
                'password': self.auth_password,
                'role': self.auth_default_role,
            }
        self.auth_users = normalized_users

    @staticmethod
    def hash_password(password: str, *, salt: str | None = None, iterations: int = 390000) -> str:
        clean_password = password.strip()
        if not clean_password:
            raise ValueError('password cannot be empty')
        resolved_salt = salt or os.urandom(16).hex()
        digest = hashlib.pbkdf2_hmac(
            'sha256',
            clean_password.encode('utf-8'),
            resolved_salt.encode('utf-8'),
            iterations,
        ).hex()
        return f'{PBKDF2_SCHEME}${iterations}${resolved_salt}${digest}'

    @staticmethod
    def verify_password_hash(password: str, encoded_hash: str) -> bool:
        try:
            scheme, raw_iterations, salt, expected = encoded_hash.split('$', maxsplit=3)
        except ValueError:
            return False
        if scheme != PBKDF2_SCHEME:
            return False
        try:
            iterations = int(raw_iterations)
        except ValueError:
            return False
        candidate = AppConfig.hash_password(password, salt=salt, iterations=iterations)
        return hmac.compare_digest(candidate, encoded_hash)

    @classmethod
    def from_env(cls) -> 'AppConfig':
        _load_env_file(os.getenv('IABS_ENV_FILE', str(BASE_DIR / '.env')))
        data_dir = Path(os.getenv('IABS_DATA_DIR', str(DEFAULT_DATA_DIR))).expanduser().resolve()
        memory_dsn = _resolve_memory_dsn()
        raw_memory_path = os.getenv('IABS_MEMORY_PATH')
        if raw_memory_path is None:
            memory_path = None if memory_dsn else (data_dir / 'memory_store.sqlite')
        else:
            memory_path = _parse_path_or_none(raw_memory_path)
            if memory_path is None and not memory_dsn:
                memory_path = data_dir / 'memory_store.sqlite'
        state_path = Path(os.getenv('IABS_STATE_PATH', str(data_dir / 'brain_state.json'))).expanduser().resolve()
        audit_log_path = Path(os.getenv('IABS_AUDIT_LOG_PATH', str(data_dir / 'audit.log'))).expanduser().resolve()
        refresh_token_store_path = Path(
            os.getenv('IABS_REFRESH_TOKEN_STORE_PATH', str(data_dir / 'refresh_tokens.json'))
        ).expanduser().resolve()
        auth_default_role = os.getenv('IABS_AUTH_DEFAULT_ROLE', 'admin')
        auth_users = _parse_auth_users(os.getenv('IABS_AUTH_USERS_JSON'))
        role_permissions = _parse_role_permissions(os.getenv('IABS_ROLE_PERMISSIONS_JSON'))
        return cls(
            seed=int(os.getenv('IABS_SEED', '42')),
            memory_capacity=int(os.getenv('IABS_MEMORY_CAPACITY', '1000')),
            memory_path=memory_path,
            memory_dsn=memory_dsn,
            state_path=state_path,
            audit_log_path=audit_log_path,
            refresh_token_store_path=refresh_token_store_path,
            postgres_pool_min_size=int(os.getenv('IABS_POSTGRES_POOL_MIN_SIZE', '1')),
            postgres_pool_max_size=int(os.getenv('IABS_POSTGRES_POOL_MAX_SIZE', '5')),
            host=os.getenv('IABS_HOST', '0.0.0.0'),
            port=int(os.getenv('IABS_PORT', '8000')),
            log_level=os.getenv('IABS_LOG_LEVEL', 'INFO'),
            log_format=os.getenv('IABS_LOG_FORMAT', 'json'),
            autoload_state=_parse_bool(os.getenv('IABS_AUTOLOAD_STATE'), True),
            app_title=os.getenv('IABS_APP_TITLE', 'Integrated Artificial Brain API'),
            app_version=os.getenv('IABS_APP_VERSION', '2.16.0'),
            environment=os.getenv('IABS_ENVIRONMENT', 'development'),
            auth_enabled=_parse_bool(os.getenv('IABS_AUTH_ENABLED'), True),
            auth_username=os.getenv('IABS_AUTH_USERNAME', 'admin'),
            auth_password=os.getenv('IABS_AUTH_PASSWORD', DEFAULT_AUTH_PASSWORD),
            auth_default_role=auth_default_role,
            auth_users=auth_users,
            role_permissions=role_permissions,
            jwt_secret=os.getenv('IABS_JWT_SECRET', DEFAULT_JWT_SECRET),
            jwt_algorithm=os.getenv('IABS_JWT_ALGORITHM', 'HS256'),
            jwt_exp_minutes=int(os.getenv('IABS_JWT_EXP_MINUTES', '60')),
            jwt_refresh_exp_minutes=int(os.getenv('IABS_JWT_REFRESH_EXP_MINUTES', str(60 * 24 * 7))),
            rate_limit_enabled=_parse_bool(os.getenv('IABS_RATE_LIMIT_ENABLED'), True),
            rate_limit_requests=int(os.getenv('IABS_RATE_LIMIT_REQUESTS', '60')),
            rate_limit_window_seconds=int(os.getenv('IABS_RATE_LIMIT_WINDOW_SECONDS', '60')),
            rate_limit_backend=os.getenv('IABS_RATE_LIMIT_BACKEND', 'memory'),
            redis_url=os.getenv('IABS_REDIS_URL', 'redis://redis:6379/0'),
            redis_rate_limit_prefix=os.getenv('IABS_REDIS_RATE_LIMIT_PREFIX', 'iabs:ratelimit'),
            metrics_enabled=_parse_bool(os.getenv('IABS_METRICS_ENABLED'), True),
            max_request_body_bytes=int(os.getenv('IABS_MAX_REQUEST_BODY_BYTES', str(1024 * 1024))),
            memory_search_default_limit=int(os.getenv('IABS_MEMORY_SEARCH_DEFAULT_LIMIT', '5')),
            memory_encryption_key=os.getenv('IABS_MEMORY_ENCRYPTION_KEY'),
            llm_enabled=_parse_bool(os.getenv('IABS_LLM_ENABLED'), False),
            llm_api_key=os.getenv('IABS_LLM_API_KEY'),
            llm_api_base=os.getenv('IABS_LLM_API_BASE', 'https://api.openai.com/v1'),
            llm_model=os.getenv('IABS_LLM_MODEL'),
            llm_timeout_seconds=float(os.getenv('IABS_LLM_TIMEOUT_SECONDS', '12')),
            llm_temperature=float(os.getenv('IABS_LLM_TEMPERATURE', '0.7')),
            llm_max_tokens=int(os.getenv('IABS_LLM_MAX_TOKENS', '350')),
            knowledge_automation_enabled=_parse_bool(os.getenv('IABS_KNOWLEDGE_AUTOMATION_ENABLED'), True),
            knowledge_scheduler_poll_seconds=int(os.getenv('IABS_KNOWLEDGE_SCHEDULER_POLL_SECONDS', '300')),
            knowledge_registry_path=Path(os.getenv('IABS_KNOWLEDGE_REGISTRY_PATH', str(data_dir / 'knowledge_automation.json'))).expanduser().resolve(),
        )

    def ensure_directories(self) -> None:
        if self.memory_path is not None:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.refresh_token_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_registry_path.parent.mkdir(parents=True, exist_ok=True)

    def get_auth_user(self, username: str) -> dict[str, str] | None:
        return self.auth_users.get(username)

    def verify_credentials(self, username: str, password: str) -> dict[str, str] | None:
        user = self.get_auth_user(username)
        if not user:
            return None
        stored_password = user.get('password')
        stored_hash = user.get('password_hash')
        verified = False
        if stored_hash:
            verified = self.verify_password_hash(password, stored_hash)
        elif stored_password is not None:
            verified = hmac.compare_digest(stored_password, password)
        if not verified:
            return None
        return {
            'username': username,
            'role': user.get('role', self.auth_default_role),
        }

    def permissions_for_role(self, role: str) -> set[str]:
        return set(self.role_permissions.get(role.strip().lower(), set()))

    def role_has_permissions(self, role: str, required_permissions: set[str]) -> bool:
        if not required_permissions:
            return True
        role_permissions = self.permissions_for_role(role)
        return required_permissions.issubset(role_permissions)

    def security_warnings(self) -> list[str]:
        warnings: list[str] = []
        if not self.auth_enabled:
            warnings.append('Authentication is disabled')
        if str(self.auth_username).strip() == 'admin' and self.auth_password == DEFAULT_AUTH_PASSWORD:
            warnings.append('Default admin password is still configured')
        if self.jwt_secret == DEFAULT_JWT_SECRET:
            warnings.append('Default JWT secret is still configured')
        if len(self.jwt_secret.strip()) < MIN_RECOMMENDED_JWT_SECRET_LENGTH:
            warnings.append('JWT secret is shorter than the recommended 32 characters')
        plaintext_users = sorted(
            username
            for username, user in self.auth_users.items()
            if isinstance(user, dict) and user.get('password') and not user.get('password_hash')
        )
        if plaintext_users:
            warnings.append(f'Plaintext auth passwords configured for users: {", ".join(plaintext_users)}')
        if not self.memory_encryption_key:
            warnings.append('Memory encryption at rest is disabled')
        if not self.rate_limit_enabled:
            warnings.append('Rate limiting is disabled')
        if self.environment == 'production' and self.log_format != 'json':
            warnings.append('Production mode should use JSON logging for safer observability')
        if self.environment == 'production' and self.rate_limit_backend != 'redis':
            warnings.append('Production mode is using in-memory rate limiting instead of Redis')
        return warnings

    def security_posture_summary(self) -> dict[str, Any]:
        warnings = self.security_warnings()
        plaintext_users = sorted(
            username
            for username, user in self.auth_users.items()
            if isinstance(user, dict) and user.get('password') and not user.get('password_hash')
        )
        critical_warnings = [
            warning
            for warning in warnings
            if warning in {
                'Authentication is disabled',
                'Default admin password is still configured',
                'Default JWT secret is still configured',
            }
        ]
        hardening_score = 100
        hardening_score -= len(warnings) * 8
        hardening_score -= len(critical_warnings) * 8
        if self.environment == 'production' and warnings:
            hardening_score -= 6
        if not self.memory_encryption_key:
            hardening_score -= 4
        if not self.metrics_enabled:
            hardening_score -= 4
        hardening_score = int(max(0, min(100, hardening_score)))
        return {
            'status': 'hardened' if not warnings else ('warning' if len(warnings) <= 2 else 'needs_attention'),
            'environment': self.environment,
            'warning_count': len(warnings),
            'warnings': warnings,
            'critical_warning_count': len(critical_warnings),
            'critical_warnings': critical_warnings,
            'hardening_score': hardening_score,
            'defaults_replaced': self.auth_password != DEFAULT_AUTH_PASSWORD and self.jwt_secret != DEFAULT_JWT_SECRET,
            'memory_encryption_enabled': bool(self.memory_encryption_key),
            'rate_limit_enabled': self.rate_limit_enabled,
            'rate_limit_backend': self.rate_limit_backend,
            'auth_enabled': self.auth_enabled,
            'auth_user_count': len(self.auth_users),
            'plaintext_user_count': len(plaintext_users),
            'jwt_secret_length': len(self.jwt_secret.strip()),
            'max_request_body_bytes': self.max_request_body_bytes,
            'security_headers_expected': True,
            'observability_ready': bool(self.metrics_enabled and self.log_format == 'json'),
            'production_ready': self.environment != 'production' or not warnings,
        }


    def storage_summary(self) -> dict[str, Any]:
        backend = 'postgresql' if self.memory_dsn else ('sqlite' if self.memory_path and self.memory_path.suffix.lower() in {'.sqlite', '.sqlite3', '.db'} else 'json')
        return {
            'backend': backend,
            'path': str(self.memory_path) if self.memory_path else None,
            'dsn': _mask_dsn(self.memory_dsn),
            'postgres_pool_min_size': self.postgres_pool_min_size,
            'postgres_pool_max_size': self.postgres_pool_max_size,
            'memory_encryption_enabled': bool(self.memory_encryption_key),
            'knowledge_registry_path': str(self.knowledge_registry_path),
            'knowledge_automation_enabled': self.knowledge_automation_enabled,
        }

    def public_auth_summary(self) -> dict[str, Any]:
        return {
            'enabled': self.auth_enabled,
            'users': sorted(self.auth_users.keys()),
            'roles': sorted({user.get('role', 'user') for user in self.auth_users.values()}),
            'access_token_minutes': self.jwt_exp_minutes,
            'refresh_token_minutes': self.jwt_refresh_exp_minutes,
            'rbac_roles': {role: sorted(list(perms)) for role, perms in self.role_permissions.items()},
        }

    def sanitized_runtime_settings(self) -> dict[str, Any]:
        return {
            'app': {
                'title': self.app_title,
                'version': self.app_version,
                'environment': self.environment,
                'host': self.host,
                'port': self.port,
                'autoload_state': self.autoload_state,
            },
            'logging': {
                'level': self.log_level,
                'format': self.log_format,
            },
            'auth': {
                'enabled': self.auth_enabled,
                'default_role': self.auth_default_role,
                'user_count': len(self.auth_users),
                'roles': sorted({user.get('role', 'user') for user in self.auth_users.values()}),
            },
            'rate_limit': {
                'enabled': self.rate_limit_enabled,
                'requests': self.rate_limit_requests,
                'window_seconds': self.rate_limit_window_seconds,
                'backend': self.rate_limit_backend,
                'redis_url_masked': _mask_dsn(self.redis_url),
            },
            'request_guard': {
                'max_request_body_bytes': self.max_request_body_bytes,
            },
            'storage': self.storage_summary(),
            'llm': {
                'enabled': self.llm_enabled,
                'api_base': self.llm_api_base,
                'model': self.llm_model,
                'timeout_seconds': self.llm_timeout_seconds,
                'temperature': self.llm_temperature,
                'max_tokens': self.llm_max_tokens,
            },
            'knowledge_automation': {
                'enabled': self.knowledge_automation_enabled,
                'poll_seconds': self.knowledge_scheduler_poll_seconds,
                'registry_path': str(self.knowledge_registry_path),
            },
            'metrics_enabled': self.metrics_enabled,
        }
