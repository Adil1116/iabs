from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, Optional
import json
import re
import sqlite3
import time

import numpy as np

from brain.core.schemas import MemoryRecord
from brain.security import MemoryCipher, MemoryCipherUnavailableError

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency at runtime
    psycopg = None

try:
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional dependency at runtime
    ConnectionPool = None


class Hippocampus:
    """يحاكي تخزين واسترجاع الذكريات قصيرة وطويلة الأمد مع بحث هجين أقرب للدلالي."""

    SQLITE_SUFFIXES = {'.sqlite', '.sqlite3', '.db'}
    _ARABIC_TRANSLATION = str.maketrans(
        {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه',
            'ـ': '',
        }
    )
    _SYNONYM_GROUPS = [
        {'جوع', 'جوعان', 'جائع', 'اكل', 'أكل', 'طعام', 'وجبه', 'وجبة', 'hungry', 'food', 'eat'},
        {'ذاكره', 'ذاكرة', 'ذكرى', 'memory', 'memories', 'recall'},
        {'سياق', 'context', 'contextual'},
        {'هدف', 'اهداف', 'أهداف', 'مهمه', 'مهمة', 'goal', 'task', 'plan'},
        {'تعلم', 'تعليم', 'تحسين', 'feedBack', 'feedback', 'learning', 'selfimprovement', 'self-improvement'},
        {'نوم', 'نام', 'sleep', 'dream', 'rest'},
    ]

    def __init__(
        self,
        capacity: int = 1000,
        storage_path: Optional[str | Path] = None,
        storage_dsn: str | None = None,
        autosave: bool = True,
        autoload: bool = True,
        postgres_pool_min_size: int = 1,
        postgres_pool_max_size: int = 5,
        encryption_key: str | None = None,
    ):
        self.capacity = max(1, int(capacity))
        self.short_term: Deque[MemoryRecord] = deque(maxlen=self.capacity)
        self.long_term: Dict[str, MemoryRecord] = {}
        self.storage_path = Path(storage_path).expanduser().resolve() if storage_path else None
        self.storage_dsn = storage_dsn.strip() if isinstance(storage_dsn, str) and storage_dsn.strip() else None
        self.autosave = autosave
        self.postgres_pool_min_size = max(1, int(postgres_pool_min_size))
        self.postgres_pool_max_size = max(self.postgres_pool_min_size, int(postgres_pool_max_size))
        self.encryption_key = (encryption_key or '').strip() or None
        try:
            self.cipher = MemoryCipher(self.encryption_key) if self.encryption_key else None
        except MemoryCipherUnavailableError as exc:
            raise RuntimeError(str(exc)) from exc
        self.storage_backend = self._detect_storage_backend()
        self._postgres_pool: ConnectionPool | None = None
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage_backend == 'sqlite':
            self._init_sqlite()
        elif self.storage_backend == 'postgresql':
            self._init_postgres()
        if autoload and self._storage_exists():
            self.load()

    def _detect_storage_backend(self) -> str:
        if self.storage_dsn and self.storage_dsn.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        if self.storage_path and self.storage_path.suffix.lower() in self.SQLITE_SUFFIXES:
            return 'sqlite'
        return 'json'

    def _storage_exists(self) -> bool:
        if self.storage_backend == 'postgresql':
            return True
        return bool(self.storage_path and self.storage_path.exists())

    def _configure_sqlite_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=5000')

    @contextmanager
    def _connect_sqlite(self) -> Iterator[sqlite3.Connection]:
        if not self.storage_path:
            raise RuntimeError('SQLite storage path is not configured')
        conn = sqlite3.connect(self.storage_path, timeout=5)
        self._configure_sqlite_connection(conn)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_postgres_pool(self) -> None:
        if self.storage_backend != 'postgresql' or not self.storage_dsn:
            return
        if self._postgres_pool is None and ConnectionPool is not None:
            self._postgres_pool = ConnectionPool(
                conninfo=self.storage_dsn,
                min_size=self.postgres_pool_min_size,
                max_size=self.postgres_pool_max_size,
                kwargs={'autocommit': False},
                open=True,
            )

    @contextmanager
    def _connect_postgres(self) -> Iterator[Any]:
        if not self.storage_dsn:
            raise RuntimeError('PostgreSQL DSN is not configured')
        if psycopg is None:
            raise RuntimeError('psycopg is required for PostgreSQL backend but is not installed')
        self._ensure_postgres_pool()
        if self._postgres_pool is not None:
            with self._postgres_pool.connection() as conn:
                yield conn
            return
        with psycopg.connect(self.storage_dsn) as conn:
            yield conn

    def _init_sqlite(self) -> None:
        with self._connect_sqlite() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS memory_records (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK(memory_type IN ('short_term', 'long_term'))
                )
                '''
            )
            conn.commit()

    def _init_postgres(self) -> None:
        with self._connect_postgres() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS memory_records (
                        key TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        importance DOUBLE PRECISION NOT NULL,
                        timestamp DOUBLE PRECISION NOT NULL,
                        source TEXT NOT NULL,
                        memory_type TEXT NOT NULL CHECK(memory_type IN ('short_term', 'long_term'))
                    )
                    '''
                )
                cur.execute('CREATE INDEX IF NOT EXISTS idx_memory_records_timestamp ON memory_records (timestamp DESC)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_memory_records_source ON memory_records (source)')
            conn.commit()

    def _sanitize_for_json(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self._sanitize_for_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(item) for item in value]
        return value

    def _serialize_record(self, record: MemoryRecord) -> Dict[str, Any]:
        payload = asdict(record)
        payload['data'] = self._sanitize_for_json(payload['data'])
        return payload

    def _encode_storage_payload(self, value: Any) -> str:
        sanitized = self._sanitize_for_json(value)
        if self.cipher is None:
            return json.dumps(sanitized, ensure_ascii=False)
        encrypted = self.cipher.encrypt_json(sanitized)
        return json.dumps(
            {
                '__encrypted__': True,
                'algorithm': 'fernet-sha256-derived-key',
                'ciphertext': encrypted,
            },
            ensure_ascii=False,
        )

    def _decode_storage_payload(self, raw_value: Any) -> Any:
        if isinstance(raw_value, (dict, list)):
            raw_value = json.dumps(raw_value, ensure_ascii=False)
        if raw_value is None:
            return None
        try:
            payload = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return raw_value
        if isinstance(payload, dict) and payload.get('__encrypted__'):
            ciphertext = str(payload.get('ciphertext', '')).strip()
            if not ciphertext:
                return payload
            if self.cipher is None:
                raise ValueError('الملف يحتوي على بيانات مشفرة لكن لم يتم توفير مفتاح فك التشفير')
            return self.cipher.decrypt_json(ciphertext)
        return payload

    def _serialize_record_for_storage(self, record: MemoryRecord) -> Dict[str, Any]:
        payload = self._serialize_record(record)
        encoded = self._encode_storage_payload(payload.get('data'))
        try:
            payload['data'] = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            payload['data'] = encoded
        return payload

    def _deserialize_record(self, payload: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            key=str(payload['key']),
            data=payload.get('data'),
            importance=float(payload.get('importance', 0.5)),
            timestamp=float(payload.get('timestamp', time.time())),
            source=str(payload.get('source', 'unknown')),
        )

    def _deserialize_record_from_storage(self, payload: Dict[str, Any]) -> MemoryRecord:
        decoded_payload = dict(payload)
        decoded_payload['data'] = self._decode_storage_payload(decoded_payload.get('data'))
        return self._deserialize_record(decoded_payload)

    def _save_if_needed(self) -> None:
        if self.autosave and (self.storage_path or self.storage_dsn):
            self.save()

    def _normalize_text(self, text: str) -> str:
        lowered = str(text).strip().lower().translate(self._ARABIC_TRANSLATION)
        lowered = re.sub(r'[\u064b-\u065f\u0670]', '', lowered)
        lowered = re.sub(r'[^\w\s]+', ' ', lowered, flags=re.UNICODE)
        return ' '.join(lowered.split())

    def _extract_text_fragments(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, np.integer, np.floating)):
            return [str(value)]
        if isinstance(value, dict):
            parts: list[str] = []
            for key, item in value.items():
                parts.append(str(key))
                parts.extend(self._extract_text_fragments(item))
            return parts
        if isinstance(value, (list, tuple, set)):
            parts: list[str] = []
            for item in value:
                parts.extend(self._extract_text_fragments(item))
            return parts
        return [str(value)]

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        return [token for token in re.findall(r'\w+', normalized, flags=re.UNICODE) if token]

    def _expand_query_tokens(self, tokens: list[str]) -> list[str]:
        expanded = list(tokens)
        normalized_groups = [
            {self._normalize_text(token) for token in group if self._normalize_text(token)}
            for group in self._SYNONYM_GROUPS
        ]
        for token in list(tokens):
            normalized_token = self._normalize_text(token)
            for group in normalized_groups:
                if normalized_token in group:
                    for synonym in group:
                        if synonym and synonym not in expanded:
                            expanded.append(synonym)
        return expanded

    def _semantic_vector(self, text: str, dims: int = 128) -> np.ndarray:
        vector = np.zeros(dims, dtype=np.float64)
        tokens = self._tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            vector[hash(token) % dims] += 1.0
            if len(token) > 2:
                for idx in range(len(token) - 2):
                    trigram = token[idx: idx + 3]
                    vector[hash(f'ng:{trigram}') % dims] += 0.35
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _record_text_blob(self, record: MemoryRecord) -> str:
        payload = self._serialize_record(record)
        fragments = self._extract_text_fragments(payload)
        return self._normalize_text(' '.join(fragments))

    def _lexical_score(self, normalized_query: str, tokens: list[str], blob: str) -> float:
        score = 0.0
        if normalized_query in blob:
            score += 6.0
        token_hits = sum(1 for token in tokens if token in blob)
        score += token_hits * 1.75
        return score

    def _semantic_score(self, query_vector: np.ndarray, blob: str) -> float:
        if not np.any(query_vector):
            return 0.0
        record_vector = self._semantic_vector(blob, dims=query_vector.shape[0])
        if not np.any(record_vector):
            return 0.0
        return float(np.dot(query_vector, record_vector))

    def store_memory(self, key: str, data: Any, importance: float = 0.5, source: str = 'system') -> MemoryRecord:
        importance = float(np.clip(importance, 0.0, 1.0))
        record = MemoryRecord(
            key=key,
            data=self._sanitize_for_json(data),
            importance=importance,
            timestamp=time.time(),
            source=source,
        )
        if importance >= 0.8:
            self.long_term[key] = record
            self.short_term = deque((item for item in self.short_term if item.key != key), maxlen=self.capacity)
        else:
            self.long_term.pop(key, None)
            self.short_term = deque((item for item in self.short_term if item.key != key), maxlen=self.capacity)
            self.short_term.append(record)
        self._save_if_needed()
        return record

    def recall(self, key: str) -> Any:
        if key in self.long_term:
            return self.long_term[key].data
        for record in reversed(self.short_term):
            if record.key == key:
                return record.data
        return 'ذكرى غير موجودة في الذاكرة.'

    def latest_short_term(self) -> Optional[MemoryRecord]:
        return self.short_term[-1] if self.short_term else None

    def consolidate_recent_memories(self, min_importance: float = 0.7) -> int:
        promoted = 0
        retained_short_term: list[MemoryRecord] = []
        for record in list(self.short_term):
            if record.importance >= min_importance and record.key not in self.long_term:
                self.long_term[record.key] = record
                promoted += 1
            else:
                retained_short_term.append(record)
        self.short_term = deque(retained_short_term, maxlen=self.capacity)
        if promoted:
            self._save_if_needed()
        return promoted

    def iter_records(self) -> Iterable[MemoryRecord]:
        yield from self.long_term.values()
        yield from self.short_term

    def recent_memories(self, limit: int = 5, include_long_term: bool = True) -> list[MemoryRecord]:
        limit = max(1, int(limit))
        records = list(self.short_term)
        if include_long_term:
            records.extend(self.long_term.values())
        return sorted(records, key=lambda item: item.timestamp, reverse=True)[:limit]

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        source: str | None = None,
        strategy: str = 'hybrid',
    ) -> list[MemoryRecord]:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []
        tokens = [token for token in normalized_query.split(' ') if token]
        expanded_tokens = self._expand_query_tokens(tokens)
        expanded_query = ' '.join(expanded_tokens)
        strategy = (strategy or 'hybrid').strip().lower()
        if strategy not in {'hybrid', 'lexical', 'semantic'}:
            strategy = 'hybrid'
        query_vector = self._semantic_vector(expanded_query)
        candidates: list[tuple[float, MemoryRecord]] = []
        now = time.time()
        for record in self.iter_records():
            if source and record.source != source:
                continue
            if record.importance < min_importance:
                continue
            blob = self._record_text_blob(record)
            lexical_score = self._lexical_score(normalized_query, expanded_tokens, blob)
            semantic_score = self._semantic_score(query_vector, blob)
            score = 0.0
            if strategy == 'lexical':
                score = lexical_score
            elif strategy == 'semantic':
                score = semantic_score * 8.0
            else:
                score = lexical_score + (semantic_score * 4.0)
            score += record.importance * 2.0
            age_hours = max(0.0, (now - record.timestamp) / 3600.0)
            score += 1.0 / (1.0 + age_hours / 24.0)
            if score > 0:
                candidates.append((score, record))
        candidates.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        return [record for _, record in candidates[: max(1, int(limit))]]

    def delete_memory(self, key: str) -> bool:
        existed = key in self.long_term or any(record.key == key for record in self.short_term)
        if not existed:
            return False
        self.long_term.pop(key, None)
        self.short_term = deque((record for record in self.short_term if record.key != key), maxlen=self.capacity)
        self._save_if_needed()
        return True

    def _storage_target(self) -> str | None:
        if self.storage_backend == 'postgresql':
            if not self.storage_dsn:
                return None
            if '@' not in self.storage_dsn or '://' not in self.storage_dsn:
                return self.storage_dsn
            scheme, rest = self.storage_dsn.split('://', maxsplit=1)
            credentials, location = rest.split('@', maxsplit=1)
            if ':' in credentials:
                username, _ = credentials.split(':', maxsplit=1)
                return f'{scheme}://{username}:***@{location}'
            return f'{scheme}://***@{location}'
        return str(self.storage_path) if self.storage_path else None

    def encryption_status(self) -> Dict[str, Any]:
        return {
            'enabled': self.cipher is not None,
            'algorithm': 'fernet-sha256-derived-key' if self.cipher is not None else None,
        }

    def healthcheck(self) -> Dict[str, Any]:
        try:
            if self.storage_backend == 'postgresql':
                with self._connect_postgres() as conn:
                    with conn.cursor() as cur:
                        cur.execute('SELECT 1')
                        value = cur.fetchone()
                return {
                    'status': 'ok',
                    'backend': 'postgresql',
                    'pool_enabled': self._postgres_pool is not None,
                    'target': self._storage_target(),
                    'result': value[0] if value else None,
                    'encryption': self.encryption_status(),
                }
            if self.storage_backend == 'sqlite':
                with self._connect_sqlite() as conn:
                    value = conn.execute('SELECT 1').fetchone()
                return {
                    'status': 'ok',
                    'backend': 'sqlite',
                    'pool_enabled': False,
                    'target': self._storage_target(),
                    'result': value[0] if value else None,
                    'encryption': self.encryption_status(),
                }
            writable = bool(self.storage_path is None or self.storage_path.parent.exists())
            return {
                'status': 'ok' if writable else 'degraded',
                'backend': 'json',
                'pool_enabled': False,
                'target': self._storage_target(),
                'writable_parent': writable,
                'encryption': self.encryption_status(),
            }
        except Exception as exc:
            return {
                'status': 'error',
                'backend': self.storage_backend,
                'pool_enabled': self._postgres_pool is not None,
                'target': self._storage_target(),
                'error': str(exc),
                'encryption': self.encryption_status(),
            }

    def stats(self) -> Dict[str, Any]:
        records = list(self.iter_records())
        total = len(self.short_term) + len(self.long_term)
        latest_timestamp = max((record.timestamp for record in records), default=None)
        average_importance = float(np.mean([record.importance for record in records])) if records else 0.0
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'total_memories': total,
            'capacity': self.capacity,
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'storage_target': self._storage_target(),
            'storage_exists': self._storage_exists(),
            'storage_backend': self.storage_backend,
            'autosave': self.autosave,
            'latest_timestamp': latest_timestamp,
            'average_importance': average_importance,
            'search_capabilities': ['lexical', 'semantic', 'hybrid'],
            'sqlite_tuning': {
                'wal_mode': self.storage_backend == 'sqlite',
                'busy_timeout_ms': 5000 if self.storage_backend == 'sqlite' else None,
                'temp_store_memory': self.storage_backend == 'sqlite',
            },
            'encryption': self.encryption_status(),
            'health': self.healthcheck(),
        }

    def insights(self) -> Dict[str, Any]:
        records_with_type = [
            *({'record': record, 'memory_type': 'long_term'} for record in self.long_term.values()),
            *({'record': record, 'memory_type': 'short_term'} for record in self.short_term),
        ]
        now = time.time()
        source_counter: Counter[str] = Counter()
        memory_type_counter: Counter[str] = Counter()
        importance_bands = {'low': 0, 'medium': 0, 'high': 0}
        freshness = {'last_hour': 0, 'last_day': 0, 'last_week': 0, 'older': 0}
        latest_record: MemoryRecord | None = None

        for item in records_with_type:
            record = item['record']
            memory_type = str(item['memory_type'])
            source_counter.update([str(record.source or 'unknown')])
            memory_type_counter.update([memory_type])
            if latest_record is None or record.timestamp > latest_record.timestamp:
                latest_record = record
            if record.importance >= 0.8:
                importance_bands['high'] += 1
            elif record.importance >= 0.4:
                importance_bands['medium'] += 1
            else:
                importance_bands['low'] += 1
            age_seconds = max(0.0, now - float(record.timestamp))
            if age_seconds <= 3600:
                freshness['last_hour'] += 1
            elif age_seconds <= 86400:
                freshness['last_day'] += 1
            elif age_seconds <= 604800:
                freshness['last_week'] += 1
            else:
                freshness['older'] += 1

        return {
            'total_memories': len(records_with_type),
            'by_memory_type': dict(memory_type_counter),
            'by_source': dict(source_counter),
            'top_sources': [
                {'source': source, 'count': count}
                for source, count in source_counter.most_common(8)
            ],
            'importance_bands': importance_bands,
            'freshness': freshness,
            'latest_memory': (
                {
                    'key': latest_record.key,
                    'source': latest_record.source,
                    'importance': latest_record.importance,
                    'timestamp': latest_record.timestamp,
                }
                if latest_record is not None
                else None
            ),
            'encryption': self.encryption_status(),
            'storage_backend': self.storage_backend,
        }

    def export_memories_json(self) -> str:
        payload = {
            'short_term': [self._serialize_record(item) for item in self.short_term],
            'long_term': {key: self._serialize_record(value) for key, value in self.long_term.items()},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _save_json(self) -> None:
        if not self.storage_path:
            return
        payload = {
            'version': 6,
            'backend': 'json',
            'capacity': self.capacity,
            'saved_at': time.time(),
            'short_term': [self._serialize_record_for_storage(item) for item in self.short_term],
            'long_term': {key: self._serialize_record_for_storage(value) for key, value in self.long_term.items()},
        }
        temp_path = self.storage_path.with_suffix(self.storage_path.suffix + '.tmp')
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_path.replace(self.storage_path)

    def _save_sqlite(self) -> None:
        if not self.storage_path:
            return
        self._init_sqlite()
        with self._connect_sqlite() as conn:
            conn.execute('DELETE FROM memory_records')
            for record in self.short_term:
                conn.execute(
                    'INSERT OR REPLACE INTO memory_records(key, data, importance, timestamp, source, memory_type) VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        record.key,
                        self._encode_storage_payload(record.data),
                        record.importance,
                        record.timestamp,
                        record.source,
                        'short_term',
                    ),
                )
            for record in self.long_term.values():
                conn.execute(
                    'INSERT OR REPLACE INTO memory_records(key, data, importance, timestamp, source, memory_type) VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        record.key,
                        self._encode_storage_payload(record.data),
                        record.importance,
                        record.timestamp,
                        record.source,
                        'long_term',
                    ),
                )
            conn.commit()

    def _save_postgres(self) -> None:
        self._init_postgres()
        with self._connect_postgres() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM memory_records')
                for record in self.short_term:
                    cur.execute(
                        'INSERT INTO memory_records(key, data, importance, timestamp, source, memory_type) VALUES (%s, %s, %s, %s, %s, %s) '
                        'ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, importance = EXCLUDED.importance, timestamp = EXCLUDED.timestamp, source = EXCLUDED.source, memory_type = EXCLUDED.memory_type',
                        (
                            record.key,
                            self._encode_storage_payload(record.data),
                            record.importance,
                            record.timestamp,
                            record.source,
                            'short_term',
                        ),
                    )
                for record in self.long_term.values():
                    cur.execute(
                        'INSERT INTO memory_records(key, data, importance, timestamp, source, memory_type) VALUES (%s, %s, %s, %s, %s, %s) '
                        'ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, importance = EXCLUDED.importance, timestamp = EXCLUDED.timestamp, source = EXCLUDED.source, memory_type = EXCLUDED.memory_type',
                        (
                            record.key,
                            self._encode_storage_payload(record.data),
                            record.importance,
                            record.timestamp,
                            record.source,
                            'long_term',
                        ),
                    )
            conn.commit()

    def save(self) -> None:
        if not self.storage_path and not self.storage_dsn:
            return
        if self.storage_backend == 'postgresql':
            self._save_postgres()
            return
        if self.storage_backend == 'sqlite':
            self._save_sqlite()
            return
        self._save_json()

    def _load_json(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.short_term = deque(maxlen=self.capacity)
            self.long_term = {}
            return
        self.capacity = max(1, int(payload.get('capacity', self.capacity)))
        short_term_payload = payload.get('short_term', [])
        long_term_payload = payload.get('long_term', {})
        self.short_term = deque((self._deserialize_record_from_storage(item) for item in short_term_payload), maxlen=self.capacity)
        self.long_term = {str(key): self._deserialize_record_from_storage(value) for key, value in long_term_payload.items()}

    def _load_sqlite(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        self._init_sqlite()
        short_records: list[MemoryRecord] = []
        long_records: Dict[str, MemoryRecord] = {}
        with self._connect_sqlite() as conn:
            rows = conn.execute(
                'SELECT key, data, importance, timestamp, source, memory_type FROM memory_records ORDER BY timestamp ASC'
            ).fetchall()
        for key, data, importance, timestamp, source, memory_type in rows:
            parsed_data = self._decode_storage_payload(data)
            record = MemoryRecord(
                key=str(key),
                data=parsed_data,
                importance=float(importance),
                timestamp=float(timestamp),
                source=str(source),
            )
            if memory_type == 'long_term':
                long_records[record.key] = record
            else:
                short_records.append(record)
        self.short_term = deque(short_records, maxlen=self.capacity)
        self.long_term = long_records

    def _load_postgres(self) -> None:
        self._init_postgres()
        short_records: list[MemoryRecord] = []
        long_records: Dict[str, MemoryRecord] = {}
        with self._connect_postgres() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT key, data, importance, timestamp, source, memory_type FROM memory_records ORDER BY timestamp ASC')
                rows = cur.fetchall()
        for key, data, importance, timestamp, source, memory_type in rows:
            parsed_data = self._decode_storage_payload(data)
            record = MemoryRecord(
                key=str(key),
                data=parsed_data,
                importance=float(importance),
                timestamp=float(timestamp),
                source=str(source),
            )
            if memory_type == 'long_term':
                long_records[record.key] = record
            else:
                short_records.append(record)
        self.short_term = deque(short_records, maxlen=self.capacity)
        self.long_term = long_records

    def load(self) -> None:
        if self.storage_backend == 'postgresql':
            self._load_postgres()
            return
        if not self.storage_path or not self.storage_path.exists():
            return
        if self.storage_backend == 'sqlite':
            self._load_sqlite()
            return
        self._load_json()

    def close(self) -> None:
        if self._postgres_pool is not None:
            self._postgres_pool.close()
            self._postgres_pool = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup during GC
        try:
            self.close()
        except Exception:
            pass
