from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any


logger = logging.getLogger('iabs.audit')


class AuditLogger:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).expanduser().resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log(
        self,
        event_type: str,
        *,
        actor: str,
        role: str,
        outcome: str,
        action: str,
        details: dict[str, Any] | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'actor': actor,
            'role': role,
            'outcome': outcome,
            'action': action,
            'details': details or {},
            'request': request_context or {},
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.file_path.open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')
        logger.info('AUDIT %s', line)
        return payload

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        if not self.file_path.exists():
            return []
        lines: deque[str] = deque(maxlen=limit)
        with self._lock:
            with self.file_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    cleaned = line.strip()
                    if cleaned:
                        lines.append(cleaned)
        entries: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _filtered_recent(
        self,
        *,
        limit: int = 200,
        event_type: str | None = None,
        outcome: str | None = None,
        actor: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        events = self.recent(limit=limit)
        normalized_filters = {
            'event_type': str(event_type).strip().lower() if event_type else None,
            'outcome': str(outcome).strip().lower() if outcome else None,
            'actor': str(actor).strip().lower() if actor else None,
            'role': str(role).strip().lower() if role else None,
        }

        def matches(entry: dict[str, Any]) -> bool:
            for key, expected in normalized_filters.items():
                if expected is None:
                    continue
                candidate = str(entry.get(key, '')).strip().lower()
                if candidate != expected:
                    return False
            return True

        return [entry for entry in events if matches(entry)]

    def summary(
        self,
        *,
        limit: int = 200,
        event_type: str | None = None,
        outcome: str | None = None,
        actor: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        events = self._filtered_recent(limit=limit, event_type=event_type, outcome=outcome, actor=actor, role=role)
        event_counter = Counter(str(item.get('event_type', 'unknown')) for item in events)
        outcome_counter = Counter(str(item.get('outcome', 'unknown')) for item in events)
        role_counter = Counter(str(item.get('role', 'unknown')) for item in events)
        actor_counter = Counter(str(item.get('actor', 'unknown')) for item in events)
        timestamps = [str(item.get('timestamp', '')) for item in events if str(item.get('timestamp', '')).strip()]
        return {
            'filters': {
                'limit': max(1, int(limit)),
                'event_type': event_type,
                'outcome': outcome,
                'actor': actor,
                'role': role,
            },
            'evaluated_events': len(events),
            'by_event_type': dict(event_counter),
            'by_outcome': dict(outcome_counter),
            'by_role': dict(role_counter),
            'top_actors': [
                {'actor': actor_name, 'count': count}
                for actor_name, count in actor_counter.most_common(5)
            ],
            'latest_timestamp': max(timestamps) if timestamps else None,
        }
