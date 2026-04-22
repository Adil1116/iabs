from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any
import json
import time


@dataclass
class RefreshTokenRecord:
    jti: str
    subject: str
    role: str
    issued_at: float
    expires_at: float
    parent_jti: str | None = None
    revoked: bool = False
    revoked_at: float | None = None
    revoked_reason: str | None = None
    replaced_by: str | None = None


class RefreshTokenStore:
    def __init__(
        self,
        file_path: str | Path | None = None,
        *,
        autosave: bool = True,
        autoload: bool = True,
        max_records: int = 10000,
    ):
        self.file_path = Path(file_path).expanduser().resolve() if file_path else None
        self.autosave = autosave
        self.max_records = max(100, int(max_records))
        self._records: dict[str, RefreshTokenRecord] = {}
        self._lock = Lock()
        if self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if autoload:
            self.load()

    def _serialize(self, record: RefreshTokenRecord) -> dict[str, Any]:
        return asdict(record)

    def _deserialize(self, payload: dict[str, Any]) -> RefreshTokenRecord:
        return RefreshTokenRecord(
            jti=str(payload['jti']),
            subject=str(payload['subject']),
            role=str(payload.get('role', 'user')),
            issued_at=float(payload.get('issued_at', time.time())),
            expires_at=float(payload.get('expires_at', time.time())),
            parent_jti=payload.get('parent_jti'),
            revoked=bool(payload.get('revoked', False)),
            revoked_at=float(payload['revoked_at']) if payload.get('revoked_at') is not None else None,
            revoked_reason=payload.get('revoked_reason'),
            replaced_by=payload.get('replaced_by'),
        )

    def _save_if_needed(self) -> None:
        if self.autosave and self.file_path is not None:
            self.save()

    def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            expired = [jti for jti, record in self._records.items() if record.expires_at < now - 86400]
            for jti in expired:
                self._records.pop(jti, None)
                removed += 1
            if len(self._records) > self.max_records:
                survivors = sorted(self._records.values(), key=lambda item: item.issued_at, reverse=True)[: self.max_records]
                self._records = {item.jti: item for item in survivors}
        if removed:
            self._save_if_needed()
        return removed

    def register(self, jti: str, subject: str, role: str, issued_at: float, expires_at: float, parent_jti: str | None = None) -> RefreshTokenRecord:
        record = RefreshTokenRecord(
            jti=jti,
            subject=subject,
            role=role,
            issued_at=float(issued_at),
            expires_at=float(expires_at),
            parent_jti=parent_jti,
        )
        with self._lock:
            self._records[jti] = record
        self.cleanup_expired()
        self._save_if_needed()
        return record

    def get(self, jti: str) -> RefreshTokenRecord | None:
        with self._lock:
            return self._records.get(jti)

    def is_active(self, jti: str) -> bool:
        record = self.get(jti)
        if record is None:
            return False
        if record.revoked:
            return False
        if record.expires_at <= time.time():
            return False
        return True

    def revoke(self, jti: str, *, reason: str = 'manual', replaced_by: str | None = None) -> bool:
        with self._lock:
            record = self._records.get(jti)
            if record is None:
                return False
            if not record.revoked:
                record.revoked = True
                record.revoked_at = time.time()
                record.revoked_reason = reason
                record.replaced_by = replaced_by
            else:
                if replaced_by and not record.replaced_by:
                    record.replaced_by = replaced_by
        self._save_if_needed()
        return True

    def rotate(
        self,
        *,
        current_jti: str,
        new_jti: str,
        subject: str,
        role: str,
        issued_at: float,
        expires_at: float,
    ) -> RefreshTokenRecord:
        self.revoke(current_jti, reason='rotated', replaced_by=new_jti)
        return self.register(new_jti, subject, role, issued_at, expires_at, parent_jti=current_jti)

    def save(self) -> None:
        if self.file_path is None:
            return
        with self._lock:
            payload = {
                'version': 1,
                'saved_at': time.time(),
                'records': [self._serialize(item) for item in self._records.values()],
            }
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + '.tmp')
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp_path.replace(self.file_path)

    def load(self) -> None:
        if self.file_path is None or not self.file_path.exists():
            return
        try:
            payload = json.loads(self.file_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        records = payload.get('records', [])
        loaded: dict[str, RefreshTokenRecord] = {}
        for item in records:
            if not isinstance(item, dict) or 'jti' not in item or 'subject' not in item:
                continue
            record = self._deserialize(item)
            loaded[record.jti] = record
        with self._lock:
            self._records = loaded
        self.cleanup_expired()

    def stats(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            records = list(self._records.values())
        active = sum(1 for item in records if (not item.revoked and item.expires_at > now))
        revoked = sum(1 for item in records if item.revoked)
        expired = sum(1 for item in records if item.expires_at <= now)
        return {
            'total': len(records),
            'active': active,
            'revoked': revoked,
            'expired': expired,
            'storage_path': str(self.file_path) if self.file_path else None,
        }
