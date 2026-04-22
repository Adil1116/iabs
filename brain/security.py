from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover - optional dependency at runtime
    Fernet = None
    InvalidToken = Exception


class MemoryCipherUnavailableError(RuntimeError):
    pass


@dataclass
class MemoryCipher:
    raw_secret: str

    def __post_init__(self) -> None:
        if not str(self.raw_secret).strip():
            raise ValueError('Memory encryption key cannot be empty')
        if Fernet is None:
            raise MemoryCipherUnavailableError(
                'Memory encryption requested but cryptography package is not installed'
            )
        digest = hashlib.sha256(self.raw_secret.encode('utf-8')).digest()
        self.fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt_json(self, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        return self.fernet.encrypt(payload).decode('utf-8')

    def decrypt_json(self, token: str) -> Any:
        try:
            payload = self.fernet.decrypt(token.encode('utf-8'))
        except InvalidToken as exc:  # pragma: no cover - depends on runtime secrets
            raise ValueError('تعذّر فك تشفير الذاكرة: مفتاح غير صحيح أو البيانات تالفة') from exc
        return json.loads(payload.decode('utf-8'))

    def status(self) -> dict[str, Any]:
        return {
            'enabled': True,
            'algorithm': 'fernet-sha256-derived-key',
            'library_available': Fernet is not None,
        }
