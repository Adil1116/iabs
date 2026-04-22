from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from fastapi import WebSocket


class WebSocketConnectionManager:
    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket) -> str:
        connection_id = uuid4().hex
        async with self._lock:
            self._connections[connection_id] = websocket
        return connection_id

    async def unregister(self, connection_id: str) -> None:
        async with self._lock:
            self._connections.pop(connection_id, None)

    async def close_all(self, code: int = 1012, reason: str = 'server_shutdown') -> None:
        async with self._lock:
            items = list(self._connections.items())
            self._connections.clear()
        if not items:
            return
        await asyncio.gather(
            *(self._safe_close(websocket, code=code, reason=reason) for _, websocket in items),
            return_exceptions=True,
        )

    async def _safe_close(self, websocket: WebSocket, *, code: int, reason: str) -> None:
        try:
            await websocket.close(code=code, reason=reason)
        except Exception:
            return

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            items = list(self._connections.items())
        if not items:
            return
        stale_ids: list[str] = []
        for connection_id, websocket in items:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale_ids.append(connection_id)
        if stale_ids:
            async with self._lock:
                for connection_id in stale_ids:
                    self._connections.pop(connection_id, None)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {'active_connections': len(self._connections)}
