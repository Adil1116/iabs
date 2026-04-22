from __future__ import annotations

import sqlite3

from brain.memory import Hippocampus
from brain.system import IntegratedArtificialBrain


class TrackingConnection:
    def __init__(self, inner: sqlite3.Connection):
        self.inner = inner
        self.closed = False

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def close(self) -> None:
        self.closed = True
        self.inner.close()


def test_sqlite_context_manager_closes_connection(tmp_path, monkeypatch):
    db_path = tmp_path / 'memory.sqlite'
    opened: list[TrackingConnection] = []
    original_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = TrackingConnection(original_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr('brain.memory.sqlite3.connect', tracked_connect)

    memory = Hippocampus(storage_path=db_path, autoload=False)
    with memory._connect_sqlite() as conn:
        conn.execute('SELECT 1').fetchone()

    assert opened
    assert opened[-1].closed is True


def test_brain_close_is_safe_and_persists_state(tmp_path):
    brain = IntegratedArtificialBrain(
        seed=23,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=False,
    )
    brain.memory.store_memory('cleanup_test', {'text': 'hello'}, importance=0.9, source='test')
    saved_state = brain.save_state()

    brain.close()
    brain.close()

    assert saved_state.exists()
    assert (tmp_path / 'memory.sqlite').exists()
