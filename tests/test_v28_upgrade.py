from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from api.main import create_app
from brain.config import AppConfig
from brain.llm_bridge import LLMGenerationResult
from brain.memory import Hippocampus


class FakeLLMBridge:
    def status(self):
        return {
            'enabled': True,
            'configured': True,
            'provider': 'openai-compatible',
            'model': 'fake-test-model',
            'api_base': 'https://example.test/v1',
            'timeout_seconds': 3,
            'max_tokens': 128,
            'temperature': 0.2,
        }

    def generate_reply(self, **_: object) -> LLMGenerationResult:
        return LLMGenerationResult(
            text='رد إبداعي مطور من طبقة اللغة.',
            used=True,
            provider='openai-compatible',
            model='fake-test-model',
            latency_ms=7,
            reason='success',
        )


def make_config(tmp_path, *, encryption_key: str | None = None) -> AppConfig:
    return AppConfig(
        state_path=tmp_path / 'brain_state.json',
        memory_path=tmp_path / 'memory.sqlite',
        audit_log_path=tmp_path / 'audit.log',
        refresh_token_store_path=tmp_path / 'refresh_tokens.json',
        auth_enabled=True,
        auth_username='admin',
        auth_password='change-me-now',
        log_format='text',
        memory_encryption_key=encryption_key,
    )


def get_access_token(client: TestClient) -> str:
    response = client.post('/auth/token', json={'username': 'admin', 'password': 'change-me-now'})
    assert response.status_code == 200, response.text
    return response.json()['access_token']


def test_encrypted_memory_roundtrip_sqlite(tmp_path):
    storage_path = tmp_path / 'encrypted_memory.sqlite'
    memory = Hippocampus(
        capacity=10,
        storage_path=storage_path,
        autoload=False,
        encryption_key='super-secret-passphrase',
    )
    memory.store_memory(
        key='enc_test_1',
        data={'user_text': 'لازم بيانات الذاكرة تبقى مشفرة', 'tag': 'privacy'},
        importance=0.92,
        source='unit-test',
    )
    memory.save()

    conn = sqlite3.connect(storage_path)
    try:
        row = conn.execute('SELECT data FROM memory_records WHERE key = ?', ('enc_test_1',)).fetchone()
    finally:
        conn.close()
    assert row is not None
    raw_payload = row[0]
    assert 'لازم بيانات الذاكرة تبقى مشفرة' not in raw_payload
    assert '__encrypted__' in raw_payload

    reloaded = Hippocampus(
        capacity=10,
        storage_path=storage_path,
        autoload=True,
        encryption_key='super-secret-passphrase',
    )
    recalled = reloaded.recall('enc_test_1')
    assert recalled['user_text'] == 'لازم بيانات الذاكرة تبقى مشفرة'
    assert reloaded.encryption_status()['enabled'] is True


def test_chat_exposes_llm_metadata_and_capabilities(tmp_path):
    app = create_app(make_config(tmp_path, encryption_key='super-secret-passphrase'))
    with TestClient(app) as client:
        client.app.state.text_interface.llm_bridge = FakeLLMBridge()
        access_token = get_access_token(client)
        headers = {'Authorization': f'Bearer {access_token}'}

        chat_response = client.post(
            '/chat',
            json={'text': 'طوّر قدرات الردود وخليها أذكى ومبدعة', 'importance': 0.85},
            headers=headers,
        )
        assert chat_response.status_code == 200, chat_response.text
        payload = chat_response.json()
        assert payload['reply'] == 'رد إبداعي مطور من طبقة اللغة.'
        assert payload['reply_mode'] == 'llm_enhanced'
        assert payload['reply_metadata']['used_llm'] is True
        assert payload['deterministic_reply']

        capabilities = client.get('/brain/capabilities', headers=headers)
        assert capabilities.status_code == 200, capabilities.text
        capabilities_payload = capabilities.json()
        assert capabilities_payload['features']['creative_reply_layer'] is True
        assert capabilities_payload['features']['memory_encryption_at_rest'] is True
        assert capabilities_payload['llm']['enabled'] is True
        assert capabilities_payload['memory']['encryption']['enabled'] is True
