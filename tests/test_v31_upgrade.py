from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import create_app
from brain.config import AppConfig


def make_config(tmp_path, **overrides) -> AppConfig:
    config = AppConfig(
        state_path=tmp_path / 'brain_state.json',
        memory_path=tmp_path / 'memory.sqlite',
        audit_log_path=tmp_path / 'audit.log',
        refresh_token_store_path=tmp_path / 'refresh_tokens.json',
        auth_enabled=True,
        auth_username='admin',
        auth_password='change-me-now',
        log_format='text',
        **overrides,
    )
    return config


def get_token(client: TestClient, username: str = 'admin', password: str = 'change-me-now') -> str:
    response = client.post('/auth/token', json={'username': username, 'password': password})
    assert response.status_code == 200, response.text
    return response.json()['access_token']


def test_config_posture_endpoint_exposes_sanitized_runtime_and_recommendations(tmp_path):
    app = create_app(
        make_config(
            tmp_path,
            environment='production',
            jwt_secret='x' * 40,
            max_request_body_bytes=2048,
        )
    )
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}
        response = client.get('/brain/config/posture', headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['security']['environment'] == 'production'
        assert payload['security']['production_ready'] is False
        assert payload['security']['hardening_score'] < 100
        assert isinstance(payload['security']['critical_warnings'], list)
        assert payload['runtime']['app']['environment'] == 'production'
        assert payload['runtime']['request_guard']['max_request_body_bytes'] == 2048
        assert payload['recommendations']
        serialized = json.dumps(payload, ensure_ascii=False)
        assert 'change-me-now' not in serialized
        assert 'x' * 40 not in serialized


def test_request_size_guard_rejects_oversized_chat_payload(tmp_path):
    app = create_app(make_config(tmp_path, max_request_body_bytes=1024))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}
        oversized_text = 'س' * 3000
        response = client.post('/chat', json={'text': oversized_text}, headers=headers)
        assert response.status_code == 413, response.text
        assert response.headers['X-Request-ID']
        assert response.json()['detail'] == 'حجم الطلب أكبر من الحد المسموح به.'


def test_sqlite_tuning_and_new_capabilities_are_exposed(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}
        capabilities_response = client.get('/brain/capabilities', headers=headers)
        assert capabilities_response.status_code == 200, capabilities_response.text
        payload = capabilities_response.json()
        assert payload['features']['request_body_size_guard'] is True
        assert payload['features']['runtime_config_posture_endpoint'] is True
        assert payload['features']['security_posture_hardening_score'] is True
        assert payload['features']['openapi_tag_catalog'] is True
        assert payload['memory']['sqlite_tuning']['wal_mode'] is True
        assert payload['memory']['sqlite_tuning']['busy_timeout_ms'] == 5000



def test_app_config_from_env_uses_current_version_default(monkeypatch):
    monkeypatch.delenv('IABS_APP_VERSION', raising=False)
    config = AppConfig.from_env()
    assert config.app_version == '2.16.0'
