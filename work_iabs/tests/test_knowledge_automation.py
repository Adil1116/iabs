from __future__ import annotations

import api.main as main_module
import brain.autonomous_learning as auto_module
from fastapi.testclient import TestClient

from api.main import create_app
from brain.config import AppConfig


def make_config(tmp_path) -> AppConfig:
    return AppConfig(
        state_path=tmp_path / 'brain_state.json',
        memory_path=tmp_path / 'memory.sqlite',
        audit_log_path=tmp_path / 'audit.log',
        refresh_token_store_path=tmp_path / 'refresh_tokens.json',
        knowledge_registry_path=tmp_path / 'knowledge_automation.json',
        auth_enabled=True,
        auth_username='admin',
        auth_password='change-me-now',
        log_format='text',
        knowledge_scheduler_poll_seconds=3600,
    )


def get_access_token(client: TestClient) -> str:
    response = client.post('/auth/token', json={'username': 'admin', 'password': 'change-me-now'})
    assert response.status_code == 200, response.text
    return response.json()['access_token']


def test_knowledge_automation_register_run_and_learn_topics(tmp_path, monkeypatch):
    async def fake_discover(source, *, timeout_seconds: float = 20.0):
        return [
            {'url': 'https://example.com/articles/ai-1', 'title': 'AI Workflow 1', 'score': 2.1, 'discovered_via': 'rss'},
            {'url': 'https://example.com/articles/ai-2', 'title': 'AI Workflow 2', 'score': 2.0, 'discovered_via': 'rss'},
        ]

    async def fake_fetch(url: str, *, timeout_seconds: float = 20.0):
        title = 'AI Workflow 1' if url.endswith('ai-1') else 'AI Workflow 2'
        return {
            'final_url': url,
            'content_type': 'text/html; charset=utf-8',
            'status_code': 200,
            'title': title,
            'text': (
                'This autonomous learning article explains workflow automation, memory context, source ranking, and ai orchestration in detail. '
                'The system learns from repeated successful ingests and improves the source score over time. '
            ) * 12,
        }

    monkeypatch.setattr(auto_module, 'discover_source_candidates', fake_discover)
    monkeypatch.setattr(main_module.knowledge_ingestion, 'fetch_url_text', fake_fetch)

    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_access_token(client)}'}
        register_response = client.post(
            '/knowledge/automation/sources',
            json={
                'name': 'AI Automation Feed',
                'feed_url': 'https://example.com/feed.xml',
                'mode': 'rss',
                'tags': ['ai', 'automation'],
                'keywords': ['workflow', 'memory'],
                'interval_seconds': 3600,
                'max_items_per_run': 2,
            },
            headers=headers,
        )
        assert register_response.status_code == 200, register_response.text
        source = register_response.json()['source']
        assert source['source_id']

        list_response = client.get('/knowledge/automation/sources', headers=headers)
        assert list_response.status_code == 200, list_response.text
        assert list_response.json()['count'] == 1

        run_response = client.post(
            f"/knowledge/automation/sources/{source['source_id']}/run",
            json={'force': True},
            headers=headers,
        )
        assert run_response.status_code == 200, run_response.text
        run_payload = run_response.json()['run']
        assert run_payload['ingested'] >= 1
        assert run_payload['source_score'] > 0
        assert run_response.json()['status']['top_topics']

        rerun_response = client.post(
            f"/knowledge/automation/sources/{source['source_id']}/run",
            json={'force': True},
            headers=headers,
        )
        assert rerun_response.status_code == 200, rerun_response.text
        rerun_payload = rerun_response.json()['run']
        assert rerun_payload['skipped_seen'] >= 1 or rerun_payload['duplicate_items'] >= 1


def test_knowledge_automation_status_and_delete_source(tmp_path, monkeypatch):
    async def fake_discover(source, *, timeout_seconds: float = 20.0):
        return []

    monkeypatch.setattr(auto_module, 'discover_source_candidates', fake_discover)

    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_access_token(client)}'}
        register_response = client.post(
            '/knowledge/automation/sources',
            json={
                'name': 'Docs Source',
                'feed_url': 'https://example.com/docs',
                'mode': 'page',
                'keywords': ['docs'],
            },
            headers=headers,
        )
        assert register_response.status_code == 200, register_response.text
        source_id = register_response.json()['source']['source_id']

        status_response = client.get('/knowledge/automation/status', headers=headers)
        assert status_response.status_code == 200, status_response.text
        status_payload = status_response.json()
        assert status_payload['enabled'] is True
        assert status_payload['source_count'] == 1

        delete_response = client.delete(f'/knowledge/automation/sources/{source_id}', headers=headers)
        assert delete_response.status_code == 200, delete_response.text
        assert delete_response.json()['deleted'] is True

        missing_response = client.delete(f'/knowledge/automation/sources/{source_id}', headers=headers)
        assert missing_response.status_code == 404
