from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from brain.config import AppConfig


def make_config(tmp_path) -> AppConfig:
    return AppConfig(
        state_path=tmp_path / 'brain_state.json',
        memory_path=tmp_path / 'memory.sqlite',
        audit_log_path=tmp_path / 'audit.log',
        refresh_token_store_path=tmp_path / 'refresh_tokens.json',
        auth_enabled=True,
        auth_username='admin',
        auth_password='change-me-now',
        log_format='text',
    )


def get_token(client: TestClient) -> str:
    response = client.post('/auth/token', json={'username': 'admin', 'password': 'change-me-now'})
    assert response.status_code == 200, response.text
    return response.json()['access_token']


def test_knowledge_briefing_endpoint_returns_ranked_sources_and_highlights(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}

        payloads = [
            {
                'source_name': 'دليل workflow الذكي',
                'source_url': 'https://example.com/workflow-guide',
                'tags': ['workflow', 'hooks', 'automation'],
                'chunk_size_chars': 320,
                'overlap_chars': 40,
                'text': (
                    'هذا الدليل يشرح workflow عملي يربط بين hooks وautomation وapi داخل النظام. '
                    'كما يوضح خطوات مراقبة التنفيذ وتسجيل الأحداث واختبار النتائج بشكل متكرر. '
                ) * 10,
            },
            {
                'source_name': 'خطة roadmap للذاكرة',
                'source_url': 'https://example.com/memory-roadmap',
                'tags': ['roadmap', 'memory', 'testing'],
                'chunk_size_chars': 320,
                'overlap_chars': 40,
                'text': (
                    'تشرح هذه الخطة تطوير memory وcontext مع roadmap واضحة للاختبارات والتحسين التدريجي. '
                    'كما تربط بين القياس والتتبع وتحديد أولويات التنفيذ القادمة. '
                ) * 10,
            },
        ]

        for payload in payloads:
            response = client.post('/knowledge/ingest/text', json=payload, headers=headers)
            assert response.status_code == 200, response.text

        briefing_response = client.get(
            '/knowledge/briefing',
            params={'query': 'workflow hooks automation', 'source_limit': 3, 'highlight_limit': 4},
            headers=headers,
        )
        assert briefing_response.status_code == 200, briefing_response.text
        payload = briefing_response.json()

        assert payload['total_sources'] >= 2
        assert payload['matched_sources'] >= 1
        assert payload['briefing']['headline'].startswith('Briefing معرفي')
        assert payload['briefing']['coverage_ratio'] > 0
        assert payload['source_snapshots'][0]['source_name'] == 'دليل workflow الذكي'
        assert payload['source_snapshots'][0]['score'] > 0
        assert len(payload['highlights']) >= 1
        assert payload['highlights'][0]['source_name'] == 'دليل workflow الذكي'
        assert payload['briefing']['recommended_questions']


def test_capabilities_expose_knowledge_briefing_feature(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}
        response = client.get('/brain/capabilities', headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['features']['knowledge_briefing'] is True
