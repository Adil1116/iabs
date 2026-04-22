from __future__ import annotations

import api.main as main_module
from fastapi.testclient import TestClient

from api.main import create_app
from brain.knowledge_ingestion import split_text_into_chunks, text_fingerprint
from brain.config import AppConfig



def make_config(tmp_path, *, auth_enabled: bool = True) -> AppConfig:
    return AppConfig(
        state_path=tmp_path / 'brain_state.json',
        memory_path=tmp_path / 'memory.sqlite',
        audit_log_path=tmp_path / 'audit.log',
        refresh_token_store_path=tmp_path / 'refresh_tokens.json',
        auth_enabled=auth_enabled,
        auth_username='admin',
        auth_password='change-me-now',
        log_format='text',
    )



def get_access_token(client: TestClient) -> str:
    response = client.post('/auth/token', json={'username': 'admin', 'password': 'change-me-now'})
    assert response.status_code == 200, response.text
    return response.json()['access_token']



def test_arabic_chunking_and_fingerprint_are_stable():
    text = (
        'الفصل الأول: كان البطل يسير في المدينة القديمة ويتأمل الشوارع الطويلة. '
        'ثم توقف أمام مكتبة صغيرة وبدأ يقرأ ما كُتب على الواجهة. '\
        * 8
    )
    chunks = split_text_into_chunks(text, max_chars=260, overlap_chars=40, min_chunk_chars=120)
    assert len(chunks) >= 2
    assert all(len(chunk) >= 100 for chunk in chunks)
    assert text_fingerprint(text) == text_fingerprint(text)



def test_knowledge_text_ingestion_search_and_duplicate_guard(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        access_token = get_access_token(client)
        headers = {'Authorization': f'Bearer {access_token}'}
        payload = {
            'source_name': 'مكتبة الرواية',
            'source_url': 'https://example.com/library/story-1',
            'tags': ['رواية', 'تعلم'],
            'chunk_size_chars': 320,
            'overlap_chars': 60,
            'min_chunk_chars': 120,
            'importance': 0.78,
            'text': (
                'الفصل الأول: كان البطل يسير في المدينة القديمة ويتأمل الشوارع الطويلة. '
                'ثم توقف أمام مكتبة صغيرة وبدأ يقرأ ما كُتب على الواجهة. '
                'لاحظ أن كل صفحة تحمل معنى مختلفاً وتفصيلاً جديداً يساعده على الفهم. '
                'وفي نهاية اليوم دوّن ملاحظاته حتى يعود إليها لاحقاً. '\
                * 10
            ),
        }
        response = client.post('/knowledge/ingest/text', json=payload, headers=headers)
        assert response.status_code == 200, response.text
        knowledge = response.json()['knowledge']
        assert knowledge['status'] == 'ingested'
        assert knowledge['duplicate'] is False
        assert knowledge['chunk_count'] >= 2
        assert knowledge['source_name'] == 'مكتبة الرواية'

        duplicate = client.post('/knowledge/ingest/text', json=payload, headers=headers)
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()['knowledge']['duplicate'] is True
        assert duplicate.json()['knowledge']['status'] == 'already_ingested'

        search_response = client.get('/knowledge/search', params={'query': 'مكتبة المدينة'}, headers=headers)
        assert search_response.status_code == 200, search_response.text
        assert search_response.json()['count'] >= 1
        assert search_response.json()['matches'][0]['source_name'] == 'مكتبة الرواية'

        sources_response = client.get('/knowledge/sources', headers=headers)
        assert sources_response.status_code == 200, sources_response.text
        assert sources_response.json()['count'] >= 1
        assert sources_response.json()['sources'][0]['source_name'] == 'مكتبة الرواية'



def test_knowledge_url_ingestion_uses_fetcher_and_stores_source(tmp_path, monkeypatch):
    async def fake_fetch(url: str, *, timeout_seconds: float = 20.0):
        return {
            'final_url': url,
            'content_type': 'text/html; charset=utf-8',
            'status_code': 200,
            'title': 'رواية عربية مجانية',
            'text': (
                'هذه صفحة تجريبية تحتوي على فصل طويل من رواية عربية مجانية. '
                'يتم تقسيم النص إلى مقاطع صغيرة ثم إرسالها إلى الذاكرة. '\
                * 12
            ),
        }

    monkeypatch.setattr(main_module.knowledge_ingestion, 'fetch_url_text', fake_fetch)

    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        access_token = get_access_token(client)
        headers = {'Authorization': f'Bearer {access_token}'}
        response = client.post(
            '/knowledge/ingest/url',
            json={
                'target_url': 'https://example.com/free-arabic-story',
                'tags': ['رواية', 'ويب'],
                'chunk_size_chars': 340,
                'overlap_chars': 50,
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        knowledge = response.json()['knowledge']
        assert knowledge['source_name'] == 'رواية عربية مجانية'
        assert knowledge['source_url'] == 'https://example.com/free-arabic-story'
        assert knowledge['fetch']['status_code'] == 200
        assert knowledge['chunk_count'] >= 2



def test_knowledge_source_details_analytics_and_delete(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        access_token = get_access_token(client)
        headers = {'Authorization': f'Bearer {access_token}'}

        ingest_response = client.post(
            '/knowledge/ingest/text',
            json={
                'source_name': 'وثيقة ترقية المشروع',
                'source_url': 'https://example.com/docs/upgrade-plan',
                'tags': 'ترقية, مشروع, roadmap',
                'chunk_size_chars': 320,
                'overlap_chars': 40,
                'text': (
                    'هذه وثيقة تفصيلية تشرح ترقية المشروع خطوة بخطوة مع تقسيم واضح للأولويات والاختبارات وربط الويب هوك. '
                    'كما تحتوي على تفاصيل عن تحسين الذاكرة والسياق وإدارة مصادر المعرفة داخل النظام. '
                    * 12
                ),
            },
            headers=headers,
        )
        assert ingest_response.status_code == 200, ingest_response.text
        knowledge = ingest_response.json()['knowledge']
        ingest_id = knowledge['ingest_id']

        details_response = client.get(f'/knowledge/sources/{ingest_id}', headers=headers)
        assert details_response.status_code == 200, details_response.text
        details = details_response.json()
        assert details['source']['source_name'] == 'وثيقة ترقية المشروع'
        assert details['chunk_count'] >= 2
        assert details['returned_chunk_count'] >= 1
        assert all(item['ingest_id'] == ingest_id for item in details['chunks'])

        analytics_response = client.get('/knowledge/analytics', headers=headers)
        assert analytics_response.status_code == 200, analytics_response.text
        analytics = analytics_response.json()
        assert analytics['total_sources'] >= 1
        assert analytics['total_chunks'] >= details['chunk_count']
        assert any(item['name'] == 'example.com' for item in analytics['top_domains'])
        assert any(item['name'] == 'ترقية' for item in analytics['top_tags'])

        delete_response = client.delete(f'/knowledge/sources/{ingest_id}', headers=headers)
        assert delete_response.status_code == 200, delete_response.text
        delete_payload = delete_response.json()
        assert delete_payload['deleted'] is True
        assert delete_payload['deleted_chunk_count'] >= 1

        missing_details = client.get(f'/knowledge/sources/{ingest_id}', headers=headers)
        assert missing_details.status_code == 404
