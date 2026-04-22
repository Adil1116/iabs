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


def test_knowledge_verify_exposes_trust_and_contradiction_signals(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}

        sources = [
            {
                'source_name': 'Official Workflow Spec',
                'source_url': 'https://docs.example.edu/spec/workflow',
                'tags': ['workflow', 'automation', 'limits'],
                'chunk_size_chars': 320,
                'overlap_chars': 40,
                'text': (
                    'The workflow engine processes 3 approval steps before deployment. '
                    'This official reference also explains automation checks and retry handling in detail. '
                ) * 10,
            },
            {
                'source_name': 'Community workflow note',
                'source_url': 'https://blog.example.com/workflow-note',
                'tags': ['workflow', 'automation', 'community'],
                'chunk_size_chars': 320,
                'overlap_chars': 40,
                'text': (
                    'The workflow engine does not process 3 approval steps before deployment. '
                    'This community note claims there are only 2 steps and mentions exceptions briefly. '
                ) * 10,
            },
        ]

        for payload in sources:
            response = client.post('/knowledge/ingest/text', json=payload, headers=headers)
            assert response.status_code == 200, response.text

        verify_response = client.get(
            '/knowledge/verify',
            params={'query': 'workflow approval steps before deployment', 'source_limit': 5, 'evidence_limit': 6},
            headers=headers,
        )
        assert verify_response.status_code == 200, verify_response.text
        payload = verify_response.json()

        assert payload['matched_sources'] >= 2
        assert payload['matched_evidence'] >= 2
        assert payload['trust_overview']['average_trust_score'] > 0
        assert payload['sources'][0]['trust_score'] >= payload['sources'][-1]['trust_score']
        assert payload['contradiction_candidates']
        assert payload['confidence_explanation']['score'] > 0
        assert payload['confidence_explanation']['label'] in {'low', 'medium', 'high'}
        assert payload['consensus_summary']['state'] in {'contested', 'aligned', 'emerging'}
        assert payload['provenance_chain']
        assert payload['contradiction_clusters']
        assert payload['sources'][0]['provenance']['ingest_id']
        assert payload['evidence'][0]['confidence_label'] in {'low', 'medium', 'high'}
        assert payload['verification_status'] in {'mixed_evidence', 'limited_evidence', 'well_supported'}

        analytics_response = client.get('/knowledge/analytics', headers=headers)
        assert analytics_response.status_code == 200, analytics_response.text
        analytics = analytics_response.json()
        assert analytics['average_trust_score'] > 0
        assert any(item['name'] in {'high', 'medium', 'low'} for item in analytics['trust_distribution'])


def test_root_and_capabilities_include_v214_knowledge_features(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        headers = {'Authorization': f'Bearer {get_token(client)}'}

        root_response = client.get('/')
        assert root_response.status_code == 200, root_response.text
        root_payload = root_response.json()
        assert root_payload['version'] == '2.16.0'
        assert 'knowledge_source_trust_scoring' in root_payload['features']
        assert 'knowledge_query_verification' in root_payload['features']
        assert 'knowledge_contradiction_detection' in root_payload['features']
        assert 'knowledge_confidence_explanations' in root_payload['features']
        assert 'knowledge_consensus_summary' in root_payload['features']
        assert 'knowledge_provenance_chain' in root_payload['features']
        assert 'knowledge_automation_registry' in root_payload['features']
        assert 'knowledge_scheduler_background_loop' in root_payload['features']
        assert 'adaptive_source_scoring' in root_payload['features']

        capabilities_response = client.get('/brain/capabilities', headers=headers)
        assert capabilities_response.status_code == 200, capabilities_response.text
        feature_flags = capabilities_response.json()['features']
        assert feature_flags['knowledge_source_trust_scoring'] is True
        assert feature_flags['knowledge_query_verification'] is True
        assert feature_flags['knowledge_contradiction_detection'] is True
        assert feature_flags['knowledge_confidence_explanations'] is True
        assert feature_flags['knowledge_consensus_summary'] is True
        assert feature_flags['knowledge_provenance_chain'] is True
        assert feature_flags['knowledge_automation_registry'] is True
        assert feature_flags['knowledge_scheduler_background_loop'] is True
        assert feature_flags['adaptive_source_scoring'] is True


def test_openapi_schema_groups_core_routes(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        schema = client.get('/openapi.json').json()
        assert schema['info']['x-iabs-release']['version'] == '2.16.0'
        tag_names = {item['name'] for item in schema['tags']}
        assert {'Auth', 'System', 'Brain', 'Memory', 'Knowledge', 'Audit'}.issubset(tag_names)
        assert schema['paths']['/knowledge/verify']['get']['summary'] == 'Verify claim against ingested knowledge'
        assert schema['paths']['/brain/config/posture']['get']['tags'] == ['System']
