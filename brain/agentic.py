from __future__ import annotations

from collections import Counter
from typing import Any
from uuid import uuid4
import re
import time

import httpx


class UserModelEngine:
    _STOPWORDS = {
        'في', 'من', 'على', 'الى', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'انا', 'أنا', 'انت', 'أنت',
        'هو', 'هي', 'تم', 'بعد', 'قبل', 'كان', 'كانت', 'the', 'and', 'for', 'that', 'this', 'then', 'have',
        'عايز', 'اريد', 'أريد', 'محتاج', 'لازم', 'please', 'project', 'مشروع', 'عاوز', 'ممكن', 'هل',
    }
    _DOMAIN_KEYWORDS = {
        'architecture': {'architecture', 'architect', 'معماري', 'معمارية', 'هيكل', 'design', 'system design'},
        'workflow': {'workflow', 'flow', 'pipeline', 'سير', 'تدفق', 'وركفلو', 'workflows'},
        'spec': {'spec', 'specs', 'مواصفات', 'متطلبات', 'وثيقة', 'requirement'},
        'testing': {'test', 'tests', 'pytest', 'unit', 'qa', 'testing', 'اختبار', 'اختبارات'},
        'api': {'api', 'apis', 'endpoint', 'endpoints', 'webhook', 'webhooks', 'واجهة', 'واجهات'},
        'memory': {'memory', 'memories', 'ذاكره', 'ذاكرة', 'context', 'سياق'},
        'automation': {'automate', 'automation', 'hook', 'hooks', 'agent', 'agentic', 'تنفيذ', 'ربط'},
        'roadmap': {'roadmap', 'sprint', 'milestone', 'خارطة', 'خريطه', 'مرحلة', 'مراحل'},
    }
    _PAIN_MARKERS = {'مشكلة', 'مشكله', 'ناقص', 'ضعيف', 'يفتقد', 'مفقود', 'bug', 'issue', 'broken', 'slow'}

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = str(text).strip().lower()
        normalized = normalized.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ة', 'ه').replace('ى', 'ي')
        normalized = re.sub(r'[\u064b-\u065f\u0670]', '', normalized)
        normalized = re.sub(r'[^\w\s-]+', ' ', normalized, flags=re.UNICODE)
        return ' '.join(normalized.split())

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        normalized = cls._normalize(text)
        return [token for token in re.findall(r'\w+', normalized, flags=re.UNICODE) if len(token) >= 3 and token not in cls._STOPWORDS]

    @staticmethod
    def empty_profile() -> dict[str, Any]:
        return {
            'version': '3.0-lite',
            'updated_at': 0.0,
            'interaction_count': 0,
            'top_interests': [],
            'stated_goals': [],
            'preferences': {},
            'communication_style': {
                'verbosity': 'balanced',
                'format': 'practical',
                'language': 'ar-EG',
            },
            'recurring_requests': [],
            'pain_points': [],
            'predicted_needs': [],
            'last_intent': None,
            'confidence': 0.0,
        }

    @classmethod
    def update_profile(
        cls,
        profile: dict[str, Any] | None,
        *,
        text: str,
        context_topics: list[str] | None = None,
        affect_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = cls.empty_profile()
        if isinstance(profile, dict):
            current.update({key: value for key, value in profile.items() if value is not None})
            communication_style = dict(cls.empty_profile()['communication_style'])
            communication_style.update(current.get('communication_style') or {})
            current['communication_style'] = communication_style
        clean_text = str(text).strip()
        tokens = cls._tokens(clean_text)
        topic_counter = Counter(str(item) for item in current.get('top_interests', []) if str(item).strip())
        topic_counter.update(tokens)
        topic_counter.update(str(item).strip() for item in (context_topics or []) if str(item).strip())
        current['top_interests'] = [item for item, _ in topic_counter.most_common(6)]

        preferences = dict(current.get('preferences') or {})
        normalized_blob = cls._normalize(clean_text)
        for label, keyword_group in cls._DOMAIN_KEYWORDS.items():
            score = float(preferences.get(label, 0.0))
            matches = sum(1 for keyword in keyword_group if cls._normalize(keyword) in normalized_blob)
            if matches:
                score = min(1.0, score + (0.18 * matches))
            elif label in current['top_interests']:
                score = min(1.0, score + 0.05)
            if score > 0:
                preferences[label] = round(score, 4)
        current['preferences'] = dict(sorted(preferences.items(), key=lambda item: item[1], reverse=True)[:8])

        verbosity = 'balanced'
        if len(clean_text) > 120 or any(marker in normalized_blob for marker in ['تفصيل', 'شرح', 'وسع', 'اكمل', 'طور', 'roadmap', 'spec']):
            verbosity = 'detailed'
        elif any(marker in normalized_blob for marker in ['مختصر', 'تلخيص', 'quick', 'short']):
            verbosity = 'concise'

        response_format = 'practical'
        if any(marker in normalized_blob for marker in ['roadmap', 'sprint', 'strategy', 'خطه', 'خارطه']):
            response_format = 'strategic'
        if any(marker in normalized_blob for marker in ['patch', 'code', 'endpoint', 'spec', 'workflow', 'webhook']):
            response_format = 'implementation'

        current['communication_style'] = {
            'verbosity': verbosity,
            'format': response_format,
            'language': 'ar-EG' if any('\u0600' <= ch <= '\u06ff' for ch in clean_text) else 'en',
        }

        stated_goals = [str(item) for item in current.get('stated_goals', []) if str(item).strip()]
        if any(marker in normalized_blob for marker in ['عايز', 'اريد', 'محتاج', 'لازم', 'ابدأ', 'اكمل', 'طور', 'نفذ']):
            compact = re.sub(r'\s+', ' ', clean_text)[:120]
            if compact and compact not in stated_goals:
                stated_goals.append(compact)
        current['stated_goals'] = stated_goals[-6:]

        recurring_requests = [str(item) for item in current.get('recurring_requests', []) if str(item).strip()]
        for label in current['preferences'].keys():
            if label not in recurring_requests:
                recurring_requests.append(label)
        current['recurring_requests'] = recurring_requests[-8:]

        pain_points = [str(item) for item in current.get('pain_points', []) if str(item).strip()]
        if any(marker in normalized_blob for marker in cls._PAIN_MARKERS):
            pain = re.sub(r'\s+', ' ', clean_text)[:100]
            if pain not in pain_points:
                pain_points.append(pain)
        if affect_state and float(affect_state.get('stress', 0.0)) > 0.6 and 'ضغط سياقي أو تنفيذي' not in pain_points:
            pain_points.append('ضغط سياقي أو تنفيذي')
        current['pain_points'] = pain_points[-6:]

        predicted_needs: list[str] = []
        ranked_preferences = list(current['preferences'].keys())
        if 'architecture' in ranked_preferences or 'roadmap' in ranked_preferences:
            predicted_needs.append('roadmap تنفيذية قصيرة')
        if 'spec' in ranked_preferences or 'workflow' in ranked_preferences:
            predicted_needs.append('spec واضحة للمنطق الجديد')
        if 'api' in ranked_preferences or 'automation' in ranked_preferences:
            predicted_needs.append('action hooks قابلة للاختبار')
        if 'testing' in ranked_preferences:
            predicted_needs.append('اختبارات تغطي السيناريوهات الجديدة')
        if not predicted_needs:
            predicted_needs.append('تلخيص عملي للخطوة التالية')
        current['predicted_needs'] = predicted_needs[:4]

        current['interaction_count'] = int(current.get('interaction_count', 0)) + 1
        current['last_intent'] = cls.infer_tom(
            profile=current,
            text=clean_text,
            recent_context=[],
            affect_state=affect_state or {},
        ).get('intent')
        confidence = 0.48 + min(0.36, 0.06 * len(current['preferences'])) + min(0.12, 0.02 * len(current['stated_goals']))
        current['confidence'] = round(min(0.97, confidence), 4)
        current['updated_at'] = time.time()
        return current

    @classmethod
    def infer_tom(
        cls,
        *,
        profile: dict[str, Any] | None,
        text: str,
        recent_context: list[dict[str, Any]] | list[str] | None = None,
        affect_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_text = str(text).strip()
        normalized_blob = cls._normalize(clean_text)
        recent_blob = ' '.join(
            str(item.get('user_text', '')) if isinstance(item, dict) else str(item)
            for item in (recent_context or [])
        )
        intent = 'general_request'
        if any(token in normalized_blob for token in ['اصلح', 'fix', 'debug', 'bug', 'مشكله', 'issue']):
            intent = 'fix_or_stabilize'
        elif any(token in normalized_blob for token in ['طور', 'اكمل', 'add', 'patch', 'نفذ', 'implementation']):
            intent = 'develop_and_extend'
        elif any(token in normalized_blob for token in ['roadmap', 'خطه', 'خارطه', 'sprint', 'plan']):
            intent = 'plan_and_sequence'
        elif any(token in normalized_blob for token in ['webhook', 'hooks', 'api', 'endpoint', 'automation']):
            intent = 'connect_to_external_actions'
        elif any(token in normalized_blob for token in ['spec', 'مواصفات', 'متطلبات', 'requirement']):
            intent = 'specify_design'

        expected_outcome = 'رد عملي مرتبط بالسياق'
        if intent == 'develop_and_extend':
            expected_outcome = 'كود أو Patch يضيف قدرات جديدة'
        elif intent == 'plan_and_sequence':
            expected_outcome = 'خطة تنفيذية واضحة قابلة للتقسيم'
        elif intent == 'connect_to_external_actions':
            expected_outcome = 'ربط القرار بأفعال خارجية قابلة للاختبار'
        elif intent == 'specify_design':
            expected_outcome = 'Spec مركزة ومنطق معماري واضح'
        elif intent == 'fix_or_stabilize':
            expected_outcome = 'حل مباشر مع اختبارات تمنع التكرار'

        emotional_need = 'clarity'
        if any(token in normalized_blob for token in ['عاجل', 'urgent', 'حالا', 'فورا', 'critical']):
            emotional_need = 'reassurance_and_speed'
        elif any(token in normalized_blob for token in ['اكمل', 'معايا', 'سوا', 'طور']):
            emotional_need = 'collaboration'
        elif affect_state and float(affect_state.get('stress', 0.0)) > 0.58:
            emotional_need = 'certainty'

        communication_style = {
            'verbosity': 'balanced',
            'format': 'practical',
        }
        if isinstance(profile, dict):
            communication_style.update(profile.get('communication_style') or {})

        next_best_actions = []
        if intent == 'develop_and_extend':
            next_best_actions = ['إضافة طبقة User Modeling', 'ربط Action Hooks', 'توسيع Dream Engine']
        elif intent == 'connect_to_external_actions':
            next_best_actions = ['تسجيل hook', 'اختبار dry-run', 'توصيل webhook فعلي لاحقاً']
        elif intent == 'plan_and_sequence':
            next_best_actions = ['ترتيب الأولويات', 'تقسيم العمل إلى سبرنتات', 'تحديد أسرع Patch']
        else:
            next_best_actions = ['تلخيص السياق', 'تحديد الفجوة', 'اقتراح الخطوة التالية']

        confidence = 0.46
        if clean_text:
            confidence += 0.12
        if recent_blob:
            confidence += 0.1
        if isinstance(profile, dict) and profile.get('preferences'):
            confidence += min(0.22, 0.04 * len(profile.get('preferences', {})))
        if intent != 'general_request':
            confidence += 0.1

        return {
            'timestamp': time.time(),
            'intent': intent,
            'expected_outcome': expected_outcome,
            'emotional_need': emotional_need,
            'preferred_response_style': communication_style,
            'contextual_clues': [item for item in cls._tokens(clean_text)[:6]],
            'next_best_actions': next_best_actions,
            'confidence': round(min(0.97, confidence), 4),
        }


class ActionHookGateway:
    @staticmethod
    def empty_registry() -> dict[str, Any]:
        return {'hooks': [], 'events': []}

    @staticmethod
    def _normalize(text: str) -> str:
        return UserModelEngine._normalize(text)

    @classmethod
    def register_hook(
        cls,
        registry: dict[str, Any] | None,
        *,
        name: str,
        event: str,
        action_type: str = 'webhook',
        target_url: str | None = None,
        method: str = 'POST',
        headers: dict[str, str] | None = None,
        payload_template: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        cooldown_seconds: int = 0,
        active: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = cls.empty_registry()
        if isinstance(registry, dict):
            current.update(registry)
            current['hooks'] = list(registry.get('hooks', []))
            current['events'] = list(registry.get('events', []))
        now = time.time()
        hook = {
            'hook_id': f'hook_{uuid4().hex[:10]}',
            'name': str(name).strip() or 'Unnamed Hook',
            'event': str(event).strip() or 'decision.made',
            'action_type': str(action_type).strip() or 'webhook',
            'target_url': str(target_url).strip() if target_url else None,
            'method': str(method).strip().upper() or 'POST',
            'headers': {str(key): str(value) for key, value in (headers or {}).items() if str(key).strip()},
            'payload_template': payload_template or {},
            'keywords': list(dict.fromkeys(str(item).strip() for item in (keywords or []) if str(item).strip())),
            'cooldown_seconds': max(0, int(cooldown_seconds)),
            'active': bool(active),
            'created_at': now,
            'updated_at': now,
            'last_triggered_at': None,
            'trigger_count': 0,
        }
        current['hooks'].append(hook)
        return current, hook

    @classmethod
    def get_hook(cls, registry: dict[str, Any] | None, hook_id: str) -> dict[str, Any]:
        for hook in list((registry or {}).get('hooks', [])):
            if str(hook.get('hook_id')) == str(hook_id):
                return hook
        raise ValueError('Action hook غير موجود')

    @classmethod
    def update_hook(
        cls,
        registry: dict[str, Any] | None,
        hook_id: str,
        updates: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = cls.empty_registry()
        if isinstance(registry, dict):
            current.update(registry)
            current['hooks'] = list(registry.get('hooks', []))
            current['events'] = list(registry.get('events', []))
        clean_updates = dict(updates or {})
        for hook in current['hooks']:
            if str(hook.get('hook_id')) != str(hook_id):
                continue
            if 'name' in clean_updates and str(clean_updates.get('name') or '').strip():
                hook['name'] = str(clean_updates.get('name')).strip()
            if 'event' in clean_updates and str(clean_updates.get('event') or '').strip():
                hook['event'] = str(clean_updates.get('event')).strip()
            if 'action_type' in clean_updates and str(clean_updates.get('action_type') or '').strip():
                hook['action_type'] = str(clean_updates.get('action_type')).strip()
            if 'target_url' in clean_updates:
                hook['target_url'] = str(clean_updates.get('target_url')).strip() if clean_updates.get('target_url') else None
            if 'method' in clean_updates and str(clean_updates.get('method') or '').strip():
                hook['method'] = str(clean_updates.get('method')).strip().upper()
            if 'headers' in clean_updates:
                hook['headers'] = {
                    str(key): str(value)
                    for key, value in (clean_updates.get('headers') or {}).items()
                    if str(key).strip()
                }
            if 'payload_template' in clean_updates:
                hook['payload_template'] = clean_updates.get('payload_template') or {}
            if 'keywords' in clean_updates:
                hook['keywords'] = list(
                    dict.fromkeys(
                        str(item).strip() for item in (clean_updates.get('keywords') or []) if str(item).strip()
                    )
                )
            if 'cooldown_seconds' in clean_updates and clean_updates.get('cooldown_seconds') is not None:
                hook['cooldown_seconds'] = max(0, int(clean_updates.get('cooldown_seconds', 0)))
            if 'active' in clean_updates:
                hook['active'] = bool(clean_updates.get('active'))
            hook['updated_at'] = time.time()
            return current, hook
        raise ValueError('Action hook غير موجود')

    @classmethod
    def delete_hook(cls, registry: dict[str, Any] | None, hook_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        current = cls.empty_registry()
        if isinstance(registry, dict):
            current.update(registry)
            current['hooks'] = list(registry.get('hooks', []))
            current['events'] = list(registry.get('events', []))
        remaining: list[dict[str, Any]] = []
        deleted_hook: dict[str, Any] | None = None
        for hook in current['hooks']:
            if deleted_hook is None and str(hook.get('hook_id')) == str(hook_id):
                deleted_hook = hook
                continue
            remaining.append(hook)
        if deleted_hook is None:
            raise ValueError('Action hook غير موجود')
        current['hooks'] = remaining
        return current, deleted_hook

    @classmethod
    def list_hooks(cls, registry: dict[str, Any] | None, *, active_only: bool = False) -> list[dict[str, Any]]:
        hooks = list((registry or {}).get('hooks', []))
        if active_only:
            hooks = [hook for hook in hooks if hook.get('active')]
        return hooks

    @classmethod
    def overview(cls, registry: dict[str, Any] | None) -> dict[str, Any]:
        hooks = cls.list_hooks(registry)
        active_count = sum(1 for hook in hooks if hook.get('active'))
        return {
            'total_hooks': len(hooks),
            'active_hooks': active_count,
            'events_logged': len(list((registry or {}).get('events', []))),
            'recent_hooks': hooks[-3:],
        }

    @classmethod
    def dispatch(
        cls,
        registry: dict[str, Any] | None,
        *,
        event: str,
        text: str = '',
        decision: str | None = None,
        topics: list[str] | None = None,
        dry_run: bool = True,
        allow_network: bool = False,
        timeout_seconds: float = 4.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = cls.empty_registry()
        if isinstance(registry, dict):
            current.update(registry)
            current['hooks'] = list(registry.get('hooks', []))
            current['events'] = list(registry.get('events', []))
        normalized_blob = cls._normalize(' '.join([text or '', decision or '', ' '.join(topics or [])]))
        matched: list[dict[str, Any]] = []
        now = time.time()
        for hook in current['hooks']:
            if not hook.get('active'):
                continue
            if str(hook.get('event', '')).strip() != str(event).strip():
                continue
            if hook.get('keywords'):
                normalized_keywords = [cls._normalize(item) for item in hook.get('keywords', []) if cls._normalize(item)]
                if normalized_keywords and not any(keyword in normalized_blob for keyword in normalized_keywords):
                    continue
            last_triggered_at = hook.get('last_triggered_at')
            cooldown_seconds = max(0, int(hook.get('cooldown_seconds', 0)))
            if last_triggered_at and cooldown_seconds and (now - float(last_triggered_at) < cooldown_seconds):
                continue
            matched.append(hook)

        dispatches: list[dict[str, Any]] = []
        for hook in matched:
            payload = {
                'event': event,
                'text': text,
                'decision': decision,
                'topics': list(topics or []),
                'timestamp': now,
                'hook': {
                    'hook_id': hook.get('hook_id'),
                    'name': hook.get('name'),
                    'action_type': hook.get('action_type'),
                },
                'template': hook.get('payload_template', {}),
            }
            executed_remotely = bool(not dry_run and allow_network and hook.get('target_url'))
            status = 'simulated'
            http_status = None
            detail = 'dry-run'
            if executed_remotely:
                try:
                    with httpx.Client(timeout=timeout_seconds) as client:
                        response = client.request(
                            str(hook.get('method', 'POST')).upper(),
                            str(hook.get('target_url')),
                            headers=hook.get('headers') or {},
                            json=payload,
                        )
                    http_status = response.status_code
                    status = 'delivered' if response.status_code < 400 else 'failed'
                    detail = f'http_status={response.status_code}'
                except Exception as exc:  # pragma: no cover - network dependent
                    status = 'failed'
                    detail = str(exc)
            if executed_remotely:
                hook['last_triggered_at'] = now
                hook['trigger_count'] = int(hook.get('trigger_count', 0)) + 1
            dispatches.append(
                {
                    'hook_id': hook.get('hook_id'),
                    'name': hook.get('name'),
                    'status': status,
                    'http_status': http_status,
                    'detail': detail,
                    'payload_preview': payload,
                }
            )
        event_log = {
            'event_id': f'action_event_{uuid4().hex[:10]}',
            'event': event,
            'timestamp': now,
            'matched_hooks': len(matched),
            'dispatches': dispatches,
            'dry_run': bool(dry_run or not allow_network),
        }
        current['events'].append(event_log)
        current['events'] = current['events'][-100:]
        summary = {
            'event': event,
            'matched_hooks': len(matched),
            'delivered_hooks': sum(1 for item in dispatches if item.get('status') == 'delivered'),
            'simulated_hooks': sum(1 for item in dispatches if item.get('status') == 'simulated'),
            'failed_hooks': sum(1 for item in dispatches if item.get('status') == 'failed'),
            'dispatches': dispatches,
        }
        return current, summary


class DreamEngine:
    @staticmethod
    def synthesize(
        *,
        trigger: str,
        context_window: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        affect_state: dict[str, Any],
    ) -> dict[str, Any]:
        texts = [str(item.get('user_text') or item.get('decision') or '') for item in context_window if isinstance(item, dict)]
        tokens = []
        for text in texts:
            tokens.extend(UserModelEngine._tokens(text))
        topic_counter = Counter(tokens)
        for goal in goals:
            if isinstance(goal, dict):
                topic_counter.update(UserModelEngine._tokens(str(goal.get('title', ''))))
                topic_counter.update(UserModelEngine._tokens(str(goal.get('description', ''))))
        dream_topics = [topic for topic, _ in topic_counter.most_common(6)]
        active_goal_titles = [str(goal.get('title', '')) for goal in goals if isinstance(goal, dict)][:3]
        simulated_paths = []
        for topic in dream_topics[:3]:
            simulated_paths.append(
                {
                    'topic': topic,
                    'scenario': f'لو ركزنا على {topic} في دورة أوفلاين قصيرة، فالأرجح هنطلع بربط أقوى بين الذاكرة والتنفيذ.',
                    'expected_gain': 'تحسين الترابط واتخاذ القرار',
                }
            )
        hypotheses = []
        if active_goal_titles:
            for title in active_goal_titles[:2]:
                hypotheses.append(f'قد يكون أسرع اختصار للتقدم هو تحويل هدف "{title}" إلى Patch صغيرة قابلة للاختبار.')
        for topic in dream_topics[:2]:
            hypotheses.append(f'ممكن يكون {topic} هو المفتاح المشترك بين أغلب الطلبات الأخيرة.')
        if not hypotheses:
            hypotheses.append('البيانات الحالية تشير إلى أن أفضل استخدام لوضع النوم هو إعادة ترتيب الذكريات وتقليل الضوضاء.')
        recommended_experiments = []
        if 'webhook' in dream_topics or 'api' in dream_topics:
            recommended_experiments.append('جرّب Action Hook واحدة بنمط dry-run قبل التوصيل الخارجي الكامل.')
        if 'ذاكره' in dream_topics or 'memory' in dream_topics or 'سياق' in dream_topics:
            recommended_experiments.append('اربط ملخصات الحلقات الزمنية بملف User Model لرفع جودة الاستدعاء.')
        if active_goal_titles:
            recommended_experiments.append('حوّل أعلى هدف نشط إلى Sprint صغيرة من 2-3 مهام قابلة للقياس.')
        if not recommended_experiments:
            recommended_experiments.append('التجربة التالية: استخراج موضوعين متكررين وبناء spec قصيرة لهما.')
        return {
            'trigger': trigger,
            'timestamp': time.time(),
            'dream_topics': dream_topics,
            'simulated_paths': simulated_paths,
            'hypotheses': hypotheses[:5],
            'recommended_experiments': recommended_experiments[:4],
            'affect_snapshot': affect_state,
            'source_items': min(10, len(context_window)),
        }
