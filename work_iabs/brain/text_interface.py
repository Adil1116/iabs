from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from brain.core.schemas import DecisionResult, MemoryRecord
from brain.anomaly import PromptAnomalyDetector
from brain.llm_bridge import OptionalLLMBridge
from brain.system import IntegratedArtificialBrain


@dataclass
class ChatCycleResult:
    reply: str
    decision: str
    confidence: float
    top_probabilities: dict[str, float]
    memory_key: str | None
    related_memories: list[dict[str, Any]]
    inferred_importance: float
    context_summary: dict[str, Any]
    personality: dict[str, Any]
    affect_state: dict[str, Any]
    goals_overview: dict[str, Any]
    user_model: dict[str, Any]
    theory_of_mind: dict[str, Any] | None
    action_hook_summary: dict[str, Any] | None
    reply_mode: str
    reply_metadata: dict[str, Any]
    deterministic_reply: str
    self_critique: dict[str, Any]
    episode_memory_key: str | None
    episode_summary: dict[str, Any] | None
    anomaly_report: dict[str, Any]


class TextBrainInterface:
    def __init__(self, brain: IntegratedArtificialBrain, llm_bridge: OptionalLLMBridge | None = None):
        self.brain = brain
        self.llm_bridge = llm_bridge or OptionalLLMBridge()
        self.anomaly_detector = PromptAnomalyDetector()
        self._decision_guidance = {
            'تحرك للأمام': 'الدماغ شايف إن فيه فرصة للاستمرار والتقدم.',
            'توقف': 'فيه إشارة حذر، فالأفضل نهدّي ونراجع الوضع.',
            'تحدث': 'الرد اللفظي هو المسار الأقوى حالياً.',
            'تذكر': 'المدخل الحالي قريب من ذاكرة مهمة مخزنة.',
            'انظر يميناً': 'فيه حاجة جانبية محتاجة انتباه سريع ناحية اليمين.',
            'انظر يساراً': 'فيه إشارة جانبية تستحق مراجعة ناحية اليسار.',
            'اهرب': 'فيه نمط دفاعي أو خطر محتمل في المدخل.',
            'ابحث عن طعام': 'المدخل يوحي بحاجة أو هدف لازم يتلبّى.',
            'نم': 'الاستجابة الداخلية تميل لتقليل النشاط والتهدئة.',
            'استكشف': 'الدماغ شايف إن أفضل خطوة هي التجربة وجمع معلومات جديدة.',
        }
        self._urgent_keywords = {
            'عاجل', 'سريع', 'خطر', 'كارثة', 'ضروري', 'مهم', 'حالاً', 'فوراً', 'انقذ', 'ساعد', 'urgent', 'emergency'
        }
        self._action_markers = {
            'ابدأ', 'ابدا', 'جرّب', 'جرب', 'راجع', 'نفّذ', 'نفذ', 'الخطوة التالية', 'أنصح', 'اقترح', 'next step'
        }

    @staticmethod
    def text_to_vector(text: str, size: int) -> np.ndarray:
        vector = np.zeros(size, dtype=np.float64)
        encoded = text.encode('utf-8')
        if not encoded:
            return vector
        for index, byte in enumerate(encoded):
            vector[index % size] += byte / 255.0
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        for index, byte in enumerate(digest):
            vector[(index * 37) % size] += byte / 255.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def text_to_position(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        left = int(digest[:16], 16) % 10000 / 100.0
        right = int(digest[16:32], 16) % 10000 / 100.0
        return np.array([left, right], dtype=np.float64)

    @staticmethod
    def _compact_text(text: str, limit: int = 120) -> str:
        normalized = re.sub(r'\s+', ' ', text.strip())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)] + '...'

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r'\w+', text.lower(), flags=re.UNICODE) if len(token) >= 3]

    def _estimate_importance(self, text: str, explicit_importance: float | None) -> float:
        if explicit_importance is not None:
            return float(np.clip(explicit_importance, 0.0, 1.0))
        lowered = text.lower()
        score = 0.45
        if len(text.strip()) > 80:
            score += 0.1
        if '?' in text or '؟' in text:
            score += 0.05
        if any(keyword in lowered for keyword in self._urgent_keywords):
            score += 0.25
        exclamation_count = text.count('!') + text.count('！')
        score += min(0.1, exclamation_count * 0.03)
        return float(np.clip(score, 0.1, 1.0))

    def _memory_preview(self, record: MemoryRecord) -> dict[str, Any]:
        data = record.data if isinstance(record.data, dict) else {'value': record.data}
        snippet_source = data.get('user_text') or data.get('text') or data.get('decision') or str(data)
        preview = {
            'key': record.key,
            'source': record.source,
            'importance': record.importance,
            'timestamp': record.timestamp,
            'snippet': self._compact_text(str(snippet_source), limit=90),
        }
        if record.source == 'episode':
            preview['modalities'] = {
                'text': bool(data.get('text')),
                'image': bool(data.get('image_refs')),
                'audio': bool(data.get('audio_refs')),
            }
        return preview

    @staticmethod
    def _affect_sentence(affect_state: dict[str, Any]) -> str:
        mood = affect_state.get('mood_label', 'متوازن')
        if mood == 'فضولي':
            return 'حالتي الحالية فضولية ومهيأة للاستكشاف.'
        if mood == 'مضغوط':
            return 'فيه توتر داخلي بسيط، فبميل للمراجعة والحذر.'
        if mood == 'مرهق':
            return 'طاقة النظام منخفضة نسبياً، فبميل للتبسيط والتركيز.'
        if mood == 'واثق':
            return 'أنا حالياً في وضع واثق ومستقر.'
        if mood == 'هادئ':
            return 'أنا في حالة هادئة ومركّزة.'
        return 'حالتي الداخلية متوازنة.'

    @staticmethod
    def _goal_sentence(goals_overview: dict[str, Any]) -> str:
        top_goals = goals_overview.get('top_active_goals') or []
        if not top_goals:
            return 'مافيش هدف نشط واضح حالياً، فبشتغل على الفهم العام وتحسين السياق.'
        top_goal = top_goals[0]
        progress_pct = round(float(top_goal.get('progress', 0.0)) * 100, 1)
        return f'أعلى هدف نشط دلوقتي هو "{top_goal.get("title", "بدون عنوان")}" وتقدمه {progress_pct}%. '

    def _build_reply(
        self,
        text: str,
        result: DecisionResult,
        related_memories: list[dict[str, Any]],
        context_summary: dict[str, Any],
        personality: dict[str, Any],
        affect_state: dict[str, Any],
        goals_overview: dict[str, Any],
    ) -> str:
        short_text = self._compact_text(text)
        confidence_pct = round(result.confidence * 100, 1)
        guidance = self._decision_guidance.get(result.decision, 'تم تحليل المدخل بنجاح.')
        memory_note = ''
        if related_memories:
            memory_note = f' رجّعت كمان {len(related_memories)} ذكريات مرتبطة تساعد على السياق.'
        context_topics = context_summary.get('topics') or []
        context_note = ''
        if context_topics:
            context_note = f' الموضوعات المسيطرة حالياً: {", ".join(context_topics[:3])}.'
        display_name = personality.get('display_name', 'IABS')
        tone = personality.get('tone', 'balanced')
        affect_note = self._affect_sentence(affect_state)
        goal_note = self._goal_sentence(goals_overview)
        return (
            f'أنا {display_name}، وبحافظ على نبرة {tone}. '
            f'فهمت رسالتك: "{short_text}". '
            f'{affect_note} '
            f'القرار التنفيذي الحالي هو: {result.decision} '
            f'بمستوى ثقة {confidence_pct}%. {guidance}'
            f'{memory_note} {goal_note}{context_note}'
        ).strip()

    def _contains_actionable_step(self, reply: str) -> bool:
        lowered = reply.lower()
        return any(marker.lower() in lowered for marker in self._action_markers)

    def _keyword_overlap_score(self, source_text: str, reply: str) -> float:
        source_tokens = set(self._tokenize(source_text))
        if not source_tokens:
            return 0.0
        reply_tokens = set(self._tokenize(reply))
        if not reply_tokens:
            return 0.0
        return len(source_tokens & reply_tokens) / max(1, len(source_tokens))

    def _run_self_critic(
        self,
        *,
        user_text: str,
        reply: str,
        related_memories: list[dict[str, Any]],
        context_summary: dict[str, Any],
        personality: dict[str, Any],
    ) -> dict[str, Any]:
        reply_len = len(reply.strip())
        clarity = 0.55
        if 60 <= reply_len <= 420:
            clarity += 0.25
        elif reply_len > 30:
            clarity += 0.15
        if 'json' not in reply.lower() and '{' not in reply and '}' not in reply:
            clarity += 0.1

        tone_alignment = 0.62
        tone = str(personality.get('tone', '')).strip().lower()
        if tone and tone in reply.lower():
            tone_alignment += 0.08
        if any(token in reply for token in ['مفهوم', 'أقترح', 'أنصح', 'أقدر', 'ممكن']):
            tone_alignment += 0.14

        memory_alignment = 0.5
        overlap = self._keyword_overlap_score(user_text, reply)
        memory_alignment += min(0.25, overlap * 0.5)
        if related_memories:
            memory_alignment += 0.1
            memory_text = ' '.join(item.get('snippet', '') for item in related_memories)
            memory_alignment += min(0.15, self._keyword_overlap_score(memory_text, reply) * 0.3)
        if context_summary.get('topics'):
            topics_blob = ' '.join(str(item) for item in context_summary.get('topics', []))
            memory_alignment += min(0.1, self._keyword_overlap_score(topics_blob, reply) * 0.3)

        actionability = 0.4
        if self._contains_actionable_step(reply):
            actionability = 0.9
        elif user_text.endswith('?') or '؟' in user_text:
            actionability = 0.55
        elif reply_len > 80:
            actionability = 0.62

        scores = {
            'clarity': float(np.clip(clarity, 0.0, 1.0)),
            'tone_alignment': float(np.clip(tone_alignment, 0.0, 1.0)),
            'memory_alignment': float(np.clip(memory_alignment, 0.0, 1.0)),
            'actionability': float(np.clip(actionability, 0.0, 1.0)),
        }
        overall = float(np.mean(list(scores.values())))
        suggestions: list[str] = []
        if scores['memory_alignment'] < 0.7:
            suggestions.append('اربط الرد أكثر بالسياق أو الذكريات المرتبطة.')
        if scores['actionability'] < 0.7:
            suggestions.append('أضف خطوة عملية واحدة واضحة في النهاية.')
        if scores['clarity'] < 0.7:
            suggestions.append('اجعل الصياغة أبسط وأكثر مباشرة.')
        needs_refinement = overall < 0.74 or bool(suggestions)
        return {
            'enabled': True,
            'scores': scores,
            'overall_score': overall,
            'needs_refinement': needs_refinement,
            'suggestions': suggestions,
        }

    def _refine_reply(
        self,
        *,
        reply: str,
        critique: dict[str, Any],
        context_summary: dict[str, Any],
        related_memories: list[dict[str, Any]],
    ) -> str:
        refined = reply.strip()
        if critique.get('scores', {}).get('memory_alignment', 1.0) < 0.7:
            topics = context_summary.get('topics') or []
            if topics:
                refined += f' كمان واضح إن محور الكلام الحالي مرتبط بـ {", ".join(topics[:2])}.'
            elif related_memories:
                refined += f' وده متسق مع ذاكرة سابقة مرتبطة بعنوان: {related_memories[0].get("snippet", "سياق سابق")}.'
        if critique.get('scores', {}).get('actionability', 1.0) < 0.7:
            refined += ' الخطوة التالية: ابدأ بأصغر جزء قابل للتنفيذ، وراجعه بعد أول نتيجة فعلية.'
        return refined.strip()

    def _generate_final_reply(
        self,
        *,
        text: str,
        deterministic_reply: str,
        result: DecisionResult,
        related_memories: list[dict[str, Any]],
        context_summary: dict[str, Any],
        personality: dict[str, Any],
        affect_state: dict[str, Any],
        goals_overview: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        llm_result = self.llm_bridge.generate_reply(
            user_text=text,
            deterministic_reply=deterministic_reply,
            decision=result.decision,
            confidence=result.confidence,
            personality=personality,
            context_summary=context_summary,
            affect_state=affect_state,
            goals_overview=goals_overview,
            related_memories=related_memories,
        )
        reply_mode = 'llm_enhanced' if llm_result.used else 'deterministic'
        reply_text = llm_result.text
        self_critique = self._run_self_critic(
            user_text=text,
            reply=reply_text,
            related_memories=related_memories,
            context_summary=context_summary,
            personality=personality,
        )
        apply_self_critic = bool(self_critique.get('needs_refinement')) and not llm_result.used
        if apply_self_critic:
            reply_text = self._refine_reply(
                reply=reply_text,
                critique=self_critique,
                context_summary=context_summary,
                related_memories=related_memories,
            )
            reply_mode = f'{reply_mode}_self_critic'
        metadata = {
            'used_llm': llm_result.used,
            'provider': llm_result.provider,
            'model': llm_result.model,
            'latency_ms': llm_result.latency_ms,
            'reason': llm_result.reason,
            'self_critic_applied': apply_self_critic,
        }
        return reply_text, reply_mode, metadata, self_critique

    def llm_status(self) -> dict[str, Any]:
        return self.llm_bridge.status()

    def process_text(
        self,
        text: str,
        importance: float | None = None,
        *,
        image_refs: list[str] | None = None,
        audio_refs: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ChatCycleResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError('النص المطلوب تحليله لا يمكن أن يكون فارغاً')
        inferred_importance = self._estimate_importance(normalized_text, importance)
        anomaly_report = self.anomaly_detector.detect(
            normalized_text,
            image_refs=list(image_refs or []),
            audio_refs=list(audio_refs or []),
            tags=list(tags or []),
        )
        if anomaly_report.get('severity') == 'high':
            inferred_importance = float(np.clip(max(inferred_importance, 0.92), 0.0, 1.0))
        related_records = self.brain.memory.search_memories(
            normalized_text,
            limit=4,
            min_importance=0.15,
            strategy='hybrid',
        )
        related_memories = [self._memory_preview(record) for record in related_records]
        fake_image = self.text_to_vector(normalized_text, 64 * 64).reshape(64, 64)
        fake_audio = self.text_to_vector(f'{normalized_text}_audio', 1024)
        position = self.text_to_position(normalized_text)
        result = self.brain.live_cycle(
            fake_image,
            fake_audio,
            position,
            importance=inferred_importance,
            source='chat',
            extra_memory={
                'user_text': normalized_text,
                'channel': 'chat',
                'related_memory_keys': [item['key'] for item in related_memories],
                'inferred_importance': inferred_importance,
                'image_refs': list(image_refs or []),
                'audio_refs': list(audio_refs or []),
                'tags': list(tags or []),
                'anomaly_report': anomaly_report,
            },
        )
        context_summary = self.brain.context_snapshot(limit=4, query=normalized_text)
        personality = self.brain.get_personality_profile()
        affect_state = self.brain.get_affect_state()
        goals_overview = self.brain.goals_overview()
        user_model = self.brain.get_user_model()
        theory_of_mind = self.brain.last_tom_inference or self.brain.infer_user_mind(normalized_text)
        action_hook_summary = self.brain.last_action_hook_result
        deterministic_reply = self._build_reply(
            normalized_text,
            result,
            related_memories,
            context_summary,
            personality,
            affect_state,
            goals_overview,
        )
        final_reply, reply_mode, reply_metadata, self_critique = self._generate_final_reply(
            text=normalized_text,
            deterministic_reply=deterministic_reply,
            result=result,
            related_memories=related_memories,
            context_summary=context_summary,
            personality=personality,
            affect_state=affect_state,
            goals_overview=goals_overview,
        )
        if anomaly_report.get('suspicious'):
            self.brain.register_anomaly_event(
                source='chat',
                text=normalized_text,
                report=anomaly_report,
                related_memory_key=self.brain.last_memory_key,
            )
            if anomaly_report.get('severity') == 'high':
                final_reply = (
                    f'{final_reply} ملحوظة أمنية: رصدت مؤشرات غير معتادة في الطلب، '
                    'فتم الاحتفاظ بسجل تحذيري ومراعاة الرد بحذر.'
                ).strip()
        episode = self.brain.record_episode(
            text=normalized_text,
            image_refs=image_refs,
            audio_refs=audio_refs,
            tags=tags,
            importance=max(inferred_importance, 0.72),
            related_memory_keys=[item['key'] for item in related_memories if item.get('key')],
            metadata={
                'decision': result.decision,
                'decision_confidence': result.confidence,
                'reply_mode': reply_mode,
                'anomaly_report': anomaly_report,
            },
        )
        return ChatCycleResult(
            reply=final_reply,
            decision=result.decision,
            confidence=result.confidence,
            top_probabilities=result.top_probabilities,
            memory_key=self.brain.last_memory_key,
            related_memories=related_memories,
            inferred_importance=inferred_importance,
            context_summary=context_summary,
            personality=personality,
            affect_state=affect_state,
            goals_overview=goals_overview,
            user_model=user_model,
            theory_of_mind=theory_of_mind,
            action_hook_summary=action_hook_summary,
            reply_mode=reply_mode,
            reply_metadata=reply_metadata,
            deterministic_reply=deterministic_reply,
            self_critique=self_critique,
            episode_memory_key=episode.get('key'),
            episode_summary=episode,
            anomaly_report=anomaly_report,
        )
