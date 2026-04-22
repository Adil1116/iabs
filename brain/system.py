from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque, Optional
from uuid import uuid4
import json
import re
import time

import numpy as np

from brain.core.schemas import DecisionResult, GoalRecord, PersonalityProfile
from brain.executive import FrontalLobe
from brain.lobes.audio import TemporalLobe
from brain.lobes.navigation import EntorhinalCortex
from brain.lobes.vision import OccipitalLobe
from brain.memory import Hippocampus
from brain.agentic import ActionHookGateway, DreamEngine, UserModelEngine



class IntegratedArtificialBrain:
    """النظام الذي يجمع وحدات الرؤية والصوت والملاحة والذاكرة واتخاذ القرار."""

    _TOPIC_STOPWORDS = {
        'في', 'من', 'على', 'الى', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'انا', 'أنا', 'انت', 'أنت',
        'هو', 'هي', 'تم', 'بعد', 'قبل', 'كان', 'كانت', 'the', 'and', 'for', 'that', 'this', 'then', 'have',
        'want', 'need', 'goal', 'task', 'انا', 'عايز', 'اريد', 'محتاج', 'لازم',
    }
    _URGENT_KEYWORDS = {'عاجل', 'خطر', 'سريع', 'كارثة', 'urgent', 'asap', 'emergency', 'critical'}
    _GOAL_KEYWORDS = {'عايز', 'اريد', 'أريد', 'محتاج', 'لازم', 'هدف', 'goal', 'plan', 'مهمه', 'مهمة'}

    def __init__(
        self,
        seed: Optional[int] = 42,
        memory_capacity: int = 1000,
        storage_path: Optional[str | Path] = None,
        storage_dsn: str | None = None,
        state_path: Optional[str | Path] = None,
        autoload_state: bool = True,
        postgres_pool_min_size: int = 1,
        postgres_pool_max_size: int = 5,
        memory_encryption_key: str | None = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vision = OccipitalLobe(self.rng)
        self.audio = TemporalLobe(self.rng)
        self.navigation = EntorhinalCortex(self.rng)
        base_dir = Path(__file__).resolve().parent.parent / 'data'
        self.storage_path = Path(storage_path).expanduser().resolve() if storage_path else (None if storage_dsn else (base_dir / 'memory_store.sqlite'))
        self.storage_dsn = storage_dsn.strip() if isinstance(storage_dsn, str) and storage_dsn.strip() else None
        self.state_path = Path(state_path).expanduser().resolve() if state_path else (base_dir / 'brain_state.json')
        if self.storage_path is not None:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = Hippocampus(
            capacity=memory_capacity,
            storage_path=self.storage_path,
            storage_dsn=self.storage_dsn,
            autosave=True,
            autoload=True,
            postgres_pool_min_size=postgres_pool_min_size,
            postgres_pool_max_size=postgres_pool_max_size,
            encryption_key=memory_encryption_key,
        )
        self.executive = FrontalLobe(self.rng, input_size=512)
        self.state = 'مستيقظ'
        self.cycles_completed = 0
        self.last_decision: Optional[DecisionResult] = None
        self.last_memory_key: Optional[str] = None
        self.decision_history: Deque[dict[str, Any]] = deque(maxlen=50)
        self.learning_history: Deque[dict[str, Any]] = deque(maxlen=100)
        self.pending_feedback: dict[str, dict[str, Any]] = {}
        now = time.time()
        self.personality = PersonalityProfile(created_at=now, updated_at=now)
        self.context_window: Deque[dict[str, Any]] = deque(maxlen=12)
        self.self_improvement_history: Deque[dict[str, Any]] = deque(maxlen=50)
        self.goals: dict[str, GoalRecord] = {}
        self.sleep_reports: Deque[dict[str, Any]] = deque(maxlen=20)
        self.last_sleep_report: dict[str, Any] | None = None
        self.episode_history: Deque[dict[str, Any]] = deque(maxlen=50)
        self.last_episode_key: str | None = None
        self.anomaly_events: Deque[dict[str, Any]] = deque(maxlen=100)
        self.user_model: dict[str, Any] = UserModelEngine.empty_profile()
        self.action_hooks: dict[str, Any] = ActionHookGateway.empty_registry()
        self.last_tom_inference: dict[str, Any] | None = None
        self.dream_history: Deque[dict[str, Any]] = deque(maxlen=20)
        self.last_dream_report: dict[str, Any] | None = None
        self.last_action_hook_result: dict[str, Any] | None = None
        self.affect_state: dict[str, Any] = {
            'curiosity': 0.58,
            'stress': 0.22,
            'satisfaction': 0.48,
            'energy': 0.82,
            'updated_at': now,
            'mood_label': 'متوازن',
        }
        if autoload_state and self.state_path.exists():
            self.load_state()
        else:
            self._refresh_affect_state()

    def _validate_numeric_array(self, name: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != expected_shape:
            raise ValueError(f'{name} يجب أن تكون أبعاده {expected_shape} لكن تم استلام {value.shape}')
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{name} يحتوي على قيم غير صالحة مثل NaN أو Infinity')
        return value

    def _integrate_features(
        self,
        visual_features: np.ndarray,
        audio_features: np.ndarray,
        spatial_features: np.ndarray,
        target_size: int = 512,
    ) -> np.ndarray:
        combined = np.concatenate([visual_features, audio_features, spatial_features])
        current_size = combined.shape[0]
        if current_size < target_size:
            padding = np.zeros(target_size - current_size, dtype=np.float64)
            combined = np.concatenate([combined, padding])
        elif current_size > target_size:
            combined = combined[:target_size]
        return combined

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (PersonalityProfile, GoalRecord)):
            return asdict(value)
        if isinstance(value, dict):
            return {str(key): IntegratedArtificialBrain._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [IntegratedArtificialBrain._to_jsonable(item) for item in value]
        return value

    def _weights_path(self) -> Path:
        return self.state_path.with_name(f'{self.state_path.stem}_components.npz')

    def _save_component_weights(self) -> Path:
        weights_path = self._weights_path()
        np.savez_compressed(
            weights_path,
            vision_v1=self.vision.v1_weights,
            vision_v2=self.vision.v2_weights,
            audio_weights=self.audio.audio_weights,
            language_weights=self.audio.language_weights,
            navigation_centers=self.navigation.centers,
            executive_weights=self.executive.decision_weights,
        )
        return weights_path

    def _load_component_weights(self) -> None:
        weights_path = self._weights_path()
        if not weights_path.exists():
            return
        with np.load(weights_path) as payload:
            self.vision.v1_weights = np.asarray(payload['vision_v1'], dtype=np.float64)
            self.vision.v2_weights = np.asarray(payload['vision_v2'], dtype=np.float64)
            self.audio.audio_weights = np.asarray(payload['audio_weights'], dtype=np.float64)
            self.audio.language_weights = np.asarray(payload['language_weights'], dtype=np.float64)
            self.navigation.centers = np.asarray(payload['navigation_centers'], dtype=np.float64)
            self.executive.decision_weights = np.asarray(payload['executive_weights'], dtype=np.float64)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _derive_mood_label(self) -> str:
        curiosity = float(self.affect_state.get('curiosity', 0.5))
        stress = float(self.affect_state.get('stress', 0.2))
        satisfaction = float(self.affect_state.get('satisfaction', 0.5))
        energy = float(self.affect_state.get('energy', 0.5))
        if stress >= 0.72:
            return 'مضغوط'
        if energy <= 0.28:
            return 'مرهق'
        if curiosity >= 0.7 and energy >= 0.55:
            return 'فضولي'
        if satisfaction >= 0.7 and stress <= 0.3:
            return 'واثق'
        if satisfaction >= 0.58 and stress <= 0.4:
            return 'هادئ'
        return 'متوازن'

    def _refresh_affect_state(self) -> dict[str, Any]:
        self.affect_state['mood_label'] = self._derive_mood_label()
        self.affect_state['updated_at'] = time.time()
        return self.get_affect_state()

    def _shift_affect(
        self,
        *,
        curiosity: float = 0.0,
        stress: float = 0.0,
        satisfaction: float = 0.0,
        energy: float = 0.0,
    ) -> dict[str, Any]:
        self.affect_state['curiosity'] = self._clip01(float(self.affect_state.get('curiosity', 0.5)) + curiosity)
        self.affect_state['stress'] = self._clip01(float(self.affect_state.get('stress', 0.2)) + stress)
        self.affect_state['satisfaction'] = self._clip01(float(self.affect_state.get('satisfaction', 0.5)) + satisfaction)
        self.affect_state['energy'] = self._clip01(float(self.affect_state.get('energy', 0.7)) + energy)
        return self._refresh_affect_state()

    def get_affect_state(self) -> dict[str, Any]:
        return {
            'curiosity': float(self.affect_state.get('curiosity', 0.5)),
            'stress': float(self.affect_state.get('stress', 0.2)),
            'satisfaction': float(self.affect_state.get('satisfaction', 0.5)),
            'energy': float(self.affect_state.get('energy', 0.7)),
            'updated_at': float(self.affect_state.get('updated_at', time.time())),
            'mood_label': str(self.affect_state.get('mood_label', 'متوازن')),
        }

    def get_user_model(self) -> dict[str, Any]:
        return self._to_jsonable(self.user_model)

    def rebuild_user_model(self) -> dict[str, Any]:
        profile = UserModelEngine.empty_profile()
        for entry in list(self.context_window):
            if not isinstance(entry, dict):
                continue
            text = str(entry.get('user_text') or '').strip()
            if not text:
                continue
            profile = UserModelEngine.update_profile(
                profile,
                text=text,
                context_topics=list(entry.get('context_topics', [])),
                affect_state=entry.get('affect') or self.get_affect_state(),
            )
        self.user_model = profile
        return self.get_user_model()

    def infer_user_mind(self, text: str) -> dict[str, Any]:
        self.last_tom_inference = UserModelEngine.infer_tom(
            profile=self.user_model,
            text=text,
            recent_context=list(self.context_window)[-6:],
            affect_state=self.get_affect_state(),
        )
        return self._to_jsonable(self.last_tom_inference)

    def register_action_hook(
        self,
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
    ) -> dict[str, Any]:
        self.action_hooks, hook = ActionHookGateway.register_hook(
            self.action_hooks,
            name=name,
            event=event,
            action_type=action_type,
            target_url=target_url,
            method=method,
            headers=headers,
            payload_template=payload_template,
            keywords=keywords,
            cooldown_seconds=cooldown_seconds,
            active=active,
        )
        self.memory.store_memory(
            key=f'{hook["hook_id"]}_registered',
            data={'event': 'action_hook_registered', 'hook': hook},
            importance=0.74,
            source='action_hook',
        )
        return hook

    def get_action_hook(self, hook_id: str) -> dict[str, Any]:
        return self._to_jsonable(ActionHookGateway.get_hook(self.action_hooks, hook_id))

    def update_action_hook(self, hook_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        self.action_hooks, hook = ActionHookGateway.update_hook(self.action_hooks, hook_id, updates)
        self.memory.store_memory(
            key=f'{hook_id}_updated_{int(time.time() * 1000)}',
            data={'event': 'action_hook_updated', 'hook': hook},
            importance=0.72,
            source='action_hook',
        )
        return self._to_jsonable(hook)

    def delete_action_hook(self, hook_id: str) -> dict[str, Any]:
        self.action_hooks, hook = ActionHookGateway.delete_hook(self.action_hooks, hook_id)
        self.memory.store_memory(
            key=f'{hook_id}_deleted_{int(time.time() * 1000)}',
            data={'event': 'action_hook_deleted', 'hook': hook},
            importance=0.69,
            source='action_hook',
        )
        return self._to_jsonable(hook)

    def list_action_hooks(self, active_only: bool = False) -> list[dict[str, Any]]:
        return self._to_jsonable(ActionHookGateway.list_hooks(self.action_hooks, active_only=active_only))

    def action_hooks_overview(self) -> dict[str, Any]:
        return self._to_jsonable(ActionHookGateway.overview(self.action_hooks))

    def recent_action_hook_events(self, limit: int = 10) -> list[dict[str, Any]]:
        events = list((self.action_hooks or {}).get('events', []))
        return self._to_jsonable(events[-max(1, int(limit)):])

    def trigger_action_hooks(
        self,
        *,
        event: str,
        text: str = '',
        decision: str | None = None,
        topics: list[str] | None = None,
        dry_run: bool = True,
        allow_network: bool = False,
    ) -> dict[str, Any]:
        self.action_hooks, summary = ActionHookGateway.dispatch(
            self.action_hooks,
            event=event,
            text=text,
            decision=decision,
            topics=topics,
            dry_run=dry_run,
            allow_network=allow_network,
        )
        self.last_action_hook_result = summary
        if summary.get('matched_hooks'):
            self.memory.store_memory(
                key=f'action_hook_dispatch_{int(time.time() * 1000)}',
                data={'event': 'action_hook_dispatch', 'summary': summary},
                importance=0.7,
                source='action_hook',
            )
        return self._to_jsonable(summary)

    def run_dream_engine(self, trigger: str = 'manual_dream') -> dict[str, Any]:
        report = DreamEngine.synthesize(
            trigger=trigger,
            context_window=list(self.context_window)[-10:],
            goals=[asdict(goal) for goal in self.goals.values() if goal.status == 'active'],
            affect_state=self.get_affect_state(),
        )
        self.last_dream_report = report
        self.dream_history.append(report)
        self.memory.store_memory(
            key=f'dream_engine_{int(report["timestamp"] * 1000)}',
            data=report,
            importance=0.79,
            source='dream_engine',
        )
        return self._to_jsonable(report)

    def _record_decision(self, result: DecisionResult, source: str) -> None:
        self.decision_history.append(
            {
                'decision': result.decision,
                'decision_index': result.decision_index,
                'confidence': float(result.confidence),
                'timestamp': time.time(),
                'source': source,
            }
        )

    def _remember_feedback_context(self, memory_key: str, integrated_features: np.ndarray, result: DecisionResult, source: str) -> None:
        self.pending_feedback[memory_key] = {
            'integrated_features': self._to_jsonable(integrated_features),
            'decision_index': int(result.decision_index),
            'decision': result.decision,
            'confidence': float(result.confidence),
            'source': source,
            'created_at': time.time(),
        }
        if len(self.pending_feedback) > 200:
            oldest_key = min(self.pending_feedback.items(), key=lambda item: item[1].get('created_at', 0.0))[0]
            self.pending_feedback.pop(oldest_key, None)

    def _cleanup_pending_feedback(self, max_age_seconds: float = 3600 * 24 * 3) -> int:
        now = time.time()
        expired = [
            key for key, value in self.pending_feedback.items()
            if now - float(value.get('created_at', now)) > max_age_seconds
        ]
        for key in expired:
            self.pending_feedback.pop(key, None)
        return len(expired)

    def _extract_topics(self, texts: list[str], limit: int = 5) -> list[str]:
        counter: Counter[str] = Counter()
        for text in texts:
            for token in re.findall(r'\w+', str(text).lower(), flags=re.UNICODE):
                if len(token) < 3 or token in self._TOPIC_STOPWORDS:
                    continue
                counter[token] += 1
        return [token for token, _ in counter.most_common(max(1, int(limit)))]

    def _extract_text_fragments(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, np.integer, np.floating)):
            return [str(value)]
        if isinstance(value, dict):
            parts: list[str] = []
            for key, item in value.items():
                parts.append(str(key))
                parts.extend(self._extract_text_fragments(item))
            return parts
        if isinstance(value, (list, tuple, set, deque)):
            parts: list[str] = []
            for item in value:
                parts.extend(self._extract_text_fragments(item))
            return parts
        return [str(value)]

    def _goal_keywords_from_text(self, title: str, description: str = '', extra_keywords: list[str] | None = None) -> list[str]:
        fragments = [title or '', description or '']
        ordered: list[str] = []
        for token in extra_keywords or []:
            normalized = self.memory._normalize_text(token)
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        tokens = self._extract_topics(fragments, limit=12)
        for token in tokens:
            normalized = self.memory._normalize_text(token)
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        return ordered[:8]

    def create_goal(
        self,
        title: str,
        description: str = '',
        *,
        priority: float = 0.5,
        target_keywords: list[str] | None = None,
        status: str = 'active',
    ) -> dict[str, Any]:
        now = time.time()
        goal = GoalRecord(
            goal_id=f'goal_{uuid4().hex[:12]}',
            title=title.strip(),
            description=description.strip(),
            status=status.strip().lower() or 'active',
            priority=self._clip01(priority),
            progress=0.0,
            target_keywords=self._goal_keywords_from_text(title, description, target_keywords),
            created_at=now,
            updated_at=now,
        )
        self.goals[goal.goal_id] = goal
        goal_payload = asdict(goal)
        self.memory.store_memory(
            key=f'{goal.goal_id}_created',
            data={'goal': goal_payload, 'event': 'goal_created'},
            importance=max(0.72, goal.priority),
            source='goal_system',
        )
        return goal_payload

    def update_goal(self, goal_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        goal = self.goals.get(goal_id)
        if goal is None:
            raise ValueError('الهدف المطلوب غير موجود')
        if 'title' in updates and isinstance(updates['title'], str) and updates['title'].strip():
            goal.title = updates['title'].strip()
        if 'description' in updates and isinstance(updates['description'], str):
            goal.description = updates['description'].strip()
        if 'status' in updates and isinstance(updates['status'], str) and updates['status'].strip():
            goal.status = updates['status'].strip().lower()
        if 'priority' in updates and updates['priority'] is not None:
            goal.priority = self._clip01(float(updates['priority']))
        if 'progress' in updates and updates['progress'] is not None:
            goal.progress = self._clip01(float(updates['progress']))
        if 'note' in updates and isinstance(updates['note'], str) and updates['note'].strip():
            goal.notes.append(updates['note'].strip())
            goal.notes = goal.notes[-10:]
        if 'target_keywords' in updates and isinstance(updates['target_keywords'], list):
            goal.target_keywords = self._goal_keywords_from_text(goal.title, goal.description, updates['target_keywords'])
        if goal.progress >= 1.0 and goal.status == 'active':
            goal.status = 'completed'
        goal.updated_at = time.time()
        self.memory.store_memory(
            key=f'{goal.goal_id}_updated_{int(goal.updated_at * 1000)}',
            data={'goal': asdict(goal), 'event': 'goal_updated'},
            importance=max(0.68, goal.priority),
            source='goal_system',
        )
        return asdict(goal)

    def list_goals(self, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        goals = list(self.goals.values())
        if status:
            goals = [goal for goal in goals if goal.status == status.strip().lower()]
        goals.sort(key=lambda item: (item.status != 'active', item.priority, item.updated_at), reverse=True)
        return [asdict(goal) for goal in goals[: max(1, int(limit))]]

    def goals_overview(self) -> dict[str, Any]:
        goals = list(self.goals.values())
        counts = Counter(goal.status for goal in goals)
        active_goals = [goal for goal in goals if goal.status == 'active']
        active_goals.sort(key=lambda item: (item.priority, item.progress, item.updated_at), reverse=True)
        return {
            'total': len(goals),
            'counts': dict(counts),
            'top_active_goals': [asdict(goal) for goal in active_goals[:3]],
            'completed_ratio': (sum(1 for goal in goals if goal.status == 'completed') / len(goals)) if goals else 0.0,
        }

    def _goal_signal_strength(self, text_blob: str, goal: GoalRecord) -> tuple[float, list[str]]:
        normalized_blob = self.memory._normalize_text(text_blob)
        matched = [keyword for keyword in goal.target_keywords if keyword and self.memory._normalize_text(keyword) in normalized_blob]
        if not matched:
            return 0.0, []
        score = min(1.0, (len(matched) / max(1, len(goal.target_keywords))) + (goal.priority * 0.25))
        return score, matched

    def _auto_create_goal_from_text(self, text: str) -> dict[str, Any] | None:
        lowered = str(text).lower().strip()
        if not lowered:
            return None
        if not any(token in lowered for token in {item.lower() for item in self._GOAL_KEYWORDS}):
            return None
        existing_titles = {goal.title.strip().lower() for goal in self.goals.values()}
        compact_title = re.sub(r'\s+', ' ', str(text).strip())
        compact_title = compact_title[:90]
        if compact_title.lower() in existing_titles:
            return None
        return self.create_goal(
            title=compact_title,
            description='تم توليد الهدف تلقائياً من محادثة المستخدم.',
            priority=0.62,
            target_keywords=self._extract_topics([text], limit=6),
        )

    def _update_goals_from_payload(self, memory_key: str, memory_payload: dict[str, Any], importance: float) -> list[dict[str, Any]]:
        fragments = self._extract_text_fragments(memory_payload)
        text_blob = ' '.join(fragments)
        if not text_blob.strip():
            return []
        created_goal = None
        if 'user_text' in memory_payload:
            created_goal = self._auto_create_goal_from_text(str(memory_payload.get('user_text', '')))
        updates: list[dict[str, Any]] = []
        now = time.time()
        for goal in self.goals.values():
            if goal.status != 'active':
                continue
            strength, matched_keywords = self._goal_signal_strength(text_blob, goal)
            if strength <= 0.0:
                continue
            increment = min(0.22, 0.04 + (strength * 0.12) + (importance * 0.06))
            goal.progress = self._clip01(goal.progress + increment)
            goal.signal_count += 1
            goal.last_signal_at = now
            goal.updated_at = now
            note = f'إشارة من {memory_key}: {", ".join(matched_keywords[:4])}'
            goal.notes.append(note)
            goal.notes = goal.notes[-10:]
            if goal.progress >= 1.0:
                goal.status = 'completed'
            updates.append(
                {
                    'goal_id': goal.goal_id,
                    'title': goal.title,
                    'progress': goal.progress,
                    'status': goal.status,
                    'matched_keywords': matched_keywords,
                }
            )
        if created_goal is not None:
            updates.append({'auto_created_goal': created_goal})
        return updates

    def _capture_context(self, memory_key: str, source: str, memory_payload: dict[str, Any]) -> None:
        entry = {
            'memory_key': memory_key,
            'timestamp': time.time(),
            'source': source,
            'decision': memory_payload.get('decision'),
            'confidence': float(memory_payload.get('confidence', 0.0)),
            'user_text': memory_payload.get('user_text'),
            'related_memory_keys': memory_payload.get('related_memory_keys', []),
            'context_topics': memory_payload.get('context_topics', []),
            'affect': memory_payload.get('affect_state'),
            'goal_updates': memory_payload.get('goal_updates', []),
        }
        self.context_window.append(entry)

    def _update_affect_from_interaction(self, memory_payload: dict[str, Any], importance: float) -> dict[str, Any]:
        fragments = ' '.join(self._extract_text_fragments(memory_payload)).lower()
        curiosity_delta = 0.04 if '?' in fragments or '؟' in fragments else 0.0
        if any(token in fragments for token in ['استكشف', 'explore', 'learn', 'تعلم', 'بحث', 'ابحث']):
            curiosity_delta += 0.04
        stress_delta = 0.06 if any(keyword in fragments for keyword in self._URGENT_KEYWORDS) else -0.015
        satisfaction_delta = 0.02 if importance >= 0.7 else 0.0
        energy_delta = -0.03 - (importance * 0.03)
        if memory_payload.get('decision') == 'نم':
            energy_delta += 0.04
            stress_delta -= 0.03
        return self._shift_affect(
            curiosity=curiosity_delta,
            stress=stress_delta,
            satisfaction=satisfaction_delta,
            energy=energy_delta,
        )

    def _update_affect_from_feedback(self, reward: float) -> dict[str, Any]:
        reward_value = float(np.clip(reward, -1.0, 1.0))
        return self._shift_affect(
            satisfaction=reward_value * 0.18,
            stress=-reward_value * 0.14,
            curiosity=0.03 if reward_value < 0 else 0.0,
            energy=-0.01 if reward_value < 0 else 0.02,
        )

    def get_personality_profile(self) -> dict[str, Any]:
        return asdict(self.personality)

    def update_personality_profile(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            'display_name', 'tone', 'style', 'identity_statement', 'behavioral_rules', 'preferred_language'
        }
        for key, value in updates.items():
            if key not in allowed_fields:
                continue
            if key == 'behavioral_rules' and isinstance(value, list):
                cleaned = [str(item).strip() for item in value if str(item).strip()]
                if cleaned:
                    self.personality.behavioral_rules = cleaned[:10]
                continue
            if isinstance(value, str) and value.strip():
                setattr(self.personality, key, value.strip())
        self.personality.updated_at = time.time()
        return self.get_personality_profile()

    def context_snapshot(self, limit: int = 5, query: str | None = None) -> dict[str, Any]:
        recent_entries = list(self.context_window)[-max(1, int(limit)):]
        related = []
        if query:
            related = [
                {
                    'key': record.key,
                    'source': record.source,
                    'importance': record.importance,
                    'timestamp': record.timestamp,
                }
                for record in self.memory.search_memories(query, limit=min(5, limit), min_importance=0.1, strategy='hybrid')
            ]
        text_fragments = [str(item.get('user_text') or item.get('decision') or '') for item in recent_entries]
        for item in related:
            text_fragments.append(str(item.get('key', '')))
        return {
            'window_size': len(self.context_window),
            'recent_items': recent_entries,
            'related_memories': related,
            'topics': self._extract_topics(text_fragments),
            'affect': self.get_affect_state(),
            'goals': self.goals_overview(),
            'recent_episodes': list(self.episode_history)[-min(max(1, int(limit)), 3):],
        }

    def record_episode(
        self,
        *,
        text: str,
        image_refs: list[str] | None = None,
        audio_refs: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.75,
        related_memory_keys: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = 'episode',
    ) -> dict[str, Any]:
        clean_text = str(text).strip()
        if not clean_text:
            raise ValueError('نص الحلقة الزمنية لا يمكن أن يكون فارغاً')
        previous_episode_key = self.last_episode_key
        summary = clean_text if len(clean_text) <= 160 else clean_text[:157] + '...'
        topics = self._extract_topics([clean_text] + [str(tag) for tag in (tags or [])], limit=4)
        episode_key = f'episode_{self.cycles_completed + 1}_{int(time.time() * 1000)}'
        payload = {
            'type': 'episodic_memory',
            'text': clean_text,
            'summary': summary,
            'image_refs': [str(item).strip() for item in (image_refs or []) if str(item).strip()],
            'audio_refs': [str(item).strip() for item in (audio_refs or []) if str(item).strip()],
            'tags': [str(item).strip() for item in (tags or []) if str(item).strip()],
            'related_memory_keys': [str(item).strip() for item in (related_memory_keys or []) if str(item).strip()],
            'temporal_link': {
                'previous_episode_key': previous_episode_key,
                'cycle_number': self.cycles_completed,
                'recorded_at': time.time(),
            },
            'topics': topics,
            'metadata': self._to_jsonable(metadata or {}),
        }
        stored = self.memory.store_memory(
            key=episode_key,
            data=payload,
            importance=float(np.clip(importance, 0.0, 1.0)),
            source=source,
        )
        episode_summary = {
            'key': stored.key,
            'source': stored.source,
            'importance': stored.importance,
            'timestamp': stored.timestamp,
            'summary': summary,
            'topics': topics,
            'modalities': {
                'text': True,
                'image': bool(payload['image_refs']),
                'audio': bool(payload['audio_refs']),
            },
            'temporal_link': payload['temporal_link'],
            'related_memory_keys': payload['related_memory_keys'],
        }
        self.last_episode_key = episode_key
        self.episode_history.append(episode_summary)
        self._capture_context(
            episode_key,
            source,
            {
                'user_text': clean_text,
                'decision': metadata.get('decision') if isinstance(metadata, dict) else None,
                'confidence': float(metadata.get('decision_confidence', 0.0)) if isinstance(metadata, dict) else 0.0,
                'related_memory_keys': payload['related_memory_keys'],
                'context_topics': topics,
                'affect_state': self.get_affect_state(),
                'goal_updates': [],
            },
        )
        return episode_summary

    def search_episodes(self, query: str, limit: int = 5, min_importance: float = 0.0) -> list[dict[str, Any]]:
        matches = self.memory.search_memories(
            query,
            limit=limit,
            min_importance=min_importance,
            source='episode',
            strategy='hybrid',
        )
        results: list[dict[str, Any]] = []
        for record in matches:
            data = record.data if isinstance(record.data, dict) else {'summary': str(record.data)}
            results.append(
                {
                    'key': record.key,
                    'source': record.source,
                    'importance': record.importance,
                    'timestamp': record.timestamp,
                    'summary': data.get('summary') or data.get('text') or '',
                    'topics': data.get('topics', []),
                    'modalities': {
                        'text': bool(data.get('text')),
                        'image': bool(data.get('image_refs')),
                        'audio': bool(data.get('audio_refs')),
                    },
                    'temporal_link': data.get('temporal_link', {}),
                    'related_memory_keys': data.get('related_memory_keys', []),
                }
            )
        return results

    def live_cycle(
        self,
        visual_input: np.ndarray,
        audio_input: np.ndarray,
        position: np.ndarray,
        importance: Optional[float] = None,
        source: str = 'live_cycle',
        extra_memory: Optional[dict[str, Any]] = None,
    ) -> DecisionResult:
        if self.state != 'مستيقظ':
            raise RuntimeError("لا يمكن تنفيذ دورة حياة بينما النظام ليس في حالة 'مستيقظ'.")
        visual_input = self._validate_numeric_array('visual_input', visual_input, self.vision.resolution)
        audio_input = self._validate_numeric_array('audio_input', audio_input, (self.audio.audio_channels,))
        position = self._validate_numeric_array('position', position, (2,))
        visual_features = self.vision.process_image(visual_input)
        audio_features = self.audio.process_audio(audio_input)
        spatial_features = self.navigation.get_spatial_activity(position)
        integrated_features = self._integrate_features(
            visual_features,
            audio_features,
            spatial_features,
            target_size=self.executive.input_size,
        )
        decision_result = self.executive.make_decision(integrated_features)
        memory_key = f'cycle_{self.cycles_completed + 1}_{int(time.time() * 1000)}'
        if importance is None:
            importance = float(np.clip(decision_result.confidence, 0.1, 1.0))
        memory_payload: dict[str, Any] = {
            'decision': decision_result.decision,
            'decision_index': decision_result.decision_index,
            'confidence': decision_result.confidence,
            'position': np.asarray(position, dtype=float).tolist(),
            'top_probabilities': decision_result.top_probabilities,
            'cycle_number': self.cycles_completed + 1,
            'personality_name': self.personality.display_name,
            'personality_tone': self.personality.tone,
        }
        if extra_memory:
            memory_payload.update(self._to_jsonable(extra_memory))
        user_text = str(memory_payload.get('user_text') or '').strip()
        if user_text:
            self.user_model = UserModelEngine.update_profile(
                self.user_model,
                text=user_text,
                context_topics=list(memory_payload.get('context_topics', [])),
                affect_state=self.get_affect_state(),
            )
            memory_payload['user_model'] = self.get_user_model()
            memory_payload['theory_of_mind'] = self.infer_user_mind(user_text)
        affect_snapshot = self._update_affect_from_interaction(memory_payload, float(importance))
        goal_updates = self._update_goals_from_payload(memory_key, memory_payload, float(importance))
        memory_payload['affect_state'] = affect_snapshot
        memory_payload['goal_updates'] = goal_updates
        if user_text:
            action_hook_summary = self.trigger_action_hooks(
                event='chat.message',
                text=user_text,
                decision=decision_result.decision,
                topics=list(memory_payload.get('context_topics', [])),
                dry_run=True,
                allow_network=False,
            )
            memory_payload['action_hook_summary'] = action_hook_summary
        self.memory.store_memory(
            key=memory_key,
            data=memory_payload,
            importance=importance,
            source=source,
        )
        self._remember_feedback_context(memory_key, integrated_features, decision_result, source)
        self._capture_context(memory_key, source, memory_payload)
        self.cycles_completed += 1
        self.last_decision = decision_result
        self.last_memory_key = memory_key
        self._record_decision(decision_result, source)
        return decision_result

    def apply_feedback(
        self,
        reward: float,
        memory_key: str | None = None,
        *,
        feedback_text: str | None = None,
        source: str = 'feedback',
    ) -> dict[str, Any]:
        resolved_memory_key = memory_key or self.last_memory_key
        if not resolved_memory_key:
            raise ValueError('لا يوجد memory_key متاح لتطبيق التعلم عليه')
        context = self.pending_feedback.get(resolved_memory_key)
        if context is None:
            raise ValueError('هذا القرار غير متاح للتعلم حالياً أو انتهت صلاحية سياقه')
        integrated_features = np.asarray(context['integrated_features'], dtype=np.float64)
        reward_value = float(np.clip(reward, -1.0, 1.0))
        learning_event = self.executive.learn(
            integrated_features,
            int(context['decision_index']),
            reward_value,
            reason=feedback_text or source,
        )
        affect_snapshot = self._update_affect_from_feedback(reward_value)
        recalled = self.memory.recall(resolved_memory_key)
        goal_feedback_updates = self._update_goals_from_payload(
            f'feedback_{resolved_memory_key}',
            {'recalled': recalled, 'feedback_text': feedback_text or '', 'reward': reward_value},
            min(1.0, abs(reward_value) + 0.2),
        )
        result = {
            'memory_key': resolved_memory_key,
            'reward': reward_value,
            'decision': context['decision'],
            'decision_index': int(context['decision_index']),
            'confidence_at_decision': float(context.get('confidence', 0.0)),
            'source': source,
            'feedback_text': feedback_text,
            'applied_at': time.time(),
            'learning': learning_event,
            'affect_state': affect_snapshot,
            'goal_updates': goal_feedback_updates,
        }
        self.learning_history.append(result)
        feedback_memory_key = f'feedback_{resolved_memory_key}_{int(time.time() * 1000)}'
        self.memory.store_memory(
            key=feedback_memory_key,
            data={
                'target_memory_key': resolved_memory_key,
                'reward': reward_value,
                'decision': context['decision'],
                'decision_index': int(context['decision_index']),
                'feedback_text': feedback_text,
                'learning_event': learning_event,
                'affect_state': affect_snapshot,
                'goal_updates': goal_feedback_updates,
            },
            importance=float(np.clip(max(0.6, abs(reward_value)), 0.0, 1.0)),
            source=source,
        )
        auto_self_improvement = None
        if self.executive.total_feedback_events and self.executive.total_feedback_events % 3 == 0:
            auto_self_improvement = self.run_self_improvement(trigger='auto_feedback_loop')
        result['auto_self_improvement'] = auto_self_improvement
        return result

    def run_self_improvement(self, trigger: str = 'manual') -> dict[str, Any]:
        recent_learning = list(self.learning_history)[-10:]
        avg_reward = float(np.mean([item['reward'] for item in recent_learning])) if recent_learning else 0.0
        old_lr = float(self.executive.learning_rate)
        if avg_reward < -0.2:
            new_lr = self.executive.set_learning_rate(old_lr * 0.9)
        elif avg_reward > 0.35:
            new_lr = self.executive.set_learning_rate(old_lr * 1.05)
        else:
            new_lr = self.executive.set_learning_rate(old_lr)
        promoted = self.memory.consolidate_recent_memories(min_importance=0.72)
        cleaned_feedback_contexts = self._cleanup_pending_feedback()
        dominant_decisions = Counter(item['decision'] for item in recent_learning if 'decision' in item).most_common(3)
        stagnant_goals = [goal.title for goal in self.goals.values() if goal.status == 'active' and goal.progress < 0.35][:3]
        recommendations: list[str] = []
        if avg_reward < 0:
            recommendations.append('قلّل الثقة في الأنماط التي تلقت تقييمات سلبية متكررة.')
        if promoted > 0:
            recommendations.append('تم ترقية ذكريات مهمة إلى الذاكرة الطويلة لتحسين الاستدعاء لاحقاً.')
        if stagnant_goals:
            recommendations.append(f'راجع الأهداف البطيئة: {", ".join(stagnant_goals)}.')
        if not recommendations:
            recommendations.append('الأداء مستقر حالياً؛ استمر في جمع تغذية راجعة متنوعة.')
        event = {
            'trigger': trigger,
            'timestamp': time.time(),
            'average_recent_reward': avg_reward,
            'learning_rate_before': old_lr,
            'learning_rate_after': new_lr,
            'promoted_memories': promoted,
            'cleaned_pending_feedback': cleaned_feedback_contexts,
            'dominant_decisions': dominant_decisions,
            'recommendations': recommendations,
            'goal_summary': self.goals_overview(),
            'affect_state': self.get_affect_state(),
        }
        self.self_improvement_history.append(event)
        self.memory.store_memory(
            key=f'self_improvement_{int(event["timestamp"] * 1000)}',
            data=event,
            importance=0.82,
            source='self_improvement',
        )
        return event

    def run_sleep_cycle(self, trigger: str = 'manual_sleep') -> dict[str, Any]:
        affect_before = self.get_affect_state()
        recent_context = list(self.context_window)[-10:]
        recent_texts = [str(item.get('user_text') or item.get('decision') or '') for item in recent_context]
        dream_engine_report = self.run_dream_engine(trigger=f'{trigger}_dream_engine')
        dream_topics = dream_engine_report.get('dream_topics') or self._extract_topics(recent_texts, limit=6)
        promoted = self.memory.consolidate_recent_memories(min_importance=0.68)
        self_improvement = self.run_self_improvement(trigger='sleep_cycle')
        self._shift_affect(curiosity=-0.04, stress=-0.14, satisfaction=0.05, energy=0.22)
        report = {
            'trigger': trigger,
            'timestamp': time.time(),
            'dream_topics': dream_topics,
            'dream_engine': dream_engine_report,
            'promoted_memories': promoted,
            'goal_summary': self.goals_overview(),
            'affect_before': affect_before,
            'affect_after': self.get_affect_state(),
            'self_improvement': self_improvement,
        }
        self.last_sleep_report = report
        self.sleep_reports.append(report)
        self.memory.store_memory(
            key=f'sleep_cycle_{int(report["timestamp"] * 1000)}',
            data=report,
            importance=0.8,
            source='sleep_cycle',
        )
        return report

    def recent_sleep_reports(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self.sleep_reports)[-max(1, int(limit)):]

    def register_anomaly_event(
        self,
        *,
        source: str,
        text: str,
        report: dict[str, Any],
        related_memory_key: str | None = None,
    ) -> dict[str, Any]:
        event = {
            'event_id': f'anomaly_{int(time.time() * 1000)}',
            'timestamp': time.time(),
            'source': source,
            'score': float(report.get('score', 0.0)),
            'severity': str(report.get('severity', 'none')),
            'suspicious': bool(report.get('suspicious', False)),
            'recommended_action': str(report.get('recommended_action', 'allow')),
            'triggers': [str(item) for item in report.get('triggers', []) if str(item).strip()],
            'summary': (str(text).strip()[:157] + '...') if len(str(text).strip()) > 160 else str(text).strip(),
            'related_memory_key': related_memory_key,
        }
        self.anomaly_events.append(event)
        if event['suspicious']:
            self.memory.store_memory(
                key=f"{event['event_id']}_memory",
                data={
                    'type': 'anomaly_event',
                    'report': self._to_jsonable(report),
                    'summary': event['summary'],
                    'related_memory_key': related_memory_key,
                },
                importance=min(1.0, max(0.76, event['score'])),
                source='anomaly_detector',
            )
            stress_bump = min(0.18, event['score'] * 0.18)
            self._shift_affect(stress=stress_bump, curiosity=-min(0.08, event['score'] * 0.06), satisfaction=-min(0.06, event['score'] * 0.04))
        return event

    def recent_anomalies(self, limit: int = 10) -> list[dict[str, Any]]:
        return list(self.anomaly_events)[-max(1, int(limit)):]

    def anomaly_overview(self) -> dict[str, Any]:
        events = list(self.anomaly_events)
        suspicious_events = [item for item in events if item.get('suspicious')]
        high_events = [item for item in events if item.get('severity') == 'high']
        average_score = float(np.mean([float(item.get('score', 0.0)) for item in events])) if events else 0.0
        return {
            'total_events': len(events),
            'suspicious_events': len(suspicious_events),
            'high_severity_events': len(high_events),
            'average_score': average_score,
            'recent': self.recent_anomalies(limit=5),
        }

    def dashboard_snapshot(self) -> dict[str, Any]:
        memory_stats = self.memory.stats()
        active_goals = [goal for goal in self.goals.values() if goal.status == 'active']
        gauges = {
            'memory_load_ratio': round(memory_stats.get('total_records', 0) / max(1, self.memory.capacity), 4),
            'active_goal_ratio': round(len(active_goals) / max(1, len(self.goals) or 1), 4),
            'anomaly_pressure': round(min(1.0, self.anomaly_overview().get('average_score', 0.0) + (0.1 * self.anomaly_overview().get('high_severity_events', 0))), 4),
            'energy_level': round(float(self.affect_state.get('energy', 0.0)), 4),
        }
        return {
            'state': self.state,
            'cycles_completed': self.cycles_completed,
            'personality': self.get_personality_profile(),
            'affect': self.get_affect_state(),
            'goals': self.goals_overview(),
            'memory': memory_stats,
            'anomalies': self.anomaly_overview(),
            'gauges': gauges,
            'user_model': self.get_user_model(),
            'action_hooks': self.action_hooks_overview(),
            'last_dream_report': self.last_dream_report,
            'recent_episodes': list(self.episode_history)[-5:],
            'recent_context': list(self.context_window)[-5:],
            'recent_learning': list(self.learning_history)[-5:],
        }

    def next_actions_snapshot(self, limit: int = 5) -> dict[str, Any]:
        resolved_limit = max(1, int(limit))
        goals = self.goals_overview()
        top_goals = goals.get('top_active_goals') or []
        anomalies = self.anomaly_overview()
        affect = self.get_affect_state()
        user_model = self.get_user_model()
        actions: list[dict[str, Any]] = []

        if anomalies.get('high_severity_events', 0) > 0 or anomalies.get('suspicious_events', 0) > 0:
            severity = 'عالية' if anomalies.get('high_severity_events', 0) > 0 else 'متوسطة'
            actions.append(
                {
                    'action_id': 'review_anomalies',
                    'title': 'راجع التنبيهات الشاذة الأخيرة',
                    'focus': 'prompt_safety',
                    'priority': 0.96 if anomalies.get('high_severity_events', 0) > 0 else 0.88,
                    'source': 'anomaly_overview',
                    'reason': f'تم رصد {anomalies.get("suspicious_events", 0)} حدث شاذ وآخر مستوى خطورة {severity}.',
                    'done_when': 'تصنيف أحدث التنبيهات وتأكيد الإجراء المناسب لكل حالة.',
                }
            )

        if top_goals:
            top_goal = top_goals[0]
            progress_pct = round(float(top_goal.get('progress', 0.0)) * 100, 1)
            priority = min(0.94, 0.65 + (float(top_goal.get('priority', 0.5)) * 0.25) + (0.1 if progress_pct < 40 else 0.0))
            actions.append(
                {
                    'action_id': 'advance_top_goal',
                    'title': f'ادفع الهدف الأعلى أولوية: {top_goal.get("title", "هدف بدون عنوان")}',
                    'focus': top_goal.get('title', 'goal'),
                    'priority': round(priority, 4),
                    'source': 'goal_system',
                    'reason': f'الهدف النشط الأهم تقدمه الحالي {progress_pct}% ويحتاج دفعة تنفيذية مباشرة.',
                    'done_when': 'إضافة خطوة تنفيذية واحدة أو رفع التقدم الفعلي للهدف.',
                }
            )

        if float(affect.get('energy', 0.0)) < 0.35:
            actions.append(
                {
                    'action_id': 'recover_energy',
                    'title': 'خفف الحمل ونفّذ دورة استعادة قصيرة',
                    'focus': affect.get('mood_label', 'state'),
                    'priority': 0.82,
                    'source': 'affect_state',
                    'reason': f'مستوى الطاقة الحالي منخفض ({round(float(affect.get("energy", 0.0)) * 100, 1)}%).',
                    'done_when': 'تنفيذ sleep cycle أو تقليل حجم المهمة التالية قبل الاستكمال.',
                }
            )

        predicted_needs = [str(item) for item in user_model.get('predicted_needs', []) if str(item).strip()]
        if predicted_needs:
            actions.append(
                {
                    'action_id': 'serve_predicted_need',
                    'title': f'حضّر المخرج المتوقع التالي: {predicted_needs[0]}',
                    'focus': predicted_needs[0],
                    'priority': 0.78,
                    'source': 'user_model',
                    'reason': 'نموذج المستخدم يشير إلى أن هذا هو أقرب مخرج عملي متوقع حالياً.',
                    'done_when': 'توليد مخرج مباشر يلبي الاحتياج المتوقع بدون خطوات زائدة.',
                }
            )

        if self.self_improvement_history:
            latest = list(self.self_improvement_history)[-1]
            recommendations = [str(item) for item in latest.get('recommendations', []) if str(item).strip()]
            if recommendations:
                actions.append(
                    {
                        'action_id': 'apply_self_improvement',
                        'title': 'نفّذ أحدث توصية تحسين ذاتي',
                        'focus': recommendations[0],
                        'priority': 0.74,
                        'source': 'self_improvement',
                        'reason': recommendations[0],
                        'done_when': 'تحويل التوصية إلى تعديل أو سلوك عملي يمكن مراجعته.',
                    }
                )

        if not actions:
            actions.append(
                {
                    'action_id': 'gather_more_signal',
                    'title': 'اجمع إشارة إضافية قبل التوسّع',
                    'focus': 'context',
                    'priority': 0.6,
                    'source': 'system',
                    'reason': 'لا توجد ضغوط عالية حالياً، والأفضل جمع سياق أوضح قبل أي تصعيد.',
                    'done_when': 'استقبال مدخل جديد أو إنشاء هدف نشط أو تسجيل تغذية راجعة.',
                }
            )

        actions.sort(key=lambda item: float(item.get('priority', 0.0)), reverse=True)
        trimmed = actions[:resolved_limit]
        return {
            'summary': {
                'state': self.state,
                'top_priority': trimmed[0]['title'] if trimmed else None,
                'active_goals': len(top_goals),
                'suspicious_events': anomalies.get('suspicious_events', 0),
                'energy_level': round(float(affect.get('energy', 0.0)), 4),
                'suggested_mode': 'stabilize' if anomalies.get('high_severity_events', 0) > 0 else ('focus' if top_goals else 'explore'),
            },
            'actions': trimmed,
        }

    def roadmap_snapshot(self, limit: int = 4, include_completed: bool = False) -> dict[str, Any]:
        resolved_limit = max(1, int(limit))
        goals = self.list_goals(limit=12)
        active_goals = [goal for goal in goals if goal.get('status') == 'active']
        completed_goals = [goal for goal in goals if goal.get('status') == 'completed']
        anomalies = self.anomaly_overview()
        next_actions = self.next_actions_snapshot(limit=max(3, resolved_limit)).get('actions', [])
        user_model = self.get_user_model()
        predicted_needs = [str(item) for item in user_model.get('predicted_needs', []) if str(item).strip()]
        hooks_summary = self.action_hooks_overview()
        context_topics = self.context_snapshot(limit=5).get('topics', [])

        phases: list[dict[str, Any]] = []
        if anomalies.get('suspicious_events', 0) > 0:
            phases.append(
                {
                    'phase_id': 'stabilize_signal',
                    'title': 'تثبيت الإشارات وتنظيف المخاطر',
                    'objective': 'تقليل الضوضاء قبل أي توسع جديد.',
                    'priority': 0.96 if anomalies.get('high_severity_events', 0) > 0 else 0.84,
                    'items': [
                        'راجع أحدث anomaly وحدد هل هي ضوضاء ولا خطر حقيقي.',
                        'فعّل قاعدة واضحة لأي prompts مشبوهة قبل متابعة التطوير.',
                    ],
                    'exit_criteria': 'انخفاض الأحداث الشاذة أو تصنيفها بوضوح.',
                    'signals': {'suspicious_events': anomalies.get('suspicious_events', 0)},
                }
            )

        if active_goals:
            top_goal = active_goals[0]
            phases.append(
                {
                    'phase_id': f'goal_{top_goal.get("goal_id")}',
                    'title': f'دفع الهدف الأساسي: {top_goal.get("title", "هدف بدون عنوان")}',
                    'objective': top_goal.get('description') or 'رفع التقدم الفعلي للهدف الأعلى أولوية.',
                    'priority': round(min(0.95, 0.62 + float(top_goal.get('priority', 0.5)) * 0.3), 4),
                    'items': [
                        action.get('title') for action in next_actions[:2] if action.get('focus') == top_goal.get('title')
                    ] or [
                        'قسّم الهدف إلى Patch صغيرة أو endpoint واحدة قابلة للاختبار.',
                        'ارفع progress بعد أول نتيجة تشغيل حقيقية.',
                    ],
                    'exit_criteria': 'الوصول إلى زيادة واضحة في progress أو إغلاق أول مهمة تنفيذية.',
                    'signals': {'progress': top_goal.get('progress', 0.0), 'priority': top_goal.get('priority', 0.0)},
                }
            )

        if predicted_needs or context_topics:
            phases.append(
                {
                    'phase_id': 'shape_next_output',
                    'title': 'تشكيل المخرج العملي التالي',
                    'objective': 'تجهيز أقرب مخرج متوقع من منظور المستخدم والسياق.',
                    'priority': 0.78,
                    'items': predicted_needs[:2] or ['حوّل أهم موضوعين متكررين إلى خطة تنفيذية قصيرة.'],
                    'exit_criteria': 'توليد مخرج واحد واضح وقابل للتنفيذ فوراً.',
                    'signals': {'predicted_needs': predicted_needs[:3], 'topics': context_topics[:4]},
                }
            )

        if hooks_summary.get('total_hooks', 0) > 0 or any('hook' in item.lower() or 'webhook' in item.lower() for item in predicted_needs):
            phases.append(
                {
                    'phase_id': 'automation_bridge',
                    'title': 'توصيل المنطق بالأفعال الخارجية',
                    'objective': 'ترجمة القرارات أو الرسائل المهمة إلى action hooks قابلة للاختبار.',
                    'priority': 0.74,
                    'items': [
                        'حدّث hook واحدة على الأقل بمفاتيح مطابقة أو cooldown مناسب.',
                        'نفّذ dry-run وتأكد إن payload preview مناسب قبل التوصيل الحقيقي.',
                    ],
                    'exit_criteria': 'وجود hook مفعلة ومجرّبة بدون أخطاء واضحة.',
                    'signals': {'total_hooks': hooks_summary.get('total_hooks', 0), 'active_hooks': hooks_summary.get('active_hooks', 0)},
                }
            )

        if self.last_dream_report:
            phases.append(
                {
                    'phase_id': 'reflect_and_expand',
                    'title': 'استثمار آخر دورة تحليل أو حلم',
                    'objective': 'استخدام الاستنتاجات المتراكمة لاختيار أسرع تحسين بعد التثبيت.',
                    'priority': 0.66,
                    'items': [
                        str(item) for item in self.last_dream_report.get('recommended_experiments', [])[:2]
                    ] or ['استخرج تجربة واحدة من أحدث insight وابدأ بها.'],
                    'exit_criteria': 'اختيار تجربة واحدة وبدء تنفيذها فعلياً.',
                    'signals': {'dream_topics': self.last_dream_report.get('dream_topics', [])[:3]},
                }
            )

        if include_completed and completed_goals:
            phases.append(
                {
                    'phase_id': 'completed_reference',
                    'title': 'الرجوع للمنجزات المكتملة',
                    'objective': 'استخدام الأهداف المكتملة كنمط مرجعي للتنفيذ التالي.',
                    'priority': 0.42,
                    'items': [goal.get('title', 'هدف مكتمل') for goal in completed_goals[:2]],
                    'exit_criteria': 'استخراج نمط نجاح واحد من الأهداف المكتملة.',
                    'signals': {'completed_goals': len(completed_goals)},
                }
            )

        if not phases:
            phases.append(
                {
                    'phase_id': 'explore_signal',
                    'title': 'جمع إشارة أوضح قبل التوسيع',
                    'objective': 'لا توجد ضغوط أو أهداف كفاية لتوليد roadmap متخصصة.',
                    'priority': 0.58,
                    'items': ['سجّل هدف واحد واضح أو message فيها نية تنفيذية صريحة.'],
                    'exit_criteria': 'وصول سياق جديد يسمح بترتيب أولويات أقوى.',
                    'signals': {'state': self.state},
                }
            )

        phases.sort(key=lambda item: float(item.get('priority', 0.0)), reverse=True)
        trimmed = phases[:resolved_limit]
        for index, phase in enumerate(trimmed, start=1):
            phase['stage_order'] = index
        return {
            'summary': {
                'state': self.state,
                'active_goals': len(active_goals),
                'completed_goals': len(completed_goals),
                'suspicious_events': anomalies.get('suspicious_events', 0),
                'active_hooks': hooks_summary.get('active_hooks', 0),
                'top_focus': trimmed[0]['title'] if trimmed else None,
            },
            'phases': trimmed,
        }

    def diagnostics(self) -> dict[str, Any]:
        history = list(self.decision_history)
        counter = Counter(item['decision'] for item in history)
        average_confidence = float(np.mean([item['confidence'] for item in history])) if history else 0.0
        strongest = max(history, key=lambda item: item['confidence'], default=None)
        return {
            'state': self.state,
            'cycles_completed': self.cycles_completed,
            'history_size': len(history),
            'average_confidence': average_confidence,
            'dominant_decisions': counter.most_common(5),
            'strongest_recent_decision': strongest,
            'recent_decisions': history[-5:],
            'memory_stats': self.memory.stats(),
            'learning_stats': self.executive.learning_stats(),
            'pending_feedback_count': len(self.pending_feedback),
            'recent_learning_events': list(self.learning_history)[-5:],
            'recent_self_improvement_events': list(self.self_improvement_history)[-5:],
            'personality': self.get_personality_profile(),
            'context': self.context_snapshot(limit=5),
            'affect': self.get_affect_state(),
            'goals': self.goals_overview(),
            'recent_sleep_reports': self.recent_sleep_reports(limit=3),
            'anomalies': self.anomaly_overview(),
            'user_model': self.get_user_model(),
            'last_tom_inference': self.last_tom_inference,
            'action_hooks': self.action_hooks_overview(),
            'recent_action_hook_events': self.recent_action_hook_events(limit=5),
            'last_dream_report': self.last_dream_report,
            'dashboard': self.dashboard_snapshot(),
            'recommended_focus': (
                'زيادة الاستكشاف وجمع بيانات جديدة.'
                if not counter
                else f"أكثر نمط متكرر حالياً هو: {counter.most_common(1)[0][0]}"
            ),
        }

    def save_state(self) -> Path:
        self.memory.save()
        weights_path = self._save_component_weights()
        payload = {
            'version': 8,
            'saved_at': time.time(),
            'seed': self.seed,
            'state': self.state,
            'cycles_completed': self.cycles_completed,
            'last_memory_key': self.last_memory_key,
            'last_decision': asdict(self.last_decision) if self.last_decision else None,
            'memory_path': str(self.storage_path) if self.storage_path else None,
            'memory_dsn': self.storage_dsn,
            'weights_path': str(weights_path),
            'rng_state': self._to_jsonable(self.rng.bit_generator.state),
            'decision_history': list(self.decision_history),
            'learning_history': list(self.learning_history),
            'pending_feedback': self.pending_feedback,
            'personality': self.get_personality_profile(),
            'context_window': list(self.context_window),
            'self_improvement_history': list(self.self_improvement_history),
            'goals': [asdict(goal) for goal in self.goals.values()],
            'affect_state': self.get_affect_state(),
            'sleep_reports': list(self.sleep_reports),
            'last_sleep_report': self.last_sleep_report,
            'episode_history': list(self.episode_history),
            'last_episode_key': self.last_episode_key,
            'anomaly_events': list(self.anomaly_events),
            'user_model': self.get_user_model(),
            'last_tom_inference': self.last_tom_inference,
            'action_hooks': self.action_hooks,
            'dream_history': list(self.dream_history),
            'last_dream_report': self.last_dream_report,
            'last_action_hook_result': self.last_action_hook_result,
            'components': {
                'vision_resolution': list(self.vision.resolution),
                'audio_channels': self.audio.audio_channels,
                'navigation_cells': self.navigation.num_cells,
                'navigation_scale': self.navigation.scale,
                'executive_input_size': self.executive.input_size,
                'decisions': list(self.executive.decisions),
            },
        }
        temp_path = self.state_path.with_suffix(self.state_path.suffix + '.tmp')
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_path.replace(self.state_path)
        return self.state_path

    def load_state(self, force_awake: bool = False) -> bool:
        if not self.state_path.exists():
            return False
        try:
            payload = json.loads(self.state_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        self.memory.load()
        self._load_component_weights()
        self.seed = payload.get('seed', self.seed)
        self.state = payload.get('state', self.state)
        self.cycles_completed = int(payload.get('cycles_completed', self.cycles_completed))
        self.last_memory_key = payload.get('last_memory_key', self.last_memory_key)
        last_decision = payload.get('last_decision')
        self.last_decision = DecisionResult(**last_decision) if isinstance(last_decision, dict) else None
        history_payload = payload.get('decision_history', [])
        self.decision_history = deque(
            (
                {
                    'decision': str(item.get('decision', 'غير معروف')),
                    'decision_index': int(item.get('decision_index', -1)),
                    'confidence': float(item.get('confidence', 0.0)),
                    'timestamp': float(item.get('timestamp', time.time())),
                    'source': str(item.get('source', 'unknown')),
                }
                for item in history_payload
                if isinstance(item, dict)
            ),
            maxlen=50,
        )
        learning_payload = payload.get('learning_history', [])
        self.learning_history = deque(
            (self._to_jsonable(item) for item in learning_payload if isinstance(item, dict)),
            maxlen=100,
        )
        pending_feedback = payload.get('pending_feedback', {})
        self.pending_feedback = {
            str(key): value for key, value in pending_feedback.items() if isinstance(value, dict)
        }
        personality_payload = payload.get('personality', {})
        if isinstance(personality_payload, dict):
            merged = self.get_personality_profile()
            merged.update({key: value for key, value in personality_payload.items() if value is not None})
            self.personality = PersonalityProfile(**merged)
        context_payload = payload.get('context_window', [])
        self.context_window = deque(
            [item for item in context_payload if isinstance(item, dict)],
            maxlen=12,
        )
        self_improvement_payload = payload.get('self_improvement_history', [])
        self.self_improvement_history = deque(
            [item for item in self_improvement_payload if isinstance(item, dict)],
            maxlen=50,
        )
        goals_payload = payload.get('goals', [])
        loaded_goals: dict[str, GoalRecord] = {}
        for item in goals_payload:
            if not isinstance(item, dict) or 'goal_id' not in item or 'title' not in item:
                continue
            goal = GoalRecord(
                goal_id=str(item['goal_id']),
                title=str(item['title']),
                description=str(item.get('description', '')),
                status=str(item.get('status', 'active')),
                priority=self._clip01(float(item.get('priority', 0.5))),
                progress=self._clip01(float(item.get('progress', 0.0))),
                target_keywords=[str(keyword) for keyword in item.get('target_keywords', []) if str(keyword).strip()],
                created_at=float(item.get('created_at', time.time())),
                updated_at=float(item.get('updated_at', time.time())),
                last_signal_at=float(item['last_signal_at']) if item.get('last_signal_at') is not None else None,
                signal_count=int(item.get('signal_count', 0)),
                notes=[str(note) for note in item.get('notes', []) if str(note).strip()],
            )
            loaded_goals[goal.goal_id] = goal
        self.goals = loaded_goals
        affect_payload = payload.get('affect_state', {})
        if isinstance(affect_payload, dict):
            self.affect_state = {
                'curiosity': self._clip01(float(affect_payload.get('curiosity', 0.58))),
                'stress': self._clip01(float(affect_payload.get('stress', 0.22))),
                'satisfaction': self._clip01(float(affect_payload.get('satisfaction', 0.48))),
                'energy': self._clip01(float(affect_payload.get('energy', 0.82))),
                'updated_at': float(affect_payload.get('updated_at', time.time())),
                'mood_label': str(affect_payload.get('mood_label', 'متوازن')),
            }
        sleep_payload = payload.get('sleep_reports', [])
        self.sleep_reports = deque([item for item in sleep_payload if isinstance(item, dict)], maxlen=20)
        self.last_sleep_report = payload.get('last_sleep_report') if isinstance(payload.get('last_sleep_report'), dict) else None
        episode_payload = payload.get('episode_history', [])
        self.episode_history = deque([item for item in episode_payload if isinstance(item, dict)], maxlen=50)
        self.last_episode_key = str(payload.get('last_episode_key')) if payload.get('last_episode_key') else None
        anomaly_payload = payload.get('anomaly_events', [])
        self.anomaly_events = deque([item for item in anomaly_payload if isinstance(item, dict)], maxlen=100)
        user_model_payload = payload.get('user_model', {})
        if isinstance(user_model_payload, dict):
            self.user_model = UserModelEngine.empty_profile()
            self.user_model.update(user_model_payload)
        last_tom_payload = payload.get('last_tom_inference')
        self.last_tom_inference = last_tom_payload if isinstance(last_tom_payload, dict) else None
        action_hooks_payload = payload.get('action_hooks', {})
        if isinstance(action_hooks_payload, dict):
            self.action_hooks = ActionHookGateway.empty_registry()
            self.action_hooks.update(action_hooks_payload)
            self.action_hooks['hooks'] = [item for item in action_hooks_payload.get('hooks', []) if isinstance(item, dict)]
            self.action_hooks['events'] = [item for item in action_hooks_payload.get('events', []) if isinstance(item, dict)][-100:]
        dream_payload = payload.get('dream_history', [])
        self.dream_history = deque([item for item in dream_payload if isinstance(item, dict)], maxlen=20)
        self.last_dream_report = payload.get('last_dream_report') if isinstance(payload.get('last_dream_report'), dict) else None
        self.last_action_hook_result = payload.get('last_action_hook_result') if isinstance(payload.get('last_action_hook_result'), dict) else None
        rng_state = payload.get('rng_state')
        if isinstance(rng_state, dict):
            self.rng.bit_generator.state = rng_state
        decisions = payload.get('components', {}).get('decisions')
        if isinstance(decisions, list) and decisions:
            self.executive.decisions = [str(item) for item in decisions]
        self._refresh_affect_state()
        if force_awake:
            self.state = 'مستيقظ'
            self._shift_affect(energy=0.06, stress=-0.04)
        return True

    def sleep(self) -> dict[str, Any]:
        report = self.run_sleep_cycle(trigger='sleep')
        self.state = 'نائم'
        self.save_state()
        return report

    def wake_up(self) -> None:
        loaded = self.load_state(force_awake=True)
        if not loaded:
            self.memory.load()
            self.state = 'مستيقظ'
            self._shift_affect(energy=0.08, stress=-0.03)

    def system_status(self) -> dict[str, Any]:
        return {
            'state': self.state,
            'seed': self.seed,
            'cycles_completed': self.cycles_completed,
            'last_memory_key': self.last_memory_key,
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'storage_dsn': self.storage_dsn,
            'state_path': str(self.state_path),
            'weights_snapshot_path': str(self._weights_path()),
            'memory_stats': self.memory.stats(),
            'last_decision': asdict(self.last_decision) if self.last_decision else None,
            'decision_history_size': len(self.decision_history),
            'learning_stats': self.executive.learning_stats(),
            'pending_feedback_count': len(self.pending_feedback),
            'personality': self.get_personality_profile(),
            'context_window_size': len(self.context_window),
            'self_improvement_events': len(self.self_improvement_history),
            'episode_count': len(self.episode_history),
            'last_episode_key': self.last_episode_key,
            'affect': self.get_affect_state(),
            'goals': self.goals_overview(),
            'user_model': self.get_user_model(),
            'last_tom_inference': self.last_tom_inference,
            'action_hooks': self.action_hooks_overview(),
            'last_action_hook_result': self.last_action_hook_result,
            'last_sleep_report': self.last_sleep_report,
            'last_dream_report': self.last_dream_report,
            'recent_episodes': list(self.episode_history)[-3:],
            'anomalies': self.anomaly_overview(),
        }

    def close(self) -> None:
        self.memory.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup during GC
        try:
            self.close()
        except Exception:
            pass
