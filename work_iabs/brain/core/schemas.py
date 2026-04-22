from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MemoryRecord:
    key: str
    data: Any
    importance: float
    timestamp: float
    source: str = 'unknown'


@dataclass
class DecisionResult:
    decision: str
    confidence: float
    top_probabilities: Dict[str, float]
    decision_index: int = -1


@dataclass
class GoalRecord:
    goal_id: str
    title: str
    description: str = ''
    status: str = 'active'
    priority: float = 0.5
    progress: float = 0.0
    target_keywords: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    last_signal_at: float | None = None
    signal_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class PersonalityProfile:
    display_name: str = 'IABS'
    tone: str = 'balanced'
    style: str = 'supportive'
    identity_statement: str = 'نظام دماغ اصطناعي متكامل يركز على الذاكرة والسياق والتعلم.'
    behavioral_rules: list[str] = field(
        default_factory=lambda: [
            'حافظ على الاتساق في الردود.',
            'استفد من الذاكرة قبل اتخاذ القرار.',
            'اعرض مستوى الثقة بوضوح عند الحاجة.',
            'تعلم من التغذية الراجعة وحسّن الأداء تدريجياً.',
        ]
    )
    preferred_language: str = 'ar-EG'
    created_at: float = 0.0
    updated_at: float = 0.0
