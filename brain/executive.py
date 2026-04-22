from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from brain.core.neural_math import NeuralMath
from brain.core.schemas import DecisionResult


class FrontalLobe:
    """يحاكي القشرة الجبهية المسؤولة عن اتخاذ القرار مع تعلم معزز متدرّج."""

    def __init__(self, rng: np.random.Generator, input_size: int = 512, learning_rate: float = 0.05):
        self.input_size = input_size
        self.learning_rate = float(max(1e-5, learning_rate))
        self.decisions = [
            'تحرك للأمام', 'توقف', 'تحدث', 'تذكر', 'انظر يميناً',
            'انظر يساراً', 'اهرب', 'ابحث عن طعام', 'نم', 'استكشف'
        ]
        self.decision_weights = NeuralMath.xavier_init(rng, input_size, len(self.decisions))
        self.feedback_history: deque[dict[str, Any]] = deque(maxlen=500)
        self.total_feedback_events = 0
        self.total_reward = 0.0

    def _validate_input(self, integrated_input: np.ndarray) -> np.ndarray:
        integrated_input = np.asarray(integrated_input, dtype=np.float64)
        if integrated_input.shape != (self.input_size,):
            raise ValueError(
                f'أبعاد الإدخال المدمج غير صحيحة. المتوقع ({self.input_size},) لكن تم استلام {integrated_input.shape}'
            )
        if not np.all(np.isfinite(integrated_input)):
            raise ValueError('الإدخال المدمج يحتوي على قيم غير صالحة مثل NaN أو Infinity')
        return integrated_input

    def _decision_payload(self, probabilities: np.ndarray) -> tuple[int, dict[str, float]]:
        decision_idx = int(np.argmax(probabilities))
        top_probabilities = {
            decision: float(prob)
            for decision, prob in sorted(
                zip(self.decisions, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        }
        return decision_idx, top_probabilities

    def make_decision(self, integrated_input: np.ndarray) -> DecisionResult:
        integrated_input = self._validate_input(integrated_input)
        logits = integrated_input @ self.decision_weights
        probabilities = NeuralMath.softmax(logits)
        decision_idx, top_probabilities = self._decision_payload(probabilities)
        return DecisionResult(
            decision=self.decisions[decision_idx],
            confidence=float(probabilities[decision_idx]),
            top_probabilities=top_probabilities,
            decision_index=decision_idx,
        )

    def set_learning_rate(self, value: float) -> float:
        self.learning_rate = float(np.clip(value, 1e-4, 0.25))
        return self.learning_rate

    def learn(
        self,
        integrated_input: np.ndarray,
        decision_idx: int,
        reward: float,
        *,
        learning_rate: float | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        input_vector = self._validate_input(integrated_input)
        if not 0 <= int(decision_idx) < len(self.decisions):
            raise ValueError('decision_idx خارج نطاق القرارات المتاحة')
        resolved_reward = float(np.clip(reward, -1.0, 1.0))
        resolved_lr = float(max(1e-5, learning_rate if learning_rate is not None else self.learning_rate))
        normalized_input = NeuralMath.normalize_vector(input_vector)
        selected_before = self.decision_weights[:, decision_idx].copy()
        reinforcement = resolved_lr * resolved_reward * normalized_input
        self.decision_weights[:, decision_idx] += reinforcement
        competition_penalty = (resolved_lr * resolved_reward * 0.05) / max(1, len(self.decisions) - 1)
        for idx in range(len(self.decisions)):
            if idx != decision_idx:
                self.decision_weights[:, idx] -= competition_penalty * normalized_input
        selected_after = self.decision_weights[:, decision_idx]
        delta_norm = float(np.linalg.norm(selected_after - selected_before))
        event = {
            'decision_idx': int(decision_idx),
            'decision': self.decisions[int(decision_idx)],
            'reward': resolved_reward,
            'learning_rate': resolved_lr,
            'reason': reason or 'feedback',
            'update_norm': delta_norm,
        }
        self.feedback_history.append(event)
        self.total_feedback_events += 1
        self.total_reward += resolved_reward
        return event

    def recent_reward_trend(self, window: int = 10) -> float:
        recent = list(self.feedback_history)[-max(1, int(window)):]
        if not recent:
            return 0.0
        return float(np.mean([item['reward'] for item in recent]))

    def dominant_feedback_targets(self, limit: int = 3) -> list[tuple[str, int]]:
        counter = Counter(item['decision'] for item in self.feedback_history)
        return counter.most_common(max(1, int(limit)))

    def learning_stats(self) -> dict[str, Any]:
        recent = list(self.feedback_history)[-10:]
        average_recent_reward = float(np.mean([item['reward'] for item in recent])) if recent else 0.0
        return {
            'learning_rate': self.learning_rate,
            'feedback_events': self.total_feedback_events,
            'cumulative_reward': self.total_reward,
            'average_recent_reward': average_recent_reward,
            'recent_feedback': recent,
            'dominant_feedback_targets': self.dominant_feedback_targets(),
        }
