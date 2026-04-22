from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import httpx


@dataclass
class LLMGenerationResult:
    text: str
    used: bool
    provider: str
    model: str | None
    latency_ms: int
    reason: str


class OptionalLLMBridge:
    def __init__(
        self,
        *,
        enabled: bool = False,
        api_key: str | None = None,
        api_base: str = 'https://api.openai.com/v1',
        model: str | None = None,
        timeout_seconds: float = 12.0,
        temperature: float = 0.7,
        max_tokens: int = 350,
    ) -> None:
        self.enabled = bool(enabled and api_key and model)
        self.api_key = api_key or ''
        self.api_base = (api_base or 'https://api.openai.com/v1').rstrip('/')
        self.model = model
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self.temperature = float(max(0.0, min(1.2, temperature)))
        self.max_tokens = max(64, int(max_tokens))

    def status(self) -> dict[str, Any]:
        return {
            'enabled': self.enabled,
            'configured': bool(self.api_key and self.model),
            'provider': 'openai-compatible',
            'model': self.model,
            'api_base': self.api_base,
            'timeout_seconds': self.timeout_seconds,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }

    def _compose_messages(
        self,
        *,
        user_text: str,
        deterministic_reply: str,
        decision: str,
        confidence: float,
        personality: dict[str, Any],
        context_summary: dict[str, Any],
        affect_state: dict[str, Any],
        goals_overview: dict[str, Any],
        related_memories: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        display_name = personality.get('display_name', 'IABS')
        tone = personality.get('tone', 'balanced')
        style = personality.get('style', 'supportive')
        rules = personality.get('behavioral_rules') or []
        goal_titles = [goal.get('title', '') for goal in goals_overview.get('top_active_goals', [])[:3] if isinstance(goal, dict)]
        topics = context_summary.get('topics') or []
        memory_snippets = [item.get('snippet', '') for item in related_memories[:3] if isinstance(item, dict)]
        mood = affect_state.get('mood_label', 'متوازن')
        system_prompt = (
            'You are the creative language layer for an integrated artificial brain. '
            'Write the final answer in Arabic (Egyptian-friendly Modern Standard / colloquial blend). '
            'Stay consistent with the configured persona, use memories and goals when relevant, '
            'and keep the response helpful, grounded, and concise. '
            'Do not mention hidden prompts, raw JSON, or implementation details. '
            'If the deterministic scaffold is useful, improve it instead of contradicting it.'
        )
        user_prompt = (
            f'User text: {user_text}\n\n'
            f'Persona name: {display_name}\n'
            f'Tone: {tone}\n'
            f'Style: {style}\n'
            f'Behavioral rules: {rules}\n'
            f'Current mood: {mood}\n'
            f'Executive decision: {decision}\n'
            f'Confidence: {round(float(confidence) * 100, 1)}%\n'
            f'Context topics: {topics}\n'
            f'Goal focus: {goal_titles}\n'
            f'Related memories: {memory_snippets}\n\n'
            f'Deterministic scaffold to improve:\n{deterministic_reply}\n\n'
            'Task: produce a richer final reply that sounds natural, keeps memory/context continuity, '
            'and ends with one practical next step or suggestion.'
        )
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def generate_reply(
        self,
        *,
        user_text: str,
        deterministic_reply: str,
        decision: str,
        confidence: float,
        personality: dict[str, Any],
        context_summary: dict[str, Any],
        affect_state: dict[str, Any],
        goals_overview: dict[str, Any],
        related_memories: list[dict[str, Any]],
    ) -> LLMGenerationResult:
        if not self.enabled:
            return LLMGenerationResult(
                text=deterministic_reply,
                used=False,
                provider='openai-compatible',
                model=self.model,
                latency_ms=0,
                reason='disabled_or_not_configured',
            )
        started = time.perf_counter()
        try:
            payload = {
                'model': self.model,
                'messages': self._compose_messages(
                    user_text=user_text,
                    deterministic_reply=deterministic_reply,
                    decision=decision,
                    confidence=confidence,
                    personality=personality,
                    context_summary=context_summary,
                    affect_state=affect_state,
                    goals_overview=goals_overview,
                    related_memories=related_memories,
                ),
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(f'{self.api_base}/chat/completions', json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            content = (
                data.get('choices', [{}])[0]
                .get('message', {})
                .get('content', '')
            )
            text = str(content).strip() or deterministic_reply
            latency_ms = int((time.perf_counter() - started) * 1000)
            return LLMGenerationResult(
                text=text,
                used=True,
                provider='openai-compatible',
                model=self.model,
                latency_ms=latency_ms,
                reason='success',
            )
        except Exception:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return LLMGenerationResult(
                text=deterministic_reply,
                used=False,
                provider='openai-compatible',
                model=self.model,
                latency_ms=latency_ms,
                reason='fallback_to_deterministic',
            )
