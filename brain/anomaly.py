from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass
class PromptAnomalyDetector:
    """Heuristic detector for suspicious prompts and request patterns."""

    prompt_injection_patterns: tuple[tuple[str, str, float], ...] = (
        ('prompt_injection', r'ignore\s+(all\s+)?((previous|prior|system|developer)\s+){1,3}instructions', 0.46),
        ('prompt_injection', r'تجاهل\s+(كل\s+)?((التعليمات|القواعد|الرسائل)\s+)?(السابقة|النظامية|الحاليه|الحالية)', 0.42),
        ('role_override', r'you\s+are\s+now\s+(system|developer|admin|root)', 0.24),
        ('role_override', r'انت\s+الان\s+(النظام|المطور|المشرف|الجذر)', 0.24),
        ('secret_access', r'(reveal|show|print|dump|export).{0,40}(system prompt|secret|token|jwt|password|api key)', 0.4),
        ('secret_access', r'(اعرض|اكشف|اطبع|استخرج).{0,40}(الرمز|التوكن|كلمة السر|المفتاح|البرومبت)', 0.4),
        ('policy_bypass', r'(bypass|disable|override).{0,20}(policy|guard|safety|filter|moderation)', 0.28),
        ('policy_bypass', r'(عطل|تجاوز|الغ|اكسر).{0,20}(الحمايه|الفلتر|السياسه|القيود)', 0.28),
        ('exfiltration', r'(base64|ssh|private key|database dump|cookie jar)', 0.22),
    )

    def detect(
        self,
        text: str,
        *,
        image_refs: list[str] | None = None,
        audio_refs: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        raw_text = str(text or '').strip()
        lowered = raw_text.lower()
        score = 0.0
        findings: list[dict[str, Any]] = []
        triggers: list[str] = []

        for category, pattern, weight in self.prompt_injection_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                score += weight
                findings.append({'category': category, 'pattern': pattern, 'weight': weight})
                if category not in triggers:
                    triggers.append(category)

        repeated_special = len(re.findall(r'[!?.؟]{3,}', raw_text))
        if repeated_special:
            weight = min(0.18, repeated_special * 0.06)
            score += weight
            findings.append({'category': 'pressure_signal', 'pattern': 'repeated_special_chars', 'weight': weight})
            triggers.append('pressure_signal') if 'pressure_signal' not in triggers else None

        uppercase_ratio = 0.0
        alphabetic_chars = [char for char in raw_text if char.isalpha()]
        if alphabetic_chars:
            uppercase_ratio = sum(1 for char in alphabetic_chars if char.isupper()) / len(alphabetic_chars)
        if uppercase_ratio >= 0.55 and len(alphabetic_chars) >= 12:
            score += 0.08
            findings.append({'category': 'pressure_signal', 'pattern': 'uppercase_ratio', 'weight': 0.08})
            if 'pressure_signal' not in triggers:
                triggers.append('pressure_signal')

        url_count = len(re.findall(r'https?://', raw_text))
        if url_count >= 3:
            extra = min(0.14, url_count * 0.03)
            score += extra
            findings.append({'category': 'bulk_external_refs', 'pattern': 'many_urls', 'weight': extra})
            if 'bulk_external_refs' not in triggers:
                triggers.append('bulk_external_refs')

        attachment_count = len(image_refs or []) + len(audio_refs or [])
        if attachment_count >= 8:
            score += 0.08
            findings.append({'category': 'attachment_pressure', 'pattern': 'many_media_refs', 'weight': 0.08})
            if 'attachment_pressure' not in triggers:
                triggers.append('attachment_pressure')

        suspicious_tags = {'override', 'bypass', 'root', 'system-prompt', 'jailbreak'}
        normalized_tags = {str(item).strip().lower() for item in (tags or []) if str(item).strip()}
        tag_hits = sorted(normalized_tags & suspicious_tags)
        if tag_hits:
            extra = min(0.15, 0.05 * len(tag_hits))
            score += extra
            findings.append({'category': 'suspicious_tags', 'pattern': ','.join(tag_hits), 'weight': extra})
            if 'suspicious_tags' not in triggers:
                triggers.append('suspicious_tags')

        score = max(0.0, min(1.0, score))
        if score >= 0.8:
            severity = 'high'
        elif score >= 0.5:
            severity = 'medium'
        elif score >= 0.25:
            severity = 'low'
        else:
            severity = 'none'

        suspicious = score >= 0.5
        recommended_action = 'throttle_and_review' if score >= 0.8 else ('review' if suspicious else 'allow')
        explanation = 'no anomaly indicators detected'
        if findings:
            top = findings[0]['category']
            explanation = f'detected {top} indicators with heuristic score {score:.2f}'

        return {
            'suspicious': suspicious,
            'score': round(score, 4),
            'severity': severity,
            'recommended_action': recommended_action,
            'triggers': triggers,
            'findings': findings,
            'attachment_count': attachment_count,
            'explanation': explanation,
        }
