from __future__ import annotations

from collections import Counter
import hashlib
import html
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from brain.system import IntegratedArtificialBrain


_ARABIC_TRANSLATION = str.maketrans(
    {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
        'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه',
        'ـ': '',
    }
)

_STOPWORDS = {
    'الى', 'على', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي', 'كان', 'كانت', 'كما', 'لكن', 'وقد', 'فقد', 'وهو', 'وهي',
    'في', 'من', 'عن', 'مع', 'ثم', 'بعد', 'قبل', 'كل', 'لما', 'لان', 'إن', 'ان', 'او', 'أو', 'حتى', 'ضمن', 'عند',
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'into', 'your', 'have', 'will', 'you', 'are', 'was',
}

_NEGATION_TOKENS = {
    'لا', 'ليس', 'لم', 'لن', 'بدون', 'غير', 'ما', 'no', 'not', 'never', 'none', 'without', 'cannot', "can't",
    "doesn't", 'doesnt', "isn't", 'isnt', "won't", 'wont',
}

_TRUSTED_HOST_HINTS = (
    '.gov', '.edu', '.mil', '.gov.uk', '.ac.uk',
)

_HIGH_TRUST_HINTS = {
    'official', 'docs', 'documentation', 'reference', 'manual', 'spec', 'policy',
    'دليل', 'مرجع', 'وثيقه', 'توثيق', 'مواصفات',
}

_LOW_TRUST_HINTS = {
    'blog', 'blogs', 'forum', 'forums', 'community', 'gist', 'mirror',
    'مدونه', 'منتدي',
}


def _normalize_text(text: str) -> str:
    lowered = str(text).strip().lower().translate(_ARABIC_TRANSLATION)
    lowered = re.sub(r'[\u064b-\u065f\u0670]', '', lowered)
    lowered = re.sub(r'\s+', ' ', lowered, flags=re.UNICODE)
    return lowered.strip()


def clean_text(text: str) -> str:
    normalized = html.unescape(str(text or ''))
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'\u00a0', ' ', normalized)
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


def extract_title_from_html(markup: str) -> str | None:
    match = re.search(r'<title[^>]*>(.*?)</title>', markup or '', flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = clean_text(re.sub(r'<[^>]+>', ' ', html.unescape(match.group(1))))
    return title[:180] if title else None


def html_to_text(markup: str) -> str:
    text = str(markup or '')
    text = re.sub(r'(?is)<script[^>]*>.*?</script>', ' ', text)
    text = re.sub(r'(?is)<style[^>]*>.*?</style>', ' ', text)
    text = re.sub(r'(?i)</(p|div|section|article|h1|h2|h3|h4|h5|h6|li|blockquote|br|tr)>', '\n', text)
    text = re.sub(r'(?is)<[^>]+>', ' ', text)
    return clean_text(text)


def text_fingerprint(text: str) -> str:
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def split_text_into_chunks(
    text: str,
    *,
    max_chars: int = 900,
    overlap_chars: int = 120,
    min_chunk_chars: int = 180,
) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    max_chars = max(300, int(max_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chars // 3))
    min_chunk_chars = max(80, min(int(min_chunk_chars), max_chars))
    if len(cleaned) <= max_chars:
        return [cleaned]

    raw_units = [
        part.strip() for part in re.split(r'\n{2,}|(?<=[\.\!؟\?])\s+', cleaned, flags=re.UNICODE)
        if part and part.strip()
    ]
    units: list[str] = []
    for part in raw_units:
        if len(part) <= max_chars:
            units.append(part)
            continue
        start = 0
        while start < len(part):
            end = min(len(part), start + max_chars)
            window = part[start:end]
            if end < len(part):
                soft_break = max(window.rfind('،'), window.rfind(' '), window.rfind('؛'))
                if soft_break >= min_chunk_chars:
                    end = start + soft_break + 1
                    window = part[start:end]
            window = window.strip()
            if window:
                units.append(window)
            start = max(end - overlap_chars, end) if overlap_chars == 0 else max(start + 1, end - overlap_chars)
            if end >= len(part):
                break

    chunks: list[str] = []
    current = ''
    for unit in units:
        candidate = f'{current} {unit}'.strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            overlap = current[-overlap_chars:].strip() if overlap_chars else ''
            current = f'{overlap} {unit}'.strip() if overlap else unit
        else:
            current = candidate
    if current:
        chunks.append(current.strip())

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk_chars:
            candidate = f'{merged[-1]} {chunk}'.strip()
            if len(candidate) <= max_chars + overlap_chars:
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return merged


def _source_topics(source_name: str, text: str, tags: list[str]) -> list[str]:
    counter: Counter[str] = Counter()
    fragments = [source_name or '', text[:4000], ' '.join(tags or [])]
    for fragment in fragments:
        normalized = _normalize_text(fragment)
        for token in re.findall(r'\w+', normalized, flags=re.UNICODE):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            counter[token] += 1
    return [token for token, _ in counter.most_common(8)]


def _source_profile(source_name: str, source_url: str | None, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    cleaned_name = clean_text(source_name)
    cleaned_text = clean_text(text)
    parsed = urlparse(clean_text(source_url or '')) if source_url else None
    host = parsed.netloc.lower() if parsed and parsed.netloc else None
    scheme = parsed.scheme.lower() if parsed and parsed.scheme else None
    metadata = metadata or {}

    score = 0.38
    signals: list[str] = []

    if scheme == 'https':
        score += 0.16
        signals.append('https_source')
    elif scheme == 'http':
        score += 0.06
        signals.append('http_source')
    else:
        score -= 0.03
        signals.append('no_source_url')

    if host:
        if any(host.endswith(hint) for hint in _TRUSTED_HOST_HINTS):
            score += 0.18
            signals.append('institutional_domain')
        if any(hint in host for hint in ('docs.', 'api.', 'developer.', 'reference.')):
            score += 0.11
            signals.append('documentation_domain')
        if any(hint in host for hint in _LOW_TRUST_HINTS):
            score -= 0.08
            signals.append('community_or_blog_domain')
    else:
        signals.append('local_or_manual_source')

    normalized_name = _normalize_text(cleaned_name)
    if any(hint in normalized_name for hint in _HIGH_TRUST_HINTS):
        score += 0.08
        signals.append('authoritative_title_hint')
    if any(hint in normalized_name for hint in _LOW_TRUST_HINTS):
        score -= 0.05
        signals.append('casual_title_hint')

    char_count = len(cleaned_text)
    if char_count >= 5000:
        score += 0.12
        signals.append('rich_content')
    elif char_count >= 1800:
        score += 0.08
        signals.append('substantial_content')
    elif char_count < 400:
        score -= 0.08
        signals.append('thin_content')

    content_type = str(metadata.get('content_type') or '').lower()
    if content_type.startswith('text/html') or content_type.startswith('text/plain'):
        score += 0.04
        signals.append('structured_text_content')

    source_type = str(metadata.get('source_type') or ('url' if source_url else 'text')).strip().lower()
    if source_type == 'url':
        score += 0.03
    elif source_type == 'text':
        score += 0.01

    score = round(float(np.clip(score, 0.05, 0.98)), 4)
    if score >= 0.75:
        band = 'high'
    elif score >= 0.55:
        band = 'medium'
    else:
        band = 'low'

    return {
        'score': score,
        'band': band,
        'host': host,
        'scheme': scheme,
        'signals': signals[:6],
        'character_count': char_count,
        'source_type': source_type,
    }


def _split_into_sentences(text: str, *, sentence_chars: int = 260) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<=[\.\!؟\?])\s+|\n+', cleaned, flags=re.UNICODE)
        if sentence and sentence.strip()
    ]
    result: list[str] = []
    for sentence in sentences:
        compact = sentence[:sentence_chars].strip()
        if len(compact) >= 25 and compact not in result:
            result.append(compact)
    return result or [cleaned[:sentence_chars].strip()]


def _sentence_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    return {
        token
        for token in re.findall(r'\w+', normalized, flags=re.UNICODE)
        if len(token) >= 3 and token not in _STOPWORDS and token not in _NEGATION_TOKENS
    }


def _has_negation(text: str) -> bool:
    normalized = _normalize_text(text)
    tokens = set(re.findall(r'\w+', normalized, flags=re.UNICODE))
    return any(token in tokens for token in _NEGATION_TOKENS)


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r'\d+(?:\.\d+)?', clean_text(text)))


def _contradiction_signal(sentence_a: str, sentence_b: str) -> tuple[float, str] | None:
    tokens_a = _sentence_tokens(sentence_a)
    tokens_b = _sentence_tokens(sentence_b)
    if not tokens_a or not tokens_b:
        return None
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    negation_mismatch = _has_negation(sentence_a) != _has_negation(sentence_b)
    numbers_a = _numeric_tokens(sentence_a)
    numbers_b = _numeric_tokens(sentence_b)
    number_conflict = bool(numbers_a and numbers_b and numbers_a != numbers_b)
    if overlap < 0.45 or (not negation_mismatch and not number_conflict):
        return None
    score = (overlap * 0.62) + (0.22 if negation_mismatch else 0.0) + (0.16 if number_conflict else 0.0)
    if score < 0.58:
        return None
    reasons: list[str] = []
    if negation_mismatch:
        reasons.append('اختلاف واضح في النفي أو الإثبات')
    if number_conflict:
        reasons.append('تعارض رقمي محتمل')
    return round(float(np.clip(score, 0.0, 0.99)), 4), ' + '.join(reasons)


def _build_chunk_tags(source_name: str, tags: list[str] | None = None) -> list[str]:
    ordered = ['knowledge', 'ingest']
    for item in tags or []:
        candidate = clean_text(item)
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    if source_name:
        short = clean_text(source_name)[:40]
        if short and short not in ordered:
            ordered.append(short)
    return ordered[:10]


def _memory_key_exists(brain: IntegratedArtificialBrain, key: str) -> bool:
    return any(record.key == key for record in brain.memory.iter_records())


def ingest_text_into_memory(
    brain: IntegratedArtificialBrain,
    *,
    text: str,
    source_name: str,
    source_url: str | None = None,
    tags: list[str] | None = None,
    chunk_size_chars: int = 900,
    overlap_chars: int = 120,
    min_chunk_chars: int = 180,
    importance: float = 0.76,
    promote_manifest_to_long_term: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = clean_text(text)
    if len(cleaned) < 40:
        raise ValueError('النص قصير جداً للإدخال إلى الذاكرة')
    source_name = clean_text(source_name) or 'web_source'
    source_url = clean_text(source_url) if source_url else None
    tags = [clean_text(item) for item in (tags or []) if clean_text(item)]
    fingerprint = text_fingerprint(cleaned)
    manifest_key = f'knowledge_manifest_{fingerprint[:24]}'
    if _memory_key_exists(brain, manifest_key):
        existing = brain.memory.recall(manifest_key)
        if isinstance(existing, dict):
            response = dict(existing)
            response['duplicate'] = True
            response['status'] = 'already_ingested'
            return response

    ingest_id = f'knowledge_{int(time.time() * 1000)}_{fingerprint[:8]}'
    chunks = split_text_into_chunks(
        cleaned,
        max_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
        min_chunk_chars=min_chunk_chars,
    )
    if not chunks:
        raise ValueError('تعذر تقسيم النص إلى مقاطع صالحة')

    chunk_keys: list[str] = []
    previous_chunk_key: str | None = None
    chunk_tags = _build_chunk_tags(source_name, tags)
    topics = _source_topics(source_name, cleaned, chunk_tags)
    source_profile = _source_profile(source_name, source_url, cleaned, metadata=metadata)
    chunk_importance = float(np.clip(min(importance, 0.79), 0.35, 0.79))

    for index, chunk in enumerate(chunks, start=1):
        episode = brain.record_episode(
            text=chunk,
            tags=chunk_tags,
            importance=chunk_importance,
            related_memory_keys=[previous_chunk_key] if previous_chunk_key else [],
            metadata={
                'kind': 'knowledge_chunk',
                'ingest_id': ingest_id,
                'source_name': source_name,
                'source_url': source_url,
                'source_type': 'text',
                'fingerprint': fingerprint,
                'chunk_index': index,
                'total_chunks': len(chunks),
                'topics': topics,
                **(metadata or {}),
            },
            source='episode',
        )
        previous_chunk_key = episode['key']
        chunk_keys.append(episode['key'])

    manifest_payload = {
        'kind': 'knowledge_manifest',
        'ingest_id': ingest_id,
        'source_name': source_name,
        'source_url': source_url,
        'fingerprint': fingerprint,
        'chunk_count': len(chunks),
        'character_count': len(cleaned),
        'chunk_size_chars': int(chunk_size_chars),
        'overlap_chars': int(overlap_chars),
        'tags': chunk_tags,
        'topics': topics,
        'chunk_keys': chunk_keys,
        'created_at': time.time(),
        'duplicate': False,
        'status': 'ingested',
        'preview': cleaned[:240],
        'trust_score': source_profile['score'],
        'trust_band': source_profile['band'],
        'source_profile': source_profile,
        'metadata': metadata or {},
    }
    manifest_importance = 0.82 if promote_manifest_to_long_term else max(0.72, float(np.clip(importance, 0.0, 1.0)))
    brain.memory.store_memory(
        key=manifest_key,
        data=manifest_payload,
        importance=manifest_importance,
        source='knowledge_ingest',
    )
    return manifest_payload


async def fetch_url_text(url: str, *, timeout_seconds: float = 20.0) -> dict[str, Any]:
    target_url = clean_text(url)
    if not target_url:
        raise ValueError('الرابط مطلوب')
    parsed = urlparse(target_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('الرابط يجب أن يكون http أو https صالح')
    headers = {
        'User-Agent': 'IABS Knowledge Ingest/1.0',
        'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1',
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        response = await client.get(target_url)
        response.raise_for_status()
        content_type = str(response.headers.get('content-type', '')).lower()
        raw_text = response.text
        title = extract_title_from_html(raw_text) if 'html' in content_type or '<html' in raw_text.lower() else None
        if 'html' in content_type or '<html' in raw_text.lower():
            text = html_to_text(raw_text)
        elif content_type.startswith('text/') or not content_type:
            text = clean_text(raw_text)
        else:
            raise ValueError('المحتوى المستهدف ليس صفحة HTML أو نصاً مباشراً')
        return {
            'final_url': str(response.url),
            'content_type': content_type,
            'status_code': int(response.status_code),
            'title': title,
            'text': text,
        }


async def ingest_url_into_memory(
    brain: IntegratedArtificialBrain,
    *,
    target_url: str,
    source_name: str | None = None,
    tags: list[str] | None = None,
    chunk_size_chars: int = 900,
    overlap_chars: int = 120,
    min_chunk_chars: int = 180,
    importance: float = 0.76,
) -> dict[str, Any]:
    fetched = await fetch_url_text(target_url)
    parsed = urlparse(fetched['final_url'])
    resolved_source_name = clean_text(source_name or fetched.get('title') or parsed.netloc or 'web_source')
    payload = ingest_text_into_memory(
        brain,
        text=fetched['text'],
        source_name=resolved_source_name,
        source_url=fetched['final_url'],
        tags=tags,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
        min_chunk_chars=min_chunk_chars,
        importance=importance,
        metadata={
            'fetched_from_url': fetched['final_url'],
            'content_type': fetched['content_type'],
            'status_code': fetched['status_code'],
            'source_type': 'url',
            'page_title': fetched.get('title'),
        },
    )
    payload['fetch'] = {
        'final_url': fetched['final_url'],
        'content_type': fetched['content_type'],
        'status_code': fetched['status_code'],
        'title': fetched.get('title'),
    }
    return payload


def knowledge_sources_summary(brain: IntegratedArtificialBrain, *, limit: int = 10) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for record in brain.memory.iter_records():
        if record.source != 'knowledge_ingest' or not isinstance(record.data, dict):
            continue
        if record.data.get('kind') != 'knowledge_manifest':
            continue
        manifest = dict(record.data)
        manifest['importance'] = record.importance
        manifest['timestamp'] = record.timestamp
        manifests.append(manifest)
    manifests.sort(key=lambda item: float(item.get('created_at', 0.0) or 0.0), reverse=True)
    limited = manifests[:max(1, int(limit))]
    return {
        'count': len(manifests),
        'total_chunks': sum(int(item.get('chunk_count', 0) or 0) for item in manifests),
        'sources': limited,
    }


def search_ingested_chunks(
    brain: IntegratedArtificialBrain,
    *,
    query: str,
    limit: int = 5,
    min_importance: float = 0.0,
) -> dict[str, Any]:
    records = brain.memory.search_memories(
        query,
        limit=max(int(limit) * 6, int(limit)),
        min_importance=min_importance,
        source='episode',
        strategy='hybrid',
    )
    results: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record.data, dict):
            continue
        metadata = record.data.get('metadata') or {}
        if metadata.get('kind') != 'knowledge_chunk':
            continue
        results.append(
            {
                'key': record.key,
                'importance': record.importance,
                'timestamp': record.timestamp,
                'summary': record.data.get('summary') or record.data.get('text', '')[:200],
                'topics': record.data.get('topics', []),
                'source_name': metadata.get('source_name'),
                'source_url': metadata.get('source_url'),
                'ingest_id': metadata.get('ingest_id'),
                'chunk_index': metadata.get('chunk_index'),
                'total_chunks': metadata.get('total_chunks'),
            }
        )
        if len(results) >= max(1, int(limit)):
            break
    return {
        'query': query,
        'count': len(results),
        'matches': results,
    }



def _manifest_records(brain: IntegratedArtificialBrain) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for record in brain.memory.iter_records():
        if record.source != 'knowledge_ingest' or not isinstance(record.data, dict):
            continue
        if record.data.get('kind') != 'knowledge_manifest':
            continue
        manifest = dict(record.data)
        manifest['importance'] = record.importance
        manifest['timestamp'] = record.timestamp
        manifest['_memory_key'] = record.key
        manifests.append(manifest)
    manifests.sort(key=lambda item: float(item.get('created_at', 0.0) or 0.0), reverse=True)
    return manifests


def _knowledge_chunk_records(brain: IntegratedArtificialBrain, *, ingest_id: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in brain.memory.iter_records():
        if record.source != 'episode' or not isinstance(record.data, dict):
            continue
        metadata = record.data.get('metadata') or {}
        if metadata.get('kind') != 'knowledge_chunk':
            continue
        if metadata.get('ingest_id') != ingest_id:
            continue
        chunks.append(
            {
                'key': record.key,
                'importance': record.importance,
                'timestamp': record.timestamp,
                'summary': record.data.get('summary') or record.data.get('text', '')[:200],
                'text_preview': clean_text(record.data.get('text', ''))[:260],
                'topics': record.data.get('topics', []),
                'tags': record.data.get('tags', []),
                'source_name': metadata.get('source_name'),
                'source_url': metadata.get('source_url'),
                'ingest_id': metadata.get('ingest_id'),
                'chunk_index': metadata.get('chunk_index'),
                'total_chunks': metadata.get('total_chunks'),
            }
        )
    chunks.sort(key=lambda item: (int(item.get('chunk_index') or 0), float(item.get('timestamp') or 0.0)))
    return chunks


def _counter_to_ranked_list(counter: Counter[str], *, limit: int = 10) -> list[dict[str, Any]]:
    return [
        {'name': name, 'count': count}
        for name, count in counter.most_common(max(1, int(limit)))
        if str(name).strip()
    ]



def get_knowledge_source_details(
    brain: IntegratedArtificialBrain,
    *,
    ingest_id: str,
    chunk_limit: int = 20,
    include_chunks: bool = True,
) -> dict[str, Any]:
    cleaned_ingest_id = clean_text(ingest_id)
    if not cleaned_ingest_id:
        raise ValueError('ingest_id مطلوب')
    manifests = _manifest_records(brain)
    manifest = next((item for item in manifests if item.get('ingest_id') == cleaned_ingest_id), None)
    if manifest is None:
        raise ValueError('مصدر المعرفة غير موجود')
    response_source = dict(manifest)
    response_source.pop('_memory_key', None)
    chunks = _knowledge_chunk_records(brain, ingest_id=cleaned_ingest_id) if include_chunks else []
    limited_chunks = chunks[:max(1, int(chunk_limit))] if include_chunks else []
    return {
        'source': response_source,
        'chunk_count': len(chunks) if include_chunks else int(response_source.get('chunk_count', 0) or 0),
        'returned_chunk_count': len(limited_chunks),
        'chunks': limited_chunks,
    }



def delete_knowledge_source(brain: IntegratedArtificialBrain, *, ingest_id: str) -> dict[str, Any]:
    cleaned_ingest_id = clean_text(ingest_id)
    if not cleaned_ingest_id:
        raise ValueError('ingest_id مطلوب')
    manifests = _manifest_records(brain)
    manifest = next((item for item in manifests if item.get('ingest_id') == cleaned_ingest_id), None)
    if manifest is None:
        raise ValueError('مصدر المعرفة غير موجود')
    manifest_key = str(manifest.get('_memory_key') or '')
    expected_chunk_keys = [str(item).strip() for item in manifest.get('chunk_keys', []) if str(item).strip()]
    fallback_chunk_keys = [item['key'] for item in _knowledge_chunk_records(brain, ingest_id=cleaned_ingest_id)]
    chunk_keys = list(dict.fromkeys(expected_chunk_keys + fallback_chunk_keys))
    deleted_chunk_keys = [key for key in chunk_keys if brain.memory.delete_memory(key)]
    deleted_manifest = brain.memory.delete_memory(manifest_key) if manifest_key else False
    return {
        'deleted': bool(deleted_manifest or deleted_chunk_keys),
        'ingest_id': cleaned_ingest_id,
        'source_name': manifest.get('source_name'),
        'deleted_manifest': deleted_manifest,
        'deleted_chunk_count': len(deleted_chunk_keys),
        'expected_chunk_count': len(chunk_keys),
        'deleted_chunk_keys': deleted_chunk_keys,
    }



def knowledge_analytics(brain: IntegratedArtificialBrain, *, limit: int = 10) -> dict[str, Any]:
    manifests = _manifest_records(brain)
    source_type_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    trust_band_counter: Counter[str] = Counter()
    total_characters = 0
    total_chunks = 0
    total_trust_score = 0.0

    recent_sources: list[dict[str, Any]] = []
    for manifest in manifests:
        metadata = manifest.get('metadata') or {}
        source_type = str(metadata.get('source_type') or ('url' if manifest.get('source_url') else 'text')).strip()
        source_type_counter.update([source_type or 'unknown'])
        total_characters += int(manifest.get('character_count', 0) or 0)
        total_chunks += int(manifest.get('chunk_count', 0) or 0)
        parsed = urlparse(str(manifest.get('source_url') or ''))
        if parsed.netloc:
            domain_counter.update([parsed.netloc])
        trust_band = str(manifest.get('trust_band') or 'unknown').strip() or 'unknown'
        trust_band_counter.update([trust_band])
        total_trust_score += float(manifest.get('trust_score', 0.0) or 0.0)
        for tag in manifest.get('tags', []) or []:
            tag_counter.update([clean_text(tag)])
        for topic in manifest.get('topics', []) or []:
            topic_counter.update([clean_text(topic)])
        recent_sources.append(
            {
                'ingest_id': manifest.get('ingest_id'),
                'source_name': manifest.get('source_name'),
                'source_url': manifest.get('source_url'),
                'chunk_count': manifest.get('chunk_count'),
                'character_count': manifest.get('character_count'),
                'created_at': manifest.get('created_at'),
                'source_type': source_type or 'unknown',
                'trust_score': manifest.get('trust_score'),
                'trust_band': manifest.get('trust_band'),
            }
        )

    total_sources = len(manifests)
    return {
        'total_sources': total_sources,
        'total_chunks': total_chunks,
        'total_characters': total_characters,
        'average_trust_score': round(total_trust_score / max(1, total_sources), 4),
        'by_source_type': _counter_to_ranked_list(source_type_counter, limit=limit),
        'top_domains': _counter_to_ranked_list(domain_counter, limit=limit),
        'top_tags': _counter_to_ranked_list(tag_counter, limit=limit),
        'top_topics': _counter_to_ranked_list(topic_counter, limit=limit),
        'trust_distribution': _counter_to_ranked_list(trust_band_counter, limit=limit),
        'recent_sources': recent_sources[:max(1, int(limit))],
    }



def _query_tokens(text: str | None) -> list[str]:
    normalized = _normalize_text(text or '')
    tokens = [
        token
        for token in re.findall(r'\w+', normalized, flags=re.UNICODE)
        if len(token) >= 3 and token not in _STOPWORDS
    ]
    return list(dict.fromkeys(tokens))



def _relevance_score(query_tokens: list[str], *fragments: Any, importance: float = 0.0) -> float:
    if not query_tokens:
        return round(float(np.clip(0.35 + (importance * 0.55), 0.0, 1.0)), 4)
    normalized_blob = _normalize_text(' '.join(clean_text(fragment) for fragment in fragments if fragment))
    if not normalized_blob:
        return 0.0
    unique_hits = sum(1 for token in query_tokens if token in normalized_blob)
    density = unique_hits / max(1, len(query_tokens))
    score = (density * 0.82) + (float(np.clip(importance, 0.0, 1.0)) * 0.18)
    return round(float(np.clip(score, 0.0, 1.0)), 4)



def _excerpt_sentences(text: str, *, limit: int = 2, sentence_chars: int = 220) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<=[\.\!؟\?])\s+|\n+', cleaned, flags=re.UNICODE)
        if sentence and sentence.strip()
    ]
    excerpts: list[str] = []
    for sentence in sentences:
        compact = sentence[:sentence_chars].strip()
        if compact and compact not in excerpts:
            excerpts.append(compact)
        if len(excerpts) >= max(1, int(limit)):
            break
    if not excerpts:
        excerpts.append(cleaned[:sentence_chars].strip())
    return excerpts





def _provenance_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get('metadata') or {}
    source_url = str(manifest.get('source_url') or '')
    parsed = urlparse(source_url) if source_url else None
    return {
        'ingest_id': manifest.get('ingest_id'),
        'source_name': manifest.get('source_name'),
        'source_url': source_url or None,
        'domain': parsed.netloc if parsed and parsed.netloc else None,
        'scheme': parsed.scheme if parsed and parsed.scheme else None,
        'source_type': str(metadata.get('source_type') or ('url' if source_url else 'text')).strip() or 'unknown',
        'content_type': metadata.get('content_type'),
        'status_code': metadata.get('status_code'),
        'title': metadata.get('title'),
        'created_at': manifest.get('created_at'),
        'fingerprint': manifest.get('fingerprint'),
    }



def _trust_policy_notes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    profile = manifest.get('source_profile') or {}
    signals = {str(item).strip() for item in profile.get('signals', []) if str(item).strip()}
    notes: list[dict[str, Any]] = []
    if 'institutional_domain' in signals or 'documentation_domain' in signals:
        notes.append(
            {
                'rule': 'prefer_authoritative_hosts',
                'applied': True,
                'reason': 'المصدر صادر من نطاق مؤسسي أو توثيقي وتم رفع وزنه.',
            }
        )
    if 'community_or_blog_domain' in signals or 'casual_title_hint' in signals:
        notes.append(
            {
                'rule': 'downgrade_casual_or_community_sources',
                'applied': True,
                'reason': 'المصدر يبدو مجتمعياً/غير رسمي ولذلك يحتاج تأكيداً إضافياً.',
            }
        )
    if 'thin_content' in signals:
        notes.append(
            {
                'rule': 'penalize_thin_content',
                'applied': True,
                'reason': 'المحتوى قصير نسبياً وتم خفض الثقة تلقائياً.',
            }
        )
    if not notes:
        notes.append(
            {
                'rule': 'balanced_default_ranking',
                'applied': False,
                'reason': 'لا توجد تعديلات استثنائية على الثقة بخلاف التقييم الأساسي.',
            }
        )
    return notes



def _confidence_label(score: float) -> str:
    clipped = float(np.clip(score, 0.0, 1.0))
    if clipped >= 0.78:
        return 'high'
    if clipped >= 0.58:
        return 'medium'
    return 'low'



def _confidence_explanation(
    *,
    verification_status: str,
    avg_trust: float,
    source_count: int,
    evidence_count: int,
    contradiction_count: int,
) -> dict[str, Any]:
    raw_score = (avg_trust * 0.55) + (min(source_count, 4) / 4.0 * 0.2) + (min(evidence_count, 6) / 6.0 * 0.25)
    if contradiction_count:
        raw_score -= min(0.28, contradiction_count * 0.08)
    if verification_status == 'well_supported':
        raw_score += 0.08
    elif verification_status in {'no_knowledge', 'no_match'}:
        raw_score = 0.0
    score = round(float(np.clip(raw_score, 0.0, 1.0)), 4)
    reasons: list[str] = []
    if source_count >= 2:
        reasons.append('التحقق اعتمد على أكثر من مصدر بدلاً من مصدر منفرد.')
    if avg_trust >= 0.72:
        reasons.append('متوسط الثقة مرتفع نسبياً مقارنة بالمصادر المتاحة.')
    elif avg_trust > 0:
        reasons.append('متوسط الثقة متوسط ويحتاج تدعيم بمصادر أوثق.')
    if contradiction_count:
        reasons.append('تم خفض الثقة بسبب وجود تعارضات بين أفضل المقاطع المطابقة.')
    if evidence_count >= 4:
        reasons.append('عدد الأدلة المطابقة جيد ويساعد على تثبيت الاستنتاج.')
    if not reasons:
        reasons.append('المعرفة الحالية غير كافية لتوليد درجة ثقة عالية.')
    return {
        'score': score,
        'label': _confidence_label(score),
        'reasons': reasons[:4],
    }



def _cluster_contradictions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        reason = clean_text(candidate.get('reason') or 'general_conflict') or 'general_conflict'
        bucket = grouped.setdefault(
            reason,
            {
                'reason': reason,
                'count': 0,
                'max_score': 0.0,
                'sources': [],
                'examples': [],
            },
        )
        bucket['count'] += 1
        bucket['max_score'] = max(float(bucket['max_score']), float(candidate.get('score', 0.0) or 0.0))
        for source_name in (candidate.get('left_source_name'), candidate.get('right_source_name')):
            cleaned_name = clean_text(source_name or '')
            if cleaned_name and cleaned_name not in bucket['sources']:
                bucket['sources'].append(cleaned_name)
        if len(bucket['examples']) < 2:
            bucket['examples'].append(
                {
                    'left_excerpt': candidate.get('left_excerpt'),
                    'right_excerpt': candidate.get('right_excerpt'),
                }
            )
    clusters = list(grouped.values())
    clusters.sort(key=lambda item: (int(item.get('count', 0)), float(item.get('max_score', 0.0))), reverse=True)
    return clusters[:4]



def _consensus_summary(
    query: str,
    selected_sources: list[dict[str, Any]],
    contradiction_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not selected_sources:
        return {
            'state': 'unresolved',
            'summary': 'لا يوجد إجماع لأن الاستعلام لم يجد مصادر كافية مرتبطة به.',
            'top_topics': [],
            'top_sources': [],
        }
    topic_counter: Counter[str] = Counter()
    top_sources = [str(item.get('source_name') or '') for item in selected_sources[:3] if str(item.get('source_name') or '').strip()]
    for item in selected_sources:
        for topic in item.get('topics', []) or []:
            cleaned_topic = clean_text(topic)
            if cleaned_topic:
                topic_counter.update([cleaned_topic])
    top_topics = [name for name, _ in topic_counter.most_common(4)]
    lead_excerpt = clean_text(selected_sources[0].get('best_excerpt') or '')
    if contradiction_candidates:
        state = 'contested'
        summary = (
            f'أفضل المصادر المطابقة للاستعلام "{query}" متفقة جزئياً فقط؛ '
            f'المعلومة الأساسية تظهر في {", ".join(top_sources[:2])} لكن توجد تعارضات تحتاج مراجعة بشرية.'
        )
    elif len(selected_sources) >= 2:
        state = 'aligned'
        summary = (
            f'يوجد تقارب واضح بين المصادر الأعلى صلة حول "{query}"، '
            f'وأبرزها {", ".join(top_sources[:2])}. {lead_excerpt[:180]}'
        ).strip()
    else:
        state = 'emerging'
        summary = f'يوجد اتجاه مبدئي من مصدر رئيسي واحد حول "{query}", لكن يلزم تدعيمه بمصادر إضافية.'
    return {
        'state': state,
        'summary': summary,
        'top_topics': top_topics,
        'top_sources': top_sources,
    }



def knowledge_verify(
    brain: IntegratedArtificialBrain,
    *,
    query: str,
    source_limit: int = 5,
    evidence_limit: int = 8,
) -> dict[str, Any]:
    cleaned_query = clean_text(query)
    if len(cleaned_query) < 3:
        raise ValueError('query مطلوب للتحقق المعرفي')

    manifests = _manifest_records(brain)
    if not manifests:
        return {
            'query': cleaned_query,
            'verification_status': 'no_knowledge',
            'summary': 'لا توجد مصادر معرفة كافية للتحقق حالياً.',
            'matched_sources': 0,
            'matched_evidence': 0,
            'trust_overview': {'average_trust_score': 0.0, 'high_trust_sources': 0},
            'confidence_explanation': {'score': 0.0, 'label': 'low', 'reasons': ['ابدأ أولاً بإدخال مصادر قابلة للتحقق.']},
            'consensus_summary': {
                'state': 'unresolved',
                'summary': 'لا يمكن بناء إجماع بدون corpus معرفي.',
                'top_topics': [],
                'top_sources': [],
            },
            'provenance_chain': [],
            'evidence': [],
            'contradiction_candidates': [],
            'contradiction_clusters': [],
            'recommended_followups': ['ابدأ بإدخال مصادر نصية أو روابط قبل طلب التحقق.'],
        }

    query_tokens = _query_tokens(cleaned_query)
    evidence_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for manifest in manifests:
        chunks = _knowledge_chunk_records(brain, ingest_id=str(manifest.get('ingest_id') or ''))
        source_best_score = 0.0
        source_best_excerpt = manifest.get('preview')
        source_profile = manifest.get('source_profile') or {}
        provenance = _provenance_snapshot(manifest)
        trust_policy_notes = _trust_policy_notes(manifest)
        for chunk in chunks:
            relevance = _relevance_score(
                query_tokens,
                chunk.get('summary'),
                chunk.get('text_preview'),
                chunk.get('source_name'),
                chunk.get('topics'),
                importance=float(chunk.get('importance', 0.0) or 0.0),
            )
            trust_score = float(manifest.get('trust_score', 0.0) or 0.0)
            blended_score = round(float(np.clip((relevance * 0.78) + (trust_score * 0.22), 0.0, 1.0)), 4)
            if blended_score <= 0.0:
                continue
            candidate_excerpt = chunk.get('text_preview') or chunk.get('summary') or ''
            if (
                blended_score > source_best_score
                or (
                    blended_score == source_best_score
                    and len(clean_text(candidate_excerpt)) > len(clean_text(source_best_excerpt or ''))
                )
            ):
                source_best_score = blended_score
                source_best_excerpt = candidate_excerpt or source_best_excerpt
            evidence_rows.append(
                {
                    'ingest_id': chunk.get('ingest_id'),
                    'source_name': chunk.get('source_name'),
                    'source_url': chunk.get('source_url'),
                    'chunk_key': chunk.get('key'),
                    'chunk_index': chunk.get('chunk_index'),
                    'score': blended_score,
                    'relevance_score': relevance,
                    'trust_score': trust_score,
                    'trust_band': manifest.get('trust_band'),
                    'excerpt': chunk.get('text_preview') or chunk.get('summary') or '',
                    'topics': chunk.get('topics', []),
                    'trust_signals': source_profile.get('signals', []),
                    'provenance': provenance,
                }
            )
        if source_best_score > 0.0:
            source_rows.append(
                {
                    'ingest_id': manifest.get('ingest_id'),
                    'source_name': manifest.get('source_name'),
                    'source_url': manifest.get('source_url'),
                    'score': source_best_score,
                    'trust_score': manifest.get('trust_score', 0.0),
                    'trust_band': manifest.get('trust_band'),
                    'best_excerpt': source_best_excerpt,
                    'topics': manifest.get('topics', []),
                    'trust_signals': source_profile.get('signals', []),
                    'trust_policy_notes': trust_policy_notes,
                    'provenance': provenance,
                }
            )

    source_rows.sort(key=lambda item: (float(item.get('score', 0.0)), float(item.get('trust_score', 0.0))), reverse=True)
    evidence_rows.sort(key=lambda item: (float(item.get('score', 0.0)), float(item.get('trust_score', 0.0))), reverse=True)
    selected_sources = source_rows[:max(1, int(source_limit))]
    selected_evidence = evidence_rows[:max(1, int(evidence_limit))]

    contradiction_candidates: list[dict[str, Any]] = []
    for index, left in enumerate(selected_sources):
        for right in selected_sources[index + 1:]:
            if left.get('ingest_id') == right.get('ingest_id'):
                continue
            best_signal: tuple[float, str] | None = None
            best_pair: tuple[str, str] | None = None
            for left_sentence in _split_into_sentences(str(left.get('best_excerpt') or ''))[:3]:
                for right_sentence in _split_into_sentences(str(right.get('best_excerpt') or ''))[:3]:
                    signal = _contradiction_signal(left_sentence, right_sentence)
                    if signal and (best_signal is None or signal[0] > best_signal[0]):
                        best_signal = signal
                        best_pair = (left_sentence, right_sentence)
            if best_signal and best_pair:
                contradiction_candidates.append(
                    {
                        'score': best_signal[0],
                        'reason': best_signal[1],
                        'left_source_name': left.get('source_name'),
                        'left_source_url': left.get('source_url'),
                        'left_excerpt': best_pair[0],
                        'right_source_name': right.get('source_name'),
                        'right_source_url': right.get('source_url'),
                        'right_excerpt': best_pair[1],
                    }
                )

    contradiction_candidates.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
    contradiction_candidates = contradiction_candidates[:5]
    contradiction_clusters = _cluster_contradictions(contradiction_candidates)

    avg_trust = round(
        sum(float(item.get('trust_score', 0.0) or 0.0) for item in selected_sources) / max(1, len(selected_sources)),
        4,
    )
    high_trust_sources = sum(1 for item in selected_sources if float(item.get('trust_score', 0.0) or 0.0) >= 0.75)

    if not selected_sources:
        verification_status = 'no_match'
        summary = 'لم يتم العثور على أدلة مرتبطة مباشرة بالاستعلام داخل المعرفة الحالية.'
    elif contradiction_candidates:
        verification_status = 'mixed_evidence'
        summary = 'تم العثور على أدلة مرتبطة بالاستعلام لكن توجد إشارات تعارض تستحق مراجعة بشرية أو مصادر أوثق.'
    elif high_trust_sources >= 2 and avg_trust >= 0.7:
        verification_status = 'well_supported'
        summary = 'الاستعلام مدعوم بعدة مصادر جيدة الثقة داخل corpus الحالي بدون تعارضات بارزة.'
    else:
        verification_status = 'limited_evidence'
        summary = 'فيه أدلة مبدئية، لكن الثقة أو التغطية ما زالت محدودة وتحتاج مصادر إضافية.'

    recommended_followups: list[str] = []
    if contradiction_candidates:
        recommended_followups.append('راجع المقاطع المتعارضة وحدد المصدر الأوثق قبل الاعتماد على الاستنتاج.')
    if high_trust_sources < 2:
        recommended_followups.append('أضف مصدرين موثوقين على الأقل لرفع جودة التحقق.')
    if selected_sources:
        recommended_followups.append(f'ابدأ بمراجعة المصدر الأعلى ترتيباً: {selected_sources[0]["source_name"]}.')

    confidence_explanation = _confidence_explanation(
        verification_status=verification_status,
        avg_trust=avg_trust,
        source_count=len(selected_sources),
        evidence_count=len(selected_evidence),
        contradiction_count=len(contradiction_candidates),
    )
    consensus_summary = _consensus_summary(cleaned_query, selected_sources, contradiction_candidates)

    return {
        'query': cleaned_query,
        'query_tokens': query_tokens,
        'verification_status': verification_status,
        'summary': summary,
        'matched_sources': len(source_rows),
        'matched_evidence': len(evidence_rows),
        'trust_overview': {
            'average_trust_score': avg_trust,
            'high_trust_sources': high_trust_sources,
        },
        'confidence_explanation': confidence_explanation,
        'consensus_summary': consensus_summary,
        'provenance_chain': [item.get('provenance') for item in selected_sources],
        'sources': [
            {
                **item,
                'best_excerpt': (_excerpt_sentences(str(item.get('best_excerpt') or ''), limit=1, sentence_chars=220) or [''])[0],
                'confidence_label': _confidence_label(float(item.get('score', 0.0) or 0.0)),
            }
            for item in selected_sources
        ],
        'evidence': [
            {
                **item,
                'excerpt': (_excerpt_sentences(str(item.get('excerpt') or ''), limit=1, sentence_chars=220) or [''])[0],
                'confidence_label': _confidence_label(float(item.get('score', 0.0) or 0.0)),
            }
            for item in selected_evidence
        ],
        'contradiction_candidates': contradiction_candidates,
        'contradiction_clusters': contradiction_clusters,
        'recommended_followups': recommended_followups[:3],
    }


def knowledge_briefing(
    brain: IntegratedArtificialBrain,
    *,
    query: str | None = None,
    source_limit: int = 4,
    highlight_limit: int = 6,
) -> dict[str, Any]:
    manifests = _manifest_records(brain)
    cleaned_query = clean_text(query or '') or None
    query_tokens = _query_tokens(cleaned_query)
    resolved_source_limit = max(1, int(source_limit))
    resolved_highlight_limit = max(1, int(highlight_limit))
    if not manifests:
        return {
            'query': cleaned_query,
            'total_sources': 0,
            'focus_topics': [],
            'source_snapshots': [],
            'highlights': [],
            'briefing': {
                'headline': 'لا توجد معرفة مُدخلة بعد',
                'summary_points': ['ابدأ بإدخال نصوص أو روابط قبل طلب briefing تحليلي.'],
                'recommended_questions': ['ما أول مصدر تحب تدخله للنظام؟'],
                'coverage_ratio': 0.0,
            },
        }

    source_rows: list[dict[str, Any]] = []
    highlight_rows: list[dict[str, Any]] = []
    focus_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    total_chunks = 0

    for manifest in manifests:
        chunks = _knowledge_chunk_records(brain, ingest_id=str(manifest.get('ingest_id') or ''))
        total_chunks += len(chunks)
        metadata = manifest.get('metadata') or {}
        source_type = str(metadata.get('source_type') or ('url' if manifest.get('source_url') else 'text')).strip() or 'unknown'
        source_type_counter.update([source_type])
        parsed = urlparse(str(manifest.get('source_url') or ''))
        if parsed.netloc:
            domain_counter.update([parsed.netloc])

        best_chunk: dict[str, Any] | None = None
        best_chunk_score = 0.0
        for chunk in chunks:
            chunk_score = _relevance_score(
                query_tokens,
                chunk.get('summary'),
                chunk.get('text_preview'),
                chunk.get('source_name'),
                chunk.get('topics'),
                importance=float(chunk.get('importance', 0.0) or 0.0),
            )
            if not query_tokens or chunk_score > 0.0:
                highlight_rows.append(
                    {
                        'ingest_id': chunk.get('ingest_id'),
                        'source_name': chunk.get('source_name'),
                        'source_url': chunk.get('source_url'),
                        'chunk_key': chunk.get('key'),
                        'chunk_index': chunk.get('chunk_index'),
                        'total_chunks': chunk.get('total_chunks'),
                        'score': chunk_score,
                        'importance': chunk.get('importance'),
                        'excerpt': chunk.get('text_preview') or chunk.get('summary'),
                        'topics': chunk.get('topics', []),
                    }
                )
            if chunk_score >= best_chunk_score:
                best_chunk_score = chunk_score
                best_chunk = chunk

        source_score = _relevance_score(
            query_tokens,
            manifest.get('source_name'),
            manifest.get('preview'),
            manifest.get('topics'),
            manifest.get('tags'),
            importance=float(manifest.get('importance', 0.0) or 0.0),
        )
        if best_chunk_score > source_score:
            source_score = best_chunk_score

        if query_tokens and source_score <= 0.0:
            continue

        for topic in manifest.get('topics', []) or []:
            cleaned_topic = clean_text(topic)
            if cleaned_topic:
                focus_counter.update([cleaned_topic])
        source_rows.append(
            {
                'ingest_id': manifest.get('ingest_id'),
                'source_name': manifest.get('source_name'),
                'source_url': manifest.get('source_url'),
                'source_type': source_type,
                'chunk_count': manifest.get('chunk_count'),
                'character_count': manifest.get('character_count'),
                'topics': manifest.get('topics', []),
                'tags': manifest.get('tags', []),
                'preview': manifest.get('preview'),
                'created_at': manifest.get('created_at'),
                'score': source_score,
                'best_chunk_index': best_chunk.get('chunk_index') if best_chunk else None,
                'best_chunk_excerpt': best_chunk.get('text_preview') if best_chunk else None,
            }
        )

    if not source_rows:
        return {
            'query': cleaned_query,
            'total_sources': len(manifests),
            'focus_topics': [],
            'source_snapshots': [],
            'highlights': [],
            'briefing': {
                'headline': 'لا توجد نتائج مطابقة للاستعلام الحالي',
                'summary_points': ['جرّب كلمات أدق أو أوسع علشان briefing يطلع بنتائج أقوى.'],
                'recommended_questions': ['هل تحب توسّع الاستعلام أو تلغيه؟'],
                'coverage_ratio': 0.0,
            },
        }

    source_rows.sort(key=lambda item: (float(item.get('score', 0.0)), float(item.get('created_at', 0.0) or 0.0)), reverse=True)
    highlight_rows.sort(key=lambda item: (float(item.get('score', 0.0)), float(item.get('importance', 0.0) or 0.0)), reverse=True)
    focus_topics = _counter_to_ranked_list(focus_counter, limit=6)
    selected_sources = source_rows[:resolved_source_limit]
    selected_highlights = highlight_rows[:resolved_highlight_limit]
    coverage_ratio = round(len(source_rows) / max(1, len(manifests)), 4)

    dominant_topic = focus_topics[0]['name'] if focus_topics else None
    dominant_source_type = source_type_counter.most_common(1)[0][0] if source_type_counter else 'unknown'
    dominant_domain = domain_counter.most_common(1)[0][0] if domain_counter else None
    headline = 'ملخص معرفي سريع للمصادر الحالية'
    if cleaned_query:
        headline = f'Briefing معرفي للاستعلام: {cleaned_query}'
    elif dominant_topic:
        headline = f'ملخص معرفي سريع حول: {dominant_topic}'

    summary_points: list[str] = []
    if dominant_topic:
        summary_points.append(f'أكتر محور ظاهر حالياً هو «{dominant_topic}» وظهر {focus_topics[0]["count"] if isinstance(focus_topics[0], tuple) else focus_topics[0]["count"]} مرات عبر المصادر المختارة.')
    summary_points.append(f'تمت مراجعة {len(source_rows)} مصدر معرفي بإجمالي {total_chunks} مقطع، والنمط الغالب للمصادر هو {dominant_source_type}.')
    if dominant_domain:
        summary_points.append(f'أكتر نطاق متكرر بين الروابط الحالية هو {dominant_domain}.')
    if cleaned_query and selected_sources:
        summary_points.append(f'أعلى مصدر صلة بالاستعلام هو «{selected_sources[0]["source_name"]}» بدرجة {selected_sources[0]["score"]:.2f}.')
    elif selected_sources:
        summary_points.append(f'أحدث مصدر بارز في الملخص هو «{selected_sources[0]["source_name"]}» وبيغطي {selected_sources[0]["chunk_count"]} مقاطع.')

    recommended_questions: list[str] = []
    if len(focus_topics) >= 2:
        recommended_questions.append(f'إزاي يرتبط {focus_topics[0]["name"]} مع {focus_topics[1]["name"]} داخل نفس corpus؟')
    if focus_topics:
        recommended_questions.append(f'إيه التفاصيل الناقصة أو الحالات الحدّية المرتبطة بموضوع {focus_topics[0]["name"]}؟')
    if selected_sources:
        recommended_questions.append(f'هل نحتاج مصدر إضافي يؤكد أو يراجع ما جاء في {selected_sources[0]["source_name"]}؟')

    return {
        'query': cleaned_query,
        'query_tokens': query_tokens,
        'total_sources': len(manifests),
        'matched_sources': len(source_rows),
        'focus_topics': focus_topics,
        'source_snapshots': selected_sources,
        'highlights': [
            {
                **item,
                'excerpt': _excerpt_sentences(str(item.get('excerpt') or ''), limit=1, sentence_chars=220)[0],
            }
            for item in selected_highlights
        ],
        'briefing': {
            'headline': headline,
            'summary_points': summary_points[:4],
            'recommended_questions': recommended_questions[:3],
            'coverage_ratio': coverage_ratio,
        },
    }
