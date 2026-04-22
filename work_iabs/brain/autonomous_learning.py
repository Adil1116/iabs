from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx

from brain import knowledge_ingestion
from brain.system import IntegratedArtificialBrain


logger = logging.getLogger('iabs.autonomous_learning')


def _clean_text(value: Any) -> str:
    text = str(value or '').strip()
    text = re.sub(r'\s+', ' ', text, flags=re.UNICODE)
    return text.strip()


def _normalize_keywords(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_items = re.split(r'[,\n]+', values)
    else:
        raw_items = list(values)
    ordered: list[str] = []
    for item in raw_items:
        candidate = _clean_text(item).lower()
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered[:12]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def _safe_host(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower().strip()
    return host or None


def _safe_url(url: str, base_url: str | None = None) -> str | None:
    candidate = _clean_text(url)
    if not candidate:
        return None
    if base_url:
        candidate = urljoin(base_url, candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return None
    return candidate


def _html_anchor_candidates(markup: str, source_url: str, keywords: list[str], max_candidates: int = 40) -> list[dict[str, Any]]:
    host = _safe_host(source_url)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = re.compile(r'<a\b[^>]*href=["\'](?P<href>[^"\']+)["\'][^>]*>(?P<label>.*?)</a>', flags=re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(markup or ''):
        href = _safe_url(match.group('href'), source_url)
        if not href or href in seen:
            continue
        href_host = _safe_host(href)
        if host and href_host and href_host != host:
            continue
        label = _clean_text(re.sub(r'<[^>]+>', ' ', match.group('label')))
        searchable = f'{label} {href}'.lower()
        keyword_hits = sum(1 for keyword in keywords if keyword and keyword in searchable)
        topical_bonus = 0.25 if re.search(r'/20\d{2}/|article|news|blog|post|docs|guide', href, flags=re.IGNORECASE) else 0.0
        score = (keyword_hits * 1.4) + topical_bonus + (0.15 if label else 0.0)
        results.append({
            'url': href,
            'title': label or href.rsplit('/', maxsplit=1)[-1],
            'score': round(score, 4),
            'discovered_via': 'html',
        })
        seen.add(href)
    results.sort(key=lambda item: (item.get('score', 0.0), len(item.get('title', ''))), reverse=True)
    return results[:max_candidates]


def _parse_feed_xml(raw_text: str, source_url: str, keywords: list[str], max_candidates: int = 40) -> list[dict[str, Any]]:
    text = _clean_text(raw_text)
    if not text or ('<rss' not in text.lower() and '<feed' not in text.lower() and '<rdf' not in text.lower()):
        return []
    try:
        root = ET.fromstring(raw_text)
    except ET.ParseError:
        return []

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in root.findall('.//item') + root.findall('.//{*}entry'):
        title = _clean_text(item.findtext('title') or item.findtext('{*}title') or '')
        link_text = _clean_text(item.findtext('link') or item.findtext('{*}link') or '')
        if not link_text:
            link_node = item.find('link') or item.find('{*}link')
            if link_node is not None:
                link_text = _clean_text(link_node.attrib.get('href') or link_node.text or '')
        href = _safe_url(link_text, source_url)
        if not href or href in seen:
            continue
        published = _clean_text(
            item.findtext('pubDate')
            or item.findtext('{*}published')
            or item.findtext('{*}updated')
            or item.findtext('updated')
            or ''
        )
        searchable = f'{title} {href}'.lower()
        keyword_hits = sum(1 for keyword in keywords if keyword and keyword in searchable)
        score = (keyword_hits * 1.6) + (0.25 if published else 0.0) + 0.3
        entries.append({
            'url': href,
            'title': title or href.rsplit('/', maxsplit=1)[-1],
            'published_at': published or None,
            'score': round(score, 4),
            'discovered_via': 'rss',
        })
        seen.add(href)
    entries.sort(key=lambda item: (item.get('score', 0.0), bool(item.get('published_at'))), reverse=True)
    return entries[:max_candidates]


async def discover_source_candidates(source: dict[str, Any], *, timeout_seconds: float = 20.0) -> list[dict[str, Any]]:
    feed_url = str(source.get('feed_url') or '').strip()
    if not feed_url:
        return []
    mode = str(source.get('mode') or 'auto').strip().lower()
    keywords = _normalize_keywords(list(source.get('keywords') or []) + list(source.get('learned_keywords') or []))
    headers = {
        'User-Agent': 'IABS Autonomous Learning/2.16',
        'Accept': 'application/rss+xml,application/atom+xml,text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.1',
    }
    async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        response = await client.get(feed_url)
        response.raise_for_status()
        raw_text = response.text
        final_url = str(response.url)

    if mode in {'auto', 'rss'}:
        rss_candidates = _parse_feed_xml(raw_text, final_url, keywords, max_candidates=max(10, int(source.get('max_items_per_run', 5)) * 4))
        if rss_candidates or mode == 'rss':
            return rss_candidates

    if mode in {'auto', 'html'}:
        html_candidates = _html_anchor_candidates(raw_text, final_url, keywords, max_candidates=max(10, int(source.get('max_items_per_run', 5)) * 4))
        if html_candidates or mode == 'html':
            return html_candidates

    fallback_url = _safe_url(final_url)
    if fallback_url and mode in {'auto', 'page'}:
        return [{
            'url': fallback_url,
            'title': _clean_text(source.get('name') or fallback_url),
            'score': 0.5,
            'discovered_via': 'page',
        }]
    return []


class KnowledgeAutomationService:
    def __init__(
        self,
        registry_path: str | Path,
        *,
        enabled: bool = True,
        poll_interval_seconds: int = 300,
    ) -> None:
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(enabled)
        self.poll_interval_seconds = max(15, int(poll_interval_seconds))
        self.state: dict[str, Any] = {
            'version': '2.16.0',
            'sources': {},
            'recent_runs': [],
            'topic_profiles': {},
            'scheduler': {
                'last_tick_at': None,
                'last_run_at': None,
            },
        }
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.load()

    def load(self) -> None:
        if not self.registry_path.exists():
            self.save()
            return
        try:
            payload = json.loads(self.registry_path.read_text(encoding='utf-8'))
            if isinstance(payload, dict):
                self.state.update(payload)
        except Exception as exc:
            logger.warning('Failed to load knowledge automation state: %s', exc)

    def save(self) -> None:
        snapshot = json.dumps(self.state, ensure_ascii=False, indent=2)
        self.registry_path.write_text(snapshot, encoding='utf-8')

    def _recent_runs(self) -> list[dict[str, Any]]:
        runs = self.state.setdefault('recent_runs', [])
        if not isinstance(runs, list):
            runs = []
            self.state['recent_runs'] = runs
        return runs

    def _sources(self) -> dict[str, dict[str, Any]]:
        sources = self.state.setdefault('sources', {})
        if not isinstance(sources, dict):
            sources = {}
            self.state['sources'] = sources
        return sources

    def _topic_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = self.state.setdefault('topic_profiles', {})
        if not isinstance(profiles, dict):
            profiles = {}
            self.state['topic_profiles'] = profiles
        return profiles

    def status(self) -> dict[str, Any]:
        sources = list(self._sources().values())
        topic_profiles = self._topic_profiles()
        top_topics = sorted(
            (
                {
                    'topic': topic,
                    'weight': float(item.get('weight', 0.0)),
                    'ingested_count': int(item.get('ingested_count', 0)),
                    'last_seen_at': item.get('last_seen_at'),
                }
                for topic, item in topic_profiles.items()
            ),
            key=lambda item: (item['weight'], item['ingested_count']),
            reverse=True,
        )[:10]
        return {
            'enabled': self.enabled,
            'poll_interval_seconds': self.poll_interval_seconds,
            'source_count': len(sources),
            'active_source_count': sum(1 for item in sources if item.get('active', True)),
            'scheduler': self.state.get('scheduler', {}),
            'recent_runs': self._recent_runs()[-10:],
            'top_topics': top_topics,
        }

    def list_sources(self) -> dict[str, Any]:
        sources = sorted(self._sources().values(), key=lambda item: item.get('updated_at') or 0.0, reverse=True)
        return {
            'count': len(sources),
            'sources': sources,
        }

    def get_source(self, source_id: str) -> dict[str, Any]:
        source = self._sources().get(source_id)
        if not source:
            raise ValueError('مصدر التعلم التلقائي غير موجود')
        return source

    def register_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        source_id = f'autosrc_{uuid4().hex[:12]}'
        source = {
            'source_id': source_id,
            'name': _clean_text(payload.get('name') or 'Untitled Source'),
            'feed_url': _clean_text(payload.get('feed_url')),
            'mode': (_clean_text(payload.get('mode') or 'auto') or 'auto').lower(),
            'tags': _normalize_keywords(payload.get('tags')),
            'keywords': _normalize_keywords(payload.get('keywords')),
            'learned_keywords': _normalize_keywords(payload.get('learned_keywords')),
            'active': bool(payload.get('active', True)),
            'interval_seconds': max(300, int(payload.get('interval_seconds', 86400))),
            'max_items_per_run': max(1, min(12, int(payload.get('max_items_per_run', 5)))),
            'importance': _clamp(float(payload.get('importance', 0.77)), 0.3, 0.95),
            'last_run_at': None,
            'last_success_at': None,
            'created_at': now,
            'updated_at': now,
            'seen_urls': [],
            'stats': {
                'runs': 0,
                'successes': 0,
                'failures': 0,
                'items_seen': 0,
                'items_ingested': 0,
                'duplicate_items': 0,
                'skipped_seen': 0,
                'source_score': 0.5,
            },
            'last_summary': None,
        }
        if source['mode'] not in {'auto', 'rss', 'html', 'page'}:
            raise ValueError('mode يجب أن تكون auto أو rss أو html أو page')
        if not _safe_url(source['feed_url']):
            raise ValueError('feed_url لازم يكون http أو https صالح')
        self._sources()[source_id] = source
        self.save()
        return source

    def update_source(self, source_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        source = self.get_source(source_id)
        if 'name' in updates and _clean_text(updates.get('name')):
            source['name'] = _clean_text(updates['name'])
        if 'feed_url' in updates and updates.get('feed_url') is not None:
            feed_url = _clean_text(updates['feed_url'])
            if not _safe_url(feed_url):
                raise ValueError('feed_url لازم يكون http أو https صالح')
            source['feed_url'] = feed_url
        if 'mode' in updates and updates.get('mode') is not None:
            mode = _clean_text(updates['mode']).lower()
            if mode not in {'auto', 'rss', 'html', 'page'}:
                raise ValueError('mode يجب أن تكون auto أو rss أو html أو page')
            source['mode'] = mode
        if 'tags' in updates and updates.get('tags') is not None:
            source['tags'] = _normalize_keywords(updates['tags'])
        if 'keywords' in updates and updates.get('keywords') is not None:
            source['keywords'] = _normalize_keywords(updates['keywords'])
        if 'active' in updates and updates.get('active') is not None:
            source['active'] = bool(updates['active'])
        if 'interval_seconds' in updates and updates.get('interval_seconds') is not None:
            source['interval_seconds'] = max(300, int(updates['interval_seconds']))
        if 'max_items_per_run' in updates and updates.get('max_items_per_run') is not None:
            source['max_items_per_run'] = max(1, min(12, int(updates['max_items_per_run'])))
        if 'importance' in updates and updates.get('importance') is not None:
            source['importance'] = _clamp(float(updates['importance']), 0.3, 0.95)
        source['updated_at'] = time.time()
        self.save()
        return source

    def delete_source(self, source_id: str) -> dict[str, Any]:
        source = self._sources().pop(source_id, None)
        if source is None:
            raise ValueError('مصدر التعلم التلقائي غير موجود')
        self.save()
        return source

    def _remember_topics(self, source: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
        topics = [topic for topic in manifest.get('topics', []) if _clean_text(topic)]
        if not topics:
            return []
        learned_keywords = _normalize_keywords(list(source.get('learned_keywords') or []) + topics)
        source['learned_keywords'] = learned_keywords[:12]
        profiles = self._topic_profiles()
        now = time.time()
        for topic in topics:
            profile = profiles.setdefault(topic, {'weight': 0.0, 'ingested_count': 0, 'source_ids': [], 'last_seen_at': None})
            profile['weight'] = round(float(profile.get('weight', 0.0)) + 1.0, 4)
            profile['ingested_count'] = int(profile.get('ingested_count', 0)) + 1
            profile['last_seen_at'] = now
            if source['source_id'] not in profile.get('source_ids', []):
                profile.setdefault('source_ids', []).append(source['source_id'])
                profile['source_ids'] = profile['source_ids'][:10]
        return topics[:8]

    def _update_source_score(self, source: dict[str, Any], summary: dict[str, Any]) -> float:
        stats = source.setdefault('stats', {})
        runs = max(1, int(stats.get('runs', 0)))
        successes = int(stats.get('successes', 0))
        failures = int(stats.get('failures', 0))
        items_ingested = int(stats.get('items_ingested', 0))
        duplicates = int(stats.get('duplicate_items', 0))
        items_seen = max(1, int(stats.get('items_seen', 0)))
        success_ratio = successes / runs
        ingest_ratio = items_ingested / items_seen
        duplicate_penalty = duplicates / items_seen
        freshness_bonus = 0.2 if summary.get('ingested', 0) > 0 else 0.0
        failure_penalty = min(0.35, failures / max(1, runs) * 0.35)
        score = 0.32 + (success_ratio * 0.28) + (ingest_ratio * 0.28) - (duplicate_penalty * 0.18) - failure_penalty + freshness_bonus
        stats['source_score'] = round(_clamp(score, 0.05, 0.99), 4)
        return float(stats['source_score'])

    async def run_source(self, source_id: str, brain: IntegratedArtificialBrain, brain_lock: Lock, *, force: bool = False) -> dict[str, Any]:
        source = self.get_source(source_id)
        if not source.get('active', True) and not force:
            raise ValueError('المصدر غير مفعل حالياً')
        now = time.time()
        if not force and source.get('last_run_at') and (now - float(source['last_run_at'])) < int(source.get('interval_seconds', 86400)):
            return {
                'source_id': source_id,
                'status': 'skipped_not_due',
                'seconds_until_next_run': int(source.get('interval_seconds', 86400) - (now - float(source['last_run_at']))),
                'source': source,
            }

        run_id = f'auto_run_{uuid4().hex[:12]}'
        summary = {
            'run_id': run_id,
            'source_id': source_id,
            'source_name': source.get('name'),
            'started_at': now,
            'discovered': 0,
            'processed': 0,
            'ingested': 0,
            'duplicate_items': 0,
            'skipped_seen': 0,
            'errors': [],
            'topics': [],
            'status': 'ok',
        }
        stats = source.setdefault('stats', {})
        stats['runs'] = int(stats.get('runs', 0)) + 1

        try:
            candidates = await discover_source_candidates(source)
            summary['discovered'] = len(candidates)
            seen_urls = list(source.get('seen_urls') or [])
            seen_set = set(seen_urls)
            for candidate in candidates[: max(1, int(source.get('max_items_per_run', 5)))]:
                candidate_url = _safe_url(str(candidate.get('url') or ''))
                if not candidate_url:
                    continue
                stats['items_seen'] = int(stats.get('items_seen', 0)) + 1
                if candidate_url in seen_set:
                    summary['skipped_seen'] += 1
                    stats['skipped_seen'] = int(stats.get('skipped_seen', 0)) + 1
                    continue
                fetched = await knowledge_ingestion.fetch_url_text(candidate_url)
                with brain_lock:
                    manifest = knowledge_ingestion.ingest_text_into_memory(
                        brain,
                        text=fetched['text'],
                        source_name=_clean_text(candidate.get('title') or fetched.get('title') or source.get('name') or candidate_url),
                        source_url=fetched['final_url'],
                        tags=list(source.get('tags') or []) + ['autonomous_learning', source.get('source_id')],
                        chunk_size_chars=900,
                        overlap_chars=120,
                        min_chunk_chars=180,
                        importance=float(source.get('importance', 0.77)),
                        metadata={
                            'automation_source_id': source['source_id'],
                            'automation_source_name': source.get('name'),
                            'automation_run_id': run_id,
                            'automation_feed_url': source.get('feed_url'),
                            'automation_discovered_via': candidate.get('discovered_via'),
                            'source_type': 'autonomous_learning',
                            'page_title': fetched.get('title'),
                            'published_at': candidate.get('published_at'),
                            'candidate_score': candidate.get('score', 0.0),
                        },
                    )
                summary['processed'] += 1
                if manifest.get('duplicate'):
                    summary['duplicate_items'] += 1
                    stats['duplicate_items'] = int(stats.get('duplicate_items', 0)) + 1
                else:
                    summary['ingested'] += 1
                    stats['items_ingested'] = int(stats.get('items_ingested', 0)) + 1
                    summary['topics'] = _normalize_keywords(summary.get('topics', []) + self._remember_topics(source, manifest))
                seen_urls.append(candidate_url)
                seen_set.add(candidate_url)
            source['seen_urls'] = seen_urls[-2000:]
            stats['successes'] = int(stats.get('successes', 0)) + 1
            if summary['ingested'] > 0:
                source['last_success_at'] = time.time()
        except Exception as exc:
            logger.warning('Autonomous learning run failed for source %s: %s', source_id, exc)
            summary['status'] = 'error'
            summary['errors'].append(str(exc))
            stats['failures'] = int(stats.get('failures', 0)) + 1

        summary['finished_at'] = time.time()
        summary['source_score'] = self._update_source_score(source, summary)
        source['last_run_at'] = summary['finished_at']
        source['updated_at'] = summary['finished_at']
        source['last_summary'] = summary
        self.state.setdefault('scheduler', {})['last_run_at'] = summary['finished_at']
        runs = self._recent_runs()
        runs.append(summary)
        self.state['recent_runs'] = runs[-60:]
        self.save()
        return summary

    async def run_due_sources(self, brain: IntegratedArtificialBrain, brain_lock: Lock) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        now = time.time()
        self.state.setdefault('scheduler', {})['last_tick_at'] = now
        for source in self._sources().values():
            if not source.get('active', True):
                continue
            last_run_at = float(source.get('last_run_at') or 0.0)
            interval_seconds = int(source.get('interval_seconds', 86400))
            if not last_run_at or (now - last_run_at) >= interval_seconds:
                summaries.append(await self.run_source(source['source_id'], brain, brain_lock, force=True))
        self.save()
        return {
            'status': 'ok',
            'trigger': 'scheduler',
            'run_count': len(summaries),
            'runs': summaries,
        }

    async def run_all(self, brain: IntegratedArtificialBrain, brain_lock: Lock, *, force: bool = False) -> dict[str, Any]:
        summaries = []
        for source in self._sources().values():
            if not source.get('active', True) and not force:
                continue
            summaries.append(await self.run_source(source['source_id'], brain, brain_lock, force=force))
        return {
            'status': 'ok',
            'trigger': 'manual',
            'run_count': len(summaries),
            'runs': summaries,
        }

    async def _scheduler_loop(self, brain: IntegratedArtificialBrain, brain_lock: Lock) -> None:
        while not self._stop_event.is_set():
            try:
                if self.enabled:
                    await self.run_due_sources(brain, brain_lock)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('Autonomous learning scheduler tick failed: %s', exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def start(self, brain: IntegratedArtificialBrain, brain_lock: Lock) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._scheduler_loop(brain, brain_lock))

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
        self.save()
