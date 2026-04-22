from pathlib import Path


def replace_or_fail(text: str, old: str, new: str, file_name: str) -> str:
    if old not in text:
        raise RuntimeError(f'Pattern not found in {file_name}: {old[:80]!r}')
    return text.replace(old, new, 1)


base = Path('/home/user/work/project')

# --- patch brain/system.py ---
system_path = base / 'brain/system.py'
system_text = system_path.read_text(encoding='utf-8')
system_text = replace_or_fail(
    system_text,
    "from brain.memory import Hippocampus\n",
    "from brain.memory import Hippocampus\nfrom brain.agentic import ActionHookGateway, DreamEngine, UserModelEngine\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "        self.anomaly_events: Deque[dict[str, Any]] = deque(maxlen=100)\n        self.affect_state: dict[str, Any] = {\n",
    "        self.anomaly_events: Deque[dict[str, Any]] = deque(maxlen=100)\n        self.user_model: dict[str, Any] = UserModelEngine.empty_profile()\n        self.action_hooks: dict[str, Any] = ActionHookGateway.empty_registry()\n        self.last_tom_inference: dict[str, Any] | None = None\n        self.dream_history: Deque[dict[str, Any]] = deque(maxlen=20)\n        self.last_dream_report: dict[str, Any] | None = None\n        self.last_action_hook_result: dict[str, Any] | None = None\n        self.affect_state: dict[str, Any] = {\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "    def get_affect_state(self) -> dict[str, Any]:\n        return {\n            'curiosity': float(self.affect_state.get('curiosity', 0.5)),\n            'stress': float(self.affect_state.get('stress', 0.2)),\n            'satisfaction': float(self.affect_state.get('satisfaction', 0.5)),\n            'energy': float(self.affect_state.get('energy', 0.7)),\n            'updated_at': float(self.affect_state.get('updated_at', time.time())),\n            'mood_label': str(self.affect_state.get('mood_label', 'متوازن')),\n        }\n\n",
    "    def get_affect_state(self) -> dict[str, Any]:\n        return {\n            'curiosity': float(self.affect_state.get('curiosity', 0.5)),\n            'stress': float(self.affect_state.get('stress', 0.2)),\n            'satisfaction': float(self.affect_state.get('satisfaction', 0.5)),\n            'energy': float(self.affect_state.get('energy', 0.7)),\n            'updated_at': float(self.affect_state.get('updated_at', time.time())),\n            'mood_label': str(self.affect_state.get('mood_label', 'متوازن')),\n        }\n\n    def get_user_model(self) -> dict[str, Any]:\n        return self._to_jsonable(self.user_model)\n\n    def rebuild_user_model(self) -> dict[str, Any]:\n        profile = UserModelEngine.empty_profile()\n        for entry in list(self.context_window):\n            if not isinstance(entry, dict):\n                continue\n            text = str(entry.get('user_text') or '').strip()\n            if not text:\n                continue\n            profile = UserModelEngine.update_profile(\n                profile,\n                text=text,\n                context_topics=list(entry.get('context_topics', [])),\n                affect_state=entry.get('affect') or self.get_affect_state(),\n            )\n        self.user_model = profile\n        return self.get_user_model()\n\n    def infer_user_mind(self, text: str) -> dict[str, Any]:\n        self.last_tom_inference = UserModelEngine.infer_tom(\n            profile=self.user_model,\n            text=text,\n            recent_context=list(self.context_window)[-6:],\n            affect_state=self.get_affect_state(),\n        )\n        return self._to_jsonable(self.last_tom_inference)\n\n    def register_action_hook(\n        self,\n        *,\n        name: str,\n        event: str,\n        action_type: str = 'webhook',\n        target_url: str | None = None,\n        method: str = 'POST',\n        headers: dict[str, str] | None = None,\n        payload_template: dict[str, Any] | None = None,\n        keywords: list[str] | None = None,\n        cooldown_seconds: int = 0,\n        active: bool = True,\n    ) -> dict[str, Any]:\n        self.action_hooks, hook = ActionHookGateway.register_hook(\n            self.action_hooks,\n            name=name,\n            event=event,\n            action_type=action_type,\n            target_url=target_url,\n            method=method,\n            headers=headers,\n            payload_template=payload_template,\n            keywords=keywords,\n            cooldown_seconds=cooldown_seconds,\n            active=active,\n        )\n        self.memory.store_memory(\n            key=f'{hook[\"hook_id\"]}_registered',\n            data={'event': 'action_hook_registered', 'hook': hook},\n            importance=0.74,\n            source='action_hook',\n        )\n        return hook\n\n    def list_action_hooks(self, active_only: bool = False) -> list[dict[str, Any]]:\n        return self._to_jsonable(ActionHookGateway.list_hooks(self.action_hooks, active_only=active_only))\n\n    def action_hooks_overview(self) -> dict[str, Any]:\n        return self._to_jsonable(ActionHookGateway.overview(self.action_hooks))\n\n    def recent_action_hook_events(self, limit: int = 10) -> list[dict[str, Any]]:\n        events = list((self.action_hooks or {}).get('events', []))\n        return self._to_jsonable(events[-max(1, int(limit)):])\n\n    def trigger_action_hooks(\n        self,\n        *,\n        event: str,\n        text: str = '',\n        decision: str | None = None,\n        topics: list[str] | None = None,\n        dry_run: bool = True,\n        allow_network: bool = False,\n    ) -> dict[str, Any]:\n        self.action_hooks, summary = ActionHookGateway.dispatch(\n            self.action_hooks,\n            event=event,\n            text=text,\n            decision=decision,\n            topics=topics,\n            dry_run=dry_run,\n            allow_network=allow_network,\n        )\n        self.last_action_hook_result = summary\n        if summary.get('matched_hooks'):\n            self.memory.store_memory(\n                key=f'action_hook_dispatch_{int(time.time() * 1000)}',\n                data={'event': 'action_hook_dispatch', 'summary': summary},\n                importance=0.7,\n                source='action_hook',\n            )\n        return self._to_jsonable(summary)\n\n    def run_dream_engine(self, trigger: str = 'manual_dream') -> dict[str, Any]:\n        report = DreamEngine.synthesize(\n            trigger=trigger,\n            context_window=list(self.context_window)[-10:],\n            goals=[asdict(goal) for goal in self.goals.values() if goal.status == 'active'],\n            affect_state=self.get_affect_state(),\n        )\n        self.last_dream_report = report\n        self.dream_history.append(report)\n        self.memory.store_memory(\n            key=f'dream_engine_{int(report[\"timestamp\"] * 1000)}',\n            data=report,\n            importance=0.79,\n            source='dream_engine',\n        )\n        return self._to_jsonable(report)\n\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "        if extra_memory:\n            memory_payload.update(self._to_jsonable(extra_memory))\n        affect_snapshot = self._update_affect_from_interaction(memory_payload, float(importance))\n        goal_updates = self._update_goals_from_payload(memory_key, memory_payload, float(importance))\n        memory_payload['affect_state'] = affect_snapshot\n        memory_payload['goal_updates'] = goal_updates\n",
    "        if extra_memory:\n            memory_payload.update(self._to_jsonable(extra_memory))\n        user_text = str(memory_payload.get('user_text') or '').strip()\n        if user_text:\n            self.user_model = UserModelEngine.update_profile(\n                self.user_model,\n                text=user_text,\n                context_topics=list(memory_payload.get('context_topics', [])),\n                affect_state=self.get_affect_state(),\n            )\n            memory_payload['user_model'] = self.get_user_model()\n            memory_payload['theory_of_mind'] = self.infer_user_mind(user_text)\n        affect_snapshot = self._update_affect_from_interaction(memory_payload, float(importance))\n        goal_updates = self._update_goals_from_payload(memory_key, memory_payload, float(importance))\n        memory_payload['affect_state'] = affect_snapshot\n        memory_payload['goal_updates'] = goal_updates\n        if user_text:\n            action_hook_summary = self.trigger_action_hooks(\n                event='chat.message',\n                text=user_text,\n                decision=decision_result.decision,\n                topics=list(memory_payload.get('context_topics', [])),\n                dry_run=True,\n                allow_network=False,\n            )\n            memory_payload['action_hook_summary'] = action_hook_summary\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "    def run_sleep_cycle(self, trigger: str = 'manual_sleep') -> dict[str, Any]:\n        affect_before = self.get_affect_state()\n        recent_context = list(self.context_window)[-10:]\n        recent_texts = [str(item.get('user_text') or item.get('decision') or '') for item in recent_context]\n        dream_topics = self._extract_topics(recent_texts, limit=6)\n        promoted = self.memory.consolidate_recent_memories(min_importance=0.68)\n        self_improvement = self.run_self_improvement(trigger='sleep_cycle')\n        self._shift_affect(curiosity=-0.04, stress=-0.14, satisfaction=0.05, energy=0.22)\n        report = {\n            'trigger': trigger,\n            'timestamp': time.time(),\n            'dream_topics': dream_topics,\n            'promoted_memories': promoted,\n            'goal_summary': self.goals_overview(),\n            'affect_before': affect_before,\n            'affect_after': self.get_affect_state(),\n            'self_improvement': self_improvement,\n        }\n",
    "    def run_sleep_cycle(self, trigger: str = 'manual_sleep') -> dict[str, Any]:\n        affect_before = self.get_affect_state()\n        recent_context = list(self.context_window)[-10:]\n        recent_texts = [str(item.get('user_text') or item.get('decision') or '') for item in recent_context]\n        dream_engine_report = self.run_dream_engine(trigger=f'{trigger}_dream_engine')\n        dream_topics = dream_engine_report.get('dream_topics') or self._extract_topics(recent_texts, limit=6)\n        promoted = self.memory.consolidate_recent_memories(min_importance=0.68)\n        self_improvement = self.run_self_improvement(trigger='sleep_cycle')\n        self._shift_affect(curiosity=-0.04, stress=-0.14, satisfaction=0.05, energy=0.22)\n        report = {\n            'trigger': trigger,\n            'timestamp': time.time(),\n            'dream_topics': dream_topics,\n            'dream_engine': dream_engine_report,\n            'promoted_memories': promoted,\n            'goal_summary': self.goals_overview(),\n            'affect_before': affect_before,\n            'affect_after': self.get_affect_state(),\n            'self_improvement': self_improvement,\n        }\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "        return {\n            'state': self.state,\n            'cycles_completed': self.cycles_completed,\n            'personality': self.get_personality_profile(),\n            'affect': self.get_affect_state(),\n            'goals': self.goals_overview(),\n            'memory': memory_stats,\n            'anomalies': self.anomaly_overview(),\n            'gauges': gauges,\n            'recent_episodes': list(self.episode_history)[-5:],\n            'recent_context': list(self.context_window)[-5:],\n            'recent_learning': list(self.learning_history)[-5:],\n        }\n",
    "        return {\n            'state': self.state,\n            'cycles_completed': self.cycles_completed,\n            'personality': self.get_personality_profile(),\n            'affect': self.get_affect_state(),\n            'goals': self.goals_overview(),\n            'memory': memory_stats,\n            'anomalies': self.anomaly_overview(),\n            'gauges': gauges,\n            'user_model': self.get_user_model(),\n            'action_hooks': self.action_hooks_overview(),\n            'last_dream_report': self.last_dream_report,\n            'recent_episodes': list(self.episode_history)[-5:],\n            'recent_context': list(self.context_window)[-5:],\n            'recent_learning': list(self.learning_history)[-5:],\n        }\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "            'recent_sleep_reports': self.recent_sleep_reports(limit=3),\n            'anomalies': self.anomaly_overview(),\n            'dashboard': self.dashboard_snapshot(),\n            'recommended_focus': (\n",
    "            'recent_sleep_reports': self.recent_sleep_reports(limit=3),\n            'anomalies': self.anomaly_overview(),\n            'user_model': self.get_user_model(),\n            'last_tom_inference': self.last_tom_inference,\n            'action_hooks': self.action_hooks_overview(),\n            'recent_action_hook_events': self.recent_action_hook_events(limit=5),\n            'last_dream_report': self.last_dream_report,\n            'dashboard': self.dashboard_snapshot(),\n            'recommended_focus': (\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "            'version': 7,\n",
    "            'version': 8,\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "            'anomaly_events': list(self.anomaly_events),\n            'components': {\n",
    "            'anomaly_events': list(self.anomaly_events),\n            'user_model': self.get_user_model(),\n            'last_tom_inference': self.last_tom_inference,\n            'action_hooks': self.action_hooks,\n            'dream_history': list(self.dream_history),\n            'last_dream_report': self.last_dream_report,\n            'last_action_hook_result': self.last_action_hook_result,\n            'components': {\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "        anomaly_payload = payload.get('anomaly_events', [])\n        self.anomaly_events = deque([item for item in anomaly_payload if isinstance(item, dict)], maxlen=100)\n        rng_state = payload.get('rng_state')\n",
    "        anomaly_payload = payload.get('anomaly_events', [])\n        self.anomaly_events = deque([item for item in anomaly_payload if isinstance(item, dict)], maxlen=100)\n        user_model_payload = payload.get('user_model', {})\n        if isinstance(user_model_payload, dict):\n            self.user_model = UserModelEngine.empty_profile()\n            self.user_model.update(user_model_payload)\n        last_tom_payload = payload.get('last_tom_inference')\n        self.last_tom_inference = last_tom_payload if isinstance(last_tom_payload, dict) else None\n        action_hooks_payload = payload.get('action_hooks', {})\n        if isinstance(action_hooks_payload, dict):\n            self.action_hooks = ActionHookGateway.empty_registry()\n            self.action_hooks.update(action_hooks_payload)\n            self.action_hooks['hooks'] = [item for item in action_hooks_payload.get('hooks', []) if isinstance(item, dict)]\n            self.action_hooks['events'] = [item for item in action_hooks_payload.get('events', []) if isinstance(item, dict)][-100:]\n        dream_payload = payload.get('dream_history', [])\n        self.dream_history = deque([item for item in dream_payload if isinstance(item, dict)], maxlen=20)\n        self.last_dream_report = payload.get('last_dream_report') if isinstance(payload.get('last_dream_report'), dict) else None\n        self.last_action_hook_result = payload.get('last_action_hook_result') if isinstance(payload.get('last_action_hook_result'), dict) else None\n        rng_state = payload.get('rng_state')\n",
    'brain/system.py',
)
system_text = replace_or_fail(
    system_text,
    "            'personality': self.get_personality_profile(),\n            'context_window_size': len(self.context_window),\n            'self_improvement_events': len(self.self_improvement_history),\n            'episode_count': len(self.episode_history),\n            'last_episode_key': self.last_episode_key,\n            'affect': self.get_affect_state(),\n            'goals': self.goals_overview(),\n            'last_sleep_report': self.last_sleep_report,\n            'recent_episodes': list(self.episode_history)[-3:],\n            'anomalies': self.anomaly_overview(),\n        }\n",
    "            'personality': self.get_personality_profile(),\n            'context_window_size': len(self.context_window),\n            'self_improvement_events': len(self.self_improvement_history),\n            'episode_count': len(self.episode_history),\n            'last_episode_key': self.last_episode_key,\n            'affect': self.get_affect_state(),\n            'goals': self.goals_overview(),\n            'user_model': self.get_user_model(),\n            'last_tom_inference': self.last_tom_inference,\n            'action_hooks': self.action_hooks_overview(),\n            'last_action_hook_result': self.last_action_hook_result,\n            'last_sleep_report': self.last_sleep_report,\n            'last_dream_report': self.last_dream_report,\n            'recent_episodes': list(self.episode_history)[-3:],\n            'anomalies': self.anomaly_overview(),\n        }\n",
    'brain/system.py',
)
system_path.write_text(system_text, encoding='utf-8')

# --- patch brain/text_interface.py ---
text_path = base / 'brain/text_interface.py'
text_text = text_path.read_text(encoding='utf-8')
text_text = replace_or_fail(
    text_text,
    "    goals_overview: dict[str, Any]\n    reply_mode: str\n",
    "    goals_overview: dict[str, Any]\n    user_model: dict[str, Any]\n    theory_of_mind: dict[str, Any] | None\n    action_hook_summary: dict[str, Any] | None\n    reply_mode: str\n",
    'brain/text_interface.py',
)
text_text = replace_or_fail(
    text_text,
    "        goals_overview = self.brain.goals_overview()\n        deterministic_reply = self._build_reply(\n",
    "        goals_overview = self.brain.goals_overview()\n        user_model = self.brain.get_user_model()\n        theory_of_mind = self.brain.last_tom_inference or self.brain.infer_user_mind(normalized_text)\n        action_hook_summary = self.brain.last_action_hook_result\n        deterministic_reply = self._build_reply(\n",
    'brain/text_interface.py',
)
text_text = replace_or_fail(
    text_text,
    "            goals_overview=goals_overview,\n            reply_mode=reply_mode,\n",
    "            goals_overview=goals_overview,\n            user_model=user_model,\n            theory_of_mind=theory_of_mind,\n            action_hook_summary=action_hook_summary,\n            reply_mode=reply_mode,\n",
    'brain/text_interface.py',
)
text_path.write_text(text_text, encoding='utf-8')

# --- patch api/main.py ---
api_path = base / 'api/main.py'
api_text = api_path.read_text(encoding='utf-8')
api_text = replace_or_fail(
    api_text,
    "class SelfImproveInput(BaseModel):\n    trigger: str = Field(default='manual', min_length=2, max_length=60)\n\n\nclass GoalCreateInput(BaseModel):\n",
    "class SelfImproveInput(BaseModel):\n    trigger: str = Field(default='manual', min_length=2, max_length=60)\n\n\nclass TheoryOfMindInput(BaseModel):\n    text: str = Field(..., min_length=1, max_length=4000)\n\n\nclass ActionHookCreateInput(BaseModel):\n    name: str = Field(..., min_length=2, max_length=120)\n    event: str = Field(..., min_length=3, max_length=80)\n    action_type: str = Field(default='webhook', min_length=3, max_length=40)\n    target_url: str | None = Field(default=None, max_length=500)\n    method: str = Field(default='POST', min_length=3, max_length=10)\n    headers: dict[str, str] = Field(default_factory=dict)\n    payload_template: dict[str, Any] = Field(default_factory=dict)\n    keywords: list[str] = Field(default_factory=list, max_length=10)\n    cooldown_seconds: int = Field(default=0, ge=0, le=86400)\n    active: bool = True\n\n\nclass ActionHookTriggerInput(BaseModel):\n    event: str = Field(..., min_length=3, max_length=80)\n    text: str = Field(default='', max_length=4000)\n    decision: str | None = Field(default=None, max_length=120)\n    topics: list[str] = Field(default_factory=list, max_length=10)\n    dry_run: bool = True\n    allow_network: bool = False\n\n\nclass GoalCreateInput(BaseModel):\n",
    'api/main.py',
)
api_text = replace_or_fail(
    api_text,
    "    '/brain/affect', '/brain/goals', '/brain/goals/{goal_id}', '/brain/context', '/brain/self-improve', '/brain/sleep-report',\n    '/brain/dashboard', '/brain/anomalies',\n",
    "    '/brain/affect', '/brain/user-model', '/brain/user-model/rebuild', '/brain/theory-of-mind',\n    '/brain/action-hooks', '/brain/action-hooks/trigger', '/brain/action-hooks/events',\n    '/brain/goals', '/brain/goals/{goal_id}', '/brain/context', '/brain/self-improve', '/brain/sleep-report',\n    '/brain/dashboard', '/brain/anomalies',\n",
    'api/main.py',
)
api_text = replace_or_fail(
    api_text,
    "                    'prompt_anomaly_detection': True,\n                    'neural_dashboard_snapshot': True,\n                    'memory_encryption_at_rest': brain.memory.encryption_status().get('enabled', False),\n                },\n            }\n\n    @app.post('/brain/personality')\n",
    "                    'prompt_anomaly_detection': True,\n                    'neural_dashboard_snapshot': True,\n                    'user_modeling': True,\n                    'theory_of_mind_lite': True,\n                    'action_hooks_gateway': True,\n                    'dream_engine': True,\n                    'memory_encryption_at_rest': brain.memory.encryption_status().get('enabled', False),\n                },\n            }\n\n    @app.get('/brain/user-model')\n    def brain_user_model(_: TokenPayload = Depends(require_permissions('system:read'))) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            return {'user_model': brain.get_user_model(), 'last_tom_inference': brain.last_tom_inference}\n\n    @app.post('/brain/user-model/rebuild')\n    def rebuild_brain_user_model(\n        request: Request,\n        user: TokenPayload = Depends(require_permissions('system:write')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            profile = brain.rebuild_user_model()\n        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='rebuild_user_model')\n        return {'user_model': profile}\n\n    @app.post('/brain/theory-of-mind')\n    def brain_theory_of_mind(\n        payload: TheoryOfMindInput,\n        request: Request,\n        user: TokenPayload = Depends(require_permissions('system:read')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            inference = brain.infer_user_mind(payload.text)\n        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='theory_of_mind_inference')\n        return {'theory_of_mind': inference, 'user_model': brain.get_user_model()}\n\n    @app.get('/brain/action-hooks')\n    def brain_action_hooks(\n        active_only: bool = Query(default=False),\n        _: TokenPayload = Depends(require_permissions('system:read')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            return {'hooks': brain.list_action_hooks(active_only=active_only), 'summary': brain.action_hooks_overview()}\n\n    @app.post('/brain/action-hooks')\n    def create_brain_action_hook(\n        payload: ActionHookCreateInput,\n        request: Request,\n        user: TokenPayload = Depends(require_permissions('system:write')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            hook = brain.register_action_hook(\n                name=payload.name,\n                event=payload.event,\n                action_type=payload.action_type,\n                target_url=payload.target_url,\n                method=payload.method,\n                headers=payload.headers,\n                payload_template=payload.payload_template,\n                keywords=payload.keywords,\n                cooldown_seconds=payload.cooldown_seconds,\n                active=payload.active,\n            )\n        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='create_action_hook', details={'hook_id': hook['hook_id']})\n        return {'hook': hook, 'summary': brain.action_hooks_overview()}\n\n    @app.post('/brain/action-hooks/trigger')\n    def trigger_brain_action_hooks(\n        payload: ActionHookTriggerInput,\n        request: Request,\n        user: TokenPayload = Depends(require_permissions('system:write')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            summary = brain.trigger_action_hooks(\n                event=payload.event,\n                text=payload.text,\n                decision=payload.decision,\n                topics=payload.topics,\n                dry_run=payload.dry_run,\n                allow_network=payload.allow_network,\n            )\n        _audit(request, 'system_action', actor=str(user['sub']), role=str(user['role']), outcome='success', action='trigger_action_hooks', details={'event': payload.event, 'matched_hooks': summary['matched_hooks']})\n        return {'action_hook_summary': summary, 'hooks': brain.list_action_hooks()}\n\n    @app.get('/brain/action-hooks/events')\n    def brain_action_hook_events(\n        limit: int = Query(default=10, ge=1, le=50),\n        _: TokenPayload = Depends(require_permissions('system:read')),\n    ) -> dict[str, Any]:\n        brain, brain_lock, _ = runtime()\n        with brain_lock:\n            return {'events': brain.recent_action_hook_events(limit=limit), 'summary': brain.action_hooks_overview()}\n\n    @app.post('/brain/personality')\n",
    'api/main.py',
)
api_text = replace_or_fail(
    api_text,
    "                    'goals_overview': result.goals_overview,\n                    'reply_mode': result.reply_mode,\n",
    "                    'goals_overview': result.goals_overview,\n                    'user_model': result.user_model,\n                    'theory_of_mind': result.theory_of_mind,\n                    'action_hook_summary': result.action_hook_summary,\n                    'reply_mode': result.reply_mode,\n",
    'api/main.py',
)
api_text = replace_or_fail(
    api_text,
    "                            'personality': result.personality,\n                            'reply_mode': result.reply_mode,\n",
    "                            'personality': result.personality,\n                            'user_model': result.user_model,\n                            'theory_of_mind': result.theory_of_mind,\n                            'action_hook_summary': result.action_hook_summary,\n                            'reply_mode': result.reply_mode,\n",
    'api/main.py',
)
api_path.write_text(api_text, encoding='utf-8')

# --- patch tests/test_learning.py ---
learning_path = base / 'tests/test_learning.py'
learning_text = learning_path.read_text(encoding='utf-8')
learning_text += """


def test_user_model_action_hooks_and_dream_engine(tmp_path):
    brain = IntegratedArtificialBrain(
        seed=19,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=False,
    )
    hook = brain.register_action_hook(
        name='Project Hook',
        event='chat.message',
        keywords=['مشروع', 'workflow'],
        target_url='https://example.com/webhook',
    )
    assert hook['hook_id']

    visual = np.ones((64, 64), dtype=np.float64)
    audio = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    position = np.array([7.0, 1.0], dtype=np.float64)
    brain.live_cycle(
        visual,
        audio,
        position,
        source='chat',
        extra_memory={'user_text': 'اكمل تطوير المشروع وابني workflow أوضح مع webhook عملية'},
    )

    user_model = brain.get_user_model()
    assert user_model['interaction_count'] >= 1
    assert user_model['communication_style']['verbosity'] in {'balanced', 'detailed', 'concise'}
    assert brain.last_tom_inference is not None
    assert brain.last_action_hook_result is not None
    assert brain.last_action_hook_result['matched_hooks'] >= 1

    dream_report = brain.run_dream_engine(trigger='unit-test-dream')
    assert dream_report['trigger'] == 'unit-test-dream'
    assert 'recommended_experiments' in dream_report and dream_report['recommended_experiments']

    brain.save_state()
    reloaded = IntegratedArtificialBrain(
        seed=19,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=True,
    )
    assert reloaded.get_user_model()['interaction_count'] >= 1
    assert reloaded.action_hooks_overview()['total_hooks'] >= 1
    assert reloaded.last_dream_report is not None
"""
learning_path.write_text(learning_text, encoding='utf-8')

# --- patch tests/test_api.py ---
api_test_path = base / 'tests/test_api.py'
api_test_text = api_test_path.read_text(encoding='utf-8')
api_test_text += """


def test_user_model_and_action_hooks_api(tmp_path):
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        access_token = get_access_token(client)
        headers = {'Authorization': f'Bearer {access_token}'}

        hook_create = client.post(
            '/brain/action-hooks',
            json={
                'name': 'Project Automation',
                'event': 'chat.message',
                'keywords': ['مشروع', 'workflow'],
                'target_url': 'https://example.com/webhook',
            },
            headers=headers,
        )
        assert hook_create.status_code == 200, hook_create.text
        assert hook_create.json()['hook']['hook_id']

        chat_response = client.post(
            '/chat',
            json={'text': 'اكمل تطوير المشروع واعمل workflow أوضح للـ webhook', 'importance': 0.88},
            headers=headers,
        )
        assert chat_response.status_code == 200, chat_response.text
        payload = chat_response.json()
        assert 'user_model' in payload
        assert payload['theory_of_mind']['intent'] in {
            'develop_and_extend', 'connect_to_external_actions', 'plan_and_sequence', 'specify_design', 'general_request'
        }
        assert payload['action_hook_summary']['matched_hooks'] >= 1

        user_model_response = client.get('/brain/user-model', headers=headers)
        assert user_model_response.status_code == 200, user_model_response.text
        assert user_model_response.json()['user_model']['interaction_count'] >= 1

        tom_response = client.post('/brain/theory-of-mind', json={'text': 'نفذ spec جديدة للمشروع'}, headers=headers)
        assert tom_response.status_code == 200, tom_response.text
        assert tom_response.json()['theory_of_mind']['expected_outcome']

        trigger_response = client.post(
            '/brain/action-hooks/trigger',
            json={
                'event': 'chat.message',
                'text': 'هذا مشروع يحتاج workflow',
                'topics': ['workflow'],
                'dry_run': True,
                'allow_network': False,
            },
            headers=headers,
        )
        assert trigger_response.status_code == 200, trigger_response.text
        assert trigger_response.json()['action_hook_summary']['matched_hooks'] >= 1

        events_response = client.get('/brain/action-hooks/events', headers=headers)
        assert events_response.status_code == 200, events_response.text
        assert len(events_response.json()['events']) >= 1
"""
api_test_path.write_text(api_test_text, encoding='utf-8')

print('Patched successfully')
