from __future__ import annotations

import numpy as np

from brain.system import IntegratedArtificialBrain


def test_feedback_learning_changes_weights_and_semantic_memory(tmp_path):
    brain = IntegratedArtificialBrain(
        seed=7,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=False,
    )
    visual = np.ones((64, 64), dtype=np.float64)
    audio = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    position = np.array([2.0, 4.0], dtype=np.float64)

    result = brain.live_cycle(
        visual,
        audio,
        position,
        source='test_cycle',
        extra_memory={'user_text': 'المشروع محتاج تطوير للذاكرة والسياق'},
    )
    assert brain.last_memory_key is not None

    before = brain.executive.decision_weights[:, result.decision_index].copy()
    feedback = brain.apply_feedback(1.0, memory_key=brain.last_memory_key, feedback_text='قرار ممتاز', source='test_feedback')
    after = brain.executive.decision_weights[:, result.decision_index]

    assert feedback['memory_key'] == brain.last_memory_key
    assert feedback['reward'] == 1.0
    assert not np.allclose(before, after)
    assert brain.executive.learning_stats()['feedback_events'] >= 1

    matches = brain.memory.search_memories('تطوير الذاكره والسياق', limit=5, strategy='semantic')
    assert matches


def test_personality_context_goals_affect_and_self_improvement_cycle(tmp_path):
    brain = IntegratedArtificialBrain(
        seed=11,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=False,
    )
    personality = brain.update_personality_profile({'display_name': 'IABS Prime', 'tone': 'analytical'})
    assert personality['display_name'] == 'IABS Prime'

    goal = brain.create_goal(
        'تحسين الذاكرة والسياق',
        'تقوية البحث الدلالي وربط الرسائل بالسياق',
        priority=0.9,
        target_keywords=['ذاكره', 'سياق', 'memory'],
    )
    assert goal['goal_id']

    visual = np.ones((64, 64), dtype=np.float64)
    audio = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    position = np.array([1.0, 3.0], dtype=np.float64)
    brain.live_cycle(
        visual,
        audio,
        position,
        source='chat',
        extra_memory={'user_text': 'عايز ذاكرة اقوى وسياق افضل', 'context_topics': ['ذاكره', 'سياق']},
    )

    snapshot = brain.context_snapshot(limit=5, query='ذاكره وسياق')
    assert snapshot['window_size'] >= 1
    assert snapshot['topics']
    assert snapshot['goals']['total'] >= 1
    assert snapshot['affect']['mood_label']

    goal_after_signal = brain.list_goals(limit=5)[0]
    assert goal_after_signal['progress'] > 0.0

    brain.apply_feedback(0.8, memory_key=brain.last_memory_key, feedback_text='مفيد', source='feedback-1')
    event = brain.run_self_improvement(trigger='test-suite')
    assert event['trigger'] == 'test-suite'
    assert 'recommendations' in event and event['recommendations']

    sleep_report = brain.run_sleep_cycle(trigger='unit-test-sleep')
    assert sleep_report['trigger'] == 'unit-test-sleep'
    assert 'dream_topics' in sleep_report
    assert brain.last_sleep_report is not None


def test_semantic_synonyms_and_state_persistence(tmp_path):
    brain = IntegratedArtificialBrain(
        seed=17,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=False,
    )
    brain.create_goal('ابحث عن طعام', 'هدف متعلق بالجوع والطعام', priority=0.7, target_keywords=['جوعان', 'طعام'])
    visual = np.ones((64, 64), dtype=np.float64)
    audio = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    position = np.array([5.0, 2.0], dtype=np.float64)
    brain.live_cycle(
        visual,
        audio,
        position,
        source='chat',
        extra_memory={'user_text': 'أنا جوعان جداً ومحتاج آكل بسرعة'},
    )

    semantic_matches = brain.memory.search_memories('عايز طعام', limit=5, strategy='semantic')
    assert semantic_matches

    brain.save_state()
    reloaded = IntegratedArtificialBrain(
        seed=17,
        storage_path=tmp_path / 'memory.sqlite',
        state_path=tmp_path / 'brain_state.json',
        autoload_state=True,
    )
    assert reloaded.goals_overview()['total'] >= 1
    assert reloaded.get_affect_state()['mood_label']



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
