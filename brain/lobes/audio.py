from __future__ import annotations

import numpy as np

from brain.core.neural_math import NeuralMath


class TemporalLobe:
    """يحاكي القشرة السمعية ومنطقة معالجة اللغة بشكل مبسط."""

    def __init__(self, rng: np.random.Generator, audio_channels: int = 1024):
        self.audio_channels = audio_channels
        self.audio_weights = NeuralMath.he_init(rng, audio_channels, 256)
        self.language_weights = NeuralMath.xavier_init(rng, 256, 128)

    def process_audio(self, audio_signal: np.ndarray) -> np.ndarray:
        audio_signal = np.asarray(audio_signal, dtype=np.float64)
        if audio_signal.shape != (self.audio_channels,):
            raise ValueError(
                f'أبعاد الصوت غير صحيحة. المتوقع ({self.audio_channels},) لكن تم استلام {audio_signal.shape}'
            )
        normalized = NeuralMath.normalize_vector(audio_signal)
        auditory_features = NeuralMath.relu(normalized @ self.audio_weights)
        return NeuralMath.sigmoid(auditory_features @ self.language_weights)
