from __future__ import annotations

from typing import Optional
import numpy as np


class NeuralMath:
    """مجموعة دوال رياضية مستقرة عدديًا للعمليات العصبية."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if axis is None:
            axis = -1
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_values = np.exp(shifted)
        denominator = np.sum(exp_values, axis=axis, keepdims=True)
        denominator = np.where(denominator == 0, 1.0, denominator)
        return exp_values / denominator

    @staticmethod
    def normalize_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mean = np.mean(x)
        std = np.std(x)
        if std < eps:
            return x - mean
        return (x - mean) / (std + eps)

    @staticmethod
    def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(1, fan_in))
        return rng.normal(0.0, scale, size=(fan_in, fan_out))

    @staticmethod
    def xavier_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(1, fan_in + fan_out))
        return rng.normal(0.0, scale, size=(fan_in, fan_out))
