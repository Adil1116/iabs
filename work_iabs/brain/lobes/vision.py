from __future__ import annotations

from typing import Tuple
import numpy as np

from brain.core.neural_math import NeuralMath


class OccipitalLobe:
    """يحاكي القشرة البصرية بشكل مبسط."""

    def __init__(self, rng: np.random.Generator, resolution: Tuple[int, int] = (64, 64)):
        self.resolution = resolution
        input_size = resolution[0] * resolution[1]
        self.v1_weights = NeuralMath.he_init(rng, input_size, 128)
        self.v2_weights = NeuralMath.xavier_init(rng, 128, 64)

    def process_image(self, image_data: np.ndarray) -> np.ndarray:
        image_data = np.asarray(image_data, dtype=np.float64)
        if image_data.shape != self.resolution:
            raise ValueError(
                f'أبعاد الصورة غير صحيحة. المتوقع {self.resolution} لكن تم استلام {image_data.shape}'
            )
        flattened = NeuralMath.normalize_vector(image_data.flatten())
        layer1 = NeuralMath.relu(flattened @ self.v1_weights)
        return NeuralMath.sigmoid(layer1 @ self.v2_weights)
