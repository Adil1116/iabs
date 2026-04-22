from __future__ import annotations

import numpy as np


class EntorhinalCortex:
    """يحاكي الخلايا الشبكية للملاحة المكانية."""

    def __init__(self, rng: np.random.Generator, num_grid_cells: int = 200, scale: float = 10.0):
        self.num_cells = num_grid_cells
        self.scale = max(1e-6, float(scale))
        self.centers = rng.uniform(0.0, 100.0, (num_grid_cells, 2))

    def get_spatial_activity(self, position: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError(f'إحداثيات الموقع يجب أن تكون متجهًا ثنائيًا (2,) لكن تم استلام {position.shape}')
        distances = np.linalg.norm(self.centers - position, axis=1)
        values = np.cos(distances / self.scale)
        return np.maximum(0.0, values)
