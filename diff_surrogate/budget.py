"""Budget-aware training data generation for expensive solvers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingBudget:
    """Allocates a fixed compute budget across input regions and properties."""

    total_solver_calls: int = 1000
    n_regions: int = 4
    properties: list[str] = field(default_factory=lambda: ["value"])
    accuracy_target: float = 0.01
    pressure_threshold: float = 0.8

    def __post_init__(self):
        if self.total_solver_calls < 1:
            raise ValueError(f"total_solver_calls must be >= 1, got {self.total_solver_calls}")
        if self.n_regions < 1:
            raise ValueError(f"n_regions must be >= 1, got {self.n_regions}")
        self._calls_per_region: dict[int, int] = {i: 0 for i in range(self.n_regions)}
        self._accuracy_per_region: dict[int, float] = {i: 1.0 for i in range(self.n_regions)}
        self._total_calls: int = 0

    def allocate(self, region: int) -> int:
        remaining = self.total_solver_calls - self._total_calls
        if remaining <= 0:
            return 0
        weights = {
            r: max(self._accuracy_per_region.get(r, 1.0) / self.accuracy_target, 1.0)
            for r in range(self.n_regions)
        }
        total_w = sum(weights.values())
        share = remaining * weights[region] / total_w
        allocated = max(int(share), 1)
        return min(allocated, remaining)

    def record_accuracy(self, region: int, mse: float):
        self._accuracy_per_region[region] = mse

    def record_calls(self, region: int, n: int):
        self._calls_per_region[region] = self._calls_per_region.get(region, 0) + n
        self._total_calls += n

    @property
    def is_exhausted(self) -> bool:
        return self._total_calls >= self.total_solver_calls

    @property
    def budget_remaining(self) -> int:
        return max(0, self.total_solver_calls - self._total_calls)

    @property
    def pressure(self) -> float:
        return self._total_calls / max(1, self.total_solver_calls)
