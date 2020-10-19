from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Metric:
    metrics: Dict[str, float] = field(default_factory=dict)
    target_value: Optional[float] = None

    def __repr__(self) -> str:
        metric_str = ", ".join([f"{k}: {v:.03f}" for k, v in self.metrics.items()])
        return f"{metric_str}"
