"""Per-run resource accounting. Synchronous reservations are atomic on the event loop."""
from dataclasses import dataclass, field

from .config import Settings


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunBudget:
    settings: Settings
    searches: int = 0
    fetches: int = 0
    model_calls: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    usage_incomplete: bool = False

    def take(self, resource: str) -> None:
        limit = getattr(self.settings, f"max_{resource}")
        if getattr(self, resource) >= limit:
            raise BudgetExceeded(f"run {resource} budget exhausted ({limit})")
        setattr(self, resource, getattr(self, resource) + 1)

    def start_model(self) -> None:
        if sum(self.usage.values()) >= self.settings.max_total_tokens:
            raise BudgetExceeded("run token budget exhausted")
        self.take("model_calls")

    def account(self, usage: dict) -> None:
        self.usage["input_tokens"] += int(usage.get("inputTokens", 0))
        self.usage["output_tokens"] += int(usage.get("outputTokens", 0))
