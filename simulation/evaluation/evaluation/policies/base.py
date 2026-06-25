"""Base interface for evaluation policy adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evaluation.specs import PolicyOutput


class EvaluationPolicy(ABC):
    """A method under evaluation.

    The context argument is intentionally typed as Any because sim and non-sim
    runners may expose different runtime handles while sharing the same output
    contract.
    """

    name: str

    @abstractmethod
    def predict(self, context: Any) -> PolicyOutput:
        """Return an action intent for one evaluation episode."""

