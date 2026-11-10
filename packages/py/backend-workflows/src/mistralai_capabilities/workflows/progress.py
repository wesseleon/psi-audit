"""Optional progress query shared by workflows and execution observers.

Platform execution status remains authoritative. A running workflow can expose a
``progress`` query returning this record to distinguish processing from a human wait.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["WorkflowProgress"]


class WorkflowProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: Literal["processing", "awaiting_review"]
    review_url: str | None = None
