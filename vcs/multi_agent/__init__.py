"""Multi-agent orchestration package for Hypothesis-Driven Video Seeking."""

from .actor import ActorRunner
from .orchestrator import MultiAgentCoordinator
from .state import (
    SharedVideoState,
    compute_uncovered_intervals,
    has_explicit_timestamp,
    merge_intervals,
)

__all__ = [
    "ActorRunner",
    "MultiAgentCoordinator",
    "SharedVideoState",
    "compute_uncovered_intervals",
    "has_explicit_timestamp",
    "merge_intervals",
]
