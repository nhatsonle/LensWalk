from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union

class TimeInterval(BaseModel):
    start_sec: float = Field(..., description="Start time in seconds.")
    end_sec: float = Field(..., description="End time in seconds.")

class Observation(BaseModel):
    interval: Union[TimeInterval, List[TimeInterval]] = Field(..., description="Time interval(s) covered by this observation.")
    description: str = Field(..., description="Text description of observed events, objects, or scenes.")
    confidence: float = Field(default=1.0, description="Confidence score for the observation.")
    subject_registry: Dict[str, Any] = Field(default_factory=dict, description="Structured subject registry for observed entities.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Budget(BaseModel):
    sampled_frames_count: int = Field(..., description="Total sampled frame count.")
    frame_indices: List[int] = Field(..., description="Sampled frame indices.")

class Provenance(BaseModel):
    tool_name: str = Field(..., description="Tool name.")
    tool_params: Dict[str, Any] = Field(..., description="Raw tool parameters.")

class ToolOutput(BaseModel):
    observations: List[Observation] = Field(..., description="Structured observations produced by this tool call.")
    coverage: List[TimeInterval] = Field(..., description="All time intervals covered by this tool call.")
    used_budget: Budget = Field(..., description="Sampling budget consumed by this tool call.")
    provenance: Provenance = Field(..., description="Tool-call provenance.")

class SubjectRegistryState(BaseModel):
    subject_registry: Dict[str, Any] = Field(default_factory=dict, description="Aggregated subject registry.")
