from typing import Dict, Any, Optional

from .base_observer import BaseVideoObserver
from ..data_models import ToolOutput, TimeInterval, Observation, Provenance
from ..utils import extract_and_resize_frames
from ..prompts import (
    DEFAULT_BIN_TIME_RANGE_PROMPT,
    DEFAULT_GROUP_TIME_RANGE_PROMPT,
    DEFAULT_USER_PROMPT,
    SINGLE_DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from utils.tool_registry import register_tool


@register_tool("single_segment_observer")
class SingleSegmentObserver(BaseVideoObserver):
    def __init__(
        self,
        *,
        base_fps: float = 1.0,
        base_frame_num: int = 32,
        base_max_edge_length: int = 720,
        base_prompt_time_range: float = 10.0,
        openai_base_url: Optional[str] = None,
        openai_key: Optional[str] = None,
        mllm_backend: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        group_time_range_prompt: Optional[str] = None,
        bin_time_range_prompt: Optional[str] = None,
        arguments: Optional[dict] = None,
        **base_kwargs
    ):
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.user_prompt = DEFAULT_USER_PROMPT if user_prompt is None else user_prompt

        self.group_time_range_prompt = (
            DEFAULT_GROUP_TIME_RANGE_PROMPT if group_time_range_prompt is None else group_time_range_prompt
        )
        self.bin_time_range_prompt = (
            DEFAULT_BIN_TIME_RANGE_PROMPT if bin_time_range_prompt is None else bin_time_range_prompt
        )

        merged_arguments = {
            "interval": "Time interval to observe.",
            "query": "Question to ask the MLLM.",
            "video_path": "Path to the video file.",
            "fps": None,
            "max_edge_length": None,
            "max_total_frames": None,
            "num_workers": None,
            "subject_registry": None,
        }
        if arguments:
            merged_arguments.update(arguments)
        
        self.anyof_groups = []
        if base_kwargs.get('name', None) is None:
            base_kwargs['name'] = "segment_probe"
        super().__init__(
            arguments=merged_arguments,
            base_fps=base_fps,
            base_frame_num=base_frame_num,
            base_max_edge_length=base_max_edge_length,
            openai_base_url=openai_base_url,
            openai_key=openai_key,
            mllm_backend=mllm_backend,
            **base_kwargs
        )
        self.base_prompt_time_range = float(base_prompt_time_range)
        
    def _postprocess_output(self, output: ToolOutput) -> str:
        """Convert tool output into a text observation."""
        obs_sorted = sorted(output.observations, key=lambda o: o.interval.start_sec)
        return "\n".join(
            f"{o.description}"
            for o in obs_sorted
        )
    
    def _func_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        interval = TimeInterval(**params["interval"])
        max_edge_length = params.get("max_edge_length", self.base_max_edge_length)
        num_workers: Optional[int] = params.get("num_workers", None)

        fps_param = params.get("fps", None)
        if fps_param is None or float(fps_param) <= 0:
            fps_param = float(self.base_fps)

        max_total_frames = params.get("max_total_frames", None)
        if max_total_frames is None:
            max_total_frames = int(self.base_frame_num)
            
        max_total_frames = min(max_total_frames, self.base_frame_num)

        segments_to_sample = [{"interval": params["interval"], "fps": float(fps_param)}]

        prompt_time_range = None
        if self.use_time_prompt:
            user_ptr = params.get('prompt_time_range', None)
            try:
                user_ptr_val = float(user_ptr) if user_ptr is not None else None
            except Exception:
                user_ptr_val = None
            if user_ptr_val is not None and user_ptr_val > 0:
                prompt_time_range = user_ptr_val
            else:
                seg_len = max(0.0, interval.end_sec - interval.start_sec)
                prompt_time_range = max(self.base_prompt_time_range, (seg_len / 5.0) if seg_len > 0 else self.base_prompt_time_range)

        frames, budget, meta = extract_and_resize_frames(
            video_path=params["video_path"],
            segments=segments_to_sample,
            max_edge_length=max_edge_length,
            max_total_frames=max_total_frames,
            num_workers=num_workers,
            prompt_time_range=prompt_time_range
        )

        use_subject_registry = bool(params.get("subject_registry", False))

        mllm_results = self.call_mllm_backend(
            frame_groups=[frames],
            query=params["query"],
            group_intervals=[interval],
            meta=meta,
            subject_registry=use_subject_registry,
        )

        observations = [Observation(**res) for res in mllm_results]
        output = ToolOutput(
            observations=observations,
            coverage=[interval],
            used_budget=budget,
            provenance=Provenance(tool_name=self.name, tool_params=params)
        )
        observe_text_message = self._postprocess_output(output)
        subtitles = params.get("subtitles")
        if subtitles is not None:
            subtitles = self._retrieve_subtitles(interval, subtitles)
            if subtitles is not None:
                observe_text_message += f"[subtitle]{subtitles}"
        return observe_text_message, output.model_dump()
