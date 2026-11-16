"""Config-driven tool orchestrator for LensWalk VCS tools.

Goals
-----
- Wrap tool calls to:
  - inject `video_path` for observers
  - clip and validate time intervals against the video duration
  - pass through optional call-level metadata such as `prompt_time_range`
- Keep tools authoritative for their own defaults/SDK wiring (no key/env duplication here).

Compatible tools
----------------
- segment_observer
- stitched_observer
- scan_observer
- finish

Usage
-----
from utils.tool_patch import ToolDispatcher
from vcs_dispatcher import VCSDispatcher

d = VCSDispatcher()
d.load_config("/path/to/tools.yaml")  # same schema as ToolDispatcher expects

# Example: single segment
res = d.call_tool(
    {"name": "op.single.segment_probe", "arguments": {
        "interval": {"start_sec": 12.0, "end_sec": 18.5},
        "query": "Describe the main action"
    }},
    video_path="/data/video.mp4",
    duration_sec=243.2,
    prompt_time_range=1.0,   # optional, forwarded as-is
)
print(res)

# Example: stitched
res = d.call_tool(
    {"name": "op.multi.stitched_probe", "arguments": {
        "segments": [
            {"interval": {"start_sec": 5.0, "end_sec": 8.0}, "fps": 6.0},
            {"interval": {"start_sec": 9.0, "end_sec": 12.5}},  # fps optional
        ],
        "global_interval": {"start_sec": 5.0, "end_sec": 13.0},
        "query": "Outline the narrative beats"
    }},
    video_path="/data/video.mp4",
    duration_sec=243.2,
)

# Example: scan
res = d.call_tool(
    {"name": "op.scan.partitioned_parallel", "arguments": {
        "global_interval": {"start_sec": 0.0, "end_sec": 30.0},
        "num_slices": 10,
        "query": "Where does a car appear?"
    }},
    video_path="/data/video.mp4",
    duration_sec=243.2,
)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
from utils.tool_patch import ToolDispatcher  # base + registry loading
from utils.tool_registry import get_tool_class
from pathlib import Path
import yaml
from copy import deepcopy

# Importing this package registers the VCS tool classes via decorators.
import vcs.function  # noqa: F401


@dataclass
class CallContext:
    """Per-call context supplied *by the caller* (not probed from video).

    Attributes
    ----------
    video_path: str
        Absolute or relative path to the video file. Injected into tool args.
    duration_sec: float
        Total duration of the video, used to clip intervals.
    prompt_time_range: Optional[float]
        If provided, forwarded verbatim as `prompt_time_range` to tools. Tools
        that support windowed time prompts will use it; others will ignore it.
    clip_policy: str
        Interval handling when out of bounds. One of {"intersect", "drop"}.
        - intersect: clamp to [0, duration]; drop if empty after clamp.
        - drop: drop any interval with parts outside [0, duration].
    min_span_sec: float
        Minimum positive span required for an interval to be considered valid.
    """

    video_path: str
    subtitles: Optional[dict]
    duration_sec: float
    prompt_time_range: Optional[float] = None
    clip_policy: str = "intersect"
    min_span_sec: float = 1e-6
    allowed_interval: Optional[Tuple[float, float]] = None


class VCSDispatcher(ToolDispatcher):
    """Config-driven dispatcher with argument normalization for VCS tools.

    This class sticks to wiring/orchestration only. It delegates *all* tool
    semantics (defaults, OpenAI client setup, concurrency, etc.) to the tools
    themselves.
    """
    def __init__(
        self,
        *,
        agent_tool_call: str = 'return', 
        strict_call: bool = True,
        subject_registry: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_tool_call = agent_tool_call
        self.strict_call = strict_call
        self.subject_registry = bool(subject_registry)
        
    def parse_tool_call(self, raw: Any):
        if getattr(self, "agent_tool_call", None) == "return":
            raw_json = raw.model_dump() if raw is not None else None
            try:
                args = json.loads(raw_json["function"]["arguments"])
                if "database" in args:
                    args.pop("database")
            except json.JSONDecodeError as exc:
                print(f"Error decoding arguments: {exc!s}")
                return None
                
            tool_args_dict = {
                'name': raw_json["function"]['name'],
                'arguments': args
            }
            return tool_args_dict, raw_json
        return super().parse_tool_call(raw)
    
    # --- public API -----------------------------------------------------
    def call_tool(
        self,
        tool_call: Dict[str, Any],
        *,
        video_path: str,
        subtitles: Any = None,
        duration_sec: float,
        prompt_time_range: Optional[float] = None,
        clip_policy: str = "intersect",
        min_span_sec: float = 1e-6,
        allowed_interval: Optional[Tuple[float, float]] = None,
        max_budget_cap: Optional[int] = None,
    ) -> Any:
        """Invoke a registered tool with normalized arguments.

        Parameters
        ----------
        tool_call: dict
            {"name": <tool name>, "arguments": {...}} - same shape as
            ToolDispatcher.load_config expects for tool names (no JSON needed).
        video_path: str
            Path to the video; injected into arguments as `video_path`.
        duration_sec: float
            Total duration used to clip validation windows.
        prompt_time_range: float, optional
            Forwarded as `prompt_time_range` if provided.
        clip_policy: {"intersect", "drop"}
            How to handle out-of-bounds intervals.
        min_span_sec: float
            Minimum span required after clipping.
        allowed_interval: tuple(float, float), optional
            Optional sub-window ``(start, end)`` constraining Actor tool calls.
        max_budget_cap: int, optional
            Optional frame cap enforcing Actor or phase frame budgets.
        """
        if not isinstance(tool_call, dict) or "name" not in tool_call:
            raise ValueError("tool_call must be a dict with a 'name' key")

        args = dict(tool_call.get("arguments", {}) or {})
        name = tool_call["name"].lower()
        if self.subject_registry and isinstance(args, dict):
            if name and name not in {"finish", "submit_report"}:
                args.setdefault("subject_registry", True)

        if max_budget_cap is not None and int(max_budget_cap) > 0:
            cap = int(max_budget_cap)
            if name == "overview_observer":
                req_n = args.get("num_frames")
                args["num_frames"] = min(int(req_n), cap) if req_n is not None else cap
            elif name not in {"finish", "submit_report"}:
                req_max = args.get("max_total_frames")
                args["max_total_frames"] = min(int(req_max), cap) if req_max is not None else cap

        ctx = CallContext(
            video_path=video_path,
            duration_sec=float(duration_sec),
            prompt_time_range=prompt_time_range,
            clip_policy=clip_policy,
            min_span_sec=float(min_span_sec),
            subtitles=subtitles,
            allowed_interval=allowed_interval,
        )

        args["video_path"] = ctx.video_path
        args["subtitles"] = ctx.subtitles
        if ctx.prompt_time_range is not None:
            args["prompt_time_range"] = float(ctx.prompt_time_range)

        if name == "segment_probe":
            args = self._normalize_single(args, ctx)
        elif name == "stitched_probe":
            args = self._normalize_stitched(args, ctx)
        elif name == "partitioned_parallel":
            args = self._normalize_scan(args, ctx)
        else:
            # Other tools: still clip any common interval keys if present
            args = self._normalize_best_effort(args, ctx)

        tool = self._registry.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")
        return tool(args)

    def load_config(self, path: str | Path) -> None:
        cfg = yaml.safe_load(Path(path).read_text())
        tools_cfg = cfg.get("tools", cfg) if isinstance(cfg, dict) else cfg
        if not isinstance(tools_cfg, list):
            raise ValueError("Config must be a list or contain 'tools' list")
        self._bench = cfg.get("bench") if isinstance(cfg, dict) else None

        need_schema = getattr(self, "agent_tool_call", None) == "return"
        if need_schema:
            self._tool_format_schema: list[dict[str, Any]] = []

        for entry in tools_cfg:
            tool_name = entry["name"]
            class_path = entry.get("class", tool_name)          
            
            tool_cls = get_tool_class(class_path)
            if tool_cls is None:
                raise ValueError(f"Tool class is not registered: {class_path}")
            params = {k: v for k, v in entry.items() if k not in ["class", "defs", "description"]}
            arguments = dict(entry.get("arguments", {}))
            params["arguments"] = arguments
            tool_instance = tool_cls(**params)
            print(f"Loaded tool: {tool_name} -> {tool_cls.__name__}")

            args_schema = entry.get("arguments", {})
            setattr(tool_instance, "_arguments_schema", args_schema)

            if need_schema:
                self._tool_format_schema.append(
                    {
                        "function": self._entry_to_schema(entry),
                        "type": "function",
                    }
                )
            self._registry[tool_name.lower()] = tool_instance
            
    @property
    def tool_format_schema(self) -> list[dict[str, Any]] | None:
        return getattr(self, "_tool_format_schema", None)

    def get_tool_schemas(self, allowed_names: Optional[set[str]] = None) -> list[dict[str, Any]]:
        """Return OpenAI tool schemas optionally filtered by tool name."""
        schemas = self.tool_format_schema or []
        if allowed_names is None:
            return list(schemas)
        allowed_lower = {n.lower() for n in allowed_names}
        return [
            t for t in schemas
            if t.get("type") == "function"
            and (t.get("function", {}).get("name") or "").lower() in allowed_lower
        ]
    
    def _entry_to_schema(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a YAML tool entry into an OpenAI function JSON Schema.

        Strict mode follows the Structured Outputs subset:
        - root and nested objects have additionalProperties=False
        - every property is listed in required
        - originally optional fields are nullable
        """
        name = entry["name"]
        description = entry.get("description", "")
        strict = bool(getattr(self, "strict_call", False))

        params_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }

        if "defs" in entry and isinstance(entry["defs"], dict):
            params_schema["$defs"] = deepcopy(entry["defs"])

        required_flags: Dict[str, bool] = {}
        for arg_name, arg_spec_raw in entry.get("arguments", {}).items():
            spec = {} if arg_spec_raw is None else dict(arg_spec_raw)

            desc = spec.pop("description", "")
            is_required = bool(spec.pop("required", False))

            ref = spec.get("$ref")
            t = spec.get("type")

            prop: Dict[str, Any] = {}
            if ref:
                prop["$ref"] = ref
            else:
                if t:
                    prop["type"] = t
                for k in (
                    "enum", "const", "default", "format", "pattern",
                    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                    "minLength", "maxLength",
                    "minItems", "maxItems",
                    "items", "properties", "additionalProperties"
                ):
                    if k in spec:
                        prop[k] = spec[k]

            if desc:
                prop["description"] = desc

            params_schema["properties"][arg_name] = prop
            required_flags[arg_name] = is_required

        if strict:
            params_schema["required"] = list(params_schema["properties"].keys())

            for arg_name, was_required in required_flags.items():
                if was_required:
                    continue
                prop = params_schema["properties"][arg_name]
                if "$ref" in prop:
                    ref = prop.pop("$ref")
                    desc = prop.pop("description", None)
                    new_prop = {"anyOf": [ {"$ref": ref}, {"type": "null"} ]}
                    if desc:
                        new_prop["description"] = desc
                    params_schema["properties"][arg_name] = new_prop
                else:
                    if "type" in prop:
                        typ = prop["type"]
                        if isinstance(typ, list):
                            if "null" not in typ:
                                prop["type"] = typ + ["null"]
                        elif isinstance(typ, str):
                            if typ != "null":
                                prop["type"] = [typ, "null"]
                    else:
                        prop["type"] = ["string", "null"]
        else:
            req = [k for k, v in required_flags.items() if v]
            if req:
                params_schema["required"] = req

        self._enforce_objects_additional_properties_false(params_schema)

        return {
            "name": name,
            "description": description,
            "strict": strict,
            "parameters": params_schema,
        }


    def _enforce_objects_additional_properties_false(self, schema: Dict[str, Any]) -> None:
        """
        Recursively set additionalProperties=False on every object node.
        """
        def _walk(node: Any):
            if not isinstance(node, dict):
                return

            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
                props = node.get("properties", {})
                if isinstance(props, dict):
                    for v in props.values():
                        _walk(v)

            if "items" in node:
                _walk(node["items"])

            if "anyOf" in node and isinstance(node["anyOf"], list):
                for s in node["anyOf"]:
                    _walk(s)

            if "$defs" in node and isinstance(node["$defs"], dict):
                for v in node["$defs"].values():
                    _walk(v)

        _walk(schema)

    # --- normalization helpers -----------------------------------------
    def _normalize_single(self, args: Dict[str, Any], ctx: CallContext) -> Dict[str, Any]:
        if "interval" not in args or not isinstance(args["interval"], dict):
            raise ValueError("segment_probe requires 'interval' dict")
        clipped = self._clip_interval(args["interval"], ctx)
        if clipped is None:
            raise ValueError("interval is outside video duration after clipping")
        out = dict(args)
        out["interval"] = clipped
        return out

    def _normalize_stitched(self, args: Dict[str, Any], ctx: CallContext) -> Dict[str, Any]:
        if "segments" not in args or not isinstance(args["segments"], list) or not args["segments"]:
            raise ValueError("stitched_probe requires non-empty 'segments' list")

        segs: List[Dict[str, Any]] = []
        for seg in args["segments"]:
            if not isinstance(seg, dict) or "interval" not in seg:
                # skip malformed entries
                continue
            ci = self._clip_interval(seg["interval"], ctx)
            if ci is None:
                continue  # drop empty/out-of-range pieces
            # carry optional fps as-is; do not assign defaults here
            segs.append({**seg, "interval": ci})
        if not segs:
            raise ValueError("no valid segments remain after clipping")

        out = dict(args)
        out["segments"] = segs

        if "global_interval" in args and isinstance(args["global_interval"], dict):
            gi = self._clip_interval(args["global_interval"], ctx)
            if gi is not None:
                out["global_interval"] = gi
            else:
                # Remove invalid global_interval; tool will fallback to per-segment coverage
                out.pop("global_interval", None)
        return out

    def _normalize_scan(self, args: Dict[str, Any], ctx: CallContext) -> Dict[str, Any]:
        if "global_interval" not in args or not isinstance(args["global_interval"], dict):
            raise ValueError("partitioned_parallel requires 'global_interval' dict")
        gi = self._clip_interval(args["global_interval"], ctx)
        if gi is None:
            raise ValueError("global_interval is outside video duration after clipping")

        # Respect the tool contract: exactly one of num_slices / slice_duration_sec
        ns = args.get("num_slices")
        sd = args.get("slice_duration_sec")
        if (ns is None and sd is None) or (ns is not None and sd is not None):
            # Do not fix silently; surface to caller for clarity
            raise ValueError("Provide exactly one of 'num_slices' or 'slice_duration_sec'")

        out = dict(args)
        out["global_interval"] = gi
        return out

    def _normalize_best_effort(self, args: Dict[str, Any], ctx: CallContext) -> Dict[str, Any]:
        out = dict(args)
        for key in ("interval", "global_interval"):
            if key in out and isinstance(out[key], dict):
                ci = self._clip_interval(out[key], ctx)
                if ci is None:
                    lo = 0.0 if ctx.allowed_interval is None else max(0.0, ctx.allowed_interval[0])
                    hi = ctx.duration_sec if ctx.allowed_interval is None else min(ctx.duration_sec, ctx.allowed_interval[1])
                    raise ValueError(
                        f"'{key}' {out[key]} has no overlap with allowed bounds [{lo:.1f}, {hi:.1f}]."
                    )
                out[key] = ci
        if "segments" in out and isinstance(out["segments"], list):
            segs: List[Dict[str, Any]] = []
            for seg in out["segments"]:
                if not isinstance(seg, dict):
                    continue
                if "interval" in seg and isinstance(seg["interval"], dict):
                    ci = self._clip_interval(seg["interval"], ctx)
                    if ci is not None:
                        segs.append({**seg, "interval": ci})
                elif "start_sec" in seg and "end_sec" in seg:
                    ci = self._clip_interval(seg, ctx)
                    if ci is not None:
                        updated = {**seg, **ci}
                        segs.append(updated)
            if out["segments"] and not segs:
                raise ValueError("No valid segments remain after clipping to allowed bounds.")
            out["segments"] = segs
        return out

    # --- low-level interval utilities ----------------------------------
    def _clip_interval(self, itv: Dict[str, Any], ctx: CallContext) -> Optional[Dict[str, float]]:
        """Clip [start, end] to [0, duration] and optional ``allowed_interval``."""
        try:
            s = float(itv.get("start_sec", 0.0))
            e = float(itv.get("end_sec", 0.0))
        except Exception as exc:
            raise ValueError(f"invalid interval fields: {itv}") from exc

        lo = 0.0
        hi = ctx.duration_sec
        if ctx.allowed_interval is not None:
            lo = max(lo, float(ctx.allowed_interval[0]))
            hi = min(hi, float(ctx.allowed_interval[1]))

        if ctx.clip_policy not in {"intersect", "drop"}:
            raise ValueError("clip_policy must be 'intersect' or 'drop'")

        if ctx.clip_policy == "drop":
            if s < lo or e > hi or e <= s:
                return None
            return {"start_sec": s, "end_sec": e}

        s = max(lo, min(s, hi))
        e = max(lo, min(e, hi))
        if e - s < ctx.min_span_sec:
            return None
        return {"start_sec": s, "end_sec": e}
