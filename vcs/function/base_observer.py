
from abc import abstractmethod
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np
import base64
import os
import cv2
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from utils.tool_base import ToolBase
from ..data_models import Observation, TimeInterval
from ..prompts import SUBJECT_REGISTRY_JSON_PROMPT

try:
    from openai import OpenAI, OpenAIError
except Exception:
    OpenAI = None
    class OpenAIError(Exception):
        pass

class BaseVideoObserver(ToolBase):
    """Base class for video observation tools.

    The base class owns runtime defaults, OpenAI-compatible API wiring, prompt
    rendering, frame-to-message conversion, retries, and response parsing. Tool
    subclasses remain responsible for sampling strategy and output packaging.
    """
    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        base_fps: float = 1.0,
        base_frame_num: int = 1200,
        base_max_edge_length: int = 512,
        openai_base_url: Optional[str] = None,
        openai_key: Optional[str] = None,
        mllm_backend: Optional[Dict[str, Any]] = None,
        api_workers: Optional[int] = None,
        use_time_prompt: bool = True,
        type:str ="func",
        **base_kwargs
    ):
        super().__init__(name=name, type=type, arguments=arguments)
        self.base_fps = float(base_fps)
        self.base_frame_num = int(base_frame_num)
        self.base_max_edge_length = int(base_max_edge_length)

        self.openai_base_url = openai_base_url or os.getenv("TOOL_OPENAI_BASE_URL")
        self.openai_key = openai_key or os.getenv("TOOL_OPENAI_API_KEY")
        if os.getenv("MLLM_BACKEND") is not None or isinstance(mllm_backend, str):
            self.mllm_backend = {"type": "api", "model": os.getenv("MLLM_BACKEND") or mllm_backend}
        else:
            self.mllm_backend = mllm_backend or {"type": "api", "model": "gemini-2.5-flash"}        
        self.use_time_prompt = use_time_prompt
        if self.use_time_prompt:
            print(f"{self.__class__.__name__}: using timestamps for VLM calls.")
        else:
            print(f"{self.__class__.__name__}: not adding extra timestamps to VLM calls.")
        self.api_workers = api_workers

        if not hasattr(self, "system_prompt") or not isinstance(getattr(self, "system_prompt"), str):
            self.system_prompt: str = "You are a helpful visual analysis assistant."
        if not hasattr(self, "user_prompt") or not isinstance(getattr(self, "user_prompt"), str):
            self.user_prompt: str = (
                "{query}\n\n"
                "{time_range_text}"
            )
        if not hasattr(self, "group_time_range_prompt"):
            self.group_time_range_prompt: Optional[str] = None

        if not hasattr(self, "bin_time_range_prompt"):
            self.bin_time_range_prompt: Optional[str] = "[{start_sec:g}-{end_sec:g}]:"
            
    def _url_call(self, params: Dict[str, Any], *, identity: Optional[str], bench: Optional[str]) -> Any:
        raise NotImplementedError

    def _validate(self, params: Dict[str, Any]) -> None:
        """Validate required fields while preserving YAML-default behavior."""

        def _meta_is_required(meta: Any) -> bool:
            return isinstance(meta, dict) and bool(meta.get("required"))

        def _provided_any_source(key: str) -> bool:
            if params.get(key, None) is not None:
                return True
            v_direct = getattr(self, key, None)
            if v_direct is not None:
                return True
            v_base = getattr(self, f"base_{key}", None)
            return v_base is not None

        def _provided_from_params_only(key: str) -> bool:
            return params.get(key, None) is not None

        errors: list[str] = []

        required_keys = [k for k, meta in (self.arguments or {}).items() if _meta_is_required(meta)]
        missing = [k for k in required_keys if not _provided_any_source(k)]
        if missing:
            errors.append("Missing required argument(s): " + ", ".join(missing))

        groups = list(self.anyof_groups or [])
        group_errs: list[str] = []
        for idx, group in enumerate(groups, 1):
            try:
                keys = list(group)
            except Exception:
                group_errs.append(f"Group {idx}: invalid any-of group type; expected iterable of keys")
                continue

            unknown = [k for k in keys if k not in (self.arguments or {})]
            if unknown:
                group_errs.append(
                    f"Group {idx}: unknown key(s) in any-of group: {', '.join(unknown)}"
                )
                keys = [k for k in keys if k in (self.arguments or {})]

            if keys and not any(_provided_from_params_only(k) for k in keys):
                group_errs.append(
                    f"Group {idx}: at least one of [{', '.join(keys)}] must be provided in call params (all missing or None)"
                )

        if group_errs:
            errors.extend(group_errs)

        if errors:
            raise ValueError(f"Tool '{self.name}': " + " ; ".join(errors))
        
    @abstractmethod
    def _func_call(self, params: Dict[str, Any]) -> Any:
        pass

    def _img_to_data_url(self, img_bgr) -> str:
        """OpenCV BGR -> JPEG -> base64 data URL (image/jpeg)"""
        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _decide_workers(self, group_count: int) -> int:
        """Choose API worker count from group count and available CPU cores."""
        if group_count <= 1:
            return 1
        if self.api_workers and self.api_workers > 0:
            return min(self.api_workers, group_count)
        try:
            hw = cpu_count()
        except Exception:
            hw = 4
        return max(1, min(hw, group_count))

    def _build_api_client(self):
        """Build the OpenAI-compatible API client and model name."""
        if OpenAI is None:
            raise RuntimeError("The openai SDK is not installed. Install it with `pip install openai`.")
        if not self.openai_key:
            raise ValueError("API backend requires openai_key.")
        client = OpenAI(api_key=self.openai_key, base_url=self.openai_base_url)
        model_name = self.mllm_backend.get("model", "gpt-4o-mini")
        return client, model_name
    
    def _retrieve_subtitles(
        self,
        time_interval: Union["TimeInterval", List["TimeInterval"]],
        subtitle: dict
    ) -> Optional[str]:
        """Extract subtitle lines that overlap the requested interval(s)."""

        def _time_to_seconds(t: str) -> float:
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        raw_subs = subtitle

        subs = []
        if raw_subs is None:
            return None
        for row in raw_subs:
            s = _time_to_seconds(row["start"])
            e = _time_to_seconds(row["end"])
            text = (
                row.get("line", "")
                .replace("&nbsp;", " ")
                .replace("\xa0", " ")
                .strip()
            )
            subs.append((s, e, text))

        intervals = time_interval if isinstance(time_interval, (list, tuple)) else [time_interval]

        seen = set()
        lines: List[str] = []

        for itv in intervals:
            istart, iend = float(itv.start_sec), float(itv.end_sec)
            for s, e, text in subs:
                if e <= istart or s >= iend:
                    continue
                if (s, e, text) in seen:
                    continue
                seen.add((s, e, text))
                o_start = max(s, istart)
                o_end   = min(e, iend)
                lines.append(f"[{o_start}-{o_end}] {text}")
        if not lines:
            return None
        return "\n".join(lines)

    def _build_user_content(
        self,
        query: str,
        frames: List[np.ndarray],
        *,
        group_meta: Optional[Dict[str, Any]] = None,
        total_interval: Optional[Any] = None,
        extra_user_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build Chat Completions user content from text and sampled frames."""
        time_range_text = ""
        if total_interval is not None:
            tpl = getattr(self, "group_time_range_prompt", None)

            if isinstance(total_interval, dict):
                ts = float(total_interval.get("start_sec", 0.0))
                te = float(total_interval.get("end_sec", 0.0))
                interval_text = f"[{ts:g}-{te:g}]"
                if isinstance(tpl, str) and tpl:
                    try:
                        rendered = tpl.format(start_sec=ts, end_sec=te, interval_text=interval_text)
                        if ("{start_sec" not in tpl) and ("{end_sec" not in tpl) and ("{interval_text" not in tpl):
                            time_range_text = f"{rendered} {interval_text}".strip()
                        else:
                            time_range_text = rendered
                    except Exception:
                        time_range_text = f"Frames are from {interval_text} of original video."
                else:
                    time_range_text = f"Frames are from {interval_text} of original video."

            elif isinstance(total_interval, list):
                interval_pairs: List[str] = []
                for it in total_interval:
                    try:
                        ts = float(it.get("start_sec", 0.0))
                        te = float(it.get("end_sec", 0.0))
                    except Exception:
                        continue
                    interval_pairs.append(f"[{ts:g}-{te:g}]")
                intervals_text = ",".join(interval_pairs)
                if isinstance(tpl, str) and tpl:
                    try:
                        rendered = tpl.format(intervals_text=intervals_text)
                        if ("{" not in tpl) and ("}" not in tpl):
                            time_range_text = f"{rendered} {intervals_text}".strip()
                        else:
                            time_range_text = rendered
                    except Exception:
                        time_range_text = f"Frames are from {intervals_text} of original video."
                else:
                    time_range_text = f"Frames are from {intervals_text} of original video."
            else:
                time_range_text = ""

        try:
            first_text = self.user_prompt.format(query=query, time_range_text=time_range_text)
        except Exception:
            first_text = f"{query}\n\n{time_range_text}".strip()

        if extra_user_prompt:
            extra_text = extra_user_prompt.strip()
            if extra_text:
                if first_text:
                    first_text = f"{first_text.rstrip()}\n\n{extra_text}"
                else:
                    first_text = extra_text

        items: List[Dict[str, Any]] = [{"type": "text", "text": first_text}]

        if group_meta:
            bins = group_meta.get("bins") or []
            gmap: Dict[int, List[int]] = group_meta.get("group_to_indices") or {}
            for i, bin_itv in enumerate(bins):
                indices = gmap.get(i, [])
                if not indices:
                    continue

                bs = float(bin_itv.get("start_sec", 0.0))
                be = float(bin_itv.get("end_sec", 0.0))
                tpl = getattr(self, "bin_time_range_prompt", None)
                if isinstance(tpl, str) and tpl:
                    try:
                        bin_text = tpl.format(start_sec=bs, end_sec=be)
                    except Exception:
                        bin_text = f"[{bs:g}-{be:g}]:"
                else:
                    bin_text = f"[{bs:g}-{be:g}]:"

                items.append({"type": "text", "text": bin_text})

                for j in indices:
                    if 0 <= j < len(frames):
                        items.append({
                            "type": "image_url",
                            "image_url": {"url": self._img_to_data_url(frames[j])}
                        })
        else:
            for img in frames:
                items.append({
                    "type": "image_url",
                    "image_url": {"url": self._img_to_data_url(img)}
                })
        return items

    def _build_messages(self, user_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build API messages with the tool-level system prompt."""
        sys_prompt = getattr(self, "system_prompt", "You are a helpful visual analysis assistant.")
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def _is_transient_error(self, exc: Exception) -> bool:
        """Return whether an exception is worth retrying."""
        msg = str(exc)
        return (
            isinstance(exc, OpenAIError)
            or "rate limit" in msg.lower()
            or "timed out" in msg.lower()
            or "too many requests" in msg.lower()
            or "forbidden for url" in msg.lower()
            or "internal" in msg.lower()
        )
    
    def _call_with_backoff(
        self,
        func,
        *args,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        max_retries: int = 8,
        **kwargs
    ):
        """Call a function with exponential backoff on transient errors."""
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry = True
                if self._is_transient_error(e) and retry:
                    num_retries += 1
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None
                    delay *= exponential_base * (1 + (random.random() if jitter else 0.0))
                    delay = min(20, delay)
                    print(f"Retrying in {delay:.2f}s due to: {str(e)}")
                    time.sleep(delay)
                else:
                    print(str(e))
                    raise

    def _api_chat_completions(
        self,
        client,
        model_name: str,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Make the OpenAI-compatible Chat Completions request."""
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format:
            kwargs["response_format"] = response_format
        parts = str(kwargs["model"]).rsplit("-", 1)
        if len(parts) == 2 and parts[1] in {"minimal", "low", "medium", "high"}: 
            kwargs.update(model=parts[0], reasoning_effort=parts[1])

        if 'qwen' in model_name:
            kwargs["extra_body"] = {"enable_thinking": False}
        return client.chat.completions.create(**kwargs)

    def _extract_text_from_response(self, resp) -> str:
        """Extract assistant text from a Chat Completions response."""
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _parse_observation_payload(
        self,
        raw_text: str,
        *,
        subject_registry: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        """Parse optional subject-registry JSON returned by the observer model."""
        if not subject_registry:
            return raw_text, {}

        text = raw_text.strip()
        if not text:
            return "", {}

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return raw_text, {}

        subject_registry = payload.get("subject_registry")
        if not isinstance(subject_registry, dict):
            subject_registry = {}

        observation = payload.get("observation")
        if isinstance(observation, str):
            obs_text = observation
        elif observation is None:
            obs_text = raw_text
        else:
            try:
                obs_text = json.dumps(observation, ensure_ascii=False)
            except (TypeError, ValueError):
                obs_text = str(observation)

        return obs_text, subject_registry

    def _interval_dict_for_group(self, group_intervals: Optional[List[TimeInterval]], idx: int) -> Dict[str, float]:
        """Convert a group interval into the serialized observation format."""
        if group_intervals and idx < len(group_intervals):
            itv = group_intervals[idx]
            if isinstance(itv,list):
                return [{"start_sec": it.start_sec, "end_sec": it.end_sec} for it in itv]
            return {"start_sec": itv.start_sec, "end_sec": itv.end_sec}
        return {"start_sec": 0.0, "end_sec": 0.0}

    def _make_observation(
        self,
        text: str,
        interval_dict: Dict[str, float],
        subject_registry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the serialized observation payload."""
        return {
            "interval": interval_dict,
            "description": text if text else "(model returned no content)",
            "subject_registry": subject_registry or {},
        }

    def call_mllm_backend(
        self,
        frame_groups: List[List[np.ndarray]],
        query: str,
        group_intervals: Optional[List[TimeInterval]] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        *,
        subject_registry: bool = False,
    ) -> List[Dict[str, Any]]:
        """Call the configured multimodal backend for each frame group."""
        backend_type = (self.mllm_backend.get("type") if self.mllm_backend else None) or "api"
        use_subject_registry = bool(subject_registry)

        if backend_type == "api":
            client, model_name = self._build_api_client()

            def _task(idx: int):
                frames = frame_groups[idx]
                gmeta = meta[idx] if (meta and idx < len(meta)) else None

                group_total_interval: Optional[Dict[str, float]] = None
                if isinstance(gmeta, dict) and ("start_sec" in gmeta) and ("end_sec" in gmeta):
                    group_total_interval = {
                        "start_sec": float(gmeta.get("start_sec", 0.0)),
                        "end_sec": float(gmeta.get("end_sec", 0.0))
                    }
                elif group_intervals and idx < len(group_intervals):
                    itv = group_intervals[idx]
                    if not isinstance(itv, list):
                        group_total_interval = {"start_sec": itv.start_sec, "end_sec": itv.end_sec}
                    else:
                        group_total_interval = [{"start_sec": r.start_sec, "end_sec": r.end_sec} for r in itv]

                user_content = self._build_user_content(
                    query,
                    frames,
                    group_meta=gmeta,
                    total_interval=group_total_interval,
                    extra_user_prompt=SUBJECT_REGISTRY_JSON_PROMPT if use_subject_registry else None,
                )
                messages = self._build_messages(user_content)

                try:
                    resp = self._call_with_backoff(
                        self._api_chat_completions,
                        client,
                        model_name,
                        messages,
                        response_format={"type": "json_object"} if use_subject_registry else None,
                    )
                except Exception as e:
                    print(f"Group {idx} observer call failed: {str(e)}")
                    raise

                raw_text = self._extract_text_from_response(resp) if resp is not None else ""
                parsed_text, registry_payload = self._parse_observation_payload(
                    raw_text,
                    subject_registry=use_subject_registry,
                )
                interval_dict = self._interval_dict_for_group(group_intervals, idx)
                obs = self._make_observation(parsed_text, interval_dict, registry_payload)
                return idx, obs

            n = len(frame_groups)
            if n == 0:
                return []

            max_workers = self._decide_workers(n)
            results: List[tuple] = []

            if max_workers == 1:
                for i in range(n):
                    results.append(_task(i))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_task, i): i for i in range(n)}
                    for fut in as_completed(futures):
                        results.append(fut.result())

            results.sort(key=lambda x: x[0])
            return [obs for _, obs in results]

        elif backend_type == "url":
            raise NotImplementedError("mllm_backend.type='url' is not implemented. Use 'api'.")
        elif backend_type == "model":
            raise NotImplementedError("mllm_backend.type='model' is not implemented. Use 'api'.")
        else:
            raise ValueError(f"Unknown mllm_backend.type: {backend_type}")
