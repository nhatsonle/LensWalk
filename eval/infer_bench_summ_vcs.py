import os
import re
import json
import time
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
from openai import OpenAI, OpenAIError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from vcs.vcs_dispatcher import VCSDispatcher
from vcs.vcs_utils import StopException, encode_video, strip_image_urls
from vcs.data_models import SubjectRegistryState
import vcs.prompts as prompts
from benchmarks.benchmarks import load_benchmark
from threading import Lock
from utils.static_analysis import summarize_tool_calls
from utils.eval_utils import eval_open_ended


RUN_START_TIME = time.strftime('%Y%m%d_%H%M%S')


def _short_model_name(m: Optional[str]) -> Optional[str]:
    return m.split("/")[-1] if m else m


def _safe_path_label(value: Optional[str]) -> str:
    raw = os.path.basename(value or "") or "benchmark"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "benchmark"


@dataclass
class ScriptArgs:
    model_name_or_path: Optional[str] = field(default='gpt-4o')
    data_start: Optional[int] = field(default=0)
    data_end: Optional[int] = field(default=-1)
    index_filter_file: Optional[str] = field(default="")
    bench_name: Optional[str] = field(default='')
    tool_config: Optional[str] = field(default='')
    max_turns: Optional[int] = field(default=5)
    tool_call_strict_schema: Optional[bool] = field(default=True)
    max_scan_calls: Optional[int] = field(default=1)
    limit_scan_calls: Optional[bool] = field(default=False) 

    resume: Optional[bool] = field(default=False)
    resume_log_path: Optional[str] = field(default="")
    batch_size: Optional[int] = field(default=15)
    workers: Optional[int] = field(default=8)
    system_prompt_ref: Optional[str] = field(default="VCS_REASON_PROMPT")
    usr_prompt_format_ref: Optional[str] = field(default="VCS_REASON_USER_PROMPT")
    custom_name: Optional[str] = field(default="")
    insert_frames: Optional[bool] = field(default=False)
    frame_fps: Optional[int] = field(default=None)
    max_frames: Optional[int] = field(default=16)
    video_root: Optional[str] = field(default="")
    force_stitch_summary: Optional[bool] = field(default=False)
    subject_registry: Optional[bool] = field(default=False)
    subtitle_root: Optional[str] = field(default="")
    use_subtitle: Optional[bool] = field(default=False)



SKIP_KEYS = {"resume", "resume_log_path", "workers", "batch_size"}

def _resolve_prompt(ref: str, **fmt_kwargs) -> str:
    prompt = getattr(prompts, ref, ref)
    if fmt_kwargs:
        try:
            return prompt.format(**fmt_kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return prompt

def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    if not args.bench_name:
        raise ValueError("Please pass --bench_name with a JSON/JSONL path or a registered benchmark name.")
    SCAN_LIMIT_ENABLED = bool(args.limit_scan_calls)
    SCAN_CAP = args.max_scan_calls if args.max_scan_calls is not None else -1
    SUBJECT_REGISTRY_ENABLED = bool(args.subject_registry)
    subject_registry_system_prompt = prompts.SUBJECT_REGISTRY_UPDATE_PROMPT
    subject_registry_model_name = args.model_name_or_path

    def scan_allowed(current_count: int) -> bool:
        if not SCAN_LIMIT_ENABLED:
            return True
        if SCAN_CAP is None or SCAN_CAP < 0:
            return True
        if SCAN_CAP == 0:
            return False
        return current_count < SCAN_CAP


    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_BASE_URL")
    bench_label = _safe_path_label(args.bench_name)

    chat_pool = ThreadPoolExecutor(max_workers=args.workers)
    tool_pool = ThreadPoolExecutor(max_workers=args.workers)
    frame_pool = ThreadPoolExecutor(max_workers=args.workers)
    subject_registry_pool = ThreadPoolExecutor(max_workers=args.workers)

    completed_indices = set()
    if args.resume and args.resume_log_path:
        with open(os.path.join(args.resume_log_path, "run_settings.json"), "r", encoding="utf-8") as f:
            prev_cfg = json.load(f)
        for k, v in prev_cfg["args"].items():
            if k not in SKIP_KEYS:
                setattr(args, k, v)
        conv_path = os.path.join(args.resume_log_path, "all_conversations.jsonl")
        if os.path.exists(conv_path):
            with open(conv_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        completed_indices.add(json.loads(line)["index"])
                    except Exception:
                        pass
        print(f"[Resume] Loaded previous run from: {args.resume_log_path}")
        print(f"[Resume] Completed items detected: {len(completed_indices)}")
    mllm = os.getenv("MLLM_BACKEND")
    run_dir_name = (
        f"run_lods/{bench_label}_{_short_model_name(args.model_name_or_path)}"
        f"_maxturn{args.max_turns}_s{args.data_start}_e{args.data_end}_{RUN_START_TIME}_{mllm}_maxscan{args.max_scan_calls}"
        f"_limitScan{int(bool(args.limit_scan_calls))}"
    )
    if args.custom_name:
        run_dir_name += f"_{args.custom_name}"
    save_dir = os.path.join("results", run_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    conversations_file = os.path.join(save_dir, "all_conversations.jsonl")
    callback_log_file = os.path.join(save_dir, "all_tool_calls.jsonl")


    vcs_agent = VCSDispatcher(
        agent_tool_call='return',
        strict_call=args.tool_call_strict_schema,
        subject_registry=SUBJECT_REGISTRY_ENABLED,
    )

    tool_cfg_content = ""
    if args.tool_config and os.path.exists(args.tool_config):
        with open(args.tool_config, 'r', encoding='utf-8') as _tc:
            tool_cfg_content = _tc.read()
        print(f"Loading tool config from {args.tool_config} ...")
        vcs_agent.load_config(args.tool_config)

    if SCAN_LIMIT_ENABLED and SCAN_CAP == 0 and vcs_agent.tool_format_schema:
        vcs_agent._tool_format_schema = [
            t for t in vcs_agent.tool_format_schema
            if not (
                t.get("type") == "function" and
                t.get("function", {}).get("name") == "scan_observer"
            )
        ]
    
    tool_names = [
        (t.get("function", {}).get("name") or "").lower()
        for t in (vcs_agent.tool_format_schema or [])
        if t.get("type") == "function"
    ]
    has_finish_tool = "finish" in tool_names
    has_stitched_tool = "stitched_observer" in tool_names
    print(f"Loaded tool schemas: {', '.join(tool_names)}")
    run_settings = {
        "start_time": RUN_START_TIME,
        "args": {k: v for k, v in vars(args).items() if k not in SKIP_KEYS},
        "tool_config_path": args.tool_config,
        "tool_config_content": tool_cfg_content,
    }
    with open(os.path.join(save_dir, "run_settings.json"), "w", encoding="utf-8") as _rsf:
        json.dump(run_settings, _rsf, ensure_ascii=False, indent=2)

    print(f'Loading dataset {args.bench_name} ...')

    system_prompt = _resolve_prompt(args.system_prompt_ref, max_calls=args.max_turns)
    usr_prompt_format = _resolve_prompt(args.usr_prompt_format_ref)

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    def _json_stringify(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(obj)

    def _normalize_tool_result_payload(result: Any) -> tuple[str, Any]:
        payload = None
        primary_text = ""
        if isinstance(result, tuple):
            for part in result:
                if isinstance(part, dict) and payload is None:
                    payload = part
                elif isinstance(part, str) and not primary_text:
                    primary_text = part
            if not primary_text:
                primary_text = _json_stringify(result)
        elif isinstance(result, dict):
            payload = result
            primary_text = _json_stringify(result)
        elif isinstance(result, str):
            primary_text = result
        else:
            primary_text = _json_stringify(result)
        return primary_text, payload

    def _format_interval_label(interval_obj: Any) -> str:
        def _extract_bounds(it: Any) -> Optional[Tuple[float, float]]:
            if isinstance(it, dict):
                start = it.get("start_sec", it.get("start"))
                end = it.get("end_sec", it.get("end"))
            else:
                start = getattr(it, "start_sec", getattr(it, "start", None))
                end = getattr(it, "end_sec", getattr(it, "end", None))
            if start is None or end is None:
                return None
            try:
                return float(start), float(end)
            except Exception:
                return None

        bounds_list: List[Tuple[float, float]] = []
        if isinstance(interval_obj, list):
            for part in interval_obj:
                bounds = _extract_bounds(part)
                if bounds:
                    bounds_list.append(bounds)
        else:
            bounds = _extract_bounds(interval_obj)
            if bounds:
                bounds_list.append(bounds)

        if not bounds_list:
            return "[unknown]"

        def _fmt_pair(pair: Tuple[float, float]) -> str:
            return f"[{pair[0]:g},{pair[1]:g}]"

        if len(bounds_list) == 1:
            return _fmt_pair(bounds_list[0])
        return " ".join(_fmt_pair(p) for p in bounds_list)

    def _collect_subject_registry(payload: Any) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
        subjects: Dict[str, Any] = {}
        interval_entries: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(payload, dict):
            observations = payload.get("observations")
            if isinstance(observations, list):
                for obs in observations:
                    if not isinstance(obs, dict):
                        continue
                    registry = obs.get("subject_registry")
                    if isinstance(registry, dict) and registry:
                        interval_label = _format_interval_label(obs.get("interval"))
                        interval_entries.append((interval_label, registry))
                        for key, value in registry.items():
                            subjects[key] = value
        return subjects, interval_entries

    @dataclass
    class _SubjectRegistryUpdateResult:
        failed: bool
        observation_summary: str
        updated_subject_registry: Dict[str, Any]
        error_message: Optional[str] = None

    def _format_subject_registry_message(state: SubjectRegistryState) -> str:
        registry = state.subject_registry or {}
        if registry:
            return "[Subject Registry]\n" + _json_stringify(registry)
        return "[Subject Registry]\nNo subject registry entries recorded."

    def _remove_previous_subject_registry_message(session: dict) -> None:
        idx = session.get("subject_registry_message_index")
        if idx is not None:
            if 0 <= idx < len(session["chat_history"]):
                candidate = session["chat_history"][idx]
                if (
                    isinstance(candidate, dict)
                    and candidate.get("role") == "user"
                    and candidate.get("content", "").startswith("[Subject Registry]")
                ):
                    session["chat_history"].pop(idx)
                else:
                    for ridx in range(len(session["chat_history"]) - 1, -1, -1):
                        cand = session["chat_history"][ridx]
                        if (
                            isinstance(cand, dict)
                            and cand.get("role") == "user"
                            and cand.get("content", "").startswith("[Subject Registry]")
                        ):
                            session["chat_history"].pop(ridx)
                            break
            session["subject_registry_message_index"] = None

    def _refresh_subject_registry_message(session: dict) -> None:
        if not SUBJECT_REGISTRY_ENABLED:
            return
        _remove_previous_subject_registry_message(session)
        message = _format_subject_registry_message(session["subject_registry_state"])
        session["chat_history"].append({"role": "user", "content": message})
        session["subject_registry_message_index"] = len(session["chat_history"]) - 1

    def _prepare_subject_registry_update(
        session: dict,
        *,
        tool_name: str,
        tool_call: Optional[Dict[str, Any]],
        raw_text: str,
        raw_payload: Any,
    ) -> Optional[Dict[str, Any]]:
        if not SUBJECT_REGISTRY_ENABLED:
            return None
        if tool_name.lower() == "finish":
            return None

        state: SubjectRegistryState = session.setdefault(
            "subject_registry_state",
            SubjectRegistryState(),
        )
        turn_idx = session.get("observation_step", 0) + 1
        records: List[Dict[str, Any]] = session.setdefault("subject_registry_records", [])
        per_turn_subjects, per_turn_subject_entries = _collect_subject_registry(raw_payload)

        item = session.get("item", {})
        video_duration = item.get("duration") or item.get("video_duration") or item.get("video_length")
        try:
            video_duration_str = f"{float(video_duration):.2f}" if video_duration is not None else "unknown"
        except Exception:
            video_duration_str = str(video_duration)

        context_lines: List[str] = [
            "[Meta]",
            f"Question: {item.get('question', '')}",
            f"Video duration (seconds): {video_duration_str}",
            "",
            "[Current Subject Registry]",
            _json_stringify(state.subject_registry or {}),
        ]

        if records:
            context_lines.extend(["", "[History]"])
            for record in records:
                turn_num = record.get("turn_idx", "unknown")
                context_lines.append(f"[History Turn {turn_num}]")
                context_lines.append("Tool call:")
                context_lines.append(_json_stringify(record.get("tool_call") or {}))
                context_lines.append("Observation summary:")
                context_lines.append(record.get("observation_summary", ""))

        context_lines.extend([
            "",
            f"[Current Turn {turn_idx}]",
            "Tool call:",
            _json_stringify(tool_call or {"name": tool_name}),
            "Raw observation:",
        ])
        raw_block = raw_text
        context_lines.append(raw_block)

        if per_turn_subjects:
            context_lines.append(f"[New observed subject registry of turn {turn_idx}]")
            context_lines.append(_json_stringify(per_turn_subjects))
            if per_turn_subject_entries:
                context_lines.append(f"[Subject registry intervals turn {turn_idx}]")
                for interval_label, registry in per_turn_subject_entries:
                    try:
                        registry_text = json.dumps(registry, ensure_ascii=False)
                    except Exception:
                        registry_text = str(registry)
                    context_lines.append(f"{interval_label}: {registry_text}")

        payload = {
            "model": subject_registry_model_name,
            "messages": [
                {"role": "system", "content": subject_registry_system_prompt},
                {"role": "user", "content": "\n".join(context_lines)},
            ],
            "response_format": {"type": "json_object"},
        }
        return {
            "payload": payload,
            "turn_idx": turn_idx,
            "raw_block": raw_block,
            "raw_text": raw_text,
            "per_turn_subjects": per_turn_subjects,
            "per_turn_subject_entries": per_turn_subject_entries,
            "tool_call": tool_call or {"name": tool_name},
            "tool_name": tool_name,
        }

    def _extract_subject_registry_update(
        parsed: Any,
        *,
        raw_text: str,
        prev_subject_registry: Dict[str, Any],
        per_turn_subjects: Dict[str, Any],
        raw_content: str,
    ) -> _SubjectRegistryUpdateResult:
        updated_subject_registry = dict(prev_subject_registry)
        if isinstance(parsed, dict):
            subject_value = parsed.get("updated_subject_registry")
            if subject_value is None:
                subject_value = parsed.get("subject_registry")
            if isinstance(subject_value, dict):
                updated_subject_registry.update(subject_value)
            elif per_turn_subjects:
                updated_subject_registry.update(per_turn_subjects)
            else:
                return _SubjectRegistryUpdateResult(
                    failed=True,
                    observation_summary=raw_text,
                    updated_subject_registry=prev_subject_registry,
                    error_message=raw_content or "Parsed payload missing 'updated_subject_registry'",
                )
        else:
            return _SubjectRegistryUpdateResult(
                failed=True,
                observation_summary=raw_text,
                updated_subject_registry=prev_subject_registry,
                error_message=raw_content or "Non-JSON response",
            )

        if per_turn_subjects:
            updated_subject_registry.update(per_turn_subjects)

        return _SubjectRegistryUpdateResult(
            failed=False,
            observation_summary=raw_text,
            updated_subject_registry=updated_subject_registry,
        )

    def _subject_registry_task(payload: Dict[str, Any]) -> Optional[Any]:
        request_payload = dict(payload)
        use_json = "response_format" in request_payload
        parts = str(request_payload.get("model", "")).rsplit("-", 1)
        if len(parts) == 2 and parts[1] in {"minimal", "low", "medium", "high"}:
            request_payload.update(model=parts[0], reasoning_effort=parts[1])
        for attempt in range(5):
            try:
                return client.chat.completions.create(**request_payload)
            except OpenAIError as exc:
                print(f"[Subject Registry] OpenAIError attempt {attempt + 1}: {exc}")
                if use_json and "response_format" in request_payload:
                    request_payload = dict(request_payload)
                    request_payload.pop("response_format", None)
                    use_json = False
                if attempt == 4:
                    return exc
                time.sleep(3 * (2 ** attempt))
        return None

    def _decode_subject_registry_output(response: Any) -> tuple[Optional[Dict[str, Any]], str]:
        if response is None:
            return None, ""
        if isinstance(response, Exception):
            return None, ""
        content = ""
        try:
            choices = getattr(response, "choices", None)
            if choices:
                content = choices[0].message.content or ""
        except Exception:
            content = ""
        if not content:
            try:
                content = str(response)
            except Exception:
                content = ""
        parsed = None
        if content:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.S)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = None
        return parsed, content

    ds = load_benchmark(
        args.bench_name,
        data_root=args.video_root or None,
        subtitle_root=args.subtitle_root or None,
        shuffle=True,
        shuffle_seed=42,
    )


    acc = 0
    cnt = 0
    results = []

    if args.resume and args.resume_log_path:
        conv_path = os.path.join(args.resume_log_path, "all_conversations.jsonl")
        if os.path.exists(conv_path):
            with open(conv_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_indices.add(rec["index"])
                        acc += rec.get("score", 0)
                        cnt += 1
                        results.append(rec)
                    except Exception:
                        pass
        if cnt:
            print(f"[Resume] Loaded {cnt} finished items with cumulative accuracy {acc / cnt:.4f}")

    def _chat_task(chat_hist: list[dict], allowed_names: set[str]):
        for attempt in range(5):
            try:
                model = args.model_name_or_path
                kwargs = dict(model=model, messages=chat_hist)
                parts = str(kwargs["model"]).rsplit("-", 1)
                if len(parts) == 2 and parts[1] in {"minimal", "low", "medium", "high"}: 
                    kwargs.update(model=parts[0], reasoning_effort=parts[1])
                if vcs_agent.tool_format_schema:
                    all_tools = vcs_agent.tool_format_schema

                    filtered_tools = [
                        t for t in all_tools
                        if t.get("type") == "function"
                        and t.get("function", {}).get("name") in allowed_names
                    ]

                    if filtered_tools:
                        kwargs["tools"] = filtered_tools
                if model == "qwen3-8b":
                    kwargs["extra_body"] = {"enable_thinking": False}
                res =  client.chat.completions.create(**kwargs).choices[0].message
                return res
            except OpenAIError as e:
                print(f"[OpenAIError] {type(e).__name__}: {e}")
                if attempt == 4:
                    print("[Error] Chat request failed after 5 attempts.")
                    import traceback
                    traceback.print_exc()
                    return None
                time.sleep(3 * (2 ** attempt))

    tool_log_lock = Lock()

    def _tool_task(pquery: dict, item: dict):
        session_id = item.get('id')
        start_time = time.time()
        err_msg = None
        try:
            video_path = item.get('video_path') or item.get('video_path_abs')
            duration = float(item.get('duration') or item.get('video_duration') or 0.0)
            if args.use_subtitle:
                result = vcs_agent.call_tool(
                    pquery,
                    video_path=video_path,
                    duration_sec=duration,
                    subtitles = item['subtitles'],
                )
            else:
                result = vcs_agent.call_tool(
                    pquery,
                    video_path=video_path,
                    duration_sec=duration,
                )
            is_stop = False
        except StopException as exc:
            err_msg = str(exc)
            result = str(exc)
            is_stop = True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_msg = str(exc)
            result = str(exc) or repr(exc)
            is_stop = False
            
        end_time = time.time()
        result_text, result_payload = _normalize_tool_result_payload(result)

        callback_info = {
            "session_id": session_id,
            "tool_name": pquery.get("name"),
            "tool_args": pquery.get("arguments"),
            "result": result_text,
            "interval": result_payload.get("observations", [{}])[0].get("interval") if isinstance(result_payload, dict) else None,
            "frames_used": result_payload.get("used_budget", {}).get("sampled_frames_count") if isinstance(result_payload, dict) else None,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "error": err_msg,
        }

        with tool_log_lock:
            with open(callback_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(callback_info, ensure_ascii=False) + "\n")

        return result, is_stop

    keep_indices = None
    if args.index_filter_file:
        try:
            keep_indices = set()
            with open(args.index_filter_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    for tok in re.split(r"[,\s]+", s):
                        if tok:
                            keep_indices.add(int(tok))
            print(f"[Filter] Loaded {len(keep_indices)} indices from {args.index_filter_file}")
        except Exception as e:
            print(f"[Filter] Failed to load index_filter_file '{args.index_filter_file}': {e}")
            keep_indices = None

    pending_all = [
        (idx, itm) for idx, itm in enumerate(ds)
        if idx >= args.data_start
        and (args.data_end <= 0 or idx < args.data_end)
        and idx not in completed_indices
        and (keep_indices is None or idx in keep_indices)
    ]
    pending_iter = iter(pending_all)
    max_parallel_sessions = max(1, args.batch_size or 1)

    active_sessions: Dict[int, dict] = {}
    ready_sessions: set[int] = set()
    frame_fut2sess: Dict[Any, dict] = {}
    chat_fut2sess: Dict[Any, dict] = {}
    tool_fut2info: Dict[Any, tuple] = {}
    subject_registry_fut2info: Dict[Any, dict] = {}
    all_futures: set = set()
    dataset_exhausted = False

    def _initialize_session(idx: int, item: dict) -> dict:
        question = item['question']
        gt = item.get('golden_answer') or item.get('answer')
        imgs = item.get('image_url')
        subtitle_path = item.get('subtitle_path')
        if subtitle_path is not None and args.use_subtitle == True:
            try:
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    item['subtitles'] = json.load(f)
            except:
                raise
        if isinstance(gt, list):
            gt = gt[0]

        usr_prompt = (
            usr_prompt_format
            .replace("VIDEO_LENGTH", str(f"{item.get('duration', ''):.2f}"))
            .replace("QUESTION_PLACEHOLDER", question).replace('MAX_FRAMES', '50')
        )

        return {
            "i": idx,
            "item": item,
            "gt": gt,
            "usr_prompt": usr_prompt,
            "qs_url" : imgs,
            "imgs": None,
            "chat_history": [],
            "turn_cnt": 0,
            "stop_all": False,
            "output_text": "",
            "score": 0.0,
            "pending_tools": 0,
            "pending_chat": False,
            "pending_subject_registry": 0,
            "finish_prompt_added": False,
            "finalized": False,
            "question_type": item.get("question_type"),
            "video_path": item.get('video_path') or item.get('video_path_abs'),
            "scan_calls": 0,
            "has_stitched": False,
            "stitch_prompt_added": False,
            "subject_registry_state": SubjectRegistryState(),
            "subject_registry_records": [],
            "subject_registry_message_index": None,
            "observation_step": 0,
        }

    def _ensure_initial_history(session: dict) -> None:
        if session["chat_history"]:
            return
        imgs = session.get("imgs") or []
        if imgs:
            user_msg_content = (
                [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im}"}} for im in imgs]
                + [{"type": "text", "text": session["usr_prompt"]}]
            )
        else:
            user_msg_content = session["usr_prompt"]
        session["chat_history"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg_content},
        ]

    def _finalize_session(session: dict) -> None:
        nonlocal acc, cnt
        if session.get("finalized"):
            return
        if not session["output_text"]:
            last_msg = next(
                (m for m in reversed(session["chat_history"]) if m.get("role") != "tool"),
                {},
            )
            session["output_text"] = last_msg.get("content", "")
        if session["question_type"] == "open-ended":
            if session["output_text"] is not None:
                if len(session["output_text"]) == len(session["gt"]):
                    session["score"] = 1.0 if session["output_text"].lower() == session["gt"].lower() else 0.0
                else:
                    example = {
                    "id": session["i"],
                    "question": session["item"]["question"],
                    "answer": session["gt"],
                    "response": session["output_text"],
                    }
                    eval_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
                    is_correct, _, _ = eval_open_ended(example, eval_client)
                    session["score"] = 1.0 if is_correct else 0.0
            else:
                session["score"] = 0.0
        else:
            session["score"] = (
                vcs_agent.compute_score(session["output_text"], session["gt"])
                if hasattr(vcs_agent, "compute_score")
                else 0.0
            )
        acc += session["score"]
        cnt += 1

        conversation_record = {
            "index": session["i"],
            "question": session["item"]["question"],
            "golden_answer": session["gt"],
            "answer": session["output_text"],
            "score": session["score"],
            "chat_history": strip_image_urls(session["chat_history"]),
            "subject_registry": session["subject_registry_state"].subject_registry,
        }
        with open(conversations_file, "a", encoding="utf-8") as _cf:
            _cf.write(json.dumps(conversation_record, ensure_ascii=False) + "\n")

        results.append({**session["item"], "answer": session["output_text"], "score": session["score"]})
        print(f"[idx {session['i']}] score={session['score']} | current ACC={acc / cnt:.4f}")

        session["finalized"] = True
        sid = session["i"]
        active_sessions.pop(sid, None)
        ready_sessions.discard(sid)

    def _maybe_finalize_session(session: dict) -> bool:
        if session.get("finalized"):
            return True
        if session["pending_chat"] or session["pending_tools"] > 0 or session.get("pending_subject_registry", 0) > 0:
            return False
        if not session["stop_all"] and session["turn_cnt"] < args.max_turns:
            return False
        _finalize_session(session)
        session['item'] = {k: v for k, v in session["item"].items() if k != "subtitles"}
        print("session ended", session["item"])
        return True

    while not dataset_exhausted or active_sessions or all_futures or ready_sessions:
        while len(active_sessions) < max_parallel_sessions and not dataset_exhausted:
            try:
                idx, item = next(pending_iter)
            except StopIteration:
                dataset_exhausted = True
                break

            session = _initialize_session(idx, item)
            active_sessions[idx] = session

            if args.insert_frames and session["video_path"]:
                fut = frame_pool.submit(
                    encode_video,
                    session["video_path"],
                    args.max_frames,
                    fps=args.frame_fps,
                )
                frame_fut2sess[fut] = session
                all_futures.add(fut)
            else:
                session["imgs"] = session.get("imgs") or []
                _ensure_initial_history(session)
                ready_sessions.add(idx)

        BASE_ALLOWED = {"segment_observer", "stitched_observer", "scan_observer", "finish"}
        LIMIT_TOOL_NAMES = {"scan_observer"}
        for sid in list(ready_sessions):
            session = active_sessions.get(sid)
            ready_sessions.discard(sid)
            if not session or session.get("finalized"):
                continue
            _ensure_initial_history(session)
            if session["stop_all"]:
                _maybe_finalize_session(session)
                continue
            if session["pending_chat"] or session["pending_tools"] > 0 or session.get("pending_subject_registry", 0) > 0:
                continue
            if session["turn_cnt"] >= args.max_turns:
                session["stop_all"] = True
                _maybe_finalize_session(session)
                continue
            if has_finish_tool and session["turn_cnt"] == args.max_turns - 1 and not session["finish_prompt_added"]:
                session["chat_history"].append(
                    {"role": "user", "content": "Please call the `finish` function to finish the task."}
                )
                session["finish_prompt_added"] = True
            if (
                args.force_stitch_summary
                and has_stitched_tool
                and session["turn_cnt"] == args.max_turns - 2
                and not session.get("has_stitched")
                and not session.get("stitch_prompt_added")
            ):
                session["chat_history"].append(
                    {"role": "user", "content": prompts.FORCE_STITCH_USER_PROMPT}
                )
                session["stitch_prompt_added"] = True
                print("has added forced stitch tool prompt.")
            allowed_names = set(BASE_ALLOWED)
            if not scan_allowed(session.get("scan_calls", 0)):
                allowed_names -= LIMIT_TOOL_NAMES

            fut = chat_pool.submit(_chat_task, session["chat_history"], allowed_names)
            chat_fut2sess[fut] = session
            all_futures.add(fut)
            session["pending_chat"] = True

        if not all_futures:
            # No pending futures; check if any sessions can finish before the next loop.
            for session in list(active_sessions.values()):
                _maybe_finalize_session(session)
            if dataset_exhausted and not active_sessions:
                break
            continue

        done, _ = wait(all_futures, timeout=0.1, return_when=FIRST_COMPLETED)
        if not done:
            continue

        for fut in done:
            all_futures.discard(fut)
            if fut in frame_fut2sess:
                session = frame_fut2sess.pop(fut)
                try:
                    frames = fut.result() or []
                    orig = session.get("imgs") or []
                    session["imgs"] = orig + frames
                except Exception as exc:
                    print(f"[frame] session {session['i']} failed to encode video: {exc}")
                    session["stop_all"] = True
                if session["stop_all"]:
                    _maybe_finalize_session(session)
                else:
                    _ensure_initial_history(session)
                    ready_sessions.add(session["i"])
                continue

            if fut in chat_fut2sess:
                session = chat_fut2sess.pop(fut)
                session["pending_chat"] = False
                output_message = fut.result()
                if output_message is None:
                    session["stop_all"] = True
                    _maybe_finalize_session(session)
                    continue

                tool_calls = getattr(output_message, "tool_calls", None) or []
                last = any(
                    getattr(tc.function, "name", "").lower() == "finish"
                    for tc in tool_calls
                    if getattr(tc, "function", None)
                )
                stitch_insert_rule = (
                    args.force_stitch_summary
                    and has_stitched_tool
                    and not session["finish_prompt_added"]
                    and session["turn_cnt"] < args.max_turns - 1
                    and not session.get("has_stitched")
                    and not session.get("stitch_prompt_added")
                )
                if stitch_insert_rule and last:
                    session["chat_history"].append(
                        {"role": "user", "content": prompts.FORCE_STITCH_USER_PROMPT}
                    )
                    session["stitch_prompt_added"] = True
                    print("has added forced stitch tool prompt.")
                    ready_sessions.add(session["i"])
                    continue

                session["chat_history"].append(output_message.model_dump())

                tool_specs = []
                if tool_calls:
                    for tool_call in tool_calls:
                        parsed = vcs_agent.parse_tool_call(tool_call)
                        if isinstance(parsed, tuple):
                            parsed, tool_call_json = parsed
                        else:
                            parsed, tool_call_json = parsed, tool_call
                        if not parsed:
                            print(f"[tool] session {session['i']} received unparsable tool call: {tool_call}")
                            continue
                        tool_specs.append((parsed, tool_call_json))

                if tool_specs:
                    session["pending_tools"] = 0
                    for parsed, tool_call_json in tool_specs:
                        tool_name = (parsed.get("name") if isinstance(parsed, dict) else None) \
                                    or tool_call_json.get("function", {}).get("name")
                        if tool_name in LIMIT_TOOL_NAMES:
                            if not scan_allowed(session.get("scan_calls", 0)):
                                session["chat_history"].append({
                                    "tool_call_id": tool_call_json["id"],
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": f"[tool disabled] '{tool_name}' reached max scan calls ({args.max_scan_calls}) in this session.",
                                })
                                try:
                                    _tool_name = tool_call_json.get("function", {}).get("name", "")
                                except Exception:
                                    _tool_name = ""
                                if isinstance(_tool_name, str) and _tool_name.lower() == "stitched_observer":
                                    session["has_stitched"] = True
                                disabled_log = {
                                    "session_id": session["item"].get('id'),
                                    "tool_name": tool_name,
                                    "tool_args": parsed.get("arguments") if isinstance(parsed, dict) else None,
                                    "result": "[disabled]",
                                    "interval": None,
                                    "frames_used": None,
                                    "start_time": time.time(),
                                    "end_time": time.time(),
                                    "duration": 0.0,
                                    "error": f"tool '{tool_name}' disabled by max scan calls ({args.max_scan_calls})",
                                }
                                with tool_log_lock:
                                    with open(callback_log_file, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(disabled_log, ensure_ascii=False) + "\n")
                                continue
                            else:
                                session["scan_calls"] = session.get("scan_calls", 0) + 1
                        fut_tool = tool_pool.submit(_tool_task, parsed, session["item"])
                        tool_fut2info[fut_tool] = (session, tool_call_json, parsed)
                        all_futures.add(fut_tool)
                        session["pending_tools"] += 1
                else:
                    session["output_text"] = output_message.content

                session["turn_cnt"] += 1
                if session["turn_cnt"] >= args.max_turns:
                    session["stop_all"] = True
                    if not session["output_text"]:
                        last_msg = next(
                            (m for m in reversed(session["chat_history"]) if m.get("role") != "tool"),
                            {},
                        )
                        session["output_text"] = last_msg.get("content", "")

                if session["pending_tools"] == 0 and not session["stop_all"]:
                    ready_sessions.add(session["i"])

                _maybe_finalize_session(session)
                continue

            if fut in tool_fut2info:
                session, tool_call_json, parsed = tool_fut2info.pop(fut)
                try:
                    result, is_stop = fut.result()
                except Exception as exc:
                    result, is_stop = str(exc) or repr(exc), False

                tool_name = (tool_call_json.get("function") or {}).get("name", "")
                raw_text, raw_payload = _normalize_tool_result_payload(result)
                tool_name = (tool_call_json.get("function") or {}).get("name", "")
                if isinstance(tool_name, str) and tool_name.lower() == "stitched_observer":
                    session["has_stitched"] = True

                tool_call_dict = parsed if isinstance(parsed, dict) else None

                try:
                    with open(callback_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "event": "tool_raw",
                            "session_id": session["item"].get("id"),
                            "tool_call_id": tool_call_json["id"],
                            "tool_name": tool_name,
                            "tool_args": tool_call_dict.get("arguments") if tool_call_dict else None,
                            "raw_text": raw_text,
                            "raw_payload": raw_payload,
                            "timestamp": time.time(),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                
                display_content = raw_text
                registry_scheduled = False
                registry_meta: Optional[Dict[str, Any]] = None

                should_update_registry = (
                    SUBJECT_REGISTRY_ENABLED
                    and not is_stop
                    and (tool_name or "").lower() != "finish"
                )
                if should_update_registry:
                    try:
                        registry_meta = _prepare_subject_registry_update(
                            session,
                            tool_name=tool_name or "",
                            tool_call=tool_call_dict,
                            raw_text=raw_text,
                            raw_payload=raw_payload,
                        )
                    except Exception as exc:
                        print(f"[Subject Registry] Prepare failed: {exc}")
                        registry_meta = None
                    if registry_meta and registry_meta.get("payload"):
                        fut_registry = subject_registry_pool.submit(_subject_registry_task, registry_meta["payload"])
                        subject_registry_fut2info[fut_registry] = {
                            "session": session,
                            "tool_call": tool_call_json,
                            "tool_name": tool_name,
                            "raw_text": raw_text,
                            "raw_payload": raw_payload,
                            "is_stop": is_stop,
                            "meta": registry_meta,
                        }
                        all_futures.add(fut_registry)
                        session["pending_subject_registry"] = session.get("pending_subject_registry", 0) + 1
                        registry_scheduled = True
                    else:
                        should_update_registry = False

                if not registry_scheduled:
                    if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], str):
                        display_content = result[0]
                    session["chat_history"].append(
                        {
                            "tool_call_id": tool_call_json["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": str(display_content),
                        }
                    )
                    if should_update_registry:
                        session["observation_step"] = session.get("observation_step", 0) + 1
                    if is_stop:
                        session["stop_all"] = True
                        session["output_text"] = display_content

                session["pending_tools"] = max(session["pending_tools"] - 1, 0)
                if session["pending_tools"] == 0 and session.get("pending_subject_registry", 0) == 0:
                    if not session["stop_all"] and session["turn_cnt"] < args.max_turns:
                        ready_sessions.add(session["i"])
                _maybe_finalize_session(session)
                continue

            if fut in subject_registry_fut2info:
                info = subject_registry_fut2info.pop(fut)
                session = info["session"]
                tool_call_json = info["tool_call"]
                tool_name = info.get("tool_name", "")
                raw_text = info.get("raw_text", "")
                meta = info.get("meta", {})
                turn_idx = meta.get("turn_idx", session.get("observation_step", 0) + 1)
                raw_block = meta.get("raw_block", raw_text)
                per_turn_subjects = meta.get("per_turn_subjects") or {}
                try:
                    response = fut.result()
                except Exception as exc:
                    response = exc

                parsed, raw_content = _decode_subject_registry_output(response)
                registry_state: SubjectRegistryState = session.setdefault(
                    "subject_registry_state",
                    SubjectRegistryState(),
                )
                prev_subject_registry = dict(registry_state.subject_registry)

                update_result = _extract_subject_registry_update(
                    parsed,
                    raw_text=raw_text,
                    prev_subject_registry=prev_subject_registry,
                    per_turn_subjects=per_turn_subjects,
                    raw_content=raw_content,
                )
                observation_summary = update_result.observation_summary
                updated_subject_registry = update_result.updated_subject_registry

                registry_failed = bool(update_result.failed)
                if registry_failed and update_result.error_message:
                    print(f"[Subject Registry] {update_result.error_message}")

                session["observation_step"] = turn_idx

                if registry_failed:
                    session["chat_history"].append(
                        {
                            "tool_call_id": tool_call_json["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": str(raw_text),
                        }
                    )
                    try:
                        with open(callback_log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "event": "subject_registry_update_failed",
                                "session_id": session["item"].get("id"),
                                "tool_call_id": tool_call_json["id"],
                                "tool_name": tool_name,
                                "raw_subject_registry_response": raw_content,
                                "turn_idx": turn_idx,
                                "timestamp": time.time(),
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                else:
                    new_state = SubjectRegistryState(subject_registry=updated_subject_registry)
                    session["subject_registry_state"] = new_state
                    session.setdefault("subject_registry_records", []).append(
                        {
                            "turn_idx": turn_idx,
                            "tool_name": meta.get("tool_name", tool_name),
                            "tool_call": meta.get("tool_call"),
                            "observation_summary": observation_summary,
                            "raw_text": raw_block,
                        }
                    )

                    session["chat_history"].append(
                        {
                            "tool_call_id": tool_call_json["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": str(observation_summary),
                        }
                    )

                    try:
                        with open(callback_log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "event": "subject_registry_update",
                                "session_id": session["item"].get("id"),
                                "tool_call_id": tool_call_json["id"],
                                "tool_name": tool_name,
                                "observation_summary": observation_summary,
                                "updated_subject_registry": updated_subject_registry,
                                "raw_subject_registry_response": raw_content,
                                "turn_idx": turn_idx,
                                "timestamp": time.time(),
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                    _refresh_subject_registry_message(session)

                session["pending_subject_registry"] = max(session.get("pending_subject_registry", 1) - 1, 0)
                if session["pending_tools"] == 0 and session.get("pending_subject_registry", 0) == 0:
                    if not session["stop_all"] and session["turn_cnt"] < args.max_turns:
                        ready_sessions.add(session["i"])
                _maybe_finalize_session(session)
                continue

    result_filename = os.path.join(
        save_dir,
        f"{bench_label}_{_short_model_name(args.model_name_or_path)}_maxturn{args.max_turns}_s{args.data_start}_e{args.data_end}_summary_{RUN_START_TIME}.json",
    )
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=4)

    chat_pool.shutdown(wait=True)
    tool_pool.shutdown(wait=True)
    subject_registry_pool.shutdown(wait=True)
    frame_pool.shutdown(wait=True)

    print(f"Results saved to {result_filename}")
    print(f'Accuracy: {acc / cnt if cnt else 0.0}')
    summarize_tool_calls(callback_log_file, save_dir)


if __name__ == "__main__":
    main()
