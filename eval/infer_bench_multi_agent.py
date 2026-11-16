"""Benchmark and single-video inference runner for Multi-Agent LensWalk."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from benchmarks.benchmarks import load_benchmark
from utils.eval_utils import eval_open_ended
from utils.tool_patch import ToolDispatcher
from vcs.llm_client import build_llm_client
from vcs.multi_agent import MultiAgentCoordinator
from vcs.vcs_dispatcher import VCSDispatcher
from vcs.vcs_utils import strip_image_urls


RUN_START_TIME = time.strftime("%Y%m%d_%H%M%S")


def _safe_path_label(value: Optional[str]) -> str:
    raw = os.path.basename(value or "") or "single_video"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "single_video"


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def _load_subtitles_any(subtitle_path: Optional[str]) -> Any:
    """Load JSON or SRT subtitles into a list of dicts."""
    if not subtitle_path or not os.path.exists(subtitle_path):
        return None
    if subtitle_path.lower().endswith(".json"):
        with open(subtitle_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    def _to_sec(h: str, m: str, s: str, ms: str) -> float:
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    rows: List[Dict[str, Any]] = []
    for block in re.split(r"\n\n+", content.strip()):
        match = pattern.search(block)
        if not match:
            continue
        start = _to_sec(*match.groups()[:4])
        end = _to_sec(*match.groups()[4:8])
        text = block[match.end() :].strip().replace("\n", " ")
        rows.append({"start_time": round(start, 2), "end_time": round(end, 2), "subtitle": text})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Orchestrator - Parallel Actors - Stitched Verify inference.")
    p.add_argument("--tool_config", default="tool_configs/vcs_multi_agent.yaml")
    p.add_argument("--bench_name", default="", help="JSON/JSONL benchmark path (or omit to use --video_path).")
    p.add_argument("--video_root", default="")
    p.add_argument("--subtitle_root", default="")
    p.add_argument("--video_path", default="", help="Single video path when --bench_name is omitted.")
    p.add_argument("--subtitle_path", default="", help="Optional .json or .srt subtitle path for single video.")
    p.add_argument("--question", default="", help="Question for single video mode.")
    p.add_argument("--question_type", default="open-ended", choices=["multiple-choice", "open-ended"])

    p.add_argument("--model_name_or_path", default="gpt-5")
    p.add_argument("--actor_model", default="", help="Optional separate model for Actor agents.")
    p.add_argument("--mllm_backend", default="", help="Optional visual tool model override (sets MLLM_BACKEND).")
    p.add_argument("--llm_backend", default="auto", choices=["auto", "litellm", "openai"])

    p.add_argument("--total_frame_budget", type=int, default=192)
    p.add_argument("--overview_frames", type=int, default=64)
    p.add_argument("--verify_reserve_frames", type=int, default=32)
    p.add_argument("--min_actor_frames", type=int, default=16)
    p.add_argument("--max_actor_turns", type=int, default=3)
    p.add_argument("--max_replans", type=int, default=1)
    p.add_argument("--conditional_overview", type=_str2bool, default=True)
    p.add_argument("--force_overview", type=_str2bool, default=None)

    p.add_argument("--subject_registry", type=_str2bool, default=False)
    p.add_argument("--use_subtitle", type=_str2bool, default=False)
    p.add_argument("--tool_call_strict_schema", type=_str2bool, default=True)

    p.add_argument("--data_start", type=int, default=0)
    p.add_argument("--data_end", type=int, default=-1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_actor_workers", type=int, default=3)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--custom_name", default="")
    p.add_argument("--resume", type=_str2bool, default=False)
    p.add_argument("--resume_log_path", default="")
    p.add_argument("--verbose", action="store_true", help="Print real-time progress across Orchestrator, Actors, and Tools.")
    return p


def _strip_actor_histories(actor_histories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return {hid: strip_image_urls(hist) for hid, hist in (actor_histories or {}).items()}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.mllm_backend:
        os.environ["MLLM_BACKEND"] = args.mllm_backend
    elif not os.getenv("MLLM_BACKEND"):
        os.environ["MLLM_BACKEND"] = args.model_name_or_path

    if not args.bench_name and not (args.video_path and args.question):
        raise ValueError("Provide either --bench_name or both --video_path and --question.")

    dispatcher = VCSDispatcher(
        agent_tool_call="return",
        strict_call=args.tool_call_strict_schema,
        subject_registry=args.subject_registry,
    )
    dispatcher.load_config(args.tool_config)

    llm_client = build_llm_client(for_tool=False, backend=args.llm_backend)
    coordinator = MultiAgentCoordinator(
        dispatcher=dispatcher,
        llm_client=llm_client,
        orchestrator_model=args.model_name_or_path,
        actor_model=args.actor_model or args.model_name_or_path,
        total_frame_budget=args.total_frame_budget,
        overview_frames=args.overview_frames,
        verify_reserve_frames=args.verify_reserve_frames,
        min_actor_frames=args.min_actor_frames,
        max_actor_turns=args.max_actor_turns,
        max_replans=args.max_replans,
        conditional_overview=args.conditional_overview,
        max_actor_workers=args.max_actor_workers,
        verbose=args.verbose,
    )

    label = _safe_path_label(args.bench_name or args.video_path)
    model_short = args.model_name_or_path.split("/")[-1]
    run_name = f"run_multi_agent/{label}_{model_short}_B{args.total_frame_budget}_{RUN_START_TIME}"
    if args.custom_name:
        run_name += f"_{args.custom_name}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    conversations_file = save_dir / "all_conversations.jsonl"
    tool_calls_file = save_dir / "all_tool_calls.jsonl"
    (save_dir / "run_settings.json").write_text(
        json.dumps({"start_time": RUN_START_TIME, "args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Single-video quick execution mode
    if not args.bench_name:
        subtitles = _load_subtitles_any(args.subtitle_path) if args.subtitle_path else None
        t0 = time.time()
        out = coordinator.run(
            question=args.question,
            video_path=args.video_path,
            subtitles=subtitles,
            session_id="single_0",
            force_overview=args.force_overview,
        )
        elapsed = time.time() - t0
        record = {
            "index": 0,
            "question": args.question,
            "video_path": args.video_path,
            "answer": out["final_answer"],
            "budget": out["budget"],
            "replan_count": out["replan_count"],
            "has_stitched_verified": out["has_stitched_verified"],
            "overview_skipped_reason": out["overview_skipped_reason"],
            "plan": out["plan"],
            "actor_reports": out["actor_reports"],
            "decision_trace": out["decision_trace"],
            "stitched_observation": out["stitched_observation"],
            "actor_histories": _strip_actor_histories(out["actor_histories"]),
            "elapsed_sec": round(elapsed, 2),
        }
        conversations_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        with tool_calls_file.open("w", encoding="utf-8") as tf:
            for row in out["tool_call_logs"]:
                tf.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Question: {args.question}")
        print(f"Final Answer: {out['final_answer']}")
        print(f"Budget Breakdown: {out['budget']}")
        print(f"Saved logs under: {save_dir}")
        return {"summary": record, "save_dir": str(save_dir)}

    # Benchmark mode
    ds = load_benchmark(
        args.bench_name,
        data_root=args.video_root or None,
        subtitle_root=args.subtitle_root or None,
        shuffle=False,
    )
    completed: set[int] = set()
    prior_records: List[Dict[str, Any]] = []
    if args.resume and args.resume_log_path:
        prev_conv = Path(args.resume_log_path) / "all_conversations.jsonl"
        if prev_conv.exists():
            with prev_conv.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed.add(int(rec["index"]))
                        prior_records.append(rec)
                    except Exception:
                        pass

    pending = [
        (idx, item)
        for idx, item in enumerate(ds)
        if idx >= args.data_start
        and (args.data_end <= 0 or idx < args.data_end)
        and idx not in completed
    ]

    write_lock = Lock()
    all_records = list(prior_records)

    def _run_item(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        video_path = item.get("video_path") or item.get("video_path_abs")
        duration = float(item.get("duration") or item.get("video_duration") or 0.0) or None
        question = item["question"]
        gt = item.get("golden_answer") or item.get("answer")
        if isinstance(gt, list):
            gt = gt[0]
        q_type = item.get("question_type") or args.question_type

        subtitles = None
        if args.use_subtitle and item.get("subtitle_path"):
            subtitles = _load_subtitles_any(item["subtitle_path"])

        t0 = time.time()
        out = coordinator.run(
            question=question,
            video_path=video_path,
            duration_sec=duration,
            subtitles=subtitles,
            session_id=item.get("id", idx),
            force_overview=args.force_overview,
        )
        elapsed = time.time() - t0
        pred = out["final_answer"]

        score = 0.0
        if gt is not None:
            if q_type == "open-ended":
                if pred and str(pred).strip().lower() == str(gt).strip().lower():
                    score = 1.0
                else:
                    ok, _, _ = eval_open_ended(
                        {"id": idx, "question": question, "answer": gt, "response": pred},
                        llm_client,
                        model=args.model_name_or_path,
                    )
                    score = 1.0 if ok else 0.0
            else:
                score = ToolDispatcher.compute_score(pred, str(gt))

        rec = {
            "index": idx,
            "id": item.get("id", idx),
            "question": question,
            "golden_answer": gt,
            "answer": pred,
            "score": score,
            "budget": out["budget"],
            "replan_count": out["replan_count"],
            "has_stitched_verified": out["has_stitched_verified"],
            "overview_skipped_reason": out["overview_skipped_reason"],
            "plan": out["plan"],
            "actor_reports": out["actor_reports"],
            "decision_trace": out["decision_trace"],
            "stitched_observation": out["stitched_observation"],
            "actor_histories": _strip_actor_histories(out["actor_histories"]),
            "subject_registry": out["subject_registry"],
            "elapsed_sec": round(elapsed, 2),
        }

        with write_lock:
            with conversations_file.open("a", encoding="utf-8") as cf:
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with tool_calls_file.open("a", encoding="utf-8") as tf:
                for row in out["tool_call_logs"]:
                    tf.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rec

    if args.workers <= 1 or len(pending) <= 1:
        for idx, item in pending:
            rec = _run_item(idx, item)
            all_records.append(rec)
            print(f"[idx {idx}] score={rec['score']} frames={rec['budget']['total_frames_used']} ans={rec['answer']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_run_item, idx, item) for idx, item in pending]
            for fut in as_completed(futs):
                rec = fut.result()
                all_records.append(rec)
                print(
                    f"[idx {rec['index']}] score={rec['score']} "
                    f"frames={rec['budget']['total_frames_used']} ans={rec['answer']}"
                )

    n = len(all_records)
    summary = {
        "total_items": n,
        "accuracy": (sum(r.get("score", 0.0) for r in all_records) / n) if n else 0.0,
        "avg_total_frames": (sum(r["budget"]["total_frames_used"] for r in all_records) / n) if n else 0.0,
        "avg_overview_frames": (sum(r["budget"]["overview_frames"] for r in all_records) / n) if n else 0.0,
        "avg_actor_frames": (sum(r["budget"]["actor_frames"] for r in all_records) / n) if n else 0.0,
        "avg_verify_frames": (sum(r["budget"]["verify_frames"] for r in all_records) / n) if n else 0.0,
        "avg_fallback_frames": (sum(r["budget"]["fallback_frames"] for r in all_records) / n) if n else 0.0,
        "stitched_verify_rate": (sum(1 for r in all_records if r.get("has_stitched_verified")) / n) if n else 0.0,
        "fallback_rate": (sum(1 for r in all_records if r.get("replan_count", 0) > 0) / n) if n else 0.0,
    }
    (save_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
    return {"summary": summary, "save_dir": str(save_dir)}


if __name__ == "__main__":
    main()
