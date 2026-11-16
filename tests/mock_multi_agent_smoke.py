"""Offline smoke tests verifying Orchestrator - Parallel Actors - Stitched Verify."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcs.llm_client import _CompatMessage, _CompatToolCall  # noqa: E402
from vcs.multi_agent import MultiAgentCoordinator  # noqa: E402
from vcs.vcs_dispatcher import VCSDispatcher  # noqa: E402


FIXTURE_DIR = ROOT / "tests" / "fixtures"
VIDEO_DIR = FIXTURE_DIR / "videos"
VIDEO_PATH = VIDEO_DIR / "mock_multi_agent.mp4"


def _ensure_synthetic_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    try:
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 2.0, (96, 64))
        for i in range(60):  # 30 seconds at 2 fps
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (255, 0, 0) if i < 30 else (0, 255, 0)
            writer.write(frame)
        writer.release()
        return
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=96x64:d=30:r=2",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(f"Could not generate synthetic test video: {proc.stderr}")


class ScenarioFakeCompletions:
    """Deterministic fake LLM/VLM client supporting Scenario 1, 2, and 3."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.actor_turn_counts: dict[str, int] = {}
        self.synth_calls = 0

    def create(self, **kwargs):
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools")
        joined = "\n".join(str(m.get("content", "")) for m in messages)

        # 1. Visual tool observer calls (no tools argument passed)
        if not tools:
            if "2x4 grid images" in joined:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=_CompatMessage(
                                content=json.dumps(
                                    {
                                        "frames": [
                                            {"timestamp": "0.0s", "description": "Scene A at start."},
                                            {"timestamp": "25.0s", "description": "Scene B near end."},
                                        ]
                                    }
                                )
                            )
                        )
                    ]
                )

            if "Synthesis & Cross-Verification phase" in joined:
                self.synth_calls += 1
                if self.scenario == "stitched_verify" and self.synth_calls == 1:
                    payload = {
                        "action": "STITCHED_VERIFY",
                        "reasoning": "Need to compare H1 (0-8s) and H2 (20-28s) side by side.",
                        "final_answer": None,
                        "verify_segments": [
                            {"interval": {"start_sec": 0.0, "end_sec": 6.0}, "fps": 1.0},
                            {"interval": {"start_sec": 20.0, "end_sec": 26.0}, "fps": 1.0},
                        ],
                        "verify_query": "Compare colors across both segments.",
                    }
                elif self.scenario == "fallback_replan" and self.synth_calls == 1:
                    payload = {
                        "action": "FALLBACK_REPLAN",
                        "reasoning": "Both H1 and H2 returned NOT_FOUND; search uncovered middle gap.",
                        "final_answer": None,
                        "fallback_intervals": [{"start_sec": 10.0, "end_sec": 20.0}],
                        "fallback_query": "Inspect middle segment for the answer.",
                    }
                else:
                    payload = {
                        "action": "FINISH",
                        "reasoning": "Actor reports contain sufficient direct evidence.",
                        "final_answer": "A",
                    }
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=_CompatMessage(content=json.dumps(payload)))]
                )

            if "finalizing the answer after direct visual cross-verification" in joined:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=_CompatMessage(
                                content=json.dumps({"reasoning": "Verified across stitched segments.", "final_answer": "B"})
                            )
                        )
                    ]
                )

            if "Orchestrator Agent of a multi-agent video understanding system" in joined:
                if self.scenario == "explicit_ts":
                    plan = {
                        "reasoning": "Explicit timestamp given; use K=1 hypothesis.",
                        "hypotheses": [
                            {
                                "hypothesis_id": "H1",
                                "interval": {"start_sec": 0.0, "end_sec": 10.0},
                                "sub_query": "Check background color at 0-10s.",
                                "rationale": "Question mentions 0:05.",
                                "allocated_frames": 32,
                            }
                        ],
                    }
                else:
                    # Deliberately return two overlapping-free hypotheses leaving [10, 20] uncovered for scenario 3
                    plan = {
                        "reasoning": "Two plausible regions near start and end.",
                        "hypotheses": [
                            {
                                "hypothesis_id": "H1",
                                "interval": {"start_sec": 0.0, "end_sec": 8.0},
                                "sub_query": "Check early region.",
                                "rationale": "Start candidate.",
                                "allocated_frames": 32,
                            },
                            {
                                "hypothesis_id": "H2",
                                "interval": {"start_sec": 22.0, "end_sec": 30.0},
                                "sub_query": "Check late region.",
                                "rationale": "End candidate.",
                                "allocated_frames": 32,
                            },
                        ],
                    }
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=_CompatMessage(content=json.dumps(plan)))]
                )

            # Generic visual MLLM observer response (dynamic_focus / stitched_observer)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_CompatMessage(content="Observed solid blue frames cleanly."))]
            )

        # 2. Actor tool-calling turns
        hid = "unknown"
        for line in joined.splitlines():
            if line.startswith("Hypothesis ID:"):
                hid = line.split(":", 1)[1].strip()
                break
        turn = self.actor_turn_counts.get(hid, 0) + 1
        self.actor_turn_counts[hid] = turn

        if turn == 1:
            start_s, end_s = (10.0, 16.0) if hid.startswith("F") else ((22.0, 28.0) if hid == "H2" else (0.0, 6.0))
            tc = _CompatToolCall(
                f"call_df_{hid}_{turn}",
                "dynamic_focus",
                json.dumps(
                    {
                        "interval": {"start_sec": start_s, "end_sec": end_s},
                        "query": "Inspect visual details.",
                        "fps": 1.0,
                        "detail_level": "high",
                        "max_total_frames": 4,
                    }
                ),
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=_CompatMessage(tool_calls=[tc]))])

        # Turn 2: submit_report
        if self.scenario == "fallback_replan" and not hid.startswith("F"):
            report_args = {
                "status": "NOT_FOUND",
                "candidate_answer": None,
                "confidence": "low",
                "evidences": [],
                "uncertainties": "No target object seen in this interval.",
            }
        elif self.scenario == "stitched_verify":
            report_args = {
                "status": "PARTIAL",
                "candidate_answer": "Needs comparison with other segment",
                "confidence": "medium",
                "evidences": [
                    {
                        "interval": {"start_sec": 0.0 if hid == "H1" else 22.0, "end_sec": 6.0 if hid == "H1" else 28.0},
                        "visual_fact": f"Observed key visual cue in {hid}.",
                        "relevance": "partial",
                    }
                ],
                "uncertainties": "Requires cross-segment comparison.",
            }
        else:
            report_args = {
                "status": "FOUND",
                "candidate_answer": "A",
                "confidence": "high",
                "evidences": [
                    {
                        "interval": {"start_sec": 10.0 if hid.startswith("F") else 0.0, "end_sec": 16.0 if hid.startswith("F") else 6.0},
                        "visual_fact": "Blue background verified at high resolution.",
                        "relevance": "direct",
                    }
                ],
                "uncertainties": "",
            }

        tc = _CompatToolCall(f"call_rep_{hid}_{turn}", "submit_report", json.dumps(report_args))
        return SimpleNamespace(choices=[SimpleNamespace(message=_CompatMessage(tool_calls=[tc]))])


class ScenarioFakeClient:
    def __init__(self, scenario: str) -> None:
        self.chat = SimpleNamespace(completions=ScenarioFakeCompletions(scenario))


def _run_scenario(scenario: str, question: str) -> dict:
    fake_client = ScenarioFakeClient(scenario)
    import vcs.function.base_observer as base_observer

    class _ToolFakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = fake_client.chat

    base_observer.OpenAI = _ToolFakeOpenAI
    os.environ["OPENAI_API_KEY"] = "mock-key"
    os.environ["OPENAI_API_BASE"] = "http://mock.local/v1"

    dispatcher = VCSDispatcher(agent_tool_call="return", strict_call=True, subject_registry=False)
    dispatcher.load_config(str(ROOT / "tool_configs" / "vcs_multi_agent.yaml"))

    coord = MultiAgentCoordinator(
        dispatcher=dispatcher,
        llm_client=fake_client,
        orchestrator_model="mock-orchestrator",
        actor_model="mock-actor",
        total_frame_budget=160,
        overview_frames=16,
        verify_reserve_frames=24,
        max_actor_turns=3,
        max_replans=1,
        conditional_overview=True,
    )
    return coord.run(question=question, video_path=str(VIDEO_PATH), duration_sec=30.0, session_id=scenario)


def main() -> None:
    os.chdir(ROOT)
    _ensure_synthetic_video(VIDEO_PATH)
    shutil.rmtree(ROOT / "results" / "run_multi_agent", ignore_errors=True)

    # Case 1: Explicit timestamp skips overview and finishes directly
    out1 = _run_scenario("explicit_ts", "At 0:05, what color is shown?\n(A) Blue\n(B) Red")
    assert out1["overview_skipped_reason"] == "explicit_timestamp", out1
    assert out1["budget"]["overview_frames"] == 0, out1["budget"]
    assert out1["final_answer"] == "A", out1
    print("[PASS] Case 1 (Explicit timestamp -> conditional overview skip -> FINISH)")

    # Case 2: Multi-hypothesis -> Stitched Verify -> final answer B
    out2 = _run_scenario("stitched_verify", "Compare the color at the beginning and end.\n(A) Same\n(B) Different")
    assert out2["budget"]["overview_frames"] > 0, out2["budget"]
    assert out2["has_stitched_verified"] is True, out2
    assert out2["budget"]["verify_frames"] > 0, out2["budget"]
    assert out2["final_answer"] == "B", out2
    print("[PASS] Case 2 (Overview -> 2 Actors -> STITCHED_VERIFY -> FINISH)")

    # Case 3: Falsification-first NOT_FOUND -> Fallback Replan on uncovered intervals
    out3 = _run_scenario("fallback_replan", "Where does the clue appear?\n(A) Middle\n(B) Nowhere")
    assert out3["replan_count"] == 1, out3
    assert out3["budget"]["fallback_frames"] > 0, out3["budget"]
    assert any(r["status"] == "NOT_FOUND" and r["candidate_answer"] is None for r in out3["actor_reports"]), out3
    assert out3["final_answer"] == "A", out3
    print("[PASS] Case 3 (NOT_FOUND anti-bias -> FALLBACK_REPLAN on uncovered gap -> FINISH)")

    # Verify CLI runner end-to-end import + single-video invocation with monkeypatched build_llm_client
    import eval.infer_bench_multi_agent as runner_mod
    runner_mod.build_llm_client = lambda **kwargs: ScenarioFakeClient("explicit_ts")
    res = runner_mod.main(
        [
            "--video_path",
            str(VIDEO_PATH),
            "--question",
            "At 0:05, what color is shown? (A) Blue (B) Red",
            "--model_name_or_path",
            "mock-orch",
            "--custom_name",
            "smoke",
            "--verbose",
        ]
    )
    assert Path(res["save_dir"]).exists()
    print(f"[PASS] CLI Runner Smoke Test -> {res['save_dir']}")


if __name__ == "__main__":
    main()
