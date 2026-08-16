"""Small, dependency-free verification helpers for two-clock experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SAFE_WRITE_EXPERIMENTS = {"D", "E"}


def _uniform_int(value: Any, name: str) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    integers = [int(item) for item in values]
    if not integers or any(item != integers[0] for item in integers):
        raise ValueError(f"{name} must be non-empty and uniform per video")
    return integers[0]


def build_clip_trace(
    outputs: Iterable[dict[str, Any]],
    *,
    experiment: str,
    encoder_calls: int,
    teacher_frozen: bool,
    teacher_registered: bool,
) -> dict[str, Any]:
    """Convert one real model forward into a compact, reviewable trace."""
    frames = []
    for stage, output in enumerate(outputs):
        refresh = bool(_uniform_int(output["two_clock_refresh"], "refresh"))
        raw = _uniform_int(output["two_clock_raw_frame"], "raw frame")
        source = _uniform_int(output["two_clock_source_raw"], "source frame")
        age = _uniform_int(output["two_clock_age"], "age")
        frames.append(
            {
                "stage": stage,
                "raw_frame": raw,
                "action": "REFRESH" if refresh else "REUSE",
                "source_frame": source,
                "age": age,
                "encoder_called": refresh,
                "spatial_write": bool(output["two_clock_spatial_write"]),
                "memory_write_feature": output["two_clock_memory_write_feature"],
                "pointer_write": output.get("obj_ptr") is not None,
            }
        )
    trace = {
        "experiment": experiment,
        "encoder_calls": int(encoder_calls),
        "teacher_frozen": bool(teacher_frozen),
        "teacher_registered": bool(teacher_registered),
        "frames": frames,
    }
    validate_clip_trace(trace)
    return trace


def validate_clip_trace(trace: dict[str, Any], *, max_age: int = 5) -> None:
    """Fail closed when a recorded clip violates the registered state machine."""
    frames = trace.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("flight trace has no frames")
    if frames[0].get("action") != "REFRESH":
        raise ValueError("the prompted anchor must refresh")

    refresh_count = 0
    for frame in frames:
        raw = int(frame["raw_frame"])
        source = int(frame["source_frame"])
        age = int(frame["age"])
        refresh = frame.get("action") == "REFRESH"
        if age != raw - source:
            raise ValueError("feature age does not equal raw minus source frame")
        if age < 0 or age > max_age:
            raise ValueError("feature age is outside the registered range")
        if refresh != (age == 0 and raw == source):
            raise ValueError("refresh action, source frame, and age disagree")
        if bool(frame.get("encoder_called")) != refresh:
            raise ValueError("per-frame encoder call does not match refresh action")
        if not bool(frame.get("pointer_write")):
            raise ValueError("tracked frame did not produce an object pointer")
        if frame.get("memory_write_feature") != "raw":
            raise ValueError("memory encoder did not receive raw cached features")
        if trace.get("experiment") in SAFE_WRITE_EXPERIMENTS:
            if bool(frame.get("spatial_write")) != refresh:
                raise ValueError("safe-write variant wrote spatial memory on reuse")
        elif not bool(frame.get("spatial_write")):
            raise ValueError("non-safe-write variant skipped spatial memory")
        refresh_count += int(refresh)

    if int(trace.get("encoder_calls", -1)) != refresh_count:
        raise ValueError("clip encoder-call count does not match refresh count")
    if not bool(trace.get("teacher_frozen")):
        raise ValueError("online teacher is not frozen")
    if bool(trace.get("teacher_registered")):
        raise ValueError("online teacher entered the student module graph")


def append_clip_trace(path: Path, trace: dict[str, Any]) -> None:
    """Append one validated trace; each DDP rank owns a separate file."""
    validate_clip_trace(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, sort_keys=True) + "\n")


def load_clip_traces(paths: Iterable[Path]) -> list[dict[str, Any]]:
    traces = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                trace = json.loads(line)
                validate_clip_trace(trace)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid flight trace {path}:{line_number}: {error}") from error
            traces.append(trace)
    if not traces:
        raise ValueError("no flight traces were recorded")
    return traces


def validate_optimizer_audit(payload: dict[str, Any]) -> None:
    if payload.get("status") != "pass":
        raise ValueError("optimizer audit status is not pass")
    if payload.get("missing_trainable_parameters"):
        raise ValueError("trainable parameters are missing from the optimizer")
    if payload.get("unexpected_optimizer_parameters"):
        raise ValueError("optimizer contains unexpected parameters")
    teacher = payload.get("teacher", {})
    if not teacher.get("parameters"):
        raise ValueError("optimizer audit did not observe an online teacher")
    if not teacher.get("frozen"):
        raise ValueError("teacher parameters are trainable")
    if teacher.get("registered_with_student"):
        raise ValueError("teacher is registered with the student")
    if teacher.get("optimizer_parameters", 0):
        raise ValueError("teacher parameters entered the optimizer")
