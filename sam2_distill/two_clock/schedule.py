"""Deterministic causal refresh schedules."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RefreshTrajectory:
    refresh: tuple[bool, ...]
    source: tuple[int, ...]
    age: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.refresh)
        if not size or len(self.source) != size or len(self.age) != size:
            raise ValueError("refresh, source, and age must have one non-empty length")
        last_refresh = None
        for frame, (refresh, source, age) in enumerate(
            zip(self.refresh, self.source, self.age)
        ):
            if refresh:
                last_refresh = frame
            if last_refresh is None:
                raise ValueError("the first frame must refresh")
            if source != last_refresh or age != frame - last_refresh:
                raise ValueError(
                    f"invalid causal state at frame {frame}: "
                    f"refresh={refresh}, source={source}, age={age}"
                )

    @property
    def encoder_calls(self) -> int:
        return sum(self.refresh)


def _trajectory_from_refresh(refresh: list[bool], max_age: int) -> RefreshTrajectory:
    source = []
    age = []
    last_refresh = -1
    for frame, should_refresh in enumerate(refresh):
        if should_refresh:
            last_refresh = frame
        if last_refresh < 0:
            raise ValueError("the first frame must refresh")
        current_age = frame - last_refresh
        if current_age > max_age:
            raise ValueError(
                f"feature age {current_age} exceeds configured maximum {max_age}"
            )
        source.append(last_refresh)
        age.append(current_age)
    return RefreshTrajectory(tuple(refresh), tuple(source), tuple(age))


def fixed_refresh_trajectory(
    num_frames: int,
    interval: int,
    *,
    max_age: int = 5,
) -> RefreshTrajectory:
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if interval < 1 or interval > max_age + 1:
        raise ValueError(f"interval must be in [1, {max_age + 1}]")
    refresh = [frame % interval == 0 for frame in range(num_frames)]
    return _trajectory_from_refresh(refresh, max_age=max_age)


def balanced_refresh_phase(
    video_id: str | int,
    interval: int,
    *,
    seed: int = 250107256,
) -> int:
    """Return a paired per-video phase that breaks annotation-cadence aliasing."""
    if interval < 1:
        raise ValueError("interval must be positive")
    payload = f"two-clock-eval:{seed}:{video_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % interval


def fixed_refresh_source(
    frame: int,
    *,
    anchor: int,
    interval: int,
    phase: int = 0,
) -> int:
    """Resolve the causal source frame for an anchor-preserving periodic policy."""
    if interval < 1:
        raise ValueError("interval must be positive")
    if not 0 <= phase < interval:
        raise ValueError("phase must be in [0, interval)")
    if frame < anchor:
        raise ValueError("frame cannot precede the prompt anchor")
    relative = frame - anchor
    if relative == 0:
        return anchor
    first_refresh = interval if phase == 0 else phase
    if relative < first_refresh:
        return anchor
    return anchor + first_refresh + (
        (relative - first_refresh) // interval
    ) * interval


def _stable_seed(seed: int, epoch: int, video_id: str | int) -> int:
    payload = f"{seed}:{epoch}:{video_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def training_trajectory(
    num_frames: int,
    *,
    seed: int,
    epoch: int,
    video_id: str | int,
    max_age: int = 5,
    full_refresh_probability: float = 0.25,
) -> RefreshTrajectory:
    """Generate the registered full-refresh/reuse-run training mixture."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if max_age < 1:
        raise ValueError("max_age must be positive")
    if not 0.0 <= full_refresh_probability <= 1.0:
        raise ValueError("full_refresh_probability must be in [0, 1]")

    rng = random.Random(_stable_seed(seed, epoch, video_id))
    if rng.random() < full_refresh_probability:
        return _trajectory_from_refresh([True] * num_frames, max_age=max_age)

    refresh = [False] * num_frames
    refresh[0] = True
    frame = 0
    while frame < num_frames:
        reuse_run = rng.randint(1, max_age)
        frame += reuse_run + 1
        if frame < num_frames:
            refresh[frame] = True
    return _trajectory_from_refresh(refresh, max_age=max_age)
