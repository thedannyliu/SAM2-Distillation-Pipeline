"""Pre-registered experiment definitions for two-clock SAM2.1-L reuse."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoClockExperiment:
    name: str
    train: bool
    train_with_reuse: bool
    age_conditioning: bool
    recency_conditioning: bool
    freshness_conditioning: bool
    suppress_stale_spatial_writes: bool
    privileged_state_distillation: bool

    def validate(self) -> None:
        if self.freshness_conditioning and not self.recency_conditioning:
            raise ValueError("freshness conditioning requires recency conditioning")
        if self.suppress_stale_spatial_writes and not self.freshness_conditioning:
            raise ValueError("stale-write suppression requires the C-rf read path")
        if self.privileged_state_distillation and not self.suppress_stale_spatial_writes:
            raise ValueError("state distillation requires stale-write suppression")
        if not self.train and self.name not in {"O0", "O1"}:
            raise ValueError(f"only O0/O1 are evaluation-only, got {self.name}")


EXPERIMENTS = {
    "O0": TwoClockExperiment("O0", False, False, False, False, False, False, False),
    "O1": TwoClockExperiment("O1", False, False, False, False, False, False, False),
    "O2": TwoClockExperiment("O2", True, False, False, False, False, False, False),
    "A": TwoClockExperiment("A", True, True, False, False, False, False, False),
    "B": TwoClockExperiment("B", True, True, True, False, False, False, False),
    "C": TwoClockExperiment("C", True, True, True, True, True, False, False),
    "C-r": TwoClockExperiment("C-r", True, True, True, True, False, False, False),
    "D": TwoClockExperiment("D", True, True, True, True, True, True, False),
    "E": TwoClockExperiment("E", True, True, True, True, True, True, True),
}

for _experiment in EXPERIMENTS.values():
    _experiment.validate()


def get_experiment(name: str) -> TwoClockExperiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"unknown two-clock experiment {name!r}; choose {choices}") from error
