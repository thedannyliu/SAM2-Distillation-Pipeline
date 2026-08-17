from __future__ import annotations

import pytest
import torch

from sam2_distill.two_clock.experiments import EXPERIMENTS, get_experiment
from sam2_distill.two_clock.losses import (
    discounted_refresh_value,
    group_normalized_supervision,
)
from sam2_distill.two_clock.schedule import (
    balanced_refresh_phase,
    fixed_refresh_source,
    fixed_refresh_trajectory,
    training_trajectory,
)
from sam2_distill.two_clock.temporal import (
    FeatureAgeConditioner,
    TwoClockMemoryConditioner,
)


def test_fixed_refresh_trajectory_r4() -> None:
    trajectory = fixed_refresh_trajectory(8, 4)
    assert trajectory.refresh == (True, False, False, False, True, False, False, False)
    assert trajectory.source == (0, 0, 0, 0, 4, 4, 4, 4)
    assert trajectory.age == (0, 1, 2, 3, 0, 1, 2, 3)
    assert trajectory.encoder_calls == 2


def test_training_trajectory_is_deterministic_and_bounded() -> None:
    kwargs = dict(num_frames=32, seed=250107256, epoch=3, video_id="sav_001234")
    first = training_trajectory(**kwargs)
    second = training_trajectory(**kwargs)
    assert first == second
    assert first.refresh[0]
    assert max(first.age) <= 5


def test_fixed_refresh_rejects_unsupported_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        fixed_refresh_trajectory(8, 7)


@pytest.mark.parametrize("interval,target", [(2, 8), (4, 8), (6, 12)])
def test_phase_balancing_covers_every_age_at_sav_gt_cadence(
    interval: int,
    target: int,
) -> None:
    ages = {
        target
        - fixed_refresh_source(
            target,
            anchor=0,
            interval=interval,
            phase=phase,
        )
        for phase in range(interval)
    }
    assert ages == set(range(interval))


def test_balanced_refresh_phase_is_stable_and_bounded() -> None:
    first = balanced_refresh_phase("sav_000123", 6)
    assert first == balanced_refresh_phase("sav_000123", 6)
    assert 0 <= first < 6


def test_registered_experiment_ladder() -> None:
    assert list(EXPERIMENTS) == ["O0", "O1", "O2", "A", "B", "C", "C-r", "D", "E"]
    assert not get_experiment("B").recency_conditioning
    assert get_experiment("C").freshness_conditioning
    assert get_experiment("D").suppress_stale_spatial_writes
    assert get_experiment("E").privileged_state_distillation


def test_age_zero_is_exact_identity() -> None:
    module = FeatureAgeConditioner()
    feature = torch.randn(2, 256, 4, 4)
    output = module.condition_bchw("deep", feature, torch.zeros(2, dtype=torch.long))
    assert torch.equal(output, feature)


def test_age_film_zero_output_projection_receives_first_gradient() -> None:
    module = FeatureAgeConditioner()
    feature = torch.randn(2, 256, 2, 2)
    age = torch.tensor([1, 3])
    module.condition_bchw("deep", feature, age).square().mean().backward()
    gradient = module.heads["deep"].net[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_two_clock_residual_is_identity_initialized_but_trainable() -> None:
    module = TwoClockMemoryConditioner()
    recency = torch.tensor([1, 4])
    freshness = torch.tensor([0, 3])
    delta = module.spatial_delta(recency, freshness)
    assert torch.count_nonzero(delta) == 0
    delta.sum().backward()
    gradient = module.spatial_recency.net[-1].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0
    assert not any(name.startswith("alpha") for name, _ in module.named_parameters())


def test_age_rejects_out_of_range_instead_of_clipping() -> None:
    module = FeatureAgeConditioner()
    with pytest.raises(ValueError, match="feature age"):
        module.condition_bchw("deep", torch.randn(1, 256, 2, 2), torch.tensor([8]))


def test_group_normalization_is_invariant_to_pseudo_count() -> None:
    gt = torch.tensor([2.0, 100.0])
    gt_valid = torch.tensor([True, False])
    short = group_normalized_supervision(
        gt,
        gt_valid,
        torch.tensor([4.0, 4.0]),
        torch.tensor([True, True]),
    )
    long = group_normalized_supervision(
        gt,
        gt_valid,
        torch.tensor([4.0] * 8),
        torch.tensor([True] * 8),
    )
    assert short.item() == pytest.approx(3.0)
    assert long.item() == pytest.approx(short.item())


def test_discounted_refresh_value_has_fixed_horizon() -> None:
    reuse = torch.tensor([[0.4, 0.6, 0.8]])
    refresh = torch.tensor([[0.2, 0.3, 0.5]])
    value = discounted_refresh_value(reuse, refresh)
    assert value.item() == pytest.approx(0.2 + 0.8 * 0.3 + 0.64 * 0.3)
    with pytest.raises(ValueError, match="horizon"):
        discounted_refresh_value(reuse[:, :2], refresh[:, :2])
